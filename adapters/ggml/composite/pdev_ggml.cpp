// pdev over a ggml backend device, through the public ggml API only (ggml.h, ggml-alloc.h, ggml-backend.h).
//
// The official ggml-vulkan backend (RX 6400 / WX 2100), ggml-cpu and Metal are used this way. The pdev creates its own ggml_backend_t on the
// device, next to the one llama.cpp uses for the layers on the same GPU. Shared with llama.cpp's instance: the Vulkan device, its queues and
// its memory (ggml-vulkan keeps them in one device object that every backend instance of that GPU points to); everything else (command
// pools, fence, scratch buffers) is per instance. All calls are made from the one host thread that runs graph_compute, so the queue is never
// submitted to from two threads at once (docs/benchmarks/moe-ep-20261021/sglx-lmhead/README.md, "Vulkan device sharing").
//
// Asynchrony: tensor_set_async + graph_compute_async + tensor_get_async only enqueue on a device that supports it (ggml-vulkan, Metal);
// wait() is ggml_backend_synchronize. A backend without the async entries (ggml-cpu) does the work inside the calls, which is also correct.

#include "composite.h"
#include "pdev.h"

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <list>
#include <condition_variable>
#include <functional>
#include <map>
#include <mutex>
#include <thread>
#include <string>
#include <vector>

namespace sglx {
namespace {

struct ggml_piece : piece {
    ggml_type type = GGML_TYPE_F32;
    int64_t   K    = 0;
    int64_t   rows = 0;

    ggml_context *        wctx  = nullptr;
    ggml_backend_buffer_t wbuf  = nullptr;
    ggml_tensor *         W     = nullptr;
    ggml_gallocr_t        galloc = nullptr;

    // the MUL_MAT graph for N columns (N = 1 is the token generation, the others are prompt batches), least recently used last
    struct graph_entry {
        int64_t        N  = 0;
        ggml_context * ctx = nullptr;
        ggml_cgraph *  gf = nullptr;
        ggml_tensor *  x  = nullptr;
        ggml_tensor *  y  = nullptr;
    };
    std::list<graph_entry> graphs;
    int64_t                reserved_N = 0;

    // pinned host staging (the device's host buffer type) for the input columns and the result: an async copy to or from pageable memory
    // makes ggml-vulkan submit and wait at once, which would serialize the devices
    struct staging {
        ggml_backend_buffer_t buf = nullptr;
        float *               ptr = nullptr;
        size_t                cap = 0;
        std::vector<float>    plain; // when the device has no host buffer type
    };
    staging hx, ho;
    float * out_host = nullptr;
    size_t  out_bytes = 0;

    ~ggml_piece() override {
        if (hx.buf != nullptr) {
            ggml_backend_buffer_free(hx.buf);
        }
        if (ho.buf != nullptr) {
            ggml_backend_buffer_free(ho.buf);
        }
        for (auto & g : graphs) {
            ggml_free(g.ctx);
        }
        if (galloc != nullptr) {
            ggml_gallocr_free(galloc);
        }
        if (wbuf != nullptr) {
            ggml_backend_buffer_free(wbuf);
        }
        if (wctx != nullptr) {
            ggml_free(wctx);
        }
    }
};


// the expert slots of a ggml device (pdev::moe_*): three 3-D tensors [n_embd, n_ff, S] (gate, up) and [n_ff, n_embd, S] (down) in the device memory;
// one expert is the slice `slot` of each. A computation is a small graph: MUL_MAT_ID(up), MUL_MAT_ID(gate), SWIGLU, MUL_MAT_ID(down).
struct ggml_moe_slots : moe_slots {
    moe_shape sh;
    int       n_slots = 0;
    ggml_context *        ctx  = nullptr;
    ggml_backend_buffer_t buf  = nullptr;
    ggml_tensor *         g    = nullptr;
    ggml_tensor *         u    = nullptr;
    ggml_tensor *         d    = nullptr;
    ggml_gallocr_t        galloc_dec = nullptr;   // decode graphs (one token, 1..8 ids)
    ggml_gallocr_t        galloc_pair = nullptr;  // pair graphs (P pairs, one id each)
    int64_t               reserved_P = 0;
    bool                  dec_reserved = false;

    struct graph_entry {
        int            mode = 0;      // 0: one token, `a` = number of ids; 1: pairs, `a` = N_x, `b` = P
        int64_t        a = 0;
        int64_t        b = 0;
        ggml_context * ctx = nullptr;
        ggml_cgraph *  gf = nullptr;
        ggml_tensor *  x = nullptr;
        ggml_tensor *  tok = nullptr;
        ggml_tensor *  ids = nullptr;
        ggml_tensor *  y = nullptr;
    };
    std::list<graph_entry> graphs;
    ggml_piece::staging hin, hout;
    float * out_host = nullptr;
    size_t  out_bytes = 0;
    int     out_pairs = 0;
    // streaming (moe_stream): three staging areas for a group of experts, and the backend that does the DMA
    ggml_backend_buffer_t stg_buf[3] = { nullptr, nullptr, nullptr };
    char *                stg_ptr[3] = { nullptr, nullptr, nullptr };
    std::vector<char>     stg_plain[3];
    size_t                stg_cap = 0;
    ggml_backend_t        ub = nullptr;

    ~ggml_moe_slots() override {
        for (auto & g_ : graphs) {
            ggml_free(g_.ctx);
        }
        if (galloc_dec != nullptr) {
            ggml_gallocr_free(galloc_dec);
        }
        if (galloc_pair != nullptr) {
            ggml_gallocr_free(galloc_pair);
        }
        if (hin.buf != nullptr) {
            ggml_backend_buffer_free(hin.buf);
        }
        if (hout.buf != nullptr) {
            ggml_backend_buffer_free(hout.buf);
        }
        for (auto * b : stg_buf) {
            if (b != nullptr) {
                ggml_backend_buffer_free(b);
            }
        }
        if (ub != nullptr) {
            ggml_backend_free(ub);
        }
        if (buf != nullptr) {
            ggml_backend_buffer_free(buf);
        }
        if (ctx != nullptr) {
            ggml_free(ctx);
        }
    }
};

class pdev_ggml final : public pdev {
public:
    explicit pdev_ggml(ggml_backend_dev_t dev) : dev_(dev) {
        name_ = ggml_backend_dev_name(dev);
        desc_ = ggml_backend_dev_description(dev);
        be_   = ggml_backend_dev_init(dev, nullptr);
        if (be_ == nullptr) {
            fprintf(stderr, "SGLX: cannot initialize the ggml device %s\n", name_.c_str());
        }
        weight_ = default_weight_for(desc_, ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU);
        host_buft_ = ggml_backend_dev_host_buffer_type(dev);
        if (getenv("GGML_SGLX_NO_PINNED") != nullptr) {
            host_buft_ = nullptr;
        }
        // Vulkan1, Vulkan2, ...: no pinned memory, own thread (GGML_SGLX_THREADS=0 / 1 forces it off / on for every ggml device)
        const bool vk_other = name_.rfind("Vulkan", 0) == 0 && name_ != "Vulkan0";
        const char * te = getenv("GGML_SGLX_THREADS");
        threaded_ = te != nullptr ? atoi(te) != 0 : vk_other;
        if (threaded_) {
            host_buft_ = nullptr;
            th_ = std::thread([this] {
                std::unique_lock<std::mutex> lk(mu_);
                for (;;) {
                    cv_.wait(lk, [&] { return job_.pending; });
                    job_t j = job_;
                    lk.unlock();
                    bool ok = j.fn ? j.fn() : submit_impl(j.p, j.x, j.N, j.out);
                    ok = wait_impl() && ok;
                    lk.lock();
                    ok_ = ok;
                    job_.pending = false;
                    cv_.notify_all();
                }
            });
        }
    }

    ~pdev_ggml() override {
        if (be_ != nullptr) {
            ggml_backend_free(be_);
        }
    }

    bool ok() const { return be_ != nullptr; }

    const std::string & name() const override { return name_; }
    const std::string & description() const override { return desc_; }
    double default_weight() const override { return weight_; }

    bool supports_mul_mat(ggml_type type) override {
        if (be_ == nullptr) {
            return false;
        }
        auto it = sup_.find(type);
        if (it != sup_.end()) {
            return it->second;
        }
        bool ok = true;
        // ask the device with probe tensors (no data is allocated): a few columns and one column
        for (int64_t N : { (int64_t) 1, (int64_t) 8 }) {
            ggml_init_params ip = { ggml_tensor_overhead() * 4 + 1024, nullptr, true };
            ggml_context * ctx = ggml_init(ip);
            const int64_t K = 512; // a multiple of every block size
            ggml_tensor * w = ggml_new_tensor_2d(ctx, type, K, 64);
            ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, N);
            ggml_tensor * y = ggml_mul_mat(ctx, w, x);
            ok = ok && ggml_backend_dev_supports_op(dev_, y);
            ggml_free(ctx);
        }
        sup_[type] = ok;
        return ok;
    }

    piece * piece_create(ggml_type type, int64_t K, int64_t rows) override {
        if (be_ == nullptr) {
            return nullptr;
        }
        auto * p = new ggml_piece();
        p->type = type;
        p->K    = K;
        p->rows = rows;
        ggml_init_params ip = { ggml_tensor_overhead() * 2 + 1024, nullptr, true };
        p->wctx = ggml_init(ip);
        p->W    = ggml_new_tensor_2d(p->wctx, type, K, rows);
        ggml_set_name(p->W, "sglx_w");
        p->wbuf = ggml_backend_alloc_ctx_tensors(p->wctx, be_);
        if (p->wbuf == nullptr) {
            fprintf(stderr, "SGLX: %s: cannot allocate %.1f MiB for %lld rows of %s\n", name_.c_str(),
                    (double) ggml_nbytes(p->W) / 1048576.0, (long long) rows, ggml_type_name(type));
            delete p;
            return nullptr;
        }
        ggml_backend_buffer_set_usage(p->wbuf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        p->galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(be_));
        return p;
    }

    bool piece_upload(piece * pp, const void * data, size_t bytes) override {
        auto * p = static_cast<ggml_piece *>(pp);
        if (bytes != ggml_nbytes(p->W)) {
            return false;
        }
        ggml_backend_tensor_set(p->W, data, 0, bytes);
        return true;
    }

    void piece_free(piece * pp) override {
        delete static_cast<ggml_piece *>(pp);
    }

    float * stage(ggml_piece::staging & s, size_t bytes) {
        if (bytes > s.cap) {
            if (s.buf != nullptr) {
                ggml_backend_buffer_free(s.buf);
                s.buf = nullptr;
            }
            s.ptr = nullptr;
            const size_t cap = std::max<size_t>(bytes, s.cap * 2);
            if (host_buft_ != nullptr) {
                s.buf = ggml_backend_buft_alloc_buffer(host_buft_, cap);
                if (s.buf != nullptr) {
                    s.ptr = (float *) ggml_backend_buffer_get_base(s.buf);
                }
            }
            if (s.ptr == nullptr) {
                s.plain.resize(cap / sizeof(float) + 1);
                s.ptr = s.plain.data();
            }
            s.cap = cap;
        }
        return s.ptr;
    }

    bool submit_impl(piece * pp, const float * x, int64_t N, float * out) {
        auto * p = static_cast<ggml_piece *>(pp);
        const size_t xbytes = (size_t) p->K * (size_t) N * sizeof(float);
        const size_t obytes = (size_t) p->rows * (size_t) N * sizeof(float);
        float * xs = stage(p->hx, xbytes);
        float * os = stage(p->ho, obytes);
        memcpy(xs, x, xbytes);
        const auto t0 = clk::now();
        ggml_piece::graph_entry * g = graph_for(p, N);
        if (g == nullptr) {
            return false;
        }
        const auto t1 = clk::now();
        if (!ggml_gallocr_alloc_graph(p->galloc, g->gf)) {
            fprintf(stderr, "SGLX: %s: cannot allocate the compute buffer for %lld columns\n", name_.c_str(), (long long) N);
            return false;
        }
        const auto t2 = clk::now();
        ggml_backend_tensor_set_async(be_, g->x, xs, 0, xbytes);
        const auto t3 = clk::now();
        const ggml_status st = ggml_backend_graph_compute_async(be_, g->gf);
        const auto t4 = clk::now();
        if (st != GGML_STATUS_SUCCESS) {
            fprintf(stderr, "SGLX: %s: graph_compute failed (%d)\n", name_.c_str(), (int) st);
            return false;
        }
        ggml_backend_tensor_get_async(be_, g->y, os, 0, obytes);
        p->out_host = out;
        p->out_bytes = obytes;
        pending_.push_back(p);
        const auto t5 = clk::now();
        auto us = [](clk::time_point a, clk::time_point b) { return std::chrono::duration<double, std::micro>(b - a).count(); };
        if (N == 1 && calls_++ > 2) { // not the first calls (shader compilation, first use)
            tm_[0] += us(t0, t1); tm_[1] += us(t1, t2); tm_[2] += us(t2, t3); tm_[3] += us(t3, t4); tm_[4] += us(t4, t5); ++tn_;
        }
        return true;
    }

    bool wait_impl() {
        if (be_ != nullptr) {
            ggml_backend_synchronize(be_);
        }
        for (ggml_piece * p : pending_) {
            memcpy(p->out_host, p->ho.ptr, p->out_bytes);
        }
        pending_.clear();
        for (ggml_moe_slots * m : moe_pending_) {
            memcpy(m->out_host, m->hout.ptr, m->out_bytes);
        }
        moe_pending_.clear();
        return true;
    }

    // A device other than the first Vulkan one cannot use pinned host memory (ggml-vulkan pins on device 0 only), so its async copies to and
    // from pageable memory submit and wait at once. Such a device runs submit + wait on its own thread, which keeps the host thread free to
    // start the other devices (it sleeps on a condition variable, no polling).
    bool submit(piece * p, const float * x, int64_t N, float * out) override {
        if (!threaded_) {
            return submit_impl(p, x, N, out);
        }
        std::unique_lock<std::mutex> lk(mu_);
        job_ = { p, x, N, out, true, nullptr };
        ok_ = true;
        cv_.notify_all();
        return true;
    }

    bool wait() override {
        if (!threaded_) {
            return wait_impl();
        }
        std::unique_lock<std::mutex> lk(mu_);
        cv_.wait(lk, [&] { return !job_.pending; });
        return ok_;
    }


    // ---- MoE expert slots ----

    bool supports_moe(const moe_shape & sh) override {
        if (be_ == nullptr) {
            return false;
        }
        ggml_init_params ip = { ggml_tensor_overhead() * 64 + 4096, nullptr, true };
        ggml_context * c = ggml_init(ip);
        const int64_t S = 4;
        ggml_tensor * g = ggml_new_tensor_3d(c, sh.t_gate, sh.n_embd, sh.n_ff, S);
        ggml_tensor * u = ggml_new_tensor_3d(c, sh.t_up, sh.n_embd, sh.n_ff, S);
        ggml_tensor * d = ggml_new_tensor_3d(c, sh.t_down, sh.n_ff, sh.n_embd, S);
        bool ok = true;
        for (int64_t n : { (int64_t) 1, (int64_t) 4 }) {
            ggml_tensor * x   = ggml_new_tensor_3d(c, GGML_TYPE_F32, sh.n_embd, 1, 1);
            ggml_tensor * ids = ggml_new_tensor_2d(c, GGML_TYPE_I32, n, 1);
            ggml_tensor * a = ggml_mul_mat_id(c, u, x, ids);
            ggml_tensor * b = ggml_mul_mat_id(c, g, x, ids);
            ggml_tensor * h = ggml_swiglu_split(c, b, a);
            ggml_tensor * y = ggml_mul_mat_id(c, d, h, ids);
            ok = ok && ggml_backend_dev_supports_op(dev_, a) && ggml_backend_dev_supports_op(dev_, b) && ggml_backend_dev_supports_op(dev_, h) &&
                 ggml_backend_dev_supports_op(dev_, y);
        }
        ggml_tensor * xs  = ggml_new_tensor_2d(c, GGML_TYPE_F32, sh.n_embd, 8);
        ggml_tensor * tok = ggml_new_tensor_1d(c, GGML_TYPE_I32, 16);
        ggml_tensor * xp  = ggml_get_rows(c, xs, tok);
        ok = ok && ggml_backend_dev_supports_op(dev_, xp);
        {
            ggml_tensor * x3  = ggml_reshape_3d(c, xp, sh.n_embd, 1, 16);
            ggml_tensor * ids = ggml_new_tensor_2d(c, GGML_TYPE_I32, 1, 16);
            ggml_tensor * a = ggml_mul_mat_id(c, u, x3, ids);
            ok = ok && ggml_backend_dev_supports_op(dev_, a);
        }
        ggml_free(c);
        return ok;
    }

    bool memory(size_t * free_bytes, size_t * total_bytes) override {
        ggml_backend_dev_memory(dev_, free_bytes, total_bytes);
        return *total_bytes != 0;
    }

    moe_slots * moe_create(const moe_shape & sh, int n_slots) override {
        if (be_ == nullptr || n_slots <= 0) {
            return nullptr;
        }
        auto * m = new ggml_moe_slots();
        m->sh = sh;
        m->n_slots = n_slots;
        ggml_init_params ip = { ggml_tensor_overhead() * 4 + 1024, nullptr, true };
        m->ctx = ggml_init(ip);
        m->g = ggml_new_tensor_3d(m->ctx, sh.t_gate, sh.n_embd, sh.n_ff, n_slots);
        m->u = ggml_new_tensor_3d(m->ctx, sh.t_up, sh.n_embd, sh.n_ff, n_slots);
        m->d = ggml_new_tensor_3d(m->ctx, sh.t_down, sh.n_ff, sh.n_embd, n_slots);
        ggml_set_name(m->g, "sglx_moe_gate");
        ggml_set_name(m->u, "sglx_moe_up");
        ggml_set_name(m->d, "sglx_moe_down");
        m->buf = ggml_backend_alloc_ctx_tensors(m->ctx, be_);
        if (m->buf == nullptr) {
            fprintf(stderr, "SGLX: %s: cannot allocate %d expert slots (%.1f MiB)\n", name_.c_str(), n_slots,
                    (double) (ggml_nbytes(m->g) + ggml_nbytes(m->u) + ggml_nbytes(m->d)) / 1048576.0);
            delete m;
            return nullptr;
        }
        ggml_backend_buffer_set_usage(m->buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        m->galloc_dec  = ggml_gallocr_new(ggml_backend_get_default_buffer_type(be_));
        m->galloc_pair = ggml_gallocr_new(ggml_backend_get_default_buffer_type(be_));
        return m;
    }

    void moe_free(moe_slots * s) override {
        if (be_ != nullptr) {
            ggml_backend_synchronize(be_);
        }
        moe_pending_.erase(std::remove(moe_pending_.begin(), moe_pending_.end(), static_cast<ggml_moe_slots *>(s)), moe_pending_.end());
        delete static_cast<ggml_moe_slots *>(s);
    }

    bool moe_upload(moe_slots * s, int slot, const void * gate, const void * up, const void * down) override {
        auto * m = static_cast<ggml_moe_slots *>(s);
        if (slot < 0 || slot >= m->n_slots) {
            return false;
        }
        ggml_backend_tensor_set(m->g, gate, (size_t) slot * m->g->nb[2], m->g->nb[2]);
        ggml_backend_tensor_set(m->u, up,   (size_t) slot * m->u->nb[2], m->u->nb[2]);
        ggml_backend_tensor_set(m->d, down, (size_t) slot * m->d->nb[2], m->d->nb[2]);
        return true;
    }


    bool moe_stream(moe_slots * s, int ring_first, int ring_n, const moe_src * ex, int n_exp, const float * x, int64_t N_x, const int32_t * pair_exp,
                    const int32_t * pair_tok, int n_pairs, float * out) override {
        auto * m = static_cast<ggml_moe_slots *>(s);
        if (n_exp <= 0 || n_pairs <= 0) {
            return true;
        }
        if (ring_n < 2 || ring_first < 0 || ring_first + ring_n > m->n_slots || N_x > 1024) {
            return false;
        }
        const int64_t K = m->sh.n_embd;
        const int C = ring_n / 2;                                  // experts per group; two groups in the ring
        const int groups = (n_exp + C - 1) / C;
        const size_t bg = m->g->nb[2], bu = m->u->nb[2], bd = m->d->nb[2];
        const size_t cap = (size_t) C * (bg + bu + bd);
        if (cap > m->stg_cap) {
            for (int i = 0; i < 3; ++i) {
                if (m->stg_buf[i] != nullptr) {
                    ggml_backend_buffer_free(m->stg_buf[i]);
                    m->stg_buf[i] = nullptr;
                }
                m->stg_ptr[i] = nullptr;
                if (host_buft_ != nullptr) {
                    m->stg_buf[i] = ggml_backend_buft_alloc_buffer(host_buft_, cap);
                    if (m->stg_buf[i] != nullptr) {
                        m->stg_ptr[i] = (char *) ggml_backend_buffer_get_base(m->stg_buf[i]);
                    }
                }
                if (m->stg_ptr[i] == nullptr) {
                    m->stg_plain[i].resize(cap + 64);
                    m->stg_ptr[i] = m->stg_plain[i].data();
                }
            }
            m->stg_cap = cap;
        }
        if (m->ub == nullptr) {
            m->ub = ggml_backend_dev_init(dev_, nullptr);
            if (m->ub == nullptr) {
                return false;
            }
        }
        // the pairs grouped by expert
        std::vector<int> start((size_t) n_exp + 1, 0), order((size_t) n_pairs);
        for (int i = 0; i < n_pairs; ++i) {
            if (pair_exp[i] < 0 || pair_exp[i] >= n_exp) {
                return false;
            }
            ++start[(size_t) pair_exp[i] + 1];
        }
        for (int e = 0; e < n_exp; ++e) {
            start[(size_t) e + 1] += start[(size_t) e];
        }
        {
            std::vector<int> fill(start.begin(), start.end() - 1);
            for (int i = 0; i < n_pairs; ++i) {
                order[(size_t) fill[(size_t) pair_exp[i]]++] = i;
            }
        }
        std::mutex mu;
        std::condition_variable cvs;
        int staged = 0, uploaded = 0, computed = 0;
        bool failed = false;
        auto fail = [&] { std::lock_guard<std::mutex> lk(mu); failed = true; cvs.notify_all(); };
        // The DMA thread copies a staged group into its half of the ring with the blocking tensor_set (ggml-vulkan sleeps in waitForFences there; the
        // asynchronous path would spin on a core for the whole transfer, and the host has two). The main thread stages the groups ahead of the DMA (memcpy from the
        // mmap into pinned memory) and computes the groups that arrived, so the three overlap.
        std::thread t_dma([&] {
            for (int i = 0; i < groups; ++i) {
                {
                    std::unique_lock<std::mutex> lk(mu);
                    cvs.wait(lk, [&] { return failed || (staged > i && computed >= i - 1); });
                    if (failed) { return; }
                }
                const char * src = m->stg_ptr[i % 3];
                const int half = i & 1;
                const int e0 = i * C, e1 = std::min(n_exp, e0 + C);
                const size_t n = (size_t) (e1 - e0);
                ggml_tensor * ts[3] = { m->g, m->u, m->d };
                const size_t bs[3] = { bg, bu, bd };
                size_t off = 0;
                for (int k = 0; k < 3; ++k) {
                    const size_t slot0 = (size_t) ring_first + (size_t) half * (size_t) C;
                    ggml_backend_tensor_set(ts[k], src + off, slot0 * bs[k], n * bs[k]);
                    off += (size_t) C * bs[k];
                }
                {
                    std::lock_guard<std::mutex> lk(mu);
                    uploaded = i + 1;
                }
                cvs.notify_all();
            }
        });
        auto stage_group = [&](int i) {
            char * dst = m->stg_ptr[i % 3];
            const int e0 = i * C, e1 = std::min(n_exp, e0 + C);
            for (int k = 0; k < 3; ++k) {
                const size_t b = k == 0 ? bg : (k == 1 ? bu : bd);
                const size_t off = (size_t) C * (k == 0 ? 0 : (k == 1 ? bg : bg + bu));
                for (int e = e0; e < e1; ++e) {
                    const void * src = k == 0 ? ex[e].gate : (k == 1 ? ex[e].up : ex[e].down);
                    memcpy(dst + off + (size_t) (e - e0) * b, src, b);
                }
            }
            {
                std::lock_guard<std::mutex> lk(mu);
                staged = i + 1;
            }
            cvs.notify_all();
        };
        bool ok = true;
        std::vector<int32_t> ps, pt;
        std::vector<float> tmp;
        for (int i = 0; i < groups && ok; ++i) {
            // stage ahead: a staging area is free once its group has been uploaded
            for (;;) {
                int st_n, up_n;
                { std::lock_guard<std::mutex> lk(mu); st_n = staged; up_n = uploaded; }
                if (st_n >= groups || st_n - up_n >= 3) { break; }
                stage_group(st_n);
            }
            {
                std::unique_lock<std::mutex> lk(mu);
                cvs.wait(lk, [&] { return failed || uploaded > i; });
                if (failed) { ok = false; break; }
            }
            const int e0 = i * C, e1 = std::min(n_exp, e0 + C);
            const int p0 = start[(size_t) e0], p1 = start[(size_t) e1];
            for (int p = p0; p < p1 && ok; p += 1024) {
                const int n = std::min(1024, p1 - p);
                ps.resize((size_t) n);
                pt.resize((size_t) n);
                for (int q = 0; q < n; ++q) {
                    const int pi = order[(size_t) (p + q)];
                    ps[(size_t) q] = ring_first + (i & 1) * C + (pair_exp[pi] - e0);
                    pt[(size_t) q] = pair_tok[pi];
                }
                tmp.resize((size_t) K * (size_t) n);
                ok = moe_submit_impl(m, x, N_x, ps.data(), pt.data(), n, tmp.data());
                // while the device computes, stage the next group
                if (ok) {
                    int st_n, up_n;
                    { std::lock_guard<std::mutex> lk(mu); st_n = staged; up_n = uploaded; }
                    if (st_n < groups && st_n - up_n < 3) { stage_group(st_n); }
                    ok = wait_impl();
                }
                if (ok) {
                    for (int q = 0; q < n; ++q) {
                        memcpy(out + (size_t) order[(size_t) (p + q)] * (size_t) K, tmp.data() + (size_t) q * (size_t) K, (size_t) K * sizeof(float));
                    }
                }
            }
            {
                std::lock_guard<std::mutex> lk(mu);
                computed = i + 1;
                if (!ok) { failed = true; }
            }
            cvs.notify_all();
        }
        if (!ok) { fail(); }
        t_dma.join();
        return ok;
    }

    bool moe_submit(moe_slots * s, const float * x, int64_t N_x, const int32_t * pair_slot, const int32_t * pair_tok, int n_pairs, float * out) override {
        auto * m = static_cast<ggml_moe_slots *>(s);
        if (!threaded_) {
            return moe_submit_impl(m, x, N_x, pair_slot, pair_tok, n_pairs, out);
        }
        std::unique_lock<std::mutex> lk(mu_);
        job_ = { nullptr, nullptr, 0, nullptr, true, [=] { return moe_submit_impl(m, x, N_x, pair_slot, pair_tok, n_pairs, out); } };
        ok_ = true;
        cv_.notify_all();
        return true;
    }

    std::string stats() override {
        if (tn_ == 0) {
            return std::string();
        }
        char b[200];
        snprintf(b, sizeof(b), "graph %.0f, alloc %.0f, set %.0f, compute %.0f, get %.0f us (N=1 mean of %llu)", tm_[0] / tn_, tm_[1] / tn_, tm_[2] / tn_, tm_[3] / tn_, tm_[4] / tn_, (unsigned long long) tn_);
        return b;
    }

private:
    // the graph for `mode` (0: n_ids ids for one token, 1: P pairs of N_x tokens); nullptr if it cannot be made
    ggml_moe_slots::graph_entry * moe_graph(ggml_moe_slots * m, int mode, int64_t a, int64_t b) {
        for (auto it = m->graphs.begin(); it != m->graphs.end(); ++it) {
            if (it->mode == mode && it->a == a && it->b == b) {
                if (it != m->graphs.begin()) {
                    m->graphs.splice(m->graphs.begin(), m->graphs, it);
                }
                return &m->graphs.front();
            }
        }
        const moe_shape & sh = m->sh;
        ggml_moe_slots::graph_entry e;
        e.mode = mode; e.a = a; e.b = b;
        ggml_init_params ip = { ggml_tensor_overhead() * 24 + ggml_graph_overhead_custom(24, false) + 1024, nullptr, true };
        e.ctx = ggml_init(ip);
        ggml_tensor * x3 = nullptr;
        if (mode == 0) {
            e.x   = ggml_new_tensor_3d(e.ctx, GGML_TYPE_F32, sh.n_embd, 1, 1);
            e.ids = ggml_new_tensor_2d(e.ctx, GGML_TYPE_I32, a, 1);
            x3 = e.x;
        } else {
            e.x   = ggml_new_tensor_2d(e.ctx, GGML_TYPE_F32, sh.n_embd, a);
            e.tok = ggml_new_tensor_1d(e.ctx, GGML_TYPE_I32, b);
            e.ids = ggml_new_tensor_2d(e.ctx, GGML_TYPE_I32, 1, b);
            ggml_set_input(e.tok);
            x3 = ggml_reshape_3d(e.ctx, ggml_get_rows(e.ctx, e.x, e.tok), sh.n_embd, 1, b);
        }
        ggml_set_input(e.x);
        ggml_set_input(e.ids);
        ggml_tensor * up = ggml_mul_mat_id(e.ctx, m->u, x3, e.ids);
        ggml_tensor * gt = ggml_mul_mat_id(e.ctx, m->g, x3, e.ids);
        ggml_tensor * h  = ggml_swiglu_split(e.ctx, gt, up);
        e.y = ggml_mul_mat_id(e.ctx, m->d, h, e.ids);
        ggml_set_output(e.y);
        e.gf = ggml_new_graph_custom(e.ctx, 24, false);
        ggml_build_forward_expand(e.gf, e.y);
        ggml_gallocr_t ga = mode == 0 ? m->galloc_dec : m->galloc_pair;
        if ((mode == 0 && !m->dec_reserved) || (mode == 1 && b > m->reserved_P)) {
            // the compute buffer of the largest graph of this kind; smaller graphs fit in it (see graph_for)
            if (!ggml_gallocr_reserve(ga, e.gf)) {
                fprintf(stderr, "SGLX: %s: cannot reserve the MoE compute buffer (mode %d, %lld/%lld)\n", name_.c_str(), mode, (long long) a, (long long) b);
                ggml_free(e.ctx);
                return nullptr;
            }
            for (auto it = m->graphs.begin(); it != m->graphs.end();) {
                if (it->mode == mode) {
                    ggml_free(it->ctx);
                    it = m->graphs.erase(it);
                } else {
                    ++it;
                }
            }
            if (mode == 1) {
                m->reserved_P = b;
            } else {
                m->dec_reserved = true;
            }
        }
        m->graphs.push_front(e);
        int n_dec = 0, n_pair = 0;
        for (auto it = m->graphs.begin(); it != m->graphs.end();) {
            int & cnt = it->mode == 0 ? n_dec : n_pair;
            if (++cnt > (it->mode == 0 ? 9 : 4)) {
                ggml_free(it->ctx);
                it = m->graphs.erase(it);
            } else {
                ++it;
            }
        }
        return &m->graphs.front();
    }

    bool moe_submit_impl(ggml_moe_slots * m, const float * x, int64_t N_x, const int32_t * pair_slot, const int32_t * pair_tok, int n_pairs, float * out) {
        const int64_t K = m->sh.n_embd;
        const bool one = N_x == 1;
        int64_t P = n_pairs;
        if (!one) {
            P = P <= 64 ? 64 : (P <= 256 ? 256 : 1024);
            if (n_pairs > 1024 || N_x > 1024) {
                fprintf(stderr, "SGLX: %s: moe_submit: at most 1024 pairs and 1024 columns per call\n", name_.c_str());
                return false;
            }
            // the largest graph first, so that the compute buffer is reserved once
            if (m->reserved_P == 0 && moe_graph(m, 1, 1024, 1024) == nullptr) {
                return false;
            }
        } else if (n_pairs < 1 || n_pairs > 8) {
            fprintf(stderr, "SGLX: %s: moe_submit: 1..8 ids for one token\n", name_.c_str());
            return false;
        } else if (!m->dec_reserved && moe_graph(m, 0, 8, 0) == nullptr) {
            return false;
        }
        ggml_moe_slots::graph_entry * g = one ? moe_graph(m, 0, n_pairs, 0) : moe_graph(m, 1, 1024, P); // the x tensor of a pair graph has 1024 columns
        if (g == nullptr) {
            return false;
        }
        const size_t xb = (size_t) K * (size_t) N_x * sizeof(float);
        const size_t ib = (size_t) (one ? n_pairs : P) * sizeof(int32_t);
        const size_t ob = (size_t) K * (size_t) (one ? n_pairs : P) * sizeof(float);
        auto up16 = [](size_t v) { return (v + 15) & ~(size_t) 15; };
        const size_t off_tok = up16(xb), off_ids = off_tok + up16(one ? 0 : ib), total = off_ids + up16(ib);
        char * in = (char *) stage(m->hin, total);
        float * os = stage(m->hout, ob);
        memcpy(in, x, xb);
        int32_t * tokv = (int32_t *) (in + off_tok);
        int32_t * idsv = (int32_t *) (in + off_ids);
        for (int64_t i = 0; i < (one ? n_pairs : P); ++i) {
            const bool real = i < n_pairs;
            idsv[i] = real ? pair_slot[i] : 0;
            if (!one) {
                tokv[i] = real ? pair_tok[i] : 0;
            }
        }
        ggml_gallocr_t ga = one ? m->galloc_dec : m->galloc_pair;
        if (!ggml_gallocr_alloc_graph(ga, g->gf)) {
            fprintf(stderr, "SGLX: %s: cannot allocate the MoE compute buffer\n", name_.c_str());
            return false;
        }
        ggml_backend_tensor_set_async(be_, g->x, in, 0, xb);
        if (!one) {
            ggml_backend_tensor_set_async(be_, g->tok, tokv, 0, ib);
        }
        ggml_backend_tensor_set_async(be_, g->ids, idsv, 0, ib);
        const ggml_status st = ggml_backend_graph_compute_async(be_, g->gf);
        if (st != GGML_STATUS_SUCCESS) {
            fprintf(stderr, "SGLX: %s: MoE graph_compute failed (%d)\n", name_.c_str(), (int) st);
            return false;
        }
        const size_t obytes = (size_t) K * (size_t) n_pairs * sizeof(float);
        ggml_backend_tensor_get_async(be_, g->y, os, 0, obytes);
        m->out_host = out;
        m->out_bytes = obytes;
        m->out_pairs = n_pairs;
        moe_pending_.push_back(m);
        return true;
    }

    ggml_piece::graph_entry * graph_for(ggml_piece * p, int64_t N) {
        for (auto it = p->graphs.begin(); it != p->graphs.end(); ++it) {
            if (it->N == N) {
                if (it != p->graphs.begin()) {
                    p->graphs.splice(p->graphs.begin(), p->graphs, it);
                }
                return &p->graphs.front();
            }
        }
        ggml_piece::graph_entry g;
        g.N = N;
        ggml_init_params ip = { ggml_tensor_overhead() * 8 + ggml_graph_overhead_custom(8, false) + 1024, nullptr, true };
        g.ctx = ggml_init(ip);
        g.x   = ggml_new_tensor_2d(g.ctx, GGML_TYPE_F32, p->K, N);
        ggml_set_name(g.x, "sglx_x");
        ggml_set_input(g.x);
        g.y   = ggml_mul_mat(g.ctx, p->W, g.x);
        ggml_set_name(g.y, "sglx_y");
        ggml_set_output(g.y);
        g.gf  = ggml_new_graph_custom(g.ctx, 8, false);
        ggml_build_forward_expand(g.gf, g.y);
        if (N > p->reserved_N) {
            // the compute buffer of the largest graph; smaller graphs fit in it
            if (!ggml_gallocr_reserve(p->galloc, g.gf)) {
                fprintf(stderr, "SGLX: %s: cannot reserve the compute buffer for %lld columns\n", name_.c_str(), (long long) N);
                ggml_free(g.ctx);
                return nullptr;
            }
            p->reserved_N = N;
            // ggml_gallocr_reserve replaced the compute buffer: the tensors of every graph made before still point at the freed buffer (their
            // data pointers equal the new ones, so ggml_gallocr_alloc_graph would not notice and set_tensor_async would read a dead
            // ggml_backend_buffer, an "unsupported buffer type" assert in ggml-vulkan: llama-perplexity -ub 320 after a smaller graph). Drop them.
            for (auto & old : p->graphs) {
                ggml_free(old.ctx);
            }
            p->graphs.clear();
        }
        p->graphs.push_front(g);
        while (p->graphs.size() > 4) {
            ggml_free(p->graphs.back().ctx);
            p->graphs.pop_back();
        }
        return &p->graphs.front();
    }

    ggml_backend_dev_t dev_ = nullptr;
    ggml_backend_t     be_  = nullptr;
    ggml_backend_buffer_type_t host_buft_ = nullptr;
    bool                       threaded_ = false;
    struct job_t { piece * p = nullptr; const float * x = nullptr; int64_t N = 0; float * out = nullptr; bool pending = false; std::function<bool()> fn; };
    job_t                      job_;
    bool                       ok_ = true;
    std::mutex                 mu_;
    std::condition_variable    cv_;
    std::thread                th_;
    std::vector<ggml_piece *> pending_;
    std::vector<ggml_moe_slots *> moe_pending_;
    std::string        name_;
    std::string        desc_;
    double             weight_ = 1.0;
    std::map<ggml_type, bool> sup_;
    using clk = std::chrono::steady_clock;
    double   tm_[5] = { 0, 0, 0, 0, 0 };
    uint64_t tn_ = 0;
    uint64_t calls_ = 0;
};

} // namespace

std::unique_ptr<pdev> make_pdev_ggml(ggml_backend_dev_t dev) {
    auto p = std::make_unique<pdev_ggml>(dev);
    if (!p->ok()) {
        return nullptr;
    }
    return p;
}

} // namespace sglx
