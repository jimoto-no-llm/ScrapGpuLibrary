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
                    bool ok = submit_impl(j.p, j.x, j.N, j.out);
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
        job_ = { p, x, N, out, true };
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

    std::string stats() override {
        if (tn_ == 0) {
            return std::string();
        }
        char b[200];
        snprintf(b, sizeof(b), "graph %.0f, alloc %.0f, set %.0f, compute %.0f, get %.0f us (N=1 mean of %llu)", tm_[0] / tn_, tm_[1] / tn_, tm_[2] / tn_, tm_[3] / tn_, tm_[4] / tn_, (unsigned long long) tn_);
        return b;
    }

private:
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
    struct job_t { piece * p = nullptr; const float * x = nullptr; int64_t N = 0; float * out = nullptr; bool pending = false; };
    job_t                      job_;
    bool                       ok_ = true;
    std::mutex                 mu_;
    std::condition_variable    cv_;
    std::thread                th_;
    std::vector<ggml_piece *> pending_;
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
