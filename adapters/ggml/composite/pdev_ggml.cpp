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
#include <cstdio>
#include <list>
#include <map>
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

    ~ggml_piece() override {
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

    bool submit(piece * pp, const float * x, int64_t N, float * out) override {
        auto * p = static_cast<ggml_piece *>(pp);
        ggml_piece::graph_entry * g = graph_for(p, N);
        if (g == nullptr) {
            return false;
        }
        if (!ggml_gallocr_alloc_graph(p->galloc, g->gf)) {
            fprintf(stderr, "SGLX: %s: cannot allocate the compute buffer for %lld columns\n", name_.c_str(), (long long) N);
            return false;
        }
        ggml_backend_tensor_set_async(be_, g->x, x, 0, (size_t) p->K * (size_t) N * sizeof(float));
        const ggml_status st = ggml_backend_graph_compute_async(be_, g->gf);
        if (st != GGML_STATUS_SUCCESS) {
            fprintf(stderr, "SGLX: %s: graph_compute failed (%d)\n", name_.c_str(), (int) st);
            return false;
        }
        ggml_backend_tensor_get_async(be_, g->y, out, 0, (size_t) p->rows * (size_t) N * sizeof(float));
        return true;
    }

    bool wait() override {
        if (be_ != nullptr) {
            ggml_backend_synchronize(be_);
        }
        return true;
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
    std::string        name_;
    std::string        desc_;
    double             weight_ = 1.0;
    std::map<ggml_type, bool> sup_;
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
