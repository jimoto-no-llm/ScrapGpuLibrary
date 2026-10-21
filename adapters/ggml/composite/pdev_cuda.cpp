// pdev over an NVIDIA GPU through the CUDA backend of this repository (src/backends/cuda: the Driver API loaded with dlopen, hand-made PTX
// kernels; no CUDA toolkit). Named "CUDA<i>" in GGML_SGLX_SPLIT, i = the index in the device list of the backend: the GPUs of the driver in its
// own order, or the ones named in SCRAPGPU_CUDA_DEVICES (comma-separated substrings of the device names) in that order, e.g.
//
//   SCRAPGPU_CUDA_DEVICES="GT 730,GT 710" GGML_SGLX_SPLIT=Vulkan0:0.5,Vulkan1:0.2,CUDA0:0.2,CUDA1:0.1
//
// Weights are repacked to the R32 layout on the host while they are uploaded (cuda/r32_transfer.h, the layout of the CL11 kernels: coalesced
// 128-bit loads, one row per lane). The host side follows docs/benchmarks/moe-ep-20261021/microbench MB1 / MB2: pinned staging for the input columns and the result (a transfer from
// pageable memory is slower), submit() = copy + kernel + copy on the device's stream and returns at once, wait() = one stream synchronize.
//
// A Fermi card and a Kepler card cannot share a process with the 390 driver: the second generation fails to create its context, piece_create()
// returns nullptr and the composite reports it. Use two Kepler cards (GT 730 + GT 710), or run the Fermi card in another process.
//
// Built only with SGLX_WITH_CUDA (adapters/ggml/CMakeLists.txt option CL11_WITH_CUDA); otherwise make_pdev_cuda() returns nullptr.

#include "composite.h"
#include "pdev.h"

#ifdef SGLX_WITH_CUDA

#include "cuda/device.h"
#include "cuda/r32_transfer.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace sglx {
namespace {

using namespace scrapgpu_cuda;

bool wtype_of(ggml_type t, WType & out) {
    if (t == GGML_TYPE_Q4_0) { out = WType::Q4_0; return true; }
    if (t == GGML_TYPE_Q6_K) { out = WType::Q6_K; return true; }
    return false;
}

struct cuda_piece : piece {
    WType    type = WType::Q4_0;
    int64_t  K = 0;
    int64_t  rows = 0;
    uint64_t w = 0;     // device: rows * row_bytes
    uint64_t xd = 0;    // device: the input columns
    uint64_t od = 0;    // device: the result
    float *  xh = nullptr; // pinned
    float *  oh = nullptr; // pinned
    size_t   xcap = 0;
    size_t   ocap = 0;
    // the result of the last submit(), copied out of the pinned memory by wait()
    float *  out_host = nullptr;
    size_t   out_bytes = 0;
};

class pdev_cuda final : public pdev {
public:
    pdev_cuda(Device * d, int index) : d_(d) {
        name_ = "CUDA" + std::to_string(index);
        desc_ = d->info().name;
        weight_ = default_weight_for(desc_, false);
    }

    const std::string & name() const override { return name_; }
    const std::string & description() const override { return desc_; }
    double default_weight() const override { return weight_; }

    bool supports_mul_mat(ggml_type type) override {
        WType t;
        return wtype_of(type, t) && d_->init();
    }

    piece * piece_create(ggml_type type, int64_t K, int64_t rows) override {
        WType t;
        if (!wtype_of(type, t) || K % wtype_block_size(t) != 0 || K <= 0 || rows <= 0) {
            return nullptr;
        }
        if (!d_->init()) {
            fprintf(stderr, "SGLX: %s: %s\n", name_.c_str(), d_->last_error().c_str());
            return nullptr;
        }
        auto * p = new cuda_piece();
        p->type = t;
        p->K = K;
        p->rows = rows;
        const uint64_t bytes = wtype_row_bytes(t, K) * (uint64_t) rows;
        if (!d_->alloc((size_t) bytes, p->w)) {
            fprintf(stderr, "SGLX: %s: cannot allocate %.1f MiB: %s\n", name_.c_str(), (double) bytes / 1048576.0, d_->last_error().c_str());
            delete p;
            return nullptr;
        }
        return p;
    }

    bool piece_upload(piece * pp, const void * data, size_t bytes) override {
        auto * p = static_cast<cuda_piece *>(pp);
        if (bytes != (size_t) (wtype_row_bytes(p->type, p->K) * (uint64_t) p->rows)) {
            return false;
        }
        if (!r32_upload(*d_, p->type, p->K, p->rows, data, p->w, d_->stream_default())) {
            fprintf(stderr, "SGLX: %s: upload failed: %s\n", name_.c_str(), d_->last_error().c_str());
            return false;
        }
        return true;
    }

    void piece_free(piece * pp) override {
        auto * p = static_cast<cuda_piece *>(pp);
        // a submit() that was not waited for still reads these buffers
        d_->stream_sync(d_->stream_default());
        pending_.erase(std::remove(pending_.begin(), pending_.end(), p), pending_.end());
        d_->free(p->w);
        d_->free(p->xd);
        d_->free(p->od);
        d_->host_free(p->xh);
        d_->host_free(p->oh);
        delete p;
    }

    bool submit(piece * pp, const float * x, int64_t N, float * out) override {
        auto * p = static_cast<cuda_piece *>(pp);
        const size_t xbytes = (size_t) p->K * (size_t) N * sizeof(float);
        const size_t obytes = (size_t) p->rows * (size_t) N * sizeof(float);
        if (!grow(p, xbytes, obytes)) {
            return false;
        }
        CUstream s = d_->stream_default();
        memcpy(p->xh, x, xbytes);
        if (!d_->h2d(p->xd, p->xh, xbytes, s) ||
            !d_->mul_mat(s, p->type, p->w, (uint32_t) p->rows, (uint32_t) p->K, p->xd, (uint32_t) N, p->od) ||
            !d_->d2h(p->oh, p->od, obytes, s)) {
            fprintf(stderr, "SGLX: %s: submit failed: %s\n", name_.c_str(), d_->last_error().c_str());
            return false;
        }
        p->out_host = out;
        p->out_bytes = obytes;
        pending_.push_back(p);
        return true;
    }

    bool wait() override {
        if (pending_.empty()) {
            return true;
        }
        const bool ok = d_->stream_sync(d_->stream_default());
        if (!ok) {
            fprintf(stderr, "SGLX: %s: stream synchronize failed: %s\n", name_.c_str(), d_->last_error().c_str());
        }
        for (cuda_piece * p : pending_) {
            if (ok) {
                memcpy(p->out_host, p->oh, p->out_bytes);
            }
        }
        pending_.clear();
        return ok;
    }

private:
    // device and pinned buffers for the input columns and the result of at least xbytes / obytes (grown with headroom)
    bool grow(cuda_piece * p, size_t xbytes, size_t obytes) {
        if (xbytes > p->xcap) {
            const size_t cap = std::max<size_t>(xbytes, p->xcap * 2);
            d_->stream_sync(d_->stream_default());
            d_->free(p->xd);
            d_->host_free(p->xh);
            p->xd = 0;
            p->xh = nullptr;
            p->xcap = 0;
            p->xh = static_cast<float *>(d_->host_alloc(cap));
            if (p->xh == nullptr || !d_->alloc(cap, p->xd)) {
                return false;
            }
            p->xcap = cap;
        }
        if (obytes > p->ocap) {
            const size_t cap = std::max<size_t>(obytes, p->ocap * 2);
            d_->stream_sync(d_->stream_default());
            d_->free(p->od);
            d_->host_free(p->oh);
            p->od = 0;
            p->oh = nullptr;
            p->ocap = 0;
            p->oh = static_cast<float *>(d_->host_alloc(cap));
            if (p->oh == nullptr || !d_->alloc(cap, p->od)) {
                return false;
            }
            p->ocap = cap;
        }
        return true;
    }

    Device *                  d_ = nullptr;
    std::string               name_;
    std::string               desc_;
    double                    weight_ = 1.0;
    std::vector<cuda_piece *> pending_;
};

} // namespace

std::unique_ptr<pdev> make_pdev_cuda(int index) {
    Device * d = Device::get(index);
    if (d == nullptr) {
        return nullptr;
    }
    return std::make_unique<pdev_cuda>(d, index);
}

int cuda_device_count() {
    return device_count();
}

} // namespace sglx

#else // !SGLX_WITH_CUDA

namespace sglx {

std::unique_ptr<pdev> make_pdev_cuda(int) {
    return nullptr;
}

int cuda_device_count() {
    return 0;
}

} // namespace sglx

#endif
