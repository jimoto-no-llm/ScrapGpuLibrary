// pdev over a CL11 device of this plugin (the OpenCL 1.1 backend of src/backends/opencl11): the existing tuned mat-vec / mat-mat kernels
// (cl11_op_mul_mat) on a slice of the rows of a weight in the R32 layout.
//
// The host side follows the measurements of docs/benchmarks/moe-ep-20261021/microbench: the input columns and the result are staged in
// pinned host memory (CL_MEM_ALLOC_HOST_PTR, kept mapped; a non-blocking transfer from pageable memory is slow on NVIDIA 390 and takes
// 100 ms on Fermi), submit() enqueues write + kernel + read and flushes, wait() is one clFinish. The device mutex is taken for every call
// into the CL11 state (the llama.cpp backends of the same device use the same cl11_device).

#include "composite.h"
#include "pdev.h"
#include "sgl_trace.h"

#include "opencl11/buffer.h"
#include "opencl11/context.h"
#include "opencl11/device_state.h"
#include "opencl11/ops.h"
#include "opencl11/r32.h"
#include "opencl11/tensor.h"
#include "tensor/tensor.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <string>
#include <vector>

namespace sglx {
namespace {

scrapgpu_dtype dtype_of(ggml_type type) {
    switch (type) {
        case GGML_TYPE_F32:  return SCRAPGPU_DTYPE_F32;
        case GGML_TYPE_F16:  return SCRAPGPU_DTYPE_F16;
        case GGML_TYPE_Q4_0: return SCRAPGPU_DTYPE_Q4_0;
        case GGML_TYPE_Q8_0: return SCRAPGPU_DTYPE_Q8_0;
        case GGML_TYPE_Q4_K: return SCRAPGPU_DTYPE_Q4_K;
        case GGML_TYPE_Q6_K: return SCRAPGPU_DTYPE_Q6_K;
        default:             return SCRAPGPU_DTYPE_UNKNOWN;
    }
}

// pinned host memory of the device: a cl_mem created with CL_MEM_ALLOC_HOST_PTR and kept mapped; falls back to malloc'd memory
struct pinned {
    cl_mem mem  = nullptr;
    void * ptr  = nullptr;
    size_t size = 0;
    bool   fallback = false;
};

struct cl_piece : piece {
    cl11_device *  d = nullptr;
    ggml_type      type = GGML_TYPE_F32;
    scrapgpu_dtype dt = SCRAPGPU_DTYPE_UNKNOWN;
    int64_t        K = 0;
    int64_t        rows = 0;
    bool           repacked = false;
    size_t         wbytes = 0;
    cl_mem         w = nullptr;

    cl_mem xm = nullptr; // device copy of the input columns
    cl_mem om = nullptr; // device result
    size_t xcap = 0;
    size_t ocap = 0;
    pinned xp, op;

    // the transfer of the last submit() that wait() copies out of the pinned memory
    float * out_host = nullptr;
    size_t  out_bytes = 0;
};

class pdev_cl11 final : public pdev {
public:
    explicit pdev_cl11(cl11_device * d) : d_(d) {
        name_ = d->name;
        desc_ = d->info.name;
        weight_ = default_weight_for(desc_, false);
    }

    const std::string & name() const override { return name_; }
    const std::string & description() const override { return desc_; }
    double default_weight() const override { return weight_; }

    bool supports_mul_mat(ggml_type type) override {
        auto it = sup_.find(type);
        if (it != sup_.end()) {
            return it->second;
        }
        bool ok = false;
        const scrapgpu_dtype dt = dtype_of(type);
        if (dt != SCRAPGPU_DTYPE_UNKNOWN) {
            std::lock_guard<std::mutex> lock(d_->mutex);
            if (cl11_ensure_context(d_)) {
                const int64_t K = 512;
                const int64_t ne_w[4] = { K, 64, 1, 1 };
                const int64_t ne_x[4] = { K, 1, 1, 1 };
                const int64_t ne_y[4] = { 64, 1, 1, 1 };
                const sgt::TensorDesc s0 = sgt::make_contiguous(dt, ne_w);
                const sgt::TensorDesc s1 = sgt::make_contiguous(SCRAPGPU_DTYPE_F32, ne_x);
                const sgt::TensorDesc dd = sgt::make_contiguous(SCRAPGPU_DTYPE_F32, ne_y);
                ok = cl11_supports_mul_mat(s0, sgt::r32_eligible(s0), s1, dd);
            }
        }
        sup_[type] = ok;
        return ok;
    }

    piece * piece_create(ggml_type type, int64_t K, int64_t rows) override {
        const scrapgpu_dtype dt = dtype_of(type);
        if (dt == SCRAPGPU_DTYPE_UNKNOWN) {
            return nullptr;
        }
        std::lock_guard<std::mutex> lock(d_->mutex);
        if (!cl11_ensure_context(d_) || !cl11_ensure_program(d_)) {
            return nullptr;
        }
        auto * p = new cl_piece();
        p->d = d_;
        p->type = type;
        p->dt = dt;
        p->K = K;
        p->rows = rows;
        const int64_t ne[4] = { K, rows, 1, 1 };
        const sgt::TensorDesc desc = sgt::make_contiguous(dt, ne);
        p->repacked = sgt::r32_eligible(desc);
        p->wbytes = (size_t) desc.nb[1] * (size_t) rows;
        size_t alloc = 0;
        if (!cl11_alloc_buffer(d_->context, d_->queue, d_->info, d_->name.c_str(), "sglx pdev_cl11", d_->strict_alloc_dev, p->wbytes, p->w, alloc)) {
            delete p;
            return nullptr;
        }
        return p;
    }

    bool piece_upload(piece * pp, const void * data, size_t bytes) override {
        auto * p = static_cast<cl_piece *>(pp);
        if (bytes != p->wbytes) {
            return false;
        }
        std::lock_guard<std::mutex> lock(d_->mutex);
        if (p->repacked) {
            const int64_t ne[4] = { p->K, p->rows, 1, 1 };
            const sgt::TensorDesc desc = sgt::make_contiguous(p->dt, ne);
            return cl11_rp_transfer(d_, p->w, desc, (uint8_t *) const_cast<void *>(data), 0, bytes, true);
        }
        return clEnqueueWriteBuffer(d_->queue, p->w, CL_TRUE, 0, bytes, data, 0, nullptr, nullptr) == CL_SUCCESS;
    }

    void piece_free(piece * pp) override {
        auto * p = static_cast<cl_piece *>(pp);
        {
            std::lock_guard<std::mutex> lock(d_->mutex);
            release(p);
        }
        delete p;
    }

    bool submit(piece * pp, const float * x, int64_t N, float * out) override {
        auto * p = static_cast<cl_piece *>(pp);
        const size_t xbytes = (size_t) p->K * (size_t) N * sizeof(float);
        const size_t obytes = (size_t) p->rows * (size_t) N * sizeof(float);

        std::lock_guard<std::mutex> lock(d_->mutex);
        if (!grow(p, xbytes, obytes)) {
            return false;
        }
        memcpy(p->xp.ptr, x, xbytes);
        cl_int err = clEnqueueWriteBuffer(d_->queue, p->xm, CL_FALSE, 0, xbytes, p->xp.ptr, 0, nullptr, nullptr);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "SGLX: %s: clEnqueueWriteBuffer failed: %s\n", name_.c_str(), cl11_err_str(err));
            return false;
        }

        const int64_t ne_w[4] = { p->K, p->rows, 1, 1 };
        const int64_t ne_x[4] = { p->K, N, 1, 1 };
        const int64_t ne_y[4] = { p->rows, N, 1, 1 };
        cl11_tensor s0, s1, dd;
        s0.mem = p->w;  s0.d = sgt::make_contiguous(p->dt, ne_w);                 s0.repacked = p->repacked;
        s1.mem = p->xm; s1.d = sgt::make_contiguous(SCRAPGPU_DTYPE_F32, ne_x);
        dd.mem = p->om; dd.d = sgt::make_contiguous(SCRAPGPU_DTYPE_F32, ne_y);
        // the executor caches things made from src1 (prepared columns, MUL_MAT_ID ids) by buffer and offset: the columns changed
        for (auto & xs : d_->xprep) {
            xs.valid = false;
        }
        d_->mid_ids.valid = false;
        if (!cl11_op_mul_mat(d_, dd, s0, s1)) {
            fprintf(stderr, "SGLX: %s: cl11_op_mul_mat failed\n", name_.c_str());
            return false;
        }
        cl_event done_ev = nullptr;
        err = clEnqueueReadBuffer(d_->queue, p->om, CL_FALSE, 0, obytes, p->op.ptr, 0, nullptr, SGLX_TRACE_ON() ? &done_ev : nullptr);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "SGLX: %s: clEnqueueReadBuffer failed: %s\n", name_.c_str(), cl11_err_str(err));
            return false;
        }
        if (done_ev != nullptr) {
            // the time at which the device finished this job, whatever the order in which the host waits (SGL_TRACE)
            clSetEventCallback(done_ev, CL_COMPLETE, [](cl_event, cl_int, void * tag) { SGLX_TRACE(static_cast<const char *>(tag)); }, (void *) name_.c_str());
            clReleaseEvent(done_ev);
        }
        clFlush(d_->queue);
        p->out_host = out;
        p->out_bytes = obytes;
        pending_.push_back(p);
        return true;
    }

    bool wait() override {
        if (pending_.empty()) {
            return true;
        }
        std::lock_guard<std::mutex> lock(d_->mutex);
        const cl_int err = clFinish(d_->queue);
        bool ok = err == CL_SUCCESS;
        if (!ok) {
            fprintf(stderr, "SGLX: %s: clFinish failed: %s\n", name_.c_str(), cl11_err_str(err));
        }
        for (cl_piece * p : pending_) {
            if (ok) {
                memcpy(p->out_host, p->op.ptr, p->out_bytes);
            }
        }
        pending_.clear();
        return ok;
    }

private:
    // pinned staging memory of `bytes` (kept mapped)
    bool pin_alloc(pinned & s, size_t bytes) {
        cl_int err = CL_SUCCESS;
        s.mem = clCreateBuffer(d_->context, CL_MEM_ALLOC_HOST_PTR | CL_MEM_READ_WRITE, bytes, nullptr, &err);
        if (err == CL_SUCCESS && s.mem != nullptr) {
            s.ptr = clEnqueueMapBuffer(d_->queue, s.mem, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0, bytes, 0, nullptr, nullptr, &err);
            if (err == CL_SUCCESS && s.ptr != nullptr) {
                s.size = bytes;
                return true;
            }
            clReleaseMemObject(s.mem);
            s.mem = nullptr;
        }
        s.ptr = malloc(bytes);
        s.fallback = true;
        s.size = s.ptr != nullptr ? bytes : 0;
        return s.ptr != nullptr;
    }

    void pin_free(pinned & s) {
        if (s.fallback) {
            free(s.ptr);
        } else if (s.mem != nullptr) {
            clEnqueueUnmapMemObject(d_->queue, s.mem, s.ptr, 0, nullptr, nullptr);
            clFinish(d_->queue);
            clReleaseMemObject(s.mem);
        }
        s = pinned();
    }

    void release_mem(cl_mem & m) {
        if (m != nullptr) {
            cl11_release_const_subs_for_parent(d_->const_subs, m);
            clReleaseMemObject(m);
            m = nullptr;
        }
    }

    void release(cl_piece * p) {
        release_mem(p->w);
        release_mem(p->xm);
        release_mem(p->om);
        pin_free(p->xp);
        pin_free(p->op);
        p->xcap = p->ocap = 0;
    }

    // device and pinned buffers for the input columns and the result, at least `xbytes` / `obytes` (grown with headroom)
    bool grow(cl_piece * p, size_t xbytes, size_t obytes) {
        if (xbytes > p->xcap) {
            const size_t cap = std::max<size_t>(xbytes, p->xcap * 2);
            release_mem(p->xm);
            pin_free(p->xp);
            size_t alloc = 0;
            if (!cl11_alloc_buffer(d_->context, d_->queue, d_->info, d_->name.c_str(), "sglx pdev_cl11 x", d_->strict_alloc_dev, cap, p->xm, alloc) ||
                !pin_alloc(p->xp, cap)) {
                p->xcap = 0;
                return false;
            }
            p->xcap = cap;
        }
        if (obytes > p->ocap) {
            const size_t cap = std::max<size_t>(obytes, p->ocap * 2);
            release_mem(p->om);
            pin_free(p->op);
            size_t alloc = 0;
            if (!cl11_alloc_buffer(d_->context, d_->queue, d_->info, d_->name.c_str(), "sglx pdev_cl11 out", d_->strict_alloc_dev, cap, p->om, alloc) ||
                !pin_alloc(p->op, cap)) {
                p->ocap = 0;
                return false;
            }
            p->ocap = cap;
        }
        return true;
    }

    cl11_device *        d_ = nullptr;
    std::string          name_;
    std::string          desc_;
    double               weight_ = 1.0;
    std::map<ggml_type, bool> sup_;
    std::vector<cl_piece *> pending_;
};

} // namespace

std::unique_ptr<pdev> make_pdev_cl11(cl11_device * d) {
    return std::make_unique<pdev_cl11>(d);
}

} // namespace sglx
