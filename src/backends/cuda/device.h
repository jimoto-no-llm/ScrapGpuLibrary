// One NVIDIA GPU of the ScrapGpu CUDA backend: the context, the module with the kernels (PTX, ptx_sm20.inc / ptx_sm35.inc), device and pinned
// host memory, streams, and the launches of the quantized mat-vec / mat-mat / MUL_MAT_ID kernels. Independent of ggml and of the ScrapGpu C API
// (api.cpp is the C API on top of it, adapters/ggml/composite/pdev_cuda.cpp the SGLX physical device).
//
// No exception leaves this file. Every failing call returns false and leaves the reason in last_error().
// Contexts: a Fermi card and a Kepler card cannot share a process with the 390 driver (the context of the second generation fails with
// CUDA_ERROR_INVALID_DEVICE, docs/benchmarks/moe-ep-20261021/microbench MB0), so a context is created on first use (init()), not at
// enumeration, and init() of the second generation reports the failure.

#ifndef SCRAPGPU_CUDA_DEVICE_H
#define SCRAPGPU_CUDA_DEVICE_H

#include "driver.h"

#include <cstdint>
#include <mutex>
#include <string>

namespace scrapgpu_cuda {

enum class WType { Q4_0, Q6_K };

inline int wtype_block_size(WType t) { return t == WType::Q4_0 ? 32 : 256; }
inline int wtype_block_bytes(WType t) { return t == WType::Q4_0 ? 18 : 210; }
inline uint64_t wtype_row_bytes(WType t, int64_t K) { return (uint64_t) (K / wtype_block_size(t)) * (uint64_t) wtype_block_bytes(t); }

struct DeviceInfo {
    int         ordinal = 0;        // the driver's number
    std::string name;               // "NVIDIA GeForce GT 730"
    int         cc_major = 0;
    int         cc_minor = 0;
    uint64_t    total_memory = 0;
    int         sm_count = 0;
    int         clock_khz = 0;
    int         mem_clock_khz = 0;
    int         bus_width = 0;
    bool        kepler_or_newer() const { return cc_major > 3 || (cc_major == 3 && cc_minor >= 5); }
    double      peak_gbs() const { return 2.0 * mem_clock_khz * 1e3 * (bus_width / 8.0) / 1e9; }
};

// Number of CUDA devices usable by the backend: 0 without libcuda / without a device. SCRAPGPU_CUDA_DEVICES (comma-separated substrings of
// the device names, e.g. "GT 730,GT 710") keeps only the matching devices, in the order of the list; the others are invisible.
int device_count();

class Device {
public:
    // the index-th device of the list above (nullptr when out of range); the object lives until the process ends
    static Device * get(int index);

    const DeviceInfo & info() const { return info_; }
    int index() const { return index_; }

    // Creates the context and loads the kernels (once). False when it fails (last_error() has the reason).
    bool init();
    bool ready() const { return ctx_ != nullptr; }
    const std::string & last_error() const { return err_; }

    // Serializes the users of the device (the C API and a pdev may use the same one). The methods below do not lock; they bind the context.
    std::recursive_mutex mu;

    // ---- memory ----
    bool alloc(size_t bytes, uint64_t & dptr);
    void free(uint64_t dptr);
    void * host_alloc(size_t bytes);   // pinned
    void host_free(void * p);
    bool mem_info(size_t & free_bytes, size_t & total_bytes);

    // ---- streams ----
    CUstream stream_default();         // the device's own stream
    CUstream stream_create();
    void stream_destroy(CUstream s);
    bool stream_sync(CUstream s);

    bool h2d(uint64_t dst, const void * src, size_t bytes, CUstream s);
    bool d2h(void * dst, uint64_t src, size_t bytes, CUstream s);
    bool d2d(uint64_t dst, uint64_t src, size_t bytes, CUstream s);
    bool memset8(uint64_t dst, uint8_t v, size_t bytes, CUstream s);

    // ---- kernels (all asynchronous on `s`; the pointers are device addresses) ----
    // out[c * rows + r] = sum_k W[r][k] * x[c * K + k], c < N, r < rows. W: rows rows of K elements in the natural GGUF layout, row_bytes apart
    // (a multiple of 2; K % block size == 0). x: 16-byte aligned, N columns of K floats.
    bool mul_mat(CUstream s, WType t, uint64_t w, uint64_t row_bytes, uint32_t rows, uint32_t K, uint64_t x, uint32_t N, uint64_t out);

    // MUL_MAT_ID with Q4_0 experts: dst[r, s, t] = sum_k W[ids[t * ids_tok_stride + s]][r][k] * x[t * x_tok_stride + s * x_slot_stride + k]
    // (dst = [rows, n_ids, n_tokens] contiguous). w holds n_expert experts of expert_bytes each; an id outside [0, n_expert) gives zeros.
    // n_ids * n_tokens <= 65535.
    bool mul_mat_id_q4_0(CUstream s, uint64_t w, uint64_t row_bytes, uint64_t expert_bytes, uint32_t n_expert, uint32_t rows, uint32_t K,
                         uint64_t ids, uint32_t ids_tok_stride, uint32_t n_ids, uint32_t n_tokens, uint64_t x, uint32_t x_slot_stride,
                         uint32_t x_tok_stride, uint64_t out);

private:
    Device() = default;
    bool bind();
    bool fail(const char * what, CUresult r);
    bool launch(CUfunction f, CUstream s, unsigned gx, unsigned gy, void ** args);

    DeviceInfo  info_;
    int         index_ = 0;
    CUdevice    dev_ = 0;
    CUcontext   ctx_ = nullptr;
    CUmodule    mod_ = nullptr;
    CUstream    stream_ = nullptr;
    bool        init_tried_ = false;
    std::string err_;
    unsigned    max_grid_x_ = 65535;
    CUfunction  mq4_[4] = {};
    CUfunction  mq6_[4] = {};
    CUfunction  mid4_ = nullptr;
};

} // namespace scrapgpu_cuda

#endif
