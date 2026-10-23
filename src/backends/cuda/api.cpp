// The ScrapGpu C API on the CUDA backend: scrapgpu_cuda_register() registers a backend whose devices are the NVIDIA GPUs of the CUDA Driver API
// (driver.h: libcuda is dlopen'd; no devices without it). Buffers are device memory (cuMemAlloc); a stream is a CUstream; the kernels are the
// PTX of kernels/kernels.c (device.h).
//
// Ops: MUL_MAT (src0 Q4_0 / Q6_K in the R32 layout, F32 src1 and dst, any number of columns, batches over src1's dims 2 and 3) and MUL_MAT_ID
// (src0 Q4_0 in the R32 layout, the experts of a MoE layer), plus the metadata ops (RESHAPE / VIEW / PERMUTE / TRANSPOSE). op_supported inspects
// the descriptors only. Everything else is UNSUPPORTED, the caller falls back to another backend. The quantized weights are stored in the R32
// layout (the layout the OpenCL 1.1 backend uses: coalesced 128-bit loads, one row per lane): tensor_upload / tensor_download of a tensor with
// layout = SCRAPGPU_LAYOUT_R32 repack on the host; they must cover the whole tensor.
//
// The contexts are created on first use (buffer_alloc / stream_create), not at registration: with the 390 driver a Fermi and a Kepler card cannot
// share a process, and the second generation fails there with BACKEND_ERROR.

#include "device.h"
#include "r32_transfer.h"

#include "scrapgpu/backend.h"
#include "scrapgpu/cuda.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <string>
#include <unordered_map>
#include <vector>

using scrapgpu_cuda::CUstream;
using scrapgpu_cuda::Device;
using scrapgpu_cuda::WType;

namespace {

// ---------------------------------------------------------------------------------------------
// objects
// ---------------------------------------------------------------------------------------------

struct cu_api_device {
    scrapgpu_object_header hdr; // must stay the first member
    Device *               dev = nullptr;
    std::string            name;
    std::string            vendor = "NVIDIA";
    std::string            driver_version;
    std::string            api_version;
    std::string            tuning_overrides;
};

struct cu_api_buffer {
    scrapgpu_object_header hdr;
    cu_api_device *        dev = nullptr;
    uint64_t               ptr = 0;
    size_t                 size = 0;
};

struct cu_api_stream {
    scrapgpu_object_header hdr;
    cu_api_device *        dev = nullptr;
    CUstream               s = nullptr;
};

struct cu_api_plan_node {
    scrapgpu_op_desc                   desc;
    std::vector<scrapgpu_tensor_desc>  inputs;
    std::vector<scrapgpu_tensor_desc>  outputs;
    std::vector<uint8_t>               params;
};

struct cu_api_plan {
    scrapgpu_object_header           hdr;
    cu_api_device *                  dev = nullptr;
    std::vector<cu_api_plan_node>    nodes;
    std::vector<cu_api_buffer *>     bufs; // buffer_free refuses while a plan points at the buffer
};

extern const scrapgpu_backend_vtable g_cuda_vtable;

std::vector<std::unique_ptr<cu_api_device>> g_devices;

std::mutex                                    g_plan_bufs_mutex;
std::unordered_map<cu_api_buffer *, size_t>   g_plan_bufs;

void plan_ref_buffer(cu_api_buffer * b) {
    std::lock_guard<std::mutex> lock(g_plan_bufs_mutex);
    ++g_plan_bufs[b];
}

void plan_unref_buffer(cu_api_buffer * b) {
    std::lock_guard<std::mutex> lock(g_plan_bufs_mutex);
    const auto it = g_plan_bufs.find(b);
    if (it != g_plan_bufs.end() && --it->second == 0) {
        g_plan_bufs.erase(it);
    }
}

bool plan_refs_buffer(cu_api_buffer * b) {
    std::lock_guard<std::mutex> lock(g_plan_bufs_mutex);
    return g_plan_bufs.find(b) != g_plan_bufs.end();
}

// ---------------------------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------------------------

scrapgpu_status api_invalid(const char * message) {
    scrapgpu_set_last_error(message);
    return SCRAPGPU_STATUS_INVALID_ARGUMENT;
}

scrapgpu_status api_backend_error(Device * d, const char * what) {
    const std::string msg = std::string(what) + ": " + (d != nullptr ? d->last_error() : std::string("no device"));
    scrapgpu_set_last_error(msg.c_str());
    return SCRAPGPU_STATUS_BACKEND_ERROR;
}

scrapgpu_status api_no(const char * why, const char ** reason) {
    if (reason != nullptr) {
        *reason = why;
    }
    scrapgpu_set_last_error(why);
    return SCRAPGPU_STATUS_UNSUPPORTED;
}

scrapgpu_status api_bad(const char * why, const char ** reason) {
    if (reason != nullptr) {
        *reason = why;
    }
    scrapgpu_set_last_error(why);
    return SCRAPGPU_STATUS_INVALID_ARGUMENT;
}

cu_api_device * api_device(scrapgpu_device_t * device) {
    auto * d = reinterpret_cast<cu_api_device *>(device);
    return d == nullptr || d->hdr.backend != &g_cuda_vtable ? nullptr : d;
}

cu_api_buffer * api_buffer(scrapgpu_buffer_t * buffer) {
    auto * b = reinterpret_cast<cu_api_buffer *>(buffer);
    return b == nullptr || b->hdr.backend != &g_cuda_vtable ? nullptr : b;
}

cu_api_stream * api_stream(scrapgpu_stream_t * stream) {
    auto * s = reinterpret_cast<cu_api_stream *>(stream);
    return s == nullptr || s->hdr.backend != &g_cuda_vtable ? nullptr : s;
}

scrapgpu_tensor_desc api_load_desc(const scrapgpu_tensor_desc & in) {
    scrapgpu_tensor_desc out = {};
    if (in.struct_size >= sizeof(size_t)) {
        std::memcpy(&out, &in, std::min(in.struct_size, sizeof(out)));
        out.struct_size = sizeof(out);
    }
    return out;
}

template <typename T>
T api_params(const scrapgpu_op_desc * desc) {
    T p = {};
    if (desc->params != nullptr) {
        std::memcpy(&p, desc->params, std::min(desc->params_size, sizeof(T)));
    }
    return p;
}

template <typename F>
scrapgpu_status api_guard(F && fn) noexcept {
    try {
        return fn();
    } catch (const std::bad_alloc &) {
        try { scrapgpu_set_last_error("out of host memory"); } catch (...) {}
        return SCRAPGPU_STATUS_OUT_OF_MEMORY;
    } catch (...) {
        try { scrapgpu_set_last_error("the CUDA backend threw an exception"); } catch (...) {}
        return SCRAPGPU_STATUS_BACKEND_ERROR;
    }
}

// ---------------------------------------------------------------------------------------------
// descriptor math (the public headers only; mirrors sgt:: of src/tensor/tensor.h)
// ---------------------------------------------------------------------------------------------

int64_t td_ne(const scrapgpu_tensor_desc & t, int i) { return t.ne[i] > 0 ? t.ne[i] : 1; }
size_t td_esz(const scrapgpu_tensor_desc & t) { return scrapgpu_dtype_type_size(t.dtype); }

uint64_t td_mul_sat(uint64_t a, uint64_t b) {
    return a != 0 && b > std::numeric_limits<uint64_t>::max() / a ? std::numeric_limits<uint64_t>::max() : a * b;
}

uint64_t td_add_sat(uint64_t a, uint64_t b) {
    return b > std::numeric_limits<uint64_t>::max() - a ? std::numeric_limits<uint64_t>::max() : a + b;
}

// the offset of the last byte + 1 that the tensor touches, saturating; 0 for an empty or unknown tensor
uint64_t td_nbytes(const scrapgpu_tensor_desc & t) {
    for (int i = 0; i < 4; ++i) {
        if (t.ne[i] <= 0) {
            return 0;
        }
    }
    const size_t block_size = scrapgpu_dtype_block_size(t.dtype);
    const size_t type_size  = scrapgpu_dtype_type_size(t.dtype);
    if (block_size == 0 || type_size == 0) {
        return 0;
    }
    uint64_t n;
    if (block_size == 1) {
        n = type_size;
        for (int i = 0; i < 4; ++i) {
            n = td_add_sat(n, td_mul_sat((uint64_t) (t.ne[i] - 1), t.nb[i]));
        }
    } else {
        n = td_mul_sat((uint64_t) t.ne[0], t.nb[0]) / block_size;
        for (int i = 1; i < 4; ++i) {
            n = td_add_sat(n, td_mul_sat((uint64_t) (t.ne[i] - 1), t.nb[i]));
        }
    }
    return n;
}

bool td_span_ok(const scrapgpu_tensor_desc & t) {
    const uint64_t half = std::numeric_limits<uint64_t>::max() / 2;
    if (t.offset > half) {
        return false;
    }
    uint64_t span = t.offset;
    for (int i = 0; i < 4; ++i) {
        const uint64_t count = (uint64_t) (td_ne(t, i) - 1);
        if (t.nb[i] != 0 && count > (half - span) / t.nb[i]) {
            return false;
        }
        span += count * t.nb[i];
        if (span > half) {
            return false;
        }
    }
    return true;
}

bool td_contiguous(const scrapgpu_tensor_desc & t) {
    const size_t block_size = scrapgpu_dtype_block_size(t.dtype);
    const size_t type_size  = scrapgpu_dtype_type_size(t.dtype);
    if (block_size == 0 || type_size == 0) {
        return false;
    }
    uint64_t next = type_size;
    if (t.ne[0] != (int64_t) block_size && t.nb[0] != next) {
        return false;
    }
    bool overflow = next != 0 && (uint64_t) (t.ne[0] / (int64_t) block_size) > std::numeric_limits<uint64_t>::max() / next;
    next *= (uint64_t) (t.ne[0] / (int64_t) block_size);
    for (int i = 1; i < 4; ++i) {
        if (t.ne[i] != 1 && t.nb[i] != next) {
            return false;
        }
        overflow |= next != 0 && (uint64_t) t.ne[i] > std::numeric_limits<uint64_t>::max() / next;
        next *= (uint64_t) t.ne[i];
    }
    return !overflow;
}

bool wtype_of(scrapgpu_dtype dt, WType & out) {
    if (dt == SCRAPGPU_DTYPE_Q4_0) { out = WType::Q4_0; return true; }
    if (dt == SCRAPGPU_DTYPE_Q6_K) { out = WType::Q6_K; return true; }
    return false;
}

// ---------------------------------------------------------------------------------------------
// op_supported
// ---------------------------------------------------------------------------------------------

scrapgpu_status api_check_tensors(const scrapgpu_op_desc * desc, const char ** reason) {
    auto check_one = [&](const scrapgpu_tensor_desc & raw) -> scrapgpu_status {
        const scrapgpu_tensor_desc td = api_load_desc(raw);
        if (td_esz(td) == 0) {
            *reason = "unknown tensor dtype";
            return SCRAPGPU_STATUS_UNSUPPORTED;
        }
        for (int i = 0; i < 4; ++i) {
            if (td.ne[i] <= 0) {
                *reason = "non-positive tensor dim";
                return SCRAPGPU_STATUS_INVALID_ARGUMENT;
            }
        }
        if (!td_span_ok(td)) {
            *reason = "tensor strides or offset overflow the address space";
            return SCRAPGPU_STATUS_UNSUPPORTED;
        }
        WType wt;
        if (td.layout == SCRAPGPU_LAYOUT_R32 ? !(wtype_of(td.dtype, wt) && scrapgpu_tensor_r32_eligible(&td)) : td.layout != SCRAPGPU_LAYOUT_NATURAL) {
            *reason = "the CUDA backend supports the natural layout and the R32 layout of Q4_0 / Q6_K weights";
            return SCRAPGPU_STATUS_UNSUPPORTED;
        }
        return SCRAPGPU_STATUS_OK;
    };
    for (size_t i = 0; i < desc->input_count; ++i) {
        const scrapgpu_status st = check_one(desc->inputs[i]);
        if (st != SCRAPGPU_STATUS_OK) {
            return st;
        }
    }
    for (size_t i = 0; i < desc->output_count; ++i) {
        const scrapgpu_status st = check_one(desc->outputs[i]);
        if (st != SCRAPGPU_STATUS_OK) {
            return st;
        }
    }
    return SCRAPGPU_STATUS_OK;
}

scrapgpu_status api_supports_mul_mat(const scrapgpu_op_desc * desc, const char ** reason) {
    if (desc->input_count != 2 || desc->output_count != 1) {
        return api_bad("MUL_MAT takes two inputs and one output", reason);
    }
    const auto p = api_params<scrapgpu_mul_mat_params>(desc);
    const scrapgpu_tensor_desc s0 = api_load_desc(desc->inputs[0]);
    const scrapgpu_tensor_desc s1 = api_load_desc(desc->inputs[1]);
    const scrapgpu_tensor_desc dd = api_load_desc(desc->outputs[0]);
    WType t = WType::Q4_0;
    if (!wtype_of(s0.dtype, t)) {
        return api_no("MUL_MAT: the CUDA backend has kernels for Q4_0 and Q6_K weights only", reason);
    }
    if (s0.layout != SCRAPGPU_LAYOUT_R32) {
        return api_no("MUL_MAT: quantized weights must be uploaded in the R32 layout", reason);
    }
    if (s1.dtype != SCRAPGPU_DTYPE_F32 || dd.dtype != SCRAPGPU_DTYPE_F32 || s1.layout != SCRAPGPU_LAYOUT_NATURAL || dd.layout != SCRAPGPU_LAYOUT_NATURAL) {
        return api_no("MUL_MAT expects an F32 src1 and destination", reason);
    }
    const int64_t K = td_ne(s0, 0);
    if (td_ne(s0, 0) != td_ne(s1, 0)) {
        return api_bad("MUL_MAT: the inner dimensions of src0 and src1 differ", reason);
    }
    if (td_ne(s1, 2) % td_ne(s0, 2) != 0 || td_ne(s1, 3) % td_ne(s0, 3) != 0) {
        return api_bad("MUL_MAT: src1 dims 2 and 3 must be multiples of the src0 dims", reason);
    }
    if (p.allow_broadcast == 0 && (td_ne(s1, 2) != td_ne(s0, 2) || td_ne(s1, 3) != td_ne(s0, 3))) {
        return api_no("MUL_MAT: broadcasting over dims 2/3 is not allowed", reason);
    }
    if (td_ne(dd, 0) != td_ne(s0, 1) || td_ne(dd, 1) != td_ne(s1, 1) || td_ne(dd, 2) != td_ne(s1, 2) || td_ne(dd, 3) != td_ne(s1, 3)) {
        return api_bad("MUL_MAT: dst must be [ne01, ne11, ne12, ne13]", reason);
    }
    // what the kernels need: 16-byte aligned weights (a repackable tensor already has aligned slices), contiguous columns, 16-byte aligned x
    if (s0.offset % 16 != 0) {
        return api_no("MUL_MAT: src0 must be 16-byte aligned", reason);
    }
    if (s1.nb[0] != 4 || dd.nb[0] != 4 || s1.nb[1] != (uint64_t) K * 4 || dd.nb[1] != (uint64_t) td_ne(dd, 0) * 4) {
        return api_no("MUL_MAT: src1 and dst need contiguous columns", reason);
    }
    if (s1.offset % 16 != 0 || s1.nb[2] % 16 != 0 || s1.nb[3] % 16 != 0) {
        return api_no("MUL_MAT: src1 must be 16-byte aligned", reason);
    }
    return SCRAPGPU_STATUS_OK;
}

scrapgpu_status api_supports_mul_mat_id(const scrapgpu_op_desc * desc, const char ** reason) {
    if (desc->input_count != 3 || desc->output_count != 1) {
        return api_bad("MUL_MAT_ID takes three inputs and one output", reason);
    }
    const scrapgpu_tensor_desc s0 = api_load_desc(desc->inputs[0]);
    const scrapgpu_tensor_desc s1 = api_load_desc(desc->inputs[1]);
    const scrapgpu_tensor_desc ids = api_load_desc(desc->inputs[2]);
    const scrapgpu_tensor_desc dd = api_load_desc(desc->outputs[0]);
    if (s0.dtype != SCRAPGPU_DTYPE_Q4_0) {
        return api_no("MUL_MAT_ID: the CUDA backend has a kernel for Q4_0 experts only", reason);
    }
    if (s0.layout != SCRAPGPU_LAYOUT_R32) {
        return api_no("MUL_MAT_ID: quantized weights must be uploaded in the R32 layout", reason);
    }
    if (s1.dtype != SCRAPGPU_DTYPE_F32 || dd.dtype != SCRAPGPU_DTYPE_F32 || ids.dtype != SCRAPGPU_DTYPE_I32) {
        return api_no("MUL_MAT_ID expects F32 src1/dst and I32 ids", reason);
    }
    if (td_ne(s0, 3) != 1 || td_ne(s1, 3) != 1 || td_ne(ids, 2) != 1 || td_ne(ids, 3) != 1) {
        return api_bad("MUL_MAT_ID: the tensors must be 3-dimensional at most", reason);
    }
    if (!(td_ne(s0, 0) == td_ne(s1, 0) && (td_ne(s1, 1) == 1 || td_ne(s1, 1) == td_ne(ids, 0)) && td_ne(s1, 2) == td_ne(ids, 1) &&
          td_ne(dd, 0) == td_ne(s0, 1) && td_ne(dd, 1) == td_ne(ids, 0) && td_ne(dd, 2) == td_ne(ids, 1) && td_ne(dd, 3) == 1)) {
        return api_bad("MUL_MAT_ID: shapes do not match the contract", reason);
    }
    if (s0.offset % 16 != 0) {
        return api_no("MUL_MAT_ID: src0 must be 16-byte aligned", reason);
    }
    if (s1.layout != SCRAPGPU_LAYOUT_NATURAL || dd.layout != SCRAPGPU_LAYOUT_NATURAL || ids.layout != SCRAPGPU_LAYOUT_NATURAL ||
        s1.nb[0] != 4 || dd.nb[0] != 4 || ids.nb[0] != 4 || ids.nb[1] % 4 != 0 || ids.offset % 4 != 0) {
        return api_no("MUL_MAT_ID: src1, dst and ids need contiguous elements", reason);
    }
    if (s1.offset % 16 != 0 || (td_ne(s1, 1) > 1 && s1.nb[1] % 16 != 0) || (td_ne(s1, 2) > 1 && s1.nb[2] % 16 != 0)) {
        return api_no("MUL_MAT_ID: src1 must be 16-byte aligned", reason);
    }
    if (dd.nb[1] != (uint64_t) td_ne(dd, 0) * 4 || dd.nb[2] != dd.nb[1] * (uint64_t) td_ne(dd, 1)) {
        return api_no("MUL_MAT_ID: dst must be contiguous", reason);
    }
    return SCRAPGPU_STATUS_OK;
}

scrapgpu_status api_supports(const scrapgpu_op_desc * desc, const char ** reason) {
    const char * reason_local = nullptr;
    if (reason == nullptr) {
        reason = &reason_local;
    }
    *reason = nullptr;
    if (desc == nullptr || desc->struct_size < sizeof(scrapgpu_op_desc)) {
        return api_bad("op descriptor is too small", reason);
    }
    if (desc->input_count > 32 || desc->output_count > 32 || (desc->input_count && !desc->inputs) || (desc->output_count && !desc->outputs)) {
        return api_bad("invalid tensor descriptor arrays", reason);
    }
    for (size_t i = 0; i < desc->input_count; ++i) {
        if (desc->inputs[i].struct_size < sizeof(scrapgpu_tensor_desc)) {
            return api_bad("input tensor descriptor is too small", reason);
        }
    }
    for (size_t i = 0; i < desc->output_count; ++i) {
        if (desc->outputs[i].struct_size < sizeof(scrapgpu_tensor_desc)) {
            return api_bad("output tensor descriptor is too small", reason);
        }
    }
    const scrapgpu_status valid = scrapgpu_op_desc_validate(desc);
    if (valid != SCRAPGPU_STATUS_OK) {
        *reason = "the op descriptor is invalid";
        return valid;
    }
    scrapgpu_status st = api_check_tensors(desc, reason);
    if (st != SCRAPGPU_STATUS_OK) {
        scrapgpu_set_last_error(*reason);
        return st;
    }
    switch (desc->op) {
        case SCRAPGPU_OP_NONE:
        case SCRAPGPU_OP_RESHAPE:
        case SCRAPGPU_OP_VIEW:
        case SCRAPGPU_OP_PERMUTE:
        case SCRAPGPU_OP_TRANSPOSE:
            return SCRAPGPU_STATUS_OK;
        case SCRAPGPU_OP_MUL_MAT:
            return api_supports_mul_mat(desc, reason);
        case SCRAPGPU_OP_MUL_MAT_ID:
            return api_supports_mul_mat_id(desc, reason);
        default:
            return api_no("op is not implemented by the CUDA backend", reason);
    }
}

// ---------------------------------------------------------------------------------------------
// devices
// ---------------------------------------------------------------------------------------------

size_t api_get_device_count(void * userdata) {
    (void) userdata;
    return g_devices.size();
}

scrapgpu_status api_get_device(void * userdata, size_t index, scrapgpu_device_t ** out_device) {
    (void) userdata;
    if (out_device == nullptr || index >= g_devices.size()) {
        return api_invalid("device index out of range");
    }
    *out_device = reinterpret_cast<scrapgpu_device_t *>(g_devices[index].get());
    return SCRAPGPU_STATUS_OK;
}

scrapgpu_status api_device_get_info(scrapgpu_device_t * device, scrapgpu_device_info * info) {
    return api_guard([&]() -> scrapgpu_status {
        cu_api_device * d = api_device(device);
        if (d == nullptr) {
            return api_invalid("not a CUDA device");
        }
        if (info == nullptr || info->struct_size < sizeof(size_t)) {
            return api_invalid("device info struct is too small");
        }
        const scrapgpu_cuda::DeviceInfo & di = d->dev->info();
        scrapgpu_device_info full = {};
        full.struct_size            = sizeof(full);
        full.name                   = d->name.c_str();
        full.vendor                 = d->vendor.c_str();
        full.platform_name          = "NVIDIA CUDA";
        full.driver_version         = d->driver_version.c_str();
        full.backend_name           = "cuda";
        full.reported_api_version   = d->api_version.c_str();
        full.used_api_version       = "CUDA Driver API, PTX 3.2";
        full.generation             = di.cc_major == 2 ? SCRAPGPU_DEVICE_GENERATION_NV_FERMI
                                    : di.cc_major == 3 ? SCRAPGPU_DEVICE_GENERATION_NV_KEPLER : SCRAPGPU_DEVICE_GENERATION_UNKNOWN;
        full.generation_reason      = "compute capability";
        full.total_memory_bytes     = di.total_memory;
        full.max_allocation_reported_bytes = di.total_memory;
        full.max_allocation_used_bytes     = di.total_memory;
        full.max_work_group_size    = 128;
        full.local_memory_bytes     = 0;
        full.compute_units          = (uint32_t) di.sm_count;
        full.warp_or_wave_width     = 32;
        full.quirks                 = 0;
        full.probed_quirks          = 0;
        full.tuning_profile_name    = di.kepler_or_newer() ? "sm_35" : "sm_20";
        full.tuning_overrides       = "";
        const size_t caller_size = info->struct_size;
        std::memcpy(info, &full, std::min(caller_size, sizeof(full)));
        info->struct_size = caller_size;
        return SCRAPGPU_STATUS_OK;
    });
}

// ---------------------------------------------------------------------------------------------
// buffers
// ---------------------------------------------------------------------------------------------

scrapgpu_status api_buffer_alloc(scrapgpu_device_t * device, size_t size, scrapgpu_buffer_t ** out_buffer) {
    return api_guard([&]() -> scrapgpu_status {
        cu_api_device * d = api_device(device);
        if (d == nullptr || out_buffer == nullptr) {
            return api_invalid("device or out_buffer is null");
        }
        *out_buffer = nullptr;
        if (!d->dev->init()) {
            return api_backend_error(d->dev, "CUDA context");
        }
        uint64_t p = 0;
        if (!d->dev->alloc(size, p)) {
            scrapgpu_set_last_error(d->dev->last_error().c_str());
            return SCRAPGPU_STATUS_OUT_OF_MEMORY;
        }
        std::unique_ptr<cu_api_buffer> b(new (std::nothrow) cu_api_buffer());
        if (b == nullptr) {
            d->dev->free(p);
            scrapgpu_set_last_error("out of host memory");
            return SCRAPGPU_STATUS_OUT_OF_MEMORY;
        }
        b->hdr.backend = &g_cuda_vtable;
        b->dev = d;
        b->ptr = p;
        b->size = size;
        *out_buffer = reinterpret_cast<scrapgpu_buffer_t *>(b.release());
        return SCRAPGPU_STATUS_OK;
    });
}

scrapgpu_status api_buffer_free(scrapgpu_buffer_t * buffer) {
    return api_guard([&]() -> scrapgpu_status {
        cu_api_buffer * b = api_buffer(buffer);
        if (b == nullptr) {
            return api_invalid("not a CUDA buffer");
        }
        if (plan_refs_buffer(b)) {
            return api_invalid("buffer is still referenced by a live plan");
        }
        // work already enqueued on any stream may still read the buffer
        {
            std::lock_guard<std::recursive_mutex> lock(b->dev->dev->mu);
            b->dev->dev->stream_sync(b->dev->dev->stream_default());
            b->dev->dev->free(b->ptr);
        }
        delete b;
        return SCRAPGPU_STATUS_OK;
    });
}

// the common checks of a buffer access: the stream and the buffer belong to the same device and the range fits
scrapgpu_status api_access(cu_api_stream * s, cu_api_buffer * b, uint64_t offset, size_t size, const char * what) {
    if (s == nullptr || b == nullptr) {
        return api_invalid("stream or buffer is not a CUDA object");
    }
    if (b->dev != s->dev) {
        return api_invalid("buffer and stream belong to different devices");
    }
    if (offset > (uint64_t) b->size || size > b->size - (size_t) offset) {
        return api_invalid(what);
    }
    return SCRAPGPU_STATUS_OK;
}

scrapgpu_status api_buffer_write(scrapgpu_stream_t * stream, scrapgpu_buffer_t * buffer, uint64_t offset, const void * data, size_t size, int blocking) {
    return api_guard([&]() -> scrapgpu_status {
        cu_api_stream * s = api_stream(stream);
        cu_api_buffer * b = api_buffer(buffer);
        const scrapgpu_status st = api_access(s, b, offset, size, "write exceeds the buffer");
        if (st != SCRAPGPU_STATUS_OK || size == 0) {
            return st;
        }
        Device * d = s->dev->dev;
        std::lock_guard<std::recursive_mutex> lock(d->mu);
        // a copy from pageable memory has read the host buffer when the call returns
        if (!d->h2d(b->ptr + offset, data, size, s->s) || (blocking && !d->stream_sync(s->s))) {
            return api_backend_error(d, "buffer write");
        }
        return SCRAPGPU_STATUS_OK;
    });
}

scrapgpu_status api_buffer_read(scrapgpu_stream_t * stream, scrapgpu_buffer_t * buffer, uint64_t offset, void * data, size_t size, int blocking) {
    (void) blocking; // a copy into pageable memory is complete when the call returns; we synchronize anyway
    return api_guard([&]() -> scrapgpu_status {
        cu_api_stream * s = api_stream(stream);
        cu_api_buffer * b = api_buffer(buffer);
        const scrapgpu_status st = api_access(s, b, offset, size, "read exceeds the buffer");
        if (st != SCRAPGPU_STATUS_OK || size == 0) {
            return st;
        }
        Device * d = s->dev->dev;
        std::lock_guard<std::recursive_mutex> lock(d->mu);
        if (!d->d2h(data, b->ptr + offset, size, s->s) || !d->stream_sync(s->s)) {
            return api_backend_error(d, "buffer read");
        }
        return SCRAPGPU_STATUS_OK;
    });
}

scrapgpu_status api_buffer_copy(scrapgpu_stream_t * stream, scrapgpu_buffer_t * dst, uint64_t dst_offset, scrapgpu_buffer_t * src, uint64_t src_offset, size_t size, int blocking) {
    return api_guard([&]() -> scrapgpu_status {
        cu_api_stream * s = api_stream(stream);
        cu_api_buffer * db = api_buffer(dst);
        cu_api_buffer * sb = api_buffer(src);
        scrapgpu_status st = api_access(s, db, dst_offset, size, "copy exceeds the destination buffer");
        if (st == SCRAPGPU_STATUS_OK) {
            st = api_access(s, sb, src_offset, size, "copy exceeds the source buffer");
        }
        if (st != SCRAPGPU_STATUS_OK || size == 0) {
            return st;
        }
        Device * d = s->dev->dev;
        std::lock_guard<std::recursive_mutex> lock(d->mu);
        if (db == sb) {
            // an overlap inside one buffer must copy as through a temporary; a device-to-device copy with overlap is undefined
            const uint64_t lo = std::max(dst_offset, src_offset), hi = std::min(dst_offset, src_offset) + size;
            if (lo < hi) {
                uint64_t tmp = 0;
                if (!d->alloc(size, tmp)) {
                    return api_backend_error(d, "buffer copy");
                }
                const bool ok = d->d2d(tmp, sb->ptr + src_offset, size, s->s) && d->d2d(db->ptr + dst_offset, tmp, size, s->s) && d->stream_sync(s->s);
                d->free(tmp);
                return ok ? SCRAPGPU_STATUS_OK : api_backend_error(d, "buffer copy");
            }
        }
        if (!d->d2d(db->ptr + dst_offset, sb->ptr + src_offset, size, s->s) || (blocking && !d->stream_sync(s->s))) {
            return api_backend_error(d, "buffer copy");
        }
        return SCRAPGPU_STATUS_OK;
    });
}

scrapgpu_status api_buffer_fill(scrapgpu_stream_t * stream, scrapgpu_buffer_t * buffer, uint64_t offset, size_t size, uint8_t byte, int blocking) {
    return api_guard([&]() -> scrapgpu_status {
        cu_api_stream * s = api_stream(stream);
        cu_api_buffer * b = api_buffer(buffer);
        const scrapgpu_status st = api_access(s, b, offset, size, "fill exceeds the buffer");
        if (st != SCRAPGPU_STATUS_OK || size == 0) {
            return st;
        }
        Device * d = s->dev->dev;
        std::lock_guard<std::recursive_mutex> lock(d->mu);
        if (!d->memset8(b->ptr + offset, byte, size, s->s) || (blocking && !d->stream_sync(s->s))) {
            return api_backend_error(d, "buffer fill");
        }
        return SCRAPGPU_STATUS_OK;
    });
}

// ---------------------------------------------------------------------------------------------
// streams
// ---------------------------------------------------------------------------------------------

scrapgpu_status api_stream_create(scrapgpu_device_t * device, scrapgpu_stream_t ** out_stream) {
    return api_guard([&]() -> scrapgpu_status {
        cu_api_device * d = api_device(device);
        if (d == nullptr || out_stream == nullptr) {
            return api_invalid("device or out_stream is null");
        }
        *out_stream = nullptr;
        if (!d->dev->init()) {
            return api_backend_error(d->dev, "CUDA context");
        }
        std::unique_ptr<cu_api_stream> s(new (std::nothrow) cu_api_stream());
        if (s == nullptr) {
            scrapgpu_set_last_error("out of host memory");
            return SCRAPGPU_STATUS_OUT_OF_MEMORY;
        }
        s->hdr.backend = &g_cuda_vtable;
        s->dev = d;
        s->s = d->dev->stream_create();
        if (s->s == nullptr) {
            return api_backend_error(d->dev, "stream create");
        }
        *out_stream = reinterpret_cast<scrapgpu_stream_t *>(s.release());
        return SCRAPGPU_STATUS_OK;
    });
}

scrapgpu_status api_stream_free(scrapgpu_stream_t * stream) {
    return api_guard([&]() -> scrapgpu_status {
        cu_api_stream * s = api_stream(stream);
        if (s == nullptr) {
            return api_invalid("not a CUDA stream");
        }
        {
            std::lock_guard<std::recursive_mutex> lock(s->dev->dev->mu);
            s->dev->dev->stream_sync(s->s);
            s->dev->dev->stream_destroy(s->s);
        }
        delete s;
        return SCRAPGPU_STATUS_OK;
    });
}

scrapgpu_status api_stream_sync(scrapgpu_stream_t * stream) {
    return api_guard([&]() -> scrapgpu_status {
        cu_api_stream * s = api_stream(stream);
        if (s == nullptr) {
            return api_invalid("not a CUDA stream");
        }
        std::lock_guard<std::recursive_mutex> lock(s->dev->dev->mu);
        return s->dev->dev->stream_sync(s->s) ? SCRAPGPU_STATUS_OK : api_backend_error(s->dev->dev, "stream sync");
    });
}

scrapgpu_device_t * api_stream_device(scrapgpu_stream_t * stream) {
    cu_api_stream * s = api_stream(stream);
    return s == nullptr ? nullptr : reinterpret_cast<scrapgpu_device_t *>(s->dev);
}

// ---------------------------------------------------------------------------------------------
// tensor transfers (natural layout, contiguous)
// ---------------------------------------------------------------------------------------------

scrapgpu_status api_tensor_transfer(scrapgpu_stream_t * stream, const scrapgpu_tensor_desc * raw, uint8_t * host, size_t size, bool to_dev) {
    cu_api_stream * s = api_stream(stream);
    if (s == nullptr || raw == nullptr) {
        return api_invalid("stream or tensor desc is null");
    }
    if (raw->struct_size < offsetof(scrapgpu_tensor_desc, offset) + sizeof(raw->offset)) {
        return api_invalid("tensor desc is too small");
    }
    const scrapgpu_tensor_desc td = api_load_desc(*raw);
    cu_api_buffer * b = api_buffer(td.buffer);
    if (b == nullptr) {
        return api_invalid("tensor desc has no CUDA buffer");
    }
    if (b->dev != s->dev) {
        return api_invalid("tensor buffer and stream belong to different devices");
    }
    if (td.offset > (uint64_t) b->size || size > b->size - (size_t) td.offset) {
        return api_invalid("transfer exceeds the buffer");
    }
    if (size > td_nbytes(td)) {
        return api_invalid("transfer size exceeds the tensor");
    }
    if (size == 0) {
        return SCRAPGPU_STATUS_OK;
    }
    Device * d = s->dev->dev;
    std::lock_guard<std::recursive_mutex> lock(d->mu);
    if (td.layout == SCRAPGPU_LAYOUT_R32) {
        WType wt;
        if (!wtype_of(td.dtype, wt) || !scrapgpu_tensor_r32_eligible(&td) || td.offset % 16 != 0) {
            scrapgpu_set_last_error("the tensor cannot be stored in the R32 layout (dtype, contiguity or alignment)");
            return SCRAPGPU_STATUS_UNSUPPORTED;
        }
        if (size != td_nbytes(td)) {
            scrapgpu_set_last_error("an R32 transfer must cover the whole tensor");
            return SCRAPGPU_STATUS_UNSUPPORTED;
        }
        const uint64_t slice = td.nb[2];
        for (int64_t i3 = 0; i3 < td_ne(td, 3); ++i3) {
            for (int64_t i2 = 0; i2 < td_ne(td, 2); ++i2) {
                const uint64_t off = (uint64_t) i2 * td.nb[2] + (uint64_t) i3 * td.nb[3];
                const bool ok = to_dev ? r32_upload(*d, wt, td.ne[0], td.ne[1], host + off, b->ptr + td.offset + off, s->s)
                                       : r32_download(*d, wt, td.ne[0], td.ne[1], host + off, b->ptr + td.offset + off, s->s);
                if (!ok) {
                    return api_backend_error(d, to_dev ? "tensor upload (R32)" : "tensor download (R32)");
                }
            }
        }
        (void) slice;
        return SCRAPGPU_STATUS_OK;
    }
    if (!td_contiguous(td)) {
        scrapgpu_set_last_error("tensor is not contiguous; use buffer_read / buffer_write for raw strided transfers");
        return SCRAPGPU_STATUS_UNSUPPORTED;
    }
    const bool ok = to_dev ? d->h2d(b->ptr + td.offset, host, size, s->s) : (d->d2h(host, b->ptr + td.offset, size, s->s) && d->stream_sync(s->s));
    return ok ? SCRAPGPU_STATUS_OK : api_backend_error(d, to_dev ? "tensor upload" : "tensor download");
}

scrapgpu_status api_tensor_upload(scrapgpu_stream_t * stream, const scrapgpu_tensor_desc * desc, const void * host, size_t size) {
    return api_guard([&]() -> scrapgpu_status { return api_tensor_transfer(stream, desc, (uint8_t *) const_cast<void *>(host), size, true); });
}

scrapgpu_status api_tensor_download(scrapgpu_stream_t * stream, const scrapgpu_tensor_desc * desc, void * host, size_t size) {
    return api_guard([&]() -> scrapgpu_status { return api_tensor_transfer(stream, desc, (uint8_t *) host, size, false); });
}

// ---------------------------------------------------------------------------------------------
// ops and plans
// ---------------------------------------------------------------------------------------------

scrapgpu_status api_op_supported(scrapgpu_device_t * device, const scrapgpu_op_desc * desc, const char ** reason) {
    if (api_device(device) == nullptr) {
        if (reason != nullptr) {
            *reason = "not a CUDA device";
        }
        return api_invalid("not a CUDA device");
    }
    return api_guard([&]() -> scrapgpu_status { return api_supports(desc, reason); });
}

// the device address of a tensor (offset applied), checking that its whole span lies inside the buffer of this stream's device
bool api_resolve(cu_api_stream * s, const scrapgpu_tensor_desc & td, uint64_t & ptr) {
    cu_api_buffer * b = api_buffer(td.buffer);
    if (b == nullptr || b->dev != s->dev) {
        return false;
    }
    if (td.offset > (uint64_t) b->size) {
        return false;
    }
    const uint64_t span = td_nbytes(td);
    if (span > b->size - (size_t) td.offset) {
        return false;
    }
    ptr = b->ptr + td.offset;
    return true;
}

scrapgpu_status api_exec_mul_mat(cu_api_stream * s, const scrapgpu_op_desc * desc) {
    const scrapgpu_tensor_desc s0 = api_load_desc(desc->inputs[0]);
    const scrapgpu_tensor_desc s1 = api_load_desc(desc->inputs[1]);
    const scrapgpu_tensor_desc dd = api_load_desc(desc->outputs[0]);
    uint64_t pw = 0, px = 0, pd = 0;
    if (!api_resolve(s, s0, pw) || !api_resolve(s, s1, px) || !api_resolve(s, dd, pd)) {
        return api_invalid("a tensor does not lie inside a buffer of the stream's device");
    }
    WType t = WType::Q4_0;
    wtype_of(s0.dtype, t);
    const int64_t K = td_ne(s0, 0), rows = td_ne(s0, 1), N = td_ne(s1, 1);
    const int64_t r2 = td_ne(s1, 2) / td_ne(s0, 2), r3 = td_ne(s1, 3) / td_ne(s0, 3);
    Device * d = s->dev->dev;
    std::lock_guard<std::recursive_mutex> lock(d->mu);
    for (int64_t i3 = 0; i3 < td_ne(s1, 3); ++i3) {
        for (int64_t i2 = 0; i2 < td_ne(s1, 2); ++i2) {
            const uint64_t w = pw + (uint64_t) (i2 / r2) * s0.nb[2] + (uint64_t) (i3 / r3) * s0.nb[3];
            const uint64_t x = px + (uint64_t) i2 * s1.nb[2] + (uint64_t) i3 * s1.nb[3];
            const uint64_t o = pd + (uint64_t) i2 * dd.nb[2] + (uint64_t) i3 * dd.nb[3];
            if (!d->mul_mat(s->s, t, w, (uint32_t) rows, (uint32_t) K, x, (uint32_t) N, o)) {
                return api_backend_error(d, "MUL_MAT");
            }
        }
    }
    return SCRAPGPU_STATUS_OK;
}

scrapgpu_status api_exec_mul_mat_id(cu_api_stream * s, const scrapgpu_op_desc * desc) {
    const scrapgpu_tensor_desc s0 = api_load_desc(desc->inputs[0]);
    const scrapgpu_tensor_desc s1 = api_load_desc(desc->inputs[1]);
    const scrapgpu_tensor_desc ids = api_load_desc(desc->inputs[2]);
    const scrapgpu_tensor_desc dd = api_load_desc(desc->outputs[0]);
    uint64_t pw = 0, px = 0, pi = 0, pd = 0;
    if (!api_resolve(s, s0, pw) || !api_resolve(s, s1, px) || !api_resolve(s, ids, pi) || !api_resolve(s, dd, pd)) {
        return api_invalid("a tensor does not lie inside a buffer of the stream's device");
    }
    const int64_t K = td_ne(s0, 0), rows = td_ne(s0, 1), E = td_ne(s0, 2);
    const int64_t n_ids = td_ne(ids, 0), n_tok = td_ne(ids, 1);
    const uint32_t x_slot = td_ne(s1, 1) > 1 ? (uint32_t) (s1.nb[1] / 4) : 0u;
    const uint32_t x_tok = (uint32_t) (s1.nb[2] / 4);
    const int64_t per_launch = std::max<int64_t>(1, 65535 / n_ids); // tokens per launch (gridDim.y <= 65535)
    Device * d = s->dev->dev;
    std::lock_guard<std::recursive_mutex> lock(d->mu);
    for (int64_t t0 = 0; t0 < n_tok; t0 += per_launch) {
        const int64_t nt = std::min(per_launch, n_tok - t0);
        if (!d->mul_mat_id_q4_0(s->s, pw, s0.nb[2], (uint32_t) E, (uint32_t) rows, (uint32_t) K, pi + (uint64_t) t0 * ids.nb[1],
                                (uint32_t) (ids.nb[1] / 4), (uint32_t) n_ids, (uint32_t) nt, px + (uint64_t) t0 * s1.nb[2], x_slot, x_tok,
                                pd + (uint64_t) t0 * dd.nb[2])) {
            return api_backend_error(d, "MUL_MAT_ID");
        }
    }
    return SCRAPGPU_STATUS_OK;
}

scrapgpu_status api_exec_desc(cu_api_stream * s, const scrapgpu_op_desc * desc) {
    const char * why = nullptr;
    scrapgpu_status st = api_supports(desc, &why);
    if (st != SCRAPGPU_STATUS_OK) {
        return st;
    }
    switch (desc->op) {
        case SCRAPGPU_OP_MUL_MAT:    return api_exec_mul_mat(s, desc);
        case SCRAPGPU_OP_MUL_MAT_ID: return api_exec_mul_mat_id(s, desc);
        default:                     return SCRAPGPU_STATUS_OK; // metadata ops
    }
}

scrapgpu_status api_op_execute(scrapgpu_stream_t * stream, const scrapgpu_op_desc * desc) {
    return api_guard([&]() -> scrapgpu_status {
        cu_api_stream * s = api_stream(stream);
        if (s == nullptr) {
            return api_invalid("not a CUDA stream");
        }
        if (desc == nullptr) {
            return api_invalid("op desc is null");
        }
        return api_exec_desc(s, desc);
    });
}

const char * api_op_name(int op) {
    switch (op) {
        case SCRAPGPU_OP_MUL_MAT: return "MUL_MAT";
        case SCRAPGPU_OP_MUL_MAT_ID: return "MUL_MAT_ID";
        default: return "op";
    }
}

scrapgpu_status api_plan_build(scrapgpu_device_t * device, const scrapgpu_plan_desc * desc, scrapgpu_plan_t ** out_plan) {
    return api_guard([&]() -> scrapgpu_status {
        cu_api_device * d = api_device(device);
        if (d == nullptr || desc == nullptr || out_plan == nullptr ||
            desc->struct_size < offsetof(scrapgpu_plan_desc, node_count) + sizeof(desc->node_count) ||
            desc->node_count > 4096 || (desc->node_count > 0 && desc->nodes == nullptr)) {
            return api_invalid("device, plan desc or node list is invalid");
        }
        *out_plan = nullptr;
        std::unique_ptr<cu_api_plan> plan(new (std::nothrow) cu_api_plan());
        if (plan == nullptr) {
            scrapgpu_set_last_error("out of host memory");
            return SCRAPGPU_STATUS_OUT_OF_MEMORY;
        }
        plan->hdr.backend = &g_cuda_vtable;
        plan->dev = d;
        plan->nodes.reserve(desc->node_count);
        for (size_t i = 0; i < desc->node_count; ++i) {
            const scrapgpu_plan_node & src = desc->nodes[i];
            if (src.struct_size < offsetof(scrapgpu_plan_node, op) + sizeof(src.op)) {
                return api_invalid("plan node is too small");
            }
            const char * why = nullptr;
            const scrapgpu_status st = api_supports(&src.op, &why);
            if (st != SCRAPGPU_STATUS_OK) {
                char buf[256];
                snprintf(buf, sizeof(buf), "plan node %zu (%s): %s", i, api_op_name((int) src.op.op), why != nullptr ? why : "not supported");
                scrapgpu_set_last_error(buf);
                return st;
            }
            cu_api_plan_node n;
            n.desc = src.op;
            n.desc.struct_size = sizeof(n.desc);
            for (size_t j = 0; j < src.op.input_count; ++j) {
                n.inputs.push_back(api_load_desc(src.op.inputs[j]));
            }
            for (size_t j = 0; j < src.op.output_count; ++j) {
                n.outputs.push_back(api_load_desc(src.op.outputs[j]));
            }
            if (src.op.params != nullptr && src.op.params_size > 0) {
                const uint8_t * p = static_cast<const uint8_t *>(src.op.params);
                n.params.assign(p, p + src.op.params_size);
            }
            n.desc.inputs  = n.inputs.data();
            n.desc.outputs = n.outputs.data();
            n.desc.params  = n.params.empty() ? nullptr : n.params.data();
            for (const scrapgpu_tensor_desc & t : n.inputs) {
                if (cu_api_buffer * rb = api_buffer(t.buffer)) plan->bufs.push_back(rb);
            }
            for (const scrapgpu_tensor_desc & t : n.outputs) {
                if (cu_api_buffer * rb = api_buffer(t.buffer)) plan->bufs.push_back(rb);
            }
            plan->nodes.push_back(std::move(n));
        }
        for (cu_api_buffer * rb : plan->bufs) {
            plan_ref_buffer(rb);
        }
        *out_plan = reinterpret_cast<scrapgpu_plan_t *>(plan.release());
        return SCRAPGPU_STATUS_OK;
    });
}

scrapgpu_status api_plan_execute(scrapgpu_stream_t * stream, scrapgpu_plan_t * plan_) {
    return api_guard([&]() -> scrapgpu_status {
        cu_api_stream * s = api_stream(stream);
        auto * plan = reinterpret_cast<cu_api_plan *>(plan_);
        if (s == nullptr || plan == nullptr || plan->hdr.backend != &g_cuda_vtable) {
            return api_invalid("stream or plan is not a CUDA object");
        }
        if (plan->dev != s->dev) {
            return api_invalid("plan and stream belong to different devices");
        }
        for (auto & node : plan->nodes) {
            const scrapgpu_status st = api_exec_desc(s, &node.desc);
            if (st != SCRAPGPU_STATUS_OK) {
                return st;
            }
        }
        return SCRAPGPU_STATUS_OK;
    });
}

scrapgpu_status api_plan_free(scrapgpu_plan_t * plan_) {
    return api_guard([&]() -> scrapgpu_status {
        auto * plan = reinterpret_cast<cu_api_plan *>(plan_);
        if (plan == nullptr || plan->hdr.backend != &g_cuda_vtable) {
            return api_invalid("not a CUDA plan");
        }
        for (cu_api_buffer * rb : plan->bufs) {
            plan_unref_buffer(rb);
        }
        delete plan;
        return SCRAPGPU_STATUS_OK;
    });
}

// ---------------------------------------------------------------------------------------------
// registration
// ---------------------------------------------------------------------------------------------

const scrapgpu_backend_vtable g_cuda_vtable = {
    /* .struct_size        = */ sizeof(scrapgpu_backend_vtable),
    /* .name               = */ "cuda",
    /* .userdata           = */ nullptr,
    /* .get_device_count   = */ api_get_device_count,
    /* .get_device         = */ api_get_device,
    /* .device_get_info    = */ api_device_get_info,
    /* .buffer_alloc       = */ api_buffer_alloc,
    /* .buffer_free        = */ api_buffer_free,
    /* .buffer_write       = */ api_buffer_write,
    /* .buffer_read        = */ api_buffer_read,
    /* .buffer_copy        = */ api_buffer_copy,
    /* .buffer_fill        = */ api_buffer_fill,
    /* .stream_sync        = */ api_stream_sync,
    /* .op_supported       = */ api_op_supported,
    /* .plan_build         = */ api_plan_build,
    /* .plan_execute       = */ api_plan_execute,
    /* .plan_free          = */ api_plan_free,
    /* .stream_create      = */ api_stream_create,
    /* .stream_free        = */ api_stream_free,
    /* .tensor_upload      = */ api_tensor_upload,
    /* .tensor_download    = */ api_tensor_download,
    /* .op_execute         = */ api_op_execute,
    /* .stream_device      = */ api_stream_device,
};

} // namespace

scrapgpu_status scrapgpu_cuda_register(void) {
    static std::mutex reg_mutex;
    std::lock_guard<std::mutex> lock(reg_mutex);
    for (size_t i = 0; i < scrapgpu_get_device_count(); ++i) {
        scrapgpu_device_t * existing = nullptr;
        if (scrapgpu_get_device(i, &existing) == SCRAPGPU_STATUS_OK && api_device(existing) != nullptr) {
            return SCRAPGPU_STATUS_OK;
        }
    }
    static bool registered = false;
    try {
        if (g_devices.empty()) {
            const int n = scrapgpu_cuda::device_count();
            for (int i = 0; i < n; ++i) {
                Device * dev = Device::get(i);
                auto d = std::make_unique<cu_api_device>();
                d->hdr.backend = &g_cuda_vtable;
                d->dev = dev;
                d->name = dev->info().name;
                char buf[64];
                snprintf(buf, sizeof(buf), "compute capability %d.%d", dev->info().cc_major, dev->info().cc_minor);
                d->api_version = buf;
                d->driver_version = "CUDA driver";
                g_devices.push_back(std::move(d));
            }
        }
        if (registered && g_devices.empty()) {
            return SCRAPGPU_STATUS_OK;
        }
        registered = true;
        return scrapgpu_register_backend(&g_cuda_vtable);
    } catch (const std::bad_alloc &) {
        scrapgpu_set_last_error("out of host memory");
        return SCRAPGPU_STATUS_OUT_OF_MEMORY;
    }
}
