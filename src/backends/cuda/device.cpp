#include "device.h"

#include "ptx_sm20.inc"
#include "ptx_sm35.inc"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <vector>

namespace scrapgpu_cuda {
namespace {

constexpr int kCols[4] = { 1, 2, 4, 8 }; // columns of the mq4 / mq6 kernels, in the order of the arrays

std::string lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return (char) std::tolower(c); });
    return s;
}

struct Registry {
    std::vector<std::unique_ptr<Device>> devices; // not constructible outside Device::get: filled there
    bool built = false;
    std::mutex mu;
};

Registry & registry() {
    static Registry r;
    return r;
}

} // namespace

// Device has a private constructor: the registry is filled by this friend-free trick (a local subclass is not possible either), so the
// construction goes through a static member that is allowed to call it.
Device * Device::get(int index) {
    Registry & r = registry();
    std::lock_guard<std::mutex> lock(r.mu);
    if (!r.built) {
        r.built = true;
        const Driver * d = driver();
        int n = 0;
        if (d != nullptr && d->cuDeviceGetCount(&n) == CUDA_SUCCESS && n > 0) {
            std::vector<DeviceInfo> all;
            for (int i = 0; i < n; ++i) {
                CUdevice dev = 0;
                if (d->cuDeviceGet(&dev, i) != CUDA_SUCCESS) {
                    continue;
                }
                DeviceInfo info;
                info.ordinal = i;
                char name[256] = {};
                d->cuDeviceGetName(name, (int) sizeof(name) - 1, dev);
                info.name = name;
                d->cuDeviceGetAttribute(&info.cc_major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev);
                d->cuDeviceGetAttribute(&info.cc_minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev);
                d->cuDeviceGetAttribute(&info.sm_count, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, dev);
                d->cuDeviceGetAttribute(&info.clock_khz, CU_DEVICE_ATTRIBUTE_CLOCK_RATE, dev);
                d->cuDeviceGetAttribute(&info.mem_clock_khz, CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE, dev);
                d->cuDeviceGetAttribute(&info.bus_width, CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH, dev);
                size_t total = 0;
                d->cuDeviceTotalMem(&total, dev);
                info.total_memory = total;
                all.push_back(info);
            }
            std::vector<DeviceInfo> chosen;
            const char * filter = getenv("SCRAPGPU_CUDA_DEVICES");
            if (filter != nullptr && filter[0] != '\0') {
                std::string f = filter;
                size_t pos = 0;
                while (pos <= f.size()) {
                    size_t comma = f.find(',', pos);
                    std::string tok = f.substr(pos, comma == std::string::npos ? std::string::npos : comma - pos);
                    pos = comma == std::string::npos ? f.size() + 1 : comma + 1;
                    while (!tok.empty() && tok.front() == ' ') tok.erase(tok.begin());
                    while (!tok.empty() && tok.back() == ' ') tok.pop_back();
                    if (tok.empty()) {
                        continue;
                    }
                    for (const DeviceInfo & i : all) {
                        if (lower(i.name).find(lower(tok)) != std::string::npos &&
                            std::none_of(chosen.begin(), chosen.end(), [&](const DeviceInfo & c) { return c.ordinal == i.ordinal; })) {
                            chosen.push_back(i);
                            break;
                        }
                    }
                }
            } else {
                chosen = all;
            }
            for (const DeviceInfo & i : chosen) {
                std::unique_ptr<Device> dev(new Device());
                dev->info_ = i;
                dev->index_ = (int) r.devices.size();
                d->cuDeviceGet(&dev->dev_, i.ordinal);
                r.devices.push_back(std::move(dev));
            }
        }
    }
    return index >= 0 && (size_t) index < r.devices.size() ? r.devices[(size_t) index].get() : nullptr;
}

int device_count() {
    Device::get(0); // builds the list
    Registry & r = registry();
    std::lock_guard<std::mutex> lock(r.mu);
    return (int) r.devices.size();
}

bool Device::fail(const char * what, CUresult r) {
    err_ = std::string(what) + ": " + result_name(r);
    return false;
}

bool Device::bind() {
    if (ctx_ == nullptr) {
        err_ = "the CUDA context is not initialized";
        return false;
    }
    const CUresult r = driver()->cuCtxSetCurrent(ctx_);
    return r == CUDA_SUCCESS || fail("cuCtxSetCurrent", r);
}

bool Device::init() {
    std::lock_guard<std::recursive_mutex> lock(mu);
    if (ctx_ != nullptr) {
        return true;
    }
    if (init_tried_) {
        return false; // err_ has the first failure
    }
    init_tried_ = true;
    const Driver * d = driver();
    if (d == nullptr) {
        err_ = "no CUDA driver";
        return false;
    }
    unsigned flags = 0;
    if (const char * s = getenv("SCRAPGPU_CUDA_SCHED")) {
        flags = strcmp(s, "spin") == 0 ? CU_CTX_SCHED_SPIN : strcmp(s, "yield") == 0 ? CU_CTX_SCHED_YIELD : strcmp(s, "blocking") == 0 ? CU_CTX_SCHED_BLOCKING_SYNC : 0;
    }
    CUresult r = d->cuCtxCreate(&ctx_, flags, dev_);
    if (r != CUDA_SUCCESS) {
        ctx_ = nullptr;
        return fail("cuCtxCreate (a Fermi and a Kepler card cannot share a process)", r);
    }
    d->cuCtxSetCurrent(ctx_);

    // the PTX: Kepler GK110 / GK208 and newer take the sm_35 build, the rest the sm_20 build
    const char * ptx = info_.kepler_or_newer() ? kPtxSm35 : kPtxSm20;
    char log[4096] = {};
    int opts[2] = { CU_JIT_ERROR_LOG_BUFFER, CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES };
    void * vals[2] = { log, (void *) (uintptr_t) sizeof(log) };
    r = d->cuModuleLoadDataEx(&mod_, ptx, 2, opts, vals);
    if (r != CUDA_SUCCESS) {
        mod_ = nullptr;
        fail("cuModuleLoadDataEx", r);
        if (log[0] != '\0') {
            err_ += std::string(" [") + log + "]";
        }
        d->cuCtxDestroy(ctx_);
        ctx_ = nullptr;
        return false;
    }
    bool ok = true;
    for (int i = 0; i < 4; ++i) {
        char n4[16], n6[16];
        snprintf(n4, sizeof(n4), "mq4_c%d", kCols[i]);
        snprintf(n6, sizeof(n6), "mq6_c%d", kCols[i]);
        ok = ok && d->cuModuleGetFunction(&mq4_[i], mod_, n4) == CUDA_SUCCESS && d->cuModuleGetFunction(&mq6_[i], mod_, n6) == CUDA_SUCCESS;
    }
    ok = ok && d->cuModuleGetFunction(&mid4_, mod_, "mid4") == CUDA_SUCCESS;
    if (!ok) {
        err_ = "a kernel is missing in the module";
        d->cuModuleUnload(mod_);
        mod_ = nullptr;
        d->cuCtxDestroy(ctx_);
        ctx_ = nullptr;
        return false;
    }
    int gx = 0;
    if (d->cuDeviceGetAttribute(&gx, CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X, dev_) == CUDA_SUCCESS && gx > 0) {
        max_grid_x_ = (unsigned) std::min(gx, 65535);
    }
    r = d->cuStreamCreate(&stream_, 0);
    if (r != CUDA_SUCCESS) {
        stream_ = nullptr; // the legacy stream 0 is used
    }
    return true;
}

bool Device::alloc(size_t bytes, uint64_t & dptr) {
    std::lock_guard<std::recursive_mutex> lock(mu);
    if (!bind()) return false;
    CUdeviceptr p = 0;
    const CUresult r = driver()->cuMemAlloc(&p, bytes == 0 ? 1 : bytes);
    if (r != CUDA_SUCCESS) return fail("cuMemAlloc", r);
    dptr = p;
    return true;
}

void Device::free(uint64_t dptr) {
    std::lock_guard<std::recursive_mutex> lock(mu);
    if (dptr != 0 && bind()) {
        driver()->cuMemFree(dptr);
    }
}

void * Device::host_alloc(size_t bytes) {
    std::lock_guard<std::recursive_mutex> lock(mu);
    if (!bind()) return nullptr;
    void * p = nullptr;
    const CUresult r = driver()->cuMemHostAlloc(&p, bytes == 0 ? 1 : bytes, CU_MEMHOSTALLOC_PORTABLE);
    if (r != CUDA_SUCCESS) {
        fail("cuMemHostAlloc", r);
        return nullptr;
    }
    return p;
}

void Device::host_free(void * p) {
    std::lock_guard<std::recursive_mutex> lock(mu);
    if (p != nullptr && bind()) {
        driver()->cuMemFreeHost(p);
    }
}

bool Device::mem_info(size_t & free_bytes, size_t & total_bytes) {
    std::lock_guard<std::recursive_mutex> lock(mu);
    if (!bind()) return false;
    const CUresult r = driver()->cuMemGetInfo(&free_bytes, &total_bytes);
    return r == CUDA_SUCCESS || fail("cuMemGetInfo", r);
}

CUstream Device::stream_default() { return stream_; }

CUstream Device::stream_create() {
    std::lock_guard<std::recursive_mutex> lock(mu);
    if (!bind()) return nullptr;
    CUstream s = nullptr;
    const CUresult r = driver()->cuStreamCreate(&s, 0);
    if (r != CUDA_SUCCESS) {
        fail("cuStreamCreate", r);
        return nullptr;
    }
    return s;
}

void Device::stream_destroy(CUstream s) {
    std::lock_guard<std::recursive_mutex> lock(mu);
    if (s != nullptr && bind()) {
        driver()->cuStreamDestroy(s);
    }
}

bool Device::stream_sync(CUstream s) {
    std::lock_guard<std::recursive_mutex> lock(mu);
    if (!bind()) return false;
    const CUresult r = driver()->cuStreamSynchronize(s);
    return r == CUDA_SUCCESS || fail("cuStreamSynchronize", r);
}

bool Device::h2d(uint64_t dst, const void * src, size_t bytes, CUstream s) {
    std::lock_guard<std::recursive_mutex> lock(mu);
    if (bytes == 0) return true;
    if (!bind()) return false;
    const CUresult r = driver()->cuMemcpyHtoDAsync(dst, src, bytes, s);
    return r == CUDA_SUCCESS || fail("cuMemcpyHtoDAsync", r);
}

bool Device::d2h(void * dst, uint64_t src, size_t bytes, CUstream s) {
    std::lock_guard<std::recursive_mutex> lock(mu);
    if (bytes == 0) return true;
    if (!bind()) return false;
    const CUresult r = driver()->cuMemcpyDtoHAsync(dst, src, bytes, s);
    return r == CUDA_SUCCESS || fail("cuMemcpyDtoHAsync", r);
}

bool Device::d2d(uint64_t dst, uint64_t src, size_t bytes, CUstream s) {
    std::lock_guard<std::recursive_mutex> lock(mu);
    if (bytes == 0) return true;
    if (!bind()) return false;
    const CUresult r = driver()->cuMemcpyDtoDAsync(dst, src, bytes, s);
    return r == CUDA_SUCCESS || fail("cuMemcpyDtoDAsync", r);
}

bool Device::memset8(uint64_t dst, uint8_t v, size_t bytes, CUstream s) {
    std::lock_guard<std::recursive_mutex> lock(mu);
    if (bytes == 0) return true;
    if (!bind()) return false;
    const CUresult r = driver()->cuMemsetD8Async(dst, v, bytes, s);
    return r == CUDA_SUCCESS || fail("cuMemsetD8Async", r);
}

bool Device::launch(CUfunction f, CUstream s, unsigned gx, unsigned gy, void ** args) {
    const CUresult r = driver()->cuLaunchKernel(f, gx, gy, 1, 128, 1, 1, 0, s, args, nullptr);
    return r == CUDA_SUCCESS || fail("cuLaunchKernel", r);
}

bool Device::mul_mat(CUstream s, WType t, uint64_t w, uint32_t rows, uint32_t K, uint64_t x, uint32_t N, uint64_t out) {
    std::lock_guard<std::recursive_mutex> lock(mu);
    if (!bind()) return false;
    if (rows == 0 || N == 0) return true;
    if (K % (uint32_t) wtype_block_size(t) != 0 || (x & 15) != 0 || (w & 15) != 0) {
        err_ = "mul_mat: K is not a whole number of blocks, or x / w is not 16-byte aligned";
        return false;
    }
    const unsigned gx = (unsigned) std::min<uint64_t>(max_grid_x_, ((uint64_t) rows + 31) / 32 / 4 + 1);
    uint32_t col = 0;
    while (col < N) {
        const uint32_t rem = N - col;
        const int k = rem >= 8 ? 3 : rem >= 4 ? 2 : rem >= 2 ? 1 : 0;
        uint64_t xp = x + (uint64_t) col * K * 4;
        uint64_t op = out + (uint64_t) col * rows * 4;
        CUdeviceptr wp = w;
        void * args[] = { &wp, &xp, &op, &K, &rows };
        if (!launch(t == WType::Q4_0 ? mq4_[k] : mq6_[k], s, gx, 1, args)) {
            return false;
        }
        col += (uint32_t) kCols[k];
    }
    return true;
}

bool Device::mul_mat_id_q4_0(CUstream s, uint64_t w, uint64_t slice_bytes, uint32_t n_expert, uint32_t rows, uint32_t K, uint64_t ids,
                             uint32_t ids_tok_stride, uint32_t n_ids, uint32_t n_tokens, uint64_t x, uint32_t x_slot_stride, uint32_t x_tok_stride,
                             uint64_t out) {
    std::lock_guard<std::recursive_mutex> lock(mu);
    if (!bind()) return false;
    if (rows == 0 || n_ids == 0 || n_tokens == 0) return true;
    if (K % 32 != 0 || (x & 15) != 0 || (w & 15) != 0 || (slice_bytes & 15) != 0 || (x_slot_stride & 3) != 0 || (x_tok_stride & 3) != 0 ||
        (uint64_t) n_ids * n_tokens > 65535) {
        err_ = "mul_mat_id: K is not a whole number of blocks, w / x / slices are not 16-byte aligned, or too many (slot, token) pairs";
        return false;
    }
    const unsigned gx = (unsigned) std::min<uint64_t>(max_grid_x_, ((uint64_t) rows + 31) / 32 / 4 + 1);
    CUdeviceptr wp = w, idp = ids, xp = x, op = out;
    void * args[] = { &wp, &idp, &xp, &op, &K, &rows, &slice_bytes, &n_ids, &n_expert, &ids_tok_stride, &x_slot_stride, &x_tok_stride };
    return launch(mid4_, s, gx, n_ids * n_tokens, args);
}

} // namespace scrapgpu_cuda
