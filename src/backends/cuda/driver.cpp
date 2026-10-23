#include "driver.h"

#include <cstdio>
#include <cstdlib>
#include <mutex>

#if defined(_WIN32)
#  include <windows.h>
#else
#  include <dlfcn.h>
#endif

namespace scrapgpu_cuda {
namespace {

#if defined(_WIN32)
using lib_t = HMODULE;
lib_t lib_open(const char * path) { return LoadLibraryA(path); }
void * lib_sym(lib_t l, const char * name) { return (void *) GetProcAddress(l, name); }
#else
using lib_t = void *;
lib_t lib_open(const char * path) { return dlopen(path, RTLD_NOW | RTLD_LOCAL); }
void * lib_sym(lib_t l, const char * name) { return dlsym(l, name); }
#endif

template <typename F>
bool bind(lib_t l, F & fn, const char * name) {
    fn = reinterpret_cast<F>(lib_sym(l, name));
    return fn != nullptr;
}

struct Holder {
    Driver drv;
    bool ok = false;

    Holder() {
        lib_t l = nullptr;
        const char * env = getenv("SCRAPGPU_CUDA_LIB");
        if (env != nullptr && env[0] != '\0') {
            l = lib_open(env);
        }
#if defined(_WIN32)
        if (l == nullptr) l = lib_open("nvcuda.dll");
#else
        if (l == nullptr) l = lib_open("libcuda.so.1");
        if (l == nullptr) l = lib_open("/usr/lib/x86_64-linux-gnu/libcuda.so.1");
        if (l == nullptr) l = lib_open("/usr/lib64/libcuda.so.1");
#endif
        if (l == nullptr) {
            return;
        }
        // the *_v2 entry points are the current ones of the CUDA 4+ Driver API
        bool all = bind(l, drv.cuInit, "cuInit") &&
                   bind(l, drv.cuDriverGetVersion, "cuDriverGetVersion") &&
                   bind(l, drv.cuDeviceGetCount, "cuDeviceGetCount") &&
                   bind(l, drv.cuDeviceGet, "cuDeviceGet") &&
                   bind(l, drv.cuDeviceGetName, "cuDeviceGetName") &&
                   bind(l, drv.cuDeviceGetAttribute, "cuDeviceGetAttribute") &&
                   bind(l, drv.cuDeviceTotalMem, "cuDeviceTotalMem_v2") &&
                   bind(l, drv.cuCtxCreate, "cuCtxCreate_v2") &&
                   bind(l, drv.cuCtxDestroy, "cuCtxDestroy_v2") &&
                   bind(l, drv.cuCtxSetCurrent, "cuCtxSetCurrent") &&
                   bind(l, drv.cuCtxSynchronize, "cuCtxSynchronize") &&
                   bind(l, drv.cuMemGetInfo, "cuMemGetInfo_v2") &&
                   bind(l, drv.cuModuleLoadDataEx, "cuModuleLoadDataEx") &&
                   bind(l, drv.cuModuleGetFunction, "cuModuleGetFunction") &&
                   bind(l, drv.cuModuleUnload, "cuModuleUnload") &&
                   bind(l, drv.cuMemAlloc, "cuMemAlloc_v2") &&
                   bind(l, drv.cuMemFree, "cuMemFree_v2") &&
                   bind(l, drv.cuMemHostAlloc, "cuMemHostAlloc") &&
                   bind(l, drv.cuMemFreeHost, "cuMemFreeHost") &&
                   bind(l, drv.cuMemcpyHtoDAsync, "cuMemcpyHtoDAsync_v2") &&
                   bind(l, drv.cuMemcpyDtoHAsync, "cuMemcpyDtoHAsync_v2") &&
                   bind(l, drv.cuMemcpyDtoDAsync, "cuMemcpyDtoDAsync_v2") &&
                   bind(l, drv.cuMemsetD8Async, "cuMemsetD8Async") &&
                   bind(l, drv.cuStreamCreate, "cuStreamCreate") &&
                   bind(l, drv.cuStreamDestroy, "cuStreamDestroy_v2") &&
                   bind(l, drv.cuStreamSynchronize, "cuStreamSynchronize") &&
                   bind(l, drv.cuLaunchKernel, "cuLaunchKernel");
        // optional
        bind(l, drv.cuGetErrorName, "cuGetErrorName");
        bind(l, drv.cuGetErrorString, "cuGetErrorString");
        if (!all) {
            return;
        }
        ok = drv.cuInit(0) == CUDA_SUCCESS;
    }
};

} // namespace

const Driver * driver() {
    static Holder h;
    return h.ok ? &h.drv : nullptr;
}

const char * result_name(CUresult r) {
    static thread_local char buf[96];
    const Driver * d = driver();
    const char * name = nullptr;
    if (d != nullptr && d->cuGetErrorName != nullptr && d->cuGetErrorName(r, &name) == CUDA_SUCCESS && name != nullptr) {
        snprintf(buf, sizeof(buf), "%s (%d)", name, r);
    } else {
        snprintf(buf, sizeof(buf), "CUDA error %d", r);
    }
    return buf;
}

} // namespace scrapgpu_cuda
