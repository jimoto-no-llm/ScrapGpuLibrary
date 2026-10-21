// The CUDA Driver API of the ScrapGpu CUDA backend, loaded at run time.
//
// libcuda is never linked: driver_load() dlopens it (libcuda.so.1; SCRAPGPU_CUDA_LIB overrides the path), so the backend builds and starts on
// machines without CUDA, where it simply finds 0 devices. No CUDA header is used either (there is no CUDA toolkit on the machines this is for):
// the few types and constants are declared here, with the values of the CUDA 9.1 era Driver API that the NVIDIA 390 driver implements.

#ifndef SCRAPGPU_CUDA_DRIVER_H
#define SCRAPGPU_CUDA_DRIVER_H

#include <cstddef>
#include <cstdint>

namespace scrapgpu_cuda {

using CUresult = int;
using CUdevice = int;
using CUdeviceptr = unsigned long long;
struct CUctx_st;
struct CUmod_st;
struct CUfunc_st;
struct CUstream_st;
using CUcontext = CUctx_st *;
using CUmodule = CUmod_st *;
using CUfunction = CUfunc_st *;
using CUstream = CUstream_st *;

constexpr CUresult CUDA_SUCCESS = 0;
constexpr unsigned CU_CTX_SCHED_SPIN = 0x01;
constexpr unsigned CU_CTX_SCHED_YIELD = 0x02;
constexpr unsigned CU_CTX_SCHED_BLOCKING_SYNC = 0x04;
constexpr unsigned CU_MEMHOSTALLOC_PORTABLE = 0x01;
constexpr unsigned CU_STREAM_NON_BLOCKING = 0x01;

// CUdevice_attribute
constexpr int CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X = 5;
constexpr int CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y = 6;
constexpr int CU_DEVICE_ATTRIBUTE_CLOCK_RATE = 13;
constexpr int CU_DEVICE_ATTRIBUTE_PCI_BUS_ID = 33;
constexpr int CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE = 36;
constexpr int CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH = 37;
constexpr int CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT = 16;
constexpr int CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR = 75;
constexpr int CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR = 76;

// CUjit_option
constexpr int CU_JIT_INFO_LOG_BUFFER = 3;
constexpr int CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES = 4;
constexpr int CU_JIT_ERROR_LOG_BUFFER = 5;
constexpr int CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES = 6;

struct Driver {
    CUresult (*cuInit)(unsigned flags) = nullptr;
    CUresult (*cuDriverGetVersion)(int * v) = nullptr;
    CUresult (*cuDeviceGetCount)(int * n) = nullptr;
    CUresult (*cuDeviceGet)(CUdevice * d, int ordinal) = nullptr;
    CUresult (*cuDeviceGetName)(char * name, int len, CUdevice d) = nullptr;
    CUresult (*cuDeviceGetAttribute)(int * v, int attr, CUdevice d) = nullptr;
    CUresult (*cuDeviceTotalMem)(size_t * bytes, CUdevice d) = nullptr;
    CUresult (*cuCtxCreate)(CUcontext * c, unsigned flags, CUdevice d) = nullptr;
    CUresult (*cuCtxDestroy)(CUcontext c) = nullptr;
    CUresult (*cuCtxSetCurrent)(CUcontext c) = nullptr;
    CUresult (*cuCtxSynchronize)(void) = nullptr;
    CUresult (*cuMemGetInfo)(size_t * free_bytes, size_t * total_bytes) = nullptr;
    CUresult (*cuModuleLoadDataEx)(CUmodule * m, const void * image, unsigned n, int * opts, void ** vals) = nullptr;
    CUresult (*cuModuleGetFunction)(CUfunction * f, CUmodule m, const char * name) = nullptr;
    CUresult (*cuModuleUnload)(CUmodule m) = nullptr;
    CUresult (*cuMemAlloc)(CUdeviceptr * p, size_t bytes) = nullptr;
    CUresult (*cuMemFree)(CUdeviceptr p) = nullptr;
    CUresult (*cuMemHostAlloc)(void ** p, size_t bytes, unsigned flags) = nullptr;
    CUresult (*cuMemFreeHost)(void * p) = nullptr;
    CUresult (*cuMemcpyHtoDAsync)(CUdeviceptr dst, const void * src, size_t bytes, CUstream s) = nullptr;
    CUresult (*cuMemcpyDtoHAsync)(void * dst, CUdeviceptr src, size_t bytes, CUstream s) = nullptr;
    CUresult (*cuMemcpyDtoDAsync)(CUdeviceptr dst, CUdeviceptr src, size_t bytes, CUstream s) = nullptr;
    CUresult (*cuMemsetD8Async)(CUdeviceptr dst, unsigned char v, size_t n, CUstream s) = nullptr;
    CUresult (*cuStreamCreate)(CUstream * s, unsigned flags) = nullptr;
    CUresult (*cuStreamDestroy)(CUstream s) = nullptr;
    CUresult (*cuStreamSynchronize)(CUstream s) = nullptr;
    CUresult (*cuLaunchKernel)(CUfunction f, unsigned gx, unsigned gy, unsigned gz, unsigned bx, unsigned by, unsigned bz,
                               unsigned shared, CUstream s, void ** params, void ** extra) = nullptr;
    CUresult (*cuGetErrorName)(CUresult r, const char ** s) = nullptr;
    CUresult (*cuGetErrorString)(CUresult r, const char ** s) = nullptr;
};

// The driver, loaded and cuInit(0) done, or nullptr when there is no CUDA driver (or cuInit fails: no NVIDIA device, driver mismatch).
// Thread safe, evaluated once.
const Driver * driver();

// "CUDA_ERROR_xxx (code)" for a result
const char * result_name(CUresult r);

} // namespace scrapgpu_cuda

#endif
