#ifndef SCRAPGPU_CUDA_H
#define SCRAPGPU_CUDA_H

#include "scrapgpu/api.h"
#include "scrapgpu/status.h"

#ifdef __cplusplus
extern "C" {
#endif

// Registers the CUDA backend (src/backends/cuda, built with SCRAPGPU_BUILD_CUDA): the NVIDIA devices of the CUDA Driver API, which is loaded at
// run time with dlopen (libcuda.so.1, or SCRAPGPU_CUDA_LIB); no CUDA toolkit is needed to build or to run. The kernels are PTX (sm_20 for Fermi,
// sm_35 for Kepler and newer) compiled by the driver. Without libcuda or without an NVIDIA device registration succeeds with 0 devices.
//
// SCRAPGPU_CUDA_DEVICES (comma-separated substrings of the device names, e.g. "GT 730,GT 710") keeps only the matching devices, in that order.
// A Fermi and a Kepler card cannot be used in the same process with the 390 driver: the buffer or stream allocation on the second generation
// fails with SCRAPGPU_STATUS_BACKEND_ERROR.
//
// Natural layout only (no R32). Ops: MUL_MAT (src0 Q4_0 / Q6_K, F32 src1 and dst) and MUL_MAT_ID (src0 Q4_0), plus the metadata ops. Safe to call
// more than once.
SCRAPGPU_API scrapgpu_status scrapgpu_cuda_register(void);

#ifdef __cplusplus
}
#endif

#endif
