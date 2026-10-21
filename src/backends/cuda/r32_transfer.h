// Host <-> device transfer of a quantized weight slice in the R32 layout (src/tensor/tensor.h, the layout of the OpenCL 1.1 backend; the CUDA
// kernels read it): the natural GGUF blocks are repacked on the host (cl11_rp_pack / cl11_rp_unpack of src/backends/opencl11/r32.h, which need
// no OpenCL) and copied to the areas of the slice, in chunks of whole groups of 32 rows.

#ifndef SCRAPGPU_CUDA_R32_TRANSFER_H
#define SCRAPGPU_CUDA_R32_TRANSFER_H

#include "device.h"

#include <cstdint>

namespace scrapgpu_cuda {

// `nat` = `rows` rows of K elements of `t` in the natural layout (rows * wtype_row_bytes bytes) -> the slice at the device address `dptr`
// (rows * wtype_row_bytes bytes, 16-byte aligned). Blocking: when it returns the host memory has been consumed and the copies are done.
bool r32_upload(Device & d, WType t, int64_t K, int64_t rows, const void * nat, uint64_t dptr, CUstream s);

// the slice at `dptr` -> `nat` (natural layout). Blocking.
bool r32_download(Device & d, WType t, int64_t K, int64_t rows, void * nat, uint64_t dptr, CUstream s);

} // namespace scrapgpu_cuda

#endif
