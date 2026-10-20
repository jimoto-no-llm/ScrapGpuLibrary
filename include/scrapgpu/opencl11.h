#ifndef SCRAPGPU_OPENCL11_H
#define SCRAPGPU_OPENCL11_H

#include "scrapgpu/api.h"
#include "scrapgpu/status.h"

#ifdef __cplusplus
extern "C" {
#endif

// Registers the OpenCL 1.1 backend (src/backends/opencl11): the devices of cl11_probe_device_infos,
// honouring GGML_CL11_DEVICES (SCRAPGPU_OPENCL11_DEVICES is an alias). Safe to call more than once.
SCRAPGPU_API scrapgpu_status scrapgpu_opencl11_register(void);

// With GGML_CL11_PROFILE set (every op is then timed with OpenCL events and the queue is finished after each
// op), prints the table of kernel time per op (GGML_CL11_PROFILE_SHAPES=1 splits it by the shape of src0) to
// stderr and clears it. No effect otherwise.
SCRAPGPU_API void scrapgpu_opencl11_profile_print(void);

#ifdef __cplusplus
}
#endif

#endif
