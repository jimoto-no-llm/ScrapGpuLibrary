// SGLX composite device ("SGLX0"): one ggml backend device whose weights live on several physical devices at once.
//
// First version (docs/benchmarks/moe-ep-20261021/sglx-lmhead/README.md): a weight matrix is split along its rows (for the output
// projection of a language model: along the vocabulary) over the physical devices named in GGML_SGLX_SPLIT, and every MUL_MAT with such
// a weight is computed by all the devices at the same time. llama.cpp is not modified: `-ot "token_embd\.weight=SGLX0"` puts the tensor
// into the buffers of this device, which keep the whole tensor in host memory (the host shadow) and give each physical device its rows
// when the first MUL_MAT needs them.
//
//   GGML_SGLX_SPLIT=Vulkan0:0.5,Vulkan1:0.2,CL11_0:0.15,CL11_1:0.15      device[:ratio],...   (no ratio: the default weight of the device)
//   GGML_SGLX_PROFILE=1                                                  times of the MUL_MAT (and of the uploads) on stderr
//
// The device exists only when GGML_SGLX_SPLIT is set. It lives in the CL11 plugin (libggml-cl11.so) because the CL11 devices that it uses
// are the very cl11_device objects of the plugin (one cl_context per GPU), see attach().

#ifndef SGLX_COMPOSITE_H
#define SGLX_COMPOSITE_H

#include "pdev.h"

#include "ggml-backend.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

struct cl11_device;

namespace sglx {

// a CL11 physical device (pdev_cl11.cpp)
std::unique_ptr<pdev> make_pdev_cl11(cl11_device * d);

// Called once by the CL11 registry after it probed its devices: reads GGML_SGLX_SPLIT and, when it is set, creates the device SGLX0.
// `cl11` are the CL11 devices of the plugin as (name "CL11_<i>", device); the physical devices of the split are resolved on first use.
void attach(ggml_backend_reg_t reg, const std::vector<std::pair<std::string, cl11_device *>> & cl11);

// the devices that attach() created (0 or 1) for the registry
size_t device_count();
ggml_backend_dev_t device_get(size_t index);

} // namespace sglx

#endif
