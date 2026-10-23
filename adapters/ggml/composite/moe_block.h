// SGLX0 as the home of the MoE expert weights (docs/plans/HOT_EXPERT_CACHE.md): MUL_MAT_ID on host-resident (mmap) experts, a hot / medium / cold
// split of the experts over the GPUs and the CPU, and the execution of the MoE block of llama.cpp (up, gate, SWIGLU, down) as one job.
//
//   GGML_SGLX_MOE=Vulkan0:auto,Vulkan1:auto,CL11_0:auto      the devices that keep hot experts in slots, fastest first; "1" = no device (all on the CPU)
//
// llama.cpp is not modified: `-ot 'blk\.[0-9]+\.ffn_(up|gate|down)_exps\.weight=SGLX0' -dev Vulkan0,SGLX0 -ts 1/0` puts the expert tensors in an SGLX0 buffer
// that wraps the mmap of the model file (buffer_from_host_ptr); the MoE ops of the layer are then run by SGLX0.
//
// This header is internal to the composite device (composite_device.cpp calls it).

#ifndef SGLX_MOE_BLOCK_H
#define SGLX_MOE_BLOCK_H

#include "ggml-backend.h"
#include "ggml.h"

namespace sglx {

// a ggml device by name through the process-wide registry (composite_device.cpp)
ggml_backend_dev_t find_ggml_device(const char * name);

namespace moe {

// GGML_SGLX_MOE is set (read once)
bool enabled();

// Called by the composite device. `owned` tells whether a buffer belongs to SGLX0.
struct hooks {
    bool (*owned)(ggml_backend_buffer_t buffer);
};
void init(ggml_backend_buffer_type_t buft, const hooks & h);

// buffer_init_tensor of an SGLX0 buffer: a 3-D tensor named blk.N.ffn_{up,gate,down}_exps.weight is an expert tensor
void note_tensor(ggml_tensor * t);

// can SGLX0 run `op` (MUL_MAT_ID with an SGLX0 weight, the SWIGLU / MUL / ADD of its block)?
bool supports_op(const ggml_tensor * op);

// does this module run `node`?
bool handles(const ggml_tensor * node);

// runs node `i` of the graph (a MoE block is run as a whole at its first MUL_MAT_ID; the other nodes of the block are then skipped). false = failed.
bool exec(ggml_cgraph * g, int i);

// a buffer of pinned host memory from the host buffer type of the first MoE device (nullptr if there is none)
ggml_backend_buffer_t alloc_pinned(size_t size);

// at exit: the slot arrays go back to their devices (some backends assert that no buffer is alive when their device is torn down)
void shutdown();

// GGML_SGLX_PROFILE: the time that graph_compute spent on the lazy input copies (synchronizing the producing device and reading x, ids, weights)
void note_input_copy_ms(double ms);
void print_profile();

} // namespace moe
} // namespace sglx

#endif
