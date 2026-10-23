// SGLX composite device: the physical-device interface (pdev).
//
// A pdev is one physical GPU (or the CPU) that holds a slice of the rows of a weight matrix and computes
// MUL_MAT for that slice. The composite device (composite_device.cpp) splits a weight along its rows
// (the vocabulary of the output projection), hands every pdev its rows once (piece_upload) and then, for
// every MUL_MAT, submits the same input columns to all pdevs before it waits for any of them, so the
// devices run at the same time although the host is a single thread.
//
// Implementations: pdev_ggml.cpp (any ggml backend device through the public ggml-backend.h API: the
// official Vulkan backend, ggml-cpu, Metal) and pdev_cl11.cpp (a CL11 OpenCL 1.1 device of this plugin).
// docs/plans/MOE_COMPOSITE_DEVICE.md chapter 6.2 describes the full design; this is the part that the
// first version (the output projection) needs.

#ifndef SGLX_PDEV_H
#define SGLX_PDEV_H

#include "ggml-backend.h"
#include "ggml.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

namespace sglx {

// ---- MoE expert slots (docs/plans/HOT_EXPERT_CACHE.md, composite/moe_block.cpp) ----
// The shape of one expert: gate / up are [n_embd, n_ff] (n_ff rows of n_embd), down is [n_ff, n_embd]; gate and up have the same type.
struct moe_shape {
    ggml_type t_gate = GGML_TYPE_Q4_0;
    ggml_type t_up   = GGML_TYPE_Q4_0;
    ggml_type t_down = GGML_TYPE_Q4_0;
    int64_t   n_embd = 0;
    int64_t   n_ff   = 0;
    bool operator==(const moe_shape & o) const {
        return t_gate == o.t_gate && t_up == o.t_up && t_down == o.t_down && n_embd == o.n_embd && n_ff == o.n_ff;
    }
};

// an array of expert slots on one device (the three matrices of S experts), opaque for the composite
struct moe_slots {
    virtual ~moe_slots() = default;
};

// where the three matrices of one expert are in host memory (the mmap of the model file)
struct moe_src {
    const void * gate = nullptr;
    const void * up = nullptr;
    const void * down = nullptr;
};

// the rows of one weight that live on one pdev (plus the activation scratch it needs), opaque for the composite
struct piece {
    virtual ~piece() = default;
};

class pdev {
public:
    virtual ~pdev() = default;

    // the ggml device name ("Vulkan0", "CL11_1", "CPU") and a description
    virtual const std::string & name() const = 0;
    virtual const std::string & description() const = 0;

    // relative speed of a quantized mat-vec on this device (about the effective GB/s); the default split ratio
    virtual double default_weight() const = 0;

    // can the pdev compute MUL_MAT(weight of `type`, F32 columns) (any number of columns)? Cached.
    virtual bool supports_mul_mat(ggml_type type) = 0;

    // room for `rows` rows of K columns of `type` on the device; nullptr when the allocation fails
    virtual piece * piece_create(ggml_type type, int64_t K, int64_t rows) = 0;
    // `bytes` = rows * row size: the rows in the natural ggml layout, contiguous. Blocking.
    virtual bool piece_upload(piece * p, const void * data, size_t bytes) = 0;
    virtual void piece_free(piece * p) = 0;

    // out[c * rows + r] = sum_k W[r][k] * x[c * K + k] for c < N, r < rows. Starts the work and returns (the host
    // buffers `x` and `out` must stay valid until wait() returned).
    virtual bool submit(piece * p, const float * x, int64_t N, float * out) = 0;
    // blocks until everything submitted since the last wait() is finished and written to its `out`
    virtual bool wait() = 0;
    // true when submit() only queues the job for a worker thread of the pdev (a synchronous API such as the remote backend): the composite
    // submits such pdevs first, so that their work and the wake-up of the worker overlap with the submissions of the others
    virtual bool threaded() const { return false; }
    // GGML_SGLX_PROFILE: where submit() spent its time (empty when the pdev keeps no timers)
    virtual std::string stats() { return std::string(); }

    // ---- MoE expert slots ----
    // can the device compute MUL_MAT_ID of the three matrices of `sh` (and SWIGLU between them)?
    virtual bool supports_moe(const moe_shape & sh) { (void) sh; return false; }
    // free / total device memory in bytes (false: unknown)
    virtual bool memory(size_t * free_bytes, size_t * total_bytes) { (void) free_bytes; (void) total_bytes; return false; }
    // room for n_slots experts of `sh` (nullptr when the allocation fails)
    virtual moe_slots * moe_create(const moe_shape & sh, int n_slots) { (void) sh; (void) n_slots; return nullptr; }
    virtual void moe_free(moe_slots * s) { (void) s; }
    // slot <- the three matrices of one expert in the natural ggml layout (bytes of one expert each: gate, up, down). Blocking.
    virtual bool moe_upload(moe_slots * s, int slot, const void * gate, const void * up, const void * down) {
        (void) s; (void) slot; (void) gate; (void) up; (void) down; return false;
    }
    // out[:, i] = down_{pair_slot[i]} ( silu(gate_{pair_slot[i]} x[:, pair_tok[i]]) * (up_{pair_slot[i]} x[:, pair_tok[i]]) ), i < n_pairs. x is
    // [n_embd, N_x] (columns contiguous), out is [n_embd, n_pairs]; no routing weight is applied. Starts the work and returns (x, pair_*
    // and out must stay valid until wait() returned).
    virtual bool moe_submit(moe_slots * s, const float * x, int64_t N_x, const int32_t * pair_slot, const int32_t * pair_tok, int n_pairs, float * out) {
        (void) s; (void) x; (void) N_x; (void) pair_slot; (void) pair_tok; (void) n_pairs; (void) out; return false;
    }
    // Prefill with the routing weights and the sum over the experts of a token done on the device (only the [n_embd, N_x] sums come back; the pair outputs would be 8 times as
    // much). `ids[k + n_used * t]` is the slot of the k-th expert of token t on this device, or moe_zero_slot() when the pair is not for this device; `w` is its routing weight
    // (0 for the pairs that are not for this device). The experts `ex[0..n_ex)` are copied from host memory into the slots ring_first + i first (n_ex <= the ring; the
    // staging of the next group overlaps the transfer of the current one, and `overlap` runs on the calling thread while the transfers are going on); `ids` may refer to
    // those slots. acc[:, t] = sum_k w * expert_k(x[:, t]), added in the order of k. Blocking. N_x <= 1024.
    virtual bool moe_prefill(moe_slots * s, int ring_first, const moe_src * ex, int n_ex, const float * x, int64_t N_x, int n_used, const int32_t * ids, const float * w, float * acc,
                             void (*overlap)(void *), void * overlap_ctx) {
        (void) s; (void) ring_first; (void) ex; (void) n_ex; (void) x; (void) N_x; (void) n_used; (void) ids; (void) w; (void) acc; (void) overlap; (void) overlap_ctx;
        return false;
    }
    // a slot that holds an expert of zeros (its output is exactly 0 for any input): the filler of the ids of moe_prefill. -1: none.
    virtual int moe_zero_slot(moe_slots * s) { (void) s; return -1; }
    // allocates the compute buffer of moe_prefill for `max_tokens` tokens of `n_used` experts now (false: no memory), so that the free memory that sized the slots is not spent later
    virtual bool moe_reserve(moe_slots * s, int max_tokens, int n_used) { (void) s; (void) max_tokens; (void) n_used; return true; }

    // Prefill streaming: the experts `ex[0..n_exp)` are copied from host memory into the slots [ring_first, ring_first + ring_n) a few at a time (the copy of
    // the next group overlaps the computation of the current one) and the pairs are computed there: pair i is the expert ex[pair_exp[i]] on the column
    // pair_tok[i] of x ([n_embd, N_x], N_x <= 1024); out is [n_embd, n_pairs] in pair order. Blocking. The slots of the ring are not used by anything else.
    virtual bool moe_stream(moe_slots * s, int ring_first, int ring_n, const moe_src * ex, int n_exp, const float * x, int64_t N_x, const int32_t * pair_exp,
                            const int32_t * pair_tok, int n_pairs, float * out) {
        (void) s; (void) ring_first; (void) ring_n; (void) ex; (void) n_exp; (void) x; (void) N_x; (void) pair_exp; (void) pair_tok; (void) n_pairs; (void) out;
        return false;
    }
};

// the effective mat-vec GB/s used for the default split, from a device description (substring table, see composite_device.cpp)
double default_weight_for(const std::string & description, bool is_cpu);

std::unique_ptr<pdev> make_pdev_ggml(ggml_backend_dev_t dev);

} // namespace sglx

#endif
