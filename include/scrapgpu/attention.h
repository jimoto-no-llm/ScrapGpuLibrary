#ifndef SCRAPGPU_ATTENTION_H
#define SCRAPGPU_ATTENTION_H

#include "scrapgpu/stream.h"
#include "scrapgpu/tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

// High-level scaled dot product attention:
//   query  = [D,  Q, Hq,  B]
//   key    = [D,  K, Hkv, B]
//   value  = [Dv, K, Hkv, B]
//   output = [Dv, Q, Hq,  B]
//
// All tensors must be F32, natural-layout and contiguous. Grouped query attention is supported when
// Hq is divisible by Hkv; query head h uses key/value head h / (Hq / Hkv).
//
// The optional additive mask is also F32 and contiguous:
//   mask = [K, Q, 1 or Hq, 1 or B]
//
// When causal is non-zero, key position k is visible to query position q only when:
//   k <= query_position_offset + q
//
// Temporary device workspace is bounded by max_workspace_bytes. The implementation tiles the query
// dimension and rejects the call before allocation when the budget cannot hold the transposed V
// slice plus at least one query tile.
typedef struct scrapgpu_attention_params {
    size_t struct_size;
    const scrapgpu_tensor_desc * query;
    const scrapgpu_tensor_desc * key;
    const scrapgpu_tensor_desc * value;
    const scrapgpu_tensor_desc * mask;   // optional, may be NULL
    const scrapgpu_tensor_desc * output;
    uint64_t query_position_offset;
    uint64_t max_workspace_bytes;
    int causal;
    // Explicit finite score multiplier; normally 1/sqrt(D). Zero and negative
    // values are valid. Only -INFINITY in an additive mask means fully masked.
    float scale;
    // ---- fields below were appended (ABI-compatible: a caller with a smaller struct_size gets 0 / NULL) ----
    // SCRAPGPU_ATTENTION_FLAG_* bits.
    uint32_t flags;
    // Optional caller-owned workspace buffer for SCRAPGPU_ATTENTION_FLAG_DEVICE_BATCHED (bytes from offset 0,
    // at least scrapgpu_attention_batched_workspace_bytes). NULL: the call allocates (and frees) its own.
    scrapgpu_buffer_t * workspace;
    uint64_t workspace_bytes;
} scrapgpu_attention_params;

// Run every (batch, head) pair of the attention as a few batched device ops
// (CPY transpose of V, MUL_MAT scores, SOFT_MAX, CLAMP, MUL_MAT values): the mask stays on the device,
// no host transfer happens and nothing is synchronized. The default (tiled) path downloads the mask and
// uploads it again for every (batch, head, query tile).
//  - the workspace is [V^T: Dv*K*Hkv*B floats][scores: K*Q*Hq*B floats]; max_workspace_bytes is ignored
//    (the caller sized the workspace), the tensors are not tiled;
//  - `causal` is not supported (pass an additive mask), a backend that cannot run one of the ops
//    returns SCRAPGPU_STATUS_UNSUPPORTED with the reason in scrapgpu_last_error (no tiled fallback);
//  - a query whose mask row is entirely -INFINITY produces exact zeros, like the tiled path: its NaN
//    softmax row is cleared on the device by a CLAMP(0, 1) that maps NaN to 0 (fmax/min semantics).
#define SCRAPGPU_ATTENTION_FLAG_DEVICE_BATCHED 1u

// Bytes of workspace the DEVICE_BATCHED path needs for these query/key/value/output descriptors
// (only the shapes are read). UNSUPPORTED when the size overflows.
SCRAPGPU_API scrapgpu_status scrapgpu_attention_batched_workspace_bytes(const scrapgpu_attention_params * params,
                                                                        uint64_t * out_bytes);

SCRAPGPU_API scrapgpu_status scrapgpu_scaled_dot_product_attention(scrapgpu_device_t * device,
                                                                   scrapgpu_stream_t * stream,
                                                                   const scrapgpu_attention_params * params);

#ifdef __cplusplus
}
#endif

#endif
