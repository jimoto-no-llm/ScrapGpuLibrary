"""Inference-only ScrapGpu custom ops with CPU tensors and explicit CPU/OpenCL selection."""

from __future__ import annotations

import os
from pathlib import Path

import torch
from torch import nn

__all__ = [
    "load", "ScrapLinear", "has_device", "scaled_dot_product_attention",
    # device-resident tensors (stage B)
    "to_device", "to_cpu", "torch_shape", "empty", "mul_mat", "mul_mat_into", "add", "mul", "silu", "sigmoid", "swiglu", "scale",
    "rms_norm_mul", "rope", "concat", "sdpa", "sync", "stats", "reset_stats", "pool_trim", "TransferStats",
    "profile_print",
]


def _default_ext_path() -> Path:
    return Path(__file__).resolve().parent / "_C.so"


def load(library_path: str | os.PathLike | None = None) -> None:
    """Load the extension and initialize SCRAPGPU_TORCH_BACKEND / SCRAPGPU_TORCH_DEVICE.

    Defaults are opencl11 and 0. Set environment variables before the first load,
    device query, or custom op: configuration is fixed for the process afterwards.
    """
    path = Path(library_path) if library_path is not None else _default_ext_path()
    if not path.is_file():
        raise FileNotFoundError(
            f"scrapgpu_torch extension not found at {path}. "
            "Build with -DSCRAPGPU_BUILD_PYTORCH=ON; OpenCL is optional."
        )
    torch.ops.load_library(str(path))
    torch.ops.scrapgpu.load()


def has_device() -> bool:
    """Return whether the selected backend has devices; this fixes backend configuration."""
    try:
        torch.ops.load_library(str(_default_ext_path()))
        return int(torch.ops.scrapgpu.device_count()) > 0
    except (OSError, RuntimeError):
        return False


def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: torch.Tensor | None = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    *,
    scale: float | None = None,
    enable_gqa: bool = False,
    query_position_offset: int = 0,
    max_workspace_bytes: int = 64 * 1024 * 1024,
) -> torch.Tensor:
    """Run explicit inference SDPA on contiguous CPU float32 [B,H,L,D] tensors.

    Mask is float32 additive or bool (True means visible), broadcastable to
    [B,H,L,S]. Offset supports KV-cache decoding: causal visibility is
    key_index <= query_index + query_position_offset. Workspace bounds backend
    scratch allocations, excluding input/output transfers. Dropout and autograd
    are rejected. This does not replace torch.nn.functional attention globally.
    """
    return torch.ops.scrapgpu.scaled_dot_product_attention(
        query, key, value, attn_mask, dropout_p, is_causal, scale, enable_gqa,
        query_position_offset, max_workspace_bytes,
    )


class ScrapLinear(nn.Module):
    """Linear layer using scrapgpu.linear on CPU float activations (no bias)."""

    def __init__(self, in_features: int, out_features: int, quant_type: str = "f32") -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.quant_type = quant_type
        w = torch.empty(out_features, in_features)
        nn.init.kaiming_uniform_(w, a=5**0.5)
        self.register_buffer("weight", w)
        self._scrap_weight = None
        self._weight_version = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Inference tensors have no version counter, so conservatively repack them.
        version = None if torch.is_inference(self.weight) else (self.weight.data_ptr(), self.weight._version)
        if self._scrap_weight is None or version is None or self._weight_version != version:
            self._scrap_weight = torch.classes.scrapgpu.Weight(self.weight, self.quant_type)
            self._weight_version = version
        return torch.ops.scrapgpu.linear(x, self._scrap_weight)


# ---------------------------------------------------------------------------------------------
# Device-resident tensors (stage B)
#
# A DTensor (torch.classes.scrapgpu.DTensor) lives in a ScrapGpu device buffer and uses the ggml dimension
# convention: ne[0] is the fastest-varying dimension, so a torch tensor of shape (R, N, D) is ne = [D, N, R, 1].
# Views (view/reshape/permute/narrow) share the buffer; every op below returns a new DTensor and moves no data
# to the host. Only to_device(), to_cpu() and sync() cross the host boundary. An op the backend cannot run
# raises RuntimeError with the backend's reason; there is no CPU fallback.
# ---------------------------------------------------------------------------------------------

_UNARY_SILU, _UNARY_SIGMOID = 0, 4


def to_device(t: torch.Tensor):
    """Upload a CPU float32/float16/int32 tensor (1-4 dims) to a new DTensor (one host->device transfer)."""
    return torch.ops.scrapgpu.dt_from_cpu(t)


def to_cpu(x) -> torch.Tensor:
    """Download a DTensor (made contiguous on the device first) as a 4-D tensor (ne3, ne2, ne1, ne0)."""
    return x.to_cpu()


def torch_shape(x) -> tuple[int, ...]:
    """The torch-order shape (ne3, ne2, ne1, ne0) of a DTensor."""
    return tuple(reversed(x.ne()))


def mul_mat(weight, x):
    """y[., o] = sum_i weight[o, i] * x[., i] for a Weight [out, in] and x with ne[0] == in; new DTensor."""
    return torch.ops.scrapgpu.dt_mul_mat(weight, x)


def empty(ne, dtype: str = "f32"):
    """An uninitialized dense DTensor with ggml sizes ne (4 values, ne[0] fastest); a destination for mul_mat_into."""
    return torch.ops.scrapgpu.dt_empty(list(ne), dtype)


def mul_mat_into(weight, x, dst):
    """mul_mat written into dst, an F32 tensor or view [out, x.ne1, x.ne2, x.ne3] (e.g. a narrow() column block of a
    larger result). The only op that writes an existing DTensor; returns dst."""
    return torch.ops.scrapgpu.dt_mul_mat_into(weight, x, dst)


def add(a, b):
    """a + b; b is repeated along every dimension where it is smaller (a bias row is ne = [n, 1, 1, 1])."""
    return torch.ops.scrapgpu.dt_add(a, b)


def mul(a, b):
    """a * b with the same broadcasting of b."""
    return torch.ops.scrapgpu.dt_mul(a, b)


def silu(x):
    return torch.ops.scrapgpu.dt_unary(x, _UNARY_SILU)


def sigmoid(x):
    return torch.ops.scrapgpu.dt_unary(x, _UNARY_SIGMOID)


def swiglu(x, swapped: bool = False):
    """Split ne[0] in halves (a, b): silu(a) * b, or with swapped=True silu(b) * a (torch: gate, up = chunk(2); gate * silu(up))."""
    return torch.ops.scrapgpu.dt_glu(x, 0, 1 if swapped else 0)


def scale(x, s: float, bias: float = 0.0):
    return torch.ops.scrapgpu.dt_scale(x, float(s), float(bias))


def rms_norm_mul(x, w, eps: float):
    """rms_norm(x over ne[0]) * w, with w a DTensor holding ne[0] floats."""
    return torch.ops.scrapgpu.dt_rms_norm_mul(x, w, float(eps))


def rope(x, pos, n_dims: int = 0, freq_base: float = 10000.0, *, neox: bool = True, n_ctx_orig: int = 0,
         freq_scale: float = 1.0, ext_factor: float = 0.0, attn_factor: float = 1.0,
         beta_fast: float = 32.0, beta_slow: float = 1.0):
    """Rotary embedding of x [head_dim, heads, tokens, batch] with int32 positions [tokens] shared by the batch."""
    return torch.ops.scrapgpu.dt_rope(
        x, pos, n_dims, n_ctx_orig, float(freq_base), float(freq_scale), float(ext_factor), float(attn_factor),
        float(beta_fast), float(beta_slow), 1 if neox else 0,
    )


def concat(a, b, dim: int):
    return torch.ops.scrapgpu.dt_concat(a, b, dim)


def sdpa(q, k, v, mask=None, scale: float | None = None):
    """Scaled dot-product attention on DTensors q [D,Q,Hq,B], k [D,K,Hkv,B], v [Dv,K,Hkv,B].

    mask: additive F32 DTensor [K,Q,1|Hq,1|B] (-inf blocks a key); a query with every key blocked outputs zeros
    (unlike torch, which returns NaN). All (batch, head) pairs run as batched device ops, so the mask never
    visits the host. Returns [Dv,Q,Hq,B].
    """
    if scale is None:
        scale = 1.0 / (q.ne()[0] ** 0.5)
    if mask is None:
        return torch.ops.scrapgpu.dt_sdpa(q, k, v, float(scale))
    return torch.ops.scrapgpu.dt_sdpa_mask(q, k, v, mask, float(scale))


def sync() -> None:
    """Wait for the queued device work (to_cpu also waits)."""
    torch.ops.scrapgpu.dt_sync()


class TransferStats(tuple):
    """(h2d_calls, h2d_bytes, d2h_calls, d2h_bytes, ops, pool_creates, pool_hits, pool_idle_bytes, live_bytes)."""

    __slots__ = ()
    h2d_calls = property(lambda self: self[0])
    h2d_bytes = property(lambda self: self[1])
    d2h_calls = property(lambda self: self[2])
    d2h_bytes = property(lambda self: self[3])
    ops = property(lambda self: self[4])
    pool_creates = property(lambda self: self[5])
    pool_hits = property(lambda self: self[6])
    pool_idle_bytes = property(lambda self: self[7])
    live_bytes = property(lambda self: self[8])


def stats() -> TransferStats:
    """Process-wide host<->device transfer counters of the ScrapGpu public API plus device-tensor pool counters."""
    return TransferStats(torch.ops.scrapgpu.dt_stats())


def reset_stats() -> None:
    torch.ops.scrapgpu.dt_stats_reset()


def pool_trim() -> None:
    """Free the idle pooled device buffers."""
    torch.ops.scrapgpu.dt_pool_trim()


def profile_print() -> None:
    """With GGML_CL11_PROFILE=1 (OpenCL backend): print the kernel-time table per op to stderr and clear it."""
    torch.ops.scrapgpu.opencl11_profile_print()
