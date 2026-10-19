"""Timestral-209M with the whole transformer resident on a ScrapGpu device (GT 430 class OpenCL 1.1 GPUs).

Replaces `model.transformer` (timestral.layers.Transformer) by `GpuTransformer`: weights are uploaded once (F16, ~420 MB),
the 24 layers (time / variate attention with RoPE and sigmoid gate, SwiGLU MLP, RMSNorm) run as ScrapGpu device
tensors, attention included. Patch embedding, mask generation and the quantile head stay on the CPU.

One forward moves host data only at its boundaries:
  host -> device: the input tokens x (R, N, D), the time mask, the variate mask (when R > 1), the positions
                  (one row, or R rows when the rows' positions differ)
  device -> host: the out_norm output (R, N, D)
`--check-transfers` prints the ScrapGpu transfer counters of every forward to show it.

Layout choices that matter on the GT 430 (see the comments at `linear`, `_sdpa` and `_rope_head_perm`):
  * wQKV rows are re-ordered once at upload to (qkv, head, dim), so q / k / v are plain strided views of the product,
    and q / k get the head-dim order that turns the torch RoPE (rotate_half over the whole head, zero-padded
    inv_freq) into ggml NEOX with n_dims = 32;
  * every weight product runs on the dense [D, N*R] token matrix; variate attention only re-views the product;
  * mat-mat tile planning for N*R columns (full 128-column tiles + a few mat-vec columns);
  * variate attention over R <= 6 rows runs one sdpa per query row.

env for the GT 430:
  GGML_CL11_PLATFORMS=NVIDIA GGML_CL11_DEVICES="GT 430" SCRAPGPU_TORCH_BACKEND=opencl11 SCRAPGPU_TORCH_DEVICE=0
  PYTHONPATH=<ScrapGpuLibrary>/adapters/pytorch
`GGML_CL11_PROFILE=1 GGML_CL11_PROFILE_SHAPES=1` with `--profile` prints the kernel time per op.
"""

from __future__ import annotations

import argparse
import math
import time

import numpy as np
import torch
from torch import nn

import scrapgpu_torch as sgt
from timestral import TimestralForecaster
from timestral.layers import Attention, Transformer, TwoStreamMask


def _rope_head_perm(head_dim: int, rotary_dim: int) -> list[int]:
    """New head-dim order for q and k.

    The torch RoPE rotates pairs (i, i + head_dim / 2) for i < rotary_dim / 2 (rotate_half over the whole head with a
    zero-padded inv_freq, so every other dim is an identity rotation). ggml NEOX with n_dims = rotary_dim rotates
    pairs (j, j + rotary_dim / 2) of the first rotary_dim dims. Putting dims [0 .. r/2) and [hd/2 .. hd/2 + r/2) in
    front, in that order, turns the former into the latter; the rest follow unrotated. q . k is unchanged because q
    and k get the same permutation.
    """
    half, r2 = head_dim // 2, rotary_dim // 2
    front = list(range(r2)) + list(range(half, half + r2))
    rest = [i for i in range(head_dim) if i not in set(front)]
    return front + rest


class _DAttention:
    """Device weights of one Attention module (time or variate)."""

    def __init__(self, attn: Attention, qtype: str, rope_perm: list[int]) -> None:
        h, hd, dm = attn.num_heads, attn.head_dim, attn.embed_dim
        self.is_time = attn.is_time
        self.heads, self.head_dim, self.dim = h, hd, dm
        w = attn.wQKV.weight.detach().float()  # rows: (qkv, head_dim, heads), heads fastest
        assert attn.wQKV.bias is None and attn.wO.bias is None, "attention bias is not implemented"
        w = w.reshape(3, hd, h, dm)
        perm = torch.tensor(rope_perm)
        parts = []
        for i in range(3):
            wi = w[i].permute(1, 0, 2)  # (heads, head_dim, dm): row = h * head_dim + d
            if i < 2 and attn.is_time:
                wi = wi[:, perm]  # q and k: RoPE-friendly head dim order (the same for both)
            parts.append(wi.reshape(h * hd, dm))
        self.wqkv = torch.classes.scrapgpu.Weight(torch.cat(parts).contiguous(), qtype)
        self.wo = torch.classes.scrapgpu.Weight(attn.wO.weight.detach().float().contiguous(), qtype)
        self.gate = None
        if attn._gate is not None:
            assert attn._gate.bias is None
            self.gate = torch.classes.scrapgpu.Weight(attn._gate.weight.detach().float().contiguous(), qtype)


# Column plan of the weight matrix products on a GT 430 (Fermi, `k_mm_*` mat-mat kernel, tile 64 x 128, measured with
# F16 weights, M = 4096, K = 768): a product costs ~9.3 ms per started 128-column tile whatever the number of columns
# in it (N <= 128: 7.6 ms + 0.013 ms per column) and a single-column mat-vec ~0.54 ms. A remainder of up to
# MV_COLS = 12 columns is therefore cheaper as that many mat-vecs than as one more (almost empty) tile.
MM_TILE_COLS = 128
MV_COLS = 12
VARIATE_SPLIT_MAX_ROWS = 6


def linear(w, x, tile: int = MM_TILE_COLS, mv_cols: int = MV_COLS):
    """w @ x on the packed [D, N, R] layout, as products over all N * R token columns.

    Full tiles of `tile` columns go through the mat-mat kernel, a remainder of at most `mv_cols` columns through
    one mat-vec per column, a larger remainder through one more mat-mat (padded to a tile). All pieces write
    their column block of one result tensor (mul_mat_into): no concat, no host transfer.
    """
    d, n, r, _ = x.ne()
    cols = n * r
    xf = x.reshape([d, cols, 1, 1])
    full = cols // tile * tile
    rest = cols - full
    if rest == 0 or (full == 0 and rest > mv_cols):  # nothing to split
        y = sgt.mul_mat(w, xf)
        return y.reshape([y.ne()[0], n, r, 1])
    m = w.out_features()
    y = sgt.empty([m, cols, 1, 1])
    if full:
        sgt.mul_mat_into(w, xf.narrow(1, 0, full), y.narrow(1, 0, full))
    if rest <= mv_cols:
        for c in range(full, cols):
            sgt.mul_mat_into(w, xf.narrow(1, c, 1), y.narrow(1, c, 1))
    else:
        sgt.mul_mat_into(w, xf.narrow(1, full, rest), y.narrow(1, full, rest))
    return y.reshape([m, n, r, 1])


class GpuTransformer(nn.Module):
    """Drop-in for timestral.layers.Transformer: forward(x, time_mask, variate_mask, positions) -> (R, N, D)."""

    def __init__(self, tr: Transformer, qtype: str = "f16") -> None:
        super().__init__()
        sgt.load()
        self.qtype = qtype
        r = tr.rope
        self.embed_dim = tr.out_norm.scale.numel()
        self.head_dim = r.dim
        self.rotary_dim = r.rotary_dim
        self.rope_base = float(r.base)
        self.eps = float(tr.out_norm.eps)
        perm = _rope_head_perm(self.head_dim, self.rotary_dim)
        self.layers = []
        for layer in tr.layers:
            attns = [_DAttention(a, qtype, perm) for a in layer.attns]
            norms = [sgt.to_device(n.scale.detach().float()) for n in layer.attn_norms]
            mlp = layer.mlp
            assert mlp.w12.bias is None and mlp.w3.bias is None, "MLP bias is not implemented"
            self.layers.append({
                "attns": attns,
                "attn_norms": norms,
                "mlp_norm": sgt.to_device(layer.mlp_norm.scale.detach().float()),
                "w12": torch.classes.scrapgpu.Weight(mlp.w12.weight.detach().float().contiguous(), qtype),
                "w3": torch.classes.scrapgpu.Weight(mlp.w3.weight.detach().float().contiguous(), qtype),
            })
        self.out_norm = sgt.to_device(tr.out_norm.scale.detach().float())

    # ---- one attention sublayer on the device -------------------------------------------------

    def _attention(self, a: _DAttention, xn, mask, pos):
        """xn: normalized input in the packed layout, ne [D, N, R] (token = (row, patch)), dense.

        Time attention: sequence = N patches, batch = R rows. Variate attention: sequence = R rows, batch = N patches.
        The projections always run on the dense [D, N, R] layout (a matrix product with only R columns per batch
        would be slow); the variate layout is a strided view of their outputs. Returns wO(attn) as ne [D, N, R].
        """
        h, hd, dm = a.heads, a.head_dim, a.dim
        _, n, rows, _ = xn.ne()
        qkv = linear(a.wqkv, xn)  # [3 * dm, N, R]; rows (qkv, h, d)
        row = 3 * dm * 4
        nb = [4, hd * 4, row, row * n]

        def part(i):  # [hd, heads, N, R]
            return qkv.view([hd, h, n, rows], nb, i * dm * 4)

        q, k, v = part(0), part(1), part(2)
        if a.is_time:
            q = self._rope(q, pos, n, rows)
            k = self._rope(k, pos, n, rows)
            to_attn = [0, 2, 1, 3]  # [hd, heads, N, R] -> [hd, N, heads, R]: sequence N, batch R
            from_attn = [0, 2, 1, 3]  # [hd, N, heads, R] -> [hd, heads, N, R]
        else:
            to_attn = [0, 3, 1, 2]  # [hd, heads, N, R] -> [hd, R, heads, N]: sequence R, batch N
            from_attn = [0, 2, 3, 1]  # [hd, R, heads, N] -> [hd, heads, N, R]
        o = self._sdpa(q.permute(to_attn).cont(), k.permute(to_attn).cont(), v.permute(to_attn).cont(), mask,
                       1.0 / math.sqrt(hd), split_queries=not a.is_time and rows <= VARIATE_SPLIT_MAX_ROWS)
        o = o.permute(from_attn)  # view [hd, heads, N, R]
        if a.gate is not None:
            g = sgt.sigmoid(linear(a.gate, xn))  # [dm, N, R], row = h * hd + d
            o = sgt.mul(g.reshape([hd, h, n, rows]), o)  # dense [hd, heads, N, R]
        else:
            o = o.cont()
        return linear(a.wo, o.reshape([dm, n, rows, 1]))

    @staticmethod
    def _sdpa(q, k, v, mask, scale, split_queries: bool):
        """sgt.sdpa; with split_queries one call per query position (variate attention over a few rows).

        Variate attention is a [D, R, heads, N] problem with R (the number of rows of the batch) tiny: the batched
        score/value matrix products become 2-4 element tiles of a mat-mat kernel (8 ms for R = 2). One mat-vec-shaped
        call per query is ~5x faster up to R ~ 6 (measured on the GT 430), the outputs are joined along the query axis.
        """
        if not split_queries:
            return sgt.sdpa(q, k, v, mask, scale=scale)
        outs = []
        for i in range(q.ne()[1]):
            outs.append(sgt.sdpa(q.narrow(1, i, 1), k, v, None if mask is None else mask.narrow(1, i, 1), scale=scale))
        out = outs[0]
        for o in outs[1:]:
            out = sgt.concat(out, o, 1)
        return out

    def _rope(self, x, pos, seq: int, batch: int):
        """RoPE of x [hd, heads, seq, batch] (NEOX over the first rotary_dim dims, see _rope_head_perm)."""
        if isinstance(pos, list):  # per-row positions differ: rotate each batch row on its own
            outs = [sgt.rope(x.narrow(3, i, 1), p, self.rotary_dim, self.rope_base) for i, p in enumerate(pos)]
            out = outs[0]
            for o in outs[1:]:
                out = sgt.concat(out, o, 3)
            return out
        return sgt.rope(x, pos, self.rotary_dim, self.rope_base)

    # ---- the stack ----------------------------------------------------------------------------

    @staticmethod
    def _additive(mask: torch.Tensor | None) -> torch.Tensor | None:
        """bool (True = attend) -> additive float32 (0 / -inf)."""
        if mask is None:
            return None
        return torch.zeros(mask.shape, dtype=torch.float32).masked_fill_(~mask, float("-inf"))

    @torch.inference_mode()
    def forward(self, x, time_mask, variate_mask, positions):
        rows, n, d = x.shape
        assert d == self.embed_dim and positions.shape == (rows, n)
        if isinstance(time_mask, TwoStreamMask):
            # q[:enc] over the enc keys and q[enc:] over all keys is the full mask applied once: the enc->dec quadrant
            # of the mask is all False, so the context never reads the horizon either way
            time_mask = time_mask.mask
        if time_mask is None:  # torch: is_causal=True
            time_mask = torch.tril(torch.ones(n, n, dtype=torch.bool))[None, None]
        # ---- host -> device: x, the two masks, the positions (nothing else, for the whole forward) ----
        xd = sgt.to_device(x.detach().float().contiguous())  # ne [D, N, R]
        tmask = sgt.to_device(self._additive(time_mask))  # (R|1, 1, N, N) -> ne [N, N, 1, R|1]
        vmask = None if variate_mask is None else sgt.to_device(self._additive(variate_mask))  # (N, 1, R, R)
        pos_np = positions.to(torch.int32).contiguous()
        if bool((pos_np == pos_np[:1]).all()):
            pos = sgt.to_device(pos_np[0].contiguous())
        else:
            pos = [sgt.to_device(pos_np[i].contiguous()) for i in range(rows)]
        for layer in self.layers:
            for a, norm in zip(layer["attns"], layer["attn_norms"]):
                xn = sgt.rms_norm_mul(xd, norm, self.eps)
                xd = sgt.add(xd, self._attention(a, xn, tmask if a.is_time else vmask, pos if a.is_time else None))
            xn = sgt.rms_norm_mul(xd, layer["mlp_norm"], self.eps)
            hid = sgt.swiglu(linear(layer["w12"], xn), swapped=True)  # gate * silu(up), gate = first half
            xd = sgt.add(xd, linear(layer["w3"], hid))
        out = sgt.rms_norm_mul(xd, self.out_norm, self.eps)
        # ---- device -> host: the output ----
        return sgt.to_cpu(out).reshape(rows, n, d)


def install(model, qtype: str = "f16") -> GpuTransformer:
    gpu = GpuTransformer(model.transformer, qtype)
    model.transformer = gpu
    return gpu


# ---------------------------------------------------------------------------------------------
# benchmark / check
# ---------------------------------------------------------------------------------------------


def make_context(series: int, length: int) -> np.ndarray:
    rng = np.random.default_rng(0)
    t = np.arange(length)
    return np.stack([np.sin(2 * np.pi * t / 96 + i) * 10 + 0.01 * t + rng.normal(0, 0.5, length) for i in range(series)])[:, None, :]


def timed(fc, ctx, horizon: int, reps: int):
    fc.forecast(ctx, prediction_length=horizon)  # warm-up
    times = []
    out = None
    for _ in range(reps):
        t0 = time.perf_counter()
        out = fc.forecast(ctx, prediction_length=horizon)
        times.append(time.perf_counter() - t0)
    return out, sum(times) / len(times), times


def scenarios():
    """Inputs that exercise the mask / position paths: missing leading values (per-row positions), several variates
    per series (variate mask), a single row (no variate mask), covariates (two-stream window masks)."""
    rng = np.random.default_rng(1)

    def series(b, v, t):
        x = np.arange(t)
        return np.stack([[np.sin(2 * np.pi * x / (24 + 8 * i + j)) * (3 + i) + 0.02 * x + rng.normal(0, 0.3, t) for j in range(v)]
                         for i in range(b)])

    out = {}
    out["single row"] = dict(context=series(1, 1, 512), prediction_length=64)
    c = series(2, 1, 512)
    c[1, 0, :100] = np.nan  # leading padding: first observed patch differs per row, positions differ
    out["leading NaN, per-row positions"] = dict(context=c, prediction_length=64)
    c = series(2, 3, 480)
    c[0, 1, 50:70] = np.nan
    c[1, 2, :40] = np.nan
    out["2 series x 3 variates, gaps"] = dict(context=c, prediction_length=64)
    c = series(2, 1, 480)
    out["covariates (past + known)"] = dict(context=c, prediction_length=48, past_covariates=series(2, 1, 480),
                                            known_covariates=series(2, 1, 480 + 48))
    return out


def run_scenarios(fc, qtype: str) -> int:
    cases = scenarios()
    ref = {name: np.asarray(fc.forecast(**kw).quantiles) for name, kw in cases.items()}
    gpu = install(fc.model, qtype)
    bad = 0
    for name, kw in cases.items():
        got = np.asarray(fc.forecast(**kw).quantiles)
        rel = np.nanmax(np.abs(got - ref[name])) / np.nanmax(np.abs(ref[name]))
        ok = rel <= 1e-3
        bad += not ok
        print(f"scenario {name:34s} max|diff|/max|ref| = {rel:.3e}  {'OK' if ok else 'FAIL'}")
    return bad


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mistralai/Timestral-209M")
    ap.add_argument("--qtype", default="f16")
    ap.add_argument("--series", type=int, nargs="+", default=[2])
    ap.add_argument("--context", type=int, default=2016)
    ap.add_argument("--horizon", type=int, default=96)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--check-transfers", action="store_true")
    ap.add_argument("--scenarios", action="store_true", help="compare CPU and GPU on masked / multivariate / covariate inputs")
    ap.add_argument("--skip-cpu", action="store_true")
    ap.add_argument("--profile", action="store_true", help="print the per-op kernel time table (run with GGML_CL11_PROFILE=1)")
    a = ap.parse_args()
    torch.set_num_threads(2)

    fc = TimestralForecaster.from_pretrained(a.model)
    if a.scenarios:
        raise SystemExit(1 if run_scenarios(fc, a.qtype) else 0)
    refs = {}
    for s in a.series:
        ctx = make_context(s, a.context)
        if not a.skip_cpu:
            ref, t_cpu, ts = timed(fc, ctx, a.horizon, a.reps)
            refs[s] = (ctx, np.asarray(ref.quantiles), t_cpu)
            print(f"CPU   series={s}: {t_cpu * 1000:9.1f} ms/forecast (runs: {[round(x * 1000) for x in ts]})")

    t0 = time.perf_counter()
    gpu = install(fc.model, a.qtype)
    sgt.sync()
    print(f"uploaded transformer weights ({a.qtype}) in {time.perf_counter() - t0:.1f} s; device: {torch.ops.scrapgpu.device_name(0)}")

    for s in a.series:
        ctx = make_context(s, a.context)
        out, t_gpu, ts = timed(fc, ctx, a.horizon, a.reps)
        o = np.asarray(out.quantiles)
        line = f"GPU   series={s}: {t_gpu * 1000:9.1f} ms/forecast (runs: {[round(x * 1000) for x in ts]})"
        if s in refs:
            r = refs[s][1]
            rel = np.abs(o - r).max() / np.abs(r).max()
            line += f"  max|diff|/max|ref| = {rel:.3e}  speed vs CPU x{refs[s][2] / t_gpu:.2f}"
        print(line)

    if a.profile:
        s = a.series[0]
        sgt.profile_print()  # drop what the warm-up/timing runs collected
        fc.forecast(make_context(s, a.context), prediction_length=a.horizon)
        sgt.profile_print()

    if a.check_transfers:
        s = a.series[0]
        ctx = make_context(s, a.context)
        calls = []
        orig = gpu.forward

        def counted(*args, **kw):
            sgt.sync()
            sgt.reset_stats()
            res = orig(*args, **kw)
            st = sgt.stats()
            calls.append((st.h2d_calls, st.h2d_bytes, st.d2h_calls, st.d2h_bytes, st.ops))
            return res

        gpu.forward = counted
        fc.forecast(ctx, prediction_length=a.horizon)
        gpu.forward = orig
        print("per transformer forward (h2d_calls, h2d_bytes, d2h_calls, d2h_bytes, device ops):")
        for c in calls:
            print("  ", c)


if __name__ == "__main__":
    main()
