#!/usr/bin/env python3
"""Device-resident tensors (torch.classes.scrapgpu.DTensor): every op against torch, on the selected backend.

Run once per backend, e.g.
  SCRAPGPU_TORCH_BACKEND=cpu python tests/test_device_tensor.py
  GGML_CL11_PLATFORMS=NVIDIA GGML_CL11_DEVICES="GT 430" SCRAPGPU_TORCH_BACKEND=opencl11 python tests/test_device_tensor.py

Prints "SKIP: ..." and exits 0 without a device (1 for the cpu backend, which must always exist).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import scrapgpu_torch as sgt  # noqa: E402

LIMIT = 1e-6


def nmse(got: torch.Tensor, ref: torch.Tensor) -> float:
    got = got.double().reshape(-1)
    ref = ref.double().reshape(-1)
    assert not torch.isnan(got).any(), "NaN in result"
    num = ((got - ref) ** 2).sum().item()
    den = (ref**2).sum().item()
    return num / den if den > 0 else num


def dev(t: torch.Tensor):
    return sgt.to_device(t.contiguous())


def host(x, shape=None) -> torch.Tensor:
    out = sgt.to_cpu(x)
    return out.reshape(shape) if shape is not None else out


def rejects(fn, text):
    try:
        fn()
    except (RuntimeError, TypeError) as exc:
        assert text in str(exc), str(exc)
    else:
        raise AssertionError(f"expected rejection containing: {text}")


def main() -> int:
    torch.set_num_threads(1)
    backend = os.getenv("SCRAPGPU_TORCH_BACKEND", "opencl11")
    if not sgt.has_device():
        print("SKIP: no selected ScrapGpu device")
        return 1 if backend == "cpu" else 0
    sgt.load()
    print(f"device: {torch.ops.scrapgpu.device_name(int(os.getenv('SCRAPGPU_TORCH_DEVICE', '0')))}; backend={backend}; torch={torch.__version__}")

    torch.manual_seed(7)
    results: dict[str, float] = {}

    # --- upload / download / dtype / shape (torch order is the reverse of ne) ---
    for shape in ((7,), (3, 5), (2, 3, 5), (2, 3, 4, 5)):
        t = torch.randn(*shape)
        x = dev(t)
        ne = x.ne()
        assert list(reversed(ne[: len(shape)])) == list(shape) and all(n == 1 for n in ne[len(shape):]), (shape, ne)
        results[f"roundtrip_f32_{len(shape)}d"] = nmse(host(x), t)
    ti = torch.arange(10, dtype=torch.int32)
    assert torch.equal(host(dev(ti)).reshape(-1), ti)
    th = torch.randn(6, 4).half()
    assert torch.equal(host(dev(th)).reshape(6, 4), th)
    print("PASS: upload/download f32/i32/f16")

    # --- views: permute / narrow / reshape / view / cont (descriptor only, cont on the device) ---
    t = torch.randn(3, 5, 7)  # ne = [7, 5, 3, 1]
    x = dev(t)
    # permute(order): result dim i is source dim order[i] (ggml order; here ne0<->ne1)
    p = x.permute([1, 0, 2, 3])
    assert not p.is_contiguous()
    results["permute_cont"] = nmse(host(p.cont(), (3, 7, 5)), t.transpose(1, 2))
    results["permute_download_noncontig"] = nmse(host(p, (3, 7, 5)), t.transpose(1, 2))
    nar = x.narrow(2, 1, 2)  # batch rows 1..2
    results["narrow"] = nmse(host(nar, (2, 5, 7)), t[1:3])
    nar0 = x.narrow(0, 2, 3)
    results["narrow_inner"] = nmse(host(nar0, (3, 5, 3)), t[:, :, 2:5])
    results["reshape"] = nmse(host(x.reshape([35, 3, 1, 1]), (3, 35)), t.reshape(3, 35))
    results["view_stride"] = nmse(host(x.view([7, 3, 1, 1], [4, 140, 560, 560], 0), (3, 7)), t[:, 0, :])
    rejects(lambda: p.reshape([35, 3, 1, 1]), "contiguous")
    rejects(lambda: x.narrow(1, 4, 3), "outside")
    rejects(lambda: x.permute([0, 0, 1, 2]), "permutation")
    print("PASS: views")

    # --- mul_mat with Weight (f32 and f16), 2-D / 3-D / permuted-view activations ---
    k, m = 96, 40
    w = torch.randn(m, k)
    for qt, ref_w in (("f32", w), ("f16", w.half().float())):
        wo = torch.classes.scrapgpu.Weight(w, qt)
        x2 = torch.randn(5, k)
        results[f"mul_mat_{qt}_2d"] = nmse(host(sgt.mul_mat(wo, dev(x2)), (5, m)), F.linear(x2, ref_w))
        x3 = torch.randn(3, 6, k)  # batch 3, 6 tokens
        results[f"mul_mat_{qt}_3d"] = nmse(host(sgt.mul_mat(wo, dev(x3)), (3, 6, m)), F.linear(x3, ref_w))
        # variate-attention layout: the activation is a (R, N, D) tensor viewed as (N, R, D)
        xv = dev(x3).permute([0, 2, 1, 3])
        results[f"mul_mat_{qt}_permuted_view"] = nmse(host(sgt.mul_mat(wo, xv), (6, 3, m)), F.linear(x3.transpose(0, 1), ref_w))
    rejects(lambda: sgt.mul_mat(wo, dev(torch.randn(2, k + 1))), "in_features")
    # mul_mat_into: column blocks (a tile-sized block plus single columns) written into one result tensor
    wf = torch.classes.scrapgpu.Weight(w, "f16")
    xc = torch.randn(9, k)
    xcd = dev(xc)
    dst = sgt.empty([m, 9, 1, 1])
    sgt.mul_mat_into(wf, xcd.narrow(1, 0, 6), dst.narrow(1, 0, 6))
    for c in range(6, 9):
        assert sgt.mul_mat_into(wf, xcd.narrow(1, c, 1), dst.narrow(1, c, 1)) is not None
    results["mul_mat_into_blocks"] = nmse(host(dst, (9, m)), F.linear(xc, w.half().float()))
    rejects(lambda: sgt.mul_mat_into(wf, xcd, dst.narrow(1, 0, 8)), "destination")
    print("PASS: mul_mat")

    # --- add (bias row broadcast) / mul / silu / sigmoid / swiglu ---
    a = torch.randn(4, 6, 33)
    bias = torch.randn(33)
    results["add_bias"] = nmse(host(sgt.add(dev(a), dev(bias)), (4, 6, 33)), a + bias)
    b = torch.randn(4, 6, 33)
    results["add_same"] = nmse(host(sgt.add(dev(a), dev(b)), (4, 6, 33)), a + b)
    results["mul"] = nmse(host(sgt.mul(dev(a), dev(b)), (4, 6, 33)), a * b)
    results["mul_strided_rhs"] = nmse(
        host(sgt.mul(dev(a), dev(b.transpose(0, 1).contiguous()).permute([0, 2, 1, 3])), (4, 6, 33)), a * b)
    results["silu"] = nmse(host(sgt.silu(dev(a)), a.shape), F.silu(a))
    results["sigmoid"] = nmse(host(sgt.sigmoid(dev(a)), a.shape), torch.sigmoid(a))
    g = torch.randn(5, 3, 64)
    gate, up = g.chunk(2, dim=-1)
    results["swiglu_gate_silu_up"] = nmse(host(sgt.swiglu(dev(g), swapped=True), (5, 3, 32)), gate * F.silu(up))
    results["swiglu_silu_first"] = nmse(host(sgt.swiglu(dev(g), swapped=False), (5, 3, 32)), F.silu(gate) * up)
    results["scale"] = nmse(host(sgt.scale(dev(a), 0.5, 2.0), a.shape), a * 0.5 + 2.0)
    rejects(lambda: sgt.add(dev(ti), dev(ti)), "float32")
    rejects(lambda: sgt.add(dev(a), dev(torch.randn(5))), "broadcast")
    print("PASS: elementwise")

    # --- rms_norm * weight ---
    xn = torch.randn(3, 7, 64)
    wn = torch.randn(64)
    eps = 1e-6
    ref = xn * torch.rsqrt(xn.pow(2).mean(-1, keepdim=True) + eps) * wn
    results["rms_norm_mul"] = nmse(host(sgt.rms_norm_mul(dev(xn), dev(wn), eps), xn.shape), ref)
    xs = dev(torch.randn(7, 3, 64)).permute([0, 2, 1, 3])  # strided activation view [64, 3, 7]
    xs_t = host(xs, (7, 3, 64))
    results["rms_norm_mul_view"] = nmse(host(sgt.rms_norm_mul(xs, dev(wn), eps), xs_t.shape),
                                        xs_t * torch.rsqrt(xs_t.pow(2).mean(-1, keepdim=True) + eps) * wn)
    print("PASS: rms_norm_mul")

    # --- rope: ggml layout [head_dim, heads, tokens, batch], positions shared over the batch ---
    def torch_rope(x, pos, n_dims, neox, base=10000.0):
        """x (..., tokens, heads, head_dim): rotate the first n_dims dims (neox: pairs (j, j+n/2); normal: (2j, 2j+1))."""
        half = n_dims // 2
        ang = pos.float()[:, None] * base ** (-torch.arange(half).float() * 2 / n_dims)[None, :]  # (tokens, half)
        cos, sin = ang.cos()[:, None, :], ang.sin()[:, None, :]
        y = x.clone()
        if neox:
            x0, x1 = x[..., :half], x[..., half:n_dims]
            y[..., :half] = x0 * cos - x1 * sin
            y[..., half:n_dims] = x0 * sin + x1 * cos
        else:
            x0, x1 = x[..., 0:n_dims:2], x[..., 1:n_dims:2]
            y[..., 0:n_dims:2] = x0 * cos - x1 * sin
            y[..., 1:n_dims:2] = x0 * sin + x1 * cos
        return y

    hd, nh, nt, nb = 64, 4, 9, 2
    xr = torch.randn(nb, nt, nh, hd)  # ne = [64, 4, 9, 2]
    pos = torch.randint(0, 200, (nt,), dtype=torch.int32)
    for neox, n_dims in ((True, 32), (True, 64), (False, 64)):
        got = host(sgt.rope(dev(xr), dev(pos), n_dims, 10000.0, neox=neox), xr.shape)
        results[f"rope_neox{int(neox)}_{n_dims}"] = nmse(got, torch_rope(xr, pos, n_dims, neox))
    # a strided input (the q rows of a fused qkv projection)
    qkv = torch.randn(nb, nt, 3 * nh * hd)
    qv = dev(qkv).reshape([3 * nh * hd, nt, nb, 1])
    qv = qv.view([hd, nh, nt, nb], [4, hd * 4, 3 * nh * hd * 4, 3 * nh * hd * 4 * nt], 0)
    got = host(sgt.rope(qv, dev(pos), 32, 10000.0), (nb, nt, nh, hd))
    results["rope_strided_input"] = nmse(got, torch_rope(qkv[..., : nh * hd].reshape(nb, nt, nh, hd), pos, 32, True))
    rejects(lambda: sgt.rope(dev(xr), dev(pos.float()), 32), "int32")
    print("PASS: rope")

    # --- concat ---
    c1, c2 = torch.randn(2, 3, 5), torch.randn(2, 3, 4)
    results["concat_dim0"] = nmse(host(sgt.concat(dev(c1), dev(c2), 0), (2, 3, 9)), torch.cat([c1, c2], dim=-1))
    c3 = torch.randn(2, 6, 5)
    results["concat_dim1"] = nmse(host(sgt.concat(dev(c1), dev(c3), 1), (2, 9, 5)), torch.cat([c1, c3], dim=1))
    print("PASS: concat")

    # --- sdpa: torch layout q [B, H, Q, D] <-> DTensor [D, Q, H, B] ---
    def sdpa_case(name, b, hq, hkv, ql, kl, d, mask_kind):
        q = torch.randn(b, hq, ql, d)
        kk = torch.randn(b, hkv, kl, d)
        vv = torch.randn(b, hkv, kl, d)
        scale = 0.37
        if mask_kind is None:
            add_mask = None
            dmask = None
        else:
            visible = torch.rand(b, 1 if mask_kind in ("b", "full_row") else hq, ql, kl) > 0.4
            visible[..., 0] = True
            if mask_kind == "full_row":
                visible = visible.clone()
                visible[-1, 0, 1, :] = False  # a query with no visible key
            add_mask = torch.where(visible, 0.0, -float("inf"))
            dmask = dev(add_mask)  # [b, 1|h, ql, kl] -> ne = [kl, ql, 1|h, b]
        ref = F.scaled_dot_product_attention(q, kk.repeat_interleave(hq // hkv, dim=1), vv.repeat_interleave(hq // hkv, dim=1),
                                             attn_mask=add_mask, scale=scale)
        if mask_kind == "full_row":
            # ScrapGpu defines a fully blocked query as exact zeros. Older torch returns NaN for it, newer torch
            # zeros: either way the reference below is zero there.
            blocked = ref[-1, :, 1]
            assert bool((torch.isnan(blocked) | (blocked == 0)).all()), "unexpected torch value for a fully blocked query"
            ref = torch.nan_to_num(ref, nan=0.0)
        got = sgt.sdpa(dev(q.permute(0, 1, 2, 3)), dev(kk), dev(vv), dmask, scale=scale)
        out = host(got, (b, hq, ql, d))
        results[name] = nmse(out, ref)
        if mask_kind == "full_row":
            assert (out[-1, :, 1] == 0).all(), "fully blocked query must be exactly zero"

    sdpa_case("sdpa_nomask", 2, 3, 3, 5, 7, 16, None)
    sdpa_case("sdpa_mask_per_batch", 2, 3, 3, 9, 9, 16, "b")
    sdpa_case("sdpa_mask_per_head", 2, 4, 4, 6, 8, 16, "h")
    sdpa_case("sdpa_gqa", 2, 4, 2, 6, 8, 16, "b")
    sdpa_case("sdpa_fully_masked_row", 2, 3, 3, 4, 6, 16, "full_row")
    # non-contiguous q/k/v are made contiguous on the device (head split of a fused qkv view)
    qkv = torch.randn(2, 5, 3, 4, 8)  # (B, N, qkv, H, D) -> ne = [8, 4, 3, 5, 2]; reshape to the fused row view
    fused = dev(qkv.reshape(2, 5, 3 * 4 * 8))  # ne = [96, 5, 2]
    def part(i):
        return fused.view([8, 4, 5, 2], [4, 32, 96 * 4, 96 * 5 * 4], i * 4 * 8 * 4).permute([0, 2, 1, 3])  # [D, N, H, B]
    qd, kd, vd = part(0), part(1), part(2)
    got = host(sgt.sdpa(qd, kd, vd, None, scale=0.5), (2, 4, 5, 8))
    qt_ = qkv[:, :, 0].permute(0, 2, 1, 3)
    ref = F.scaled_dot_product_attention(qt_, qkv[:, :, 1].permute(0, 2, 1, 3), qkv[:, :, 2].permute(0, 2, 1, 3), scale=0.5)
    results["sdpa_strided_qkv_views"] = nmse(got, ref)
    print("PASS: sdpa")

    # --- an op the backend cannot run is an error with the backend's reason, never a CPU fallback ---
    bad = dev(torch.randn(4, 8)).permute([1, 0, 2, 3])  # nb[0] != 4: MUL_MAT needs contiguous rows
    rejects(lambda: sgt.mul_mat(torch.classes.scrapgpu.Weight(torch.randn(3, 4), "f32"), bad), "no CPU fallback")
    print("PASS: unsupported op raises")

    # --- device residency: a chain of ops moves nothing; only the boundary calls transfer ---
    wo = torch.classes.scrapgpu.Weight(torch.randn(48, 32), "f16")
    x0 = torch.randn(2, 6, 32)
    sgt.sync()
    sgt.reset_stats()
    xd = dev(x0)
    for _ in range(3):
        h = sgt.mul_mat(wo, xd)
        h = sgt.silu(sgt.add(h, dev(torch.zeros(48))))
        xd = sgt.mul_mat(torch.classes.scrapgpu.Weight(torch.randn(32, 48), "f16"), h)
    s0 = sgt.stats()
    # 1 input upload + 3 bias uploads + 3 weight uploads (Weight construction); no downloads at all
    assert s0.d2h_calls == 0 and s0.d2h_bytes == 0, s0
    assert s0.ops == 3 * 4, s0
    host(xd)
    s1 = sgt.stats()
    assert s1.d2h_calls == 1, s1
    print(f"PASS: residency (h2d_calls={s1.h2d_calls}, d2h_calls={s1.d2h_calls} after {s1.ops} device ops)")

    # --- the buffer pool recycles: the same loop again creates no new device buffers ---
    def step():
        z = dev(x0)
        for _ in range(4):
            z = sgt.add(z, z)
        return z
    sgt.pool_trim()
    for _ in range(3):
        step()
    sgt.sync()
    before = sgt.stats()
    for _ in range(5):
        step()
    sgt.sync()
    after = sgt.stats()
    assert after.pool_creates == before.pool_creates, (before, after)
    assert after.pool_hits > before.pool_hits
    print(f"PASS: pool reuse (creates {after.pool_creates}, hits {after.pool_hits})")

    failed = []
    for name, v in results.items():
        ok = v < LIMIT
        print(f"  {name}: NMSE {v:.3e} {'OK' if ok else 'FAIL'}")
        if not ok:
            failed.append(name)
    if failed:
        print("FAILED:", ", ".join(failed))
        return 1
    print(f"all device tensor ops OK ({len(results)} numeric comparisons); backend={backend}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
