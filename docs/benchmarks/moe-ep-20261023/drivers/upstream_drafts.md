# Upstream report drafts (not posted)

Drafts for Mesa's GitLab (https://gitlab.freedesktop.org/mesa/mesa/-/issues). Posting needs a freedesktop.org account, so the user posts them. Related existing issue found: [#7432](https://gitlab.freedesktop.org/mesa/mesa/-/issues/7432) (`unknown nir_op flrp` in `nv50_ir_from_nir.cpp` on Kepler with Clover / rusticl) — the same class of missing case, a different op. No issue for `f2f16` was found (web search, 2026-10-24; the GitLab search itself was not run).

---

## Draft 1: nouveau (nvc0 codegen): `unknown nir_op f2f16` aborts rusticl on Kepler (vstore_half)

**System**: Debian 13, Linux 6.12.107, Mesa 25.0.7 (Debian `mesa-opencl-icd`), GeForce GT 730 and GT 710 (GK208B, NV106), nouveau with `modeset=1`, `RUSTICL_ENABLE=nouveau`.

**Steps**: build this kernel with `clBuildProgram` on the rusticl nouveau device (`repro/half2.cl`, `repro/clbuild.c` in our repository):

```c
__kernel void k(__global const float *a, __global float *o) {
    size_t i = get_global_id(0);
    vstore_half(a[i], i, (__global half *)o);
}
```

**Result**: the process aborts during the build:

```
ERROR: unknown nir_op f2f16
nvc0_program_translate:659 - shader translation failed: -2
```

(exit status 136). The same happens on both cards. A kernel with `fma` only and one with `vload_half` (f16 → f32) build and run.

**Expected**: either `f2f16` is lowered / supported, or the build fails with `CL_BUILD_PROGRAM_FAILURE` instead of aborting the process.

**Notes**: looks like the same kind of gap as #7432 (`flrp`). `vstore_half` is common in OpenCL code that stores fp16 (llama.cpp-style kernels hit it on every build).

---

## Draft 2 (hold): NVK on Kepler (Mesa 25.0.7): compute pipelines with f16 fail with VK_ERROR_UNKNOWN

**Hold reason**: Kepler support in NVK is only advertised from Mesa 25.2 (25.0.7 needs `NVK_I_WANT_A_BROKEN_VULKAN_DRIVER=1`). Re-test on Mesa ≥ 25.2.1 before posting; a report against 25.0.7 is likely to be closed as unsupported.

**System**: as above, `VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nouveau_icd.json NVK_I_WANT_A_BROKEN_VULKAN_DRIVER=1`, apiVersion 1.0.305.

**Result**: in llama.cpp's `test-backend-ops -b Vulkan0`, MUL_MAT (q4_0, f32), SCALE, CONT, RMS_NORM, SOFT_MAX, SSM_CONV, ARGSORT and SUM_ROWS pass; every op whose pipeline uses f16 (e.g. `add_f16_f16_f16_norepeat`, `contig_cpy_f32_f16`, `rope_norm_f16`, `clamp_f16`, `mul_mat_vec_id_q4_k_f32`) fails at `vkCreateComputePipelines` with `VK_ERROR_UNKNOWN` (list in `logs/06_stage_a_nouveau_kepler/nvk_per_op.txt`).
