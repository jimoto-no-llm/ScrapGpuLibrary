# Contributing to ScrapGpuLibrary

（日本語の要約）ビルドは core 単体、ツリー外プラグイン、ツリー内パッチの 3 経路。変更したら `test-backend-ops`、`mvm_check` / `attn_check` / `mid_check`、`ctest` で正しさを確認し、ビルドオプション・起動回数・TB250 の性能は測って記録する。古いハードを使い切ることが最優先で、API・ドライバ・拡張は何を使ってもよい（合否は正しさのみ、他は記録）。ggml の include は `adapters/ggml` の外に出さない、正しさは許容値で判定。コミットは Conventional Commits。実機の結果は GPU・ドライバ・温度・コマンドを添えて報告する。第三者のコードは [docs/REUSE.md](docs/REUSE.md) の条件を守る。詳細は以下。

## Builds

### Core, without llama.cpp

```bash
cmake -S . -B build
cmake --build build
ctest --test-dir build
```

`SCRAPGPU_BUILD_CORE=ON` builds `scrapgpu_core` (no ggml, no OpenCL). `SCRAPGPU_BUILD_OPENCL11=ON` adds the OpenCL 1.1 backend (`CL_TARGET_OPENCL_VERSION=110`), `SCRAPGPU_BUILD_GGML_ADAPTER=ON` adds `adapters/ggml`. The root project never turns the adapter on by itself. `SCRAPGPU_BUILD_TESTS=ON` (default) adds the contract tests and the dependency check that `ctest` runs.

Requirements for the OpenCL paths: an OpenCL ICD loader and the OpenCL headers. On macOS the system OpenCL framework is used (OpenCL 1.2, deprecated by Apple, fine for development).

### ggml backend as an out-of-tree plugin

llama.cpp is not modified; it is built once with `-DGGML_BACKEND_DL=ON -DBUILD_SHARED_LIBS=ON` (a static llama.cpp cannot load the plugin):

```bash
cmake -S adapters/ggml -B build-cl11 \
  -DLLAMA_CPP_SOURCE_DIR=<llama.cpp> -DLLAMA_CPP_BUILD_DIR=<llama.cpp>/build-dl
cmake --build build-cl11 -j
GGML_BACKEND_PATH=$PWD/build-cl11/libggml-cl11.so <llama.cpp>/build-dl/bin/test-backend-ops -b CL11_0
```

CL11 was developed against llama.cpp b11384 (commit `0faee50`) and additionally verified against `23b0202` (ggml 0.26.0 / backend API 3). The ggml backend API is not stable; other versions require a matching build and runtime verification (configure warns when `LLAMA_CPP_SOURCE_DIR` differs from the original baseline). See [the tested legacy-CPU flags and current CLI](docs/HETEROGENEOUS_RUNTIME.md). `-DGGML_NATIVE=ON` and `-DGGML_BACKEND_DL=ON` cannot be combined.

### ggml backend in-tree

```bash
adapters/ggml/llama.cpp/install-in-tree.sh <llama.cpp>
cd <llama.cpp>
cmake -B build-cl11 -DGGML_CL11=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-cl11 -j --target llama-cli llama-bench test-backend-ops
```

### The check tools

`tools/mvm_check`, `tools/attn_check` and `tools/mid_check` (MUL_MAT_ID of one token at MoE shapes, ids from an on-device ARGSORT, a 6-layer MoE chain against the CPU: run it on every GPU after a change to the queue handling, the sort or MUL_MAT_ID) are built by hand; the command is in the header comment of each file.

## Priority: use the old hardware to the fullest

The aim of this project is to get everything the old cards can give. Anything that makes them do more is allowed: any API (OpenCL of any version, CUDA through any toolkit or the driver API, Vulkan, HIP / ROCm, RPC / remote processes, the CPU), vendor extensions, a different driver (the 390 / 470 blobs, nouveau with rusticl or NVK, Mesa versions), old toolchains in a chroot or container, and patches to upstream drivers or libraries when they refuse an old card. Experimental paths stay opt-in until they win; a path that loses is kept behind its flag and documented, not deleted.

The only gate is correctness (below). Everything else in the table is measured and recorded with the change, but a number going the wrong way does not block it by itself.

## Checks

| Check | Command | Expected |
| --- | --- | --- |
| Clean build | the plugin with `-Wall -Wextra` | no warnings; the in-tree build must pass too |
| `test-backend-ops`, three tunings | `for g in "" kepler fermi; do GGML_CL11_FORCE_GEN=$g GGML_BACKEND_PATH=<plugin> test-backend-ops test -b CL11_0; done` | every supported case passes. Mac: 3406 / 3406 in each of the three tunings. The number of supported cases depends on the device (GT 430 and GT 730 3441 / 3442, GT 710 and RX 6400 rusticl 3442 / 3442, WX 2100 3435 / 3435), so the criterion is "all supported cases pass", never the count |
| `mvm_check` | per the header comment | 18 / 18 |
| `attn_check` | per the header comment | 8 / 8 |
| `mid_check` | per the header comment | every line ends with OK, `all OK` |
| `ctest` | `ctest --test-dir build` | the contract tests and the dependency check pass |
| Build-option identity | `GGML_CL11_PRINT_BUILD_OPTS=1` and compare | the sources and the option string passed to `clBuildProgram` are byte-for-byte the same as before the change |
| Launch count | `GGML_CL11_LAUNCH_LOG=1` | launches per token do not increase (Qwen3-0.6B Q8_0: 452 on Kepler, 564 on Fermi) |
| TB250 regression | `sweep.sh STAGES="build regress summary" BENCH_REPS=5` against `baseline.csv` | NVIDIA within ±3%, AMD within `tol_pct` (RX 6400: 20%); no `REGRESSION` line in `summary.md` |

Gate: the correctness checks (`test-backend-ops` on the backends a change touches, `mvm_check`, `attn_check`, `mid_check`, `ctest`, and the KL divergence / top-1 agreement of [Correctness](#correctness) for model-level changes). Recorded, not gating: warnings, build-option identity, launch count and the TB250 regression. Run the TB250 measurements with the bench lock held (`flock` on the bench lock), one GPU per run (`GGML_CL11_DEVICES`) unless the run is about several GPUs, after the card is below 85 °C. One configuration of the regression takes 20-50 minutes on the TB250; the `test-backend-ops` round above takes about 4 minutes on a Mac.

## Rules

### ggml stays in `adapters/ggml`

- `src/core`, `src/backends/opencl11`, `include/scrapgpu`, `tests/` and the examples hold no ggml include. `ctest` runs `cmake/check_core_dependencies.cmake`, which fails the build when one appears in `src/core` or `src/backends/opencl11`; Phase 3 extends the check to `tests/` and `examples`.
- Core and backend types must not require a ggml type: no `ggml_tensor`, no `GGML_TYPE_*`, no `GGML_OP_*`, no ggml `op_params` offsets. The adapter translates.
- Dependencies point adapter → core → backend, never the other way.

### Correctness

Use the criteria and tolerances of [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md):

- op level: NMSE against the CPU reference (the idea behind llama.cpp's `test-backend-ops`), with a tolerance per op;
- logits: KL divergence (mean and 99th percentile), top-1 agreement, maximum absolute error;
- quality: the difference in perplexity;
- sanity: no NaN / Inf, no collapse (repeated words);
- greedy token agreement is auxiliary, a fast way to notice a regression.

Generated tokens are not the correctness criterion. The CPU backend quantizes the activations to 8 bit before the dot product, CL11 computes in FP32, so Q4_K models diverge in greedy output after about 20 tokens even when CL11 is closer to the exact product.

Tolerances are calibrated against benign differences: measure the official backends against each other (for example CPU against Vulkan on the same model and text) and require that CL11 stays below that. A tolerance fixed on one model must not be applied to another: the same CL11 result that passes one model's calibrated limit fails a fixed 99th-percentile limit taken from a different model.

`op_supported` returns a reason for what it cannot do. There is no implicit CPU execution and no implicit copy: whether the CPU takes an op is the adapter's (for ggml, the scheduler's) decision.

### Optimizations and old rules as guidance

The OpenCL 1.1-only rule, the ban on per-launch overhead and the list of CL11 optimizations to keep (R32 repacking, Fermi / Kepler / AMD tuning, constant-memory mat-vec, fusions, `MUL_MAT_ID` generation kernels, Mesa `split_programs`, the lazy-allocation probe and the rest) used to be hard rules. They are now guidance: they record what made the cards fast. Replacing one is fine when a measurement on the card shows the replacement is faster or lets the card do something it could not; write the measurement in the commit.

The former rules, kept for reference:

#### OpenCL 1.1 only

- Host: the C API of OpenCL 1.1 only (`CL_TARGET_OPENCL_VERSION=110`). No `cl2.hpp`, no `clEnqueueFillBuffer`, no `clEnqueueMarkerWithWaitList`, no images, no SVM, no sub-buffers outside the backend's own internal use (memset is a kernel).
- Device: OpenCL C 1.1 (`-cl-std=CL1.1`). No `cl_khr_fp16`, no fp16 arithmetic (fp16 is a storage format), no sub-groups, no features of 1.2 or later.
- A device that reports OpenCL 3.0 (rusticl) is used with 1.1 features only.
- Work-group sizes are never hard coded: start from `CL_DEVICE_MAX_WORK_GROUP_SIZE`, reduce and rebuild the program until every kernel accepts them.

#### No per-launch overhead

The GT 430 mat-vec runs at about 80% of the memory bandwidth, so anything added in front of a launch shows up directly in decode. Some GPUs need ~10 µs per launch, with 450-700 launches per token.

- Kernel variant selection happens once, at plan build, and the result (kernel handle, work-group size, fixed argument layout) is stored on the node. No table lookups, no string work and no allocation per launch.
- Argument setting stays a POD builder on the stack; the kernel ABI structs are fixed.
- With profiling off, no event is created and no string is built. No `clFinish` per node; the whole graph is enqueued on one in-order queue.
- Node dispatch adds no virtual call, map lookup, allocation or string generation.

#### Keep the optimizations

Do not replace what CL11 already does with a generic version: the R32 repacking of Q4_0 / Q8_0 / Q4_K / Q6_K (including partial reads and writes and the ban on views), the Fermi / Kepler / AMD tuning, the Fermi denormal products, the constant-memory mat-vec, the workaround for the Kepler constant-memory bug, the 128-bit coalesced loads, the K / V cache kernels, the fusions (`RMS_NORM`+`MUL`, `ADD`+`RMS_NORM`+`MUL`, several `MUL_MAT`, `SET_ROWS` pairs, omitted `CONT`, V cache `CONT`+`MUL_MAT`), the buffer splitting for large tensors, the `MUL_MAT_ID` generation kernels, the Mesa `split_programs`, the AMD `mad()` and the lazy-allocation probe. Do not dequantize whole tensors to make a layout generic.

Do not mark an unsupported op as supported to make a test pass. Report it as unsupported with a reason instead.

### Machine rules

Driver and machine experiments are allowed. Driver switching goes through the TB250 switch script, which must be able to return to the 390 driver from every state; write the plan and the restore procedure before the first run of a new state. Results are recorded per driver state.

## Commits

Conventional Commits, one stage per commit (`feat(core):`, `refactor(opencl11):`, `docs:` are the ones in use). Keep the changes small enough that the build and the tests above can run on each commit, and record in the commit what was verified.

## Reporting hardware results

A hardware report needs the machine and the run, so that someone else can repeat it:

| Item | What to write |
| --- | --- |
| GPU | model, generation (Fermi / Kepler / GCN / RDNA2), compute capability or code name, VRAM, PCIe link speed and width |
| Driver | the OpenCL version the device reports, the driver version, and for AMD the Mesa version and whether rusticl or Clover was used (`RUSTICL_ENABLE=radeonsi`) |
| Host | OS and kernel, CPU and thread count, memory |
| Conditions | temperature when the measurement started, the GPU filter, the bench lock |
| Commands | the exact command line with its options and environment variables, the repeat count, and the mean and standard deviation |
| Models | file name, quantization and SHA256, as in the table of [docs/BASELINES.md](docs/BASELINES.md) |

Numbers come from real runs on the real device. What could not be run on hardware stays marked as unverified, and an estimate is never written in its place.

## Third-party code

Licensing and per-layer recommendations are in [docs/REUSE.md](docs/REUSE.md); the rule of thumb is "take what can be taken, do not write everything ourselves".

| License | Handling |
| --- | --- |
| MIT / BSD-3 / BSL-1.0 / CC0 | Vendoring and porting are both fine as long as the copyright notice stays |
| Apache-2.0 (CLBlast, MNN, OpenCL-CLHPP) | Compatible with this project's GPL-3.0-or-later (not with GPLv2). When vendoring or porting, ship the full Apache-2.0 text, mark the changed files (§4), carry any NOTICE, and list the entry in `THIRD_PARTY_NOTICES.md` |
| GPL-3.0 / GPL-2.0-or-later | May be ported with the notice, since this project is GPL-3.0-or-later. For now, measurement tools such as clpeak stay external processes and only their output (JSON / CSV) is used |
| GPL-2.0-only (no "or later") | Incompatible with GPL-3.0: never take it in, external processes only (check mixbench's exact terms before any use) |
| BabelStream (own license) | External tool only; published results of a modified build must say that it is modified |
| No LICENSE file (Relic) | Do not take it in. The design is a reference and it can be used as a comparison baseline, nothing more |

When porting:

1. Record the source, the commit and the license at the top of the file, or keep the file under `third_party/`, and add an entry to the third-party list at the root of the repository (`THIRD_PARTY_NOTICES.md`).
2. For Apache-2.0 code, add the full license text and note the changes in the files that were modified.
3. GPL measurement tools stay out of the repository for now: document how to run them and how to read their output.
4. A ported kernel gets the NMSE comparison against the CPU reference and a performance comparison against the CL11 kernel it replaces before it is adopted.

## License

GNU General Public License v3.0 or later (GPL-3.0-or-later, [LICENSE](LICENSE)). The parts derived from ggml / llama.cpp (MIT) and their copyright notice are listed in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

By contributing you agree that your contribution is licensed under GPL-3.0-or-later.
