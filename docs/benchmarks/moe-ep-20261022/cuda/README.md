# CUDA backend（U7）と SGLX の `pdev_cuda`（tb250 実測）

測定日 2026-10-22、tb250（ドライバ 390.157、libcuda 9.1 世代の Driver API）、GT 730 / GT 710（Kepler sm_35）、GT 430（Fermi sm_21）。数値はすべて `logs/` の実測。測定は `~/moe-ep/sglx/run.sh`（flock + 温度待ち）経由で 1 本ずつ。

## 実装

| ファイル | 内容 |
| --- | --- |
| `src/backends/cuda/driver.{h,cpp}` | libcuda を dlopen（リンクしない。無ければデバイス 0 個）。CUDA ヘッダ不要 |
| `src/backends/cuda/device.{h,cpp}` | コンテキスト（初回使用時に作成）、ストリーム、pinned メモリ、PTX モジュール、カーネル起動。sm_35 以上は sm_35 版、それ以外は sm_20 版を選ぶ |
| `src/backends/cuda/kernels/kernels.c` | カーネル本体（freestanding C。clang の NVPTX で PTX にする。`gen_ptx.sh`）。生成物 `ptx_sm20.inc` / `ptx_sm35.inc` はコミット済みで、ビルドに clang も nvcc も要らない |
| `src/backends/cuda/r32_transfer.{h,cpp}` | 量子化重みを R32 レイアウト（CL11 と同じ）に詰め直して upload / download |
| `src/backends/cuda/api.cpp` | ScrapGpu C API（`scrapgpu_cuda_register()`、`include/scrapgpu/cuda.h`）。MUL_MAT（Q4_0 / Q6_K）、MUL_MAT_ID（Q4_0）、メタ op |
| `adapters/ggml/composite/pdev_cuda.cpp` | SGLX0 の物理デバイス `CUDA<i>` |
| `CMakeLists.txt` | `SCRAPGPU_BUILD_CUDA`（既定 OFF）。`scrapgpu-device-server --backend cuda` も可（GT 430 を別プロセスで使う用） |
| `tests/cuda/cuda_kernel_tests.cpp`、`tests/ops/ops_tests.cpp`（`SCRAPGPU_TEST_BACKEND=cuda`） | CPU 参照との NMSE 比較 |
| `tools/mm_bench.cpp`、`tools/cuda_isa_bench.cpp` | 帯域測定（CUDA と CL11 を同じコードで）、命令スループット |

カーネル: Q4_0 / Q6_K × F32 の mat-vec と mat-mat（列 1,2,4,8 を一度に）、MUL_MAT_ID（Q4_0、(slot, token) ごとに blockIdx.y）。1 warp が 32 行（lane = 行、128 bit ロード、x は全 lane 共通）。重みのバイト→float は `prmt` で 128+b を作り、オフセットは x の 16 要素ごとの和（共有メモリに先に計算）で引く。Fermi / Kepler は同一ソースで `-march` だけ違う（Kepler も `ld.global.nc` は使わない、下記）。`SCRAPGPU_CUDA_DEVICES="GT 730,GT 710"` で使うカードと順序を指定（pdev の名前は `CUDA0`, `CUDA1`）。`SCRAPGPU_CUDA_PTX=<file>`、`SCRAPGPU_CUDA_NRR` はカーネル実験用。

## 正しさ

| 検査 | 結果 |
| --- | --- |
| `scrapgpu_cuda_kernel_tests`（q4_0 / q6_K、K 3 通り、行 4 通り、列 8 通り + MUL_MAT_ID 14 通り、無効 id を含む。NMSE < 1e-9） | GT 730 / GT 710 / GT 430 とも **161/161**（`logs/cuda_final_ktest_*`）。NMSE は 1e-11 以下 |
| `scrapgpu_ops_tests`（`SCRAPGPU_TEST_BACKEND=cuda`、plan の buffer 参照、tensor_copy、MUL_MAT / MUL_MAT_ID） | 3 枚とも通る（8 件実行、未対応 op は SKIP） |
| NRR（1 lane あたりの行数）1 / 2 / 4 の全てで ktest | GT 730 で 161/161（`cuda_e2b_*`） |
| Fermi と Kepler の同居 | 同一プロセスでは GT 430 の `cuCtxCreate` が 101（MB0 と同じ）。ktest は SKIP と表示して続行（`cuda_ktest_all`、初版）。Kepler 2 枚は同居可 |
| `test-backend-ops -b SGLX0 -o MUL_MAT,GET_ROWS`（統合版プラグイン） | `CUDA0:1`、`CUDA0:0.7,CUDA1:0.3` は **361/361**。`Vulkan0:0.5,Vulkan1:0.2,CUDA0:0.2,CUDA1:0.1` は 360/361（失敗は既知の Vulkan0 の `q4_0 m=40 n=9 k=32` の NaN、前回の CL11 構成と同じ。`logs/cuda_tbo_*`） |

## 帯域（GT 730、Kepler、q6_K 2048 x 128000 = 205.1 MiB の mat-vec N=1、`tools/mm_bench.cpp`）

| 実装 | GB/s | ms | 備考 |
| --- | --- | --- | --- |
| **CL11（既存）** | **28.2** | 7.63 | `cuda_final_cl11_GT730_mv_q6K` |
| **CUDA（最終）** | **26.4** | 8.15 | 94 %。`cuda_final_cuda_GT730_mv_q6K` |
| CUDA v1（自然レイアウト、行ごとの warp） | 9.1 | 23.7 | `v1_natural_*` |
| CUDA v2（R32、NRR 1、ld.global.nc） | 20.6 | 10.5 | `v2_r32_*` |

全カード・全形状（最終版、`logs/cuda_final_*`、throughput = 非同期に 20 回投入して 1 回同期。GB/s は重みの読み出し量）:

| カード | 形状 | CL11 | CUDA |
| --- | --- | --- | --- |
| GT 730 | q6_K mv N=1 | 28.2 | 26.4 |
| GT 730 | q4_0 mv N=1（72 MiB） | 27.9 | **32.7** |
| GT 730 | q6_K N=8 の時間 | 128.2 ms | **81.7 ms** |
| GT 730 | MUL_MAT_ID 1 token（experts 32、1792 x 2048、4 使用、7.9 MiB） | **0.313 ms** | 0.476 ms |
| GT 730 | MUL_MAT_ID 32 tokens | 54.6 ms | **14.0 ms** |
| GT 710 | q6_K mv N=1 | **10.2** | 9.1 |
| GT 710 | q4_0 mv N=1 | 10.7 | 10.7 |
| GT 710 | MUL_MAT_ID 1 token | **0.802 ms** | 1.199 ms |
| GT 710 | MUL_MAT_ID 32 tokens | 109.7 ms | **35.8 ms** |
| GT 430 | q6_K mv N=1 | **14.5** | 11.0 |
| GT 430 | q4_0 mv N=1 | **18.5** | 14.0 |
| GT 430 | MUL_MAT_ID 1 token | **0.794 ms** | 0.861 ms |
| GT 430 | MUL_MAT_ID 32 tokens | 258 ms | **27.6 ms** |

要点:
- Kepler の q6_K N=1 は CL11 の 94 %、q4_0 は CL11 の 117 %。複数列（prefill 形）と複数 token の MUL_MAT_ID は CUDA の方が 1.5〜9 倍速い。1 token の MUL_MAT_ID と Fermi の N=1 は CL11 の方が速い（CL11 が Fermi 専用の denormal 方式などを持つ）。
- 往復（8 KiB の書き込み → カーネル → 読み出し）は MB1 の 60 µs 台のまま（この README では再測定していない）。
- 試して外したもの（GT 730、`cuda_e1_*` `cuda_e2_*` `cuda_e3_*`）: `ld.global.nc` は plain load より遅い（q6_K で 20.4 対 26.4、q4_0 で 14.3 対 19.7）。1 lane あたり 4 行（NRR 4）は q4_0 で 25.4→32.7 になるが q6_K はレジスタ不足で 13〜16 に落ちる（q6_K は NRR 1、q4_0 は十分な warp 数があるとき NRR 4）。
- 命令スループット（`cuda_isa_*`）: GT 730 は prmt 2.3、and 3.4、fma 4.3 warp 命令/clk/SM。GT 430 は prmt 1.0、fma 1.2、shr 0.5。Kepler では x のロード（128 bit のブロードキャスト）が律速で、ALU ではない。

## SGLX C を CL11 から CUDA に替えた構成（`perf pin`、C1 → CL11 → CUDA を交互に 2 回、REPS=10、PROFILE=0、`-ub 512 -ts 5/1`）

`GGML_SGLX_SPLIT=Vulkan0:0.5,Vulkan1:0.2,CUDA0:0.2,CUDA1:0.1`、`SCRAPGPU_CUDA_DEVICES="GT 730,GT 710"`、プラグインは統合パッチ適用版（`CL11_WITH_CUDA=ON`）。`scripts/sglxcuda.sh`、`logs/cuda_sglx_*`。終了時に perf は auto に戻した（`cuda_sglx.perf.log`）。

| 構成 | pp512 | tg128 | pg512,128 |
| --- | --- | --- | --- |
| C1 #1 / #2 | 662.1 / 662.6 | 51.58 / 51.84 | 194.95 / 195.34 |
| SGLX C（CL11）#1 / #2 | 668.0 / 669.0 | 61.94 / 62.28 | 222.8 / 223.5 |
| **SGLX C（CUDA）#1 / #2** | 669.2 / 670.4 | 61.98 / 61.99 | 222.8 / 223.4 |

CUDA 版は CL11 版と同等（差は標準偏差内）。C1 に対して tg +20 %、pg +14 %。出力射影の区間は短く、GT 730 / GT 710 の担当分が RX / WX より先に終わるため、この構成では CUDA の mat-vec の差が見えない。

## 残課題

- Fermi（GT 430）の mat-vec は CL11 の 75〜76 %。CL11 の Fermi 用 denormal 方式（x を 2^k で前処理）は未移植。
- 1 token の MUL_MAT_ID は CL11 の 1.5 倍遅い（R=1 の `mid4`、1792 行 = 56 グループ x 4 slot で warp 数が足りない）。ゲート・アップ・SwiGLU・ダウンの融合カーネルは未実装（設計書の `moe_gate_up_swiglu_q4_0_mv` など）。
- PTX が 576 KB（x2）ある。JIT は初回のみ（ドライバのキャッシュ）。不要な列数 / NRR の組を削ると小さくなる。Windows（MSVC）の文字列リテラル上限は考慮していない。
- Q4_0 / Q6_K 以外（Q8_0、Q4_K）と MUL_MAT_ID の Q6_K は未対応。
- 統合: `integration.patch`（`composite.h`、`composite_device.cpp`、`adapters/ggml/CMakeLists.txt`）。親が適用する。
