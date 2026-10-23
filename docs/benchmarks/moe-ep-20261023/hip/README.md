# 段階5: tb250 の AMD GPU で HIP / ROCm（公式 Vulkan との比較）

TB250、2026-10-23。LFM2.5-8B-A1B Q4_0、llama.cpp `23b0202`（他の測定と同じソース）、ROCm 7.13.0。作業場所 `~/moe-ep/hip/`（ROCm 本体は `/mnt/hdd/rocm/`）。測定は `~/moe-ep/sglx/run.sh`（flock + 温度待ち）で 1 本ずつ。数値は `logs/` の実測。**【推定】** と書いたものだけが実測でない。perf は触っていない（`logs/perf_status_*.txt` に記録だけ。最初の測定は `auto`、最後の対照 `ctl_*` は他担当の pin（`manual`）の間に測った）。

## 結論

1. **RX 6400（gfx1034）は ROCm 7.13.0（TheRock の安定版 wheel）で `HSA_OVERRIDE_GFX_VERSION` なしにそのまま動く**。`rocminfo` が `gfx1034`（`amdgcn-amd-amdhsa--gfx1034`）を出し、HIP カーネル、llama.cpp の HIP ビルド（`-DGGML_HIP=ON`、`gfx1034`）、`test-backend-ops` 3591/3591 が通る。CPU に AVX が無くても動く。
2. **WX 2100（Polaris12、gfx803）は ROCm から見えない**。KFD のトポロジーに CPU と RX 6400 の 2 ノードしか無い。原因は **【推定】** PCIe atomics が無い（WX 2100 は PCH のルートポート 00:1c.6 の下、PCIe 2.0 x1）ため amdkfd が Polaris を外している。`dmesg` と特権付き `lspci` が読めず確認できていない。KFD はカーネル側なので古い ROCm（5.x）に替えても変わらない見込み（**【推定】**、試していない）。WX 2100 は Vulkan のまま使う。
3. **速度: 全層が GPU に載る構成では HIP は公式 Vulkan と同等**（RX を HIP、WX を Vulkan にした層分割 5/1: tg 51.3 対 51.8、pg 193.9 対 195.2、pp 697 対 666）。**勝ってはいない**。RX 単体（`-ncmoe 7`、expert の一部が CPU）は tg が同等（37.2 対 37.6〜38.8）だが、**pp は HIP が 5.7 倍遅い**（82 対 466）。HIP は 4 GB に収めるのに `-ncmoe 7` 以上が要る（Vulkan は 5 で載る）。
4. **正しさ: HIP は KLD の基準を満たさない**。KLD 0.0429 / top-1 88.9 %（基準 ≤ 0.0259 / ≥ 91.94 %）。同じ条件の Vulkan は 0.0216 で、約 2 倍。原因は未特定（下の「正しさ」）。`test-backend-ops` の MUL_MAT / MUL_MAT_ID / GET_ROWS は全部通るので、演算単体の誤りではない。
5. SGLX の pdev としては **速くなる根拠が無い**（MUL_MAT_ID は n=1 で HIP が 3.4〜4.9 倍遅い）。組み込み自体は `pdev_ggml` で名前 `ROCm0` を指定するだけで済む見込み（下に案）。

## ROCm の入手

| 項目 | 内容 |
| --- | --- |
| 方法 | pip wheel（TheRock）を `/mnt/hdd/rocm/` の venv に入れる（sudo・apt 不要）。`rocm-sdk-core`、`rocm-sdk-libraries-gfx103x-all`、`rocm-sdk-devel` |
| 動いた版 | **7.13.0（安定版）**。取得元 `https://repo.amd.com/rocm/whl/gfx103X-all/`、`pip download --no-deps`。sha256 は `logs/wheels713.sha256`。サイズ: core 0.4 GB、libraries 1.0 GB、devel 1.85 GB |
| 動かなかった版 | `7.14.0a20260612`（ナイトリー、`https://rocm.nightlies.amd.com/v2/gfx103X-all/`、sha256 は `logs/wheels.sha256`）。`rocminfo` が `rocr::AMD::GpuAgent::InitDma()` 内の `call *%rax`（`std::function` の古い manager が不正値 `0x100000001`）で SIGSEGV。`HSA_OVERRIDE_GFX_VERSION=10.3.0`、`HSA_ENABLE_SDMA=0` でも同じ。安定版に替えて解決。`rocgdb`（wheel 同梱）で特定 |
| 展開 | `rocm` メタパッケージ（sdist）は `--no-index` で build できなかったので入れず、`rocm_sdk_devel/_devel.tar` を site-packages に手で展開（9.1 GB、4.5 分）。`_rocm_sdk_devel/` が `ROCM_PATH`（core・libraries へのシンボリックリンク入り）。`scripts/env.sh` |
| ディスク | `/mnt/hdd/rocm/` 約 42 GB（ナイトリーの venv と wheel も残してある。不要なら `venv/`、`wheels/` を消せる） |
| `/dev/kfd` | `render` グループで足りた |
| RX の PCIe | host↔device の `hipMemcpy` は pageable / pinned / registered のどれも約 1.7〜1.8 GB/s（`logs/hip_bw.out`、`scripts/bw.hip`） |

`logs/rocminfo_7.13.0.log` に `rocminfo`。簡単な HIP カーネル `scripts/vadd.hip`: 結果は正しく、vadd 108 GB/s（`logs/hip_vadd.out`）。

## llama.cpp の HIP ビルド

`scripts/build-hip.sh`。`-DGGML_HIP=ON -DGPU_TARGETS=gfx1034 -DGGML_CUDA_FA=OFF`、CPU 側は既存ビルドと同じフラグ（AVX 等 OFF）、`GGML_BACKEND_DL=ON` で ggml-hip / ggml-vulkan / ggml-cpu を同じ `bin/` に置く（RX を HIP、WX を Vulkan にするため）。gfx1030 では**なく gfx1034 を直接**指定できた（コンパイラ・ランタイムとも gfx1034 を持つ）。ビルドは約 50 分（`-j2`、`nice`）。ビルドは 2 つ: `llama-build`（既定）、`llama-build-mmq`（`-DGGML_CUDA_FORCE_MMQ=ON`）。

注意: `GGML_CUDA_FORCE_MMQ` / `GGML_CUDA_FORCE_CUBLAS` は**環境変数ではなくビルド時の定義**。最初に環境変数で試した `hip_rx_nc7_ub512_mmq`、`_cublas`、`hip_kld_rocm_nc7_mmq`、`_cublas` は何も変えておらず、既定の再測定と同じ（KLD が 6 桁まで一致）。FORCE_MMQ の本当の効果は `mmq_*`（`llama-build-mmq`）。

## 速度

`llama-bench -fa 0 -t 2 -b/-ub N -r R -p 512 -n 128 -pg 512,128`（`scripts/hbench.sh`）。`-ncmoe` は expert を CPU に置く層数。表は `scripts/sumrz.py` の出力（平均 ± 標準偏差、t/s）。

### RX 6400 単体（`-ub 512`）

| 構成 | ncmoe | pp512 | tg128 | pg512,128 | 備考 |
| --- | --- | --- | --- | --- | --- |
| HIP | 5 / 6 | - | - | - | **メモリ不足**（`cudaMalloc` 269 MiB 失敗、`hip_rx_nc5_ub512_try`、`hip_rx_nc6_ub512`） |
| HIP | 7 | 81.44 ± 0.53 / 82.03 ± 0.90 | 37.07 ± 1.14 / 37.84 ± 0.58 | 65.83 ± 0.46 | `hip_rx_nc7_ub512_def2` / `_def3`（`_mmq`、`_cublas` も同じ範囲） |
| HIP | 7 | 77.86 ± 1.60 | 25.75 ± 0.41 | 55.55 ± 0.61 | `hip_rx_nc7_ub512`（**最初の run。他の再測定は 37 前後。冷えた状態の外れ値**、原因は確認していない） |
| HIP | 8 | 66.16 ± 0.30 | 22.84 ± 0.70 | 47.88 ± 0.23 | `hip_rx_nc8_ub512`（最初の run の直後。再測定なし） |
| HIP（FORCE_MMQ ビルド） | 7 | 82.15 ± 0.30 | 37.90 ± 1.07 | 65.05 ± 0.50 | `mmq_rx_nc7_ub512`（既定ビルドと差なし） |
| HIP（対照、perf pin 中） | 7 | 81.35 ± 0.60 | 36.87 ± 0.59 | 65.25 ± 0.53 | `ctl_hip_rx_nc7_ub512` |
| Vulkan（同じビルド） | 5 | 95.49 ± 0.02 | 33.83 ± 4.00 | 72.27 ± 0.39 | `vk_rx_nc5_ub512` |
| Vulkan | 6 | 101.57 ± 0.34 | 42.17 ± 0.55 | 72.35 ± 0.40 | |
| Vulkan | 7 | 466.64 ± 2.27 / 466.17 ± 2.77 | 37.50 ± 0.33 / 38.81 ± 0.24 | 141.75 ± 0.73 / 142.56 ± 0.55 | `vk_rx_nc7_ub512`、`_c` |
| Vulkan | 7（前のビルド） | 465.65 ± 1.46 | 37.56 ± 0.99 | 141.02 ± 0.89 | `vkold_rx_nc7_ub512`（対照: 新旧ビルドで同じ） |
| Vulkan（対照、perf pin 中） | 7 | 457.51 ± 16.24 | 38.16 ± 0.32 | 145.41 ± 1.08 | `ctl_vk_rx_nc7_ub512` |
| Vulkan | 8 | 411.61 ± 9.68 / 417.09 ± 1.61 | 33.75 ± 0.39 / 33.99 ± 0.10 | 127.08 ± 1.26 / 127.49 ± 0.86 | `vk_rx_nc8_ub512`、`_b` |

`-ub 128`、ncmoe 7: HIP 23.34 ± 0.47 / 36.50 ± 0.50 / 24.92 ± 0.46（`hip_rx_nc7_ub128`）、FORCE_MMQ ビルド 23.15 / 37.24 / 25.33、Vulkan 158.27 / 37.77 / 96.04（`vk_rx_nc7_ub128`）。

- **tg は HIP と Vulkan が同等**（37 前後）。**pp は HIP が 5.7 倍遅い**（ub 512）。ub 128 でも 6.8 倍。
- pp が遅い原因は**未解明**。HIP の host→device コピーは 1.7 GB/s で、Vulkan の pp（ub 512 で約 1.1 s / ub）が示す転送量と同程度なので、コピー帯域そのものではない。HIP の pp の ub あたり時間は ub 128（約 5.5 s）と ub 512（約 6.3 s）でほぼ同じ（**【推定】** ub あたりの固定費が大きい。例えば CPU 側 expert の層ごとの処理）。FORCE_MMQ でも変わらない。追跡は残課題。
- HIP は Vulkan より約 300 MB 多く VRAM を使う（ncmoe 5・6 が載らない）。

### RX（HIP）+ WX（Vulkan）の層分割 `-ts 5/1 -ncmoe 0 -ub 512`（全層が GPU）

| 構成 | pp512 | tg128 | pg512,128 | tag |
| --- | --- | --- | --- | --- |
| **HIP + Vulkan**（RX = ROCm0、WX = Vulkan1） | 697.35 ± 11.64 | 51.26 ± 0.32 | 193.94 ± 0.42 | `hipmix_nc0_ub512_b`（r5） |
| 同、1 回目 | 565.30 ± 231.55 | 51.66 ± 0.21 | 193.93 ± 0.57 | `hipmix_nc0_ub512`（r3。最初の rep が 297.9、他は 701.6 / 696.4: 冷えた最初の rep） |
| 同、FORCE_MMQ ビルド | 702.61 ± 14.04 | 51.64 ± 0.34 | 195.42 ± 1.45 | `mmq_mix_nc0_ub512`（r5） |
| 同、対照（pin 中） | 690.77 ± 23.65 | 51.27 ± 0.06 | 195.17 ± 0.85 | `ctl_hipmix_nc0_ub512` |
| C1（Vulkan0/Vulkan1、同じビルド） | 665.51 ± 4.02 / 668.18 ± 4.00 | 51.76 ± 0.04 / 51.48 ± 0.06 | 195.19 ± 0.32 / 195.13 ± 0.32 | `vkmix_nc0_ub512_b`、`vkmix_nc0_ub512` |
| C1（前のビルド） | 666.04 ± 4.98 | 51.57 ± 0.11 | 194.65 ± 0.34 | `vkoldmix_nc0_ub512`（他の測定の C1 51.7 / 197 / 695 と整合） |
| C1（対照、pin 中） | 664.30 ± 3.67 | 51.61 ± 0.14 | 194.94 ± 0.41 | `ctl_vkmix_nc0_ub512` |

**HIP 混在は C1 に対して pp +4〜5 %、tg −1〜0 %、pg −0.6〜0 %**（標準偏差と同じ桁で、有意なのは pp だけ）。

### MUL_MAT_ID の性能（`test-backend-ops perf -o MUL_MAT_ID`）

両方とも 1500 s の timeout で打ち切り（HIP 37 ケース、Vulkan 36 ケース、共通 36）。`scripts/mmid_cmp.py`。n_mats=128、n_used=8、m=768、k=2048、µs/run（HIP / Vulkan）:

| n（トークン数） | q4_0 | q8_0 | q4_K | iq2_xs | f16 |
| --- | --- | --- | --- | --- | --- |
| 1 | **181.2 / 42.4** | 189.8 / 56.2 | 199.0 / 40.5 | 196.6 / 48.4 | 254.7 / 204.9 |
| 4 | 238.0 / 218.4 | 445.7 / 419.8 | 234.2 / 219.3 | 220.2 / 184.3 | 993.5 / 779.7 |
| 8 | 434.9 / 437.1 | 836.8 / 832.1 | 813.5 / 433.4 | 931.5 / 377.8 | 1958 / 1607 |
| 32 | 2403.7 / 2832.1 | 3026.7 / 3713.2 | 3294.8 / 3353.0 | 2991.5 / 6396.7 | 9975 / 5045 |

- **n=1（tg の形）は HIP が量子化型で 3.4〜4.9 倍遅い**（約 180〜200 µs でほぼ型に依らない。**【推定】** 1 回あたりの固定費）。f32 は同じ（403.9 / 402.8）、f16 は 1.24 倍。
- n=4〜8 はほぼ同じ（q4_0 / q8_0）。n≥32 は q4_0 / q8_0 / q6_K で HIP が 15〜20 % 速い。f16 は HIP が 1.2〜2 倍遅い。
- それでも実機の tg が同等なのは、tg では層あたり MUL_MAT_ID 以外（アテンション、通信、スケジューリング）が支配的で、n=1 の MUL_MAT_ID 差が隠れるため **【推定】**（層ごとの区間は測っていない）。

## 正しさ

| 検査 | 結果 |
| --- | --- |
| `test-backend-ops test -b ROCm0 -o MUL_MAT,MUL_MAT_ID,GET_ROWS`（既定ビルド） | **3591/3591**（`hip_tbo_rocm`） |
| 同 `-o MUL_MAT_ID`（FORCE_MMQ ビルド） | **60 ケース OK、失敗 0**。ただし timeout 100 s で打ち切り（途中、`mmq_tbo_mmid2`）。最初の run `mmq_tbo_mmid` は `VK_ICD_FILENAMES` の設定忘れ（NVIDIA ICD）で 1 秒で落ちた（記録だけ残す） |
| KLD、HIP 単体 `-ncmoe 7`（`-ub 128`） | **0.042921 ± 0.002656 / top-1 88.922 %**（`hip_kld_rocm_nc7`。FORCE_MMQ ビルドの `mmq_kld_rocm_nc7` も同じ値） |
| KLD、Vulkan 単体 `-ncmoe 7`（同じビルド） | 0.021625 ± 0.001480（`hip_kld_vk_nc7`） |
| KLD、HIP + Vulkan（RX = ROCm0、WX = Vulkan1）`-ts 4/1` | **0.041622 ± 0.002591 / top-1 89.510 %**（`hip_kld_mix_nc0_ts41`、FORCE_MMQ ビルドも同じ） |
| KLD、C1（Vulkan0/Vulkan1）`-ts 5/1` | 0.021464 ± 0.001467 / top-1 92.843 %（`hip_kld_vkmix_nc0`） |
| KLD、HIP + Vulkan `-ts 5/1 -ub 128` | **abort**（2 回、`hip_kld_mix_nc0`、`_retry`）: `hipblasGemmBatchedEx` が `CUBLAS_STATUS_INTERNAL_ERROR`（`ggml_cuda_mul_mat_cublas_impl`、ggml-cuda.cu:1612）。`-ub 512` の速度測定（同じ 5/1）は通る。RX の VRAM がほぼ満杯のため hipBLAS の作業領域が取れない可能性 **【推定】**（`-ts 4/1` では通る） |

基準（SGLX の他の測定）は KLD ≤ 0.0259 / top-1 ≥ 91.94 %。**HIP は 2 つの構成とも満たさない**。

- 演算単体のテスト（MUL_MAT / MUL_MAT_ID / GET_ROWS）は通るのに KLD が 2 倍なので、誤差の出所は別の演算か、複数演算の累積。FORCE_MMQ で KLD が 6 桁まで同じ → MoE の mmq 対 dequant + hipBLAS の違いは原因ではない。
- **【推定】** アテンション等の fp16 バッチ行列積（`CUBLAS_COMPUTE_16F`、fp16 累積。ggml-cuda.cu:1397 付近にその定義がある）。未検証。確かめるには該当演算を fp32 にするか、演算ごとに CPU と比較する必要がある。

## SGLX への組み込み案（実装なし、composite のファイルは触っていない）

- `composite_device.cpp` は `GGML_SGLX_SPLIT` の名前を `ggml_backend_dev_by_name` で引き、`make_pdev_ggml(dev)` で包む。ggml-hip の装置名は `ROCm0` なので、**`GGML_SGLX_SPLIT=ROCm0:0.78,...` で `pdev_ggml` がそのまま包める見込み**（`MUL_MAT` を `supports_op` で確認する経路も同じ）。ggml-hip を `GGML_BACKEND_PATH` の同じディレクトリに置く（`LD_LIBRARY_PATH` に ROCm の `lib/` が要る）。**動かしていない**ので、以下は要確認:
  - `pdev_ggml` の `threaded_` は `Vulkan1` 以降だけ既定 ON（名前判定）。ROCm0 は既定 OFF になるので、必要なら `GGML_SGLX_THREADS=1`。
  - 同じ RX を llama.cpp のレイヤー用（HIP か Vulkan0）と pdev で共有する形（現在の Vulkan0 共有と同じ）を HIP で行うと、`ggml_backend_dev_init` が別の HIP コンテキスト / ストリームを作る。同じ GPU を Vulkan と HIP から同時に使う動作は未確認。
  - ホストピン留めメモリは `ggml_backend_dev_host_buffer_type`（`cudaMallocHost`）が使える。
- **速度の見込みは薄い**: 出力射影は `n=1` の mat-vec で、HIP の n=1 MUL_MAT_ID は Vulkan の 1/4 の速さ（上の表）。mat-vec（MUL_MAT）単体は測っていない。全層 GPU の tg は同等だったので、HIP を pdev にしても Vulkan0 より速くなる根拠は無い。**入れる価値があるとすれば、Vulkan が苦手な構成（pp や n≥32 の MUL_MAT_ID が HIP で 15〜20 % 速い）**だが、SGLX の対象は tg の n=1。
- KLD が基準を満たさないので、原因を特定するまで SGLX の合否試験（正しさ）に通らない。

## 残課題

- KLD が 2 倍になる演算の特定（演算ごとの CPU 比較、fp16 累積の仮説の検証）。
- 単体 `-ncmoe 7` の pp が Vulkan の 1/5.7 になる原因（ub あたりの固定費）。
- HIP で `-ncmoe 5 / 6` が載らない VRAM 差の内訳。
- HIP の MUL_MAT_ID perf は 36 ケースで打ち切り。MUL_MAT（mat-vec）の perf と、n=512 級の MUL_MAT_ID は未測定。
- FORCE_MMQ ビルドの MUL_MAT_ID 正しさテストは途中まで（60 ケース）。
- WX 2100 の ROCm: KFD に載らない原因（PCIe atomics か）の確認には root（`dmesg`、`lspci -vv`）が要る。
- 他担当の測定と同じ CPU を `-j2` のビルド（約 50 分 × 2）が使った。ビルド中の GPU 測定は `q10`/`q11` までは行っていない（ビルドは測定の前後）が、q9 の KLD 2 本はビルド中に走った（KLD 値は影響を受けない）。

## ファイル

| ファイル | 内容 |
| --- | --- |
| `scripts/env.sh`、`build-hip.sh` | ROCm のパス、llama.cpp の HIP ビルド |
| `scripts/hbench.sh`、`hkld.sh` | `llama-bench` / KLD（`run.sh` 経由）。`MODE` = `hip` `vk` `vkold` `mix` `vkmix` `vkoldmix` |
| `scripts/q1.sh`〜`q11.sh` | 測定の順序（q4 はビルド定義の空振り KLD、q6 は帯域、q8 は FORCE_MMQ ビルド） |
| `scripts/sumrz.py`、`mmid_cmp.py` | 表の生成 |
| `scripts/vadd.hip`、`bw.hip`、`segv.c` | HIP の簡単なカーネル、コピー帯域、SIGSEGV の調査用 `LD_PRELOAD` |
| `logs/` | 全 run の `.log` / `.out`、`rocminfo_7.13.0.log`、wheel の sha256、perf の記録 |
