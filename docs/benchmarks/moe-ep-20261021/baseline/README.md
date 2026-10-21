# MoE expert 並列 段階2: LFM2.5-8B-A1B Q4_0 の基準値（tb250）

測定日 2026-10-21。SGL（ScrapGpuLibrary）worktree `feat/moe-expert-parallel`（ベースコミット `a323e7dd1a06f8b7a49d8cfd54fd56afd4f31da2` の上にあるコミット前の作業ツリー。コードの変更はしていない）。実装変更なし、ビルドと測定のみ。すべて実測で、推定値は「推定」と明記した。

## 結論

指標は pp512 / tg128 / `-pg 512,128`（総合）。値は平均 ± 標準偏差（t/s、`-r 5`）。

| 区分 | 構成 | pp512 | tg128 | pp512+tg128 |
| --- | --- | --- | --- | --- |
| CPU のみ | Celeron G3930 2 スレッド | 12.14 ± 0.01 | 6.14 ± 0.01 | 10.04 ± 0.01 |
| 単体 pp 最速 | RX 6400 公式 Vulkan、`-ncmoe 0 -ub 512`（全 expert が GPU） | **117.99 ± 0.35** | 12.04 ± 0.00 | 40.25 ± 0.03 |
| 単体 tg 最速（C0） | RX 6400 公式 Vulkan、`-ncmoe 5 -ub 128` | 71.84 ± 0.19 | **26.98 ± 1.31** | 49.06 ± 2.08 |
| 単体 総合最速 | RX 6400 公式 Vulkan、`-ncmoe 5 -ub 512` | 94.37 ± 0.23 | 25.68 ± 1.33 | **64.00 ± 2.48** |
| 複数 最速（C1） | RX 6400 + WX 2100 公式 Vulkan 層分割、`-ts 5/1 -ub 512` | **631.75 ± 5.24** | **44.12 ± 0.44** | **170.55 ± 1.32** |
| 複数 5 枚（docs の条件） | CL11 5 枚、GT 430 自動 RPC、`-ts 8,8,18,40,18`、`-ub 128` | 33.27 ± 0.19 | 11.70 ± 0.10 | 24.07 ± 0.14 |
| 複数 5 枚（pp 寄りの分割） | 同上、`-ts 2,4,16,60,18` | 48.78 ± 0.35 | 11.33 ± 0.05 | 29.17 ± 0.10 |

- 単体の最速は tg・総合とも RX 6400 の公式 Vulkan。CL11 の NVIDIA（GT 730 tg 12.15、GT 430 10.26、GT 710 8.65）は expert を CPU に置く構成で、CPU の experts 計算が上限になる（下のプロファイル）。
- C1（AMD 2 枚の公式 Vulkan 層分割）は単体最速に対して pp512 は 5.4 倍、tg128 は 1.6 倍（C0 tg 最速比）、総合は 2.7 倍。**今回の比較で最も強い相手は C1**。C1 は `-ts` を 5/1〜13/3、`-ub` を 64〜1024 で振り、`-ub 512` 以上で pp が頭打ち（下表）。
- 5 枚の CL11 は C1 の tg の約 1/4。NVIDIA を加えた混在（mix4/mix5）も C1 より遅い。層分割では各 GPU の時間が足されるため、帯域の小さい NVIDIA を入れるほど遅くなる（docs/MULTI_GPU.md と同じ傾向）。
- KLD（CPU の logits が基準）は全構成で平均 0.020〜0.024、標準誤差 0.0013〜0.0020 で、構成間の差は標準誤差の範囲内。top-1 一致率は 92.3〜93.6 %。docs/CL11.md の CPU 同士（`-fa on/off`）の雑音床 0.0185 に近く、この MoE モデルのルーターの near-tie が支配的。

## 環境

| 項目 | 内容 |
| --- | --- |
| ホスト | tb250（BIOSTAR TB250-BTC PRO）、Debian 13、kernel 6.12.107+deb13-amd64、Celeron G3930（2 コア 2 スレッド、AVX なし）、RAM 31 GiB |
| NVIDIA | ドライバ 390.157（OpenCL のみ。Vulkan は 390 でクラッシュするため `VK_ICD_FILENAMES=radeon_icd.json` で AMD だけに限定） |
| AMD | Mesa 25.0.7-2+deb13u1（RADV Vulkan、rusticl は `RUSTICL_ENABLE=radeonsi`、Clover は WX 2100 のみ） |
| GPU | GT 430（Fermi GF108、sm_21、964 MiB、OpenCL 1.1）、GT 730（Kepler GK208B、sm_35、981 MiB、OpenCL 1.2）、GT 710（Kepler GK208B、sm_35、2002 MiB、OpenCL 1.2）、RX 6400（RDNA2 Navi24、4080 MiB）、Radeon Pro WX 2100（Polaris 12 / Lexa、2048 MiB） |
| PCIe | 負荷時のリンクは未取得。アイドル時の sysfs は省電力で下がるため参考外。docs/HARDWARE.md の値（RX 6400 Gen2 x4、他は Gen2 x1）に従う。sysfs の `max_link_speed` / `max_link_width`: RX 6400 16GT/s x16（カード内部ブリッジ）、WX 2100 8GT/s x8、GT 730 8GT/s x8、GT 710 5GT/s x8、GT 430 2.5GT/s x16 |
| 温度 | 測定開始時 NVIDIA 31〜51 ℃、AMD 33〜64 ℃（各 run のログ先頭に記録）。すべて 85 ℃未満 |
| 排他 | すべての測定は `flock ~/bench/gpu-bench.lock` を 1 run ごとに取得。別エージェント（マイクロベンチ・投機的デコード）も同じロックを使うので run が混ざらないだけで、待ち時間は含まれない |
| モデル | `LFM2.5-8B-A1B-Q4_0.gguf`（4,844,678,368 B、SHA256 `48ed1465d761311b2fd57b7fb46cf969a20b3a8281945b04e10f52bd1609e715`、docs/BASELINES.md と一致）。32 expert、4 使用 |
| llama.cpp | `23b0202a189c44a54625aadcb37a946dd1d6278d`（ggml 0.26.0、build 11552） |
| llama.cpp ビルド | `cmake -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_BACKEND_DL=ON -DBUILD_SHARED_LIBS=ON -DGGML_NATIVE=OFF -DGGML_AVX=OFF -DGGML_AVX2=OFF -DGGML_AVX512=OFF -DGGML_FMA=OFF -DGGML_F16C=OFF -DGGML_BMI2=OFF -DGGML_BLAS=OFF -DGGML_VULKAN=ON -DGGML_RPC=ON -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_SERVER=OFF -DLLAMA_CURL=OFF -DLLAMA_OPENSSL=OFF`。Vulkan ヘッダ（v1.4.309）と SPIRV-Headers は tb250 のユーザー領域に取得（`libvulkan-dev` が無く sudo 不使用）、`-DVulkan_LIBRARY=/usr/lib/x86_64-linux-gnu/libvulkan.so.1 -DVulkan_INCLUDE_DIR=… -DCMAKE_PREFIX_PATH=… -DCMAKE_CXX_FLAGS=-I…/include`。ターゲット: ggml-cpu ggml-vulkan ggml-rpc ggml-rpc-server llama-bench llama-completion llama-perplexity test-backend-ops。設定ログ `logs/llama-cfg.log` |
| SGL プラグイン | `adapters/ggml`（Mac の worktree を tar で転送）を `-DLLAMA_CPP_SOURCE_DIR=… -DLLAMA_CPP_BUILD_DIR=…` で `-DCMAKE_BUILD_TYPE=Release` ビルド。警告 0。`libggml-cl11.so` SHA256 先頭 `c5e01a0d120fd420`。`logs/plugin-cfg.log` |
| 作業場所 | tb250 `~/moe-ep/`（モデルはロック対象外の `~/bench/models` からコピー） |

## 共通の測定方法

- `llama-bench -m <モデル> -dev <デバイス> -ngl 99 -fa 0 -t 2 -b <ub> -ub <ub> -r 5 -o jsonl`、最終測定は `-p 512 -n 128 -pg 512,128`。
- 掃引（n-cpu-moe、`-ts`、`-ub` の探索）は `-r 2 -p 128 -n 64`。参考値として pp128 / tg64 を残した。
- 単体: `-ncmoe N`（`--n-cpu-moe`、先頭 N 層の expert を CPU）。複数: `-ts a/b/...`（llama-bench では `/` 区切り、`,` は別々の計測になるので注意。最初の数 run は `,` で誤って測ったため `logs/runs-bad/` に退避して捨てた）、`-sm layer`。
- 環境変数・デバイス選択は `scripts/env.sh`（`use_env` / `use_mg`）。実行は `scripts/run.sh`（ロック・温度確認・ログ）。各 run の完全なコマンドと環境変数は `logs/<tag>.log` の先頭 4 行に記録した。
- 各 run の出力: `logs/<tag>.out`（jsonl、`samples_ts` に全サンプル）と `logs/<tag>.log`（stderr、`# cmd:` 行、`# env:` 行、温度、`rc`、所要秒）。
- 構成別の環境（`use_env` / `use_mg`）:

| 名前 | 環境変数 | `-dev` |
| --- | --- | --- |
| cpu | （なし） | `none -ngl 0` |
| gt430 / gt730 / gt710 | `GGML_BACKEND_PATH=<plugin> GGML_CL11_PLATFORMS=NVIDIA GGML_CL11_DEVICES="GT 430"`（730, 710 も同様） | `CL11_0` |
| rx-cl11 | `GGML_BACKEND_PATH=<plugin> RUSTICL_ENABLE=radeonsi GGML_CL11_PLATFORMS=rusticl GGML_CL11_DEVICES="RX 6400"` | `CL11_0` |
| wx-cl11 | `GGML_BACKEND_PATH=<plugin> GGML_CL11_PLATFORMS=Clover GGML_CL11_DEVICES="WX 2100"` | `CL11_0` |
| rx-vk / wx-vk | `VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/radeon_icd.json` | `Vulkan0`（RX 6400）/ `Vulkan1`（WX 2100） |
| cl5 | `GGML_BACKEND_PATH=<plugin> RUSTICL_ENABLE=radeonsi GGML_CL11_PLATFORMS=NVIDIA,rusticl`（GT 430 は自動 RPC で RPC0） | `RPC0/CL11_0/CL11_1/CL11_2/CL11_3` = GT 430 / GT 730 / GT 710 / RX 6400 / WX 2100 |
| cl4 | 同上 + `GGML_CL11_DEVICES="GT 730,GT 710,RX 6400,WX 2100"` | `CL11_0/…/CL11_3` |
| amdvk（C1） | `VK_ICD_FILENAMES=…radeon_icd.json` | `Vulkan0/Vulkan1` |
| mix4 | `GGML_BACKEND_PATH=<plugin> GGML_CL11_PLATFORMS=NVIDIA GGML_CL11_DEVICES="GT 730,GT 710"` + 上の VK_ICD | `CL11_0/CL11_1/Vulkan0/Vulkan1` |
| mix5 | 同上、`GGML_CL11_DEVICES` なし（GT 430 が自動 RPC） | `RPC0/CL11_0/CL11_1/Vulkan0/Vulkan1` |

`<plugin>` = `~/moe-ep/src/build-cl11/libggml-cl11.so`。全 run で `VK_ICD_FILENAMES` を設定（NVIDIA の Vulkan は使わない）。

例（C1 の最終測定）:
`VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/radeon_icd.json llama-bench -m LFM2.5-8B-A1B-Q4_0.gguf -dev Vulkan0/Vulkan1 -ngl 99 -fa 0 -t 2 -b 512 -ub 512 -r 5 -ts 5/1 -sm layer -ncmoe 0 -o jsonl -p 512 -n 128 -pg 512,128`

## 1. 単体

### 最終測定（`-r 5`、pp512 / tg128 / pg512,128 が揃っている run）

`-ncmoe` は expert を CPU に置く層数（0 = 全 expert が GPU）、`-ub` は `-b` と同じ。`final_cpu` は旧指標（pp128、pp512、tg64、`-pg 512,64`）で測った最初の run、`final_cpu_ncmoe0_ub128` が新指標。

| tag | devices | -ncmoe | -ub | pp512 | tg128 | pp512+tg128 (t/s overall) | pp128 | tg64 | reps |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| final_cpu | none | 0 | 128 | 12.17 ± 0.01 | - | - | 12.12 ± 0.80 | 6.18 ± 0.02 | 5 |
| final_cpu_ncmoe0_ub128 | none | 0 | 128 | 12.14 ± 0.01 | 6.14 ± 0.01 | 10.04 ± 0.01 | - | - | 5 |
| final_gt430_ncmoe24_ub128 | CL11_0 | 24 | 128 | 15.41 ± 0.02 | 10.26 ± 0.05 | 13.97 ± 0.01 | - | - | 5 |
| final_gt710_ncmoe20_ub128 | CL11_0 | 20 | 128 | 16.65 ± 0.02 | 8.65 ± 0.02 | 13.98 ± 0.03 | - | - | 5 |
| final_gt730_ncmoe22_ub128 | CL11_0 | 22 | 128 | 17.86 ± 0.03 | 12.15 ± 0.06 | 16.26 ± 0.02 | - | - | 5 |
| final_mg_amdvk_ts11-2_ub128 | Vulkan0/Vulkan1 | 0 | 128 | 428.53 ± 3.08 | 38.57 ± 0.18 | 140.95 ± 0.96 | - | - | 5 |
| final_mg_amdvk_ts5-1_ub1024 | Vulkan0/Vulkan1 | 0 | 1024 | 631.40 ± 5.22 | 44.03 ± 0.48 | 170.36 ± 0.95 | - | - | 5 |
| final_mg_amdvk_ts5-1_ub128 | Vulkan0/Vulkan1 | 0 | 128 | 363.08 ± 3.99 | 43.88 ± 0.42 | 146.40 ± 0.35 | - | - | 5 |
| final_mg_amdvk_ts5-1_ub256 | Vulkan0/Vulkan1 | 0 | 256 | 504.99 ± 4.04 | 43.84 ± 0.30 | 161.26 ± 0.63 | - | - | 5 |
| final_mg_amdvk_ts5-1_ub512 | Vulkan0/Vulkan1 | 0 | 512 | 631.75 ± 5.24 | 44.12 ± 0.44 | 170.55 ± 1.32 | - | - | 5 |
| final_mg_amdvk_ts6-1_ub128 | Vulkan0/Vulkan1 | 0 | 128 | 429.41 ± 3.33 | 38.62 ± 0.35 | 140.13 ± 1.40 | - | - | 5 |
| final_mg_amdvk_ts9-2_ub1024 | Vulkan0/Vulkan1 | 0 | 1024 | 634.20 ± 3.82 | 43.97 ± 0.57 | 171.37 ± 1.22 | - | - | 5 |
| final_mg_amdvk_ts9-2_ub128 | Vulkan0/Vulkan1 | 0 | 128 | 363.58 ± 3.94 | 43.65 ± 0.22 | 146.56 ± 0.97 | - | - | 5 |
| final_mg_amdvk_ts9-2_ub256 | Vulkan0/Vulkan1 | 0 | 256 | 505.54 ± 4.16 | 44.23 ± 0.52 | 161.23 ± 0.85 | - | - | 5 |
| final_mg_amdvk_ts9-2_ub512 | Vulkan0/Vulkan1 | 0 | 512 | 631.45 ± 6.01 | 43.92 ± 0.22 | 170.29 ± 0.52 | - | - | 5 |
| final_mg_cl4_ts4-12-50-24_ub128 | CL11_0/CL11_1/CL11_2/CL11_3 | 0 | 128 | 55.20 ± 0.50 | 11.23 ± 0.08 | 30.60 ± 0.08 | - | - | 5 |
| final_mg_cl5_ts2-4-16-60-18_ub128 | RPC0/CL11_0/CL11_1/CL11_2/CL11_3 | 0 | 128 | 48.78 ± 0.35 | 11.33 ± 0.05 | 29.17 ± 0.10 | - | - | 5 |
| final_mg_cl5_ts8-8-18-40-18_ub128 | RPC0/CL11_0/CL11_1/CL11_2/CL11_3 | 0 | 128 | 33.27 ± 0.19 | 11.70 ± 0.10 | 24.07 ± 0.14 | - | - | 5 |
| final_mg_mix4_ts1-4-60-20_ub128 | CL11_0/CL11_1/Vulkan0/Vulkan1 | 0 | 128 | 209.61 ± 1.25 | 33.22 ± 0.16 | 100.79 ± 0.54 | - | - | 5 |
| final_mg_mix5_ts2-2-8-50-20_ub128 | RPC0/CL11_0/CL11_1/Vulkan0/Vulkan1 | 0 | 128 | 90.18 ± 0.62 | 24.17 ± 0.11 | 57.59 ± 0.11 | - | - | 5 |
| final_rx-cl11_ncmoe14_ub128 | CL11_0 | 14 | 128 | 28.49 ± 0.08 | 10.54 ± 0.31 | 21.30 ± 0.10 | - | - | 5 |
| final_rx-cl11_ncmoe6_ub128 | CL11_0 | 6 | 128 | 55.98 ± 0.95 | 9.57 ± 0.07 | 28.28 ± 0.16 | - | - | 5 |
| final_rx-vk_ncmoe0_ub512 | Vulkan0 | 0 | 512 | 117.99 ± 0.35 | 12.04 ± 0.00 | 40.25 ± 0.03 | - | - | 5 |
| final_rx-vk_ncmoe5_ub128 | Vulkan0 | 5 | 128 | 71.84 ± 0.19 | 26.98 ± 1.31 | 49.06 ± 2.08 | - | - | 5 |
| final_rx-vk_ncmoe5_ub512 | Vulkan0 | 5 | 512 | 94.37 ± 0.23 | 25.68 ± 1.33 | 64.00 ± 2.48 | - | - | 5 |
| final_rx-vk_ncmoe6_ub512 | Vulkan0 | 6 | 512 | 100.04 ± 0.26 | 23.89 ± 0.54 | 58.34 ± 1.75 | - | - | 5 |
| final_wx-cl11_ncmoe18_ub128 | CL11_0 | 18 | 128 | 20.59 ± 0.02 | 8.49 ± 0.14 | 15.98 ± 0.08 | - | - | 5 |
| final_wx-vk_ncmoe17_ub128 | Vulkan1 | 17 | 128 | 15.30 ± 0.17 | 12.48 ± 0.09 | 14.49 ± 0.16 | - | - | 5 |

### n-cpu-moe の探索（`-r 2`、pp128 / tg64）

範囲は 24（全 expert CPU）から 2 刻みで下げ、VRAM 不足でロードできなくなった値で止め、その境界と最適付近は奇数値でも追加した。失敗はそのまま記録（`VRAM不足でロード失敗` = `unable to allocate` / out of device memory）。AMD の WX 2100（Vulkan と Clover）は VRAM を超えると遅くなりながら動き続けたため、掃引を手動で打ち切った（`logs/NOTES.txt`: wx-vk は 8 まで、wx-cl11 は 12 の開始まで）。**NVIDIA の GT 430 / GT 710 は境界付近（GT 430 の 23、GT 710 の 17・19）で動くが tg が大きく落ちる**（390 の遅延確保で VRAM を超えた分が遅い、と推定。未確認）。

選んだ最速値の根拠:

| バックエンド | tg 最速の ncmoe | pp 最速の ncmoe | 最終測定に使った値 |
| --- | --- | --- | --- |
| GT 430（CL11） | 24（10.3） | 24 | 24 |
| GT 730（CL11） | 22（12.1） | 22（17.8）。21 は失敗 | 22 |
| GT 710（CL11） | 20（8.61、18/24 と同等） | 18/20 付近 | 20 |
| RX 6400 rusticl（CL11） | 14（11.0） | 6（54.5） | 14 と 6 |
| WX 2100 Clover（CL11） | 18（8.57） | 16（21.6、tg 6.8） | 18 |
| RX 6400 Vulkan | 5（`-ub 128` で 29.9、`-ub 512` で 26.7） | 0（231、`-ub 128`） | 5（ub128・ub512）、6（ub512）、0（ub512） |
| WX 2100 Vulkan | 17（12.5） | 17 | 17 |

RX 6400 Vulkan の `-ncmoe 5` は 4 GiB の VRAM の上限付近で、`-ncmoe 4` は tg が 19 まで落ちる。`-ncmoe 5` の `-ub` 依存（tg64、`-r 2`）は ub 64: 26.8、ub 128: 29.9、ub 256: 24.0、ub 512: 26.7 で、tg128 の `-r 5` は ub 128: 26.98 ± 1.31、ub 512: 25.68 ± 1.33（標準偏差が大きく ub による tg の差は有意とは言えない）。


**gt430, -ub 128**

| -ncmoe | pp128 (t/s) | tg64 (t/s) |
| --- | --- | --- |
| 24 | 15.40 | 10.32 |
| 23 | 7.22 | 3.77 |
| 22 | VRAM不足でロード失敗 | |

**gt730, -ub 128**

| -ncmoe | pp128 (t/s) | tg64 (t/s) |
| --- | --- | --- |
| 24 | 17.25 | 11.71 |
| 22 | 17.81 | 12.13 |
| 21 | VRAM不足でロード失敗 | |
| 20 | VRAM不足でロード失敗 | |

**gt710, -ub 128**

| -ncmoe | pp128 (t/s) | tg64 (t/s) |
| --- | --- | --- |
| 24 | 16.15 | 8.59 |
| 22 | 15.21 | 7.15 |
| 20 | 16.47 | 8.61 |
| 19 | 7.16 | 3.09 |
| 18 | 16.47 | 8.59 |
| 17 | 10.39 | 2.64 |
| 16 | VRAM不足でロード失敗 | |

**rx-cl11, -ub 128**

| -ncmoe | pp128 (t/s) | tg64 (t/s) |
| --- | --- | --- |
| 24 | 18.58 | 8.68 |
| 22 | 19.11 | 9.26 |
| 20 | 20.82 | 9.70 |
| 18 | 22.79 | 10.06 |
| 16 | 24.75 | 10.31 |
| 15 | 26.31 | 10.59 |
| 14 | 28.24 | 10.98 |
| 13 | 29.96 | 10.89 |
| 12 | 31.94 | 10.53 |
| 11 | 34.69 | 10.64 |
| 10 | 37.82 | 10.34 |
| 9 | 41.01 | 10.24 |
| 8 | 44.42 | 10.03 |
| 7 | 48.38 | 9.70 |
| 6 | 54.53 | 9.62 |
| 5 | 16.14 | 8.58 |
| 4 | 16.01 | 7.46 |
| 3 | 16.01 | 7.14 |
| 2 | 16.10 | 7.41 |
| 0 | 15.93 | 7.40 |

**wx-cl11, -ub 128**

| -ncmoe | pp128 (t/s) | tg64 (t/s) |
| --- | --- | --- |
| 24 | 17.41 | 7.66 |
| 22 | 17.89 | 7.79 |
| 20 | 19.21 | 8.14 |
| 19 | 10.37 | 5.59 |
| 18 | 20.12 | 8.57 |
| 16 | 21.57 | 6.78 |
| 14 | 10.12 | 1.66 |
| 12 | 10.05 | 1.35 |

**rx-vk, -ub 64**

| -ncmoe | pp128 (t/s) | tg64 (t/s) |
| --- | --- | --- |
| 5 | 48.64 | 26.80 |
| 4 | 123.42 | 19.48 |

**rx-vk, -ub 128**

| -ncmoe | pp128 (t/s) | tg64 (t/s) |
| --- | --- | --- |
| 24 | 47.44 | 11.97 |
| 22 | 49.50 | 12.74 |
| 20 | 55.08 | 13.43 |
| 18 | 61.98 | 13.74 |
| 16 | 69.49 | 15.08 |
| 14 | 78.18 | 15.98 |
| 12 | 89.50 | 16.88 |
| 10 | 103.89 | 18.72 |
| 8 | 122.27 | 20.73 |
| 7 | 140.85 | 21.89 |
| 6 | 168.16 | 23.69 |
| 5 | 72.76 | 29.89 |
| 4 | 224.62 | 19.23 |
| 2 | 230.51 | 12.04 |
| 0 | 231.13 | 12.03 |

**rx-vk, -ub 256**

| -ncmoe | pp128 (t/s) | tg64 (t/s) |
| --- | --- | --- |
| 6 | 169.31 | 23.91 |
| 5 | 71.04 | 23.97 |
| 4 | 223.84 | 19.20 |

**rx-vk, -ub 512**

| -ncmoe | pp128 (t/s) | tg64 (t/s) |
| --- | --- | --- |
| 12 | 87.27 | 16.94 |
| 10 | 103.13 | 18.99 |
| 8 | 120.89 | 21.12 |
| 7 | 140.63 | 21.85 |
| 6 | 170.34 | 24.01 |
| 5 | 72.08 | 26.67 |
| 4 | 224.23 | 19.03 |

**wx-vk, -ub 128**

| -ncmoe | pp128 (t/s) | tg64 (t/s) |
| --- | --- | --- |
| 24 | 10.42 | 9.89 |
| 22 | 10.87 | 10.43 |
| 21 | 11.37 | 10.79 |
| 20 | 11.97 | 11.23 |
| 19 | 12.58 | 11.49 |
| 18 | 13.32 | 11.84 |
| 17 | 14.14 | 12.47 |
| 16 | 5.25 | 9.09 |
| 14 | 15.28 | 3.44 |
| 12 | 5.89 | 2.09 |
| 10 | 6.14 | 1.50 |
| 8 | 6.31 | 1.15 |

## 2. 複数 GPU

`-ts` は上の `-dev` の並び順。`-sm layer`。

### 最終測定（`-r 5`）

上の最終測定表の `final_mg_*` 行。要約:

| 構成 | `-ts` | `-ub` | pp512 | tg128 | pp512+tg128 |
| --- | --- | --- | --- | --- | --- |
| C1 AMD 2 枚 Vulkan | 5/1 | 128 | 363.08 ± 3.99 | 43.88 ± 0.42 | 146.40 ± 0.35 |
| C1 | 5/1 | 256 | 504.99 ± 4.04 | 43.84 ± 0.30 | 161.26 ± 0.63 |
| **C1** | **5/1** | **512** | **631.75 ± 5.24** | 44.12 ± 0.44 | 170.55 ± 1.32 |
| C1 | 5/1 | 1024 | 631.40 ± 5.22 | 44.03 ± 0.48 | 170.36 ± 0.95 |
| C1 | 9/2 | 128 / 256 / 512 / 1024 | 363.6 / 505.5 / 631.5 / **634.2** | 43.65 / **44.23** / 43.92 / 43.97 | 146.6 / 161.2 / 170.3 / 171.4 |
| C1 | 6/1 | 128 | 429.41 ± 3.33 | 38.62 ± 0.35 | 140.13 ± 1.40 |
| C1 | 11/2 | 128 | 428.53 ± 3.08 | 38.57 ± 0.18 | 140.95 ± 0.96 |
| CL11 5 枚（docs） | 8,8,18,40,18 | 128 | 33.27 ± 0.19 | 11.70 ± 0.10 | 24.07 ± 0.14 |
| CL11 5 枚 | 2,4,16,60,18 | 128 | 48.78 ± 0.35 | 11.33 ± 0.05 | 29.17 ± 0.10 |
| CL11 4 枚（GT 430 なし） | 4,12,50,24 | 128 | 55.20 ± 0.50 | 11.23 ± 0.08 | 30.60 ± 0.08 |
| 混在 mix4（GT 730 + GT 710 CL11、AMD 2 枚 Vulkan） | 1,4,60,20 | 128 | 209.61 ± 1.25 | 33.22 ± 0.16 | 100.79 ± 0.54 |
| 混在 mix5（+ GT 430 RPC） | 2,2,8,50,20 | 128 | 90.18 ± 0.62 | 24.17 ± 0.11 | 57.59 ± 0.11 |

C1 の確定値: **`-ts 5/1`（9/2 と 13/3 も同等）、`-ub 512`（1024 でも同じ。pp は ub 128→512 で 1.74 倍、512 以上で頭打ち）**。tg は ts 5/1〜13/3、ub 64〜1024 のすべてで 43.5〜44.5 t/s の一定値（理由は未調査）。ts 6/1 以上（RX 6400 の割合 86 % 以上）では tg が 38.6 に落ち、8/1 は 19.1。ts 1/1 は WX 2100 の VRAM（2 GiB）を超えて 4.15。

pp は `-ub` を上げても 8B の Vulkan 層分割で頭打ちになる（`-ub 512` 以上）。`-ub 1024` は `-b 1024` が必要。

### 探索（`-r 2`、pp128 / tg64。`-ub` は tag 中にあるものを除き 128）

| tag | devices | -ncmoe | -ub | pp512 | tg128 | pp512+tg128 (t/s overall) | pp128 | tg64 | reps |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mgx_amdvk_ts1,1 | Vulkan0/Vulkan1 | 0 | 128 | - | - | - | 29.99 ± 0.11 | 4.15 ± 0.00 | 2 |
| mgx_amdvk_ts11,2 | Vulkan0/Vulkan1 | 0 | 128 | - | - | - | 415.56 ± 2.64 | 39.37 ± 0.86 | 2 |
| mgx_amdvk_ts13,3 | Vulkan0/Vulkan1 | 0 | 128 | - | - | - | 346.41 ± 8.69 | 43.76 ± 0.15 | 2 |
| mgx_amdvk_ts2,1 | Vulkan0/Vulkan1 | 0 | 128 | - | - | - | 202.21 ± 3.60 | 36.53 ± 0.16 | 2 |
| mgx_amdvk_ts3,1 | Vulkan0/Vulkan1 | 0 | 128 | - | - | - | 255.10 ± 3.53 | 39.88 ± 0.22 | 2 |
| mgx_amdvk_ts4,1 | Vulkan0/Vulkan1 | 0 | 128 | - | - | - | 293.13 ± 6.36 | 41.99 ± 0.13 | 2 |
| mgx_amdvk_ts5,1 | Vulkan0/Vulkan1 | 0 | 128 | - | - | - | 348.15 ± 7.74 | 43.68 ± 0.13 | 2 |
| mgx_amdvk_ts5,2 | Vulkan0/Vulkan1 | 0 | 128 | - | - | - | 227.73 ± 3.97 | 39.12 ± 0.79 | 2 |
| mgx_amdvk_ts6,1 | Vulkan0/Vulkan1 | 0 | 128 | - | - | - | 419.37 ± 7.94 | 38.46 ± 0.66 | 2 |
| mgx_amdvk_ts7,2 | Vulkan0/Vulkan1 | 0 | 128 | - | - | - | 293.81 ± 6.05 | 42.09 ± 0.12 | 2 |
| mgx_amdvk_ts8,1 | Vulkan0/Vulkan1 | 0 | 128 | - | - | - | 300.61 ± 6.40 | 19.12 ± 0.14 | 2 |
| mgx_amdvk_ts9,2 | Vulkan0/Vulkan1 | 0 | 128 | - | - | - | 347.19 ± 8.72 | 44.02 ± 0.34 | 2 |
| mgx_amdvk_ub256_ts5,1 | Vulkan0/Vulkan1 | 0 | 256 | - | - | - | 348.80 ± 8.40 | 43.75 ± 0.01 | 2 |
| mgx_amdvk_ub512_ts11,2 | Vulkan0/Vulkan1 | 0 | 512 | - | - | - | 414.81 ± 6.62 | 39.03 ± 0.10 | 2 |
| mgx_amdvk_ub512_ts4,1 | Vulkan0/Vulkan1 | 0 | 512 | - | - | - | 293.88 ± 6.61 | 42.26 ± 0.04 | 2 |
| mgx_amdvk_ub512_ts5,1 | Vulkan0/Vulkan1 | 0 | 512 | - | - | - | 348.39 ± 8.43 | 43.46 ± 0.15 | 2 |
| mgx_amdvk_ub512_ts9,2 | Vulkan0/Vulkan1 | 0 | 512 | - | - | - | 348.15 ± 8.47 | 44.48 ± 0.05 | 2 |
| mgx_amdvk_ub64_ts5,1 | Vulkan0/Vulkan1 | 0 | 64 | - | - | - | 215.28 ± 2.73 | 43.72 ± 0.19 | 2 |
| mgx_cl4_ts4,12,50,24 | CL11_0/CL11_1/CL11_2/CL11_3 | 0 | 128 | - | - | - | 52.06 ± 0.55 | 11.22 ± 0.11 | 2 |
| mgx_cl4_ts8,18,40,18 | CL11_0/CL11_1/CL11_2/CL11_3 | 0 | 128 | - | - | - | 41.17 ± 0.65 | 11.71 ± 0.00 | 2 |
| mgx_cl5_ts0,4,16,60,20 | RPC0/CL11_0/CL11_1/CL11_2/CL11_3 | 0 | 128 | - | - | - | 51.60 ± 0.49 | 11.12 ± 0.12 | 2 |
| mgx_cl5_ts2,4,16,60,18 | RPC0/CL11_0/CL11_1/CL11_2/CL11_3 | 0 | 128 | - | - | - | 46.04 ± 0.24 | 11.51 ± 0.02 | 2 |
| mgx_cl5_ts4,4,18,50,24 | RPC0/CL11_0/CL11_1/CL11_2/CL11_3 | 0 | 128 | - | - | - | 40.86 ± 0.31 | 11.36 ± 0.04 | 2 |
| mgx_cl5_ts8,8,18,40,18 | RPC0/CL11_0/CL11_1/CL11_2/CL11_3 | 0 | 128 | - | - | - | 31.37 ± 0.36 | 11.65 ± 0.02 | 2 |
| mgx_mix4_ts0,4,60,20 | CL11_0/CL11_1/Vulkan0/Vulkan1 | 0 | 128 | - | - | - | 186.68 ± 1.78 | 28.02 ± 0.16 | 2 |
| mgx_mix4_ts1,4,60,20 | CL11_0/CL11_1/Vulkan0/Vulkan1 | 0 | 128 | - | - | - | 202.94 ± 2.61 | 32.89 ± 0.23 | 2 |
| mgx_mix4_ts2,8,50,20 | CL11_0/CL11_1/Vulkan0/Vulkan1 | 0 | 128 | - | - | - | 91.27 ± 1.45 | 24.18 ± 0.15 | 2 |
| mgx_mix4_ts4,8,40,20 | CL11_0/CL11_1/Vulkan0/Vulkan1 | 0 | 128 | - | - | - | 74.85 ± 0.48 | 23.47 ± 0.10 | 2 |
| mgx_mix5_ts2,2,8,50,20 | RPC0/CL11_0/CL11_1/Vulkan0/Vulkan1 | 0 | 128 | - | - | - | 87.41 ± 1.43 | 24.07 ± 0.14 | 2 |
| mgx_mix5_ts4,4,8,40,20 | RPC0/CL11_0/CL11_1/Vulkan0/Vulkan1 | 0 | 128 | - | - | - | 61.13 ± 0.22 | 21.84 ± 0.16 | 2 |

- `mgx_cl5_ts0,4,16,60,20` は GT 430 に層を割り当てない 5 枚指定（実質 4 枚 + 空の RPC）。
- docs/MULTI_GPU.md の 5 枚（Qwen3-8B の 13.15 t/s、63 トークン）と比べ、今回の LFM2.5-8B-A1B の 5 枚は tg 11.7 t/s（`-r 5` の tg128）。同じ構成（`-ts 8,8,18,40,18`）で、docs の記録（LFM2.5-8B-A1B Q4_0、`-ub 128`、環境指定なし、13.15 t/s）より約 11 % 低い。原因は未調査（llama.cpp のバージョン差、`-fa 0` 以外の既定値の差、測定長の違いの可能性）。

## 3. 正しさ（CPU の logits との KL ダイバージェンス）

`llama-perplexity -f <text> -c 512 --chunks 4 -b 128 -ub 128 -t 2 -fa off`（テキストは `cat README.md docs/*.md`、359,423 B、SHA256 `f2c7dabd2776c7aef2aaca949ecc793560fc717346bf866164c0f5e71af90af7`。docs/CL11.md の LFM2 の手順と同じ 4 チャンク）。基準: 同じ引数で `-dev none -ngl 0 --kl-divergence-base`（CPU、PPL 23.941 ± 2.665、`logs/kld_base_cpu.log`）。比較は `--kl-divergence`。各構成の `-ncmoe` / `-ts` は最終測定の最速値。

| config | mean KLD ± SE | 99% KLD | Same top p (top-1 agreement) ± SE | PPL(Q) / PPL(CPU) | rc |
| --- | --- | --- | --- | --- | --- |
| C0_rx-vk_ncmoe5 | 0.021625 ± 0.001480 | 0.198596 | 92.941 % ± 0.802 | 0.997836 | 0 |
| C1_amdvk_ts5-1 | 0.021464 ± 0.001467 | 0.195443 | 92.843 % ± 0.808 | 0.999407 | 0 |
| cl5_ts8-8-18-40-18 | 0.021303 ± 0.001716 | 0.264785 | 92.255 % ± 0.837 | 1.022091 | 0 |
| gt430_ncmoe24 | 0.020528 ± 0.001270 | 0.220035 | 93.431 % ± 0.776 | 1.024901 | 0 |
| gt710_ncmoe20 | 0.020537 ± 0.001269 | 0.220558 | 93.431 % ± 0.776 | 1.025299 | 0 |
| gt730_ncmoe22 | 0.020538 ± 0.001270 | 0.220804 | 93.333 % ± 0.781 | 1.025106 | 0 |
| mix4_ts1-4-60-20 | 0.024063 ± 0.002021 | 0.243178 | 92.843 % ± 0.808 | 1.014486 | 0 |
| mix5_ts2-2-8-50-20 | 0.023615 ± 0.001772 | 0.236111 | 93.333 % ± 0.781 | 1.025843 | 0 |
| rx-cl11_ncmoe14 | 0.021034 ± 0.001364 | 0.203302 | 93.431 % ± 0.776 | 1.019319 | 0 |
| wx-cl11_ncmoe18 | 0.021623 ± 0.001609 | 0.224404 | 93.627 % ± 0.765 | 1.022716 | 0 |
| wx-vk_ncmoe17 | 0.021260 ± 0.001726 | 0.264837 | 92.255 % ± 0.837 | 1.022014 | 0 |

- 平均 KLD の標準誤差は 0.0013〜0.0020。全構成の差（0.0205〜0.0241）は、mix4/mix5 の 0.0236〜0.0241 を除き標準誤差の範囲内。
- C0（RX 6400 Vulkan）0.0216 ± 0.0015、C1（AMD 2 枚）0.0215 ± 0.0015、5 枚（CL11、GT 430 RPC）0.0213 ± 0.0017。top-1 一致率は 92.9 / 92.8 / 92.3 %（SE 0.8）。3 構成に有意差はない。
- docs/CL11.md の既存記録（GT 430 0.0216、GT 730 0.0216、RX 6400 Vulkan 0.0196〜0.0224）と整合する。PPL(Q)/PPL(CPU) が 1.02 前後に出る CL11 / NVIDIA 構成は、CPU が活性を 8 bit に量子化する差（docs/MULTI_GPU.md の仮説）による。
- 4 チャンク（約 1,000 トークン）の評価で、docs の `chunks 4` と同じ規模。確度を上げるなら長いテキストが必要（未実施）。

## 4. プロファイル

ログ: `logs/prof_*`、`logs/prof2_*`。`GGML_VK_PERF_LOGGER=1`、`GGML_CL11_LAUNCH_LOG=1`、`GGML_CL11_PROFILE=1`、`GGML_SCHED_DEBUG=1`（debug ログに `-v` 必要。split 数は起動時の `sched_reserve` 行）。プロファイル有効時は計測のために遅くなる（`avg_ts` は下に併記）。

| 構成 | graph split 数（tg 1 トークンのグラフ） | 備考 |
| --- | --- | --- |
| C0（RX 6400 Vulkan、ncmoe 5） | 8（pp グラフは 11） | CPU と GPU の往復が多い。入力テンソルは 1 split あたり約 8 KiB（hidden 2048 × f32） |
| C1（AMD 2 枚 Vulkan） | 3 | 境界は GPU→GPU の 1 回のみ（ホスト経由） |
| 5 枚（CL11 + GT 430 RPC） | 6 | RPC0 → CL11_0 → CL11_1 → CL11_2 → CL11_3、各境界の入力は約 8 KiB（`l_out-N`、`ffn_moe_out-N` など）+ 小さなインデックス |
| GT 730（CL11、ncmoe 22） | 42 | 22 層の MUL_MAT_ID を CPU に置くため CPU と GPU を層ごとに往復 |

**C0（RX 6400 Vulkan、ncmoe 5）** `GGML_VK_PERF_LOGGER` の 1 グラフ（tg）の GPU op 合計 19.48 ms。内訳: `MUL_MAT_ID`（up/gate と down、19 層）8.64 ms、mat-vec q4_0 3.3 ms、出力の q6_K mat-vec（m=128000）2.45 ms、残りの小さな op（GLU、ADD、CONCAT、MUL、SSM_CONV、RMS_NORM 等）約 5.1 ms。壁時計は 1/23.55 = 42.5 ms/token（ロガー有効時）。**差分 約 23 ms が CPU 側（先頭 5 層の experts と同期）**（差分であり直接測定ではない。ロガーは小さい op を大きく見せる、VULKAN_AMD.md の注意と同じ）。

**C1（AMD 2 枚）** `GGML_VK_PERF_LOGGER=1` は 2 デバイスで `GGML_ASSERT(ctx->compute_ctx.expired())` で abort（`logs/prof_c1_vk_perf.log`、rc=134）。内訳は取れていない。壁時計は 1/44.1 = 22.7 ms/token（`-r 5`）。split は 3 で、層をまたぐ転送は hidden 8 KiB が 1 回のみ（推定: 8 KiB は PCIe Gen2 x1 の実測 0.3〜0.4 GB/s で約 0.03 ms なので、帯域は無視でき、同期のレイテンシが支配）。

**5 枚（CL11 + RPC、`-ts 8,8,18,40,18`）** `GGML_CL11_PROFILE=1` の 17 グラフ分のカーネル合計と、1 トークンあたり（÷17）: GT 730（CL11_0）54.7 ms（3.2 ms）、GT 710（CL11_1）314.9 ms（18.5 ms）、RX 6400（CL11_2）329.3 ms（19.4 ms）、WX 2100（CL11_3）264.5 ms（15.6 ms）。**4 枚のカーネル合計 56.7 ms/token**。GT 430（RPC0、別プロセス）のカーネル時間は取っていない。壁時計は 1/12.04 = 83 ms/token（`launch` ログの run）なので、**カーネル以外（GT 430 の層、RPC・同期・転送、ホストの待ち）が約 26 ms/token**（差分）。層分割はカーネル時間が直列に足される。最後のデバイス CL11_3（WX 2100）の内訳では q6_K 出力 mat-vec（1 回 6.4 ms、33 GB/s）が 41 %、MUL_MAT_ID が 36 %（`logs/prof_cl5_kernels.log`）。起動回数（`GGML_CL11_LAUNCH_LOG=1`、1 トークンの 4 グラフ）: カーネル 52 / 134 / 305 / 88 の計 579、ノード数は 110 / 280 / 623 / 176。

**GT 730（CL11、ncmoe 22）** カーネル合計 457.6 ms / 17 = 26.9 ms/token（q4_0 mat-vec 30.6 %、q6_K 出力 28.3 %、MUL_MAT_ID 7.0 %）。壁時計 1/12.12 = 82.5 ms/token なので、**約 55 ms/token が CPU 側（22 層分の experts と split 間の同期）**（差分）。graph split は 42。

PCIe 転送量の見積もり（**推定**）: 層ごとの境界で運ぶのは hidden 1 ベクトル（2048 × f32 = 8 KiB）程度で、1 トークンあたり C0 で split 8 回、GT 730 で 42 回、5 枚で 6 回。合計でも数百 KiB/token で、PCIe Gen2 x1（実測 0.3〜0.4 GB/s、docs/HARDWARE.md）でも転送時間は 1 ms 未満。支配的なのは転送量ではなく、split ごとの同期（`clFinish` 相当 / フェンス待ち）と、CPU 側の experts 計算。

## 出力物

- `scripts/`: `env.sh`、`run.sh`、`bench.sh`、`sweep.sh`、`mg.sh`、`final_single.sh`、`final_mg.sh`、`kld.sh`、`profile*.sh`、`queue*.sh`、`summarize.py`、`summarize_kld.py`（`profile.sh` の `prof_c1_vk_perf` は abort のため `profile2.sh` / `profile3.sh` で補完）。
- `logs/`: 全 run の `.log` / `.out`、`queue*.log`、`llama-cfg.log`、`plugin-cfg.log`、`NOTES.txt`、`runs-bad/`（`-ts` を `,` 区切りにして誤測定した run。結果に使っていない）。

## 問題点・未検証

- 5 枚の tg は docs 記録より約 11 % 低い（原因未調査）。
- C1 の GPU 内訳は `GGML_VK_PERF_LOGGER` の abort で未取得。GT 430（RPC）のカーネル時間も未取得。
- `-r 2` の掃引値はノイズが大きい（例: RX 6400 Vulkan `-ncmoe 5` の tg が 26.7〜29.9）。最終値は `-r 5`。
- PCIe の負荷時リンク速度は未取得。アイドル時の値は省電力で変動する。
- KLD は 4 チャンクのみ。
- `wx-vk` と `wx-cl11` の掃引は VRAM 超過で急に遅くなったため手動で打ち切った。
- マイクロベンチ・投機的デコードの別エージェントが同じロックを使うため、壁時計の所要時間にはロック待ちを含む。
- コミット・push はしていない。
