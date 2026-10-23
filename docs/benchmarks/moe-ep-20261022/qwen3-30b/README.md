# Qwen3-30B-A3B の基準測定（tb250、段階8の準備）

測定日 2026-10-22〜23。SGL worktree `feat/moe-expert-parallel`、コード変更なし（SGL / llama.cpp とも）。すべて実測。推定は「推定」と明記。単位は t/s、値は平均 ± 標準偏差。

## 結論

| 区分 | 構成 | pp512 | tg128 | pg512,128 |
| --- | --- | --- | --- | --- |
| CPU のみ | 2 スレッド | 3.62 ± 0.00 | 3.10 ± 0.01 | 3.24 ± 0.00 |
| 単体 tg 最速 | RX 6400 Vulkan `-ncmoe 39` | 18.31 ± 0.13 | 7.37 ± 0.07 | 13.39 ± 0.07 |
| 単体 pg 最速 | RX 6400 Vulkan `-ncmoe 41` | 59.76 ± 0.61 | 5.96 ± 0.01 | 21.17 ± 0.05 |
| 複数 tg 最速 | AMD 2 枚 `-ncmoe 34` + `-ot`（WX に 5 層分の expert） | 16.39 ± 0.04 | **7.70 ± 0.00** | 12.72 ± 0.01 |
| C1 相当 | AMD 2 枚層分割 `-ts 42/6 -ncmoe 33` | 20.50 ± 0.06 | 7.50 ± 0.01 | 13.13 ± 0.25 |
| 複数 pg 最速（mix4） | RX+WX Vulkan + GT 730/710 CL11、`-ncmoe 29` | 30.42 ± 1.60 | 6.59 ± 0.02 | **22.87 ± 0.04** |
| mix5（GT 430 RPC 含む） | 同上 + GT 430 | 29.08 ± 1.50 | 6.52 ± 0.02 | 21.12 ± 0.12 |
| CL11 5 枚 | RX/WX rusticl・Clover + NVIDIA 3 枚、`-ub 128` | 5.79 ± 0.04 | 4.62 ± 0.05 | 4.96 ± 0.03 |

- **どの構成も tg は 7.7 t/s 止まり（CPU の 2.5 倍）**。Q4_0 は 17.4 GB で VRAM 計 約10 GB に載らず、expert の約 7 割が CPU に残る。tg の上限は CPU の expert 計算（AVX 無し 2 コア）で、帯域ではない（推定: 1 トークン約 1 GB の expert 読み出し）。GPU を足しても VRAM に載る expert 層が 9→14 層になるだけで、tg の伸びは 5 %（7.37→7.70）。
- **pp は VRAM の空きで二峰性**。`-ncmoe 39`（VRAM 満杯）では pp512 18、`-ncmoe 41`〜44 では 54〜60。CPU 側 expert を層ごとに GPU へ流して計算する op offload（`-nopo 1` で 18→8）が、1 層分の置き場が VRAM に無いと働かないため（下記 ppchk）。tg 最速と pp 最速の `-ncmoe` が違う。
- NVIDIA（GT 730/710/430）を足すと tg は下がる（7.5→6.5）が、pg は 13→22.9 に上がる（pp が 30 に伸びる。層を持つカードが増えて op offload の置き場が確保できるため。原因の切り分けは未実施）。
- Q3_K_M は CPU 側の K-quant dot が AVX 無しで遅く、tg 1.8（Q4_0 の 7.4 に対して約 1/4）。**この CPU では Q4_0 が前提**。CL11 プラグインが扱う型も Q4_0/Q8_0/Q4_K/Q6_K のみ。

## 環境

tb250 は baseline README（`../../moe-ep-20261021/baseline/README.md`）と同じ（Celeron G3930 2C AVX 無し、RAM 31 GB、RX 6400 4 GB / WX 2100 2 GB 公式 Vulkan RADV、GT 730 1 GB / GT 710 2 GB CL11、GT 430 1 GB 自動 RPC、VK_ICD は AMD のみ）。llama.cpp `23b0202a189c44a54625aadcb37a946dd1d6278d`、CL11 プラグインは `~/moe-ep/src/build-cl11/libggml-cl11.so`（baseline のもの）。測定は `~/moe-ep/q30/run.sh`（`~/bench/gpu-bench.lock` の flock、温度 85 ℃待ち、各 run の先頭に `sgl-driver-switch status` の perf を記録。perf は触っていない）。

- **llama-imatrix のために共有ビルドを更新した**。`ninja llama-imatrix` を実行したところ、ビルドツリーが古く libggml*/libllama*/libllama-common が同じソース（`23b0202`、`git status` clean）から再ビルドされた（2026-10-22 19:10〜19:19、ロック保持中）。コード変更は無いが、`~/moe-ep/src/llama-build/bin` の so のタイムスタンプが更新されている。
- perf の状態は run ごとに異なる（`logs/summary.md` の perf 列）。別担当が `pin`（manual）に切り替えていた時間帯がある。CPU 律速の tg は perf に依存しなかった（sweep の auto と final の manual で同値、例: RX n39 tg 7.37 / 7.37）。

## モデル

| 量子化 | 取得元 | サイズ | sha256 |
| --- | --- | --- | --- |
| Q4_0 | `https://huggingface.co/unsloth/Qwen3-30B-A3B-GGUF/resolve/main/Qwen3-30B-A3B-Q4_0.gguf` | 17,379,988,032 B | `8b8e9febb6ed326d91a016f5d465cf5b55a0b328d157a43e2242fda71fde9c40`（HF の LFS oid と一致、HDD 原本と SSD コピーの両方で確認） |
| Q3_K_M | 同 `Qwen3-30B-A3B-Q3_K_M.gguf` | 14,711,847,488 B | `a63f06070fccd2e3a329f3bfe5e996709a7f305d04c6acf2c8ae900c4723fffa`（同上） |

原本は tb250 `/mnt/hdd/models/`、測定用は `~/moe-ep/models/`。Q3_K_M の最初のダウンロードは HTTP/2 切断（curl 92）で途中停止し、`-C -` で再開して sha256 一致を確認。`qwen3moe 30B.A3B`、48 層、128 expert、8 使用。

## 構成ごとのコマンド（`logs/run/<tag>.log` に環境変数と完全なコマンド）

共通: `llama-bench -m ~/moe-ep/models/Qwen3-30B-A3B-Q4_0.gguf -ngl 99 -fa 0 -t 2 -b UB -ub UB -r 3 -sm layer -o jsonl -p 512 -n 128 -pg 512,128`（`~/moe-ep/src/llama-build/bin/`）。環境は `VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/radeon_icd.json`（以下、追加分のみ）。CPU は `-dev none -ngl 0`。

| tag | 追加の環境変数 | `-dev` と引数 |
| --- | --- | --- |
| final_cpu | なし | `-dev none -ngl 0 -ncmoe 0 -ub 512` |
| final2_rx_n39 / n41 / n44(ub512,1024) | なし | `-dev Vulkan0 -ncmoe N -ub 512` |
| final2_amd_ts42-6_n33（n36, n38 も） | なし | `-dev Vulkan0/Vulkan1 -ncmoe 33 -ts 42/6 -ub 512` |
| final2_ot_wx5_n34 | なし | `-dev Vulkan0/Vulkan1 -ncmoe 34 -ts 48/0 -ot 'blk\.(34|35|36|37|38)\.ffn_.*_exps\.weight=Vulkan1'`（非 expert は全部 RX、層 34〜38 の expert を WX、39〜47 は RX、0〜33 は CPU） |
| final2_mix4_…_n29 | `GGML_BACKEND_PATH=<plugin> GGML_CL11_PLATFORMS=NVIDIA GGML_CL11_DEVICES="GT 730,GT 710" GGML_CL11_AUTO_RPC=0` | `-dev Vulkan0/Vulkan1/CL11_0/CL11_1 -ncmoe 29 -ts 38/5/2/3` |
| final2_mix5b_…_n29 | `GGML_BACKEND_PATH=<plugin> GGML_CL11_PLATFORMS=NVIDIA` | `-dev Vulkan0/Vulkan1/CL11_1/CL11_0/RPC0 -ncmoe 29 -ts 36/5/4/2/1`（CL11_0=GT 730、CL11_1=GT 710、RPC0=GT 430） |
| final2_cl5b_…_n27_ub128 | `GGML_BACKEND_PATH=<plugin> RUSTICL_ENABLE=radeonsi GGML_CL11_PLATFORMS=NVIDIA,rusticl` | `-dev CL11_2/CL11_3/CL11_1/CL11_0/RPC0 -ncmoe 27 -ts 35/5/4/2/2 -b 128 -ub 128` |

層分割では先頭の層が先頭のデバイスに載り、`-ncmoe N` は先頭 N 層の expert を CPU に置く。expert を GPU に持つ層（N〜47）が後ろの層にあたるため、`-dev` の順と `-ts` を「RX（CPU expert 層＋後続の GPU 層）、WX、GT 710、GT 730、GT 430」の順に、後続デバイスが後ろの層を持つように合わせた（最初に baseline と同じ ts 5/1 を使うと GPU expert が WX の 8 層に偏り tg 2.4 まで落ちた: `logs/run/sw_amd_n40` ほか）。

## 結果の詳細

全 final の表と全 sweep は `logs/summary.md`（`scripts/summarize.py` が `logs/run/*.out` から生成）。

### 最終測定

| tag | perf | pp512 | tg128 | pg512,128 |
| --- | --- | --- | --- | --- |
| final_cpu | auto | 3.62 ± 0.00 | 3.10 ± 0.01 | 3.24 ± 0.00 |
| final2_rx_n39_ub512 | manual | 18.31 ± 0.13 | 7.37 ± 0.07 | 13.39 ± 0.07 |
| final2_rx_n41_ub512 | auto | 59.76 ± 0.61 | 5.96 ± 0.01 | 21.17 ± 0.05 |
| final2_rx_n44_ub512 | auto | 54.13 ± 2.87 | 4.67 ± 0.86 | 20.10 ± 0.12 |
| final2_rx_n44_ub1024 | auto | 56.22 ± 0.47 | 5.75 ± 0.01 | 20.26 ± 0.11 |
| final2_amd_ts42-6_n33 | manual | 20.50 ± 0.06 | 7.50 ± 0.01 | 13.13 ± 0.25 |
| final2_amd_ts42-6_n36 | auto | 19.59 ± 0.93 | 6.13 ± 0.06 | 12.38 ± 0.06 |
| final2_amd_ts42-6_n38 | auto | 20.53 ± 0.15 | 6.01 ± 0.02 | 12.46 ± 0.06 |
| final2_ot_wx5_n34 | manual | 16.39 ± 0.04 | 7.70 ± 0.00 | 12.72 ± 0.01 |
| final2_mix4_ts38-5-2-3_n29 | manual | 30.42 ± 1.60 | 6.59 ± 0.02 | 22.87 ± 0.04 |
| final2_mix5b_ts36-5-4-2-1_n29 | manual | 29.08 ± 1.50 | 6.52 ± 0.02 | 21.12 ± 0.12 |
| final2_cl5b_ts35-5-4-2-2_n27_ub128 | manual | 5.79 ± 0.04 | 4.62 ± 0.05 | 4.96 ± 0.03 |

`final2_rx_n44_ub512` の tg は標準偏差が大きい（rep ごとに差があり、他の n44 系 sweep と合わせても 4.7〜5.8）。

### pp が二峰になる理由（op offload の確認）

RX 6400 単体、`-p 512 -n 0`（または 2048）、`-r 2`（`logs/run/ppchk_*`）:

| 設定 | pp |
| --- | --- |
| `-ncmoe 39 -ub 512` | 18.2 |
| 同 + `-nopo 1`（op offload 無効） | 8.0 |
| `-ncmoe 44 -ub 512` | 56.6 |
| `-ncmoe 44 -ub 2048`（pp2048） | 146 |
| `-ncmoe 48 -ub 512`（expert 全部 CPU） | 53.6 |

CPU 上の expert を GPU へ流して計算する経路が pp を担っている。`-nopo` で 18→8 になる、expert 全部 CPU の n48 が n39 より 3 倍速い、`-ub` を大きくすると伸びる、という 3 点から、ストリーミング（PCIe 転送）が pp の主役で、VRAM が埋まると経路が止まると判断した（VRAM 満杯時にどこへフォールバックするかのコード確認はしていない）。pp128 の sweep 値が 8 と 20 に割れていたのも同じ現象。

### VRAM / RAM / mmap（`logs/run/<tag>.mem`、最大値、MiB）

| 構成 | RX VRAM | WX VRAM | NVIDIA（GT430/730/710） | システム使用 | プロセス RSS | ページキャッシュ | mmap |
| --- | --- | --- | --- | --- | --- | --- | --- |
| CPU | - | - | - | 5.8 GB | 16.9 GB | 24.7 GB | あり（mmap のまま） |
| RX n39 | 3870 | - | - | 7.0 GB | 16.7 GB | 21.1 GB | あり |
| AMD ts42/6 n33 | 3988 | 2007 | - | 12.4 GB | 16.7 GB | 21.0 GB | あり |
| AMD `-ot` n34 | 3870 | 1705 | - | 7.6 GB | 16.7 GB | 21.5 GB | あり |
| mix4 n29 | 3948 | 1777 | 72 / 717 / 984（別 run） | 17.6 GB | 0.6 GB | 19.9 GB | なし（下記） |
| mix5b n29 | 3272 | 1777 | 367 / 718 / 1385（別 run） | 17.1 GB | 0.4 GB | 21.3 GB | なし |
| cl5b n27（ub128） | 3454 | 1778 | 704 / 717 / 1385（別 run） | 18.9 GB | 9.5 GB | 20.4 GB | 一部 |

- Vulkan 単体・AMD のみでは CPU expert は mmap のページを直接読む（RSS 16.7 GB、システム使用は小さい）。CL11 を混ぜると CPU expert が Vulkan のピン留めホスト領域（RX の GTT 約 13 GB）へ複製され、RSS は 0.5 GB 程度、システム使用が 17〜24 GB に増える。mmap のまま動かない。RAM 31 GB でも他の測定が走っていると圧迫が近い（mix5b の別 run で 24.3 GB）。
- NVIDIA の VRAM は `nvidia-smi` で測ったが、サンプリングが CPU を 1 コア奪うため（下記）、時間測定とは別の短い run（`memnv_*`、`-p 64 -n 16`）で取った。GT 730 は 1 GB 中 0.72 GB、GT 710 は 2 GB 中 1.4 GB、GT 430 は 0.37 GB で、いずれも余りがあるが expert は 1 層 約 340 MB 単位なので `-ts` の刻みで入りきらない。
- split 数: 層分割のデバイス数は 1（RX）、2（AMD）、4（mix4）、5（mix5b/cl5b）。

### 打ち切り・失敗

- RX 単体の限界は `-ncmoe 39`（38 以下は VRAM 満杯で pp が半分以下）。AMD 2 枚は ts 42/6 で `-ncmoe 33`（32 は tg 6.8）。mix5b は 29（26 以下で pp が崩れる）。
- `-ncmoe` を限界より下げても OOM で落ちず、遅くなって動き続けた（ロード失敗は無し）。
- cl5b は `-ub 512 -p 512` で GT 430 の RPC バッファ 398 MB が確保できず失敗（`logs/run/invalid-nvsmi/final_cl5b_ts35-5-4-2-2_n27_ub512.log`）。baseline と同じ `-ub 128` で測り直した。
- `-ot` で attention を RX に集め、expert を WX に置く構成は clean 条件で tg 7.70 と最速だが、pp が最小（16.4）。
- Q3_K_M（Vulkan RX 単体、`-p 128 -n 32`、`-r 2`）: `-ncmoe 40…37` で tg 1.74〜1.88、pp128 21〜22。`logs/run/sw_q3rx_*`。AMD 2 枚の sweep は tg が見込めないので中止（手動で停止）。Q3_K_M の final（pp512 / tg128 / pg）は未実施。

### 測定上の無効データ（捨てた run）

`run.sh` に追加した `nvidia-smi` のサンプラーが、390 ドライバでは 1 回ごとに CPU 1 コア分を奪い、CPU 律速の tg を約 4 割下げた（CPU のみ tg 3.10→1.97、RX n39 7.4→4.4）。気づいたのは CPU のみの測り直しで差が出たとき。サンプラーを `NVSMI=1` のときだけにして、影響した run（先に取った final と sweep の一部）は `logs/run/invalid-nvsmi/` に分けて結果表から外し、全部測り直した。表の数字は `final_cpu` と `final2_*` のみ。`final_cpu` はサンプラー追加前、`sw_*`（invalid-nvsmi 以外）はサンプラー無しの run。

## KLD / top-1

実施していない。30B の CPU 基準 logits は短いテキストでも作れる見込み（CPU pp 3.6 t/s × 2048 トークン ≈ 10 分、`llama-perplexity --kl-divergence-base`）だが、今回は速度の基準と expert 頻度に時間を使い、未着手。段階8で SGLX の構成を決めたあとに、`../../moe-ep-20261021/baseline/scripts/kld.sh` と同じ形（`-c 512 --chunks 4`）で取る。

## expert のアクセス頻度（Strata 流の材料）

llama.cpp 付属の `llama-imatrix`（`ninja llama-imatrix` で追加ビルド、ソース変更なし）は MoE の expert ごとの選択回数を `*.ffn_*_exps.weight.counts` に保存する。それを `scripts/expert_freq.py`（依存なしの GGUF 読み取り）で集計した。RX 6400 + `-ncmoe 39`、`-c 512 --chunks 8`（4096 トークン）、層ごとに 32768 件（4096 × 8）。コーパス: readme = `kl/text.txt`（日本語・英語混じりの README）、en_legal = GPL/Apache/LGPL の英文、py_code = Python 標準ライブラリのソース。`logs/imat/`。

### 偏りの大きさ

層ごとの上位 K expert（128 中）が担う選択の割合（48 層平均、一様なら K/128）:

| コーパス | 上位 8 | 上位 16 | 上位 32 | 上位 64 |
| --- | --- | --- | --- | --- |
| readme | 0.403 | 0.579 | 0.776 | 0.951 |
| en_legal | 0.409 | 0.572 | 0.786 | 0.963 |
| py_code | 0.442 | 0.647 | 0.819 | 0.952 |
| 一様 | 0.063 | 0.125 | 0.25 | 0.5 |

- 80 % を覆うのに必要な expert は層平均 30〜35（最小 17〜22、最大 61〜65）。層ごとの差が大きく、浅い層（0〜5）は偏りが小さく（H/Hmax 0.9）、中盤以降は偏りが大きい（`logs/imat/*.summary.txt`）。
- 同じコーパスの前半 4 チャンクで選んだ上位 K を後半 4 チャンクに当てた被覆率（保留データ、`logs/imat/heldout_*.txt`）: readme 上位 32 で 0.760 / 0.758、py_code 上位 32 で 0.830 / 0.753。同じ領域なら偏りは再現する（一様の 0.25 を大きく超える）。
- **別領域への転用は効かない**。readme で選んだ上位 32 は en_legal の選択の 0.257、py_code の 0.383 しか覆わない（一様 0.25）。上位 64 でも 0.49 / 0.66（`logs/imat/compare.txt`）。ホットな expert は話題・言語・コード/散文で入れ替わる。
- 限界: 4096 トークン × 3 コーパスで、生成（tg）中のルーティングそのものではなく、プロンプト処理中の選択を数えている。実際のチャット用途の分布は未測定。

### SGLX / Strata 流で狙えそうな点

1. **VRAM に載せる単位を「層」から「expert」へ**。今は VRAM 約 10 GB で 9〜14 層分（128 expert 全部）を GPU に置き、残り 3 分の 2 以上の層が CPU 計算。同じ容量なら全 48 層の上位約 30 expert（容量の約 25 %）を置ける。同一領域なら選択の約 76〜83 % を覆う（上の保留データ）ので、CPU に回る expert 計算を約 7 割から約 2 割に減らせる見込み（推定。層をまたぐ CPU/GPU の混在実行と転送のコストは別に要る）。CPU が律速（tg 3.1 vs GPU 混在 7.7）なので、効くとすれば tg。
2. **領域が変わると入れ替わるので、固定ではなく動的に**。被覆率が別領域で一様並みに落ちるので、頻度を数えて入れ替える（Strata の発想）。選択回数の更新は軽いので、ホット集合を数百トークン単位で更新する設計が候補。
3. **pp の主役は expert のストリーミング**（ppchk）。VRAM を満杯にせず 1 層分の置き場を残す方が pp は 3 倍速い（18→54〜60）。ホット expert のキャッシュと、ストリーミング用のバッファの VRAM 配分が設計の中心になる。`-ub 2048` で pp2048 が 146 まで伸びたので、バッチ化が効く。
4. **NVIDIA 3 枚の使い道**。Q4_0 の expert 計算を GT 730/710 に置いても tg は伸びず（mix4/5 は RX+WX より tg が低い）、pp/pg を押し上げる側（22.9）に効いた。層ではなく「ホット expert のキャッシュ置き場」「プリフェッチ先」にする方が合うかもしれない（未検証）。
5. **量子化は Q4_0 前提**。K-quant は CPU で 1/4 の速度、CL11 も未対応。expert 側を Q4_0（または CL11 対応の Q4_K）に揃える必要がある。

## 再現

`scripts/tb250/` が tb250 `~/moe-ep/q30/` のスクリプト（`env.sh` 構成名、`run.sh`、`bench.sh TAG CFG [args]`: 環境変数 `NCMOE UB REPS TS OT MODEL`、`sweep.sh`、`queue*.sh` が実行順）。モデルのダウンロードは `dl.sh`。ログは `logs/run/`（`<tag>.log` 先頭 5 行に日時・perf・温度・環境・コマンド、`.out` が jsonl、`.mem` が 2 秒ごとの VRAM/RAM）。

## 残件

- KLD / top-1（CPU 基準）。
- Q3_K_M の final（pp512/tg128/pg）と AMD 2 枚の sweep。
- mix4/mix5 の pp が伸びる理由の切り分け、NVIDIA の VRAM に余りがある理由（`-ts` の刻み）の確認。
- 生成中のルーティングの測定（`llama-imatrix` はプロンプト処理のみ）。実装するならルーターの選択結果を出す小さなフック（eval callback で `ffn_moe_topk` を読む）が必要。
