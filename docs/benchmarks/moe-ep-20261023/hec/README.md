# Qwen3-30B-A3B ホット/ミディアム/コールド expert キャッシュ（SGLX0 の MoE モード）の実装と実測

実装 2026-10-23、tb250（Celeron G3930 2C AVX 無し、RAM 31 GB、RX 6400 4 GB / WX 2100 2 GB 公式 Vulkan RADV、GT 730 1 GB / GT 710 2 GB CL11、GT 430）。llama.cpp `23b0202`（無改造）、`Qwen3-30B-A3B-Q4_0.gguf`（unsloth、層 0〜5 の down は Q4_1）。設計は `docs/plans/HOT_EXPERT_CACHE.md`、前日の判定は `../hec-day1/README.md`。ブランチ `feat/moe-expert-parallel`。
数値はすべて `logs/` の実測（`scripts/` のキュー）。**【推定】** と書いたものだけが実測でない。perf は `sgl-driver-switch perf pin`（キューの先頭で pin、終了時に auto。ログの先頭の `# perf:` に記録）。測定は `~/bench/gpu-bench.lock` の flock + 温度待ち（`adapters/ggml/tools/hec_run.sh`）で 1 本ずつ。

## 結果の要約

構成 **N** = `GGML_SGLX_MOE=Vulkan0:auto:400,CL11_0:300`（RX 6400 がホット + ring、GT 730 が第 2 のホット、コールドは CPU）、`GGML_SGLX_MOE_THREADS=2`、`SPIN_US=0`、`--poll 0`、event 無し。他は上の「実装の構成」のとおり。以下は tb250 の実測（perf pin、1 本ずつ flock）。

### 1. 速度: 乱数トークン（`logs/f_*`、30B Q4_0、`hec_gen --bench`、ub 512）

| 構成 | pp512 | tg128 | pg512,128 | 回数 |
| --- | --- | --- | --- | --- |
| **N（SGLX MoE）** | **64.2**（62.0〜65.0） | **13.24**（13.05〜13.48） | **38.17**（37.92〜38.32） | 6 |
| `-ncmoe 41`（RX、30B 基準の最速 pp 構成） | 62.2 | 7.18 | 24.50 | 2 |
| `-ncmoe 39`（RX） | 18.3 | 7.43 | 13.45 | 2 |
| RX + WX の `-ot`（層 34〜38 を WX、`-ncmoe 34`） | 8.45 | 6.18 | 11.60 | 1 |
| 4 デバイス層分割（`mix4`、`-ncmoe 29`） | 30.4 | 6.63 | 22.99 | 1 |
| llama.cpp `moe_cache` 3000 MiB（`-ncmoe 48`） | 16.4 | 4.45 | 10.83 | 1 |
| 30B 基準（`moe-ep-20261022/qwen3-30b`、参考） | 59.8 | 7.70 | 22.87 | - |

N の測定は基準と交互に走らせた（`f_N_a/b/c/d` の間に baseline）。基準側は 1〜2 回で、`REPS 3〜5` には届いていない（表の「回数」）。N は tg が 30B 基準の 1.72 倍、pg が 1.67 倍（この回の最良の基準 `-ncmoe 41` に対して 1.56 倍）、pp は同等以上（64.2 対 62.2）。**全部の既存構成を上回った。**

### 2. 速度: 自然文（`logs/txt3_*`、fixed prompt の後 128 トークンを greedy 生成、モデルを毎回 page cache に読んでから実行）

プロンプトは `prompts/nat_{en,ja,code,chat}.txt`（49〜145 トークンと短い）。tg は decode の t/s。

| 構成 | en | ja | code | chat |
| --- | --- | --- | --- | --- |
| **N** | **12.64** | **12.89** | **11.78** | **9.56** |
| `-ncmoe 39` | 7.39 | 7.39 | 7.39 | 7.42 |
| `-ncmoe 41` | 7.16 | 7.14 | 7.15 | 7.17 |
| RX + WX（`-ncmoe 34`） | 7.68 | 7.66 | 7.68 | 7.68 |
| 4 デバイス層分割（`mix4`） | 6.18 | 6.56 | 6.55 | 6.61 |
| `moe_cache` 3000 MiB | 6.13 | 7.12 | 3.79 | 3.37 |

N は乱数トークンより自然文のほうが速い（en / ja で 12.6〜12.9）。ただし chat（49 トークン）は 9.56 で、プロンプトが短く、配置に使える頻度の統計が少ないため（prefill の終わりで 1 回配置、その後 decode で更新）と考えている（検証はしていない）。

**短いプロンプトの prefill は N のほうが遅い**: pp は N が 6.3〜16.4 t/s、`-ncmoe 39` が 8.2〜23.7 t/s、`mix4` が 10.3〜29.4 t/s。短い ubatch では 1 つの expert に集まる token が少なく、ring に流す元の条件（4 組以上）を満たす expert が少ないため CPU に回る組が増える。ubatch 512 の乱数 prefill（pp512）では N が優位だが、数十〜百数十トークンの prefill は N が最速ではない。

先に `txt2_*`（`q_base3`）で測った基準のうち en と ja の `-ncmoe 39/41`、`-ot` と `mix4` の一部は 1 t/s 前後まで落ちていた。同じ構成の chat / code は 7 前後で、ほかの処理がページキャッシュを追い出した結果とみて、`txt3_*` で毎回読み込んでから測り直した。`txt2_*` の値は使わない（ログは残してある）。

### 3. 正しさ（§8.1 形式、30B、CPU のロジットを基準）

CPU 基準 `~/moe-ep/kl/base30_6.kld`（`llama-perplexity --kl-divergence-base`、`-c 512 --chunks 6 -b 512 -ub 512`、`kl/text.txt`）。条件: KLD ≤ 最速の単独構成 ×1.2、top-1 ≥ 最速 −1 pt、自己整合 KLD < 0.001。

**prefill 経路（6 チャンク、`logs/kld3_*`）**

| 構成 | KLD 平均 | top-1 | 判定 |
| --- | --- | --- | --- |
| `-ncmoe 39`（基準。`-ncmoe 41` も同値） | 0.004181 ± 0.000220 | 97.12% | - |
| **N** | 0.004440 ± 0.000244（基準の 1.06 倍） | 96.93%（−0.2 pt） | **合格** |
| 4 デバイス（RX + GT 730 + WX + GT 710） | 0.004098 ± 0.000230 | 97.71% | **合格** |

2 チャンクで測った最初の値（N 0.00485、基準 0.00399）は N の KLD が基準 ×1.2 をわずかに超えて見えたが（`logs/kld_sglx_rx`）、誤差の範囲だった。6 チャンクで 1.06 倍になった。

**decode 経路（`hec_gen --dump`、同じテキストを teacher forcing、150 トークンの prefill の後 301 位置、`logs/kld_dec_*`）**

| 比較 | KLD 平均 | top-1 |
| --- | --- | --- |
| `-ncmoe 39` 対 CPU（基準） | 0.00341 | 97.67% |
| N 対 CPU | 0.00328 | 97.67% |
| 4 デバイス対 CPU | 0.00420（基準 ×1.23） | 98.34% |
| 全 expert を CPU に置いた SGLX（floor）対 CPU | 0.00396 | 98.01% |
| 全 CPU の SGLX 対 `-ncmoe 39` | 0.00232 | 98.34% |
| 全 CPU の SGLX 対 N | 0.00250 | 97.01% |
| N 対 4 デバイス（自己整合） | 0.00218 | 97.34% |
| `-ncmoe 39` 対 N | 0.00216 | 97.34% |

- **floor の切り分け**: expert を全部 CPU（ggml-cpu と bit 一致のカーネル）で計算しても、attention などが RX の Vulkan で動くので CPU 基準との KLD は 0.004 になる。つまり 0.003〜0.004 は GPU 対 CPU の丸めの差の床で、SGLX の expert の誤りではない。
- **自己整合 < 0.001 は不合格**（0.0022）。ただし「全 CPU の SGLX 対 `-ncmoe 39`」（0.0023）、「全 CPU 対 N」（0.0025）も同じ大きさで、正当な構成同士でも GPU の演算が入ると 0.002 程度は離れる。この条件は GPU 構成同士には厳しすぎる（Vulkan の `mul_mat_id` の丸めの差）。デバイスの異常とは言えないが、条件は満たしていない。
- decode の 4 デバイスは基準 ×1.23 で ×1.2 をわずかに超えた（301 位置では誤差内の可能性が高いが、未確認）。

**注意**: 乱数トークンを流した `hec_gen` のダンプ比較で LFM2.5-8B-A1B の KLD を測ると 0.49 / top-1 約 40% になる（通常の Vulkan 層分割でも同じ）。分布外の入力なので KLD の根拠にしない。LFM の KLD は下記のとおり実テキストで測り直した。

### 4. LFM2.5-8B-A1B（退行と一般性）

- **構成 H の退行なし**（`logs/lfm_H_*`、`GGML_SGLX_SPLIT=Vulkan0:0.78,…`、新しいプラグイン）: tg128 68.9 / 69.7、pg512,128 239.1 / 240.1、pp512 704〜706。記録の tg 69 / pg 240 と同じ。C1（Vulkan0/Vulkan1 の層分割）は tg 51.8、pg 197.0。
- **MoE モード**（`-ot ...=SGLX0`、`Vulkan0:auto`）: pp512 241、tg128 46.7（`logs/lfmmoe_bench`）。H（68.9）より遅い。LFM は 1 トークン 4 expert・32 expert の小さい MoE で、MoE モードは 30B 用に最適化した。
- **KLD**（`ub320-fix` の手順どおり: `kl/base.kld`、`-c 512 --chunks 4 -b 512 -ub 320`、`logs/kld3_lfm_*`）: C1 0.02075 / 93.73%（既存値 0.0215 / 92.8% を再現）、MoE モード（RX のみ）0.01975 / 94.12%、MoE モード（RX + GT 730）0.02015 / 93.43%。基準以下で、回帰なし。

### 5. llama.cpp `moe_cache`（`23b0202`）

`src/llama-moe-cache.cpp`（実験的、`--moe-cache-size`）。
- **仕組み**: デバイスごとに予算を持ち、GPU 側の「バンク」に expert を置き、層の group ごとの LRU でスロットを割り当てる。選択された expert が無ければ、その層の計算の前に upload（連続するスロットは束ねる）して GPU で計算する。大きな batch はキャッシュを読むだけで、更新しない。
- **重なる点**: GPU 側に固定サイズのスロット（expert 単位）、ホスト側の mmap、スロットの対応表、連続 upload の結合。
- **違う点**: moe_cache は全部 GPU で計算する（コールドは upload を待つ）。CPU が計算に加わらず、ミスは同期の upload になる。動的な LRU なので RX の転送帯域（1.2〜1.6 GB/s）がミスのたびに律速になる。SGLX は配置を静的にし、コールドを CPU で並行して計算する。複数デバイスにまたがる構成もできる。
- **測定**（RX 単独、`-ncmoe 48`）: tg128 は 2800 / 3000 / 3300 MiB で 3.83 / 4.12 / 2.11、3000 MiB で pp512 16.4、pg 10.8（N の約 1/3）。3300 MiB で落ちるのは VRAM の不足。自然文でも 3.4〜7.1 で、基準（7.4）を下回る。**baseline にならない**。

### 6. GT 430 のミディアム（`logs/g430_*`、RPC0 = ggml-rpc-server 経由、`hec_gen --bench pp256,tg128`）

| 構成 | pp256 | tg128 |
| --- | --- | --- |
| GT 430 無し（`Vulkan0:auto:250`、CL11 のデバイス無し） | 43.4 | 8.98 ± 1.97 |
| `Vulkan0:auto:250,RPC0:300`（常に使う） | 25.1 | 6.92 |
| `Vulkan0:auto:250,RPC0:300!`（遅延ゲート付き） | 24.3 | 7.66 |

**GT 430 を足しても速くならず、むしろ遅くなった。** 常に使うと tg が 23% 落ち、pp256 は 42% 落ちた。ゲート付きでも tg 7.66、pp256 24.3 で、ゲート無しより少し良いだけ。GT 430 は別プロセスの ggml-rpc-server で（`cl11_filter_open_conflicts`: NVIDIA ドライバが世代の違う GPU を同じプロセスに入れられない）、層ごとの遅延が 2.4〜3.3 ms で、設計書の損益分岐（0.3〜1.5 ms）に届かない。ゲートが prefill でも担当を残した理由は未調査。比較元の「GT 430 無し」はばらつきが大きく（標準偏差 1.97）、CL11 の GT 730 も入っていないので、N とは条件が違う。差の大きさは参考値。

### 7. 構成の選択（decode tg128、乱数トークン、`logs/s8_*` `s9_*` `s10_*`）

- 構成 N（RX + GT 730）14.63（s8）。RX のみ（Q4_1 スロット 500）13.12。RX + WX 3.36。RX + GT 730 + WX + GT 710 4.21。WX が入ると大きく遅くなる（原因は仮説のみ。「実測で分かったこと」の 10）。GT 710 だけを足した構成は測っていない。
- spin 0 / 100 / 500 µs で tg 13.75 / 12.09 / 10.75。スレッド 1 と poll 50 で 9.56。event（`GGML_SGLX_MOE_EVENT`）有り 13.21、無し 13.51 で、差は誤差内。既定は無し。
- Q4_1 のスロット数: `auto:250` の RX のみ 13.75（s9）、500 で 13.12（s8）。250 と 500 の差は 1 回ずつの測定で、違いは誤差かもしれない。
- CUDA の pdev は作っていない。CL11 の `MUL_MAT_ID`（1 token）が 0.313 ms、CUDA が 0.476 ms で、CL11 のほうが速かったため。

### 8. 残件・未検証

- 基準側は 1〜2 回で、`REPS 3〜5` に届いていない（N は 6 回）。
- 短い prefill（数十〜百数十トークン）は、N が基準より遅い。ring に流す閾値 `STREAM_TAU`（既定 3）の調整は未実施。
- decode の自己整合 < 0.001 は GPU 同士では満たせない（床 0.002）。4 デバイスの decode KLD は基準 ×1.23。
- WX 2100 が decode を遅くする原因と、GT 430 のゲートが prefill でも担当を残す原因は、仮説のみ。
- GT 710 単独、CUDA の pdev、MoE モードの PCIe 帯域の詳細な内訳は測っていない。
- Mac のチェック（`scripts/mac_check.sh`）: `MUL_MAT_ID` 945/945、`MUL_MAT` と `GET_ROWS` 1610/1610 合格、MoE ブロックの照合 all OK、tiny モデルの llama.cpp 通し（Metal スロット）は KLD 0.003 前後で CPU のみは 0.00013。


## 実装の構成

llama.cpp は無改造（`23b0202`）。SGLX0（`libggml-cl11.so`、`adapters/ggml/composite/`）が MoE の expert を引き受ける。

起動の形:

```
GGML_BACKEND_PATH=<plugin> GGML_CL11_PLATFORMS=NVIDIA GGML_CL11_DEVICES="GT 730,GT 710" GGML_CL11_AUTO_RPC=0 \
GGML_SGLX_MOE=Vulkan0:auto:250 \
hec_gen / llama-bench / llama-completion -m Qwen3-30B-A3B-Q4_0.gguf -dev Vulkan0/SGLX0 -ts 1/0 -ngl 99 \
    -ot 'blk\.[0-9]+\.ffn_(up|gate|down)_exps\.weight=SGLX0' --poll 0
```

| 部品 | ファイル | 内容 |
| --- | --- | --- |
| SGLX0 の mmap バッファ | `composite_device.cpp` | `buffer_from_host_ptr`（モデルファイルの mmap をそのまま expert の置き場にする。コピーしない）、`mmap_support`、`init_tensor` で `blk.N.ffn_{up,gate,down}_exps` を登録、MoE のとき SGLX0 の計算バッファを GPU 0 の pinned メモリに置く（sched の入力コピーを 1 回の非同期読み出しに束ねる）、`MADV_POPULATE_READ` で mmap を先に張る |
| MoE ブロック | `moe_block.{h,cpp}` | `supports_op`（`MUL_MAT_ID` と、その `SWIGLU` / `MUL` / `ADD` 連鎖）、up・gate・SWIGLU・down を 1 つの job として実行、(token, slot) の組ごとに「どのデバイスのスロットか / CPU か」を引いて実行、統計（指数減衰の選択回数）、配置（静的）、スロットの確保、prefill の経路 |
| スロット API | `pdev.h`, `pdev_ggml.cpp` | `moe_create / moe_upload / moe_submit / moe_prefill / moe_reserve / moe_zero_slot`。ggml のデバイス（Vulkan0/1、CL11_0/1）を ggml のグラフ（`MUL_MAT_ID` ×3 + `SWIGLU`）で使う。CUDA の pdev には MoE を作っていない（下記） |
| CPU のコールド | `moe_cpu.{h,cpp}` | ggml-cpu の `vec_dot`（`ggml_get_type_traits_cpu`）を直接呼ぶ mat-vec。グラフも OpenMP も使わない |

### decode（N = 1）

1. router（RX の llama 側）の ids / 重み / x を、SGLX0 の graph_compute の先頭で 3 つまとめて非同期に読む（pinned）。
2. 8 個の expert を、スロットを持つデバイスごとに分ける。デバイスごとに 1 回 `moe_submit`（x を送り、スロット番号で `MUL_MAT_ID`×3 + SWIGLU を 1 グラフで実行、expert ごとの出力 `[n_embd, n]` を読み戻す）。ワーカースレッドを持つデバイス（WX、CL11）は別スレッドで。
3. GPU に投げたあと、スロットに無い expert（コールド）を CPU で計算（呼び出しスレッド + ヘルパー 1 本、動的チャンク）。
4. GPU を待ち、`y[:, k]` に置く（重みはまだ掛けない）。そのあとの `MUL`（重み）と `ADD` 連鎖は ggml のグラフどおりの順序でホストが実行する。

### prefill（N > 1）

1. ストリーミングデバイス（`GGML_SGLX_MOE` の先頭、RX）の **ring**（既定 96 スロット + 0 埋めスロット 8 個）に、スロットに無く、ubatch の中で 4 組以上の token が選んでいる expert を、mmap から pinned のステージングに memcpy → ブロッキングの DMA で入れる（ステージングと DMA は別スレッドで重ねる）。
2. 同じデバイスの **1 つのグラフ** で、そのデバイスにある・ring に入れた expert の全部の組を計算し、routing weight を掛けて k の順に足し、`[n_embd, tokens]` の和だけを返す（llama.cpp の `build_moe_ffn` と同じ形。ペアごとの出力 8 倍を読み戻さない）。デバイスに無い組は 0 埋めスロットの expert を指し、重み 0。
3. ほかのデバイス（WX、CL11）と CPU（3 組以下の expert）の組は decode と同じペア単位の API で計算し、ホストが重みを掛けて k の順に足し込む。
4. prefill の最後の層で、数えた頻度から **配置**（static prefill 配置）を決めてスロットに入れる。次の prefill の開始時に ring に置いた expert は外す。

### 配置

- `freq[層][expert]`: 選択回数（prefill でも decode でも数える。半減期 4096 token の指数減衰）。
- 全層の候補を `freq × クラスの CPU 時間比`（Q4_1 の down を持つ層は ggml-cpu で 2.2 倍遅いので 2.2）の降順に並べ、`GGML_SGLX_MOE` の並び（RX → … ）で空いているスロットに詰める。`GGML_SGLX_MOE_MIN_MASS` で、RX 以外のデバイスが 1 つの層の選択のごく一部しか持たない場合に、その層への割り当てを取り下げる。
- スロットは「クラス」（gate / up / down の型と形）ごとの配列。Qwen3-30B-A3B の Q4_0 ファイルは層 0〜5 の down が Q4_1 なので、クラスが 2 つある（Q4_1 側のスロット数は `GGML_SGLX_MOE=Vulkan0:auto:250` の 3 つ目の数）。
- 変更分だけを各デバイスの別スレッドで upload（RX 1000 個で約 2 秒）。

### 環境変数（既定値）

| 変数 | 意味 |
| --- | --- |
| `GGML_SGLX_MOE=dev[:slots[:q41slots]],…` | MoE モードの ON。デバイスは ggml のデバイス名（`Vulkan0`、`Vulkan1`、`CL11_0`、`CL11_1`）。`slots` = `auto`（空きメモリから）または個数。`1` だけなら全部 CPU |
| `GGML_SGLX_MOE_RING` (96) | ストリーミング用 ring のスロット数（先頭デバイス） |
| `GGML_SGLX_MOE_STREAM_TAU` (3) | これ以下の組数の expert は ring に流さず CPU |
| `GGML_SGLX_MOE_THREADS` (2) | コールドの CPU スレッド数（呼び出しスレッドを含む） |
| `GGML_SGLX_MOE_SPIN_US` (0) | ヘルパーが次の job を待って回る時間 |
| `GGML_SGLX_MOE_HALFLIFE` (4096) | 頻度の半減期（token） |
| `GGML_SGLX_MOE_FREEZE=1` | 最初の配置のあとは入れ替えない |
| `GGML_SGLX_MOE_MIN_MASS` (0) | 上記 |
| `GGML_SGLX_MOE_FUSED=0` | prefill をペア単位の旧経路にする |
| `GGML_SGLX_MOE_CPUKERN=0` / `PINNED=0` / `PREFAULT=0` | 各最適化を切る |
| `GGML_SGLX_PROFILE=1`, `GGML_SGLX_MOE_VERBOSE=1` | 内訳を stderr に出す |

## 実測で分かったこと（設計書との差）

1. **ggml の OpenMP は 2 コアの旧 CPU で 10 ms/呼び出し**（`logs/s2_tiny_cold_*`）。プロセスに他に忙しいスレッドがあるとき、グラフのバリアで待たされる。tiny モデルの 8 expert が 9.5 ms（1 スレッド 0.21 ms、`GOMP_SPINCOUNT=0` 0.46 ms）。コールドの CPU 計算は ggml のグラフではなく、ggml-cpu の `vec_dot` を直接呼ぶ自前の mat-vec（`moe_cpu.cpp`、ヘルパー 1 本、動的チャンク）にした。出力は ggml-cpu のグラフと bit 一致（NMSE 0）。
2. **ヘルパースレッドを回して待たせると逆効果**。次の job を待って回る時間（`GGML_SGLX_MOE_SPIN_US`）を 500 / 100 / 0 µs にすると tg が 10.75 / 12.09 / 13.75。回っている間は、Vulkan の submit・fence の待ちとカーネルのドライバが CPU を取り合う（GPU 待ちが 0.53 → 0.17 ms）。既定 0。
3. **llama.cpp の thread pool は CPU 1 コアを持っていく**。llama-bench の既定 `--poll 50` だと、各 token の CPU グラフ（embedding など）のあとにヘルパーが約 0.3 秒回り続ける。SGLX0 で CPU を使う構成では `--poll 0` を勧める。
4. **mmap の初回アクセスは minor fault が expert 1 個（2.65 MB）で約 0.5 ms**。コールドの expert は稀にしか選ばれないので、毎回が初回アクセスに近い。mmap を張るとき `MADV_POPULATE_READ`（15.8 GiB が 0.5 秒、ページキャッシュにあるとき）。
5. **RX 6400 の転送と計算は並列に進まない**（`hec_bench stream`、`logs/s3_stream_c12`）。pinned への memcpy は 7.3 GB/s、RX への DMA は 1.4〜1.65 GB/s（1 expert 1.6〜1.9 ms）。ring に流しながらペアごとに計算すると、DMA スレッドの時間が純粋な DMA（112 ms/層）より 100 ms 長くなり、計算（101 ms/層）とほぼ直列に足されていた。
6. **ペアごとの出力をホストへ戻すのが高い**。prefill の 1 層 4096 組の出力は 33 MB で、RX から読むのに 21.7 ms（`hec_bench pairs`、`logs/s5_pairs_n512`）。重みを掛けて k の順に足す部分までデバイスの 1 グラフでやると、戻すのは `[n_embd, tokens]` の 4 MB（2.8 ms）で、計算を含めて 1 層 21〜23 ms。
7. **sched の入力コピーが層ごとに 3 回の同期読み出し**（x、ids、重み）で 1.23 ms/層。SGLX0 の計算バッファを GPU 0 の pinned メモリに置くと、3 つを 1 回の非同期 submit にでき、0.33 ms/層。
8. **Metal の `mul_mat_id` は 1 つの expert に多くの token が集まると間違う**（Mac で見つけた。prefill の「このデバイスに無い組」を 1 つの 0 埋めスロットに集めたとき）。0 埋めスロットを 8 個にして組を分散させて回避した（Vulkan では問題ない）。
9. **ggml-vulkan は小さなグラフの submit を synchronize まで遅らせる**。`ggml_backend_event_record` で投入を前倒しできるか（`GGML_SGLX_MOE_EVENT`、`logs/s10_*`）。
10. **WX 2100 を足すと decode が大きく遅くなる**（`logs/s8_*`、tg128: RX のみ 13.1〜13.8 → RX+WX 3.36、RX+GT 730+WX+GT 710 4.21。GT 730 だけ足した構成は 14.63）。原因は**仮説**で、切り分けの計測はしていない。
    - (a) WX は別プロセスではなく同じプロセスの 2 本目の Vulkan デバイスで、decode のたびに専用ワーカースレッドへ job を渡し（submit、フェンス待ち、読み戻し）、層ごとに待つ。tb250 の CPU は 2 コアで、ワーカー、呼び出しスレッド、コールドの CPU ヘルパー、llama のスレッドが取り合う。RX の 1 層あたりの遅延（GPU 待ち 0.17 ms）が 0.2 ms 増えるだけで、48 層で約 10 ms/token になる。
    - (b) 2 GB の WX にスロットを置くと RX の `moe_reserve` とは別に ggml のグラフ用バッファを確保し、PCIe の帯域と upload の待ちが増える。
    - (c) RADV の 2 本目のデバイスで、小さなグラフの submit が synchronize まで遅れる（9. の件）。層ごとに 1 回ずつ、遅延が直列に足される。
    検証案: `GGML_SGLX_MOE_VERBOSE=1` の層ごとの待ち時間の内訳を WX だけ有り／無しで比べる、WX を `NAME:slots!`（遅延が CPU の見込み時間を超えるときだけ使うゲート付き）で入れる、`GGML_SGLX_MOE_MIN_MASS` で WX の担当を狭める。ゲート付きや MIN_MASS で入れた場合の計測は未実施。
