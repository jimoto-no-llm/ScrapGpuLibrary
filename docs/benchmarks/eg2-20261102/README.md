# EmbeddingGemma 2（Text 270M）を TB250 の各 GPU に載せる

測定日 2026-11-02〜03、TB250（Celeron G3930、RAM 32GB、Debian 13、NVIDIA 390.157、Mesa 25.0.7）。llama.cpp `23b0202a1`（ggml 0.26.0）、SGL `feat/embeddinggemma2`。数値はすべて `logs/` の実測。**推定は【推定】と書いたものだけ**。

## 結論

1. **ライセンス・ゲートは問題なし**（下記）。重みは取得済み。
2. **最大の効き目は「GPU で全 op を回せるようにしたこと」**。v2 のグラフは FFN が融合 op `GEGLU`、`per_layer_model_proj` が BF16 で、どちらも CL11 は未対応だった。1 層ごとに CPU へ落ち、GT 730 で **1.03 embeddings/s**（`logs/b_gt730_Q8_0_L128_B1_prefix_no_geglu.log` に未対応 op の一覧）。CL11 に `GEGLU` / `GELU` / `NORM` を足し、`per_layer_model_proj` を F16 にして **5.48 embeddings/s（5.3 倍）**。ほかに CL11 の mat-mat 候補・タイル幅を振ったが、既定を超えるものは無かった。
3. **1 枚で最速は RX 6400 の公式 Vulkan**（L=128 で 42〜53、L=32 で 202 embeddings/s）。CL11（rusticl）の 2〜4 倍速い。WX 2100 も公式 Vulkan が CL11 の Clover / rusticl より速い。GT 730 / 710 / 430 は CL11（NVIDIA 390）。CPU（2 コア）は L=128 で 0.46〜0.59 なので、どの GPU も CPU の 4.6〜115 倍（L=128、各経路の最良値）。
4. **正しさ**: 8 経路 × 4 重み（Q8_0 / Q4_0 / Q4_K_M / F16）すべてで、同じ重みの CPU との平均コサイン類似度 ≥ 0.99977（合格基準 0.999）。変換の正しさは HF の fp32 とも照合（0.999999、1500 トークン超の文を含む 1.000000）。
5. **組み合わせ（データ並列、1 カード 1 プロセス）は単体最速を 47% 上回る**: L=128 は RX 6400 単独 43.4 → RX 6400 + WX 2100 + GT 730 で **63.9**、L=32（B=16）は 169.6 → 252〜268、L=512 は 10.9 → 16.0。**最良の部分集合は RX 6400 + WX 2100 + GT 730 の 3 枚**。GT 710 / GT 430 を足しても伸びない（Celeron 2 コアのディスパッチが律速で、RX 6400 が 43 → 38 に落ちる）。コア固定（RX 6400 を core 0、他を core 1）で全 5 枚の L=32 は 241 → 268。
6. バッチ（1 回の decode に複数の文）は **短い文（L=32）だけ効く**（B=16 で 2〜3 倍）。L ≥ 128 では B=1 かごく小さい B が最速で、大きい B は遅くなる。llama.cpp の非因果の注意は、バッチ全体のトークン数 T に対して T×T のスコアを作ってマスクするため（文の数 B に対して 2 乗で無駄が増える）。

## ライセンスとゲートの確認（重みを取る前、2026-11-02）

| 項目 | 結果 |
| --- | --- |
| `https://huggingface.co/api/models/google/embeddinggemma-2` | `gated: false`、`private: false`、`license:apache-2.0`、revision `f9972ac22e696a105c482202646979e33e4b5a1e` |
| モデルカード（README.md） | `license: apache-2.0`。本文の License 欄のリンク先は `https://ai.google.dev/gemma/docs/gemma_4_license` |
| リンク先のページ | 題名は「Apache License 2.0」（Gemma 4 の文書名だが内容は Apache-2.0）。**食い違いは文書名だけで、条件は同じ** |
| 同意操作 | 不要（`resolve/` が認証なしで 307 を返し、HF トークンは使っていない） |
| 注意 | カードに「配備は Gemma Prohibited Use Policy に従うこと」とある（利用禁止ポリシーへの言及で、ダウンロードの条件ではない） |

## 重み

`google/embeddinggemma-2` revision `f9972ac2…`、`model.safetensors` 1,488,915,288 バイト、SHA256 `197a32965d4b1105faf060417baa899e193fb73cd401f42ec9295234d5553d79`（HF の LFS oid と一致）。Text のみ（Vision / Audio は変換しない）。llama.cpp `23b0202a1` の `convert_hf_to_gguf.py` で変換。置き場は Mac の `~/dev/sandbox/eg2/`、TB250 の `~/moe-ep/eg2/models/`（共有正本ではない）。

| ファイル | 内容 | SHA256 |
| --- | --- | --- |
| `eg2-bf16.gguf` | `--outtype bf16`（CPU 基準。CL11 は BF16 重みに未対応） | `9b8a84bb0944c35476bb3e0df4ef0c2ec6d65f2ca69eed15ee3feeafd67e9ac3` |
| `eg2-F16.gguf` | `--outtype f16`（`per_layer_model_proj` も F16） | `41da4f5b6d3d37ef0e553060435359506eb4f9086b8156e354d823adccd990fc` |
| `eg2-Q8_0.gguf` | F16 から `llama-quantize` | `1b0663558c93b57b44f4c916cdfd12f4673fb854e0bfbc52f3deb054d3cc2916` |
| `eg2-Q4_0.gguf` | 同上（output は Q6_K になる） | `2912aff109a3fc7fe033e4c284a34fef3c296ae65470f61bc3cb94712aecfe34` |
| `eg2-Q4_K_M.gguf` | 同上 | `ecdad70d82b814c9e413aa8470bd7f78ea0aee6117a9f979b19d39e8ee62617f` |
| `eg2-Q6_K.gguf` | 同上（再現性のため記録。測定には使っていない） | `40e29902a8874c003faac502a148c8d0ab7b5424d0521b2e8318d28206582c14` |

- **`per_layer_model_proj` は `llama-quantize` が量子化せず BF16 のまま残す**（`llama-quant.cpp:316`）。`--tensor-type` でも変えられない。CL11 は BF16 の MUL_MAT を持たず CPU に落ちるので、**量子化版は BF16 ではなく F16 変換のファイルから作った**。BF16 → F16 は仮数が増える向きで、範囲外は極小値の下位だけ。BF16 の値との差は F16 のコサイン類似度（下表）に出ている（0.99998 以上）。
- `llama-quantize` は TB250 の `VK_ICD_FILENAMES` 無し環境で `stack smashing` で落ちる（NVIDIA 390 の Vulkan ICD を読むため、`docs/HARDWARE.md` の既知の癖）。量子化は Mac で実行。TB250 の測定スクリプトは `env.sh` で `VK_ICD_FILENAMES` を指定している。

## 正しさ

### 手順

- 基準: **CPU（TB250、SSE4.2 ビルド、2 スレッド）で計算した BF16 の埋め込み**。重みの違う量子化版は **同じ重みの CPU** と比べる（合格: 平均コサイン類似度 ≥ 0.999）。BF16 との差は別枠で記録。
- 文: `task/texts_corr.txt` 46 件（日英混在の短文・段落 44 件 + 1500 トークンに切った長文 2 件）。長文は **SWA の窓（片側 512、対称で 1024）を超える**ので、対称窓のマスクと、局所層（head 256、rope 10000）／大域層（head 512、rope 1e6）の違いを通る。短い文だけでは窓に当たらない。
- ツール: `tools/eg2bench/eg2bench.cpp`（libllama を直接使う。`--dump` で平均プーリング後・出力射影後の float32、`--bench L B reps` で embeddings/s）。比較は `scripts/cmp_emb.py`（行ごとに L2 正規化、純 Python）。
- 出力 `-fa off`（`FLASH_ATTN_EXT` 無し、`SOFT_MAX` 経路）。

### 変換の検証（Mac M2 Max CPU、`eg2-bf16.gguf`）

| 比較 | 平均 cos | 最小 |
| --- | --- | --- |
| llama.cpp BF16 対 HF sentence-transformers fp32（検索タスクの旧版 294 文。節見出しと本文。現在の `task/` は組み直したので無い） | 0.999999 | 0.999992 |
| 同、`task/texts_long.txt` の 9 文（最長 9114 トークン、SWA 窓を大きく超える） | 1.000000 | 1.000000 |

### 8 経路 × 4 重み（`logs/corr.tsv`）

同じ重みの CPU との平均 cos の最小値（8 経路のうち最悪）。**全 32 件が合格**。

| 重み | 同じ重みの CPU との平均 cos（最悪の経路） | 文ごとの最小 | BF16 の CPU との平均 cos |
| --- | --- | --- | --- |
| Q8_0 | 0.999918 | 0.999631 | 0.99984〜0.99992 |
| Q4_0 | 0.999907 | 0.999705 | 0.9847〜0.9848（別枠） |
| Q4_K_M | 0.999769 | 0.999176 | 0.9914〜0.9915（別枠） |
| F16 | 0.999981 | 0.999928 | 0.99998〜1.00000 |

経路: GT 730 / GT 710 / GT 430（CL11）、RX 6400 / WX 2100（公式 Vulkan）、RX 6400（rusticl）、WX 2100（rusticl）、WX 2100（Clover）。RX 6400 の Clover は GPU リセットを起こすので使わない。

### `test-backend-ops`（CL11 に足した op）

| 検査 | 結果 |
| --- | --- |
| Mac（M2 Max）全件、既定 / `FORCE_GEN=kepler` / `fermi` | 4176 / 4176 を 3 回（`logs/macfull_*.log.gz`） |
| `GELU,GELU_ERF,GELU_QUICK,GEGLU,GEGLU_ERF,GEGLU_QUICK,NORM,UNARY`（`logs/tbo_*`） | GT 430 / GT 730 / GT 710 は既定・kepler・fermi の 3 調整、RX 6400 rusticl、WX 2100 Clover・rusticl とも **79 / 79** |
| TB250 全件（`logs/tbofull_*.gz`） | GT 730 と GT 430: 4214 / 4215（落ちる 1 件は VRAM に入らない 1 GiB の `MUL_MAT_HADAMARD`、`docs/BASELINES.md` に既知）。RX 6400 rusticl: 4215 / 4215 |
| `ctest`（Mac、CL11 + remote 有効） | 18 / 18 |

`mvm_check` / `attn_check` / `mid_check` は今回の変更が触れない経路（GLU の種類と NORM / GELU の追加のみ）なので実行していない。

## 実装（SGL）

| 変更 | 内容 |
| --- | --- |
| `feat(cl11): add NORM, GELU and GEGLU` | `feat/cl11-norm-gelu`（90d3684）の NORM と GELU / GELU_ERF / GELU_QUICK を最新 main に載せ直し（衝突なし）、`GEGLU` / `GEGLU_ERF` / `GEGLU_QUICK`（GLU の種類）を追加。v2 の FFN は `ggml_geglu_split` なので GELU だけでは各層が CPU に落ちる。core・ref・CL11 カーネル・ggml アダプタ・テスト・docs |
| `feat(tools): eg2bench` | libllama で embeddings/s と埋め込みダンプを取るツール（`--duration` / `--start-at` で複数プロセスを同じ時間窓で測れる） |
| `docs(bench): EmbeddingGemma 2` | この文書とスクリプト、ログ、小さい検索タスク |

## 経路ごとの最良値（embeddings/s、`perf pin`、重み Q8_0 / Q4_0）

`scripts/bench_card.sh`、`flock ~/bench/gpu-bench.lock`、開始時 85 ℃未満（実際は 37〜54 ℃、組み合わせは 31〜47 ℃）、1 本ずつ、`eg2bench --bench L B reps`（warm 2、reps は文の長さと B で 2〜30）、`-ngl 99 -t 2`、`-fa off`。表は平均（個々の標準偏差は `logs/b_*_mx.out` と `scripts/summarize.py` の出力、ほとんど 1% 未満。**RX 6400 の Vulkan は B=1 で ±10%**）。文はランダムなトークン ID（固定シード、先頭は BOS）の固定長。「best」は測った B の最大（B ∈ {1, 2, 4, 8, 16, 32, 64} のうち L ごとに測った点）。

| 経路 | 重み | L32 B=1 | L32 best | L128 B=1 | L128 best | L512 B=1 | L512 best |
|---|---|---:|---:|---:|---:|---:|---:|
| **RX 6400 公式 Vulkan** | Q4_0 | 72.20 | **202.23** (B=16) | 42.35 | **52.78** (B=4) | 13.17 | **13.17** |
| RX 6400 公式 Vulkan | Q8_0 | 60.18 | 182.72 (B=16) | 35.17 | 43.98 (B=8) | 11.81 | 11.81 |
| RX 6400 CL11 rusticl | Q4_0 | 14.35 | 92.20 (B=16) | 13.49 | 23.22 (B=8) | 5.86 | 5.93 (B=2) |
| RX 6400 CL11 rusticl | Q8_0 | 14.84 | 94.35 (B=16) | 12.90 | 23.23 (B=4) | 5.88 | 5.88 |
| **WX 2100 公式 Vulkan** | Q4_0 | 19.88 | **44.03** (B=16) | 10.20 | **11.81** (B=2) | 2.75 | **2.75** |
| WX 2100 公式 Vulkan | Q8_0 | 20.29 | 44.14 (B=16) | 10.21 | 10.21 | 2.77 | 2.77 |
| WX 2100 CL11 rusticl | Q8_0 | 7.61 | 36.02 (B=16) | 5.91 | 9.00 (B=4) | 2.31 | 2.31 |
| WX 2100 CL11 Clover | Q8_0 | 6.21 | 29.26 (B=16) | 4.74 | 6.60 (B=8) | 1.86 | 1.86 |
| **GT 730 CL11** | Q8_0 | 11.97 | **19.51** (B=16) | 5.48 | **5.48** | 1.22 | **1.22** |
| GT 730 CL11 | Q4_0 | 11.09 | 19.29 (B=16) | 5.14 | 5.14 | 1.20 | 1.20 |
| **GT 710 CL11** | Q8_0 | 6.31 | **9.58** (B=16) | 2.77 | **2.77** | 0.60 | **0.60** |
| **GT 430 CL11** | Q4_0 | 2.79 | **7.58** (B=16) | 2.12 | **2.12** | 0.47 | **0.47** |
| GT 430 CL11 | Q8_0 | 2.78 | 7.52 (B=16) | 2.11 | 2.11 | 0.47 | 0.47 |
| CPU（2 スレッド、基準） | Q4_0 / Q8_0 | 2.42 / 1.95 | 同 | 0.59 / 0.46 | 同 | 0.13 / 0.10 | 同 |

- 全データ（全 B・両方の重み）は `scripts/summarize.py logs mx` と `logs ex` の出力。CPU 基準（Q8_0、L=128、B=1）は各カードの前に交互に測った。ログは同じ名前で上書きされ、**残っているのは最後の 1 本だけ**（`logs/b_cpu_Q8_0_L128_B1_cpuref.out`、2160.63 ms）。キュー開始直後に確認した値は 2160.45 ms で、両端でドリフトは無い（途中の値は残っていない）。
- **カードごとの第一候補どおりの結果**: RX 6400 と WX 2100 は公式 Vulkan > CL11 rusticl > Clover。GT 730 / 710 / 430 は CL11 + GELU / GEGLU（自前 CUDA との比較は未実施、下の「残課題」）。
- 重みの選び方: NVIDIA（CL11）は Q8_0 が Q4_0 以上（GT 730 5.48 対 5.14）。Vulkan は Q4_0 が速い（RX 6400 42 対 35）。`Q4_K_M` も測ったが Q4_0 と同程度（`logs/b_*_Q4_K_M_L128_B1_base.out`）。F16 は一番遅い（RX 6400 Vulkan 31、GT 730 4.17）。
- 達成演算率【計算】: 行列積は 1 文あたり L=128 で 35.0 GFLOP（重みのある 2 次元テンソルの K×M の合計 136.7M MAC × 2 × L）+ 注意 1.9 GFLOP。GT 730 は 182 ms で約 200 GFLOPS（理論 733 の 28%、カーネル時間の 74% が `MUL_MAT q8_0 mm`、`logs/prof_gt730.log`）。RX 6400 Vulkan の L=512（約 170 GFLOP を 76 ms）は約 2.2 TFLOPS。

## 効いた最適化・効かなかった最適化

| 内容 | 結果 |
| --- | --- |
| CL11 に `GEGLU`（と GELU / NORM）、`per_layer_model_proj` を F16 に | **効いた**。GT 730 L=128: 1.03 → 5.48 embeddings/s（5.3 倍）。結果は変えない（CPU との cos は上の表） |
| mat-mat のタイル幅 `GGML_CL11_MM_BN` 4 / 8（NVIDIA 3 枚・rusticl・Clover） | 効かず。NVIDIA は ±1%、WX 2100 は ±2%。RX 6400 rusticl は 4 が 14.33、8 が 13.15 で +9%（AMD の既定は 4、標準偏差 1 程度なので有意とは言えない）。既定は変えない |
| Kepler の mat-mat 候補（`GGML_CL11_MM_VARIANT` x84 … x88g の 13 種、GT 730 / 710） | 効かず。既定相当の `x84` が最速（GT 730 5.47、GT 710 2.77）で、ほかは GT 730 が 1.97〜4.79、GT 710 が 0.94〜2.42。`GGML_CL11_MM_VARIANT=`（空文字）は既定と同じ値にならず 4.79 で、環境変数を設定しないほうを既定として扱った。`logs/b_gt7*_mm_*` |
| バッチ（1 decode に B 文） | L=32 だけ効く（B=16: GT 730 11.97 → 19.51、RX 6400 Vulkan 72 → 202）。B=32 以上は落ちる。L ≥ 128 は B=1〜4 が最速で、大きい B は遅い（GT 730 の L=512 B=4 は B=1 の 63%、RX 6400 Vulkan は 70%）。原因は非因果の注意が T×T を作ること（上記） |
| 層分割（1 プロセスで 2 枚、`-sm layer`） | **負け**。RX 6400 + WX 2100 の Vulkan: L=128 で 21.6（RX 6400 単独 42〜53）、L=32 B=16 で 95（単独 202）。Kepler 2 枚（GT 730 + GT 710）: 4.1（GT 730 単独 5.48）。1 文は層を順に通るため並列にならず、受け渡しが足される |
| SGLX（行分割の composite device） | **試していない**。MUL_MAT の行を複数カードに割る構成は、1 文が 1 層ずつ同期するたびに PCIe（Gen2 x1〜x4）経由の活性化の往復が入る。層分割が単体より遅かったこと、データ並列で 47% 伸びたことから、優先度を下げた。要検証 |
| コア固定（`taskset`）をデータ並列に | **効いた**（下の組み合わせ） |
| `perf pin` | 全測定で使用。比較（pin 無し）は取っていない |

## 量子化の別枠評価

BF16 の CPU 出力との差（`mac_*` は Mac M2 Max の CPU、`task/texts_retrieval.txt` の 156 文）と、小さい検索タスクの結果。**検索タスクは n=78 組で、標準誤差が約 ±5 ポイントあり、量子化間の差を分解できない**（下記）。

| 重み | サイズ | BF16 との平均 cos | 最小 cos | recall@1 | recall@5 | MRR |
| --- | --- | --- | --- | --- | --- | --- |
| BF16（基準） | 542 MB | 1 | 1 | 0.2949 | 0.7436 | 0.4970 |
| F16 | 542 MB | 1.000000 | 1.000000 | 0.2949 | 0.7436 | 0.4970 |
| Q8_0 | 301 MB | 0.999838 | 0.999658 | 0.2821 | 0.7564 | 0.4868 |
| Q6_K | 235 MB | 0.998660 | 0.997079 | 0.3077 | 0.7692 | 0.5050 |
| Q4_K_M | 173 MB | 0.989852 | 0.982349 | 0.3333 | 0.7564 | 0.5098 |
| Q4_0 | 170 MB | 0.981346 | 0.967608 | 0.3205 | 0.7564 | 0.4959 |

- 検索タスク（`scripts/make_task.py`）: このリポジトリの Markdown の節の前半 500 文字を文書、後半 300 文字を問い合わせとして、同じ節を探す。正解が曖昧な節が多く recall@1 は BF16 でも 0.29。差は全体で ±0.04（MRR）に収まり、有意な劣化は検出できない。**「検索精度が落ちない」とは言えず、「このタスクでは検出できなかった」まで**。大きな検索コーパスでの評価は未実施。
- コサイン類似度で見ると Q4_0 は BF16 から 0.019 離れる。Q8_0 は 0.0002。速度優先で Q4_0 を使うと埋め込みが変わる（同じコーパスの文書どうしは同じ重みで揃えること）。
- Matryoshka 次元削減は評価していない。

## 組み合わせ（データ並列）

モデルを各カードに丸ごと載せ、**1 カード 1 プロセス**（GT 430 も別プロセスなので Fermi と Kepler が同居できる）。全プロセスが同じ絶対時刻に始め、30 秒の同じ時間窓で処理した数の合計（`scripts/combo.sh`、`eg2bench --duration 30 --start-at`）。窓の開始のずれは 0.0〜0.01 秒。重みは全カード Q8_0（埋め込みを比較できるように揃える。経路ごとの最良の重みを使うと Vulkan が少し速くなる、`logs/c_*_best_*`）。1 プロセス 1 スレッド、`perf pin`、`flock`。B は各カードの最良値（L=32: 16、L=128: RX 6400 は 8（または 4）、他は 1、L=512: 1）。**ここでのデータ並列は「各カードが自分の速度で文を取る」ときの合計処理量**（静的に分けたのではなく、同じ窓に各プロセスが自分のペースで処理した数の和）。文を配る側（キュー）は作っていない。

### 部分集合ごとの合計（embeddings/s、Q8_0）

| 構成 | L=32 | L=128 | L=512 |
|---|---:|---:|---:|
| RX 6400 のみ（Vulkan） | 169.62 | 42.14 / 42.18 / 42.29（B=8、3 回）、43.36 / 43.29（B=4、2 回） | 10.86 |
| RX 6400 + WX 2100 | 220.89 | 52.69（B=8）、54.60（B=4） | 13.99 |
| **RX 6400 + WX 2100 + GT 730** | **252.23** | 56.59（B=8）、**63.91**（B=4） | **15.99** |
| 全 5 枚（+ GT 710 + GT 430） | 241.07 | 57.54 / 57.46 / 57.50（B=8、3 回）、61.63 / 61.84（B=4、2 回） | 15.31 |
| WX 2100 + GT 730 + GT 710 + GT 430（RX 6400 無し） | 76.12 | 19.65 | 4.77 |
| RX 6400 + GT 730（q8 / best） | - | 41.62（B=1）/ 48.88 | - |
| GT 730 + GT 710 | - | 8.24（B=1） | - |
| GT 730 + GT 710 + GT 430 | - | 10.27（B=1） | - |

（`scripts/summarize_combo.py logs`。L=128 で B=1 の他の構成も `logs/c_*_L128B1.txt`。）

- **単体最速（RX 6400）に対し、3 枚（RX 6400 + WX 2100 + GT 730）で +47%（L=128 B=4: 43.3 → 63.9、L=32: 169.6 → 252.2、L=512: 10.9 → 16.0）**。最良の部分集合は 3 枚で、**GT 710 と GT 430 を足すと L=32 と L=512 と L=128（B=4）で減る**（-4%、-4%、-4%）。
- 原因は CPU: 5 プロセスが Celeron 2 コアを取り合い、RX 6400 のプロセスの処理量が 43 → 38〜42（B=4 と B=8）に落ちる。GT 710（2.7）と GT 430（2.1）が増やす分より RX 6400 が失う分のほうが大きい。RX 6400 が無い 4 枚は 19.7（L=128）。
- **コア固定（`taskset`、RX 6400 を core 0、他を core 1）**: 全 5 枚 L=32: 241.1 → **268.1**（+11%）、L=128（B=8）: 57.5 → 62.3（+8%）。3 枚 L=32: 252.2 → 256.3（+2%）、L=128（B=8）: 56.6 → 59.1（+4%）。コア固定した 5 枚の L=32 が今回の最高値（268.1）。
- 繰り返し: RX 6400 単独の L=128（B=8）は 42.14 / 42.18 / 42.29、全 5 枚は 57.54 / 57.46 / 57.50 で、ばらつきは 0.3% 以下。一方 matrix の RX 6400（B=1）は ±10% 動く。
- RX 6400 を公式 Vulkan ではなく rusticl にしたり、WX 2100 を Clover にする組み合わせは測っていない（単体で Vulkan の 1/2〜1/4 以下なので）。

### 単体最速対 最良の部分集合

| 長さ | 単体最速（RX 6400、Q8_0） | 最良の構成 | 合計 | 比 |
|---|---:|---|---:|---:|
| L=32 | 169.62（B=16） | 全 5 枚、コア固定 | 268.06 | 1.58 |
| L=32 | 169.62 | RX 6400 + WX 2100 + GT 730 | 252.23（コア固定 256.25） | 1.49〜1.51 |
| L=128 | 43.36（B=4） | RX 6400（B=4）+ WX 2100 + GT 730 | 63.91 | 1.47 |
| L=512 | 10.86 | RX 6400 + WX 2100 + GT 730 | 15.99 | 1.47 |

## 残課題・未実施

- **自前 CUDA の MUL_MAT との比較は未実施**（GT 730 / 710 / 430）。`src/backends/cuda` の mat-mat は列 1〜8 までで、この用途（N=128〜2048、演算律速）の GEMM が無い。GT 730 の `MUL_MAT q8_0` は理論値の 28% 程度なので伸びしろはあるが、Kepler 向けの GEMM を PTX で書くのが前提。CL11 の既存の候補 13 種を試して伸びなかったことは上に記録。
- SGLX の行分割は未評価（上記）。
- 文を配るキュー（本物のデータ並列ランナー）は作っていない。表の合計は同時に走らせた窓の和。
- 非因果の注意がバッチで T×T になる問題の対策（文ごとにグラフを分ける、ブロック対角の注意）は llama.cpp 側の変更が要る。B>1 で L ≥ 128 が遅くなるのはこれ。
- 検索タスクが小さい（n=78）。量子化間の検索精度の差は未決。Matryoshka 次元削減は未評価。
- Vision / Audio、`perf auto` との比較、`mvm_check` / `attn_check` / `mid_check`、`GGML_CL11_PRINT_BUILD_OPTS` による同一性は未確認。
- RX 6400 の公式 Vulkan を B=1 で測ると ±10% 動く（`logs/b_rx6400-vk_*`）。ホストの CPU ガバナの影響の可能性があるが切り分けていない。

## 再現

TB250（`~/moe-ep/eg2/`、`scripts/` はこのディレクトリの `scripts/`）。

```sh
# llama.cpp 23b0202a1 の自前ビルド（build-llama.sh、Ninja、-DGGML_BACKEND_DL=ON -DGGML_VULKAN=ON、AVX 無し）
# プラグイン: build-plugin.sh（adapters/ggml を against llama-build）
# eg2bench: g++ -O2 -std=c++17 tools/eg2bench/eg2bench.cpp -I<llama>/include -I<llama>/ggml/include -L<build>/bin -lllama -lggml -lggml-base
scripts/cpu_ref.sh                       # CPU 基準のダンプ（bf16 / 量子化）
scripts/corr_card.sh gt730 "Q8_0 Q4_0 Q4_K_M F16"   # カードのダンプと CPU との比較（logs/corr.tsv）
scripts/pinned.sh scripts/q_matrix.sh    # 全経路 × 長さ × B（perf pin、終了時 perf auto）
scripts/combo.sh NAME 30 "rx6400-vk:Q8_0:128:4 wx2100-vk:Q8_0:128:1 gt730:Q8_0:128:1"   # データ並列
PINMAP="rx6400-vk=0 default=1" scripts/combo.sh ...                                      # コア固定
python3 scripts/summarize.py logs mx; python3 scripts/best_table.py logs; python3 scripts/summarize_combo.py logs
# Mac の全件: for g in "" kepler fermi; do GGML_CL11_FORCE_GEN=$g GGML_BACKEND_PATH=<plugin> test-backend-ops test -b CL11_0; done
```

TB250 に埋め込みダンプ（`~/moe-ep/eg2/dumps/`）と重み（`models/`、上の SHA256）が残っている。
