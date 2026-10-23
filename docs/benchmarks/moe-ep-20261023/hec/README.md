# Qwen3-30B-A3B ホット/ミディアム/コールド expert キャッシュ（SGLX0 の MoE モード）の実装と実測

実装 2026-10-23、tb250（Celeron G3930 2C AVX 無し、RAM 31 GB、RX 6400 4 GB / WX 2100 2 GB 公式 Vulkan RADV、GT 730 1 GB / GT 710 2 GB CL11、GT 430）。llama.cpp `23b0202`（無改造）、`Qwen3-30B-A3B-Q4_0.gguf`（unsloth、層 0〜5 の down は Q4_1）。設計は `docs/plans/HOT_EXPERT_CACHE.md`、前日の判定は `../hec-day1/README.md`。ブランチ `feat/moe-expert-parallel`。
数値はすべて `logs/` の実測（`scripts/` のキュー）。**【推定】** と書いたものだけが実測でない。perf は `sgl-driver-switch perf pin`（キューの先頭で pin、終了時に auto。ログの先頭の `# perf:` に記録）。測定は `~/bench/gpu-bench.lock` の flock + 温度待ち（`adapters/ggml/tools/hec_run.sh`）で 1 本ずつ。

RESULTS_PLACEHOLDER

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
