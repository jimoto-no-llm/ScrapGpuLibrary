# GT 430（REMOTE0）の job 遅延の分解と、速度への寄与

TB250、2026-10-23。LFM2.5-8B-A1B Q4_0、llama.cpp `23b0202`、SGL `feat/moe-expert-parallel`。作業場所 `~/moe-ep/g430b/`（`srv/` `plug/`、`scripts/` = このディレクトリの `scripts/`）。測定は `run.sh`（flock + 温度待ち）で 1 本ずつ、速度は `perf pin`（`sess2.sh`、終了時 `perf auto`）。数値は `logs/` の実測。**【推定】** と書いたものだけが実測でない。

## 結論

1. **「1 job が単独の 3 倍」は異常ではなかった**。前回の比較は 4608 行の単独測定（0.74 ms）対 SGLX 内の担当行数（`R0:0.07` = 8960 行、`R0:0.10` = 12800 行）で、行数が違った。GT 430 の q6_K mat-vec はメモリ帯域律速（約 12.5 GB/s）で、job 時間 ≈ 0.13 ms + 0.14 ms / 1000 行。SGLX 内（G、12800 行、バッチ 1 往復）の job を区間に分けると、カーネル + 読み戻し（server 側 `BUFFER_READ`）が 1.68 ms（p10 1.65 / p90 1.71）で、12800 行の帯域見積もり（21.5 MB / 12.5 GB/s = 1.72 ms）と一致する（`logs/q21_G_job_segments.txt`）。起床・ソケット・enqueue・write は合計でも 0.2 ms 前後（クライアント worker 起床 47 µs、server 起床 55 µs、write x 70 µs、enqueue 32 µs、応答の受け取り 14 µs。いずれも p50）。
2. **GT 430 のクロック状態は原因でない**。単独の `scrapgpu_remote_bench` で 12 ms の間隔を空けて job を投げても、カーネル + 読み戻しは 0.625 ms（間隔なし 0.626 ms）で変わらない（`logs/q20_opencl11_g0_h0.out` 対 `q20_opencl11_g12_h0.out`）。`nvidia-smi -q -d CLOCK,PERFORMANCE` は 390 ドライバでクロックが N/A、P-state は常に P0 としか出ない（読めないので上の比較で代えた）。
3. **間隔を空けると CPU 側が +0.45 ms**（write 0.075 → 0.31 ms、op_execute の enqueue 0.05 → 0.26 ms。CPU の governor は `powersave`、アイドルで 800 MHz）。ホストの CPU を 2 スレッドで回し続けると（実機では llama-bench が実際にそうなる）この分は消える（`q20_*_g12_h2`: 0.86 ms）。実機の trace では CPU 周波数は問題にならない（上の区間）。
4. **実機で効いていたのはスケジューリング**。`REMOTE0` を入れると、**ホストスレッドの `submit(REMOTE0)` が約半分の呼び出しで job の完了まで（p50 1.8 ms）戻らない**（`q22_G_mm_events.txt`: `sub_e/3` p50 1806 µs、p10 7 µs）。woken した worker・server がホストスレッドと同じ CPU でそれを押しのけ、server は OpenCL の blocking 待ちで CPU を握る（server 118 ms/s、`q23_G.sched`）。その間、他のデバイス（Vulkan0 など）の submit が平均 0.9 ms 遅れる。CPU 全体は飽和していない（llama-bench 65 %）ので、使用率には出ない。
5. **直し方（環境変数、既定は変えない）**: ホスト（`taskset -c 0`）と server + worker（`SCRAPGPU_SERVER_CPU=1`、`GGML_SGLX_REMOTE_CPU=1`）を別コアに分ける。または server + worker を `SCHED_BATCH`（`SCRAPGPU_SERVER_SCHED=batch`、`GGML_SGLX_REMOTE_SCHED=batch`、特権不要）。`GGML_SGLX_THREADED_LAST=1`（worker 付きの pdev を最後に submit）。どれも MUL_MAT を 2.79 → 2.64〜2.71 ms にした（`q24_*`、G、REPS=3）。**同じ変更は REMOTE0 を持たない H にも効く**（pin で +1 %、`q29` の Hpin 対 Hplain）。
6. **分割比が本当の問題だった**。前回の G（`V0:0.6,C0:0.25,C1:0.05,R0:0.1`）は GT 730（CL11_0）が 2.7 ms で最後に終わり、Vulkan0 は 2.0 ms で待っていた。Vulkan0 は 1 行あたりの時間が極端に小さい（autotune の alone 測定: 76800 行 1.64 ms、96096 行 1.87 ms → 約 0.012 ms / 1000 行、固定費約 0.7 ms）ので、行数を増やしても終わりがほとんど遅れない。比を Vulkan0 0.70〜0.78 に寄せると MUL_MAT 2.8 → 2.05〜2.11 ms、tg 65.5 → 68.6〜69.5（`q26`、`q27`）。
7. **それでも GT 430 は速度に寄与しない**（G 対 H、下表）。理由は上の数字から計算できる: Vulkan0 の 1 行あたりの時間が GT 430 の約 1/12 なので、GT 430 が担当する約 8 % の行（約 10000 行）を Vulkan0 が引き受けると +0.12 ms（0.012 ms / 1000 行 × 10000）。1 トークン約 14.5 ms のうち **0.8 % が上限**（実測の差 −0.4〜+0.1 %）。**GT 430 を速くしても（job の固定費 0.13 ms + カーネル）、上限は同じ桁**。

## 正しさ（`logs/q28_*`、`q30_*`、perf pin なし、`-b 512 -ub 320`）

| 検査 | 結果 |
| --- | --- |
| `test-backend-ops -b SGLX0 -o MUL_MAT,GET_ROWS`、`REMOTE0:1`（GT 430 単独、CL11 サーバー、trace 付きビルド） | **658/658** |
| 同、G（`V0:0.70,C0:0.17,C1:0.05,R0:0.08`） | 656/658。失敗 2 件は `q4_0 m=40 n=9 k=32` と `m=100 n=9 k=96` の Vulkan0 の NaN。**GT 430 を含まない H でも、Vulkan0 単独（`Vulkan0:1`、1635/1637）でも同じ 2 件**（`q30_tbo_H`、`q30_tbo_V0`）。前回は 1 件だけ報告していた（もう 1 件は同じ種類の既知の不具合） |
| KLD（CPU 基準）G / H / G + スケジューリング設定（pin 相当 + THREADED_LAST） | 0.020758 ± 0.001523 / top-1 93.922 %（3 つとも同じ。基準 ≤ 0.0259 / ≥ 91.94 %） |
| 自己整合（`Vulkan0:1` 基準）G | KLD 0.000001 / top-1 99.902 %（基準 < 0.001） |

## 速度（`q29`、`perf pin`、REPS=10、`-b 512 -ub 320 -ts 5/1/0`、`-ot token_embd=SGLX0`、C1 / G / H 等を 3 周交互）

G = `V0:0.70,C0:0.17,C1:0.05,R0:0.08`、H = `V0:0.78,C0:0.17,C1:0.05`（G から GT 430 を除き、その行を Vulkan0 へ）。pin = ホスト `taskset -c 0`、server + worker を cpu 1。batch = taskset なし、server + worker を SCHED_BATCH。plain = なにもしない。表は 3 周の平均（個々の値は `logs/q29_*.out`）。

| 構成 | pp512 | tg128 | pg512,128 | tg の 3 周 |
| --- | --- | --- | --- | --- |
| C1（SGLX なし） | 695.6 | 51.7 | 197.1 | 51.8 / 51.9 / 51.5 |
| **G pin**（GT 430 あり） | 702.4 | **68.7** | **240.4** | 69.0 / 68.6 / 68.4 |
| **H pin**（GT 430 なし） | 704.2 | **69.0** | **240.4** | 69.5 / 68.9 / 68.5 |
| G batch | 700.5 | 68.0 | 237.9 | 67.8 / 68.1 / 68.0 |
| H plain | 702.0 | 67.9 | 237.6 | 68.1 / 67.7 / 67.8 |

- **SGLX（どの構成も）は C1 に対して tg +31〜33 %、pg +22 %**（前回の勝ち構成 G は +25 %、64.7 だった。比とスケジューリングの調整で 68〜69 に伸びた）。pp は +1 %。
- **G 対 H は同等**（pin: tg −0.4 %、pg ±0 %。batch 対 plain は G が +0.1 % / +0.1 %）。各構成の標準偏差は 0.4〜0.8 なので有意ではない。**GT 430 の寄与は確認できない**。
- pin は plain より tg で約 +1 %（H: 69.0 対 67.9）。

## 区間時間の見方（`SGL_TRACE`）

`SGL_TRACE=<prefix>` で `<prefix>.cli`（プラグイン: composite の `mm_b` / `sub_b` / `sub_e` / `wait_e`、REMOTE0 の worker `p_*`、remote クライアントの `c_*`、CL11 の完了コールバック）と `<prefix>.srv`（サーバー: `s_hdr` / `s_payload` / `s_sub_b/e` / `s_done`）を出す（`CLOCK_MONOTONIC`、メモリに溜めて終了時に書く。未設定なら何もしない）。`scripts/trace_an.py` が 1 job を区間に、`scripts/mm_an.py` が 1 回の MUL_MAT の中の各イベントの時刻にする。`scripts/schedsamp.py` が `/proc/<pid>/task/*/schedstat` で各スレッドの run / run-queue 待ち / CPU を取る。注意: trace 付きの G は MUL_MAT 3.5 ms と遅い（分割が悪い設定 + コールバックのオーバーヘッド）ので、区間の**内訳**を見るためにだけ使った。

## 追加したもの

| ファイル | 内容 |
| --- | --- |
| `src/backends/remote/trace.h`、`adapters/ggml/composite/sgl_trace.h` | 上の trace（既定 OFF） |
| `src/backends/remote/server.cpp` | `SCRAPGPU_SERVER_CPU` / `_SCHED` / `_NICE`（Linux）、`SCRAPGPU_SERVER_SPIN_US`（受信前のポーリング。**測っていない**、既定 0） |
| `adapters/ggml/composite/pdev_scrapgpu.cpp` | `GGML_SGLX_REMOTE_CPU` / `GGML_SGLX_REMOTE_SCHED` |
| `composite_device.cpp` | `GGML_SGLX_THREADED_LAST` |
| `pdev_cl11.cpp` | trace 時だけ読み戻しイベントの完了コールバック |
| `tools/remote_bench.cpp` | `--gap-us` / `--hog` / `--spin-gap` |

## 残課題

- **GT 430 が速度に寄与する構成は見つからなかった**。上限は約 0.8 %（上の 7）で、現在のノイズ（0.5 %）と同程度。寄与を出すには Vulkan0 の固定費（約 0.7 ms + submit 0.15 ms）が下がって Vulkan0 の 1 行あたりの時間が増える側（= 他のデバイスの出番が増える）か、GT 430 の 1 行あたりの時間を半分以下にする必要がある。後者は q6_K の Fermi 向けカーネル調整（帯域は 12.5 GB/s、理論値は約 23 GB/s【`docs/benchmarks/moe-ep-20261021` の MB3 の「34 %」からの逆算、推定】）。
- 分割比は手探り（`q26`、`q27` の格子）。`GGML_SGLX_AUTOTUNE` は alone の測定が実機の並列実行と 0.2 ms ほどずれ、ラウンドごとに比が揺れた（`q25`）。Vulkan0 の「固定費 + 小さい傾き」を考慮した当てはめ（今は全デバイスが同時に終わる比）と、並列実行中の完了時刻（CL11 の完了コールバック）を使うなら改善の余地がある。
- 区間 trace で分かった `submit(REMOTE0)` のブロックの根本対策（worker をホストの CPU から離す）は環境変数にしただけで、既定にしていない。ホストのコア数・割り当てを知らない所で自動で pin するのは危険なため。
- `SCRAPGPU_SERVER_SPIN_US`、`SCRAPGPU_CUDA_SCHED` は測っていない。
