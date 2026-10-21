# MoE 複合デバイス マイクロベンチ MB0〜MB5（TB250 実測）

設計書 `docs/plans/MOE_COMPOSITE_DEVICE.md` 9 章の「勝ち筋の判定」を、`adapters/ggml/tools/moe_rt_bench.cpp`（U1）で実測した結果。測定日 2026-10-21。この文書の数値は、特記がない限りすべて `logs/` にある実測値（中央値。p99 は CSV とログ）。**【推定】** と書いたものだけが実測ではない。

## 0. 結論（先に）

| 問い | 結果 |
| --- | --- |
| **MB0**: CUDA で Fermi（GT 430）と Kepler（GT 730 / GT 710）を 1 プロセスで使えるか | **使えない**。先に Kepler の context を作ると GT 430 の `cuCtxCreate` が、先に Fermi を作ると両方の Kepler の `cuCtxCreate` が `CUDA_ERROR_INVALID_DEVICE`（101）になる（OpenCL と同じ制約）。Kepler 同士（GT 730 + GT 710）は共存でき、1000 往復の交互起動も誤りなし。Fermi 単独も動く。sm_20 の PTX を 390.157 の JIT に渡せば、Fermi（sm_21）にも Kepler（sm_35）にも同じ PTX で起動できる。**GT 430 は別プロセス（remote）のまま** |
| **MB2**: 複数デバイスの同時実行は max か和か | **計算が主体なら max に近い**（4 枚で 1 ms 前後のカーネル: 並列 1 096 µs、最大の単独 1 007 µs、和 2 921 µs）。**往復が主体の小さい仕事は max と和の中間**（4 枚の 8 KiB 往復: 並列 258 µs、最大の単独 167 µs、和 450 µs。デバイス 1 枚増えるごとに投入のホスト時間 15〜45 µs が足される）。ホスト 1 スレッドの「全部に投入 → 順に待つ」が最良で、デバイスごとのスレッドは同じか悪く、イベントの busy-poll をスレッドで行うと 2 コアを奪い合って約 14 倍遅くなる（3.9 ms 対 0.28 ms） |
| **判定式** | **式の字面では「C2 へ進む」（利得が正）だが、余裕が薄い**。WX の slice を公式 Vulkan に、NVIDIA の slice を OpenCL にした ES（MB3x）で、1 層あたり +66 µs（router を RX 側に置く場合）〜+203 µs（router をホストに置く場合）、L_r = 6 で **+0.4〜+1.2 ms / トークン**（設計書の C1 の見積もり 22〜23 ms に対し +2〜5 %）。全部を OpenCL にした ES（MB3(ii)）では負（−2.5 ms / トークン）。詳細は 8 章 |
| **C3（全層 EP）** | **候補にならない**。ES 1 層が 671 µs、RX 側の往復が 73〜210 µs で、22 層だと 16〜19 ms。CUDA の往復（60 µs）に替えても支配項は変わらない |

**問題点（先に）**: (1) NVIDIA 390 の OpenCL は、ページ可能（malloc）メモリからの non-blocking 転送が **Fermi で 1 往復 100 ms** かかる（Kepler は 95 µs）。実装は固定（pinned）ホストメモリを必須にすること。(2) MB3 / MB5 の OpenCL カーネルはこのツール用の素朴な R32 Q4_0 カーネルで、既存の調整済み CL11 カーネルではない。到達帯域は名目の 14〜77 %（GT 430 34 %、WX 2100 43〜48 %、GT 730 54 %、GT 710 77 %、RX 6400 14 %）。判定には ggml-vulkan の実測と組み合わせて使った。(3) SGL の cpu backend は測っていない（ggml-cpu だけ）。(4) 計測中に気付いた bench lock の問題は 9 章。

## 1. 環境

| 項目 | 内容 |
| --- | --- |
| マシン | TB250（BIOSTAR TB250-BTC PRO）、Celeron G3930（2 コア、AVX なし）、メモリ 31 GiB、Debian 13、kernel 6.12.107+deb13-amd64 |
| NVIDIA | ドライバ 390.157（CUDA ドライバ 9.1 = `cuDriverGetVersion` 9010）。GT 430（Fermi sm_21、964 MiB）、GT 730（Kepler sm_35、981 MiB）、GT 710（Kepler sm_35、2 002 MiB）。OpenCL は GT 430 が 1.1、Kepler が 1.2 |
| AMD | Mesa 25.0.7-2+deb13u1（RADV、rusticl は OpenCL 3.0、Clover は OpenCL 1.1）。RX 6400（navi24）、Radeon Pro WX 2100（polaris12）。Vulkan は `VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/radeon_icd.json`（NVIDIA の Vulkan ICD は落ちるため） |
| PCIe | sysfs の `current_link_speed` / `width`（測定前のアイドル時）: RX 6400 側 03:00.0 は 16 GT/s x16（カード内ブリッジ）、WX 2100 は 5.0 GT/s x1、NVIDIA 3 枚は **2.5 GT/s x1**（アイドル時の電力状態の可能性。docs/HARDWARE.md は 5.0 GT/s と記載。負荷時は確認していない） |
| ggml / llama.cpp | `~/moe-ep/src/llama.cpp`（23b0202a1）の `~/moe-ep/src/llama-build/bin` を**読み取りだけ**で再利用（`libggml.so`、`libggml-base.so`、`libggml-vulkan.so`、`libggml-cpu.so`）。ggml-vulkan: RX 6400 は fp16 1 / int dot 1 / warp 32、WX 2100 は fp16 0 / int dot 0 / warp 64 |
| 測定条件 | すべて `~/bench/gpu-bench.lock` の下（`moe_rt_bench_run.sh`）、各測定の開始時にすべてのカードが 45 ℃以下、終了時も 45 ℃以下（85 ℃の待機は一度も発動していない）。sudo 不使用。ドライバは 390 のまま |
| ツール | `adapters/ggml/tools/moe_rt_bench.cpp`（sha256 `76f2e47ba8b688aa…`）。ビルドは同ファイルの先頭コメント。TB250 では `g++ -std=c++17 -O2 -Wall -Wextra -I<llama.cpp>/ggml/include moe_rt_bench.cpp -lOpenCL -ldl -pthread -L<build>/bin -lggml -lggml-base` で警告なし |
| 作業場所 | `~/moe-ep/rt-bench/`（ソース、バイナリ、ログ）。他のディレクトリは読み取りだけ |

Mac（M2 Max、Apple OpenCL 1.2、ggml-cpu）で `mb1 mb2 mb3 mb3g mb3x mb4 mb5 mb5g` が `all OK`（`--dup 2` で同じ GPU を 2 枚として使った経路を含む）。`mb0` / `mb1c` は libcuda が無いので SKIP。`-DMRB_NO_GGML` のビルドも通る。

## 2. 実行したコマンド

- 全体: `suite.sh`（この README と同じ場所。`./suite.sh mb0 mb1c mb1 mb2 mb3 mb3g mb3x mb4 mb5 mb5r mb5g`）。各コマンドは `adapters/ggml/tools/moe_rt_bench_run.sh LOGDIR TAG cmd...`（bench lock、温度待ち、ログのヘッダに温度・環境・コマンド）を通る。`DRY_RUN=1` でコマンドだけ出る。
- 結果: `logs/*.out`（標準出力）、`logs/*.log`（ヘッダ、標準エラー、終了コード、開始・終了温度）、`logs/all.csv`（`mb,variant,device,n,median_us,p99_us,min_us,mean_us,note`）。
- 例: `RUSTICL_ENABLE=radeonsi moe_rt_bench mb2 --platforms NVIDIA,rusticl --devices "GT 730,GT 710,WX 2100,RX 6400" --csv all.csv`
- `suite.sh` の `mb0` / `mb1c` は 2 回目の実行から Fermi 先行などの組み合わせを足した最終版。最初に実行した mb0 / mb1 / mb1c は、ツール側の不具合（CUDA の pinned 入力が未初期化、ページ可能メモリの non-blocking が 1 往復 100 ms で 10 000 回が終わらない）のため破棄して、修正後に取り直した（`logs/` にあるのは取り直し後だけ）。
- 反復回数: MB1 / MB2 / MB4 は 10 000 回、MB3 は 1 000 回。**MB5 は 10 回**（1 層で 50 ms〜1.3 s かかるため。設計書の「1 千回」は不可能）。GT 430 の OpenCL non-blocking（ページ可能）だけ 1 往復 100 ms なので 100 回。

## 3. MB0: CUDA Driver API（`logs/mb0_*.out`）

PTX は `.version 4.0` / `.target sm_20` / `.address_size 64`（ソースに埋め込み、nvcc 不要）。`dlopen("/usr/lib/x86_64-linux-gnu/libcuda.so.1")`。CUDA の列挙順は GT 730（0）、GT 430（1）、GT 710（2）。

| 作る順 | `cuCtxCreate` の結果 | 起動・結果 |
| --- | --- | --- |
| 昇順 0, 1, 2（GT 730 → GT 430 → GT 710） | GT 730 OK、**GT 430 失敗（101 INVALID_DEVICE）**、GT 710 OK | 2 枚とも JIT・起動・結果一致。交互 1000 回 × 2 枚で誤り 0 |
| 降順 2, 1, 0 | GT 710 OK、**GT 430 失敗**、GT 730 OK | 同上 |
| Fermi 先行 1, 0, 2 | GT 430 OK、**GT 730 失敗、GT 710 失敗** | GT 430 のみ |
| GT 430 単独 | OK | JIT・起動・結果一致 |

結論: **Fermi と Kepler は CUDA でも同じプロセスに入れられない**（最初に作った世代が残る）。Kepler 2 枚は共存する。`cuCtxCreate` は 26〜220 ms。

## 4. MB1: 8 KiB 往復（書き込み → ほぼ空のカーネル → 読み出し、中央値 µs、`logs/mb1_*.out`・`mb1c*.out`）

n = 10 000（※は 100）。pinned = `CL_MEM_ALLOC_HOST_PTR` を map したまま使う。

| デバイス（API） | blocking | non-blocking + clFinish | イベント busy-poll | pinned + clFinish | pinned + busy-poll | カーネル 1 個 + finish（T_k） |
| --- | --- | --- | --- | --- | --- | --- |
| GT 430（OpenCL 1.1） | 77.3 | **100 391 ※** | **100 227 ※** | **69.0** | 68.8 | 12.3 |
| GT 730（OpenCL） | 66.4 | 95.2 | 95.1 | **64.2** | 63.5 | 12.0 |
| GT 710（OpenCL） | 66.4 | 95.1 | 95.0 | **64.2** | 63.5 | 11.9 |
| WX 2100（Clover） | 127.4 | 130.8 | 128.9 | **115.7** | 113.8 | 66.0 |
| WX 2100（rusticl） | 179.1 | 165.2 | 166.2 | 165.2 | 165.4 | 106.5 |
| RX 6400（rusticl） | 168.7 | 154.3 | 155.1 | 155.0 | 155.6 | 118.6 |

CUDA Driver API（`mb1c`）:

| デバイス | 同期コピー + ctx sync | async（ページ可能）+ stream sync | async pinned + stream sync | async pinned + event 問い合わせ | カーネル + sync |
| --- | --- | --- | --- | --- | --- |
| GT 730 | 69.6 | 73.4 | **60.3** | 60.2 | 10.5 |
| GT 710 | 69.6 | 73.7 | **60.3** | 60.2 | 10.5 |
| GT 430（単独プロセス） | 69.5 | 79.2 | 79.2 | 79.0 | 10.7 |

- 8 KiB の書き込みは NVIDIA で約 28 µs、読み出しは約 30 µs。AMD の読み出しは Clover 78 µs、rusticl 100 µs。
- p99 は中央値の 1.0〜2.2 倍（NVIDIA は 1.02〜1.1 倍、RX 6400 の busy-poll が最大）。
- **GT 430 の OpenCL で、ページ可能メモリからの non-blocking 転送が毎回 100 ms**（最小 0.12〜0.66 ms の回もある）。Kepler では起きない。pinned にすると 69 µs。MB2 以降のホスト側ステージングは pinned を既定にした。
- NVIDIA は OpenCL と CUDA の差が小さい（64 → 60 µs）。CUDA にしても RT は 0.1 ms を切るが、元から切っている。設計書 3.3 節の「CUDA で RT が 10〜20 µs 級」は、**RT 全体（60 µs）では外れ**、カーネル起動だけ（10.5 µs）が当たっている。

## 5. MB2: 複数デバイス同時（`logs/mb2_*.out`、`mb1c.out`）

「全部に投入 → 順に待つ」（`1thread`）、デバイスごとのスレッド（`threads`、起床込み）、1 枚ずつ完了を待つ（`sequential`）。tiny = 8 KiB 往復、busy = 各デバイスで約 500 µs になるよう較正したカーネル（デバイスと run によって 0.5〜1.0 ms になった）。中央値 µs。

| 組み合わせ（待ち方 finish） | 負荷 | max(単独) | 和(単独) | sequential | 1thread | threads | 1thread / max | 1thread / 和 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GT 730 + GT 710 + WX（rusticl）+ RX（rusticl） | tiny | 167.1 | 449.7 | 563.1 | **257.7** | 328.3 | 1.54 | 0.57 |
| 同上 | busy | 1 007 | 2 921 | 3 513 | **1 096** | 1 882 | 1.09 | 0.38 |
| GT 730 + GT 710 | tiny | 64.3 | 128.4 | 128.9 | **77.8** | 162.1 | 1.21 | 0.61 |
| GT 730 + GT 710 | busy | 543.8 | 1 059.9 | 1 078.3 | **567.4** | 579.4 | 1.04 | 0.54 |
| GT 730 + GT 710 + WX（rusticl） | tiny | 164.9 | 293.6 | 324.7 | **221.7** | 239.2 | 1.34 | 0.76 |
| 同上 | busy | 867.4 | 1 956 | 2 056 | **693.3** | 1 388 | 0.80 | 0.35 |
| GT 730 + GT 710 + WX（Clover） | tiny | 115.6 | 244.2 | 265.5 | **162.8** | 213.4 | 1.41 | 0.67 |
| 同上 | busy | 603.3 | 1 675 | 1 935 | **935.9** | 1 205 | 1.55 | 0.56 |
| GT 730 + GT 710（CUDA、`mb1c`） | tiny | 60.4 | 120.6 | | **71.5** | | 1.18 | 0.59 |
| 同上（CUDA） | busy | 554.8 | 1 108.5 | | **568.0** | | 1.03 | 0.51 |

- **ベンダが違っても（NVIDIA + rusticl + Clover）同時に動く**。カーネルが長ければ壁時計は max に近く、和にはならない（rusticl と Clover は同時に使っていない）。
- 小さい仕事では、投入の API 時間（NVIDIA 約 25 µs、rusticl / Clover 約 40〜45 µs / デバイス）が直列に足される。そのため 4 枚の往復は max の 1.5 倍。
- Clover の組み合わせの busy は max の 1.55 倍（Clover の投入が重いまま長く CPU を使う可能性。原因は未調査）。
- スレッドは悪化または同等。busy-poll（poll）をスレッドで行うと、2 コアに 4 スレッドが張り付いて 3.9 ms（`logs/mb2_4dev_nv_rusticl.out` の `threads_..._poll`）。

## 6. MB3: 1 層ぶんのエキスパート（decode、1 トークン、k = 4、32 エキスパート、2048×1792 Q4_0、gate + up + SwiGLU + down）

重みは乱数（ggml の Q4_0 ブロック形式。R32 に詰め直して upload）。参照は倍精度の CPU（正しさ NMSE < 1e-8 を `OK`/`FAIL` で判定。OpenCL は全 run で 3.6e-14〜4.8e-14、ggml-Vulkan は RX 2.95e-05（Q8_1 活性）、WX 3.6e-14、ggml-cpu 1.0e-04）。1 層の重みの読み出し量は 24.77 MB。

### (i) 1 枚（`i_single`、OpenCL の素朴カーネル）

| デバイス | カーネルだけ（µs） | 到達帯域（名目の何割） | x 書き込み + ids + カーネル + 出力読み出し（µs） |
| --- | --- | --- | --- |
| GT 430 | 3 190 | 7.8 GB/s（34 %） | 3 244 |
| GT 710 | 2 742 | 9.0 GB/s（77 %） | 3 066 |
| GT 730 | 1 210 | 20.5 GB/s（54 %） | 1 399 |
| WX 2100（rusticl / Clover） | 1 331 / 1 192 | 18.6 / 20.8 GB/s（43 / 48 %） | 1 462 / 1 313 |
| RX 6400（rusticl） | 1 402 | 17.7 GB/s（14 %） | 1 506 |

### (iii) ggml（公式 Vulkan / ggml-cpu）の同じ層（`mb3g`、MUL_MAT_ID × 3 + SwiGLU + MUL + ADD × 3）

| デバイス | ids・重み set + graph_compute + synchronize（µs） | x・ids・重み set + 計算 + 出力 get（µs） | 到達帯域 |
| --- | --- | --- | --- |
| **RX 6400（Vulkan0）** | **301.2**（p99 352） | 356.7 | 82.3 GB/s（設計書の 82 GB/s と一致） |
| **WX 2100（Vulkan1）** | **947.3**（p99 1 081） | 1 034 | 26.2 GB/s（設計書の約 26 GB/s と一致） |
| Celeron（ggml-cpu、2 スレッド）(iv) | 2 488 | 2 500 | 10.0 GB/s |

RX 自身が 1 層を計算する時間は 0.30 ms、WX が 1 層を一人で計算する時間は 0.95 ms。

### (ii) ES（エキスパートのスライス、1 層を行方向に帯域比で分割、ホスト 1 スレッド、投入 → 待ち → ホストで合計）

| 構成 | 内訳（行） | 1 層（µs、x 書き込み・出力読み出し込み） | カーネルだけ（µs） |
| --- | --- | --- | --- |
| **WX = 公式 Vulkan、GT 730 / GT 710 = OpenCL（`mb3x`）** | WX 832 / GT 730 736 / GT 710 224 | **670.9**（p99 980） | |
| 同上、GT 710 を外す | WX 960 / GT 730 832 | 749.6（p99 823） | |
| すべて OpenCL（WX = rusticl） | 同じ分割 | 1 153.5（p99 1 779） | 1 006（p99 2 587） |
| すべて OpenCL（WX = Clover） | 同じ分割 | 1 901.8（p99 2 716） | 947 |

単独（slice だけ、全工程込み）の最長が全体を決めている: Vulkan WX 832 行 620 µs、GT 730 736 行 659 µs、GT 710 224 行 547 µs。WX の Vulkan は slice を小さくしても時間があまり減らない（960 行 632 µs、832 行 620 µs、1792 行 947 µs）ので、固定費が大きい（960 行と 1792 行の2点からの直線で約 270 µs【推定】）。GT 730 は 736 行 659 µs・832 行 742 µs（約 0.86 µs / 行）。Clover を使う構成は、x 書き込み・出力読み出しを含めると、rusticl より 750 µs 遅い。

## 7. MB4: ggml-vulkan の境界コスト（`logs/mb4.out`、中央値 µs、n = 10 000）

| 操作 | RX 6400（Vulkan0） | WX 2100（Vulkan1） |
| --- | --- | --- |
| `tensor_set` 8 KiB | 5.6 | 22.7 |
| `tensor_get` 8 KiB | **67.0**（p99 95.9） | **76.4** |
| `tensor_get` 16 B | **67.0** | 57.0 |
| `tensor_get` 8 KiB + 16 B + 16 B（router を RX で計算する場合の sched の入力 3 本） | **204.5** | 191.3 |
| `set` 8 KiB + `get` 8 KiB | 74.2 | 90.5 |
| 1 ノード（ADD）`graph_compute` のみ | 73.3 | 72.8 |
| 1 ノード `graph_compute` + `synchronize` | **73.3**（p99 115） | 72.6 |
| `synchronize`（アイドル） | 0.05 | 0.05 |
| set 8 KiB + 1 ノード + sync + get 8 KiB | 144.6 | 152.6 |

- `tensor_get` は**サイズによらず約 60〜67 µs**（16 B でも 67 µs。同期とステージングのコスト）。`tensor_set` は安い（5.6 µs）。
- **方式 E の RT_in + RT_out**: router を RX 側に置くと get 3 本 + set 1 本 = 204.5 + 5.6 = **210 µs**、router をホストで計算すると get 1 本 + set 1 本 = **約 73 µs**。
- 1 ノードの `graph_compute` は完了を待ってから返る（nosync と +sync が同じ 73 µs）。そのため ggml-vulkan の slice を NVIDIA の slice と重ねるには、NVIDIA に先に投入してから呼ぶか、ワーカースレッドが要る（MB3x は前者）。

## 8. 判定式の評価

```
decode の利得 ≈ L_r × [ T_WX単独(1層) − ( MB4往復 + MB3(ii) ) ]      L_r = 6
```

| 項 | 値 | 出典 |
| --- | --- | --- |
| T_WX単独(1 層) | **947 µs** | MB3 (iii) WX 2100 公式 Vulkan のカーネルだけ（C1 は層ごとの同期がないので、x・出力の転送は含めない） |
| MB4 往復（router を RX で） | **210 µs** | 5 章 MB4（get 3 本 + set 1 本） |
| MB4 往復（router をホストで） | **73 µs** | get 1 本 + set 1 本 |
| MB3(ii) 最良（WX = Vulkan + NVIDIA 2 枚 OpenCL） | **671 µs** | `mb3x_wxvk_gt730_gt710` |
| MB3(ii) 全部 OpenCL（WX = rusticl） | 1 153 µs | `mb3_es_rusticl` |

| 構成 | 1 層の利得 | L_r = 6 の利得 / トークン | 判定 |
| --- | --- | --- | --- |
| router を RX、ES = 最良（671） | 947 − (210 + 671) = **+66 µs** | **+0.40 ms** | 正（わずか） |
| router をホスト、ES = 最良（671） | 947 − (73 + 671) = **+203 µs** | **+1.22 ms** | 正 |
| router を RX、ES = 全部 OpenCL（1 153） | 947 − (210 + 1 153) = −416 µs | −2.5 ms | **負** |
| router をホスト、ES = 全部 OpenCL | 947 − (73 + 1 153) = −279 µs | −1.7 ms | **負** |

**判定: 式の字面では「C2（段階 1〜3）へ進む」（WX の slice に公式 Vulkan を使い、NVIDIA を OpenCL で足す構成に限る）。ただし利得は C1 の 22〜23 ms（設計書の見積もり。この文書では測っていない）に対して +2〜5 % で、損益分岐（MB3(ii) が 737 µs（router を RX）または 874 µs（ホスト）を下回ること）に対して 9〜23 % の余裕しかない。**

実装で足されるのに、この式に入っていないもの: pdev_ggml のワーカースレッド起床（MB2: スレッドは単独の `1thread` より 17〜85 µs 悪化）、パターン照合・plan の時間、ES の分割比の外れ、揺れ（ES の p99 は中央値の 1.46 倍 = 980 µs。p99 で見ると利得は負になる）、ホスト合計。これらで余裕が消えるリスクが大きい。**C2 に進むなら、(a) router をホストで計算して sched の入力を 1 本にする、(b) 分割比を実測の時間に合わせる、(c) WX は公式 Vulkan のまま包む、の 3 点が前提**。そうでない構成（全部 OpenCL、router を RX）は負けるので、設計書 9 章の「decode の並列化はやめる」側。

- 到達帯域が低い NVIDIA カーネルを調整済みにしたときの利得は【推定】: GT 730 の slice を 65 % の帯域にすると 736 行で約 0.55 ms。ただし支配しているのは WX の Vulkan slice（620 µs、固定費約 300 µs）なので、ES は 610〜650 µs 程度が下限と見込まれる（利得は +100〜+150 µs / 層程度）。
- **C3（全層 EP）**: 22 × (MB4 往復 + MB3(ii)) = 22 × (73〜210 + 671) = 16.4〜19.4 ms（エキスパートだけ）。非エキスパート（設計書 3.7 ms）と固定費を足すと 20 ms を超え、設計書の C2 の見積もり（16〜19 ms）を上回る。CUDA の RT（60 µs）に替えても、支配項は ES 671 µs と RX の `tensor_get`（67 µs）なので結論は同じ。**C3 は候補にしない**。
- 同時に、RX 自身が 1 層を計算する時間は 0.30 ms で、遠隔の ES 1 層（671 µs + 境界 73〜210 µs）の 2.5〜3 倍速い。RX に載る層は全部 RX に置く、が前提のまま。

## 9. MB5: prompt バッチ（`mb5`、`mb5r`、`mb5g`、n = 128 / 512、10 回）

1 層あたり、エキスパート 4 つ、1 トークン 88.1 MFLOP。ES = 各デバイスに全トークンと行スライス、EP = エキスパートを分けて、そのエキスパートに当たるトークンだけを送る。ホスト 1 スレッド。正しさ: 先頭 3 トークンを倍精度で照合（NMSE 1.6e-12〜2.0e-12）。

### 単独（1 層、全工程込み、`single`）

| デバイス | n = 128 | n = 512 | 実効 GFLOPS（n = 128 / 512） |
| --- | --- | --- | --- |
| RX 6400（OpenCL rusticl の素朴カーネル） | 21.1 ms | 69.2 ms | 534 / 652 |
| WX 2100（OpenCL rusticl） | 64.0 ms | 221 ms | 176 / 204 |
| GT 730（OpenCL） | 205 ms | 661 ms | 55 / 68 |
| GT 710（OpenCL） | 412 ms | 1 320 ms | 27 / 34 |
| **RX 6400（公式 Vulkan、`mb5g`、全工程込み / 計算だけ）** | **12.3 / 10.0 ms** | **22.7 / 15.6 ms** | 1 130 / 2 892（計算だけ） |
| **WX 2100（公式 Vulkan、全工程込み / 計算だけ）** | **79.6 / 71.2 ms** | **157.8 / 122.7 ms** | 158 / 367（計算だけ） |

### 遠隔層の複合（WX + GT 730 + GT 710、OpenCL の素朴カーネル）

| 分割 | 種類 | n = 128 | n = 512 |
| --- | --- | --- | --- |
| 帯域比（WX 832 / GT 730 736 / GT 710 224 行） | ES | 91.5 ms | 296 ms |
| 帯域比（エキスパート 15 / 13 / 4） | EP | 90.2 ms | 289 ms |
| **GFLOPS 比 176 : 55 : 27**（`mb5r`、WX 1 216 / GT 730 384 / GT 710 192 行） | ES | 56.6 ms | 185.5 ms |
| 同上（エキスパート 22 / 7 / 3） | EP | **51.2 ms** | **171.1 ms** |
| GFLOPS 比 176 : 55（WX + GT 730） | ES / EP | 57.2 / 58.1 ms | 191.2 / 190.9 ms |

- prefill の分割比は**帯域ではなく演算性能に比例**させないと、最も遅い GT 730 が全体を決める（帯域比: 91 ms、GFLOPS 比: 51〜57 ms）。
- EP は ES より約 10 % 速い（n = 128、GFLOPS 比）。転送量が少ないため（n = 128、帯域比の分割で EP 5.0 MB 対 ES 6.3 MB）。
- ホスト側の準備（ソート、gather、タイル表）は n = 128 で 2〜3 ms、n = 512 で 8〜12 ms。他のデバイスの計算と重ねていないので、そのまま足される（Celeron）。
- **比較**: 遠隔層の 1 層を、公式 Vulkan の WX 単独（全工程込み）と比べると、EP は **n = 128 で 1.55 倍速い（79.6 → 51.2 ms。計算だけの 71.2 ms と比べると 1.39 倍）、n = 512 で 0.92 倍（157.8 → 171.1 ms、負け。計算だけの 122.7 ms と比べると 0.72 倍）**。n = 512 で負けるのは、この MB5 の OpenCL カーネル（WX で 204 GFLOPS）が、公式 Vulkan（367 GFLOPS）より遅いため（複合デバイスの設計の問題ではなくカーネルの質）。WX も Vulkan にした prefill ES / EP は**未測定**（MB3x の prefill 版を作っていない）。単純な合算の【推定】では、Vulkan の WX 367 + GT 730 68 + GT 710 34 = 469 GFLOPS で、n = 512 の WX 単独の 1.28 倍。
- 遠隔層（EP の最良）は RX 6400 の 1 層（Vulkan、全工程込み n = 128: 12.3 ms、n = 512: 22.7 ms）の **4〜7.5 倍遅い**。設計書 3.4 節の「C2 の pp が C1 より 1.5 倍」は、n = 128 では実測の延長で成り立つ（遠隔 6 層 + RX 16 層のエキスパートだけを実測の 1 層の時間で合算した【推定】: C1 の WX 単独 6 × 79.6 + 16 × 12.3 = 674 ms に対して C2 の EP 6 × 51.2 + 16 × 12.3 = 504 ms で 1.34 倍）が、n = 512 では成り立たない（C1: 6 × 157.8 + 16 × 22.7 = 1 310 ms、C2: 6 × 171.1 + 363 = 1 390 ms で 0.94 倍。OpenCL カーネルのままの場合）。

## 10. 問題点・注意

1. **bench lock が最大で `TMO` 秒（1 500 秒）ずっと保持されることがある**: `~/moe-ep/scripts/run.sh`（baseline の `run.sh`）は、タイムアウト用の `( sleep $TMO; ... ) &` を、コマンドの終了後に `kill $W` するが、孤児になった `sleep $TMO` が flock のファイル記述子を持ったまま残り、ロックが `TMO` 秒間解放されない。この計測中に実際に起きた（run が終わった後も PPID 1 の `sleep 1500` がロックを保持し、残り約 8 分待つことになった）。孤児の `sleep`（PID 27921、コマンドは `sleep 1500` のみ、PPID 1）を 1 つだけ `kill` してロックを解放した（他の処理には触れていない）。`moe_rt_bench_run.sh` は 1 秒刻みの待ちに替えて同じ問題が起きない。**`run.sh` 側の修正を推奨**（`kill $W` のあとに子の `sleep` も止めるか、1 秒刻みにする）。
2. **MB3 / MB5 の NVIDIA カーネルは素朴**: GT 430 の到達帯域は 34 %（Fermi は整数演算が遅い）。調整済みの CL11 カーネルなら 65 % 程度が見込まれる【推定】。ES の最長は WX の Vulkan slice なので、判定への影響は 6 章末尾の見積もりの範囲。
3. 各 MB の「SGL の cpu backend（SSE2）」は未測定（ggml-cpu だけ: 1 層 2.49 ms）。設計書 4.4 節の CPU slice の比較には足りない。
4. MB2 の busy カーネルは「約 500 µs」に較正したが、デバイスと run ごとに 0.5〜1.0 ms になった（rusticl のクロック状態など）。比較は同じ run の `max` / `和` との比で見る。
5. MB4 は `graph_compute` が完了を待つ（ワーカースレッドなしでは重ねられない）。`pdev_ggml` はワーカースレッドが必須になるが、2 コアの Celeron では起床コストが MB2 のスレッドと同様に効く。
6. PCIe のリンクは、アイドル時に NVIDIA が 2.5 GT/s と表示された。測定中の値は確認していない。
7. 実測していないもの: GT 430 を含む ES（別プロセスが必要）、RX 6400 を複合デバイスの内部に入れた構成、prefill の WX = Vulkan 構成、30B（⑤）、CUDA のカーネル（MB1c は往復だけ）。

## 11. 設計書の【推定】を置き換える候補（実測）

| 設計書の項 | 設計書の値 | 実測 |
| --- | --- | --- |
| 3.2 節 RT: NVIDIA の 1 起動 | 数 µs〜10 µs 前後【推定】 | OpenCL 11.9〜12.3 µs、CUDA 10.5 µs（カーネル + sync） |
| 3.2 節 RT: rusticl / Clover の 1 起動 | rusticl 約 100 µs、Clover 15〜20 µs | カーネル + finish: Clover 66 µs、rusticl（WX）107 µs、rusticl（RX）119 µs |
| 3.3 節 RT（8 KiB 往復） | 0.46〜1.4 ms（allreduce / デバイス間の転送） | NVIDIA 64〜69 µs、Clover 116 µs、rusticl 155〜165 µs。ggml-vulkan の境界 73〜210 µs |
| 3.3 節 CUDA の RT 「10〜20 µs 級」 | 【推定】 | カーネル起動は 10.5 µs だが、8 KiB の往復全体は 60 µs（書き込み 27 + 読み出し 31） |
| 3.2 節 WX 2100 の mat-vec 実効 | 約 26 GB/s | 26.2 GB/s（公式 Vulkan、MoE 1 層） |
| 3.2 節 RX 6400 の mat-vec 実効 | 約 82 GB/s | 82.3 GB/s（公式 Vulkan、MoE 1 層） |
| 3.2 節 CPU の 1 層 | 3.6 ms | 2.49 ms（ggml-cpu、Celeron 2 スレッド、1 層の 4 エキスパート） |
| 3.2 節 WX の 1 層（C1） | 1.4 ms（8.5 ms / 6 層） | 0.95 ms（公式 Vulkan、計算だけ） |
| 3.3 節 C2 の遠隔 1 層 | 0.43 ms + T_k + RT（RT = 0.5 ms で 0.93 ms 前後） | ES 0.67 ms + 境界 0.07〜0.21 ms = 0.74〜0.88 ms |
| 6.3 節 Fermi + Kepler を 1 プロセスで（CUDA） | 【不明】 | **不可**（OpenCL と同じ） |
| 9 章 MB2 | 【不明】 | 計算が長ければ max に近く、小さい往復は max と和の中間 |

## 12. 追加の知見（設計に効くもの）

- NVIDIA 390 の OpenCL は、**ページ可能メモリからの non-blocking の write / read がドライバ内で遅延され、Fermi では 100 ms の待ちが入る**。pinned（`CL_MEM_ALLOC_HOST_PTR` + map）にすれば 69 µs。設計書 4.3 節の「ピン留めされたステージング」は**必須**。
- ggml-vulkan の `tensor_get` は 8 B でも 8 KiB でも 57〜67 µs（同期コスト）。`pdev_ggml` で包む WX は、1 層ごとの `get` が 1 本でも約 70 µs 足される。router を RX の外（ホスト）に出して sched の入力を 1 本にする案（設計書 3.5 節の 3）の効果は大きい（210 → 73 µs）。
- ES の分割比は decode では帯域、prefill では演算性能に比例させる。固定費（WX の Vulkan で約 300 µs）が大きいので、WX に載せる行数を減らしても時間はほとんど減らない。
- 遠隔 1 層は RX 1 層の 2.5〜3 倍遅い（decode）。遠隔層を少なくする（RX の VRAM を増やす、`-c` / `-ub` を絞る）効果が最も大きい。
