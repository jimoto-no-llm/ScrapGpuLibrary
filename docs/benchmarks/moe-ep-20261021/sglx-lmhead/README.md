# SGLX0: 出力射影（lm_head、`token_embd.weight`）を語彙方向に分割して複数 GPU で同時に計算する（tb250）

測定日 2026-10-21、tb250、llama.cpp `23b0202a1`（baseline と同じビルド）、モデル `LFM2.5-8B-A1B-Q4_0.gguf`。llama.cpp は変更していない。SGL は `adapters/ggml/composite/`（新規）と `ggml-cl11.cpp` の登録部（数行）。**電力設定はすべて `perf auto`**（`perf high` の再測定は親側が行う。コマンドは下の「再現手順」）。数値はすべて実測で、推定は「推定」と書いた。

## 結論

1. **正しさは通った。** 平均 KLD（CPU の logits が基準）は C1 と同じ、SGLX の 1 枚と N 枚の自己整合は KLD 0.000002（目標 < 0.001）。`test-backend-ops -b SGLX0` は GET_ROWS が全構成で 240/240、MUL_MAT は Vulkan 1 枚の既知の失敗（後述、素の `-b Vulkan0` でも同じ）以外すべて通る。
2. **出力射影の区間は短くなった。** C1 では WX 2100 が 1 回 5.6 ms（c1-profile）。SGLX の MUL_MAT（N=1）は RX+WX で 5.25 ms、RX+WX+GT 730+GT 710 で **3.67 ms**（1.9 ms 短縮）。同じ構成で逐次に待つ（`GGML_SGLX_SERIAL=1`）と 7.29 ms なので、同時実行は 2.0 ms 効いている。
3. **それでも perf auto では tg が C1 より遅い**: C1 44.0 に対し SGLX 29〜37（下表）。pp512 は同等（C1 630.8、SGLX 637〜642）。**原因は出力射影ではなく、RX 6400 のクロックが落ちること**: SGLX の RX 部分があると、他デバイス上の層の区間（GET_ROWS 終了 → MUL_MAT 開始）が C1 の約 16 ms から 28 ms に延びる（RX 部分の比率 0.2〜0.85 で同じ。VRAM 溢れではない）。同じ run の `pp_dpm_sclk` サンプルで RX が 500 MHz に張り付く時間が長い（下の「RX のクロック」）。perf high で RX のクロックが固定されれば消える可能性がある（未検証、親側の測定待ち）。
4. したがって **perf auto では 3 指標とも C1 を上回れていない**（pp512 のみ同等）。perf high での結果が判断材料になる。

## 実装（`adapters/ggml/composite/`）

| ファイル | 内容 |
| --- | --- |
| `composite_device.cpp` | ggml デバイス `SGLX0`、buffer type / buffer（host 影）、行の分割、MUL_MAT / GET_ROWS、`GGML_SGLX_PROFILE` |
| `pdev.h` | 物理デバイス（pdev）の IF: `piece_create/upload/free`、`submit`（投入だけ）、`wait` |
| `pdev_ggml.cpp` | 任意の ggml デバイス（公式 Vulkan、ggml-cpu、Metal）を公開 API だけで包む |
| `pdev_cl11.cpp` | CL11 デバイス（GT 730 / GT 710）。既存の R32 mat-vec / mat-mat カーネルを `cl11_op_mul_mat` で流用 |

- 有効にするのは `GGML_SGLX_SPLIT` を設定したときだけ。無いと SGLX0 は現れず、CL11 の既存経路は変わらない（`test-backend-ops -b CL11_0 -o GET_ROWS,MUL_MAT` が Mac で 947/947）。デバイスは `libggml-cl11.so`（`GGML_BACKEND_PATH`）の中にある（CL11 の `cl_context` を二重に作らないため）。
- buffer は host メモリ。`-ot "token_embd\.weight=SGLX0"` で載せると、テンソル全体が host 影に置かれ、**最初の MUL_MAT で行が各 pdev に分配・upload される**（205.1 MiB、q6_K 2048×128000。tb250 で 0.3〜0.4 s）。`set_tensor` などで重みが書き換わると再 upload。
- MUL_MAT: 重みの行を比率に比例した連続範囲に切り（境界は 32 行単位、CL11 の R32 のため）、入力列を**全 pdev に投入してから**順に待つ。N=1 では各 pdev が logits の自分の範囲へ直接書く。N>1 は pdev ごとの領域に書いてから結合。
- GET_ROWS（入力埋め込み）は host 影から行を引いて ggml の dequantize で F32 にする。
- 対応しない op は `supports_op=false`: バッチ次元のある MUL_MAT、転置・パディングのある重み、src1 が F32 でないもの、全 pdev が計算できない型、MUL_MAT / GET_ROWS 以外。
- 分割比 `GGML_SGLX_SPLIT=Vulkan0:0.5,Vulkan1:0.2,CL11_0:0.2,CL11_1:0.1`。比を省くと既定（実効帯域: RX 6400 116、WX 2100 41、GT 730 25、GT 710 7.6、GT 430 15、CPU 10 GB/s）。比は全員に付けるか、全員に付けない。
- `GGML_SGLX_PROFILE=1`: MUL_MAT（N=1 / N>1）の全体・pdev ごとの submit 時間と完了時刻、pdev_ggml の内訳（graph / alloc / set / compute / get）、upload 時間、トークンの区間（GET_ROWS → 他デバイスの層 → MUL_MAT → 次の GET_ROWS）。最初の 3 呼び出し（シェーダのコンパイル）は平均から除く。`GGML_SGLX_SERIAL=1`: 各 pdev を投入の直後に待つ（同時実行の効果を測る）。`GGML_SGLX_THREADS=0/1`: pdev_ggml のスレッドの強制。

### Vulkan デバイスの共有（詰まった点）

- pdev_ggml は llama.cpp 本体と**同じ RX / WX に 2 つ目の `ggml_backend_t`** を作る。ggml-vulkan では物理デバイス・キュー・メモリは `vk_device` として共有、コマンドプール・フェンス・scratch はインスタンスごと。投入はすべて 1 つのホストスレッド（graph_compute を呼ぶスレッド）から行うので、キューの同時 submit は起きない。**tb250 で動作を確認した**（KLD・test-backend-ops とも異常なし）。WX の pdev だけは後述の理由で専用スレッドから投入する（同じ WX を本体が使うのは sched が同期した後なので重ならない）。
- **非同期にならない罠**: 初版は `tensor_get_async` を pageable メモリへ出していて、ggml-vulkan はその場で submit して待つ。結果、RX の get が 3 ms かかり（実体は RX の計算待ち）WX の投入が RX の完了後になって直列化した（tg 28 t/s、MUL_MAT 6.6〜8.3 ms）。pdev ごとに pinned ホスト buffer（`ggml_backend_dev_host_buffer_type`）を使うと RX の submit は 0.2 ms になった（tg 35）。
- **pinned は Vulkan0 のみ**: ggml-vulkan のホスト pinned buffer は `vk_instance.devices[0]` に登録されるため、WX（Vulkan1）では pinned にならず、非同期 get が同期になる。Vulkan1 以降は専用ワーカースレッドで submit + wait を行い（条件変数で待つ。ポーリングなし）、ホストスレッドは他の pdev の投入を続ける。
- 2 つ目のインスタンスが RX にあると RX のクロックが下がる、という現象（結論の 3）は、インスタンスを作るだけ（Vulkan0 の pdev を `GGML_SGLX_THREADS=1` で pinned なしにしても同じ）では変わらず、WX だけの SGLX（`Vulkan1:1`）や WX+GT 730 では起きない。RX に重みの一部（piece）を持たせた構成でだけ起きた（比率 0.2〜0.85 のどれでも）。インスタンスだけ作って使わない場合は測っていない。

## 正しさ

### test-backend-ops（tb250、perf auto、`SGLX0`）

| 構成（`GGML_SGLX_SPLIT`） | MUL_MAT | GET_ROWS |
| --- | --- | --- |
| `Vulkan0` | 1396/1397（前回 1395/1397） | 240/240 |
| `Vulkan1` | 1397/1397 | 240/240 |
| `CL11_0`（GT 730） | 418/418 | 240/240 |
| `Vulkan0:0.5,Vulkan1:0.5` | 1395/1397 | 240/240 |
| `Vulkan0:0.5,Vulkan1:0.2,CL11_0:0.2,CL11_1:0.1` | 417/418 | 240/240 |

失敗はすべて `MUL_MAT(type_a=q4_0, type_b=f32, m=40, n=9, k=32)`（と m=100, n=9, k=96）の NaN で、**Vulkan0 の pdev を含む構成でだけ、実行ごとに出たり出なかったりする**。素の `-b Vulkan0 -o MUL_MAT -p "type_a=q4_0,type_b=f32,m=(40|100),n=9"` が同じ 2 件で 0/2（`logs/tbo_vk0_plain_q40.out`、`tbo_plain_vk0_q40.out`）。ggml-vulkan 自体の既知の不具合で、SGLX の問題ではない。Vulkan1 と GT 730 は全件通る。Mac（MTL0、CPU、Apple OpenCL の CL11_0、およびその組み合わせ）では MUL_MAT・GET_ROWS とも全件通った（Mac の結果はコミット `7cc8c09` のメッセージ）。MUL_MAT の「対応外」（バッチ次元など）は skip で、全体のおよそ半数。

### モデル全体（`llama-perplexity -c 512 --chunks 4 -b 128 -ub 128 -fa off`、基準 CPU の logits、baseline と同じテキスト、`-ts 5/1`）

| 構成 | 平均 KLD ± SE | top-1 一致 | PPL(Q)/PPL(CPU) |
| --- | --- | --- | --- |
| C1（SGLX なし、今回の再測定） | 0.021464 ± 0.001467 | 92.843 % | 0.999407 |
| SGLX `Vulkan0:1`（1 枚） | 0.021508 ± 0.001469 | 92.647 % | 0.999533 |
| SGLX A `Vulkan0:0.5,Vulkan1:0.5` | 0.021504 ± 0.001469 | 92.647 % | 0.999369 |
| SGLX C `Vulkan0:0.5,Vulkan1:0.2,CL11_0:0.2,CL11_1:0.1` | 0.021504 ± 0.001469 | 92.647 % | 0.999369 |

- 合格条件（§8.1）: KLD ≤ C0 の 0.021625 × 1.2 = 0.02595 → **通る**（0.0215）。top-1 ≥ C0 の 92.941 − 1 = 91.94 % → **通る**（92.65 %、SE 0.8 の範囲内）。
- 自己整合（SGLX 1 枚の logits を基準に N 枚を比較）: A **KLD 0.000002**、top-1 100 %、C **0.000002**、100 %（目標 < 0.001）。A と C の KLD が表示桁で一致したのは、pdev ごとの丸めの差が小さいためと思われる（未検証）。

## 性能（perf auto、`-r 5`、`-ub 512`、`-ts 5/1`（SGLX は `-ts 5/1/0`））

コマンドの形（`scripts/bench.sh`、`TAG SPLIT`）: C1 = `llama-bench -m MODEL -dev Vulkan0/Vulkan1 -ngl 99 -fa 0 -t 2 -b 512 -ub 512 -r 5 -sm layer -ncmoe 0 -ts 5/1 -o jsonl -p 512 -n 128 -pg 512,128`。SGLX はこれに `-dev Vulkan0/Vulkan1/SGLX0 -ts 5/1/0 -ot 'token_embd\.weight=SGLX0'` と環境変数を付ける。`noop` は `-ot no_such_tensor_zz=CPU`（`-ot` があるとパイプライン並列が切れるので、その影響の対照）。

| 構成 | pp512 | tg128 | pg512,128 | SGLX MUL_MAT (ms) | 層の区間 (ms) |
| --- | --- | --- | --- | --- | --- |
| C1 | 630.76 ± 5.20 | 44.00 ± 0.19 | 170.47 ± 0.75 | (WX 5.6) | 約 16〜17 |
| C1 + `-ot` ノーオプ | 635.22 ± 4.00 | 44.30 ± 0.27 | 171.51 ± 2.09 | | |
| C1 + MMVQ | 630.89 ± 6.05 | 44.47 ± 0.41 | 171.31 ± 2.09 | | |
| C1 `-ts 4/1` | 531.30 ± 3.23 | 42.05 ± 0.20 | 158.27 ± 0.57 | | |
| SGLX A（RX 0.5 / WX 0.5） | 639.20 ± 4.07 | 29.31 ± 0.29 | 121.12 ± 1.12 | 5.25 | 28.9 |
| SGLX A + MMVQ | 640.51 ± 6.38 | 30.79 ± 0.69 | 125.63 ± 0.77 | 5.35 | 27.2 |
| SGLX A `-ts 4/1` | 539.53 ± 4.84 | 28.49 ± 0.26 | 114.54 ± 0.83 | 5.40 | 29.8 |
| SGLX A 逐次待ち（`GGML_SGLX_SERIAL=1`） | 638.10 ± 4.12 | 27.22 ± 0.18 | 115.37 ± 0.48 | 7.29 | 29.1 |
| SGLX B（RX 0.7 / WX 0.3） | 642.49 ± 2.44 | 33.56 ± 1.17 | 141.53 ± 7.99 | 4.19 | 24.9 |
| SGLX B + MMVQ | 642.08 ± 3.93 | 37.29 ± 3.94 | 150.45 ± 20.27 | 4.16 | 22.7 |
| SGLX C（RX 0.5 / WX 0.2 / GT 730 0.2 / GT 710 0.1） | 637.24 ± 7.43 | 31.81 ± 1.38 | 127.59 ± 0.39 | 3.67 | 28.1 |
| SGLX C + MMVQ | 639.72 ± 8.05 | 34.21 ± 1.00 | 134.31 ± 3.25 | 3.73 | 26.0 |
| SGLX C `-ts 4/1` | 540.62 ± 4.75 | 29.97 ± 0.26 | 121.94 ± 0.74 | 3.75 | 29.3 |
| SGLX D（RX 0.4 / WX 0.3 / GT 730 0.2 / GT 710 0.1） | 641.24 ± 6.01 | 30.24 ± 0.35 | 125.92 ± 0.96 | 3.93 | 28.9 |

- `SGLX MUL_MAT` はプロファイルの N=1 平均（ms、最初の 3 回を除く）、`層の区間` は GET_ROWS 終了 → MUL_MAT 開始。C1 の「約 16〜17」は SGLX の `Vulkan1:1`（RX 部分なし）の 16.0 と、c1-profile の RX 10.9 + 境界 0.4 + WX の 3 層（出力射影を除く）からの値。
- **C1 に対する pp512 は +1〜2 %（同等）、tg128 は −15〜−32 %、pg は −12〜−33 %。** B のばらつき（SE が大きい）は RX のクロック状態が run の中で動くため。
- `-ts 4/1` は SGLX でも C1 でも pp512 が約 530〜540 に落ちる（RX の層が 1 つ減る）。VRAM 不足時の代案として試したが、RX の VRAM は足りていた（下）ので不要。
- 初版（逐次に待っていた版、`logs/s_ts51_*`）は tg 26.9〜28.7、MUL_MAT 6.1〜8.3 ms。pinned + WX スレッドで 29〜37 になった。

### 出力射影区間の内訳（C 構成 N=1、`logs/f_sglx_C_ts51.log`）

| pdev | submit（ホスト） | 完了時刻（MUL_MAT 開始から） | pdev_ggml の内訳（µs、N=1 平均） |
| --- | --- | --- | --- |
| Vulkan0（RX、pinned） | 0.22 ms | 3.12 ms | graph 0 / alloc 2 / set 15 / compute 193 / get 6（投入だけ。完了は wait） |
| Vulkan1（WX、専用スレッド） | 0.011 ms（スレッドに渡すだけ） | 3.43 ms | set 769 / compute 110 / get 1908（get は GPU の計算待ちを含む） |
| CL11_0（GT 730） | 0.17 ms | 3.48 ms | |
| CL11_1（GT 710） | 0.09 ms | 3.67 ms | |

- 全体 3.67 ms（gather 0.003 ms、scatter 0）。**ホストの投入コスト（graph 構築・alloc・set・compute 投入）は 1 トークンあたり、RX 0.22 ms、WX の set + compute が 0.88 ms、GT 730 / 710 が 0.26 ms で、律速ではない**。律速は各 GPU の実行（RX は約 3 ms、すなわち既定の 2 ms 弱より遅い。クロックのため）。
- A 構成では WX が 0.5 を持ち 5.25 ms で完了（RX は 3.1 ms で完了、WX が律速）。WX は約 8 ms/全行 + 固定約 1 ms、RX は約 3 ms/全行 + 約 1.2 ms（perf auto）。最良のバランスは RX 0.5〜0.7 + 他で、C で 3.67 ms。
- トークンの区間（C）: GET_ROWS 0.013 ms、層の区間 28.1 ms、MUL_MAT 後（logits 取得・サンプリング・次のグラフ）0.43 ms。

### スケジューラの split 数（`GGML_SCHED_DEBUG=1`、`-v`、`logs/splits_*.log`）

| 構成 | split 数 | 内訳 | sched copies |
| --- | --- | --- | --- |
| C1 | **3** | #0 CPU（GET_ROWS 入力埋め込み）、#1 Vulkan0（21 層）、#2 Vulkan1（3 層 + 出力射影） | 4（パイプライン並列あり） |
| C1 + SGLX0 | **4** | #0 SGLX0（GET_ROWS）、#1 Vulkan0（入力 9）、#2 Vulkan1（入力 7）、#3 SGLX0（入力 1: result_norm、MUL_MAT） | 1（`-ot` でパイプライン並列が切れる） |

層の割り当ては同じ（Vulkan0 の split と Vulkan1 の split の入力数が C1 と同じ）。境界は 1 つ増える（Vulkan1 → SGLX0 の result_norm 8 KiB の get）。SGLX0 の compute buffer は 63.5 MiB（Vulkan0 15 MiB / Vulkan1 14 MiB、C1 は 30 / 72 MiB）。`-ot` ノーオプの対照（上表）は pp / tg とも C1 と同じなので、パイプライン並列の有無は pp512 / tg には影響しない。

### VRAM とクロック

- RX の VRAM（`mem_info_vram_used`、MiB）: C1 3849、SGLX（RX 0.85）4021（上限 4080）、GTT は 20 MiB 前後で溢れなし。WX は 841 → 641（出力射影が外れた分）。RX の比率 0.2〜0.85、`-ts 5/1`・`4/1`・`3/1` で層の区間は同じ約 28〜30 ms（`logs/q4_*`）なので、**VRAM の溢れは原因ではない**。
- 層の区間（GET_ROWS 終了 → MUL_MAT 開始、ms）: RX 部分なし（`Vulkan1:1`）16.0、`Vulkan0:0.2,Vulkan1:0.8` 28.7、`0.5/0.5` 28.3、`0.85/0.15` 23〜28。`Vulkan1:0.5,CL11_0:0.5`（RX 無し）15.6。
- RX のクロック（`logs/k_c1.clk`、`k_sg.clk`、20 ms ごとの `pp_dpm_sclk`、`tg256` 1 run）: C1 は 1685〜1695 MHz の状態が約 15 %、500 MHz の状態は 1.4 %。SGLX（RX 0.5 / WX 0.5）は **500 MHz の状態が約 35〜43 %**、2320 MHz に跳ねる状態もある（約 5 %）。残りは 2 状態が同時に `*` の行（アイドル時の表示）。c1-profile にも「RX は sclk 状態 1（1.6〜1.7 GHz）で上限 2.32 GHz ではない」とある。**クロックが主因という点は、perf high での再測定で確認が必要**（この調査では電力設定を変えていない）。

## 再現手順（perf high の再測定用）

tb250 の作業場所は `~/moe-ep/sglx/`（`env.sh`、`run.sh`、`bench.sh`、`kld.sh`、`queue*.sh` を平らにコピー。リポジトリ内では `scripts/`。`queue*.sh` は `logs/` を作業場所の直下に作る）。プラグインのビルド:

```
cmake -S <repo>/adapters/ggml -B ~/moe-ep/sglx/build-cl11 -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DLLAMA_CPP_SOURCE_DIR=~/moe-ep/src/llama.cpp -DLLAMA_CPP_BUILD_DIR=~/moe-ep/src/llama-build
cmake --build ~/moe-ep/sglx/build-cl11 -j2
```

各測定は `run.sh` を通る（`flock ~/bench/gpu-bench.lock`、全カード 85 ℃未満まで待つ、1 秒刻みのウォッチドッグで孤児を残さない）。`bench.sh TAG SPLIT [llama-bench の引数]`、環境変数 `FORCE_MMVQ=1`（`GGML_VK_FORCE_MMVQ=1`）、`TS=4/1`、`REPS`、`UB`。主な構成:

| tag | コマンド |
| --- | --- |
| C1 | `./bench.sh c1 ""` |
| C1 + MMVQ | `FORCE_MMVQ=1 ./bench.sh c1_mmvq ""` |
| SGLX A | `./bench.sh A "Vulkan0:0.5,Vulkan1:0.5"` |
| SGLX B | `./bench.sh B "Vulkan0:0.7,Vulkan1:0.3"` |
| SGLX C | `./bench.sh C "Vulkan0:0.5,Vulkan1:0.2,CL11_0:0.2,CL11_1:0.1"` |
| SGLX D | `./bench.sh D "Vulkan0:0.4,Vulkan1:0.3,CL11_0:0.2,CL11_1:0.1"` |
| SGLX C + MMVQ | `FORCE_MMVQ=1 ./bench.sh C_mmvq "Vulkan0:0.5,Vulkan1:0.2,CL11_0:0.2,CL11_1:0.1"` |
| ts 4/1 | 上に `TS=4/1` を付ける |

`bench.sh` が実際に使う環境変数（SGLX 構成）: `VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/radeon_icd.json`、`GGML_BACKEND_PATH=~/moe-ep/sglx/build-cl11/libggml-cl11.so`、`GGML_CL11_PLATFORMS=NVIDIA`、`GGML_CL11_DEVICES="GT 730,GT 710"`（CL11_0 = GT 730、CL11_1 = GT 710）、`GGML_CL11_AUTO_RPC=0`、`GGML_SGLX_SPLIT=<上の分割>`、`GGML_SGLX_PROFILE=1`。実コマンド例（C 構成）:

```
llama-bench -m ~/moe-ep/models/LFM2.5-8B-A1B-Q4_0.gguf -dev Vulkan0/Vulkan1/SGLX0 -ngl 99 -fa 0 -t 2 -b 512 -ub 512 -r 5 \
  -sm layer -ncmoe 0 -ts 5/1/0 -ot 'token_embd\.weight=SGLX0' -o jsonl -p 512 -n 128 -pg 512,128
```

各 run のログの先頭 4 行（`# env:`、`# cmd:`）に実際の環境変数とコマンドが入っている。集計は `python3 scripts/summarize.py logs f_`。KLD は `kld.sh`（`queue6.sh`）、test-backend-ops は `queue7.sh`、split 数は `splits.sh`、クロックは `clocks.sh`、VRAM は `memsample.sh`。

## 未検証・限界

- **perf high は未測定**（親が実施）。クロックが主因という結論は perf auto での観測（pp_dpm_sclk と層の区間）に基づく推測で、確認には perf high での tg が要る。
- tg が C1 に届いていない以上、設計書 §8.1 の「既存構成すべてを上回る」は未達（perf auto では）。
- 5 枚（GT 430 の RPC）は対象外。SGLX の NVIDIA（GT 730 / 710）の分担は、C と A の MUL_MAT 時間の差（5.25 → 3.67 ms）に効いているが、tg への効果は RX のクロックに隠れて見えない。
- Vulkan0 の `test-backend-ops` の NaN は ggml-vulkan の問題（実行ごとに出る）。
- WX を専用スレッドから使う方法は、ggml-vulkan の pinned buffer の制約への回避策。ggml-vulkan を変更せずにできる範囲の対処で、本体のスレッドが同じ WX を使うのは sched の同期後なので衝突しないが、ggml-vulkan が複数スレッドからのインスタンス使用を保証しているかは未確認（今回の run では異常なし）。
- 分割比は手で数通り試しただけ（最適化していない）。時間モデル（RX ≈ 1.2 + 3.0·f ms、WX ≈ 0.9 + 8.1·f ms、f はその pdev の行の割合）は perf auto の数点からの当てはめ。
- Mac の MUL_MAT は CL11_0 の型が限られ（418 件）、Metal は 1370 件。

## ファイル

- `scripts/`: `env.sh`、`run.sh`、`bench.sh`、`kld.sh`、`queue1.sh`〜`queue7.sh`、`splits.sh`、`clocks.sh`、`memsample.sh`、`summarize.py`。
- `logs/`: 各 run の `.log`（ヘッダに温度・環境・コマンド）と `.out`（jsonl）。`f_*` が最終行列、`s_ts51_*` が初版、`q2_*`〜`q4_*` が比率と ts の掃引、`kld_*`、`tbo*`、`splits_*`、`k_*.clk`、`m_*.mem`、`prof*`・`ph*`・`x_*` が調査用の短い run。
