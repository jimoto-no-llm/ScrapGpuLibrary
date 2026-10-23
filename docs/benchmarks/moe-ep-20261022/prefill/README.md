# prefill（pp512）の内訳と改善

TB250、2026-10-22。LFM2.5-8B-A1B Q4_0、llama.cpp `23b0202`、SGL `feat/moe-expert-parallel`。測定はすべて `perf pin`（`sgl-driver-switch perf pin`、終了時に `perf auto`、`trap EXIT`）、`~/moe-ep/sglx/run.sh` 方式の flock + 温度待ちで 1 本ずつ（`scripts/`、作業場所 `~/moe-ep/pp/`）。数値は `logs/` の実測。**【推定】** と書いたものだけが実測でない。

## 結論

1. **pp512 は SGLX C（出力射影を 4 枚に分割）を `-ub 320` で動かすと 712 t/s。C1（`-ub 512`、既存の標準）の 662 に対して +7.6 %、C1 を `-ub 320` にした最良の 694 に対しても +2.6 %。** tg128 は 63.5（C1 51.9）、pg512,128 は 231.5（C1 195.5、C1 `-ub 320` 197.6）。pp / tg / pg のすべてで既存構成全部を上回る（REPS=10、C1 と交互に 2 回、下の表）。
2. 効いたのは 2 つ。(a) `-ub 320`（pp512 が 320 + 192 の 2 ubatch になる。C1 でも 662→694〜699）。(b) **SGLX0 の入力コピーを遅延させるコード修正**（`composite_device.cpp`）。SGLX0 は `-ub` を小さくすると C1 より大きく落ちていた（`-ub 256` で 538、C1 は 676）。原因は、出力のない ubatch でも出力射影への入力 `result_norm` をスケジューラが WX から同期コピーし、そのたびにホストが WX の完了を待って次の ubatch の RX 側が重ならなかったこと。`cpy_tensor_async` で受けて、そのテンソルを読む node があるときだけ（同期してから）コピーする。`-ub 256` で 538→695。
3. **プレフィルの律速は WX 2100 の区間**（3 層で約 410 ms、RX の 21 層は約 320 ms）。層ごとの MoE は WX が RX の約 7.5 倍かかる。RX の VRAM が足りず WX の層を RX に移せない（22 層目で約 90 MiB 不足、`-ub 128` でだけ収まるが小さい ubatch は遅い）。
4. **エキスパート並列（NVIDIA に prefill の MoE を分担させる案）は、測定した性能からは採用しない**。GT 730 の MUL_MAT_ID は CL11 で 10〜13 GFLOPS、CUDA（別担当の測定）でも約 67 GFLOPS【32 token 14.0 ms からの換算】で、WX（約 420 GFLOPS）の 16 %、GT 710 を足しても 22 % 前後。GT 730 / 710 は PCIe 2.5 GT/s x1（約 250 MB/s）で、ubatch 512 の活性（4 MiB）の往復だけで層あたり数十 ms かかる。実装していない（理由は下）。
5. 正しさ（§8.1）: KLD（CPU 基準）0.0215（`-ub 128`）/ 0.0205（`-ub 512`）≤ 0.0259、top-1 92.65 % / 94.41 % ≥ 91.94 %、自己整合（SGLX 1 枚 vs C）KLD 0.000002・top-1 100 %。

## 最終比較（`perf pin`、REPS=10、`-b 512`、`-ts 5/1`（SGLX は 5/1/0）、交互に 2 回、`logs/q28_*`）

| 構成 | pp512 | tg128 | pg512,128 |
| --- | --- | --- | --- |
| C1 `-ub 512` #1 / #2 | 662.2 ± 5.0 / 661.5 ± 4.9 | 51.86 / 51.97 | 195.4 / 195.6 |
| C1 `-ub 320` #1 / #2 | 694.0 ± 6.6 / 694.2 ± 6.5 | 51.84 / 51.79 | 197.7 / 197.5 |
| SGLX C `-ub 512` #1 / #2（修正後） | 676.3 ± 5.4 / 676.0 ± 5.2 | 63.18 / 63.49 | 227.8 / 228.0 |
| **SGLX C `-ub 320`** #1 / #2 | **712.4 ± 6.0 / 712.1 ± 6.0** | **63.58 ± 0.71 / 63.41 ± 0.81** | **231.5 ± 0.9 / 231.6 ± 1.2** |

SGLX C = `GGML_SGLX_SPLIT=Vulkan0:0.5,Vulkan1:0.2,CL11_0:0.2,CL11_1:0.1`、`-ot 'token_embd\.weight=SGLX0'`、`-dev Vulkan0/Vulkan1/SGLX0`（前回と同じ）。コマンドは `scripts/bench.sh`（`UB=320 REPS=10 PROFILE=0 ./bench.sh TAG "$C"`）。下限（SGLX C の tg 62 / pg 223）以上。前回の SGLX C（`-ub 512`、修正前）は pp 670 / tg 62.0 / pg 223、今回の `-ub 512` の 676 / 63.3 / 228 との差は、イベント / 遅延コピーのコード変更と run 間差（未切り分け）。

`-ub` の細かい掃引（SGLX C、pp のみ、REPS=5、`logs/q27_*`）: 192 → 652、224 → 644、256 → 694、288 → 689、304 → 694、**320 → 717**、336 → 693、352 → 686、384 → 682、416 → 651、448 → 652、512 → 676。320 だけが突出している（64 の倍数が 5 つ + 3 つの 2 ubatch）。C1 でも同様（`logs/q14_*`: 304 677、312 681、**320 699**、328 684、336 682、352 675、376 666）。理由は未特定（タイルの端数と思われるが確認していない）。**`-ub 320` は pp512 向けの値**で、他の長さのプロンプトで効くかは測っていない。

## 1. pp の内訳（C1、`-ts 5/1 -b 512 -ub 512`、pp512 1 pass）

### 演算別（`GGML_VK_PERF_LOGGER=1` + 同期 1 行パッチ版 `~/moe-ep/c1p/bin-patched`、`logs/prof_c1_pp512.log`、3 pass の安定値）

ロガーは演算ごとに同期するので合計は実時間より約 10 % 大きい。

| 項目 | RX 6400（21 層: MoE 19 + dense 2） | WX 2100（3 層 + 出力射影 1 token） |
| --- | --- | --- |
| 合計 | 401〜403 ms | 314〜324 ms |
| MUL_MAT_ID（gate/up + down） | 177.1 + 98.8 = **275.9 ms（69 %）** | 149.7 + 67.9 = **217.6 ms（67 %）** |
| MUL_MAT（q4_0: conv / attn / dense の射影） | 約 65.7 ms（16 %） | 約 66.3 ms（21 %） |
| MUL_MAT（f16 attention、f32 router） | 約 11 ms | 約 6.4 ms |
| 出力射影 q6_K（1 token） | - | 5.6 ms |
| その他（MUL, GLU, CONCAT, ADD, RMS_NORM, SOFT_MAX, SSM_CONV ...） | 約 48 ms | 約 26 ms |
| MoE 1 層あたり | 約 14.5 ms（3.2〜2.9 TFLOPS 相当） | 約 109 ms（0.40〜0.46 TFLOPS 相当） |
| q4_0 dense MUL_MAT の速度 | 約 6.2 TFLOPS | 約 0.68 TFLOPS |

支配的な op は **MUL_MAT_ID（両デバイスとも 67〜69 %）**。attention（f16 MUL_MAT + SOFT_MAX + RMS_NORM_ROPE）は RX で約 20 ms、全体の 5 %。

### 時間軸（`LD_PRELOAD` の Vulkan 計測 shim、`-ub 512`、`logs/q8_tl_ub512.txt.ev.gz`、`scripts/tl.py`）

1 pass 768 ms は、RX の実行 約 317 ms → 隠れ状態 4 MiB の受け渡し 約 15 ms → WX の実行 約 412 ms（最後のチャンクのフェンスを待つ約 80 ms を含む）→ logits の取得など約 23 ms の直列。**RX の稼働は pass の約 41 %、WX は約 54 %** で、2 枚は同時には動かない（C1 の層分割）。`gpu_busy_percent` の 10 ms サンプル（`logs/q4_busy_*.csv`）でも WX は pass の約 42 % で、RX の値は平滑化されていて使えなかった。WX の区間は 1 層動かすごとに約 135 ms（`-ts` 3/1 → 4/1 → 5/1 の pp 491 → 561 → 664 から）。

ホスト CPU は律速でない: pp の `llama_decode` 全体で CPU 時間は約 45 ms/pass（SIGPROF、`logs/q7_sprof_ub256.txt.gz`）。

## 2. 設定だけで試したもの（C1、pp512、REPS=3〜5）

| 設定 | pp512 | 備考 |
| --- | --- | --- |
| `-ub 512` / 256 / 128 | 664 / 676 / 559 | 小さい ubatch は MUL_MAT_ID の効率が落ちる（RX の 1 ubatch の実行時間が 256 token で 254 ms、512 token の 317 ms の半分にならない） |
| `-ub 288 / 304 / 312 / **320** / 328 / 336 / 352 / 384 / 448` | 675 / 677 / 681 / **699** / 684 / 682 / 675 / 670 / 637 | 上記 |
| `-ot no_such_tensor_zz=CPU`（パイプライン並列を切った対照） | `-ub 256` 678、`-ub 320` 699 | C1 と同じ。**この llama.cpp では `-ub` を小さくしてもパイプライン並列の利得は無い**: 同じ形の ubatch が続くとグラフを再利用し、その前に `ggml_backend_sched_synchronize`（全デバイスの完了待ち）が入る（`llama-context.cpp` の `graph_reuse` 経路）。`LLAMA_GRAPH_REUSE_DISABLE=1` にしても 256 で 675、192 で 636、128 で 559 と変わらない |
| `-ts` 3/1 / 4/1 / 6/1 / 7/1 | 491 / 561 / **139** / 139 | 6/1 は RX の VRAM 溢れ。下 |
| `-ts 6/1 -fa 1`（`-ub` 512 / 384 / 256） | 167 / 167 / 165 | `-fa 1` で RX の計算バッファ 122→112 MiB にしかならず溢れる。tg は 53.5（5/1 `-fa 0` は 51.3） |
| `-ts 6/1` `-ub` 256 / 192 / 128 / 96 | 144 / 141 / **572** / 435 | `-ub 128` でだけ収まる。5/1 の `-ub 128` 558 との差は +2.5 % |
| `-fa 1`（5/1） | 673（pp）/ tg 45.0 | tg が下がる |
| `GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM=1` | `-ub 512` 657、256 670、128 548 | tg も 50.7→48.6 に下がる。効かない |
| `GGML_VK_*` の coopmat 系 | 未測定 | どちらのカードも `matrix cores: none` |
| `-ncmoe` | 未測定 | tg が CPU 計算になり下限を割る（前回のベースラインで 21 t/s 級） |

`-ts 6/1` の VRAM（`GGML_VK_MEMORY_LOGGER`、`logs/q9_mem_*`）: RX のモデルバッファは 5/1 で 3816 MiB、6/1 で 4011 MiB。計算バッファは `-ub 512` で 122 MiB（`-ub 256` で 61 MiB、`-fa 1` で 112 / 56 MiB）、空きは 4064 MiB。22 層 + `-ub 512` で合計 4.05 GiB を超えて GTT に溢れ、pp が 1/4 になる。

## 3. 原因の切り分け（SGLX0 が `-ub` で落ちる）

- SGLX0 をデバイスに加えるだけ（`-ot` は無関係なテンソル）は C1 と同じ（`logs/q20_sglxnoop_*`: `-ub` 256 678、512 666）。`-ot 'token_embd\.weight=SGLX0'` で出力射影を SGLX0 に載せると `-ub 256` で 538。SGLX0 に載せる比率を `Vulkan1:1`（WX だけ）や `Vulkan0:1` にしても 543 / 483 で、射影の計算ではなく **SGLX0 のスプリットが入ること** が原因（`logs/q25_*`）。SGLX0 自身の `graph_compute` は 1 ubatch で数 ms（`GGML_SGLX_TRACE`）。
- 出力のない ubatch（pp の最後以外）では SGLX0 の MUL_MAT は空だが、スケジューラは入力 `result_norm` のコピーのために `ggml_backend_synchronize(WX)` を呼ぶ。C1 / ノーオプにはこの同期が無く、次の ubatch の RX を WX の実行と重ねて投入できる（shim の時間軸: SGLX0 あり `-ub 256` は RX0 → WX0 → RX1 → WX1 が直列で 1 pass 約 930 ms、ノーオプは約 750 ms）。
- 修正: `backend_cpy_tensor_async` で入力コピーを保留し、同じ `graph_compute` の node がそのテンソル（またはその view）を読むときだけ、元デバイスを同期してからコピーする。最初の版は同期を忘れて N > 1 の logits が壊れた（KLD 9.05）。同期を入れて直した（コミット `6d5ab8f`）。`GGML_SGLX_LAZY_COPY=0` で旧動作。
- 併せて SGLX0 に `async` / `events` の能力と空の event を入れた（llama.cpp はパイプライン並列を全デバイスが持つときだけ有効にする）。ただし `-ot` があるとパイプライン並列は起きない。`-ot` を使わず `-ts 0.82/0.12/0.06` で出力層を SGLX0 に回す試し（`logs/q15_C_noot_*`）は、tied 埋め込みが Vulkan_Host（ピン止めホスト）に置かれて Vulkan0 が計算し、pp 134 / tg 10.8 になったので不採用。
- `GGML_SGLX_HOSTBUF=1`（SGLX0 のバッファをホスト扱いにして GET_ROWS を CPU に任せる）は `-ub 256` で 542 のまま。効かない（オプションとして残した）。

## 4. エキスパート並列を実装しなかった理由（測定から）

- 余っているデバイスは WX の区間中の RX、RX の区間中の WX、常時 GT 730 / GT 710。RX と WX は VRAM が満杯で、相手の層のエキスパートを持てない（RX の空きは 22 層目に約 90 MiB 足りない）。
- GT 730 の MUL_MAT_ID（q4_0、n_mats 128、n_used 8、m=768、k=2048）: CL11 は n=1〜8 で約 92〜99 GFLOPS（mat-vec 経路）、n=32 で 9.9、n=64 で 13.1 GFLOPS（`logs/q13_mmid_CL11_0.out`）。CUDA は 32 token の 1792×2048 エキスパート 4 個で 14.0 ms（CL11 54.6 ms、`../cuda/README.md`）で、換算で約 67 GFLOPS【換算は推定】。WX の MUL_MAT_ID は約 420 GFLOPS（上の表）。
- 活性の転送: NVIDIA 3 枚のリンクは 2.5 GT/s x1（`/sys/class/drm/card*/device/current_link_speed`）。ubatch 512 の隠れ状態は 4 MiB（1 往復で 8 MiB）で、層ごとの往復だけで RX の 1 層（14.5 ms）や WX が NVIDIA に譲れる時間（WX の 1 層 109 ms の 15〜22 % ≒ 16〜24 ms）より大きくなりうる【リンク実効速度は未測定、250 MB/s と仮定した推定】。
- よって NVIDIA 側の分担で縮められるのは多くても WX 区間の数 % で、層ごとの同期と転送で相殺される見込み。decode（N = 1）は従来どおり出力射影の分割だけを担当する（tg 63.5）。

## 5. 正しさ

| 検査 | 結果 |
| --- | --- |
| KLD（CPU 基準、`-ub 128`、`kld.sh`、4 chunk） | C1 0.021464 / top-1 92.84 %。SGLX C（修正後）0.021504 / 92.65 %（`logs/pp2_C128`）。基準 ≤ 0.0259 / ≥ 91.94 % を満たす |
| 同 `-b 512 -ub 192` / `256` / `512` | 0.020880 / 93.53 %、0.021504 / 92.65 %、0.020535 / 94.41 % |
| 自己整合（SGLX 1 枚 vs C / vs A（RX 0.5 + WX 0.5）、`-ub 128`） | 0.000002 / 100 %、0.000002 / 100 %（`logs/pp3_*`）。基準 < 0.001 |
| 貪欲生成 24 token（約 2 KB のプロンプト、`-ub` 320 / 512） | C1 と SGLX C（320 / 512）で同一（`logs/pp7_gen_*`）。**補助確認**。生成の一致は正しさの基準ではない |
| `test-backend-ops -b SGLX0 -o MUL_MAT,GET_ROWS`（C 構成、tb250） | 656 / 658。失敗 2 件は Vulkan0 の NaN（`q4_0 m=40 n=9 k=32`、`m=100 n=9 k=96`）で、前回と同じ ggml-vulkan の既知の問題。Mac の実行は今回していない |
| 遅延コピーを入れた最初の版 | KLD 9.05 / top-1 11.9 %（同期の抜け）。修正前のコミットは使わないこと |

**【2026-10-22 追記】この問題は修正済み。原因と結果は `../ub320-fix/README.md`（`pdev_ggml` のグラフキャッシュが、reserve で解放された計算バッファを指したままだった）。** 以下は修正前の記述:

**既知の問題（今回の変更前から）**: 全 token の logits を取る `llama-perplexity -b 512 -ub 320` は `pdev_ggml` の Vulkan `set_tensor_async` の `GGML_ASSERT(... unsupported buffer type)` で落ちる（旧プラグインでも同じ、`logs/pp6_old320`）。`-ub` 192 / 256 / 512 では落ちない。pp512 の測定（出力は最後の 1 token）には影響しない。原因は未特定（スレッド競合ではない: `GGML_SGLX_THREADS=0` でも落ちる）。

## 6. 統合した CUDA pdev

`../cuda/integration.patch` を適用した（コミット `99d0789`）。`GGML_SGLX_SPLIT=...,CUDA0:x,CUDA1:y` と `SCRAPGPU_CUDA_DEVICES="GT 730,GT 710"`、`-DCL11_WITH_CUDA=ON`。この版は今回 `CL11_WITH_CUDA=OFF` でビルド・測定した（CUDA 版は別担当の README で SGLX C と同等）。CUDA を入れたビルドでの `-ub 320` は未測定。

## 7. 残課題

- WX の区間（約 410 ms）が律速。RX の VRAM を約 90 MiB 空けて 22 層目を RX に載せれば、pp は +17 % 前後になる見込み【`-ts` の傾きからの推定】。tg も約 +4 %。方法は未解決（`-ot` で一部テンソルを WX に置くと層内に境界が増え、tg が下がる）。
- `-ub 320` が突出する理由、ubatch が小さいと MUL_MAT_ID の効率が落ちる理由（ggml-vulkan のタイル選択と思われる）は未確認。そこを直せば 2 ubatch のパイプラインで WX と RX を重ねる余地がもっと出る。
- `-ot` を使うとパイプライン並列は切れる（この llama.cpp では同じ形の ubatch では元から効かない）。
- GT 730 / 710 の PCIe 実効帯域は未測定。
- `GGML_VK_FORCE_MMVQ=1`（tg +2 %）と組み合わせた KLD は未測定。

## ファイル

- `scripts/`: `env.sh`、`run.sh`、`bench.sh`（`UB` `B` `FA` `TS` `NOOT` `REPS`）、`sess.sh`（pin → キュー → 必ず auto）、`kld.sh`、`gen.sh`、`summarize.py`、`tl.py`（shim の時間軸）、`q*.sh`（各実行のキュー）、`prof1.sh`。
- `logs/`: 各 run の `.log`（ヘッダに温度・環境・コマンド）と `.out`。`q1`〜`q37` が実行順、`prof_c1_pp512` が内訳、`q8_tl_*` `q22_tl_*` `q18_tl_*` が shim の時間軸（`.ev.gz`）、`pp*_` が正しさ。
- コード: `adapters/ggml/composite/composite_device.cpp`（events / 遅延コピー / `GGML_SGLX_LAZY_COPY` `GGML_SGLX_HOSTBUF` `GGML_SGLX_TRACE` / graph_compute 計測）。プラグインのビルドは `~/moe-ep/pp/build-cl11`（`git archive HEAD` + 差分）。
