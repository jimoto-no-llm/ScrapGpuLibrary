# GT 430 を SGLX0 に参加させる（`REMOTE0`）

TB250、2026-10-22〜23。LFM2.5-8B-A1B Q4_0、llama.cpp `23b0202`、SGL `feat/moe-expert-parallel`。作業場所 `~/moe-ep/g430/`（`srv2/` = scrapgpu-device-server、`plug2/` `plug3/` = プラグイン `-DCL11_WITH_CUDA=ON -DCL11_WITH_REMOTE=ON`）。測定は `run.sh`（flock + 温度待ち）で 1 本ずつ、速度は `perf pin`（`scripts/sess2.sh`、終了時に `perf auto`、`trap EXIT`）。数値は `logs/` の実測。**【推定】** と書いたものだけが実測でない。

## 結論

1. **GT 430 は SGLX0 の物理デバイスとして動く**（`REMOTE0`）。Fermi は Kepler と同じプロセスに入らないので、別プロセスの `scrapgpu-device-server` に開かせ、C API の remote backend で使う。サーバーは SGLX0 が初回に自動起動する（`fork + exec`、プロセスの終了で止まる）。サーバー側は CL11 でも CUDA でも動く。
2. **正しさは通った**。KLD（CPU 基準、`-b 512 -ub 320`）0.020762 / top-1 93.824 %（基準 ≤ 0.0259 / ≥ 91.94 %。SGLX C は 0.020757 / 93.922 %）。自己整合（SGLX 1 枚 vs C+430）KLD 0.000002 / top-1 99.804 %（基準 < 0.001）。`test-backend-ops` は Mac 658/658（6 構成）、tb250 の GT 430 単独 658/658（CL11）・361/361（CUDA）。
3. **速度は、GT 430 を足しても速くならなかった**（下の表。tg128 は C 61.75 に対し C+430 が 60.5〜61.9、差は run 間のばらつき内）。理由は、出力射影の MUL_MAT の所要時間が各デバイスの行数ではなく WX 2100 の経路（x を送る 0.6〜0.8 ms と結果を読む 1.3〜1.6 ms）で決まっているため。GT 430 の担当分（約 7 %）が減らせるのは Vulkan0 などの計算の 0.1〜0.3 ms だけで、そこは律速でない。
4. **勝ち構成が速くなる方向は、GT 430 ではなく WX 2100 の比率を下げること**。WX を外した G / H は C に対して tg +4.7 %（64.7 対 61.75）、pg +3.5 %。GT 430 を入れた G と入れない H は同じ（64.75 対 64.55）。**`GGML_SGLX_AUTOTUNE`（新規）が同じ方向を自動で見つけた**: A（GT 430 入り）から始めて WX を 0.2 → 0.09 へ下げ、tg 64.0 / 64.1、pg 232.5 / 229.9（C は 61.75 / 223.1）。
5. 1 job（write x + op + read）を 1 往復にまとめる `scrapgpu_remote_set_batching` を作ったが、実機の job 時間は変わらなかった（2.39 ms 対 2.40 ms）。ホストの CPU が飽和している訳でもなかった（q13: サーバーは 5.5 %、llama-bench は 65 %）。job が単独時（0.74 ms）の 3 倍以上かかる原因は未特定。

## 実装

| ファイル | 内容 |
| --- | --- |
| `adapters/ggml/composite/pdev_scrapgpu.cpp`（新規） | ScrapGpu C API の remote device を pdev に。重み = `tensor_upload`（サーバーが R32 に詰め直す）、job = `buffer_write(x)` → `op_execute(MUL_MAT)` → `buffer_read`。呼び出しは同期なので、ワーカースレッド 1 本に job を渡し、`submit` は積むだけ、`wait` は完了待ち。`GGML_SGLX_PROFILE` で 1 job の内訳を出す |
| `composite_device.cpp` / `pdev.h` | `REMOTE<i>` の解釈。`pdev::threaded()` が真の pdev を先に submit する。`GGML_SGLX_AUTOTUNE` |
| `adapters/ggml/CMakeLists.txt` | オプション `CL11_WITH_REMOTE`（core と remote client をプラグインに入れる。POSIX のみ） |
| `include/scrapgpu/remote.h`、`src/backends/remote/*` | `scrapgpu_remote_set_batching`（`SGR_OP_BATCH`）。既定 OFF。**macOS のサーバーの不具合修正**: POSIX shm は `ftruncate` が 1 回しか効かないので、1 MiB を超える転送で `ftruncate` が失敗し、サーバーが古いマッピングを捨てて以後の転送が全部失敗していた（Mac の `test-backend-ops` で初めて見えた）。失敗時は古いマッピングを保つ |
| `tools/remote_bench.cpp`、`tests/ops`（`SCRAPGPU_TEST_REMOTE_BATCH`）、`tests/remote`（BATCH の異常系） | 計測と試験 |

使い方（tb250）:

```
GGML_SGLX_SPLIT=Vulkan0:0.45,Vulkan1:0.2,CL11_0:0.2,CL11_1:0.08,REMOTE0:0.07
SCRAPGPU_DEVICE_SERVER=.../scrapgpu-device-server          # 無ければ実行ファイルの隣、次に PATH
GGML_SGLX_REMOTE_BACKEND=opencl11                           # または cuda
GGML_SGLX_REMOTE_ENV="GGML_CL11_PLATFORMS=NVIDIA;GGML_CL11_DEVICES=GT 430;SCRAPGPU_CUDA_DEVICES=GT 430"
GGML_SGLX_REMOTE_SOCKET=/path.sock                          # 起動済みのサーバーに繋ぐ場合
GGML_SGLX_REMOTE_BATCH=0                                    # 往復のまとめを切る
GGML_SGLX_AUTOTUNE=2,16                                     # 比率の自動調整（ラウンド数,測定回数）
```

メインプロセスは `GGML_CL11_DEVICES="GT 730,GT 710"` で GT 430 を外す（`scripts/env.sh`）。

**サーバー側のバックエンド**: GT 430 の q6_K mat-vec は CL11 が 12.5 GB/s、CUDA（sm_20 PTX）が 10.4 GB/s（`logs/q4_rbench_*`、4608 行 x 2048、単独の job は 0.74 ms 対 0.87 ms）。ggml 経由の tg も CL11 62.6 / CUDA 61.1（`q3_C430_*`）。**CL11 を既定にした**。

## 正しさ

| 検査 | 結果 |
| --- | --- |
| Mac `test-backend-ops -b SGLX0 -o MUL_MAT,GET_ROWS`、`MTL0:0.5,REMOTE0:0.5` / `REMOTE0:1` / `CPU:0.3,REMOTE0:0.7` × バッチ ON/OFF | 6 構成すべて **658/658**（サーバーは Apple GPU の CL11。`logs/mac_tbo_*`） |
| tb250 同、`REMOTE0:1`（GT 430 単独、CL11 サーバー） | **658/658**（`q1_tbo_remote0_opencl11`） |
| 同、CUDA サーバー | **361/361**（CUDA は q4_0 / q6_K のみ）（`q1_tbo_remote0_cuda`） |
| 同、`Vulkan0:0.45,Vulkan1:0.2,CL11_0:0.2,CL11_1:0.08,REMOTE0:0.07` | 657/658。失敗は既知の Vulkan0 の NaN（`q4_0 m=40 n=9 k=32`）（`q7_tbo_C430`） |
| KLD（CPU 基準、`-b 512 -ub 320`、4 chunk）C+430（CL11 / CUDA サーバー） | 0.020762 ± 0.001523 / top-1 93.824 %（両方同じ）（`q7_C430_cpu`、`q7_C430cuda_cpu`） |
| 同 SGLX C | 0.020757 / 93.922 %（`q7_C_cpu`） |
| 自己整合（SGLX 1 枚 `Vulkan0:1` の logits を基準に C+430） | KLD 0.000002 / top-1 99.804 %（`q7_C430_self`） |
| `tests/ops` の remote（バッチ ON / OFF）、`tests/remote`（BATCH の異常系） | 通る（Mac、83 件実行） |

## 速度（`perf pin`、REPS=10、`-b 512 -ub 320 -ts 5/1/0`、`-ot 'token_embd\.weight=SGLX0'`）

交互に 2 周（`q10`）、WX 抜きの 3 構成は別の 2 周（`q12`）。表は 2 周（C は 4 本）の値。pp512 / tg128 / pg512,128（t/s）。

| 構成 | 分割 | pp512 | tg128 | pg512,128 |
| --- | --- | --- | --- | --- |
| C1 | （SGLX なし） | 694.1 / 696.3 / 695.2 | 51.6 / 51.6 / 51.5 | 196.5 / 196.4 / 196.2 |
| SGLX C | `V0:0.5,V1:0.2,C0:0.2,C1:0.1` | 700.8 / 700.4 / 700.7 / 695.9 | 61.8 / 61.8 / 62.1 / 61.3 | 223.6 / 223.4 / 223.8 / 221.5 |
| C+430 **A** | `V0:0.45,V1:0.2,C0:0.2,C1:0.08,R0:0.07` | 701.6 / 699.0 | 60.1 / 60.9 | 217.8 / 220.1 |
| C+430 B（GT 710 を GT 430 に替え） | `V0:0.5,V1:0.2,C0:0.2,R0:0.1` | 695.0 / 702.8 | 61.2 / 61.2 | 222.8 / 219.6 |
| C+430 D（WX を 0.15） | `V0:0.45,V1:0.15,C0:0.2,C1:0.08,R0:0.12` | 704.8 / 700.2 | 61.9 / 61.4 | 221.7 / 221.0 |
| A + 自動調整 | A から開始（2 ラウンド、各 16 回） | 699.1 / 695.4 | **64.0 / 64.1** | **232.5 / 229.9** |
| **G**（WX なし、GT 430 あり） | `V0:0.6,C0:0.25,C1:0.05,R0:0.1` | 646.5※ / 698.2 | **64.8 / 64.7** | **230.6 / 229.5** |
| H（G から GT 430 を除く） | `V0:0.65,C0:0.28,C1:0.07` | 703.4 / 701.1 | **64.4 / 64.7** | **229.0 / 230.2** |

※ 1 本だけ外れ値の rep を含む（標準偏差 164）。他は 5〜12。

- A / B / D の tg・pg は C と同等（差は標準偏差内）。**GT 430 を足しても速くならない**。pp も同じ（GT 430 の担当は N=1 の tg で効く設計でなく、pp の出力射影は少数のトークンだけ）。
- C に対する比: H +4.5 %、G +4.7 %（tg）、自動調整 +3.7 %。pg は G / H / 自動調整とも +3〜4 %。**GT 430 の有無（G 対 H）は 0.3 % 以内**。
- 自動調整が落ち着いた比率: Vulkan0 0.64〜0.67、Vulkan1 0.09、CL11_0 0.13〜0.17、CL11_1 0.07、REMOTE0 0.03（`q10_Aauto_*.log`）。最初のラウンドは比例配分（`rows / time`）で、2 回目から `time = a + b * rows` を 2 点で当てはめて全デバイスが同時に終わる比にする。**tg の標準偏差が 6 前後と大きいのは、最初の rep が調整中の呼び出しを含むため**。
- 先頭の `q3`〜`q8` は REPS=3 の予備（`perf pin`）。C 62.1、A（CL11 / CUDA サーバー）62.6 / 61.1、B〜F 60.9〜63.3、G 65.2。

## GT 430 の区間時間（`GGML_SGLX_PROFILE=1`、REPS=3、`perf pin`、N=1 の MUL_MAT）

`worker per job` は pdev のワーカーが 1 job にかけた時間、`done` は MUL_MAT の開始から `wait()` が返るまで（待つ順番は分割の並び順なので、後ろのデバイスは前のデバイスの完了より早くは出ない）。

| 構成（分割） | MUL_MAT 平均 | REMOTE0 の job | write x / op / read | Vulkan1 の done |
| --- | --- | --- | --- | --- |
| C（GT 430 なし） | 3.50 ms | — | — | 3.12 ms |
| A、CL11 サーバー、3 往復 | 3.63 | 2.53 ms | 0.81 / 0.41 / 1.31 | 3.36 |
| A、CUDA サーバー、3 往復 | 3.59 | 2.78 | 0.81 / 0.43 / 1.53 | 3.26 |
| A、CL11、1 往復（バッチ） | 3.76 | 2.39（read に集約） | — | 3.60 |
| A、CL11、3 往復（同じ日の再測定） | 3.41 | 2.40 | 0.67 / 0.40 / 1.33 | 3.19 |
| B（R0 0.1） | 3.56 | 3.30 | — | 3.43 |
| D（R0 0.12） | 3.31 | 2.98 | — | 3.01 |
| G（WX なし、R0 0.1） | **2.85** | 2.09 | — | — |

- **単独（ホストが空いている状態）の job は 0.74 ms**（4608 行 x 2048 の q6_K。write 0.07 + op 0.04 + カーネル 0.55 + read 0.10、`q4_rbench_opencl11_4608`）。ggml の中では 2.4〜3.3 ms で、**3 倍以上**。往復の数を 3 → 1 にしても変わらないので、往復そのものの数ではない。CPU の使用率（`q13_*.cpu`: llama-bench 65 %、サーバー 5.5 %）でも説明できない。原因は未特定（起床の遅延、GT 430 の PCIe / クロックの状態、他デバイスの待ちとの干渉などが候補。**【推定】**）。
- 行数に比例しているわけでもない。R0 の share 0.07 → 0.12 で job は 2.4 → 3.0 ms。
- **律速は Vulkan1（WX 2100）の経路**: `set` 0.6〜0.8 ms + `get` 1.3〜1.6 ms（`logs/q8_*.log`）。計算は 0.1〜0.2 ms。WX を外した G では MUL_MAT が 2.85 ms まで縮み、そのとき REMOTE0 が最後に終わる（GT 430 の job 2.09 ms）。それでも G と H の tg が同じなのは、G の MUL_MAT の短縮が、GT 430 ではなく WX を抜いたことによるため。

## 残課題

- **GT 430 が速度に寄与する構成は見つからなかった**。上限の見積もり: 出力射影は 1 トークン約 15 ms のうち MUL_MAT が約 3.5 ms、GT 430 の 7 % 分はその計算の 0.1〜0.3 ms なので、律速でない限り tg の 1〜2 % が上限。
- 実際に速くなる方向は WX 2100 の経路の短縮（x の送りと結果の読み戻しで約 2.5 ms、ggml-vulkan 側の問題）か、比率から外すこと。**`GGML_SGLX_AUTOTUNE` で自動化できた**（+3.7 %）。ただし、調整中の最初の数十トークンは 1 台ずつ直列で測るので遅い。常用するなら調整結果を保存して次回に使う仕組みが要る。
- 1 job が単独の 3 倍以上かかる理由の特定（`perf` / `strace -T` でサーバーの起床遅延を見る、GT 430 のクロックを `nvidia-smi` で確認する）。
- バッチは Mac では効果が見えない（往復が 12 µs）。GT 430 でも効果なし。remote の他の用途（往復の多い小さな op）のために残してあり、既定は OFF。
- macOS では remote の bulk が 1 MiB 超でソケット経由になる（上限 64 MiB。`ftruncate` の制約）。
- CUDA サーバーの `SCRAPGPU_CUDA_SCHED`（blocking / yield / spin）の影響は測っていない（`q9` は未実行）。

## 再現

`scripts/build.sh`（サーバーとプラグイン）、`scripts/env.sh` + `bench.sh` / `kld.sh` / `sess2.sh`、キュー `q1`〜`q13`（`q1` 機能、`q2` `q3` `q6` `q8` プロファイル、`q4` job の内訳、`q5` 初版の自動調整、`q7` 正しさ、`q10` `q12` 比較、`q11` `q13` CPU 使用率）。`res.py` で `*.out` を 1 行にする。`logs/` に全ログ（`*.log` の先頭にコマンドと環境）。
