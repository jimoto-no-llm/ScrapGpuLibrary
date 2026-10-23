# 段階 6・7: ドライバ切替の実測

状態: **中断（2026-10-23）。最初の `nv390`（ベースライン前の正規化）で停止条件に該当したため、計測は 1 つも行っていない。**
再起動など復旧は利用者判断（`logs/00_first_nv390/after_nv390_diagnosis.txt`）。

## 経緯

1. 切替前: `status` は 3 枚とも `nvidia`、NVRM 390.157、libcuda 390.157、`state: unknown (boot default)`、`userspace.installed` なし。`dmesg` と `journalctl -k` は一般ユーザでは読めない（journalctl は adm / systemd-journal グループ外で「No entries」）。llama / bench 系プロセスなし。
2. 同一セッションの 390 ベースライン用に `flock ~/bench/gpu-bench.lock sudo -n sgl-driver-switch nv390` を実行（状態ファイルを作るため）。
3. 結果（停止条件）:
   - ユーザ空間は 390.157 の再インストールで libcuda → 390.157。
   - **カーネルモジュールは 470.256.02 がロードされている**（`/proc/driver/nvidia/version`）。`nvidia_uvm` / `nvidia_modeset` / `nvidia_drm` はロードされていない。GT 730 / GT 710 は `nvidia`（470）に束縛、GT 430 は未束縛（470 は Fermi 非対応）。`nvidia-smi` は `Driver/library version mismatch`。
   - 状態ファイルは `failed(2) at line …`。`/run/sgl-driver.state` は切替スクリプトの終了時には `nv390` だったが、直後の `status` 内の ERR trap で上書きされた。
   - D 状態のプロセスなし、GPU ハングの兆候なし（ssh・コマンドは正常応答）。
4. 以後、切替・再切替・モジュール操作は行っていない（指示の停止条件）。

## 切替スクリプトの不具合候補（読み取りのみで確認、未修正）

- `status()` が ERR trap を継承する（`set -E`）。`ls /sys/bus/pci/devices/$d/drm`（ノードが無いとき rc=2）と `awk … /proc/driver/nvidia/version`（モジュール未ロードのとき rc=2）が `pipefail` で失敗し、`status` を呼ぶだけで `failed(2)…` が状態ファイルに書かれる（直接 `status` を実行しても再現）。`|| true` を足すか、`status` 内で trap を無効にする必要がある。
- nv390 後に 470 のモジュールがロードされていた原因は未特定（ログが読めない）。候補: (a) 390 のロード後にユーザ空間インストール／`nvidia-modprobe -u -c=0`、またはカーネルのエイリアス自動ロードが 470 を先に載せた（`lspci -k` では 390 と 470 の両方が全カードの候補モジュール）、(b) `modprobe` の依存解決。切替スクリプト内で `modprobe nvidia-legacy-390xx` 後に `/sys/module/nvidia/version` を読み戻して検証する処理が無いことが検出漏れの原因。

## 利用者に頼むこと

- 再起動するか、`sudo sgl-driver-switch nv390` をもう一度実行して 390 に戻すか（今回はそれ以上触っていない）。戻らなければ再起動（ブート既定は 390）。
- `nv390` が 470 を載せる原因を調べるため、`sudo journalctl -k`／`dmesg` の `nvidia` 行を共有してほしい（この権限では読めない）。
- 上記の `status` の ERR trap 修正を許可するなら、`tools/tb250/sgl-driver-switch` を直して再 install する。

## 再開 1: 再起動後の nv390 往復確認（2026-10-23）— 失敗、停止

- 再起動後の `status`: 390.157、userspace・モジュール一致、GT 3 枚とも `nvidia-smi` に表示。修正版スクリプト install 済み。
- `flock … sudo -n sgl-driver-switch nv390` の出力: `rmmod nvidia_drm` → `modprobe: FATAL: Module nvidia_drm not found.` → `FAILED at line 82 (rc=1)`、状態ファイルは `failed(1) at line 82: modprobe -r "$m"`。
- 実機の状態は変化なし（390.157 のまま、`nvidia-smi -L` で 3 枚、`/sys/module/nvidia/version` = 390.157）。unload は 1 つ目のモジュールで止まり、何も外れていない。
- 原因（推定）: lsmod 上の名前は `nvidia_drm` だが、モジュールファイルは `nvidia-legacy-390xx-drm.ko.xz` で、`modprobe -r nvidia_drm` は名前解決できない（`modinfo` で確認）。`unload_all` は `rmmod`（カーネル名）か、390 / 470 のファイル名に直す必要がある。前回の nv390 で 470 が載った経緯も、この unload が実は効いていなかった可能性がある（未確認）。
- 停止条件（切替が failed を残す）に該当したため、これ以上の切替は行っていない。段階 C / A / B は未実施。

## 再開 2: 修正版スクリプトでの往復と段階 C（2026-10-23）— 段階 C の切替で停止

### nv390 往復確認: 通過
`rmmod` 版 `unload_all` で `nv390` は成功（3 枚 nvidia、390.157、userspace 一致、`nvidia-smi -L` で 3 枚、state=`nv390`）。

### 390 ベースライン（同一セッション、`bench_cl11_cards.sh NVIDIA gt730 gt430`、LFM2.5-1.2B Q4_0、`logs/03_baseline_nv390/`）
| カード | q6_K mv | q4_0 mv | MUL_MAT_ID 32 tok (q4_0) | pp128 | tg64 |
| --- | --- | --- | --- | --- | --- |
| GT 730 | 28.19 GB/s | 28.05 GB/s | 4.82 GB/s | 121.35 ± 0.37 | 31.98 ± 0.01 |
| GT 430 | 14.45 GB/s | 18.57 GB/s | 1.02 GB/s | 45.03 | 22.41 ± 0.01 |

### 段階 C: `nv470-nouveau430` — 目的の状態に到達せず
- `flock … sgl-driver-switch nv470-nouveau430` は rc 0、state=`nv470-nouveau430` を書いたが、**実際は `nvidia` モジュールが無く（lsmod に nouveau のみ）、3 枚とも未束縛、DRM ノードなし**。スクリプトは途中で `nvidia kernel module 470.256.02` を表示している（load 直後の版確認は通った）ので、ロードは成功し、その後（ユーザ空間 `.run` のインストール中か直後）に nvidia が外れたと読める。**同じ現象が `nv390` でも起きた**: 初回の再起動前 `nv390`（最初の失敗）で 470 が残っていた件、および今回の nv470-nouveau430 → nv390 で、ユーザ空間 `.run` を実際に入れた直後に `nvidia` が無い状態（`kernel module=not loaded`）になった。ユーザ空間のインストールをスキップした呼び出し（marker 一致）では起きていない。→ `nvidia-installer` 側（`--no-kernel-module` でもロード済みモジュールを外す／入れ替える挙動、または付随する `nvidia-modprobe`）の疑い。未確認（ログ非可読）。
- 計画どおり nv390 で戻したが、1 回目は上記の理由で module 未ロード（marker 不一致のため userspace を再インストールしたため）。2 回目（userspace スキップ）で 390.157 がロードされ、GT 730 / GT 710 は束縛・`nvidia-smi -L` に表示。
- **ただし GT 430 は `Unable to determine the device handle for GPU 0000:04:00.0: The NVIDIA kernel module detected an issue with GPU interrupts`**。その結果 `nvidia-smi`（全体）と OpenCL（CL11 plugin、GT 730 指定でも）が `SKIP: no device`（mm_bench rc=77）になる。GT 430 に nouveau を一度掴ませた後で 390 を載せ直したこと（または module 入れ替えの繰り返し）で GT 430 が割り込み異常になった疑い。**再起動で戻る見込み（推定）。再起動は利用者判断。**
- 停止条件（nv390 で正常に戻らない）に該当。段階 A / B、`pstate`、probe_state は未実施（段階 C の probe も取れていない）。perf は auto のまま。

### 現在の状態
state=`nv390`、`/sys/module/nvidia/version`=390.157、userspace 390.157、GT 430 が割り込み異常、GT 730 / 710 は nvidia-smi 上は見えるが OpenCL 実行不可。

### 次にやること（利用者の判断待ち）
1. 再起動（ブート既定の 390 で GT 3 枚が戻るはず）。
2. スクリプト修正案: (a) ユーザ空間 `.run` のインストール**前**にモジュールを load せず、インストールの**後**に load する順にする（installer がモジュールを外す疑いを回避）。(b) 各切替の最後で `/sys/module/nvidia/version` と各カードの束縛先を読み戻し、期待と違えば `failed` にする。(c) nv470-nouveau430 は GT 430 を nouveau に掴ませる前に、userspace を先に入れる。

## 再開 3: 修正版（userspace 先・load 後・expect_bound）での再試行（2026-10-23）— 段階 C で停止

- `nv390 --reinstall-userspace` の往復: 通過（390.157、3 枚束縛、`nvidia-smi -L` で 3 枚）。userspace を実際に入れ直してもモジュールは外れなかった（前回の「module が消える」現象は修正版で解消）。
- 段階 C `nv470-nouveau430`: 470 のロードと Kepler 2 枚の束縛は成功（`/sys/module/nvidia/version`=470.256.02、userspace 470.256.02、`nvidia-smi -L` に GT 730 / GT 710）。**GT 430 に nouveau が束縛されなかった**（`expect_bound`: `0000:04:00.0 is bound to 'driver', wanted nouveau`、state=`failed(1) at line 221`）。`driver_override`=nouveau、`nouveau` モジュールはロード済み（lsmod に `nouveau`、`/sys/bus/pci/drivers/nouveau` あり）、`modeset` は `/sys` から一般ユーザで読めない。probe が黙って失敗した（原因はカーネルログが読めず不明）。`logs/04_stage_c_retry/after_c_failed.txt`。
- 戻し: `nv390` は state=`nv390`、3 枚束縛、モジュール 390.157 だが、**GT 430 は再び `The NVIDIA kernel module detected an issue with GPU interrupts`**（`nvidia-smi -L` に GT 730 / GT 710 のみ）。nouveau を GT 430 に掴ませようとして失敗した後に 390 を載せるとこうなる。再起動が要る兆候として、指示どおりここで停止。段階 A / B、probe、pstate は未実施。
- 所見: GT 430 の nouveau probe 失敗は、(a) `driver_override` 経由の bind が nouveau ロード時に走らない、(b) GF108 の nouveau probe 自体が失敗（ファームウェア／modeset=1 の fbcon 関連等）、のどちらか。(a) なら `echo 0000:04:00.0 > /sys/bus/pci/drivers/nouveau/bind` が要る（`drivers_probe` でも可）。`journalctl -k`／`dmesg` の nouveau 行が必要。

## GT 430 after the nouveau attempts (2026-10-23, user's journal)

After the GT 430 was handed to nouveau (stage C, twice) and the machine went back to 390, every open of the GT 430 logs `NVRM: RmInitAdapter failed! (0x12:0x45:1920)` / `rm_init_adapter failed for device bearing minor number 0` with an `nv_close_device` → `rm_disable_adapter` → `os_flush_work_queue` trace (`logs/05_gt430_journal/`). Only a reboot cleared it. The nouveau probe messages for the GT 430 were not in the captured tail. Decision (user, Q39 = a): the GT 430 is left unbound in the remaining stages; 470 and nouveau / NVK / rusticl are tried on the Kepler cards only (`nouveau-kepler` state in the switch script).

## 再開 4: 方針変更後（GT 430 はどのドライバにも束縛しない）の実測（2026-10-23〜24）

方針（利用者 Q39 = a）: GT 430 は `none`。`nv470-nouveau430` と `nouveau`（3 枚）は使わない。新状態 `nouveau-kepler`（GT 730 / GT 710 を nouveau）。順序は nv390 往復 → 段階 B → nv390 → 段階 A（Kepler）→ nv390 → pstate（GT 710 のみ）。
各切替は rc 0、`expect_bound` も通り、GT 430 に割り込み障害は出ていない。

### 段階 B: `nv470`（Kepler 2 枚 = 470.256.02、GT 430 = none）— 動いた

認識（`logs/05_stage_b_vs_390/drivers/logs/b_nv470/`）: `nvidia-smi` に GT 730 / GT 710、OpenCL（`libnvidia-opencl`）プラットフォーム "NVIDIA CUDA" に 2 枚、Vulkan（470 の ICD）に **2 枚とも出る（apiVersion 1.2.175、driverName NVIDIA）**。390 では Vulkan ICD が落ちる（HARDWARE.md）のと違い、470 の `vulkaninfo --summary` は rc 0。ggml-vulkan の動作は未測定（この段階では測っていない）。

**自前 CUDA（470、`bench_cuda470.sh`）**
- `scrapgpu_cuda_kernel_tests`: GT 730 / GT 710 とも **161 passed, 0 failed**。`scrapgpu_ops_tests`（CUDA）: 8 op runs、75 skipped（CUDA 未実装の op。390 側と同じ構成のため差は無い）。
- 帯域（mm_bench、`cuda`）: 下表。390 の値は同日の `bench_cuda390.sh`（GT 730 の mv / mid32、`logs/05_stage_b_vs_390/sglx/logs/drv_cuda390_*`）。

| 項目 | GT 730 / 470 | GT 730 / 390 | GT 710 / 470 |
| --- | --- | --- | --- |
| mv q4_0 | 32.99 GB/s | 33.01 GB/s | 10.78 GB/s |
| mv q6_K | 26.26 GB/s | 26.38 GB/s | 8.97 GB/s |
| MUL_MAT_ID 32 tok (q4_0) | 19.19 GB/s | 18.91 GB/s | 7.55 GB/s |

  → 470 と 390 は同等（差は 1.5 % 以内）。
- SGLX C（CUDA、LFM、`-p512 -n128 -pg 512,128`、perf は auto、5 回）: 470 は pp512 635.8 ± 8.2 / tg128 31.2 ± 0.4 / pg 129.4 ± 0.7、390 は 643.7 ± 5.1 / 30.6 ± 0.4 / 127.9 ± 1.4。**差は標準偏差内**。（10/22 の C = tg 62 とは perf pin の有無などで条件が異なる。同日の 390 と比較すること。）

**CL11（470 の OpenCL 1.2、`bench_cl11_cards.sh NVIDIA gt730 gt710`、LFM2.5-1.2B Q4_0）**

| カード | 項目 | 470 | 390（GT 730 のみ同日） |
| --- | --- | --- | --- |
| GT 730 | q6_K mv / q4_0 mv | 28.64 / 28.34 GB/s | 28.19 / 28.05 GB/s |
| GT 730 | MUL_MAT_ID 32 tok | 4.67 GB/s | 4.82 GB/s |
| GT 730 | pp128 / tg64 | 118.32 ± 0.27 / 32.54 ± 0.01 | 121.35 ± 0.37 / 31.98 ± 0.01 |
| GT 710 | q6_K mv / q4_0 mv | 10.25 / 10.55 GB/s | （390 未測定） |
| GT 710 | pp128 / tg64 | 59.85 ± 0.22 / 13.32 | （390 未測定） |

  → GT 730 は 470 と 390 で同等（pp128 −2.5 %、tg64 +1.7 %）。

**勝ち構成との比較（perf pin、`bench_cl11_470.sh` / `q_n_drv.sh`）**
- 構成 H（LFM、2 回交互）: tg128 470 = 51.3 ± 4.6 / 49.3 ± 4.7、390（同日）= 55.5 ± 6.7 / 49.6 ± 3.8。pg512,128 470 = 186.9 / 188.9、390 = 190.0 / 187.9。C1 は 470 = 45.2 / 45.2、390 = 45.1 / 45.4。**470 と 390 の差は標準偏差内**。ただし**今日の H 自体が記録の tg 69 / pg 240 より遅い**（C1 も記録 51.8 に対し 45）。原因: perf pin が安定していない（下の注）。記録の勝ち値との直接比較はしない。
- 構成 N（30B、`hec/cfg_n.sh`、pp512 / tg128 / pg512,128、3 回）: 470 = tg128 13.66 / 13.37、390 = 9.93 / 9.93（1 回目の 390）、**390 を再測定したところ 13.25 / 13.61**（`drv390b_N_*`）。→ **470 ≈ 390 ≈ 13.4**（記録の 14.63 より少し低い）。最初の 390 の 9.9 は perf pin が RX 6400 / WX 2100 を別のクロックに固定した結果の外れ値（pin の表示は毎回違った）。
- 注（perf pin）: `sgl-driver-switch perf pin` は表示される sclk が実行ごとに違う（RX 6400 が `0Mhz *` 2 行、`500Mhz`、WX 2100 が 734 / 1098 / 1219 MHz）。AMD 側の pin の再現性が低いので、**絶対値は記録と比べず、同一セッションで交互に測った差だけを見る**。H / N の「勝ち」の判断（標準偏差超え）は 470 でも 390 でも出ていない。
- 結論（B）: 470 は 390 と同等で、速くならない。想定どおり Kepler 側の速度は全体にほとんど効かない。ggml-cuda（段階 D）には進んでいない。

### 段階 A（Kepler 版）: `nouveau-kepler` — 認識はするが、計算が動かない

認識（`logs/06_stage_a_nouveau_kepler/drivers/logs/a_nouveau_kepler/`）:
- rusticl（`RUSTICL_ENABLE=nouveau`）: **2 枚とも出る**（デバイス名 `NV106`、OpenCL 3.0 / OpenCL C 1.2、Max clock 512 MHz、Max compute units 2）。Clover（`libMesaOpenCL`）は AMD の 2 枚だけで nouveau のカードは出ない。
- NVK（`NVK_I_WANT_A_BROKEN_VULKAN_DRIVER=1` あり）: 2 枚とも出る（`GeForce GT 730 (NVK GK208B)`、`NVK GK208B`、apiVersion **1.0.305**）。環境変数なしでは出ない（`vk-nvk-plain` rc 1）。ggml-vulkan の列挙順は GT 710（2176 MB）が index 0、GT 730（1152 MB）が index 1（`bench_vk_nvk.sh gt730` のログ名 `drv_nvk_gt730_*` は index 0 = **GT 710** を測っている）。
- nvidia の Vulkan ICD（`VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json`）は `stack smashing detected` で rc 134（390 のユーザ空間のまま nouveau 状態にしたため）。

**rusticl + CL11 プラグイン: 動かない**（`drv_nouveau-kepler_rusticl_gt730*`）
- mm_bench（mv q4_0 / q6_K / mid32）と llama-bench が全部、カーネルのビルド時に `ERROR: unknown nir_op f2f16` / `nvc0_program_translate:659 - shader translation failed: -2` を出して rc 136 で落ちる。
- **最小再現**（`repro/`）: `__kernel void k(__global const float*a,__global float*o){size_t i=get_global_id(0); vstore_half(a[i],i,(__global half*)o);}`（`half2.cl`）を `clbuild`（`clBuildProgram`）で GT 730 / GT 710 のどちらでも落とせる。`fma` のカーネル（`fl.cl`）と `vload_half`（`half3.cl`）は通る（rc 0）。→ **float→half の変換（NIR の `f2f16`）を nvc0 のコード生成が持たない**（Mesa 25.0.7、GK208B）。
- 一次情報: `unknown nir_op f2f16` と nvc0 / rusticl の組合せで WebSearch を 1 回かけたが、該当する issue は見つからなかった（Mesa の GitLab を直接検索していないので、既存 issue の有無は未確認）。
- 速度の比較は成立しない（動かない）。

**NVK + ggml-vulkan: 一部のみ動く**（`nvk_per_op.txt`）
- `test-backend-ops` の MUL_MAT（type_a=q4_0, type_b=f32）は **94/94 passed**（fp16: 0 の f32 経路）。SCALE / CONT / RMS_NORM / SOFT_MAX / SSM_CONV / ARGSORT / SUM_ROWS も通る。
- **f16 を含むパイプラインは作成に失敗する**: ADD（`add_f16_f16_f16_norepeat`）、MUL、CPY（`contig_cpy_f32_f16`）、ROPE（`rope_norm_f16`）、CLAMP（`clamp_f16`）、GET_ROWS（`get_rows_iq2_xxs_f32`）、MUL_MAT_ID（`mul_mat_vec_id_q4_k_f32`）はいずれも `vk::Device::createComputePipeline: ErrorUnknown` で abort。したがって `llama-bench`（LFM2.5-1.2B）は起動直後に同じ例外で落ちる。原因が OpenCL 側と同じ f2f16 かは未確認（f16 を含む点は共通。推定）。
- 性能（動く op のみ）: `test-backend-ops perf` の MUL_MAT q4_0（m=4096, n=1, k=14336、重み約 33 MB）が GT 710 で 47.1 ms（約 0.70 GB/s 相当）、GT 730 で 25.2 ms（約 1.3 GB/s）。390 の CL11 mv は GT 730 で約 28 GB/s なので、**再クロック前の NVK は 2 桁近く遅い**（pstate は 405 MHz / 810 MHz のまま）。

### 上流への報告候補（投稿していない）

1. Mesa nouveau（nvc0 codegen、rusticl）: float→half (`nir_op_f2f16`、`vstore_half`) で `unknown nir_op f2f16` → シェーダ変換失敗 → プロセス abort（GK208B / GF1xx 共通の可能性）。最小再現は `repro/half2.cl` + `repro/clbuild.c`。落ちずに `CL_BUILD_PROGRAM_FAILURE` を返すだけでも改善。報告先は Mesa GitLab（rusticl / nouveau）。既存 issue の有無は未確認。
2. Mesa NVK（Kepler、25.0.7）: f16 を含む compute パイプラインが `ErrorUnknown`。ggml-vulkan の `test-backend-ops` が最小再現（op 名は `nvk_per_op.txt`）。25.2.1 以降で再現するかは未確認（25.0.7 では Kepler は未適合として扱われる）。
3. 切替スクリプト／環境側: 前回までの `nvidia-installer` によるモジュール解除、GT 430 の nouveau 束縛と 390 への戻り（`RmInitAdapter failed (0x12:0x45:1920)`）は README の別節に記録済み。

### GT 710 の pstate（再クロック）
次項に追記（実行前にここまでコミットする）。
