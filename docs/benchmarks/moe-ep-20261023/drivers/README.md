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
