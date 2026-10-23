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
