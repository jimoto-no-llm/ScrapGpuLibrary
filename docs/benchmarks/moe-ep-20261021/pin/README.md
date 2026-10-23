# クロック固定（`perf pin` / `perf compute`）での C1 と SGLX C

TB250、2026-10-21〜22。LFM2.5-8B-A1B Q4_0、llama.cpp `23b0202`、SGL `feat/moe-expert-parallel`（SGLX0）。`queue_pin.sh` が `sudo sgl-driver-switch perf pin|compute` を設定し、C1 と SGLX C を交互に 2 回ずつ（`REPS=10`、`PROFILE=0`、`-ub 512 -ts 5/1`）、1 秒間隔のクロック記録つきで測り、最後に `perf auto` に戻した（`logs/pin.perf.log`）。コマンドの詳細は `../sglx-lmhead/README.md` の再現手順（`bench.sh`）。

- `pin`: `power_dpm_force_performance_level=manual` と `pp_dpm_sclk` / `pp_dpm_mclk` の最上段。RX 6400 2320 / 1000 MHz、WX 2100 1219 / 1500 MHz。記録（`logs/pin_*_clk.csv`）で RX の `average_gfxclk` は負荷中 2322 MHz。
- `compute`: `pin` + 電力プロファイル COMPUTE。
- SGLX C: `GGML_SGLX_SPLIT=Vulkan0:0.5,Vulkan1:0.2,CL11_0:0.2,CL11_1:0.1`（出力射影を RX 6400 / WX 2100 / GT 730 / GT 710 に語彙方向で分割、`-ot "token_embd\.weight=SGLX0"`）。

| 設定 | 構成 | pp512 | tg128 | pg512,128 |
| --- | --- | --- | --- | --- |
| pin | C1 #1 | 665.2 ± 5.1 | 51.9 ± 0.1 | 195.3 ± 0.9 |
| pin | **SGLX C #1** | **673.2 ± 6.7** | **62.1 ± 0.5** | **223.3 ± 0.8** |
| pin | C1 #2 | 664.1 ± 4.4 | 52.0 ± 0.1 | 195.4 ± 0.7 |
| pin | **SGLX C #2** | **669.6 ± 6.4** | **61.9 ± 0.3** | **222.8 ± 1.3** |
| compute | C1 #1 | 662.7 ± 4.6 | 51.5 ± 0.1 | 194.8 ± 0.6 |
| compute | SGLX C #1 | 669.9 ± 6.1 | 61.7 ± 0.5 | 222.4 ± 0.8 |
| compute | C1 #2 | 664.4 ± 7.2 | 51.6 ± 0.1 | 194.5 ± 0.7 |
| compute | SGLX C #2 | 670.0 ± 7.0 | 61.7 ± 0.7 | 221.8 ± 1.6 |

## 判定（§8.1）

- 既存構成の最良（設定だけで得られる C1、クロック固定）に対して、SGLX C は tg128 **+19 %**（51.5〜52.0 → 61.7〜62.1）、pg512,128 **+14 %**（194.5〜195.4 → 221.8〜223.3）。4 組すべてで同じ向き。
- pp512 は +0.8〜1.3 %（662.7〜665.2 → 669.6〜673.2）。4 組すべてで SGLX C が上だが、1 run 内の標準偏差（4〜7）と同程度の差。**pp は「同等以上」で、明確な勝ちではない**。
- 正しさ: KLD（CPU 基準）0.02150、top-1 92.65 %、SGLX 1 枚 / N 枚の自己整合 0.000002（`../sglx-lmhead/`、perf auto で測定。クロックは計算内容を変えないので固定でも同じ、ただし固定状態での再測定はしていない）。
- `pin` と `compute` の差は誤差内。以後は `pin` を標準にする。
- `perf high` だけでは RX のクロックが固定されない状態がある（`../clock-states/`）。比較は必ず `pin` で揃える。
