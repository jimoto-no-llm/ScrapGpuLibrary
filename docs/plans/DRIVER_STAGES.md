# 段階 6・7 計画: OSS ドライバ（nouveau / rusticl / NVK）と NVIDIA 470

状態: **計画のみ（2026-10-23）。この文書の作成時点でドライバの切替も GPU 計測もしていない。** 「期待値」は推定で、実測の代わりではない。実測は各段階の `docs/benchmarks/moe-ep-20261023/drivers/` に残す。

## 1. 目的と、先に言っておくこと

目的は次の 3 つ。

- (a) Fermi / Kepler を nouveau で動かし、rusticl（OpenCL 3.0）または Clover で CL11 プラグインと SGLX の `pdev_cl11` が動くか。
- (b) NVK（Vulkan）で Kepler が ggml-vulkan に載るか。
- (c) 470 で Kepler の CUDA 11.4 / Vulkan / OpenCL がどうなるか。

そして、今の勝ち構成（LFM: 構成 H、30B: 構成 N = RX 6400 + GT 730）より速くできるか、だめなら Mesa 等に PR を出す余地があるかを調べる。

**速度向上の見込みは小さい（推定）。** 構成 H では出力射影のうち GT 730 / GT 710 の担当は 0.17 / 0.05、構成 N では GT 730 は `CL11_0:300`（Vulkan0 の `auto:400` に対する副次デバイス）で、全体の律速は RX 6400 側にある。10/22 の実測でも、SGLX C を CL11 から自前 CUDA に替えて差は標準偏差内だった（pp512 668→669、tg128 61.9→62.0、[cuda/README.md](../benchmarks/moe-ep-20261022/cuda/README.md)）。Kepler 側が 20 % 速くなっても全体にはわずかしか効かない。したがってこの段階の主な成果は、(1) どの組合せが動く / 動かないかの記録、(2) 上流へ報告できる再現手順、になる可能性が高い。

## 2. 現状の確認済み事実（tb250、読み取りのみで確認）

| 項目 | 確認結果 |
| --- | --- |
| OS / カーネル | Debian 13、6.12.107+deb13-amd64。backports なし（`apt-cache policy` で Mesa の候補は 25.0.7-2+deb13u1 のみ） |
| Mesa | `mesa-vulkan-drivers` / `mesa-opencl-icd` / `libgl1-mesa-dri` 25.0.7-2+deb13u1、`libclc-19`。`libvulkan_nouveau.so`（NVK）、`pipe_nouveau.so`（Clover と rusticl が使う Gallium ドライバ）、`nouveau_dri.so` あり。OpenCL ICD は `mesa.icd`（Clover）、`rusticl.icd`、`nvidia.icd`（390 のもの）の 3 つ |
| Vulkan ICD | `/usr/share/vulkan/icd.d/nouveau_icd.json`（`api_version` 1.4.305 は Mesa 共通の表記で、Kepler の対応版ではない）。`/etc/vulkan/icd.d/nvidia_icd.json`（390 は api 1.0.65、470 の展開物は 1.2.175）。390 の Vulkan ICD は落ちる（[HARDWARE.md](../HARDWARE.md)）ので、全スクリプトが `VK_ICD_FILENAMES` を明示する |
| ファームウェア | `firmware-misc-nonfree` 20250410-2 は入っているが、`/lib/firmware/nvidia/` は tegra 系のみ。`modinfo -F firmware nouveau`（431 件、すべて `nvidia/…`）に gf1xx / gk1xx / gk2xx の名前は 0 件。GF108 / GK208 の描画用マイクロコードはカーネルに内蔵されると考えられ（推定）、外部ファームウェアの不足は当面問題にならない（映像デコードは対象外） |
| nouveau モジュール | `modeset` パラメータ（0 = 無効、1 = 有効、2 = headless）、`config`、`debug`、`noaccel`、`runpm` あり。**`pstate` パラメータは無い**（Linux 4.5 当時の `nouveau.pstate=1` 手順は古い。[Phoronix](https://www.phoronix.com/news/Linux-4.5-Nouveu-PState-HowTo)） |
| ブラックリスト | `/etc/modprobe.d/blacklist-nouveau.conf`: `blacklist nouveau` と `options nouveau modeset=0`。`modprobe -n -v nouveau modeset=1` の出力は `nouveau.ko.xz modeset=0 modeset=1`。カーネルは後の値を採用するので、スクリプトの `modprobe nouveau modeset=1` は有効になる（改訂版は読み戻して検証する） |
| 470 | DKMS 済み（`updates/dkms/nvidia-tesla-470*.ko.xz`）、`/usr/local/lib/sgl-driver/NVIDIA-470.run`（470.256.02）。展開物 `~/bench/nvidia/n470` に libcuda / libnvidia-opencl / libGLX_nvidia / `nvidia_icd.json`（1.2.175）。10/05 に 470 への切替を一度試した記録（`~/bench/driver-swap.sh`）がある |
| 390 ユーザ空間 | 現在 libcuda.so.1 → 390.157（10/05 インストール）。`/dev/dri` は GT 430 = card2/renderD130、GT 730 = card3/renderD131、GT 710 = card4/renderD132、RX 6400 = card1/renderD129、WX 2100 = card5/renderD133（nvidia_drm 経由。nouveau にすると番号が変わりうる） |
| デバッグ FS | `/sys/kernel/debug` は一般ユーザで読めない（nouveau の `pstate` は root が必要）。sudo は `status` だけ許可されている |
| 他の担当 | 確認時に llama / bench 系のプロセスが 5 つ動いていた（30B 計測）。切替はそれが終わってから |

## 3. 一次情報と、そこから読める期待

| 項目 | 一次情報 | 読み取り |
| --- | --- | --- |
| NVK と Kepler | [Mesa docs: NVK](https://docs.mesa3d.org/drivers/nvk.html): 「Kepler（GeForce 600 / 700）以降 Ada まで対応」「Linux 6.6 以上が必要」。`NVK_I_WANT_A_BROKEN_VULKAN_DRIVER=1` は「テストが乏しい / 壊れている GPU も列挙する」開発者向け。[Collabora の 25.2 記事](https://collabora.com/news-and-blog/news-and-events/mesa-25.2-brings-new-hardware-support-for-nouveau-users.html): **Mesa 25.2 から Kepler を NVK が広告**し、Kepler は Vulkan 1.2 止まり（ハード上の制限）。[25.2.1 のリリースノート](https://docs.mesa3d.org/relnotes/25.2.1.html): 25.2.0 で外れていた Kepler の適合判定を復活、`memory_model.message_passing fails on KeplerA` が未解決の不具合として載る | tb250 の **25.0.7 は Kepler を既定では出さない**（推定。25.0.7 の changelog には「Kepler の VAB サイズ修正」があり、コードは入っているので環境変数で列挙される可能性がある。実機で確認）。確実に試すなら 25.2.1 以降を自前ビルドするか、unstable から取る。**Fermi（GT 430）は NVK の対象外**（docs は Kepler 以降のみ） |
| rusticl と nouveau | [Mesa docs: envvars](https://docs.mesa3d.org/envvars.html): `RUSTICL_ENABLE` は `iris, llvmpipe, nouveau, panfrost, radeonsi` が対象（`r600` は実験的）、`name:1,3` で装置番号の指定、`RUSTICL_FEATURES=fp64` 等。[rusticl の文書](https://docs.mesa3d.org/rusticl.html): 既定では無効。25.0.7 のリリースノートに rusticl の修正は無く、Mesa 25.2 では Clover が置き換えられる記事あり（[linuxadictos](https://en.linuxadictos.com/Mesa-25-2-adds-new-drivers--implements-Rusticl-and-replaces-Clover-for-OpenCL.html)） | nouveau は rusticl の対象だが、**GF108 / GK208 で実際に動くという一次情報は見つからなかった**。動くかは実機で決まる（`clinfo` が最初の関門）。25.0.7 には **Clover もあり**、`pipe_nouveau.so` を共有する。CL11 は OpenCL 1.1 向けなので、rusticl が無理でも Clover で動く可能性がある（推定。25.2 以降は Clover が消える） |
| nouveau の再クロック | [Phoronix 4.5 の手順](https://www.phoronix.com/news/Linux-4.5-Nouveu-PState-HowTo): debugfs の `pstate`。[freedesktop bug 99505](https://bugs.freedesktop.org/show_bug.cgi?id=99505): **GK208（GT 740M）で pstate への書き込みがメモリ再クロック（`gk104_ram_prog`）でハング**。Fermi は 2018 年時点で未対応との返信あり（[lkml](https://lkml.rescloud.iu.edu/1801.1/07587.html)。古い） | GT 730 / GT 710 は既定ではブート時の低い pstate のまま（推定）で、390 の 954 MHz に対して数分の一の可能性が高い。公平な比較には再クロックが必要だが、**ハング（PCIe 全体の停止を含む）の危険**がある。GT 430 は再クロックできない見込み |
| 470 とサポート | [NVIDIA フォーラム（CUDA 11.5 リリースノートの引用）](https://forums.developer.nvidia.com/t/geforce-600-kepler-users-caution-with-driver-470-nvidia-has-done-a-wrong-act-not-is-about-drop-support-in-driver-480/189933): Kepler のドライバサポートは R495 で削除、CUDA Toolkit の開発サポートは 11.x まで。[R470 は Kepler の最後の枝](https://forums.developer.nvidia.com/t/tesla-k80-and-discontinued-driver-support-for-kepler/193903)。[cuFFT の資料](https://docs.nvidia.com/cuda/archive/12.6.2/cufft/deprecated-functionality.html): sm_35 / sm_37 は 11.x で非推奨、12.0 で削除 | 470 は GK208（sm_35）を CUDA 11.x の範囲で扱う。GT 640 + 470 + CUDA 11.4 が落ちるという利用者の報告もある（同フォーラム。公式の確認ではない）ので、`nvidia-smi` と自前 CUDA バックエンドで確かめる |
| ggml-cuda と Kepler | [REUSE.md](../REUSE.md): 上流の CMake 既定は sm_50 以上。issue #12140 は K80 を CUDA 11.4 + ドライバ 470 + `CMAKE_CUDA_ARCHITECTURES=…;37` で動かした報告、#18743 は K40c が実行時にクラッシュ。`~/bench/kepler-compat.patch`（tb250）は `common.cuh` に `__shfl_sync` 等を旧 `__shfl` に置き換えるマクロを足すだけの 21 行 | ggml-cuda を Kepler で動かすには **nvcc 11.4（約 3 GB の取得）と GCC 11 以下**が要る（Debian 13 の既定は GCC 14、推定）。自前 CUDA バックエンドは libcuda を dlopen して PTX を渡すので nvcc 不要で、470 でそのまま動くはず（390 は CUDA 9.1 世代の Driver API で動作済み）。ggml-cuda の道は後回し |

### 状態ごとの期待（すべて推定）

| 状態 | 動いてほしいもの | 見込みと理由 |
| --- | --- | --- |
| `nouveau`（3 枚すべて） | Clover / rusticl の OpenCL、GT 730 / GT 710 の NVK | OpenCL は `clinfo` に出る確率は中程度。Kepler の NVK は 25.0.7 では環境変数が要るうえ未適合で、動いても正しさに問題が出る見込み。性能は再クロックなしでは 390 に大きく負ける。同一プロセスで GT 430 + Kepler を開けない制約（390 の問題）は、プラットフォーム 1 つの rusticl では起きないはず |
| `nv470`（GT 430 なし） | 自前 CUDA、470 の OpenCL、Vulkan | CUDA / OpenCL は 390 と同等（GK208 で同じハード、PTX も同じ）の見込み。Vulkan は 470 の ICD が api 1.2.175 を名乗るので 390 のように落ちない可能性があるが、ggml-vulkan が Kepler で動くかは未確認 |
| `nv470-nouveau430` | 上記に加えて GT 430 を nouveau で同時に使う | **390 の制約（Fermi と Kepler を 1 プロセスで開けない）が、ドライバが別なら無くなる**可能性。これが唯一、構成を変えうる点。GT 430 は再クロックできず、遅い見込み（390 の CL11 は pp128 89.4 / tg64 22.3 t/s）。nouveau と nvidia 470 を同時にロードしたときの相互作用は未確認 |

## 4. 合否の基準（共通）

- 同じセッション内で 390 を再測定する。古い値と混ぜない（モデル・ビルド・温度が変わる）。測定は必ず `~/moe-ep/sglx/run.sh` 経由（flock と温度待ち）。
- 動作: `clinfo -l` / `vulkaninfo --summary` / `nvidia-smi` に当該カードが出る = 認識。`scrapgpu_cuda_kernel_tests` 161/161、`scrapgpu_ops_tests` が通る = 正しさ。
- 性能: 390 の同じ測定（同じスクリプト）に対して、帯域 / pp / tg が 90 % 以上 = 同等、105 % 以上 = 改善。100 % 未満は、再クロックの有無と pstate を併記して記録する。
- 構成の勝ち: 構成 H（LFM、perf pin で tg 69.0 / pg 240.4）、構成 N（30B）を 2 回交互に測って、差が標準偏差を超えるときだけ「勝ち」と書く。
- 失敗の扱い: 最初に失敗した段（認識 / 実行 / 正しさ）とログを残して次へ進む。原因調査は 1 項目 30 分で打ち切る。

## 5. 切替の共通手順（利用者が行う）

切替は `sudo /usr/local/sbin/sgl-driver-switch <状態>`。この作業の権限では `status` だけ可能で、**切替は利用者が実行する**。

### 5.1 切替前

1. 他の担当の計測が終わっていることを確認する（`pgrep -a 'llama|bench|sglx|hec'`）。切替中の割り込みを避けるため、`flock ~/bench/gpu-bench.lock` を保持した状態で切替コマンドを実行する: `flock ~/bench/gpu-bench.lock sudo /usr/local/sbin/sgl-driver-switch nouveau`。
2. 止めるもの: GT 430 / GT 730 / GT 710 を開いているすべてのプロセス（llama-bench、`scrapgpu-device-server`、mesh-llm、clinfo など）。RX 6400 / WX 2100 だけを使うプロセスは切替に影響しない（AMD は触らない）。
3. `busy_check` の挙動: 現行は `/dev/nvidia*` を開いているプロセスだけを見る。改訂版（§8）は GT カードの DRM ノード（`/dev/dri/cardN`、`renderDN`）も見る。見つかると PID を表示して終了コード 1 で終わり、何も変更しない。nouveau からの切替で、現行版は DRM ノードを見ないため `modprobe -r nouveau` が使用中で失敗し、中途半端な状態になりうる。改訂版を先に install すること。
4. `/run/sgl-driver.state`: 切替が最後まで成功したときだけ状態名を書く（再起動で消え、`status` は `unknown (boot default)` = 390 と読む）。改訂版は途中で失敗すると `failed(<rc>) at line …` を書く。
5. `perf` は AMD 用。nouveau 計測の前後で `sudo sgl-driver-switch perf pin` / `auto` を使う（現行どおり、各スクリプトが後始末する）。

### 5.2 切替後（毎回）

`probe_state.sh <ラベル>`（読み取りのみ、所要 2〜3 分）で認識状況を `drivers/logs/<ラベル>/` に残す。`SUMMARY.txt` に clinfo / vulkaninfo の要点が出る。

## 6. 段階

優先順位は **C（nv470-nouveau430）→ A（nouveau）→ B（nv470）** を推奨する。C は B の内容（470 の Kepler）を含み、GT 430 の nouveau も確認でき、切替が 1 回で済む。B は C で nouveau と nvidia の同居に問題が出たときだけ行う。

### 段階 C: `nv470-nouveau430`（GT 430 = nouveau、GT 730 / GT 710 = 470）

| 項目 | 内容 |
| --- | --- |
| 切替 | `flock ~/bench/gpu-bench.lock sudo /usr/local/sbin/sgl-driver-switch nv470-nouveau430`。470 のユーザ空間インストールで約 1〜2 分。`status` の最後に WARNING が出なければユーザ空間とモジュールが一致 |
| 確認 | `probe_state.sh c_nv470_nouveau430`。見るもの: GT 430 が nouveau、GT 730 / GT 710 が nvidia（`status`）、`nvidia-smi` に Kepler 2 枚、`cl-rusticl` に GT 430（`NVC1`）、`vk-nvidia`（470 の Vulkan が Kepler を出すか。api 1.2.x が出れば 390 のクラッシュは無い）、`libs` で libcuda.so.1 → 470.256.02 |
| 試す項目 | (1) `bench_cuda470.sh`: 自前 CUDA の kernel tests / ops tests / mm_bench（q6_K・q4_0・MUL_MAT_ID 32 token）/ SGLX C（CUDA）。(2) `bench_cl11_470.sh`: 470 の OpenCL で構成 H（LFM）。(3) `bench_cl11_cards.sh rusticl gt430` と `Clover gt430`: GT 430 を nouveau で。(4) 同一プロセスで GT 430 + Kepler: `GGML_CL11_PLATFORMS=NVIDIA,rusticl` + `RUSTICL_ENABLE=nouveau:<GT430 の番号>` で `llama-bench -dev CL11_0/CL11_1/CL11_2`（390 では `CL_INVALID_DEVICE`）。(5) `bench_vk_nvk.sh` は Fermi が対象外なので GT 430 には使わない |
| 期待（推定） | (1)(2) は 390 と同等（±5 %）。(3) は動けば pp は 390 の 5 割以上、tg は再クロックなしで低い。(4) は動く可能性があり、動けば 5 枚（RX 6400 + WX 2100 + 3 枚）を 1 プロセスで使える構成が初めて可能になる（390 では GT 430 は別プロセスの RPC だった） |
| 合格 | (1) 161/161、mm_bench の GB/s が 390 の 90 % 以上、(2) H が 390 の ±3 %、(3) `clinfo` に出て kernel tests 相当（`scrapgpu_ops_tests` を `SCRAPGPU_TEST_BACKEND=opencl11` で）が通る、(4) 3 枚が 1 プロセスで開く |
| 所要時間 | 切替 3 分 + probe 3 分 + bench 約 40 分（`run.sh` の温度待ちを除く） |
| 失敗時の典型 | `modprobe nvidia-tesla-470` が nouveau の残りで失敗 / `status` の WARNING（ユーザ空間不一致）/ GT 430 の nouveau が probe 失敗（dmesg を見る） |

### 段階 A: `nouveau`（3 枚とも nouveau）

| 項目 | 内容 |
| --- | --- |
| 切替 | `flock ~/bench/gpu-bench.lock sudo /usr/local/sbin/sgl-driver-switch nouveau`。`modeset` は 1 が既定。fbcon を掴んで unload が詰まる場合は、スクリプト冒頭の `NOUVEAU_MODESET` を 2（headless）にして再試行する（カードに出力が無いので描画ノードのみで足りる見込み。推定） |
| 確認 | `probe_state.sh a_nouveau`: `lsmod`・`nouveau-parm`・`dri-nodes`（カード番号の対応）、`cl-clover` / `cl-rusticl`（`RUSTICL_ENABLE=nouveau`）に 3 枚とデバイス名（`NVC1` / `NV108` 等）、`vk-nvk-plain` と `vk-nvk-broken`（Kepler が出るのは後者だけのはず）、`kernel-log` に `nouveau` の致命的なエラーが無いか |
| 試す項目 | (1) `bench_cl11_cards.sh rusticl gt730 gt710 gt430` と `bench_cl11_cards.sh Clover …`（CL11 の mm_bench と llama-bench。`GGML_CL11_PLATFORMS` はプラグインの名前指定）。(2) `bench_vk_nvk.sh gt730 gt710`（test-backend-ops → llama-bench）。(3) 再クロック（任意、§7）。(4) 動いた経路で SGLX 構成 H の Kepler 担当を入れ替えた測定（`bench_cl11_470.sh` を複製し、`GGML_CL11_PLATFORMS` と `RUSTICL_ENABLE` を差し替えて流用） |
| 期待（推定） | (1) rusticl か Clover のどちらかで `clinfo` に出る確率は中程度。出ても 390 の NVIDIA 製コンパイラ向けに調整した CL11 カーネル（`prmt`、Fermi の denormal 方式など）より遅い見込み。クロックは再クロック前で、帯域は 390 の 1 割〜5 割程度か（pstate 次第）。(2) Mesa 25.0.7 では Kepler の NVK は未適合で、列挙されても `test-backend-ops` の失敗や GPU ハングがありうる。ggml-vulkan が要求する拡張（16 bit ストレージ等）を NVK の Kepler が出すかは未確認 |
| 合格 | (1) `clinfo` 認識 → CL11 の kernel 正しさ → mm_bench が 390 の 90 % 以上。(2) vulkaninfo に出る → test-backend-ops MUL_MAT 通過 → llama-bench が走る |
| 所要時間 | 切替 2 分 + probe 3 分 + bench 約 60 分。再クロックは別に 30 分 |
| 上流への報告の材料 | `clinfo` が出ない / カーネルのビルドが落ちる場合は `RUSTICL_DEBUG=program,perf` のログと最小の `.cl`。NVK は `NVK_DEBUG` のログ。報告先は Mesa の GitLab（下の §9） |

### 段階 B: `nv470`（GT 430 はドライバなし）

| 項目 | 内容 |
| --- | --- |
| 切替 | `flock … sudo sgl-driver-switch nv470`。`driver_override` は GT 430 に `none` を書き、どのドライバも掴まない |
| 試す項目 | 段階 C の (1)(2) と同じ。GT 430 の項目だけ無い |
| 期待 | C と同じ |
| 所要時間 | 切替 3 分 + probe 3 分 + bench 約 40 分 |

### 段階 D（任意）: ggml-cuda を Kepler で

段階 B / C で自前 CUDA が 390 を上回ったときだけ検討する。nvcc 11.4 の取得（約 3 GB、**利用者の承認が要るダウンロード**）と GCC 11 以下の用意が前提で、`kepler-compat.patch` を当てて `-DCMAKE_CUDA_ARCHITECTURES=35`。見込みは SGLX の勝ち構成に効かない（上流の ggml-cuda は 1 デバイスのバックエンドで、SGLX の副次デバイスにならない）ので、優先度は低い。

## 7. nouveau の再クロック（任意。ハングの危険あり）

- 読む: `sudo /usr/local/sbin/sgl-driver-switch pstate show`（改訂版。debugfs の `pstate` を nouveau のカードごとに表示）。事前に `probe_state.sh` の `pstate` に残る。表示の読み方は `[nn]: core X-Y MHz memory Z MHz` と現在 / ブート時の印（推定。実機で確認）。
- 変える: `sudo … pstate set <id> <PCI アドレス>`。**1 枚ずつ、そのカードを使うプロセスが無い状態で。** GK208 のメモリ再クロックでハングした報告があるので、まず GT 710（失っても影響が小さい）で試し、ハングしたら §11 の復旧へ。
- 再クロック後は `perf` の測定と同様に、前後のクロックをログに残す。

## 8. 切替スクリプトの修正（`tools/tb250/sgl-driver-switch`、この変更に含めた）

| 問題 | 内容 | 修正 |
| --- | --- | --- |
| `busy_check` が DRM ノードを見ない | nouveau / nvidia_drm のノードを開いたままだと `modprobe -r` が失敗し、途中で止まる | GT 3 枚の `/dev/dri/cardN` と `renderDN`（sysfs の `drm/` から引く）も確認。AMD のノードは見ない |
| 失敗時の状態が嘘になる | `set -e` で止まると `/run/sgl-driver.state` が前の状態のまま | `set -E` と ERR trap で `failed(rc) at line …: <コマンド>` を書く |
| nouveau の `modeset` | ブラックリストの `options nouveau modeset=0` が負けるか未検証だった | `modprobe -n -v` の実測で後の値が勝つことを確認。改訂版は `/sys/module/nouveau/parameters/modeset` を読み戻し、違えば失敗にする。`NOUVEAU_MODESET`（既定 1、2 は headless） |
| ユーザ空間を毎回入れ直す | 同じ `.run` でも毎回 1 分以上かかる | `/usr/local/lib/sgl-driver/userspace.installed` に入れた `.run` の名前を残し、同じなら省略。`--reinstall-userspace` で強制。入れた後に `ldconfig` と `libcuda.so.1` の指す先を表示 |
| `status` がユーザ空間の食い違いを出さない | 470 を入れたまま再起動すると、390 のモジュールに 470 の libcuda という組になる（`CUDA_ERROR_*` / `NVML` の不一致） | `status` に `userspace libcuda=… marker=… kernel module=…` と不一致時の WARNING |
| `status` にカードとノードの対応が無い | nouveau にするとカード番号が変わる | `status` に各 GT の `drm/` ノードと `nouveau modeset=` を表示 |
| nouveau 状態の使い方が分からない | nvidia の ICD が `/etc/vulkan/icd.d` に残る | 状態が nouveau 系のとき、`VK_ICD_FILENAMES`、`NVK_I_WANT_A_BROKEN_VULKAN_DRIVER`、`RUSTICL_ENABLE` の注意を表示 |
| 再クロック手段が無い | debugfs は root のみ | `pstate show` / `pstate set <id> <pci>` を追加（nouveau が掴んだカードだけ） |

**390 と 470 のユーザ空間の入れ替えについて（未検証の点）:** `.run` の `--install-libglvnd` は Debian の libglvnd（`libGLX.so.0`、`libEGL.so.1` など）を NVIDIA 版で上書きする（tb250 には既に `libEGL.so.1.1.0` 等が入っている）。`libcuda.so.1`、`libnvidia-opencl.so.1`、`libGLX_nvidia.so.0` は版つきの実体へのシンボリックリンクで、`.run` が張り替えるので 390↔470 で切り替わる。**旧版の実体（`libcuda.so.390.157` など）が残るか、インストーラが前の版を自動で外すかは未確認。** 動作の判定は `ls -l /usr/lib/x86_64-linux-gnu/libcuda.so*` と `/etc/OpenCL/vendors/nvidia.icd`（root 所有・読み取り専用で生成される。ファイル名は同じで中身の `libnvidia-opencl.so.1` は版つきリンク経由）で行う。Mesa の ICD（`mesa.icd`、`rusticl.icd`）は触られない。

### 利用者が行う install コマンド（tb250 上。root 所有を保つ）

```
scp tools/tb250/sgl-driver-switch tb250:/tmp/        # もしくは worktree から取り出す
sudo install -o root -g root -m 0755 /tmp/sgl-driver-switch /usr/local/sbin/sgl-driver-switch
sudo /usr/local/sbin/sgl-driver-switch status          # 動作確認（WARNING が出ないこと）
```

sudoers の追加は不要（既存の `NOPASSWD: /usr/local/sbin/sgl-driver-switch` がサブコマンドを問わず効く）。これは `pstate set` と各切替が、この作業の権限外であることを意味する。**再起動は不要。**

## 9. 上流への余地（推定を含む）

| 候補 | 内容 | 状態 |
| --- | --- | --- |
| Mesa（NVK） | Kepler（GK208）で `test-backend-ops` / llama.cpp が通らない箇所の最小再現を出す。25.2.1 のノートに載る `memory_model.message_passing fails on KeplerA` と重なれば既存 issue に情報を足す | 25.2.1 以降のビルドが前提（25.0.7 での報告は古いと言われる） |
| Mesa（rusticl / nouveau） | GF108 / GK208 で `clinfo` が出ない、または CL11 カーネルが落ちる場合の報告（`RUSTICL_DEBUG`、最小 `.cl`）。CL11 は WX 2100 / RX 6400 で rusticl の `fma` 120 倍遅い問題を既に見つけており、同じ流儀で最小再現にする | 動かなければ報告できる。動けば性能差の原因（`fma` の展開など）を調べる |
| Linux / nouveau | GK208 の pstate 書き込みによるハング（bug 99505 と同一か） | 再クロックを試したときだけ |
| llama.cpp | `kepler-compat.patch`（shuffle マクロ）。上流は sm_50 以上が既定で、Kepler 用の変更が受け入れられる可能性は低い（推定） | 段階 D まで進んだ場合のみ |

## 10. 計測スクリプト（`docs/benchmarks/moe-ep-20261023/drivers/scripts/`）

**どれも実行していない**（構文は `bash -n` と `shellcheck` のみ）。`~/moe-ep/drivers/` に置いて使う。

| ファイル | 役割 |
| --- | --- |
| `env_drv.sh` | 共通の環境（ICD、モデル、バイナリの場所は環境変数で上書き可）、`need_state`（状態が違えば止まる） |
| `probe_state.sh LABEL` | 切替直後の棚卸し（clinfo を 3 つの ICD で個別に、vulkaninfo を NVK / nvidia / radeon の ICD で個別に、nvidia-smi、`status`、dmesg）。モデルは読まない |
| `bench_cl11_cards.sh PLATFORM CARD…` | PLATFORM = NVIDIA / rusticl / Clover。単体カードの mm_bench（q6_K、q4_0、MUL_MAT_ID 32 token）と llama-bench（LFM2.5-1.2B Q4_0、pp128 / tg64） |
| `bench_vk_nvk.sh CARD…` | NVK 上の ggml-vulkan: `test-backend-ops` → `llama-bench` |
| `bench_cuda470.sh` | 自前 CUDA: kernel tests / ops tests / mm_bench / SGLX C（CUDA） |
| `bench_cl11_470.sh` | 470 の OpenCL で構成 H（LFM）を 2 回交互に（C1 と比較）。30B の構成 N は既存の `hec/scripts/q_final.sh` を同じ状態で流す |

デバイス番号（nouveau 下の GT 430 / GT 730 / GT 710 の並び）は推定の既定値（0 / 1 / 2）で、`probe_state.sh` の出力に合わせて `IDX_GT430` 等で上書きする。

## 11. 390 への戻し方と、戻らなかったとき

### 通常

1. 該当カードを開いているプロセスをすべて止める。
2. `flock ~/bench/gpu-bench.lock sudo /usr/local/sbin/sgl-driver-switch nv390`（約 1〜2 分。ユーザ空間が 390 でなければ入れ直す）。
3. 確認: `sudo … status`（3 枚とも `nvidia`、`NVRM version … 390.157`、WARNING なし）、`nvidia-smi`、`clinfo -l`（NVIDIA プラットフォームに 3 枚）、`bench_cl11_cards.sh NVIDIA gt730` が 10/22〜23 の値と ±5 %。
4. `perf` は `auto` に戻す（各スクリプトの trap が行う。`status` の `perf=` で確認）。

### nv390 が失敗する（`failed(rc)…` が state に出る、または `modprobe -r` が使用中）

1. `sudo … status` で現在のドライバと `nodes:` を見る。
2. 掴んでいるプロセスを探す: `find /proc/[0-9]*/fd -lname '/dev/dri/*' -o -lname '/dev/nvidia*' 2>/dev/null | cut -d/ -f3 | sort -u`（`fuser` は未導入）。利用者のプロセスなら止めて `nv390` を再実行。
3. `rmmod` / `modprobe -r` が D 状態で戻らない、`dmesg` に nouveau の `timeout` / `fifo` / `NVRM: Xid` が出る、`nvidia-smi` が固まる、のいずれかなら **再起動が必要**（**再起動は利用者の承認が要る**）。
4. ユーザ空間 `.run` の途中失敗: `sudo … nv390 --reinstall-userspace` を再実行。`/usr/lib/x86_64-linux-gnu/libcuda.so*` が 390.157 を指すことを確認。展開済みの `~/bench/nvidia/n390` と `n470` はファイル一覧の比較に使える。

### 再起動後

ブート既定は 390（ブラックリストと DKMS の 390 が自動で読まれる。`status` は `state: unknown (boot default)`）。ただし**ユーザ空間が 470 のまま**のことがある（`userspace.installed` が `NVIDIA-470.run` を指す）ので、再起動後に必ず `sudo … nv390` を 1 回実行して 390 のユーザ空間に戻す（`status` の WARNING がなくなる）。

### 再起動しても戻らない（ssh できない）

nouveau の再クロックなどで PCIe が止まった場合は、電源の入れ直しが必要になる（物理操作。シリアルコンソール `ttyS0,115200` がカーネル引数にある）。この計画の他の項目では起きない想定。再クロックは最後に、1 枚ずつ。
