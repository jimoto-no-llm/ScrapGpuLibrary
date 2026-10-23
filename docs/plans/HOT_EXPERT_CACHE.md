# ホットエキスパートキャッシュ（Qwen3-30B-A3B、tb250）設計書

状態: **実装済み（2026-10-23、実測は `docs/benchmarks/moe-ep-20261023/hec/README.md`）**。以下の本文は実装前の設計のまま残してあり、数値は【推定】。実測との差は上記 README の「実測で分かったこと」を優先する。主な差: (1) コールドの CPU 計算は ggml のグラフではなく自前の mat-vec（2 コア CPU で ggml の OpenMP が 10 ms/呼び出し）、(2) prefill は ring に流してペア単位で計算せず、RX の 1 グラフで重みと和まで計算する、(3) 最速の構成は RX + GT 730（WX 2100 を足すと decode が大きく遅くなった）、(4) CUDA の pdev は作らず CL11 を使った。

ブランチ `feat/moe-expert-parallel`。作成 2026-10-23。設計時は read-only の調査だけで書いた。
表記: **【実測】** はリポジトリ内の既存の測定値（出典付き）、**【推定】** はこの文書の見積もり、**【不明】** は未測定・未確認で、段階 0 で測るもの。

前提文書: `docs/plans/MOE_COMPOSITE_DEVICE.md`（以下「親設計」。SGLX0、pdev、方式 E/F、§8.1 合格条件）、`docs/benchmarks/moe-ep-20261022/qwen3-30b/README.md`（以下「30B 基準」）。
Strata の docs（`HOW_IT_WORKS.md`、`DETAILS.md`）は、指定された scratchpad にクローンが残っておらず読めなかった。この文書の「Strata 流」は 30B 基準と親設計にある範囲の発想（頻度で expert を GPU に置き、残りは RAM/mmap から CPU で計算する、`--low-ram mmap`、`--expert-cache`）だけを指す。Strata の実装の細部との対応は**【不明】**。

---

## 0. 結論（先に）

1. **llama.cpp は改造しない。** `-ot 'blk\.[0-9]+\.ffn_(up|gate|down)_exps\.weight=SGLX0'` で全 48 層の expert テンソルを SGLX0 に置く。SGLX0 は `buffer_from_host_ptr` で mmap をそのままホストの影にする（コピーしないので RAM は増えない）。SGLX0 の中では、expert を **層によらない固定サイズの「スロット」**（1 expert = gate + up + down = 2.65 MB）として扱い、各 GPU の VRAM にスロットの配列を持つ。どの (層, expert) をどのスロットに置くかは表で持つ。ホットは GPU のスロットで、コールドは mmap の上で ggml-cpu が計算する。両者を**同時に**計算して、ホストがスロットの順に合計する（3 章）。
2. **tg が伸びる主因は「CPU と GPU が同時に動くこと」。偏りはその次。** 今の最速構成（tg 7.70）は層分割なので、34 層ぶんの CPU の expert 計算（1 層約 2.8 ms【推定】）と GPU の層が**直列**に足されている。expert 単位で置き分けて、各層で GPU（ホット）と CPU（コールド）を並列に動かすと、ヒット率 h（選ばれた expert が GPU にある割合）が 0.19 を超えれば、基準の 7.70 を超える見込み【推定】（6 章）。
3. **tg の見積もり【推定】**（imatrix の頻度分布から求めた。ホット集合を正しく当てた場合の上限寄りの値。6.3 節）:

   | ホットの置き場 | 同じ領域（h 0.76〜0.92） | 別の領域で固定のまま（h 0.26〜0.57） | 一様（llama-bench の乱数トークンで偏りが無い場合の最悪） |
   | --- | --- | --- | --- |
   | RX + WX | 18.9〜22.6 | 8.3〜9.7 | 約 9 |
   | + GT 730（ホット） | 21.8〜24.5 | 8.7〜10.4 | 約 9.5 |
   | + GT 710（ミディアム） | 25.3〜26.2 | 10.0〜12.7 | 約 10 |

   モデルの誤差は ±20 % 程度と見込む（`-ncmoe` の基準で校正し、n39 は合うが、n41・n44 では実測より 20 % 楽観的。5.2 節）。期待値の max で計算しているので、実際は 2〜3 割低い可能性がある。それでも **tg は 12〜20 前後**が現実的な目標【推定】。
4. **pp は、llama の op offload を使えなくなる代わりに、SGLX0 が自前で「GPU に無い expert だけ」を RX に流す。** op offload は CPU バッファにある重みにしか働かない（`ggml_backend_sched_backend_id_from_cur` の条件 `src_backend_id == n_backends - 1`）。そのため SGLX0 に置いた expert は op offload の対象外になる。SGLX0 は、ubatch で使われる expert のうち GPU に無いものだけを、RX のリング（数十 MB）に流して計算する。1 層ぶん（340 MB）の置き場が要らないので、30B 基準の「VRAM が満杯だと pp が 1/3 になる」二峰性が消える。**pp512 は 75〜105 の見込み**【推定】（基準 59.8）。pg512,128 は 40〜50【推定】（基準 22.87）。7 章。
5. **ホット／ミディアム／コールドの 3 層案**（ユーザー案、5 章）:
   - **ホット = RX 6400・WX 2100・GT 730**（同じプロセス。往復 0.06〜0.3 ms）。
   - **ミディアム（同じプロセス）= GT 710**。帯域は CPU の約 1.3 倍しかないが、同期が 60 µs と安いので、CPU と並列に動く「第 2 の CPU」として効く。tg を 21.8 → 25.3（同じ領域）に上げる見込み【推定】。
   - **ミディアム（別プロセス）= GT 430**。1 GB に約 335 expert（**層あたり 7 個**）しか入らない。コールドに回る選択を **2〜7 ポイント**（同じ領域で 2〜6、別の領域で 4〜7）減らすだけ。CPU の時間に換算すると、得は最大でもトークンあたり 3〜9 ms【推定】。
   - **損益分岐は「GT 430 の 1 job の遅延 L が、その層の CPU の残り時間より短いこと」**。ホットのヒット率が高いとき（同じ領域）は L ≲ 0.3〜0.55 ms、低いとき（別の領域）は L ≲ 1.0〜1.5 ms【推定】。いまの 2.4〜3.3 ms では、毎層同期すると tg は 7.5〜5.7 まで落ちる【推定】。単独の 0.74 ms でも、同じ領域では損になる。
   - そのため GT 430 は、**「その層の CPU の見込み時間が L を超えるときだけ使う」というゲート付き**で入れる。主な用途は **prefill**（1 層 100 ms 以上かかるので、遅延 3 ms は無視できる。RX に流す expert を 7 個 / 層減らせて、pp +5 % 前後【推定】）。
   - 単一ストリームの decode では、層 l+1 の入力が層 l の出力（ミディアムの分を含む）に依存する。そのため、層をまたいでまとめて投げることも、パイプラインにすることもできない（5.4 節）。
6. **勝ち筋の判定は、コードを大きく書く前に 2 つ測れば決まる**（9 章）:
   - (a) **生成中と llama-bench（乱数トークン）のルーティングのトレース**。そこから、キャッシュの方策ごとの h をオフラインのシミュレータで出す。
   - (b) **CPU のコールド計算（Q4_0 の 2048×768 expert を n 個、ggml-cpu 2 スレッド）と、RX の Vulkan の MUL_MAT_ID を同時に走らせたときの 1 層の時間**。
   - **llama-bench の pp/tg は乱数トークンを入力する**（`llama-bench.cpp:2181-2219`、`std::rand() % n_vocab`）。合格判定の数字はこの分布で決まる。そのため、imatrix（自然文）の偏りがそのまま効くとは限らない【不明】。

---

## 1. 前提（モデル・ハード・既存実装）

### 1.1 Qwen3-30B-A3B の定数

30B 基準: `qwen3moe 30B.A3B`、48 層、128 expert、8 使用【実測】。llama.cpp の `qwen3moe.cpp`: gating は SOFTMAX、`norm_w = true`（選んだ 8 個の重みを和で正規化）、gate と up は別テンソル（`ffn_gate_exps` / `ffn_up_exps`）【コード確認。手元の `~/dev/sandbox/j72/LLM/j72-gguf/llama.cpp`（3ce355169、2026-10-10）で読んだ。tb250 の `23b0202` と同じかは**【不明】**】。

| 量 | 値 |
| --- | --- |
| n_embd / n_ff_exp | 2048 / 768（公開の構成値。GGUF の `expert_feed_forward_length` で確認する【不明】） |
| 1 expert（Q4_0、gate + up + down） | 3 × 2048 × 768 × 18/32 B = **2,654,208 B = 2.53 MiB** |
| 1 層の expert（128 個） | 340 MB（30B 基準の「1 層約 340 MB」と一致） |
| 全 expert | 48 × 340 MB = 16.3 GB（ファイルは 17.4 GB） |
| 1 トークンで読む expert | 8 × 48 × 2.65 MB = **1.02 GB** |
| 非 expert（1 トークン） | attention は 1 層 18.9 M パラメータ × 48 層、Q4_0 で約 0.51 GB。出力 head は 151,936 × 2048（型は**【不明】**。Q4_0 なら 0.18 GB、Q6_K なら 0.26 GB）。router `ffn_gate_inp` は 128 × 2048 × 48（型は**【不明】**） |
| 全 expert が Q4_0 か | **【不明】**（unsloth の Q4_0 は一部のテンソルを別の型にしている可能性がある。CL11 / CUDA の pdev は Q4_0 / Q6_K のみ対応） |

### 1.2 デバイス（tb250）

| デバイス | 接続 | VRAM | MoE の mat-vec の実効 | 小さな往復（8 KiB） | expert スロット（推定） |
| --- | --- | --- | --- | --- | --- |
| RX 6400（Vulkan0） | x16 表示（カード内ブリッジ）、転送 1.4 GB/s | 4 GB | 70〜82 GB/s【実測、c1-profile / MB3g】 | `tensor_get` 67 µs、1 ノードの compute + sync 73 µs【MB4】 | 約 2.6〜2.8 GiB → **1,050〜1,100**（22〜23 / 層）。基準の n39 で 3870 MiB 中、expert 以外が約 950 MiB【実測からの推定】 |
| WX 2100（Vulkan1） | x1、0.36 / 0.29 GB/s | 2 GB | 26 GB/s、固定費約 0.3 ms / 呼び出し【MB3】 | 同上 76 µs | 約 1.85 GiB → **約 740**（15 / 層）。`-ot` の run で 5 層ぶん 1705 MiB【実測】 |
| GT 730（CUDA0、Kepler） | x1（アイドル時は 2.5 GT/s 表示）、0.40 GB/s | 1 GB | Q4_0 mv 32.7 GB/s、MUL_MAT_ID 1 token で約 25 GB/s【cuda README】 | pinned で 60 µs【MB1c】 | 約 850 MiB → **約 335**（7 / 層） |
| GT 710（CUDA1、Kepler） | 同上 | 2 GB | 約 10 GB/s【cuda README】 | 60 µs | 約 1.85 GiB → **約 730**（15 / 層） |
| GT 430（REMOTE0、Fermi、別プロセス） | 同上 | 1 GB（964 MiB） | q6_K 12.5 GB/s（CL11 サーバー）。Q4_0 の MUL_MAT_ID は**【不明】**（MB3 の素朴カーネルで 7.8 GB/s） | 1 job 単独 0.74 ms、SGLX0 の中で 2.4〜3.3 ms【gt430 README と依頼】 | 約 850 MiB → **約 335**（7 / 層） |
| CPU（Celeron G3930、2C、AVX なし） | - | RAM 31 GB | **約 7.6 GB/s（1 expert 0.35 ms）【推定、下記】** | - | 残り全部（mmap） |

**CPU の実効帯域の推定（GT 430 との比較用）**:
- ggml-cpu で LFM2.5 の 1 層（24.8 MB）が 2.49 ms → **10.0 GB/s**【MB3 (iv)】。
- CPU だけの 30B の tg 3.10 = 323 ms / トークンで、読む量は 1.02 GB + 非 expert 約 0.7〜0.8 GB → **約 5.4 GB/s**（attention などを含む全体）【推定】。
- `-ncmoe 34/39` の tg からの逆算では、1 層の CPU expert が約 2.8 ms → **約 7.6 GB/s**【推定、5.2 節】。
- この文書では 1 expert = **0.35 ms**（幅は 0.27〜0.49 ms）を使う。GT 430 の 12.5 GB/s は CPU の約 1.6 倍（1 expert 0.21 ms + 往復）。GT 710 の約 10 GB/s は約 1.3 倍。
- ggml-cpu が SSE4 / SSSE3 の経路でビルドされているかは**【不明】**。baseline のビルドは `-DGGML_NATIVE=OFF` で AVX 系を全部 OFF にしており、`GGML_SSE42` を明示していない。x86 の `ggml_vec_dot_q4_0_q8_0` は `__AVX2__` などの分岐で、SSE のみのときはスカラー経路になる可能性がある。`GGML_SSE42=ON` のビルドで CPU 側が速くなれば、tg の全体が上がる（基準の測り直しも要る）。9 章 MB-H1。

### 1.3 既存の SGLX0（この worktree の現状）

- `composite_device.cpp` は、**出力射影の MUL_MAT を行で分割する版**と、GET_ROWS を実装している。`supports_op` は MUL_MAT、GET_ROWS、形の op だけ。`buffer_from_host_ptr` は無い（caps は false）。`alloc_buffer_n` も未実装。
- `pdev.h` の IF は「行の piece を upload して、mat-vec / mat-mat を submit / wait する」だけ。MUL_MAT_ID（スロットの選択）の入口は無い。
- 実装済みの pdev: `pdev_ggml`（Vulkan・ggml-cpu。Vulkan1 以降はワーカースレッド）、`pdev_cl11`、`pdev_cuda`（Q4_0 / Q6_K。CUDA backend 自体には MUL_MAT_ID Q4_0 がある）、`pdev_scrapgpu`（REMOTE、ワーカースレッド）。
- 遅延入力コピー（`cpy_tensor_async` で受けて、読む node があるときだけ同期コピーする）、イベント、`GGML_SGLX_AUTOTUNE` がある。
- **別の担当が `adapters/ggml/composite/` を編集中**（`git status` で `composite_device.cpp`、`pdev_cl11.cpp`、`pdev_scrapgpu.cpp`、`sgl_trace.h` が変更・未追跡）。この設計の実装は、新しいファイルを中心にして、共有ファイルへの変更をフックだけにする（8 章）。

### 1.4 llama.cpp 側の仕組み（読んだ箇所）

- `build_moe_ffn`（`src/llama-graph.cpp:2017-2390`）: `ffn_moe_logits`（MUL_MAT）→ `ffn_moe_probs`（SOFT_MAX）→ `ffn_moe_topk`（argsort_top_k）→ `ffn_moe_weights`（GET_ROWS）→ 正規化（SUM_ROWS、CLAMP、DIV）→ `ffn_moe_up` / `ffn_moe_gate`（MUL_MAT_ID）→ `ffn_moe_swiglu`（GLU）→ `ffn_moe_down`（MUL_MAT_ID）→ `ffn_moe_weighted`（MUL）→ スロットの view の ADD の連鎖（slot 0 + 1 + … + 7 の順）→ `ffn_moe_out`。
- op の振り分け（`ggml-backend.cpp:1061-1127`）: 重みのある op は、重みのバッファの backend に置く。**op offload は、重みが最後の backend（CPU）のホストバッファにあるときだけ**、上位の backend の `offload_op`（Vulkan はバッチ ≥ 32、`GGML_OP_OFFLOAD_MIN_BATCH`）に渡す。
- `--n-cpu-moe N` は `blk.i.ffn_(up|down|gate)_exps=CPU`（i < N）を足すだけ。`-ot` と同じ仕組み。
- mmap（`llama-model.cpp:1839-1878`）: デバイスの caps の `buffer_from_host_ptr` が真で、その buft がデバイスの既定の buft なら、mmap の範囲（その ctx のテンソルを含む最初から最後まで）を `ggml_backend_dev_buffer_from_host_ptr` で渡してくる。テンソルは `ggml_backend_tensor_alloc` で置かれ（`llama-model-loader.cpp:1656`）、`set_tensor` は呼ばれない。
- ggml-vulkan の待ち（`ggml-vulkan.cpp:345-368`）: `waitForFences`（almost_ready）のあと `getFenceStatus` で回す。c1-profile では、ホストの CPU 時間の約 73 % がこの回る待ち【実測】。**2 コアで CPU が expert を計算している間に Vulkan の待ちが回ると、1 コアを奪われる**（8 章リスク）。

---

## 2. 全体の形

```
llama.cpp（無改造）
  -dev Vulkan0,SGLX0 -ngl 99 -ot 'blk\.[0-9]+\.ffn_(up|gate|down)_exps\.weight=SGLX0'   (mmap 有効のまま)
   ├─ Vulkan0 (RX 6400): 非 expert のすべて（attention、norm、router、出力 head）
   └─ SGLX0: 48 層の MoE ブロック（MUL_MAT_ID ×3 + GLU + MUL + ADD 連鎖）
        ホストの影 = mmap（buffer_from_host_ptr。コピーなし）
        expert_cache（新規）: (layer, expert) → {device, slot} の表、頻度、入れ替え
        pdev のスロット配列:
          Vulkan0  [2048,768,S0] gate / up、[768,2048,S0] down   S0 ≈ 1,050   ← ホット
          Vulkan1  同じ形 S1 ≈ 740                                          ← ホット
          CUDA0    同じ形 S2 ≈ 335（R32 配置）                               ← ホット
          CUDA1    同じ形 S3 ≈ 730（R32 配置）                               ← ミディアム（同じプロセス）
          REMOTE0  同じ形 S4 ≈ 335（サーバー側で R32）                         ← ミディアム（別プロセス、ゲート付き）
          CPU      ggml-cpu。mmap の [2048,768,128] を層ごとに view           ← コールド
        RX のリング（prefill の流し込み用、16〜32 スロット = 42〜85 MB。スロット配列の一部を流用）
```

- **非 expert は全部 RX**（親設計の方式 E と同じ）。WX・NVIDIA は expert のスロットだけを持つ。層分割をやめるので、RX と WX の直列の区間が無くなる（c1-profile の「2 枚が同時に動かない」問題を避ける）。
- 出力 head（`output.weight`）は、既存の SGLX の行分割（`GGML_SGLX_SPLIT`）に載せることもできる。ただし、30B では RX の VRAM を expert に回す方が効く。出力 head は RX に置いたまま（2〜3 ms）を既定にして、行分割は段階 7 の選択肢にする。

---

## 3. MUL_MAT_ID を SGLX0 で実行する経路

### 3.1 llama から受け取るもの

- **テンソル**: `ffn_{gate,up,down}_exps.weight` は `[2048, 768, 128]` / `[768, 2048, 128]`。SGLX0 の `init_tensor`（mmap 経由なら `tensor_alloc` から呼ばれる）で、形（3 次元、ne2 = 128、型が Q4_0）から expert テンソルと判断し、(層, 種類) を登録する。層の番号は名前の `blk.N.` から取る。名前が無ければ（`test-backend-ops`）登録順の連番にする。
- **op**: `supports_op` で次を真にする。
  - MUL_MAT_ID（src0 が SGLX0 の expert テンソル）
  - GLU（SWIGLU）、MUL、ADD、VIEW / RESHAPE / CONT のうち、src を遡ると `ffn_moe_*` に届くもの（親設計 §2.5 と同じ規則）
  - router 一式（MUL_MAT `ffn_moe_logits`、SOFT_MAX、ARGSORT / top_k、GET_ROWS、SUM_ROWS、CLAMP、DIV）は、`ffn_gate_inp` を SGLX0 に置いたときだけ（選択式、3.4 節）
- **graph_compute**: 受け取ったサブグラフを MoE ブロックの雛形と照合し（親設計 §2.6 の plan キャッシュ）、1 層ぶんを 1 つの `moe_job` として実行する。照合に失敗したら、内部の ggml-cpu でノードどおりに逐次実行する（遅いが正しい。ホストの影は本物のホストメモリなので、ggml-cpu はそのまま読める）。

### 3.2 decode（N = 1）

入力（ホストの影に sched がコピー済み）: hidden `[2048]`、ids `[8]`、weights `[8]`。

```
T0 host  : ids[8] を読み、スロットごとに置き場を引く: home(s) ∈ {GPU j のスロット番号, CPU}
T1 host  : GPU ごとに、そのデバイスのスロットの一覧 local_ids_j（長さ n_j）を作る
T2 submit: n_j > 0 の GPU j に（非同期。worker を持つ pdev から先に）
             write hidden(8 KiB) → MUL_MAT_ID(gate, up; local_ids_j) → SWIGLU → MUL_MAT_ID(down; local_ids_j)
             → read y_j [2048 × n_j]（スロットごとの down の出力。重みは掛けない）
T3 CPU   : n_cpu > 0 なら ggml-cpu（2 スレッド）で同じ計算を、mmap の層 l の [.,.,128] に対して全体の ids で
             （n_cpu ごとにキャッシュしたグラフ。重みテンソルの data だけ層ごとに差し替える）
T4 wait  : 全 GPU を待つ（submit の順に）
T5 host  : out = Σ_{s=0..7}  w_s ⊙ y_{s}      （s の順に固定。ggml の MUL → ADD 連鎖と同じ順序）
           → ffn_moe_out（ホストの影）
```

- **スロットごとの出力を返して、重みと合計はホストで行う**。こうすると、置き場が変わっても、足す順序は ggml のグラフと同じ s = 0..7 のまま変わらない。違いは各スロットの計算をしたデバイスの丸め（Vulkan は Q8_1 活性、ggml-cpu は Q8_0 活性）だけになる。転送は 8 KiB × n_j で小さい。
- **CPU の計算は呼び出し元のスレッドで、GPU への投入のあとに行う**。そうすると GPU の計算が CPU の計算に隠れる。待ちは CPU が終わってからなので、Vulkan の回る待ちも短い。
- ggml-cpu の MUL_MAT_ID は、スロットを飛ばす指定ができない（ids は 0..n_as-1 に限る）。そのため、**n_cpu = 1..8 の 8 本のグラフ**（ids `[n_cpu, 1]`）をキャッシュする。層ごとに、重みテンソルの `data` を mmap の該当層に向け直して使う（全層で形が同じなので、48 × 8 本を作らなくてよい）。
- GPU 側も同じで、スロット配列を層によらず 1 本にし、ids を局所のスロット番号にする。`ggml_mul_mat_id(slots_gate, x, local_ids)` がそのまま使える（pdev_ggml）。CUDA は既存の MUL_MAT_ID Q4_0 カーネル、CL11 は既存の生成経路（ids をホストから渡す入口は親設計 §5.3）。
- 同期点は 1 層あたり、**sched の入力コピー（RX から読む）1〜3 本 + GPU の待ち 1 回 + sched の出力コピー（RX へ書く）1 本**。親設計の方式 E と同じ。

### 3.3 prefill（N > 1）

```
T0 host  : ids[8, N] から、expert ごとの (token, slot) の組を数えて並べる（N×8 ≤ 4096、数十 µs〜数 ms）
T1 host  : expert を 3 つに分ける
             (a) GPU j にある           → j がその token の組を計算する（hidden を詰めて送る）
             (b) 無い、かつ組の数 > τ  → RX のリングに流して RX が計算する（流し込みと計算を重ねる）
             (c) 無い、かつ組の数 ≤ τ  → CPU が計算する（τ は 1〜2 を目安。MB-H5 で決める）
T2       : (a) を全 GPU に投入 → (b) の流し込みを開始（チャンク k+1 の upload と k の計算を重ねる）→ (c) を CPU で
T3 wait  → host: token ごとに、スロットの順に w を掛けて足す
```

- 流し込みは RX だけにする（帯域 1.4 GB/s。x1 のカードは 0.3〜0.4 GB/s で、流すより、置いてある分を自分で計算させる方がよい）。
- **流し込みの元は mmap**。pinned のステージング（Vulkan の host buffer type）に memcpy してから送る。ページ可能なメモリから送ると、ggml-vulkan が submit と wait を一度に行うため（pdev_ggml のコメント）。
- 1 層の prefill は 100 ms 以上かかる（7 章）。そのため、GT 430 の 3 ms 級の遅延も、ホストの並べ替えの数 ms も、ほぼ隠れる。

### 3.4 router と ids の取り方

| 方式 | sched の入力コピー / 層 | CPU の負担 | 既定 |
| --- | --- | --- | --- |
| R1: router を RX に置く（llama の既定どおり） | hidden + ids + weights の 3 本。MB4 で約 204 µs | なし | **段階 1 の既定** |
| R2: `ffn_gate_inp` も `-ot` で SGLX0 に置き、ホストで計算する | hidden 1 本、約 67 µs | 2048 × 128 の mat-vec（F32 なら 1 MB）+ softmax + top-8。約 0.1〜0.2 ms【推定】。CPU のコールド計算と直列になる | 段階 3 で測って決める |

- 48 層では、R1 の追加コピーが約 6.5 ms / トークン、R2 の router の計算が約 5〜10 ms / トークン【推定】。差が小さいので、両方を作って測る。
- どちらの方式でも、SGLX0 は ids をホストで読む。**頻度の統計は SGLX0 の中で取れる**（llama の eval callback は要らない）。

---

## 4. ホット集合の決め方と更新

### 4.1 統計

- `(layer, expert)` ごとに、指数減衰のカウンタ `f[l][e]` を持つ（半減期 H トークン。既定 256【推定値。シミュレータで決める】）。decode では 1 トークンで 8 回、prefill では ubatch の N×8 回を加える。
- 48 × 128 = 6,144 個の float。更新は 1 層あたり 8〜4096 回の加算で、無視できる。
- 起動時の初期値は、`GGML_SGLX_MOE_STATS=path` の CSV を読む（前回の終了時に書き出したもの、または imatrix の `.counts`）。ファイルが無ければ一様。
- **prompt で取った統計を生成に効かせる**: imatrix では、同じコーパスの前半の上位 32 が後半の 76〜83 % を覆った【実測】。一方、別のコーパスでは 26〜43 % だった。プロンプトは、その要求の領域を表す最もよい標本になる。prefill の終わり（N > 1 の後に最初の N = 1 が来たとき）に、ホット集合を計算し直す。
- **生成中のルーティングは未測定**【不明】（imatrix はプロンプト処理だけ）。**llama-bench の tg / pp は乱数トークン**なので、合格判定の数字のときの分布はさらに別物【不明】。段階 0 のトレースで測る。

### 4.2 配置の目的関数

1 トークンの時間（6 章）は `Σ_l max_j(G_{l,j}, C_l)`。CPU が律速している間は、ほぼ `Σ_l E[コールドの数_l] × t_cpu` に比例する。そのため、

1. **全体で貪欲に選ぶ**: すべての (l, e) を `f[l][e]` の大きい順に並べ、GPU のスロットの合計数まで取る。層ごとに同じ数を割り当てない。浅い層（0〜5）は偏りが小さい（H/Hmax 0.9）ので、自然に取られる数が少なくなる。層ごとの差は大きい（80 % を覆うのに必要な数が 17〜65【実測】）。
2. **デバイスへの割り当て**（各層の max を小さくする）: 選んだ集合を、層ごとに「速いデバイスから順に頻度の高い順」に詰める。そのうえで、デバイス j の層 l の負荷 `G_{l,j} = fixed_j·P(使う) + 8·p_{l,j}·2.65 MB / B_j` が、その層の CPU の時間 `C_l` を超えないように、LPT で調整する。
   - デバイスの順は、RX → GT 730 → WX → GT 710 → GT 430（5.3 節）。WX は帯域が GT 730 と同程度だが、固定費が 0.3 ms と大きい（Vulkan1 は worker thread）。そのため、1 層に少しだけ置くと損になる。WX には「少数の層の多めの expert」を持たせる。
   - 帯域比で負荷を配る、という親設計の考え方は、decode のデバイス間の配分に使う（prefill は演算性能比、MB5 の教訓）。
3. **複製はしない**（段階 6 で検討）。スロットが足りないうえ、入れ替えのコストが大きいため。

### 4.3 入れ替えのコスト（PCIe x1）

| 送り先 | 1 expert（2.65 MB） | 備考 |
| --- | --- | --- |
| RX 6400 | **1.9 ms**（1.4 GB/s） | ggml-vulkan の転送が計算と同じキューで直列になるかは**【不明】**（MB-H4） |
| WX 2100 | **7.4 ms**（0.36 GB/s） | |
| GT 730 / GT 710 | **6.6 ms**（0.40 GB/s）。リンクがアイドル表示の 2.5 GT/s のままなら約 10.6 ms | R32 への詰め直しをホストで行う（Celeron の CPU 時間を食う。1 expert 1〜3 ms【推定】）。CUDA は別ストリームで計算と重ねられる見込み |
| GT 430（REMOTE） | 7〜11 ms + サーバーの R32 化 | サーバーの CPU が、本体のコールド計算と 2 コアを取り合う |

**入れ替えの規則（ヒステリシス付き）**:

- 候補 e（今は CPU）と、追い出す v（GPU j にある）について、得 = `(f_e − f_v) × Δt_j`、損 = `upload_j`。
  - `Δt_j` は、CPU と GPU j の 1 expert の時間差（RX で約 0.32 ms）。
  - `f` は、1 トークンあたりの選択確率。
- **入れ替える条件**: `f_e > (1 + α) f_v`（α = 0.25）、かつ `(f_e − f_v) × Δt_j × T_horizon > upload_j`（T_horizon = 256 トークン）。
  - 例: RX で `f_e − f_v = 0.02` なら、0.02 × 0.32 × 256 = 1.6 ms で、1.9 ms に届かず入れ替えない。
  - 0.05 なら 4.1 ms > 1.9 ms で入れ替える。
  - WX / NVIDIA は upload が 4 倍なので、差が 0.1 以上のときだけ入れ替える【推定】。
- **予算**: 1 トークンあたり、デバイスごとに最大 1 expert（RX）/ 0.25 expert（x1 のカード）を、層の境目に分けて送る。1 expert = 3 行列なので、0.88 MB ずつ 3 回に分けられる。decode の 1 トークン（目標 50〜80 ms）に対して、追加は 2〜4 % 以内に抑える。
- **prefill のときの無料の入れ替え（RX）**: prefill では、RX は GPU に無い expert をほぼ全部流し込む（7 章）。流し込んだ expert がホット集合に入るべきものなら、リングのスロットを「キャッシュのスロット」に付け替え、代わりに追い出す expert のスロットをリングに回す。**転送は増えない**。RX の入れ替えは、ほぼこれで済む。
- x1 のカードの入れ替えは、decode 中に少しずつ行う。prompt の領域が変わったときに、WX / NVIDIA が追いつくには、約 100〜300 トークンかかる見込み【推定】（例: 30 % を入れ替えるなら、WX 220 expert × 7.4 ms = 1.6 s）。
- 入れ替え中のスロットは、upload が終わるまで「CPU で計算」と扱う（古い内容を読まない）。upload の完了はイベントで確かめる。

### 4.4 再現性

- 置き場が実行の履歴で変わるので、同じプロンプトでも、スロットを計算したデバイスが違えば logits がわずかに変わる。
- 判定と比較のために **`GGML_SGLX_MOE_FREEZE=1`**（統計ファイルから配置を決めて、以後は入れ替えない）を用意する。KLD はこのモードと動的モードの両方で取る。

---

## 5. 3 層案（ホット／ミディアム／コールド）の検討

### 5.1 案の整理

| 層 | 置き場 | 条件 |
| --- | --- | --- |
| ホット | RX 6400、WX 2100（+ GT 730） | 同じプロセス。1 expert の計算が CPU の 3〜10 倍速い |
| ミディアム | GT 710（同じプロセス）、GT 430（別プロセス REMOTE0） | 速さは CPU の 1.3〜1.6 倍。並列の作業者として、CPU の負担を減らす |
| コールド | CPU（mmap） | 1 expert 0.35 ms【推定】 |

各層の時間は `max(ホットの GPU, ミディアム, CPU)` + 同期。ミディアムは、**CPU と同時に動いて、CPU の担当を減らす**ときだけ得になる。

### 5.2 校正（`-ncmoe` の基準から、CPU の 1 層の時間を求める）【推定】

`T = T_fix + Σ_{CPU の層} (C_layer + 境界) + Σ_{GPU の層} G_layer` で、30B 基準の tg を説明する。
- `T_fix`: RX の非 expert 約 6 ms + 出力 head 2〜3 ms + 小さな op と境界、約 15〜19 ms。
- RX n39（9 層が GPU）: 136 ms【実測】。
- AMD `-ot` n34（RX 9 層、WX 5 層）: 130 ms【実測】。
- 差の 6 ms は、5 層を CPU から WX に移した結果で、`5 × C_layer ≈ 6 + 5 × (WX 1 層 0.8 ms + 境界 2 回 0.8 ms)` になる。ここから **`C_layer ≈ 2.8 ms`（1 expert 0.35 ms）**。
- n39 の全体（18 + 39 × 2.85 + 9 × 0.33 ≈ 132 ms）とも合う。
- ただし **n41 / n44 の tg（5.96 / 4.67〜5.75）は、このモデルより約 20 % 遅い**。原因は未特定。モデルの誤差は ±20 % と見る。

### 5.3 GT 730 と GT 710 はどちらに置くか（決定）

| カード | 帯域（CPU 比） | 同期 | VRAM | 判断 |
| --- | --- | --- | --- | --- |
| GT 730 | 25〜33 GB/s（3.3〜4.3 倍） | 60 µs | 1 GB（約 335 expert） | **ホット**。RX の次に頻度の高い帯を持たせる。WX より固定費が小さいので、多くの層に少しずつ置いてよい |
| GT 710 | 約 10 GB/s（1.3 倍） | 60 µs | 2 GB（約 730 expert） | **ミディアム（同じプロセス）**。容量が大きく、同期が安い。担当する帯（層あたり 15 個、全選択の 6〜12 %）は、1 層 8 × 0.1 × 0.26 ms ≈ 0.2 ms で、CPU の時間に隠れる |

シミュレーション（6.3 節）では、GT 730 をホットに加えると tg +11〜15 %、さらに GT 710 をミディアムに加えると +4〜20 %（同じ領域で 21.8 → 25.3、別の領域で 8.7 → 10.0、10.4 → 12.7）【推定】。どちらも加える価値がある。

### 5.4 GT 430（ミディアム、別プロセス）

**容量**:
- 964 MiB から、context・CL11 のカーネル・ステージングを引いて、約 850 MiB【推定】。
- 850 MiB / 2.53 MiB = **約 335 expert = 層あたり 7 個**。

**どれだけコールドが減るか**（imatrix の頻度から。7 個の帯がそのコーパスの選択を覆う割合。48 層平均）:

| ホットの直後に置いた場合 | 同じ領域（readme 前半→後半 / py 前半→後半） | 別の領域（readme→en_legal / readme→py / py→readme） |
| --- | --- | --- |
| AMD のホット（上位 32 / 層）の次、順位 32〜38 | 0.058 / 0.032 | 0.044 / 0.055 / 0.066 |
| AMD + GT 730（上位 39）の次 | 0.045 / 0.032 | 0.047 / 0.070 / 0.057 |
| AMD + GT 730 + GT 710（上位 53）の次 | 0.029 / 0.020 | 0.053 / 0.053 / 0.049 |

- コールドに回る選択は 2〜7 ポイント減る。1 トークンで、8 × 48 × 0.02〜0.07 = **8〜27 回**の expert 計算が CPU から GT 430 に移る。
- CPU の時間に換算すると **2.7〜9.4 ms / トークン**。これは GT 430 が完全に隠れた場合の上限。
- 計算方法: `logs/imat/*_h1/_h2.counts.json`（同じ領域）と、コーパス全体の `.counts.json`（別の領域）を、層ごとに A の順位で並べ、B の選択の割合を足した。

**損益分岐**:
- 層 l で GT 430 を使うと、その層の時間は `max(G_hot, C_l − n_m·t_cpu, L)` になる。使わなければ `max(G_hot, C_l)`。CPU が律速なら、得になる条件は **`L < C_l`**（その層で GT 430 が無いときの CPU の時間）。
- `C_l = 0.05 + 8·(1 − h_l)·0.35 ms`。

| 状況 | h（ホット + GT 710） | C_l | 損益分岐 L* |
| --- | --- | --- | --- |
| 同じ領域、ホット全部（RX・WX・GT 730・GT 710） | 0.90〜0.92 | 0.27〜0.33 ms | **約 0.3 ms** |
| 同じ領域、AMD だけがホット | 0.76〜0.83 | 0.53〜0.72 ms | **約 0.5〜0.7 ms** |
| 別の領域（配置が追いつく前） | 0.41〜0.57 | 1.25〜1.70 ms | **約 1.0〜1.5 ms** |

シミュレーション（6.3 節と同じモデル。GT 430 は、その層で 1 つでも当たったときだけ投入し、当たる確率は 15〜34 %）の tg【推定】:

| L | 同じ領域（readme） | 同じ領域（py） | 別の領域（r→legal） | 別の領域（r→py） |
| --- | --- | --- | --- | --- |
| GT 430 なし | 39.5 ms（25.3） | 38.1 ms（26.2） | 100.4 ms（10.0） | 78.6 ms（12.7） |
| 0.3 ms | 39.1 | 38.0 | 93.2 | 71.7 |
| 0.74 ms（単独の実測） | 42.3（損） | 40.3（損） | 93.2（得 7 %） | 72.4（得 8 %） |
| 1.5 ms | 49.9（損） | 45.7（損） | 96.9（得 3 %） | 80.9（損） |
| 2.4 ms（SGLX0 の中の現状） | 59.0（損 33 %） | 52.1（損） | 111.2（損） | 95.2（損） |
| 3.3 ms | 68.0 | 58.4 | 125.7 | 109.5 |
| 毎層必ず同期する場合、L = 2.4 | 133 ms（**7.5 t/s**） | 133 | 133 | 133 |

- 依頼にあった「48 層で毎回同期すると約 115 ms」は、この表の最後の行に当たる。実際は、当たった層だけ投入すれば（ゲート）損は小さくなるが、それでも得にはならない。
- **スラックによるゲート**（その層の CPU の見込み時間 `C_l` の推定値が、測った L の EMA を超えるときだけ GT 430 の帯を使う。超えなければ、その層の GT 430 の帯の expert は CPU で計算する）を入れると、損は 0 になり、別の領域では得だけが残る。上の表の「L = 2.4 でゲートあり」は全部「GT 430 なし」と同じ値になる。

**層ごとに同期しない形（数層まとめる、層をまたぐパイプライン）は、単一ストリームの decode では作れない**:
- 層 l+1 の attention の入力は、層 l の `ffn_moe_out`（GT 430 の分を含む）を足した残差。数式の上で直列。
- まとめられるのは、(i) prefill（1 回の呼び出しで N トークンを処理し、層の時間が 100 ms 以上）、(ii) 複数の要求（`-np`。llama は 1 つの ubatch に並べるので、prefill と同じ形になる）、のどちらか。
- 出力を 1 層遅れて足す、などの近似は、合格条件（§8.1 の KLD）を壊すので採らない。

**GT 430 の使いどころ（結論）**:
1. **prefill**: 層あたり 7 expert ぶん、RX への流し込みが減る（18.6 MB / 1.4 GB/s = 13 ms / 層、約 6 %）。GT 430 の計算は、その 7 expert に当たる token の組（ubatch 512 で 1 層 80〜280 組【推定】）。Q4_0 の mat-mat の速さは**【不明】**。RX の 1 層の流し込み時間（100〜130 ms）より短ければ、隠れる。
2. **decode**: スラックのゲート付きで。L が 0.3〜0.7 ms まで縮むまでは、同じ領域ではほぼ使われない。
3. 実装の優先度は低い（段階 6）。いまの L の膨らみ（2.4〜3.3 ms）の原因が CPU の取り合い（サーバーのスレッド + クライアントの worker + Celeron 2 コア）なら、この設計では CPU がコールドの計算でさらに埋まるので、**L はもっと悪くなる**恐れがある（8 章リスク）。

---

## 6. コストモデル（decode）

### 6.1 式

```
T_tok = T_fix + Σ_{l=1..48} [ B_l + max( max_j G_{l,j}, C_l ) + S_l ]
  T_fix   : RX の非 expert（attention 約 6 ms、出力 head 2〜3 ms、小さな op）≈ 10〜12 ms【推定】
  B_l     : 境界。sched の入力コピー（R1: 約 0.2 ms、R2: 約 0.07 ms）+ 出力の set（6 µs）+ RX の split の起動
            48 層で 5〜10 ms。T_fix と合わせて「固定 18 ms」とする（幅 15〜22）
  G_{l,j} : GPU j の固定費（RX 0.07、WX 0.3、CUDA 0.06 ms）+ n_{l,j} × 2.65 MB / B_j
  C_l     : 0.05 + n_{l,cpu} × 0.35 ms（n_cpu = 8 − ヒット数）
  S_l     : ホストの合計（8 × 2048 の積和、数 µs）と、入れ替えの upload の割り込み
```

### 6.2 ヒット率と tg（CPU 律速、全層で同じ h、固定 18 ms の場合）【推定】

| h | C_l (ms) | T_tok (ms) | tg（t/s） |
| --- | --- | --- | --- |
| 0.9 | 0.33 | 34 | 29.6（GPU 側が律速になるので上限） |
| 0.8 | 0.61 | 47 | 21.1 |
| 0.7 | 0.89 | 61 | 16.5 |
| 0.6 | 1.17 | 74 | 13.5 |
| 0.5 | 1.45 | 88 | 11.4 |
| 0.4 | 1.73 | 101 | 9.9 |
| 0.3 | 2.01 | 114 | 8.7 |
| **0.19** | 2.33 | **130** | **7.70（基準と同じ）** |
| 0.1 | 2.57 | 141 | 7.1 |

- 感度: 1 expert の CPU 時間が 0.27〜0.49 ms、固定が 15〜22 ms の幅で動くと、損益分岐の h は 0.05〜0.35 の範囲で動く。
- **基準（tg 7.70、pg 22.87、pp 59.8）を超える条件**:
  - tg: h ≳ 0.2〜0.35。一様（偏りなし）でも、4 枚のスロット約 2,500 で h ≈ 0.42 になるので、超える見込み。
  - pp / pg: 7 章。
  - 3 指標を**同じ構成で**超える必要がある（§8.1）。prefill の経路（7 章）が無いと pp が崩れる（SGLX0 の CPU 計算だけでは pp 3.6 級）ので、段階 5 までは pp の判定はできない。

### 6.3 構成ごとのシミュレーション【推定】

- 方法: imatrix の counts で、層ごとに A の頻度順に、RX 19・WX 13・GT 730 7・GT 710 14・GT 430 7 個（スロットの見積もりより少なめで保守的）を、この順に割り当てる（4.2 節の推奨順 RX → GT 730 → WX とは違う。順を変えた場合は未計算）。B の分布で、各デバイスと CPU の期待負荷を計算し、層ごとに max を取って足した。
- **楽観的になる点**: 期待値の max（max の期待値ではない）で計算した。配置は A の頻度で最初から正しいとした。入れ替えのコストを入れていない。
- 計算に使ったスクリプトは scratchpad の使い捨て（リポジトリには入れていない）。段階 0 の U-sim で、正式なシミュレータにする。

| 構成 | 同じ領域 readme（h / tg） | 同じ領域 py | 別の領域 r→legal | 別の領域 r→py |
| --- | --- | --- | --- | --- |
| A: RX + WX | 0.76 / 18.9 | 0.83 / 22.6 | 0.26 / 8.3 | 0.38 / 9.7 |
| B: + GT 730（ホット） | 0.82 / 21.8 | 0.86 / 24.5 | 0.30 / 8.7 | 0.44 / 10.4 |
| C: + GT 710（ミディアム） | 0.90 / 25.3 | 0.92 / 26.2 | 0.41 / 10.0 | 0.57 / 12.7 |
| D: + GT 430（L = 0.3 ms） | 0.93 / 25.7 | 0.94 / 26.4 | 0.46 / 10.7 | 0.62 / 14.0 |

- 別の領域の列は「配置が元の領域のまま固定」の値。動的な入れ替えで、数百トークン後には同じ領域の列に近づく見込み（入れ替えの速さは 4.3 節）。
- llama-bench の tg128 は乱数トークンを入力するので、どの列に近いかは**【不明】**。一様なら h ≈ スロットの割合 0.42（構成 C）で、tg 約 10。

---

## 7. pp の経路（op offload との比較・共存）

### 7.1 今の pp（基準 59.8）が何で決まっているか

- RX 単体 `-ncmoe 41` の pp512 59.8【実測】は、op offload で、CPU にある 41 層 × 340 MB = 13.9 GB を ubatch ごとに RX へ流している（実効 1.63 GB/s【推定、8.56 s / 13.9 GB】）。
- 1 層ぶん（gate / up / down の各 113 MB）の置き場が RX の VRAM に無いと、op offload が止まって 18 に落ちる【実測、ppchk】。

### 7.2 SGLX0 の流し込み（7 章の既定）

| 項目 | llama の op offload | SGLX0 の流し込み |
| --- | --- | --- |
| 送るもの | テンソル丸ごと（128 expert、すでに GPU にあるものや使わないものも含む） | ubatch で使われ、GPU のどこにも無い expert だけ |
| 置き場 | 1 テンソル分（113 MB）以上 | リング 16〜32 スロット（42〜85 MB）。二重バッファで upload と計算を重ねる |
| 他の GPU の expert | 使えない（層分割） | 置いてあるデバイスがその場で計算する（x1 では活性だけ送る） |
| 対象 | CPU バッファの重みだけ | SGLX0 の expert |

**見積もり【推定】**（ubatch 512、1 層、RX への流し込み 1.4〜1.63 GB/s）:

| 構成（層あたりの GPU 常駐数） | 流す expert / 層 | 流す時間 / 層 | 48 層 | pp512 |
| --- | --- | --- | --- | --- |
| RX + WX（約 37） | 約 91 | 148〜172 ms | 7.1〜8.3 s | 62〜72 |
| + GT 730 + GT 710（約 59） | 約 69 | 112〜131 ms | 5.4〜6.3 s | 81〜95 |
| + GT 430（約 66） | 約 62 | 101〜117 ms | 4.8〜5.6 s | 91〜106 |

- 前提: 常駐しているデバイスの計算が、RX の流し込みの時間に隠れること。
  - WX: 15 expert × 約 35 組 × 9.4 MFLOP ≈ 5 GFLOP、367 GFLOPS で 14 ms。
  - GT 730: 67 GFLOPS で約 35 ms。
  - GT 710: 約 34 GFLOPS で約 100 ms。
  - どれも流し込みの時間（100〜170 ms）より短い【推定】。
- ubatch で 1 度も使われない expert は流さない。CPU が数個の token の expert（τ 以下）を引き取ると、さらに流す量が減る。どちらの効果も**【不明】**（段階 0 のトレースで、ubatch あたりの expert ごとの token 数の分布を出す）。
- RX の計算（1 層 4096 組 × 9.4 MFLOP = 38.5 GFLOP、約 3 TFLOPS で 13 ms）は、流し込みに隠れる。
- 複数の x1 カードに同時に流すと、チップセット（B250、DMI 3.0）の上で合計帯域が伸びるかは**【不明】**（MB-H5）。
- **pg512,128**: 640 / (512 / pp + 128 / tg)。pp 80〜100、tg 15〜20 なら 6.4〜8.5 s + 5.1〜6.4 s で **約 43〜55**【推定】。基準は 22.87。

### 7.3 VRAM の取り合い

- RX は「キャッシュのスロット」と「リング」が同じスロット配列の中にある。リングの数を `GGML_SGLX_MOE_RING`（既定 24 スロット = 64 MB）で決める。prefill の無料の入れ替え（4.3 節）は、リングとキャッシュのスロットの付け替えで行う。
- llama 側の RX の compute buffer は、`-ub` に比例する（`-ub 512` で数百 MB【不明。段階 0 で `GGML_SCHED_DEBUG` / メモリのログで見る】）。`-ub` を大きくすると pp は伸びる（30B 基準の `-ub 2048` で 146）が、スロットが減って tg が下がる。**同じ構成で pp / tg / pg を全部超える**必要があるので、`-ub` は 512 前後で探す。

### 7.4 op offload と共存させる案（予備）

- 層 0..K−1 の expert は `-ot ...=CPU` のまま（llama の op offload で prefill。decode は CPU だけ）、層 K..47 は SGLX0、という層単位の併用。
- decode では CPU の層が 8 expert すべて CPU になり、キャッシュの利点を捨てることになる。SGLX0 の流し込みが、MB-H5 で 1.2 GB/s を下回るときの退避策としてだけ持つ。

---

## 8. 実装の段階分け

各段階の「正しさ」は合否、「計測」は記録（CONTRIBUTING）。KLD の基準は §8.1 を 30B に当てはめる: CPU 基準の logits に対して、平均 KLD ≤ 30B の単体最速構成（`-ncmoe 39` か `-ot` n34）の値 × 1.2、top-1 ≥ 同 − 1 ポイント。30B の CPU 基準の logits は未作成（30B 基準の残件）なので、段階 0 で作る。

### 段階 0: トレースとマイクロベンチ（コードは計測用だけ）

- **U-trace**: ルーティングのトレース。llama の公開 API と `cb_eval` で `ffn_moe_topk-*` を読み、`(token, layer, 8 ids)` を出す小さなツール。llama.cpp 本体は改造しない。3 種類を取る: (a) 自然文の生成、(b) `llama-bench` と同じ乱数トークンの tg / pp、(c) imatrix のコーパスのプロンプト処理。
- **U-sim**: オフラインのシミュレータ。トレースを再生して、方策（静的・EMA・半減期・α・予算）ごとに、h、入れ替えの数、6.1 節の式による T_tok を出す。
- マイクロベンチ MB-H1〜H6（9 章）。
- 30B の CPU 基準の logits（`llama-perplexity --kl-divergence-base`、`-c 512 --chunks 4`）と、基準構成の KLD。
- GGUF の確認: n_ff_exp、各テンソルの型、出力 head の型。
- **完了条件**: 乱数トークンと自然文の両方の h、MB-H1 / H2 の値で、6.1 節の式を評価し、go / no-go を決める（9 章の最後）。

### 段階 1: SGLX0 の MoE ブロック（CPU だけ、mmap）

- `buffer_from_host_ptr`（caps 真）、expert テンソルの登録、MoE ブロックの照合と plan、decode / prefill の CPU 実行（ggml-cpu の 8 本のグラフ）、ホストでの合計、照合失敗時の逐次実行。
- **正しさ**:
  - (a) `test-backend-ops -b SGLX0 -o MUL_MAT_ID`（と GLU、MUL、ADD）
  - (b) MoE ブロック単体の比較ツール（`moe_composite_check` の 30B 形。ggml-cpu の素のグラフと NMSE）
  - (c) 30B の KLD（CPU 基準。CPU 同士なので、ほぼ 0 になるはず）
- **計測**: tg / pp（期待値は `-ncmoe 48` 相当）と、1 層の境界・CPU・合計の内訳（`GGML_SGLX_PROFILE`）。

### 段階 2: RX のスロット（ホット、静的配置）

- pdev の IF にスロットの配列と `moe_submit` を足す（8.1 節）。`pdev_ggml`（Vulkan0）で実装する。統計ファイルからの静的配置（`FREEZE`）。
- **正しさ**: (a)〜(c)、加えて自己整合（全部 CPU と、RX にキャッシュした場合の KLD < 0.001）。
- **計測**: tg を h ごとに（統計ファイルを変えて h を動かす）。6.2 節の表と比べ、式の係数を実測に差し替える。

### 段階 3: 複数の GPU（WX、GT 730、GT 710）と配置

- `pdev_cuda` / `pdev_cl11` にスロット、配置（4.2 節）、router の R1 / R2 の比較。
- **正しさ**: 段階 2 と同じ。pdev の組み合わせごとに。
- **計測**: 構成 A / B / C の tg、層ごとの max の内訳。

### 段階 4: 動的な統計と入れ替え

- EMA、prompt の終わりの再計算、ヒステリシス、予算、upload とイベント。
- **正しさ**: 入れ替えの最中も出力が壊れないこと（入れ替えを毎トークン強制するテストモードで、KLD < 0.001）。`FREEZE` との一致。
- **計測**: 領域が変わるプロンプト（readme → py_code など）で、tg の時間変化と、h が戻るまでのトークン数。

### 段階 5: prefill の経路

- 常駐分のその場での計算、RX のリングの流し込み（pinned ステージング、二重バッファ）、CPU の端数、無料の入れ替え。
- **正しさ**: (a)〜(c)。ubatch の大きさ（1、2、32、128、512、端数）で。
- **計測**: pp512、pg512,128（`-ub` の掃引）、流し込みの実効帯域、RX のリングの使用率。

### 段階 6: GT 430（ミディアム、REMOTE、スラックのゲート）

- `pdev_scrapgpu` にスロットの job（MUL_MAT_ID）を足す。サーバー側の op は既存の MUL_MAT_ID を使う（CL11 / CUDA）。L の EMA とゲート。
- **前提**: 別の担当の調査で、L が 1 ms 未満になっていること。そうでなければ prefill だけに使う。
- **計測**: 5.4 節の表の実測版。

### 段階 7: 仕上げ

- 出力 head の行分割（既存の SGLX）との併用。`-ub` と RX のリングの数の掃引。§8.1 の全構成との最終比較（交互測定、REPS ≥ 5）。

### 8.1 pdev の IF の追加（案。`pdev.h` は別の担当と調整してから固める）

```cpp
// pdev.h に追加（既存の piece の API はそのまま）
struct moe_slots { virtual ~moe_slots() = default; };          // 1 つの形・型の expert スロット配列（gate, up, down）
class pdev {
    ...
    virtual bool       supports_moe(ggml_type t) { return false; }
    virtual moe_slots *moe_slots_create(ggml_type t, int64_t n_embd, int64_t n_ff, int n_slots) { return nullptr; }
    // slot に expert を 1 個書く。非同期。完了は moe_upload_done() で確かめる。R32 化などは pdev の中で行う
    virtual bool       moe_upload(moe_slots *, int slot, const void *gate, const void *up, const void *down, size_t bytes_each) { return false; }
    virtual bool       moe_upload_done(moe_slots *, int slot) { return true; }
    // N トークン。pairs[i] = {slot, token}（slot の昇順に並べてある）。x は [n_embd, N_x]（詰めた token）。
    // out は [n_embd, n_pairs]（pair の順。重みは掛けない）
    virtual bool       moe_submit(moe_slots *, const float *x, int64_t N_x, const int32_t *pair_slot, const int32_t *pair_tok, int n_pairs, float *out) { return false; }
    // wait() は既存のものを使う
};
```

### 8.2 Sonnet サブエージェントへの委譲の単位（ファイルが重ならないように）

| 単位 | ファイル（新規または変更） | 依存 | 完了条件 |
| --- | --- | --- | --- |
| H1 トレースツール | `adapters/ggml/tools/moe_route_trace.cpp`（新規。llama の公開 API。ビルド方法はヘッダのコメントに） | なし | Mac で小さな MoE（LFM2.5 など）のトレースが出る。乱数トークンのモードがある |
| H2 シミュレータ | `scripts/hec_sim.py`（新規、依存なし）、`docs/benchmarks/.../hec-sim/README.md` | H1 の出力形式 | imatrix の counts と、H1 のトレースの両方を入力にできる。6.3 節の値を再現する |
| H3 マイクロベンチ | `adapters/ggml/tools/hec_bench.cpp`（新規。`moe_rt_bench.cpp` の流儀） | なし | MB-H1〜H5 が Mac（ggml-cpu + Metal または CPU 2 つ）で動く。tb250 用のコマンドを `DRY_RUN=1` で出す |
| H4 キャッシュの方策 | `adapters/ggml/composite/moe_cache.{h,cpp}`（新規。ggml に依存しない C++。統計、配置、入れ替えの判定） | なし | 単体テスト（`tests/` か tools の小さなドライバ）。H2 と同じ方策で同じ判定になる |
| H5 MoE ブロックの実行 | `adapters/ggml/composite/moe_block.{h,cpp}`（新規。照合、plan、decode / prefill の振り分け、CPU 実行、合計） | H4、8.1 の IF | `test-backend-ops -b SGLX0 -o MUL_MAT_ID` が Mac の CPU だけの構成で全件通る |
| H6 SGLX0 本体へのフック | `composite_device.cpp`（`supports_op`、`graph_compute` の振り分け、`buffer_from_host_ptr`、`init_tensor`）、`composite.h`、`adapters/ggml/CMakeLists.txt` | H5。**別の担当の編集が落ち着いてから** | 既存の出力射影の SGLX の `test-backend-ops` が回帰しない。MoE モードは `GGML_SGLX_MOE=1` のときだけ |
| H7 pdev のスロット | `pdev.h`（8.1 の追加）、`pdev_ggml.cpp`、`pdev_cuda.cpp`、`pdev_cl11.cpp`、`pdev_scrapgpu.cpp` | 8.1 の IF を確定したあと | pdev ごとに、CPU 参照と NMSE が一致する（H3 のドライバで） |
| H8 計測スクリプト | `docs/benchmarks/moe-ep-<日付>/hec/scripts/*`（`run.sh` の flock・温度待ちを踏襲） | H6 | `DRY_RUN=1` でコマンドを出せる |
| H9 文書 | この文書と親設計の更新 | 各段階 | 【推定】を実測に差し替える |

- 共有ファイル（`composite_device.cpp`、`composite.h`、`pdev.h`、`pdev_*.cpp`、`CMakeLists.txt`）は H6・H7 だけが触る。いま別の担当が編集中なので、その作業のコミット後に始める。H1〜H5 と H8 は新しいファイルだけなので、すぐに並行できる。
- tb250 での実測は SSH を伴うので、ユーザーの許可を得てから行う（親設計と同じ）。

---

## 9. リスクと、最初に測る勝ち筋判定マイクロベンチ

### 9.1 リスク

| リスク | 影響 | 対処 |
| --- | --- | --- |
| **llama-bench の乱数トークンでは、ルーティングに偏りが無い** | 合格判定の tg で h がスロットの割合（約 0.42）まで落ちる | それでも基準を超える見込み（6.2 節、tg 約 10）。実際の文章での速度も別に報告する。段階 0 の U-trace (b) で先に確かめる |
| **生成中の分布がプロンプトと違う** | prompt の統計が効かない | EMA で追う。U-sim で半減期を決める |
| **2 コアの取り合い**: CPU のコールド計算（2 スレッド）、Vulkan の回る待ち、pdev のワーカースレッド（WX、REMOTE）、CUDA の同期 | 並列のはずが直列になる | 投入 → CPU → 待ちの順序を守る。WX は非同期の API で worker を使わない形を探る。`cuCtxCreate` の flags を BLOCKING_SYNC にする。MB-H2 で先に測る |
| ggml-cpu が SSE を使っていない（スカラー経路） | CPU の 1 expert が 0.35 ms より遅い、または速くできる余地がある | MB-H1 で `GGML_SSE42=ON` のビルドと比べる。SGL の cpu backend（SSE2）とも比べる |
| ggml-vulkan の upload が計算と同じキューで直列になる | 入れ替えが decode を止める | 1 行列ずつ（0.88 MB、0.6 ms）に分ける。MB-H4 |
| 30B の expert のテンソルに Q4_0 以外の型が混ざる | CUDA / CL11 の pdev が使えない層がある | GGUF で確認する。その層は Vulkan と CPU だけで扱う |
| sched が SGLX0 に MoE 以外の op を割り当てる / 48 層で split が増えすぎる | 境界が増える | `supports_op` を `ffn_moe_*` の連鎖に限る。`GGML_SCHED_DEBUG` で確かめる |
| 置き場の違いによる数値の差 | KLD が自己整合の基準（< 0.001）を超える | スロットごとの出力をホストで固定の順に合計する。`FREEZE` で切り分ける |
| GT 430 の遅延（2.4〜3.3 ms）が縮まない | ミディアムの利得 0、毎層同期すると大きく損 | スラックのゲート。prefill に限る（5.4 節） |
| mmap のページがページキャッシュから追い出される | コールドの計算がディスク待ちになる | RAM 31 GB に 17.4 GB の mmap。他の大きなプロセスと同時に動かさない。`mlock` は RAM の圧迫を見て判断する |
| 30B の非 expert と KV が RX に収まりきらない | RX のスロットが見積もり（1,050）より少ない | `-c` と `-ub` を絞る。段階 0 で実際の確保量を記録する |

### 9.2 マイクロベンチ（`hec_bench`、すべて tb250、bench lock、85 ℃待ち）

| ID | 内容 | 判定に使う量 |
| --- | --- | --- |
| **MB-H1** | CPU のコールド計算: 30B の形の Q4_0 expert（mmap 上の `[2048,768,128]`）に、N = 1、n_cpu = 1..8 の MUL_MAT_ID × 3 + SWIGLU を、ggml-cpu 2 スレッドで。ビルドは既定 / `GGML_SSE42=ON` / SGL の cpu backend | `t_cpu`（1 expert）と固定費 |
| **MB-H2** | 同時実行: MB-H1 と並行して、RX（Vulkan0）の MUL_MAT_ID（n = 0..8 スロット）、GT 730（CUDA）、WX（Vulkan1）を「投入 → CPU → 待ち」で | 1 層の壁時計が `max` か和か。CPU の計算が何 % 遅くなるか |
| MB-H3 | 48 層ぶんの境界: RX の 1 層ぶんの attention（または 1 ノード）→ `tensor_get` × 1 / 3 → set、を 48 回 | `B_l`（R1 / R2） |
| MB-H4 | 入れ替え: 2.65 MB を RX / WX / CUDA に upload しながら、MB-H2 の 1 層を回す | upload の実効帯域と、計算の遅れ |
| MB-H5 | prefill の流し込み: mmap → pinned → RX のリング（二重バッファ）で MUL_MAT_ID の mat-mat を回す。WX / NVIDIA への同時の流し込みも | 流し込みの実効帯域（1.4〜1.63 GB/s が出るか）、チップセットで合計が伸びるか |
| MB-H6 | GT 430 の REMOTE の MUL_MAT_ID job（スロット 1〜3 個、N = 1）。CPU を MB-H1 で埋めた状態と、空けた状態で | L（5.4 節の損益分岐と比べる） |

**go / no-go**（段階 0 の終わりに評価する）:

```
T_pred = 18 ms(実測の B_l と T_fix に差し替え) + Σ_l max(G_l, 0.05 + n_cpu,l · t_cpu)
   n_cpu,l は U-sim（U-trace の乱数トークン版と自然文版）の期待値、t_cpu は MB-H1、
   並列の効率は MB-H2 で補正（壁時計 / max の比を掛ける）
1000 / T_pred > 7.70 × 1.3（乱数トークン）     → 段階 1〜5 へ進む
1000 / T_pred ≤ 7.70 × 1.1                   → 中止。原因（CPU の取り合い / 偏りが無い）を記録する
その間                                        → RX + GT 730 + GT 710 の構成だけで段階 3 まで進め、実測で決める
```

最初の 1 日で決まるのは、U-trace の乱数トークン版の h（U-sim）、MB-H1 の `t_cpu`、MB-H2 の並列の効率。この 3 つで、この設計を進めるかどうかがほぼ決まる。

---

## 10. 未決事項・不明点

- Strata の expert cache の実装の細部（docs のクローンが無かった）。
- tb250 の llama.cpp `23b0202` の `build_moe_ffn`、sched、mmap の経路が、この文書で読んだ手元のツリー（3ce355169）と同じか。
- GGUF の n_ff_exp、expert と出力 head と router の型。
- 生成中、および llama-bench の乱数トークンのときのルーティングの偏り（imatrix はプロンプト処理だけ）。
- ggml-cpu の Q4_0 の dot が、このビルドで SIMD を使っているか。
- RX の非 expert + KV + compute buffer の実際の大きさ（スロットの数を決める）。
- ggml-vulkan の upload と計算が並行するか。2 インスタンスの Vulkan0（llama の分と pdev の分）で、キューを安全に共有できるか（親設計 §6.2 は「1 スレッドからなら共有できる」と実測済み。upload を別スレッドにすると未確認）。
- B250 のチップセットで、複数の x1 への同時転送の合計帯域。
- GT 430 の Q4_0 の MUL_MAT_ID の速さと、SGLX0 の中での遅延が膨らむ原因（別の担当が調査中）。
- n41 / n44 の tg がコストモデルより約 20 % 遅い理由（校正の誤差）。
