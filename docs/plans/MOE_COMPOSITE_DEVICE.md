# MoE 複合デバイス（composite device）設計書

状態: 設計（未実装）。ブランチ `feat/moe-expert-parallel`。作成 2026-10-21。この文書は read-only の調査に基づいており、ビルドも TB250 での実測もしていない。
表記: **【実測】** はリポジトリ内の文書にある既存の測定値（出典を付ける）、**【推定】** はこの文書での見積もり、**【不明】** は未測定・未確認で、最初の段階で測るもの。

---

## 0. 結論（先に）

1. **最初に測るべき比較相手は 21 t/s ではない。** 単体最速（公式 Vulkan RX 6400 + `--n-cpu-moe 8`、tg 21.0 t/s【実測】docs/CL11.md:232）は、時間の約 6 割が CPU で計算する 8 層分のエキスパートに使われていると分解できる（3.2 節）。新しいコードを書かなくても、**`-ot` / `-sm layer` で全エキスパートを GPU（RX 6400 + WX 2100 の公式 Vulkan）に置くだけで、tg は 40 t/s 前後になる見込み【推定】**。複合デバイスは「21 t/s を超える」だけでなく、この既存機能だけの構成（以下 C1）も超えないと意味がない。段階 0 で C1 を測り、それを本当の基準にする。
2. **単一ストリームの decode では、全 MoE 層でエキスパート並列（EP）をすると負ける見込みが高い【推定】。** ホストを経由する同期 1 往復が 0.46〜1.4 ms【実測】（docs/VULKAN_AMD.md:202、docs/MULTI_GPU.md:75）で、22 層 × 1 往復だけで、EP で縮められる時間（エキスパートの読み出しで約 5 ms）を食い尽くす。EP で勝つには、1 往復を **0.1〜0.2 ms 以下**にする必要がある。
3. 勝ち筋は次の**混成（C2）**。
   - RX 6400 に、非エキスパートのすべてと、載る限りのエキスパート層（約 16 層）を置く。これらの層ではデバイス間の往復は起きない。
   - 残り約 6 層のエキスパートだけを複合デバイスに入れ、WX 2100・GT 730・GT 710（・GT 430）に**同時に**計算させる。
   - 往復はこの 6 層だけで起きる。decode は C1 より 10〜20% 速く、prefill は C1 より 1.5 倍前後速い見込み【推定】。
   - 21 t/s の C0 と比べると、tg・pp とも 2 倍前後の見込み【推定】。
4. llama.cpp から見せる形は二つ。先に作るのは **方式 E（エキスパート専用の複合デバイス）**。`-ot "blk\.(1[6-9]|2[01])\.ffn_.*_exps=SGLX0"` で該当するテンソルを受け取り、ggml の scheduler が分割したサブグラフ（MoE ブロック）だけを実行する。llama.cpp からは GPU 1 台に見える。
   - **方式 F（全体を包む複合デバイス）**は、方式 E の境界コストが支配的だと分かったときに作る。RX 6400 の ggml-vulkan も内部で持ち、グラフ全体を受け取る。
5. 物理デバイスとの接続は、内部インターフェース `sglx::pdev` で差し替えられるようにする。実装は 3 種類。
   - `pdev_cl11`: 同じ .so の `cl11_device` を直接呼ぶ。
   - `pdev_ggml`: ggml の公開 API で、別の ggml backend（Vulkan・HIP・CPU・RPC）を包む。
   - `pdev_scrapgpu`: ScrapGpu C API の vtable を包む。remote・cpu・将来の cuda が対象。
   - CUDA（Driver API）は `src/backends/cuda` に ScrapGpu backend として作り、`pdev_scrapgpu` 経由で使う。ggml の include を `adapters/ggml` に閉じる規則と矛盾しない（6 章）。
6. 合格条件のうち **「top-1 ≥ 96%」は、このモデルでは CPU を基準にすると原理的に満たせない**。CPU 同士でも 93.6%、公式 Vulkan は 92.45〜94.31%【実測】（docs/CL11.md:1001-1046）。基準の取り方をユーザーに決めてもらう必要がある（8.1 節）。

---

## 1. 既存コードの該当箇所（読んだもの）

| 場所 | 内容 | 設計への影響 |
| --- | --- | --- |
| `adapters/ggml/ggml-cl11.cpp:67-71` | `cl11_device_context`。ggml の device と buft を、`src/backends/opencl11` の `cl11_device` に結ぶ | 複合デバイスも同じ形にする（ggml 側の殻と、内部の状態を分ける） |
| `ggml-cl11.cpp:276-280, 282-290` | buffer は `cl_mem` 1 個。tensor のアドレスは、ダミーの基底 `CL11_BASE=0x1000` からのオフセット | 物理デバイスのテンソルは CL11 の `cl11_tensor`（mem・td）で直接作れる |
| `ggml-cl11.cpp:395-432` | `set_tensor` / `get_tensor` は blocking（`CL_TRUE`）。R32 は `cl11_rp_transfer` | エキスパートの切り出しを upload するときも、この経路（R32 変換）を使う |
| `ggml-cl11.cpp:434-474` | `cpy_tensor` は別の `cl_context` なら false を返し、ggml がホスト経由でコピーする | デバイス間のコピーは、複合デバイスの内部で自分で行う |
| `ggml-cl11.cpp:542-555` | buft の iface（API 3 の `alloc_buffer_n` / `get_alloc_size_n` は nullptr） | 複合デバイスでは `alloc_buffer_n` を使う（どのテンソルが入るかを確保時に知るため） |
| `ggml-cl11.cpp:653` | `cl11_supports_op_impl`。`MUL_MAT_ID` の条件は 753 行 | 物理デバイスの対応判定は、ここと `src/backends/opencl11/api.cpp:488` を使う |
| `ggml-cl11.cpp:1036-1250` | `graph_compute`。`c->mutex` を全体で保持し（1040 行）、融合を判定して最後に `clFlush` だけ行う（1243 行）。完了は待たない | CL11 は「投入は非同期、完了待ちは synchronize と get_tensor」。複数のデバイスに投入してから待てば、ホスト 1 スレッドでもデバイスは同時に動く（5.2 節） |
| `ggml-cl11.cpp:1296-1313` | backend の iface。`*_async`、`cpy_tensor_async`、event は nullptr | 複合デバイスの外側も、まず同期 API で作る（sched は同期のコピーを使う） |
| `ggml-cl11.cpp:1349-1362` | caps はすべて false | 方式 E の複合デバイスも同じにする。⑤ では `buffer_from_host_ptr=true` にする |
| `ggml-cl11.cpp:1406-1409` | `supports_buft` は同じデバイスの buft だけ | 複合デバイスも、自分の buft だけ |
| `ggml-cl11.cpp:1456-1492` | reg の初期化。`cl11_probe_devices()` と、外れた GPU の自動 RPC（`cl11_serve_dropped_over_rpc`、160 行） | 複合デバイスは同じ reg の末尾に足す（6.1 節）。GT 430 の別プロセス化も、既存の仕組みを流用する |
| `src/backends/opencl11/device_state.h:16-89` | `cl11_device`。mutex 1 個、in-order の queue 1 本、`mid_ids` キャッシュ | 物理デバイスごとに queue が独立しているので、デバイス間の並列は自然に得られる。同じデバイスの中の並列はない |
| `src/backends/opencl11/ops_moe.cpp:19-130` | prompt 経路の `MUL_MAT_ID`。39 行で ids を blocking で読み戻す | 複合デバイスでは ids がホストにあるので、ids をホストから渡す入口を足して、この同期を省く（5.3 節） |
| `ops_moe.cpp:131-186, 188-201` | 生成経路（1 起動で、GPU が ids を読む）と、その振り分け | decode は生成経路をそのまま使える。融合版は 4.3 節 |
| `include/scrapgpu/backend.h:30-60` | ScrapGpu backend の vtable（buffer、stream、op_execute、plan、`stream_device`） | CUDA と remote はこの vtable を通して使う（6.3 節） |
| `src/backends/opencl11/api.cpp:488-494`、`src/backends/cpu/api.cpp:693` | C API の `SCRAPGPU_OP_MUL_MAT_ID`（opencl11 と cpu。remote は転送するだけ） | GT 430 を remote で使う経路が、新しい仕組みを作らずに使える |
| `src/backends/vulkan/api.cpp:1-9` | SGL 自前の Vulkan は ADD・MUL・RMS_NORM・MUL_MAT だけ | MUL_MAT_ID が無いので、AMD は公式の ggml-vulkan を `pdev_ggml` で包む |
| `docs/MULTI_GPU.md:75`、`docs/VULKAN_AMD.md:202` | デバイス間の転送 1 回 ≈ 1.4 ms（CL11 4 枚）、allreduce 1 回 ≈ 0.46 ms（Vulkan 2 枚） | コストモデルの往復時間の根拠 |
| llama.cpp（`~/dev/sandbox/j72/LLM/j72-gguf/llama.cpp`、ggml 0.25.3 / API 3。b11384 系と推定。23b0202 そのものではない） `ggml/src/ggml-backend-impl.h` | v3 の iface。`alloc_buffer_n`、`graph_optimize` の `add_alloc_dep`、meta backend | 設計はこの iface に合わせる。23b0202 の公開ヘッダは b11384 と同一【実測】（docs/HETEROGENEOUS_RUNTIME.md:68）。ただし、`ggml-backend-impl.h` が 23b0202 で同一かは【不明】 |
| 同 `ggml/src/ggml-backend-meta.cpp`、`src/llama-model.cpp:417-608` | `-sm tensor` の meta device。エキスパートは軸 1（n_ff）で分割し、層ごとに allreduce する | 既存の「エキスパートのスライス並列」の実装例。ホスト経由で遅いことが実測済み（VULKAN_AMD.md:202）。ただし、このデバイスにどんな split を使わせるかは llama 側が決めるので、軸 2（エキスパート）では分けられない |
| 同 `src/llama-graph.cpp:2017-2390` | `build_moe_ffn` のノード名（`ffn_moe_topk-<il>`、`ffn_moe_up`、`ffn_moe_gate`、`ffn_moe_swiglu`、`ffn_moe_down`、`ffn_moe_weighted`、`ffn_moe_out`。途中の ADD には名前が付かない） | 方式 E のパターン照合に使う |
| 同 `src/llama-model.cpp:1839-1872` | mmap の場合、既定の buft の device が `caps.buffer_from_host_ptr` を持つと、`buffer_from_host_ptr` で mmap 領域を包む | ⑤（30B）でコピーせずに RAM から供給する入口 |

---

## 2. llama.cpp から見せる形（ggml backend API v3）

### 2.1 方式 E と方式 F

| | 方式 E（エキスパート専用。先に作る） | 方式 F（全体を包む。後で、必要なら） |
| --- | --- | --- |
| llama.cpp での指定 | `-dev Vulkan0,SGLX0 -ngl 99 -ot "<遠隔層>\.ffn_(up\|gate\|down)_exps=SGLX0"`。必要なら `ffn_gate_inp` も | `-dev SGLX0 -ngl 99`（1 台だけ） |
| 複合デバイスが受け取るもの | sched が切り出した MoE ブロックのサブグラフだけ | グラフ全体（split は 1 個） |
| RX 6400 の扱い | llama が直接使う `Vulkan0`。複合デバイスの外にある | 複合デバイスの内部で ggml-vulkan を持つ（pdev_ggml） |
| 境界のコスト | 遠隔層ごとに sched の同期コピーが入力 1〜3 回、出力 1 回 | 遠隔層ごとに、自分で詰めた 1 回の読み戻しと 1 回の書き込み |
| 難しさ | 小さい（Vulkan を包まない） | 大きい（全 op を Vulkan に委譲し、split も自前で処理する） |

方式 E で C2 の構成がそのまま作れる。方式 F で得になるのは、境界 1 回あたりのコスト（sched の get/set、それぞれ同期）を自分で 1 回にまとめられる分と、複数の要求を重ねる（4.5 節）余地だけ。方式 F に進むかは、段階 2 で境界コストを測ってから決める。

### 2.2 device / buffer type / buffer の構造

```
ggml reg "CL11"（既存の reg に足す。6.1 節）
 ├─ CL11_0..n           既存の物理デバイス（複合デバイスに使ったものは列挙から外す。env で選べる）
 └─ SGLX0               複合デバイス（GGML_SGLX_SPEC があるときだけ現れる）
      device ctx: sglx_composite { vector<unique_ptr<pdev>> pdevs; placement; plan cache; pools }
      buft "SGLX0":   alloc_buffer / alloc_buffer_n / get_alignment / get_max_size(=SIZE_MAX) / is_host=false
      buffer:         sglx_buffer { host_shadow (malloc / mmap); per-tensor placement records }
```

- **ホストの影（host shadow）。** 複合デバイスの buffer は、すべてのテンソルをホストの RAM に持つ（`get_base` は本物のホストのポインタ）。そのうえで、エキスパートのテンソルは `set_tensor` のときに物理デバイスへ切り出して upload する。
  - 理由 1: sched の境界のコピー（`get_tensor` / `set_tensor`）が `memcpy` で済む。
  - 理由 2: 配置を変えるとき（②）に、ファイルを読み直さずに済む。
  - 理由 3: ⑤ の RAM 供給がそのまま同じ形になる。
  - RAM の増分は、方式 E の遠隔 6 層で約 1.2 GB【推定】（31 GB のうち）。
- **`alloc_buffer_n`（API 3）を使う。** 確保の時点でテンソルの一覧（名前・形・型）が分かるので、物理デバイスごとの必要量を計算し、物理側のプールを一度に確保できる（④）。`alloc_buffer`（一覧なし）も実装し、こちらは `init_tensor` の時点で遅延して割り当てる。
- **`init_tensor`** で、テンソルごとの配置記録（`sglx_placement`）を作る。`tensor->extra` は使わない（ggml 側が使う可能性があるため）。buffer の context の `unordered_map<const ggml_tensor*, rec>` に持つ。
- **分割の対象**: 3 次元 `[K, M, E]`、型が Q4_0 / Q8_0 / Q4_K / Q6_K / F16 / F32、かつ大きさが閾値（1 MiB）以上のテンソル。名前は判定に使わない。`test-backend-ops` が名前を付けないので、形だけで判定する。2 次元以下のテンソル（router の `ffn_gate_inp` など）は、ホストの影だけに置く。
- **compute buffer**（sched が作る中間テンソル）もホストの影。MoE ブロックの中間（up・gate・swiglu）は、実行の際に物理デバイス側で扱うので、ホストには書かない（5 章）。

### 2.3 エキスパートのテンソルの物理配置

二つのモードを同じ記録形式で持つ。

- **EP（エキスパート軸、①）**: 物理デバイス j がエキスパートの集合 S_j を持つ。物理側のテンソルは `[K, M, |S_j|]` で、局所の番号から大域の番号への表を持つ。
  - バイトの範囲は、エキスパート e が `[e*nb2, (e+1)*nb2)` なので、`set_tensor(offset, size)` がどこで切れていても、範囲を対応づけるだけで切り出せる。
  - 複製（②）では、一つの e を複数の j が持ってよい。
- **ES（エキスパートのスライス、①'）**: すべてのエキスパートを n_ff の方向に切る。
  - up / gate の `[n_embd, n_ff, E]` は行（ne1）の範囲 `[r0_j, r1_j)`。各エキスパートの中で行は連続している。
  - down の `[n_ff, n_embd, E]` は K（ne0）の範囲。Q4_0 のブロック（32 要素）単位で切り、行ごとにまとめ直す（ホストで並べ替える）。
  - 出力は部分和になり、ホストで合計する。
  - LFM2.5 は n_ff = 1792 = 56 ブロック。比は 1/56 の刻みで、帯域に比例して決められる。
- **decode の既定は ES**（帯域に比例した完全な負荷分散で、ルーティングの偏りの影響がない）。prefill と ②・⑤ は EP（エキスパートを丸ごと持つ必要があるため）。両方を実装し、段階 1 で両方を測る。
- 物理側の upload は、物理デバイスの API（CL11 なら `cl11_rp_transfer` で R32 化）で行う。R32 はスライスごとの配置なので、`[K, M_j, E_j]` の新しいテンソルとして作れば、既存の条件（`nb[2] % 16 == 0`）をそのまま満たす。M_j は 32 の倍数にそろえる。

### 2.4 非エキスパートのテンソルの置き場

- 方式 E: llama が直接使う Vulkan0（RX 6400）。複合デバイスはエキスパート（と、選択として `ffn_gate_inp` / `ffn_exp_probs_b`）だけを持つ。
- 方式 F: 主デバイス（RX 6400 の pdev_ggml）。複合デバイスの buffer は、テンソルごとに「主デバイスに置く」と記録し、主デバイスの buft で実体を確保する。meta backend の `simple_tensor` の方式（`ggml-backend-meta.cpp:403-490`）と同じ。

### 2.5 `supports_op` / `offload_op` / `supports_buft`

方式 E の `supports_op` は「MoE ブロックの op だけ」を true にする。sched が隣接するノードを広げて割り当てても、ほかの層の op を吸い込まないようにするため。

| op | 条件 |
| --- | --- |
| `MUL_MAT_ID` | src0 が SGLX の buffer にあり、2.2 節の分割の対象。src1 は F32 |
| `GLU`（SWIGLU）、`MUL`、`ADD`、`VIEW` などの形の変更 | src を遡って `ffn_moe_*` の名前のノードにたどり着ける（`ffn_moe_weighted` の view の ADD の連鎖を含む）。名前の規約は llama-graph.cpp:2017-2390 |
| router 一式（`MUL_MAT` の `ffn_moe_logits`、SIGMOID、ADD、ARGSORT、GET_ROWS、SUM_ROWS、CLAMP、DIV） | 選択時のみ（`GGML_SGLX_ROUTER=host`）。`ffn_gate_inp` が SGLX にあるとき |
| ほか | false |

- `offload_op`: nullptr（llama の op offload にエキスパートを持っていかれないようにする）。
- `supports_buft`: 自分の buft だけ。
- 名前に頼る照合は脆い（llama.cpp の更新で変わり得る）。**照合に失敗したら、全部を物理デバイス 0 で ggml の op どおりに逐次実行する**フォールバックを必ず持つ（5.1 節）。正しさは保たれ、速くないだけになる。

### 2.6 graph_compute での振り分け

1. **パターン照合（plan の作成、③）**: 受け取ったサブグラフを、MoE ブロックの雛形 `{MUL_MAT_ID up, MUL_MAT_ID gate, GLU, MUL_MAT_ID down, MUL weights, ADD × (k-1)}` と照合する。router を含む場合は、その前段も照合する。結果は `moe_block_plan`（入力のテンソル、重みの記録、出力のテンソル、物理デバイスごとのカーネル列）にする。
2. **キャッシュ**: キーは `(n_nodes, 各ノードの op・ne・data のポインタ)` のハッシュ。llama のグラフの再利用（`llama-context.cpp:1412`）と sched の split の再利用で、decode ではキャッシュに当たる見込み【推定】。外れても照合のコストは数十ノード分。
3. **実行**: 5 章の手順で、物理デバイスに同時に投入し、待って、集約する。
4. **出力**: `ffn_moe_out` をホストの影に書いて、`GGML_STATUS_SUCCESS` を返す（同期で完了させる）。sched はこのあと `get_tensor` で読み、Vulkan0 へ `set_tensor` する。

---

## 3. コストモデル

### 3.1 モデルの定数

**LFM2.5-8B-A1B Q4_0**（`lfm2moe.cpp`、`mid_check` の形 docs/CL11.md:820、`--cpu-moe` の 46 split docs/CL11.md:211 から）。

| 量 | 値 |
| --- | --- |
| 層 | 24。MoE は 22（`n_layer_dense_lead`=2。46 = 2 + 2×22 からの【推定】。GGUF で確認する） |
| n_embd / n_ff_exp / E / k | 2048 / 1792 / 32 / 4 |
| attention 層と shortconv 層の数 | 【不明】（`n_head_kv(il)==0` で決まる。GGUF で確認する） |
| エキスパート 1 個（Q4_0、up+gate+down） | 2048×1792×3×18/32 B = **6.19 MB** |
| 1 層のエキスパート / 22 層の合計 | 198 MB / **4.36 GB** |
| 1 トークンで読むエキスパート | 4×22×6.19 MB = **545 MB** |
| 非エキスパート（1 トークンで読む量） | 約 0.25〜0.32 GB【推定】（attention / conv / dense 2 層 / 出力 head） |
| hidden 1 本 / ids / 重み | 8 KiB / 16 B / 16 B（1 トークン） |

**Qwen3-30B-A3B**（次段。値は公開の構成からの【推定】で、GGUF で確認する）: 48 層、E=128、k=8、n_embd 2048、n_ff_exp 768。エキスパート 1 個 2.65 MB、1 層 340 MB、合計 16.3 GB、1 トークンで読む量 1.02 GB。

### 3.2 物理デバイスの値と、基準 C0 の分解（校正）

| デバイス | 帯域【実測】 | mat-vec の実効（×0.65【推定】、docs/HARDWARE.md の「tg は帯域の 55〜77%」から） | ホスト↔デバイス【実測】 | 1 起動【実測】 |
| --- | --- | --- | --- | --- |
| RX 6400（Gen2 x4） | 126.6 GB/s | 約 82 GB/s | 1.4 / 1.26 GB/s | rusticl 13.8 µs。Vulkan は【不明】 |
| WX 2100（x1） | 43.6 | 約 26（Vulkan） | 0.36 / 0.29 | rusticl 約 100 µs、Clover 15〜20 µs |
| GT 730（x1） | 37.9 | 約 25 | 0.40 | 【不明】（数 µs〜10 µs 前後と推定） |
| GT 710（x1） | 11.7 | 約 7.6 | 0.40 | 同上 |
| GT 430（x1、別プロセス） | 22.9 | 約 15 | 0.40 + RPC 往復 | 同上 + 11.5〜39 µs（M2 での値） |

**C0 = 21.0 t/s = 47.6 ms / トークン の分解【推定】**:

| 部分 | 計算 | 時間 |
| --- | --- | --- |
| RX 6400 | 14 層のエキスパート 347 MB + 非エキスパート約 0.3 GB = 0.65 GB を 82 GB/s で読む。固定費 2〜2.6 ms（VULKAN_AMD 系の実測） | 約 10.5 ms |
| CPU | 8 層。`--cpu-moe` の 22 層で約 80 ms【実測】（CL11.md:234）なので 1 層 3.6 ms | 約 29 ms |
| 残り | 境界 16 回（8 層 × 往復）| 約 8 ms（1 回約 0.5 ms。Vulkan の allreduce 1 回 0.46 ms【実測】と合う） |

この分解は、密モデル 8B Q4_K_M の `-sm layer -ts 8/1` の 14.06 t/s【実測】も同じ式で再現する（RX 4.1 GB / 82 + WX 0.6 GB / 26 ≈ 73 ms ≈ 13.7 t/s）。そこで以下の見積もりにも同じ係数を使う。

### 3.3 decode（1 トークン）の時間の式

- 層分割（直列）: `T = Σ_dev (bytes_dev / B_dev) + fixed + n_boundary × RT`
- 複合デバイスの遠隔層（並列）: `T_layer = RT_in + max_j(T_k_j + bytes_j / B_j) + RT_out + T_host_sum`
  - `RT_in` / `RT_out`: 主デバイスとの往復（方式 E は sched の同期コピー）。
  - `T_k_j`: 物理デバイス j の起動の合計（③ の融合で 1 層あたり 2 起動）。
  - ES のとき `bytes_j = 24.8 MB × s_j`（s_j は帯域に比例させる）。EP のとき、ルーティング次第で偏る（期待値で 1.3〜1.6 倍の最大負荷と【推定】。4〜5 枚に 4 個を割り振るため）。

| 構成 | 内容 | decode の見積もり |
| --- | --- | --- |
| C0 | 公式 Vulkan RX 6400 + `--n-cpu-moe 8` | 47.6 ms（21 t/s）【実測】 |
| C1（新しいコード不要） | `-dev Vulkan0/Vulkan1 -sm layer -ts 18/6`（WX 2100 に 6 層を丸ごと）。CPU は使わない | RX 約 8.5 ms + WX 6 層 0.22 GB / 26 ≈ 8.5 ms + 固定費 4〜5 ms + 境界 2 回 1 ms ≈ **22〜23 ms（43〜45 t/s）【推定】** |
| C1'（同上） | 遠隔 6 層のエキスパートだけを `-ot` で WX に（attention は RX のまま） | 境界が 12 回に増え、C1 より遅い見込み |
| C2（複合デバイス、方式 E） | RX に 16 層ぶんのエキスパートと非エキスパートのすべて。遠隔 6 層のエキスパートを ES で {WX, GT 730, GT 710} に（実効の和 ≈ 58 GB/s） | RX 約 8.5〜9 ms + 6 × (0.43 ms + T_k + RT) + 固定費。RT = 0.5 ms で約 19 ms（52 t/s）、RT = 0.15 ms で約 16 ms（62 t/s）【推定】 |
| C3（全層 EP、参考） | 22 層すべてで、5 枚に EP/ES | エキスパートの読み出し約 3.5〜5 ms + 非エキスパート 3.7 ms + 固定費 + 22 × RT。RT = 0.5 ms で約 22 ms、RT = 0.1 ms で約 13 ms【推定】 |

**decode で勝てる条件**:
- C2 が C1 に勝つ条件は `6 × RT + 6 × T_k < (WX が 6 層を一人で読む時間 ≈ 8.5 ms) − 6 × 0.43 ms`。つまり RT + T_k ≲ 0.55 ms。
- C3 が C2 に勝つのは、RT ≲ 0.1 ms のときだけ。CUDA の pdev（同期 10〜20 µs 級と推定）と、RX 6400 側の読み戻しが速いときに限られる。

### 3.4 prefill（多トークン）の時間の式

- 計算: `T_layer = max_j(FLOP_j / F_j + xfer_j)`。`xfer_j = n_tok × p_j × 16 KB / BW_x1`。
  - p_j: トークンが j の担当エキスパートに当たる確率（ES は 1、EP は約 `1 − (1 − f_j)^k`）。
  - 1 エキスパート 1 トークンは 22 MFLOP。RX 6400 の実効は約 3〜4 TFLOPS【推定】（8B Q3_K_S の pp 248 t/s【実測】から）。WX 約 0.5 TF、GT 730 約 0.29 TF（0.6B の pp128 241 t/s から逆算）、GT 710 約 0.15 TF。
- **x1 のカードは、全層の EP に参加させると、転送だけで損をする**。22 層に参加すると、x1 1 枚で 1 トークン約 0.3〜0.9 ms の転送が要る。RX 自身がその分を計算すると約 0.05〜0.55 ms【推定】。x1 のカードは、**RX に載らない層の計算にだけ**使う（C2 と同じ配置）。
- C2 の prefill【推定】: 遠隔 6 層は、エキスパート 0.53 GFLOP / トークンを、約 0.9 TF の合計で計算して約 0.6 ms。転送は 0.24 ms / トークン（デバイスごとに並列）。RX 側は約 0.8 ms。合わせて約 1.4〜1.6 ms / トークン → **pp 約 600〜700 t/s**。
- C1 は WX が 6 層を丸ごと計算するので約 1.5 ms + RX 0.8 ms ≈ 2.3 ms → 約 430 t/s。
- C0 の pp は**【不明】**（文書に無い）。llama の op offload（バッチ 32 以上で、CPU にある 8 層ぶん 1.58 GB の重みを 1 ubatch ごとに PCIe で送る）が律速なら、ub 512 で 300〜450 t/s 程度と【推定】。段階 0 で測る。
- ub が大きいほど、往復（RT）と ids の処理が均される。ub 512 の推奨は段階 1 で決める。

### 3.5 勝てないときの代替（効く順）

1. **CPU を使わない**: C1 だけで、C0 の 2 倍前後が見込める。まず測る。
2. **遠隔層を少なく、厚く**: 遠隔層の数を減らすと RT の回数が減る。RX に載せるエキスパートを最大にする（KV と compute buffer を `-c` と `-ub` で絞る）。
3. **router をホストで計算**（`GGML_SGLX_ROUTER=host`）: 方式 E の入力のコピーを 3 本から 1 本（hidden だけ）にする。router は 2048×32 の積で、Celeron でも数十 µs【推定】。
4. **境界の往復を速くする**: busy-poll（`clGetEventInfo`）、ピン留めされたホストメモリ（`CL_MEM_ALLOC_HOST_PTR` + map、CUDA の `cuMemHostAlloc`）、CUDA の pdev。
5. **複数の要求を重ねる**（`-np N`、合計スループット）: llama は同じ ubatch に並べるので、1 層あたりの RT が N トークンで均される。単一ストリームの tg は速くならない。
6. **方式 F**: sched の get/set（各同期）を、自分で 1 回の読み戻しにまとめる。
7. GT 710 は帯域が 7.6 GB/s しかないので、ES の比を 0 にして外す選択肢を持つ（遅いデバイスが max を決めてしまわないように）。

---

## 4. 施策 ②〜⑤ の設計と導入時期

### 4.1 ② ホットエキスパートのキャッシュ（段階 4）

- **統計**: 複合デバイスは ids をホストで見る（`GGML_SGLX_ROUTER=host` のとき、または入力の ids から）。`(layer, expert)` ごとの使用回数を数え、`GGML_SGLX_STATS=path` に CSV で出す。
  - 全 22 層の統計を取るには、RX 側の層も一時的に複合デバイスに入れて走らせる（遅くてよい。統計を取るだけ）。
  - LFM2.5 の偏り（どれだけ集中するか）は**【不明】**で、最初に測る。
- **静的配置（既定）**: 起動時に統計ファイルを読み、二つの条件で `(layer, expert) → device` を決める。
  1. RX に置くエキスパートを、使用頻度の高いものから VRAM が埋まるまで選ぶ。
  2. 遠隔の各デバイスの期待負荷（頻度 × バイト数 / 帯域）が均等になるようにする。
  - 貪欲法（LPT）で足りる。704 個（22 層 × 32）。
- **層をまたぐホットキャッシュ**（全層を EP にし、ホットなものを RX に）は、全層で ids の読み戻し（RT）が要るので、RT ≲ 0.1 ms のときだけ有効（3.3 節）。統計と RT を測ってから決める。
- **複製**: VRAM に余りがあれば、ホットなエキスパートを複数のデバイスに置く。実行時は、トークンの k 個を各デバイスに振り分ける問題（最大負荷の最小化。k×デバイス数が小さいので総当たり）を、ホストで 1 µs 程度で解く【推定】。
- **動的な移動**（実行中に入れ替える）は、PCIe x1 では 1 エキスパート 6.2 MB で 15 ms かかるので、decode の中では行わない。統計が変わったら、次の起動で静的配置をやり直す。

### 4.2 ③ 起動回数の削減（段階 3。plan の再利用は段階 1 から）

- **plan のキャッシュ**（2.6 節）: 照合の結果、物理デバイスのテンソルの記述子、カーネルの引数の固定部分を、plan に保存する（docs/API_CONTRACT.md 8 章の「選択は plan を作るとき 1 回」と同じ規則）。
- **物理デバイスでの融合（decode）**: 1 層・1 デバイスあたり 2 起動にする。
  1. `k_moe_gate_up_swiglu`: up と gate の mat-vec を 1 回で行い、出力で SWIGLU。既存の `k_mvri_q4_0` のループを流用し、2 つの重みの行を同じ work-group で読む。
  2. `k_moe_down_wsum`: down の mat-vec。局所のスロットについて `Σ_s w_s × down_s` を**スロットの順に固定して**足す（決定的にするため）。
  - ggml のグラフでは、1 層 8〜9 起動（3 つの MUL_MAT_ID + GLU + MUL + 3 つの ADD）かかっている。
- **ホストでの合計**: デバイスの部分和を、デバイスの番号順に固定して足す（2048 個の float × デバイス数で数 µs）。
- prefill は既存の prompt 経路（gather → エキスパートごとの mat-mat → scatter）を、ids をホストから渡す形で使う（5.3 節）。融合（gate と up の mat-mat を 1 回にする）は段階 3 の後半で行う。
- 計測: `GGML_CL11_LAUNCH_LOG=1` と同等の数を、複合デバイスでも出す（`GGML_SGLX_LAUNCH_LOG=1`。物理デバイスごとの起動数とトークンあたりの合計）。

### 4.3 ④ 事前確保のプール（VMM 相当。段階 1 で最小、段階 3 で完成）

- **重みのアリーナ**: `alloc_buffer_n` の一覧から、物理デバイスごとの必要量を求め、確保の上限ごとに大きな塊で確保する。
  - 上限: NVIDIA 390 は公称上限を超えても確保できる。Mesa は `CL_DEVICE_MAX_MEM_ALLOC_SIZE` を厳守する（docs/HARDWARE.md）。
  - 確保の直後に末尾へ書き込んで、遅延確保の失敗を検出する。既存の `cl11_alloc_buffer` の方式。
  - テンソルを塊の中にオフセットで置く。
- **活性のステージング**: 物理デバイスごとに `hidden_in [n_embd × ub_max]`、`ids/weights [k × ub_max]`、`partial_out [n_embd × ub_max]`、`ff_tmp [M_j × k × ub_max]` を起動時に一度だけ確保する。prefill 用の gather / scatter の作業領域も同じ。**トークンごとの確保はしない。**
- **ホスト側**: ピン留めされたステージング（OpenCL 1.1 の `CL_MEM_ALLOC_HOST_PTR` + `clEnqueueMapBuffer`、CUDA の `cuMemHostAlloc`）。ピン留めで NVIDIA の小さな転送が速くなるかは**【不明】**で、マイクロベンチ MB1 の変種で測る。
- 仮想アドレスの予約（cuMemAddressReserve 等の VMM）そのものは、OpenCL 1.1 にも CUDA 9.1 以前にも無い。「ピークに合わせた一括確保と、オフセットでの再利用」で代わりにする。

### 4.4 ⑤ mmap / 少ない RAM での供給（段階 6、30B 用）

- **入口**: 複合デバイスの caps を `buffer_from_host_ptr=true` にし、`buffer_from_host_ptr` を実装する。llama は、mmap の範囲を複合デバイスの buffer として渡してくる（`llama-model.cpp:1858-1872`。条件は既定の buft であること）。ホストの影は、コピーではなく mmap の領域を指す。
- **配置**: ホットなエキスパートは GPU に upload する（4.1 節の静的配置。VRAM 合計約 9〜10 GB）。コールドなものは mmap に置いたまま（ページキャッシュ。31 GB の RAM に 16.3 GB が収まる）。
- **コールドなエキスパートは CPU で計算する（GPU に転送しない）**:
  - 2.65 MB を x1 で送ると約 6.6 ms かかる。
  - Celeron で mat-vec すると約 0.3〜0.5 ms【推定】（RAM の帯域と、AVX なしの dequant の速さで決まる）。
  - CPU の pdev は、`pdev_ggml`（ggml-cpu）か `pdev_scrapgpu`（SGL の cpu backend、SSE2）を使う。どちらが速いかを MB3 で比べる。
- 30B の decode【推定】: GPU の命中率を h とすると、1 トークン 384 回の呼び出しのうち `(1−h) × 384` 回が CPU になる。h = 0.8 で CPU 側が約 30〜40 ms。GPU と CPU が並列でも、48 層 × RT がかかる。15〜20 t/s が上限の目安。30B の単体最速の値は**【不明】**で、段階 6 の初めに測る。
- SSD: 31 GB の RAM に収まるので、30B では不要。RAM が足りないとき mmap のページフォールトで自然に SSD から読むが、速さは保証しない。

### 4.5 パイプライン化（複数の要求）

- 単一ストリームでは、層の依存があるので重ねられない。
- `-np N` の並列要求では、llama が N トークンを 1 つの ubatch に並べる。複合デバイスから見れば、小さな prefill と同じ形になる。RT が N で均されるので、C3（全層 EP）が有利になる側に動く。
- 方式 F なら、「要求 A の層 L の遠隔エキスパート」と「要求 B の attention」を重ねられる。llama.cpp を改造しない前提では、llama の ubatch の作り方を変えられないので、**方式 F の中でも実現は難しい**（【不明】。優先度は低い）。

---

## 5. MUL_MAT_ID の分散実行の手順

### 5.1 decode（1 トークン、方式 E、ES の既定）

入力は sched が複合デバイスの buffer（ホストの影）に入れた次のもの。
- hidden `[2048, 1]`
- ids `[4, 1]`
- weights `[1, 4, 1]`
- `GGML_SGLX_ROUTER=host` のときは hidden だけ。router はホストで計算する。

```
T0  ホスト: ids と weights をホストの影から読む（同期は不要。sched がコピー済み）
T1  各 pdev j について（ホスト 1 スレッド、非同期 API）:
      write(hidden → hidden_in_j, 8 KiB, non-blocking)
      write({ids, weights} → idsw_j, 32 B, non-blocking)    // ES は全スロット、EP は局所スロットだけ
      launch k_moe_gate_up_swiglu_j, k_moe_down_wsum_j
      read(partial_out_j → host_j, 8 KiB, non-blocking) + flush
T2  各 pdev j について wait（clWaitForEvents / cuEventSynchronize / pdev_ggml は synchronize）
T3  ホスト: out = Σ_j host_j（j の番号順）→ ffn_moe_out（ホストの影）
```

- 同期点は、方式 E で 1 層あたり **sched の入力コピー（Vulkan0 からの読み戻し）1〜3 回 + 複合デバイス内の待ち 1 回（全デバイス並列）+ sched の出力コピー（Vulkan0 への書き込み）1 回**。
- **スレッド**: Celeron は 2 コアなので、物理デバイスごとにワーカースレッドを持つ（busy-wait すると CPU を奪い合う）のは避ける。非同期の投入ができる pdev（CL11、CUDA）は、呼び出し元のスレッドで「全部に投入してから、順に待つ」。待ちの時間は最大のデバイスで決まる。同期の API しかない pdev（`pdev_ggml` の `graph_compute` + `synchronize`、remote）だけ、専用のワーカースレッドを持ち、条件変数で起こす（busy-wait しない）。
- CL11 の物理デバイスを呼ぶときは、そのデバイスの `cl11_device::mutex` を取る。llama 側が同じ CL11 の物理デバイスを直接使わないように、複合デバイスに使ったデバイスは列挙から外す（2.2 節）。
- RX 6400 の rusticl の不具合（ARGSORT の直後のカーネルが ids を早く読む、docs/CL11.md:814）は、ids をホストから渡すので関係しない。ただし、物理デバイス側で router を計算する変種を作るときは、同じ対処（間に submit を入れる）が要る。
- **フォールバック**: パターン照合に失敗したら、各 op を ggml の意味どおりに、物理デバイス 0（全エキスパートを持つ必要があるので、この場合はホストの影から CPU 参照で計算する）で実行する。遅いが正しい。

### 5.2 prefill（n トークン、EP の既定）

```
T0  ホスト: ids [k, n] から、(token, slot) の組をエキスパートごとに並べる（ホストでソートする。n×k ≤ 4096 なら数十 µs）
T1  各 pdev j: そのデバイスが持つエキスパートに当たる token の hidden だけを詰めて write（hidden_in_j [n_embd, n_j]）
      + 組の表（局所のエキスパート番号、局所の列、重み）を write
      launch: エキスパートごとの mat-mat（gate, up）→ swiglu → mat-mat（down）→ 重み付きで scatter（token, slot の部分出力）
      read partial_out_j [n_embd, n_j_tokens]（token ごと。スロットは装置内で固定順に合計済み）
T2  wait（全デバイス）
T3  ホスト: out[:, t] = Σ_j partial_j[:, t]（j の順。デバイスに無い token は 0）
```

- 同期点は decode と同じ数。転送量は `n × p_j × 8 KiB` を往復で。
- ES の prefill（全デバイスに全 token を送る）は、x1 では転送が増えるので既定にしない（3.4 節）。段階 1 で比較のために測る。
- 重み付きの和を**どこで行うか**: スロットの合計はデバイスの中で（固定順）、デバイスの合計はホストで（固定順）。主デバイスで合計する変種（部分和を主デバイスへ送って ADD）は、x1 の転送が増え、主デバイスの起動も増えるので採らない。方式 F で主デバイスに部分和を送る経路は、転送が 1 回減る場合だけ使う。

### 5.3 CL11 への追加の入口（ids をホストから渡す）

- `src/backends/opencl11/ops_moe.cpp` に、`cl11_op_mul_mat_id_host(c, dst, s0, s1, const int32_t * ids_host, ...)` を足す。`cl11_op_mul_mat_id_sync` の 39 行目の読み戻しを省いた版。
- 融合カーネル `k_moe_gate_up_swiglu` / `k_moe_down_wsum`（Q4_0 / Q8_0 / Q4_K / Q6_K の R32）を、`src/backends/opencl11/kernels/moe_fused.cl` に置く。ggml の型は使わない（scrapgpu の dtype）。

### 5.4 同期点とコストのまとめ（1 層あたり）

| | 方式 E decode | 方式 E prefill | 方式 F decode |
| --- | --- | --- | --- |
| 主デバイスとの往復 | 入力 1〜3 回（router をホストで計算するなら 1 回）+ 出力 1 回 | 同じ | 詰めた読み戻し 1 回 + 書き込み 1 回 |
| 物理デバイスの待ち | 1 回（並列） | 1 回（並列） | 1 回（並列） |
| 起動 / デバイス | 2（融合後）、融合前は 6 前後 | 2 + 2 × 使用エキスパート数 | 2 |
| 転送 / デバイス | 8 KiB × 2 + 32 B | n × p_j × 8 KiB × 2 | 8 KiB × 2 |

---

## 6. 物理バックエンドの差し替え

### 6.1 プラグインの形

- 複合デバイスは `libggml-cl11.so` の中に置き、既存の reg（`ggml_backend_cl11_reg`、`ggml-cl11.cpp:1456`）のデバイス一覧の末尾に足す。
  - 別の .so にすると、CL11 の静的ライブラリが二重に入り、同じ GPU の `cl_context` を二つ作ることになる。Fermi と Kepler の衝突（docs/MULTI_GPU.md:11）と、VRAM の二重確保の原因になる。
  - `GGML_BACKEND_DL_IMPL` は 1 つの .so に 1 つの reg なので、reg は増やさない。
- 有効にする: `GGML_SGLX_SPEC="es:cl11=WX 2100:0.45,cl11=GT 730:0.43,cl11=GT 710:0.12"`。形式は `mode:kind=selector[:ratio],...`、kind は `cl11|ggml|scrapgpu|remote`。

### 6.2 内部インターフェース `sglx::pdev`（adapters/ggml/composite/pdev.h、C++。ggml の型を持たない）

```cpp
namespace sglx {
struct pdev_caps { bool async_enqueue; size_t max_alloc; size_t mem_total; double bw_gbs; };
struct pdev_tensor { void * impl; int dtype; int64_t ne[4]; size_t nb[4]; };    // dtype は scrapgpu_dtype
class pdev {
public:
    virtual ~pdev() = default;
    virtual const char * name() const = 0;
    virtual pdev_caps caps() const = 0;
    virtual bool arena_reserve(size_t bytes) = 0;                                 // ④
    virtual bool weight_create(const pdev_tensor & desc, pdev_tensor & out) = 0; // アリーナの中に
    virtual bool weight_upload(pdev_tensor & t, const void * host, size_t off, size_t size) = 0; // R32 化などはここで
    virtual bool stage_reserve(int n_embd, int m_local, int k, int ub_max) = 0;
    // decode / prefill の MoE ブロック 1 回ぶんを投入する（非同期の pdev は投入だけで戻る）
    virtual bool moe_submit(const moe_job & job) = 0;     // hidden、組の表、重み、出力先のホストのポインタ
    virtual bool wait() = 0;
};
}
```

| 実装 | 置き場 | 中身 |
| --- | --- | --- |
| `pdev_cl11` | `adapters/ggml/composite/pdev_cl11.cpp` | 同じ .so の `cl11_device` を直接呼ぶ（mutex、queue、`cl11_rp_transfer`、5.3 節の入口と融合カーネル） |
| `pdev_ggml` | `adapters/ggml/composite/pdev_ggml.cpp` | 別の ggml backend を、ggml の**公開 API**（`ggml-backend.h`）で使う。`ggml_backend_dev_by_name("Vulkan1")` など。MoE ブロックの小さなグラフ（`ggml_mul_mat_id` × 3、`ggml_swiglu`、`ggml_mul`、合計）をその backend の buffer に作り、`ggml_backend_graph_compute` + `synchronize`。ワーカースレッドで動かす |
| `pdev_scrapgpu` | `adapters/ggml/composite/pdev_scrapgpu.cpp` | ScrapGpu C API（`scrapgpu_op_execute` / plan）。remote（GT 430）、cpu、将来の cuda |

**ggml-vulkan を内部で使えるか（dlopen の問題）**:
- llama.cpp は `GGML_BACKEND_DL` で、全部の backend を `ggml_backend_load_all` で読み込む。複合デバイスは、読み込みの順番に依存しないように、**最初に使う時点**（`init_backend` か最初の `alloc_buffer`）で `ggml_backend_reg_by_name("Vulkan")` を引く。
  - 見つからなければ、`ggml_backend_load("<path>/libggml-vulkan.so")` を自分で呼ぶ（ggml-base の公開 API。登録も ggml 側で行われる）。
  - 同じ .so を二度 dlopen しても、ハンドルは同じで、reg も static なので同じものが返る。
- 同じ RX 6400 を、llama の `Vulkan0` と複合デバイスの内部の両方で使う（方式 E で RX を pdev に含める場合）と、ggml-vulkan の同じデバイスに backend のインスタンスが 2 つできる。ggml-vulkan のキューが、インスタンスをまたいで安全に共有されるかは**【不明】**。方式 E では、RX 6400 を複合デバイスに入れない構成（C2）を既定にして、この問題を避ける。
- **include の規則との整合**: `pdev_ggml` と複合デバイスの本体は ggml の型を扱うので、`adapters/ggml` に置く。`ggml-backend.h` / `ggml-backend-impl.h` 以外の ggml-vulkan の内部ヘッダは使わない。`src/` と `include/scrapgpu` には ggml の include を入れない（CONTRIBUTING の規則のまま）。
- HIP（gfx1034 を gfx1030 として扱う `HSA_OVERRIDE_GFX_VERSION=10.3.0`）も、公式の ggml-hip ビルドを `pdev_ggml` で包むだけで使える。ROCm が TB250（Gen2 x4、PCIe atomics の有無も【不明】）と Debian 13 で動くかは**【不明】**。優先度は低い。

### 6.3 CUDA（Driver API）経路

- **置き場**: `src/backends/cuda/`（ScrapGpu backend として `scrapgpu_backend_vtable` を実装し、`scrapgpu_cuda_register()` で登録する）。ggml には依存しない。
- **実行時**: `dlopen("libcuda.so.1")` で Driver API を引く。ツールキットは実行時に要らない。390 の `libcuda` が TB250 に入っているかは**【不明】**（Debian ではパッケージが分かれている）。
- **ビルド時**: `.cu` を chroot / コンテナの古い nvcc でオフラインにコンパイルする。
  - Fermi は sm_20 / sm_21（CUDA 8 が最後）。
  - Kepler は sm_35（CUDA 8 でも 9.1 でも可）。CUDA 8 だけで両方の cubin を作れるので、ツールキットは 1 つで足りる可能性がある。
  - PTX を同梱すれば、ドライバの JIT でも動く（390 は CUDA 9.1 の PTX ISA まで）。
  - 生成物（cubin / PTX）は、生成スクリプトと一緒に置く。リポジトリに入れるか、ビルド時に作るかは段階 5 で決める。
- **最小のカーネル集合**（Q4_0 を先に。ほかの型はあと）:
  1. `moe_gate_up_swiglu_q4_0_mv`（decode。ids と重みは小さなバッファから読む。Fermi は `__shfl` が無いので shared memory でリダクションする。Kepler は `__shfl`）
  2. `moe_down_wsum_q4_0_mv`（decode。スロットを固定順で重み付きに合計）
  3. `moe_gather` / `moe_scatter_wsum`（prefill。組の表はホストで作る）
  4. `mm_q4_0_f32`（prefill のエキスパートごとの mat-mat。タイル。Fermi はグリッドの次元が 65535 まで）
  5. 補助: `fill_zero`、`add_f32`
- **ホスト側**: `cuCtxCreate`、`cuModuleLoadData`、`cuMemAlloc`（アリーナ）、`cuMemHostAlloc`（ピン留め）、`cuMemcpyHtoDAsync` / `DtoHAsync`、`cuLaunchKernel`、`cuEventRecord` / `cuEventSynchronize`。
- **期待**: 小さな転送と同期が 10〜20 µs 級になる見込み【推定】。3.3 節の RT が 0.1 ms を切れば、C3（全層 EP）の候補に戻れる。
- **Fermi と Kepler を 1 プロセスで使えるか**: OpenCL では使えない【実測】が、CUDA でも同じかは**【不明】**。MB0 で最初に確かめる。使えれば GT 430 の RPC が要らなくなる。

---

## 7. 実装の段階分け

各段階の完了条件の「正しさ」は合否の基準、「計測」は記録（CONTRIBUTING の規則）。

### 段階 0: 基準とマイクロベンチ（コードは計測用だけ）

- 測る: C0（tg・pp。pp は `llama-bench -p 512 -ub 512` と `-ub 128`）、C1（`-sm layer -ts` を何通りか）、C1'。KLD は CPU を基準にした値と、C0 を基準にした値の両方（8.1 節）。
- 確認: GGUF のハイパーパラメータ（dense の先頭の層数、attention 層の数、出力 head の型）。
- マイクロベンチ MB0〜MB5（9 章）。
- 計測の枠組みは、作業ツリーにある `docs/benchmarks/moe-ep-20261021/baseline/scripts/`（`env.sh` の `use_env`、`run.sh` の bench lock と 85 ℃の待ち）をそのまま使う。
- **完了条件**: C0 / C1 の pp・tg・KLD を表に記録し、MB0〜MB5 の結果から 3.3・3.4 節の RT と T_k を実測値に差し替える。go / no-go の判定式（9 章の最後）を評価する。

### 段階 1: 方式 E の最小版（`pdev_cl11` だけ、ES と EP、融合なし）

- 複合デバイスの reg / device / buft / buffer（ホストの影、`alloc_buffer_n`、分割）、パターン照合、5.1 / 5.2 節の実行（CL11 の既存カーネル：生成経路 + 5.3 節のホスト ids の prompt 経路）、ホストでの合計、フォールバック。
- **正しさ**:
  - (a) `test-backend-ops -b SGLX0 -o MUL_MAT_ID`。デバイス 1 枚と複数で、全ケースが通る。
  - (b) 新しいツール `moe_composite_check`。`mid_check` の MoE 6 層の連鎖を、複合デバイス（1 枚 / 2 枚 / 3 枚、ES / EP）と CPU で比べる。NMSE は `mid_check` と同じ許容値。
  - (c) 複合デバイス 1 枚と N 枚の logits を、モデル全体で比べる（自己整合。平均 KLD < 0.001、top-1 ≥ 99% を目安とする）。
  - (d) 8.1 節のモデル単位の判定。
- **計測**: C2（WX は CL11 rusticl か Clover）の pp・tg。RT の内訳（sched のコピー、投入、待ち、合計）を `GGML_SGLX_PROFILE=1` で出す。

### 段階 2: `pdev_ggml`（WX 2100 を公式 Vulkan で）と `pdev_scrapgpu`（GT 430 を remote で）

- **正しさ**: 段階 1 の (a)〜(d) を、pdev の組み合わせごとに。
- **計測**: C2 の WX を、CL11 から Vulkan に替えた場合。GT 430 を足した場合。方式 F に進むかは、境界のコスト（sched の get/set）が 1 層の時間の何割かで判断する。

### 段階 3: ③ の融合カーネルと ④ のプールの完成

- **正しさ**: `moe_composite_check` に融合カーネルの場合を足す（3 種類の調整値：既定 / `FORCE_GEN=kepler` / `fermi`）。`mvm_check` / `mid_check` / `test-backend-ops` が回帰しないこと。
- **計測**: トークンあたりの起動回数（物理デバイスごと）、C2 の tg。

### 段階 4: ② 統計と配置

- **正しさ**: 配置を変えても、出力が段階 3 と一致すること（同じ pdev 構成で KLD < 0.001）。
- **計測**: 統計の偏り、配置の前後での tg・pp。

### 段階 5: CUDA の pdev（Kepler → Fermi）

- **正しさ**: CUDA backend の `tests/ops`（C API）。MUL_MAT_ID と補助の op の NMSE。`moe_composite_check` を CUDA の pdev で。
- **計測**: MB1（RT）を CUDA で。C2 と C3 の tg。

### 段階 6: ⑤ と 30B

- **正しさ**: `buffer_from_host_ptr` の経路で、段階 1 の (c)(d)。30B の KLD を、30B の単体最速の構成を基準にして。
- **計測**: 30B の単体最速（先に測る）、命中率、tg・pp。

### 段階 7（条件付き）: 方式 F

- 段階 2 で、境界のコストが支配的だと分かった場合に行う。

### Sonnet サブエージェントへの委譲の単位（ファイルが重ならないように）

| 単位 | 担当するファイル（新規または変更） | 依存 | 完了条件 |
| --- | --- | --- | --- |
| U1 マイクロベンチ | `adapters/ggml/tools/moe_rt_bench.cpp`（新規。ヘッダのコメントにビルド方法） | なし | MB0〜MB5 が Mac で動く（Mac の OpenCL で OK と出る）。TB250 のコマンドを書き出せる |
| U2 複合デバイスの本体 | `adapters/ggml/composite/composite.h`、`composite_device.cpp`（reg への追加、device、buft、buffer、ホストの影、分割）、`composite_graph.cpp`（照合、plan のキャッシュ、実行、フォールバック）。`ggml-cl11.cpp` の reg の初期化に数行（フックだけ）。`adapters/ggml/CMakeLists.txt` | U3 の `pdev.h` | `test-backend-ops -b SGLX0 -o MUL_MAT_ID` が、Mac の pdev_cl11 1 枚で全件通る |
| U3 pdev の IF と CL11 実装 | `adapters/ggml/composite/pdev.h`、`pdev_cl11.cpp`、`src/backends/opencl11/ops_moe.cpp`（ホスト ids の入口）、`ops.h` | なし（`pdev.h` を最初に固める） | `pdev.h` を確定したうえで、`pdev_cl11` の単体テスト（tools の小さなドライバ）が CPU 参照と一致する |
| U4 `pdev_ggml` | `adapters/ggml/composite/pdev_ggml.cpp` | U3 の `pdev.h` | Mac の ggml-cpu（と、あれば Metal）を包んで、U3 と同じテストが通る |
| U5 `pdev_scrapgpu` | `adapters/ggml/composite/pdev_scrapgpu.cpp` | U3 の `pdev.h` | `SCRAPGPU_TEST_BACKEND=remote` 相当で、MoE の job が通る。**出力射影（MUL_MAT）の行分割では実装済み**（`REMOTE<i>`、GT 430、`docs/benchmarks/moe-ep-20261022/gt430/README.md`）。MoE ブロックの job はまだ |
| U6 融合カーネル | `src/backends/opencl11/kernels/moe_fused.cl`、`src/backends/opencl11/ops_moe_fused.cpp`、`opencl11_kernels.cmake`（1 行）、`kernels.h` / `kernels.cpp`（カーネルの番号） | U3 | 3 種類の調整値で、NMSE が既存の生成経路と一致する。build-option identity の差分を記録する |
| U7 CUDA backend | `src/backends/cuda/*`、`include/scrapgpu/cuda.h`、ルートの `CMakeLists.txt` のオプション `SCRAPGPU_BUILD_CUDA`、`tests/` の CUDA の場合 | なし | 実機が無い環境ではビルドと `dlopen` の失敗の扱いまで。カーネルは TB250 で検証する（ユーザーの実行） |
| U8 正しさのツールと計測スクリプト | `adapters/ggml/tools/moe_composite_check.cpp`、`adapters/ggml/tools/composite-bench/*.sh` | U2・U3 | Mac で `all OK`。TB250 用のスクリプトは `DRY_RUN=1` でコマンドを出せる |
| U9 統計と配置 | `adapters/ggml/composite/placement.cpp` / `.h` | U2 | 統計の CSV から配置を出す単体テスト |
| U10 文書 | `docs/MULTI_GPU.md`（節の追加）、`docs/CL11.md`（環境変数）、この文書の更新 | 各段階 | 実測値で【推定】を置き換える |

- 共有ファイル（`adapters/ggml/CMakeLists.txt`、`ggml-cl11.cpp` の reg の初期化）は U2 だけが触る。ほかの単位は、追加するソースの一覧を U2 に渡す。
- 並列にできるのは、`pdev.h` を確定したあとの U1・U3・U7（独立）、U4・U5・U6（U3 のあと）。
- 実装の委譲先はユーザーのメモリの方針（Sonnet）。TB250 での実測は SSH を伴うので、ユーザーの許可を得てから行う。

---

## 8. 正しさの基準とリスク

### 8.1 合格条件（2026-10-21 決定）

- 比較相手: tb250 で組める既存構成**すべて**（単体の各構成、公式 Vulkan の AMD 2 枚層分割 C1、5 枚層分割、CL11 + Vulkan 混在など）。pp512・tg128・`-pg 512,128` の 3 指標それぞれで、そのすべてを上回る。
- 正しさ（案 B）: CPU の logits を基準に、平均 KLD ≤ 単体最速構成（C0）の値 × 1.2、top-1 一致率 ≥ C0 − 1 ポイント。加えて複合デバイス 1 枚と N 枚の自己整合で平均 KLD < 0.001。判定は複数チャンクの標準誤差を見て行う。
- 当初の「top-1 ≥ 96%」は Qwen3-8B の値で、LFM2.5-8B-A1B では CPU 同士でも 93.6% のため採らない。

### 8.2 リスク

| リスク | 影響 | 対処 |
| --- | --- | --- |
| RT（ホスト経由の往復）が 0.5 ms を超える | C2 の decode が C1 に勝てない | 段階 0 の MB1 / MB4 で先に判定する。router をホストで計算、busy-poll、CUDA |
| C1（コード不要）が十分速い | 複合デバイスの価値が prefill と 30B だけになる | 段階 0 の go / no-go で範囲を決め直す |
| 方式 E の名前に頼る照合が、llama.cpp の更新で壊れる | 速くなくなる（フォールバックで正しさは保つ） | 照合の失敗を警告として出す。llama.cpp の版を固定して検証する |
| sched が、複合デバイスに MoE 以外の op を割り当てる | 境界が増える | `supports_op` を狭くする。`GGML_SCHED_DEBUG` で split を確認する |
| ドライバが実際には並列に動かない（NVIDIA 390 OpenCL と rusticl を 1 プロセスで同時に） | max ではなく和になる | MB2 で確かめる |
| 2 コアの Celeron がホスト側の律速になる | 投入とソートと合計の時間が見える | ワーカースレッドを最小に、busy-wait をしない。ホストの処理時間を計測で分ける |
| ggml-vulkan の backend を 2 インスタンス、同じデバイスで使う | 不具合 | C2 では RX を複合デバイスに入れない |
| ルーティングの微小な差（FP32 と Q8_1 活性、合計の順序）で近い同点が反転する | KLD が雑音の床（0.0185）付近で揺れる | 判定は複数のチャンクで標準誤差を見る（CL11.md の方法） |
| GT 710 のように遅いデバイスが max を決める | 並列化の利得が消える | ES の比を帯域に比例させる。0 にして外せるようにする |
| CUDA: 390 の `libcuda` が無い、CUDA 8 / 9.1 のツールキットが Debian 13 で動かない | 段階 5 ができない | chroot / コンテナ（CONTRIBUTING で許可済み）。CUDA が無くても段階 1〜4 は成立する |
| 30B の命中率が低い | 30B の tg が伸びない | 段階 6 の初めに単体最速と命中率を測ってから進む |
| VRAM の見積もり（RX に 16 層ぶん）が外れる | 遠隔層が増え、RT の回数が増える | `-c` と `-ub` を絞る。`get_memory` を実際の確保に合わせる |

---

## 9. 最初に測る「勝ち筋の判定」マイクロベンチ（U1、`moe_rt_bench`）

すべて TB250 で、bench lock の下、85 ℃未満で測る。中央値と p99 を、1 万回（MB3 以降は 1 千回）で取る。

| ID | 内容 | 判定に使う量 |
| --- | --- | --- |
| MB0 | CUDA Driver API（390 の libcuda）で、GT 430 と GT 730 / GT 710 の両方に、1 プロセスで `cuCtxCreate` してカーネルを起動できるか | GT 430 を同じプロセスで使えるか |
| MB1 | 各デバイスで「8 KiB の書き込み → 空に近いカーネル 1 個 → 8 KiB の読み出し」の往復。変種は、blocking と、非 blocking + `clFinish` と、イベントの busy-poll と、ピン留めのホストメモリ | RT_j（3.3 節） |
| MB2 | MB1 を 4 デバイス（GT 730・GT 710・WX 2100・RX 6400、1 プロセス）に「全部投入 → 順に待つ」で同時に行う。ホスト 1 スレッドと、デバイスごとのスレッドで比べる | 並列が成り立つか（壁時計が max ≈ 和にならない） |
| MB3 | 1 層ぶんのエキスパートの計算（2048×1792 の Q4_0、gate+up+down、k=4）。(i) 各デバイス 1 枚で、(ii) ES で {WX, GT 730, GT 710} に同時に（比は帯域に比例）、(iii) 同じことを ggml-vulkan の RX 6400 1 枚で、(iv) CPU（ggml-cpu と SGL の cpu）で | 遠隔層 1 層の時間 `max_j(T_k_j + bytes_j/B_j)` と、RX が自分で計算する時間との比 |
| MB4 | ggml-vulkan の RX 6400 で、8 KiB の `get_tensor` / `set_tensor`（sched の境界の 1 回）と、1 ノードの `graph_compute` + `synchronize` | 方式 E の RT_in / RT_out |
| MB5 | MB3 の prefill 版（128 / 512 トークン、ES と EP、転送込み） | 3.4 節の xfer と FLOP の実効値 |

**go / no-go の判定式**（遠隔層の数 L_r ≈ 6 で）:

```
decode の利得 ≈ L_r × [ T_WX単独(1層) − ( MB4往復 + MB3(ii) ) ]
  > 0         → C2（段階 1〜3）へ進む
  ≤ 0         → decode の並列化はやめ、C1 + 起動削減 + prefill の EP（MB5 で利得があれば）に絞る
全層 EP（C3）の候補: 22 × (MB4往復 + MB3(ii)の全層版) < C2 の時間 のときだけ（CUDA の MB1 次第）
```

最初の 1 時間で決まるのは、MB1・MB2・MB3 (ii) と、C1 の tg。この 3 つで、この設計のどの枝に進むかがほぼ決まる。

---

## 10. 未決事項・不明点の一覧

- C0 の pp（未測定）、C1 の pp / tg（未測定）。
- GGUF の構成（dense の層数、attention 層の数、出力 head の型）。
- Vulkan の RX 6400 の境界 1 回のコスト（MB4）と、起動の固定費。
- NVIDIA 390 の OpenCL と rusticl が、1 プロセスで本当に並列に動くか（MB2）。
- CUDA: TB250 に 390 の `libcuda` があるか、Fermi と Kepler を 1 プロセスで使えるか（MB0）。
- LFM2.5 のルーティングの偏り（② の効果）。
- llama.cpp 23b0202 の `ggml-backend-impl.h` が、この文書で読んだ版（API 3、ggml 0.25.3 の作業ツリー）と同じか。複合デバイスの iface は、ビルド時に `GGML_BACKEND_API_VERSION` の分岐で合わせる（既存の `ggml-cl11.cpp:545-552` と同じ方式）。
- 合格条件の top-1 の基準（8.1 節）。
- 「総合」の定義（例: `llama-bench -p 512 -n 128` の合計時間、または決まった prompt での end-to-end）。段階 0 で決める。
