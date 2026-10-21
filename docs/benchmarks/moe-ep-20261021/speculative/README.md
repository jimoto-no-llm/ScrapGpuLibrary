# 投機的デコード（speculative decoding）の測定 — tb250, LFM2.5-8B-A1B Q4_0

測定日 2026-10-21。実装の変更なし（既存の llama.cpp ツールを別ビルドしただけ）。コミットなし。
表記: 数値は全て tb250 での実測。「導出」と書いたものは実測ログからの算術であり、別の実験の予測ではない。「仮説」は未検証。

## 1. 結論

- **単一ストリーム decode を速くする構成は見つからなかった。** 本体のみ 43.5〜45.5 t/s に対し、投機あり（DSpark ドラフト）の最良は 40.8 t/s（コード生成プロンプト、本体のみの 0.92 倍）。説明文・会話では 0.4〜0.65 倍。
- 理由は二つとも実測で裏付けた。
  1. 本体の検証バッチが 9 トークン以上になると MoE の `MUL_MAT_ID` が急に遅くなる（8 トークン 75 ms → 9 トークン 144〜173 ms。1 トークンは 20〜23 ms）。8 トークンでも 1 トークンの 3.3 倍かかる。
  2. ドラフト 1 回が 25〜100 ms かかる（導出、§5.3）。本体 1 トークン 22 ms を超えるので、受理が 4 トークン/ステップ前後では取り返せない。
- ドラフトの置き場は事実上 1 通りに限られる（§5.2）。公式 DSpark ドラフトは本体の `token_embd`（LM head と共有）と同じデバイスのバッファでしか動かない。GT 730 / GT 710 / GT 430 / CPU は全て起動時に abort。LFM2.5 の小型 dense モデル（230M/350M/1.2B）は語彙が違い使えない。
- 出力は本体のみの greedy と**一致しなかった**（25〜84 トークン目で分岐、§6）。

## 2. 環境

- tb250: Celeron G3930（2 コア）、RX 6400（4 GB）、WX 2100（2 GB）、GT 430 / 730 / 710。Vulkan は radv（`VK_ICD_FILENAMES=radeon_icd.json`）。
- llama.cpp 23b0202a1（build 11552 相当、`~/moe-ep/src/llama.cpp`、作業ツリーはクリーン）。別ビルド `~/moe-ep/spec/build`（`~/moe-ep/spec/build.sh`）に `llama-speculative-simple` / `llama-speculative` / `llama-server` / `llama-lookup` / `llama-completion` を作った。`GGML_VULKAN=OFF` でビルドし、`libggml-vulkan.so` と `libggml-rpc.so` は既存ビルド（同一ソース）のものへの symlink。CL11 プラグインは `GGML_BACKEND_PATH` で既存のものを読む。既存ビルドのディレクトリは書き換えていない。ビルドは `nice -n 19 -j2` で、他の測定と時間が重なった。
- ロック: 全ての GPU 測定は `flock ~/bench/gpu-bench.lock` 下、各 run の前に全カード 85℃ 未満を待つ（`run.sh` は既存のものの複製）。ロックが混んでいたため、3〜12 本を 1 回の取得でまとめて流した（`scripts/bundle.sh`、1 本ずつ直列）。終了後に自分のプロセスが残っていないことを確認した。
- 共通設定: `-ngl 99 -sm layer -fa off -t 2 -b 128 -ub 128 -c 2048 --temp 0`（既存の最良構成と同じ。`-n` は 192、§7 のみ 128）。

## 3. モデルと入手元

| ファイル | 取得元 | サイズ | sha256 |
|---|---|---:|---|
| LFM2.5-8B-A1B-Q4_0.gguf（本体、既存） | LiquidAI/LFM2.5-8B-A1B-GGUF | 4,844,678,368 | 48ed1465d761311b2fd57b7fb46cf969a20b3a8281945b04e10f52bd1609e715 |
| LFM2.5-8B-A1B-DSpark-Q8_0.gguf | https://huggingface.co/LiquidAI/LFM2.5-8B-A1B-DSpark-GGUF | 356,491,104 | 78c6edb3ec97d54fa2e6ec1ba44c394db0a1ae62fcd854d2193389c1295e0a6d |
| LFM2.5-8B-A1B-DSpark-Q4_K_M.gguf | 同上 | 199,679,840 | 017278ec4409671890f2722a690a52dba59f5f02eff604f0c1e79cf52f7c4ae8 |
| LFM2.5-8B-A1B-DSpark-F16.gguf | 同上 | 663,691,104 | a73be221783d3e832016c73945640fd81376f5de7a250025fd67a536435ce4b6 |
| LFM2.5-230M-Q4_0 / Q8_0 | LiquidAI/LFM2.5-230M-GGUF | 149,080,928 / 246,598,496 | 430fbec5…daa73 / 855be854…aa935 |
| LFM2.5-350M-Q4_0 / Q8_0 | LiquidAI/LFM2.5-350M-GGUF | 219,309,792 / 379,217,632 | 85e32858…42f50 / be036a75…a5766 |
| LFM2.5-1.2B-Instruct-Q4_0 / Q8_0 | LiquidAI/LFM2.5-1.2B-Instruct-GGUF | 695,751,488 / 1,246,253,888 | 2ea80194…9233f / f6b981dc…d26a |

全 sha256 は HF の LFS 表示と一致（完全値は `logs/sha256.txt`）。配置先は tb250 の `~/moe-ep/models/`。

**`LFM2.5-8B-A1B-DSpark-*` の正体**: Liquid AI 公式の本体専用ドラフト（DSpark 方式）。327.7M パラメータ、5 層の attention、Markov head と confidence head 付き、ブロック長 9。GGUF のアーキテクチャは `dflash`。本体の 5 層（3,7,11,15,19）の入力特徴を受けて 1 回で 9 トークンを提案する。埋め込みと LM head は本体のものを共有する（サイドカー）。llama.cpp 本流が `--spec-type draft-dspark` で対応している。**このドラフトを使う予定で取得し、実際に全ての「ドラフトあり」測定に使った。**

語彙の互換性（GGUF メタデータ、`scripts/ggufmeta.py`）:

| モデル | arch | tokenizer.model / pre | トークン数 | tokens の sha1 先頭 | BOS / EOS |
|---|---|---|---:|---|---|
| 本体 LFM2.5-8B-A1B | lfm2moe | gpt2 / lfm2 | 128000 | cf45250b8d2c | 124894 / 124900 |
| DSpark（全量子化） | dflash | gpt2 / lfm2 | 128000 | cf45250b8d2c | 124894 / 124900 |
| LFM2.5-230M / 350M / 1.2B | lfm2 | gpt2 / lfm2 | 65536 | 012d97ff7484 | 1 / 7 |

つまり小型の LFM2.5 dense は本体と語彙が別物で、ドラフトにできない（`llama-speculative-simple` は `draft model vocab type must match target model to use speculation` で終了、`logs/fail_draft350m_wx.log`）。使える互換ドラフトは DSpark のみ。

## 4. 測定方法（`llama-bench` が使えない代わり）

`llama-bench` は投機に対応しない。代わりに:

- tg: `llama-speculative-simple`（ドラフトあり）の `decoded N tokens in S seconds`、`llama-completion`（本体のみ）の `eval time … tokens per second`。どちらも prompt の評価を除いた生成フェーズ。生成 192 トークン（§7 は 128）。`llama-server` では `timings.predicted_per_second`。
- 受理率: `n_accept / n_drafted`。tokens/step = 生成数 /（生成数 − n_accept）（導出）。
- プロンプト（`prompts/`、ChatML 形式、greedy）: `explain`（ハッシュテーブルの説明）、`code`（Python の区間マージ）、`chat`（京都旅行の相談）。
- pp / tg / pg の対応値（§7）: pp = prompt 評価 t/s（本体のみは `prompt eval time`、投機は `encoded N tokens … speed`）、tg = 上記、pg = (prompt トークン + 生成トークン) /（prompt 評価秒 + 生成秒）。プロンプトは 434 トークン（`prompts/long512.txt`、512 ちょうどにはしていない）、生成 128、`-ub 128`。`-p 512` ちょうどの `llama-bench` 値とは一致しない。
- 集計: `scripts/summarize.py logs`（全 run の表は `logs/summary_all.md`）。各 run の全コマンドは `logs/<tag>.log` の先頭 `# cmd:` 行と `jobs/jobs_*.txt`。

注意（ツール差）: `llama-speculative-simple` はドラフトをほぼ出さない設定でも `llama-completion` より遅い。n-gram 実行（ドラフト 1.02〜1.1 トークン/ステップ、実質ドラフトなし）が 36〜42 t/s で、同じ構成の `llama-completion` は 44 t/s。差の原因は未調査。公平な比較には `llama-server`（本体のみ 41.7 t/s、§6）も併記する。

## 5. 結果（tg、生成 192 トークン、3 プロンプト）

### 5.1 本体のみ（`llama-completion`）

| 構成 | explain | code | chat |
|---|---:|---:|---:|
| `-dev Vulkan0,Vulkan1 -ts 5,1`（既存の最良） | 43.55 | 44.10 | 43.84 |
| `-dev Vulkan1,Vulkan0 -ts 1,2`（WX を先頭、RX が LM head） | 45.45 | 44.21 | 44.57 |
| `-dev Vulkan1,Vulkan0 -ts 1,5`（explain のみ） | 44.62 | | |

### 5.2 DSpark ドラフトの置き場

| ドラフトの置き場 | 結果 | 理由（ログ） |
|---|---|---|
| WX 2100（Vulkan1、本体 `-ts 5,1` で LM head がある側） | 動く。§5.3 | |
| RX 6400（本体を逆順 `-dev Vulkan1,Vulkan0 -ts 1,2`） | 動く。§5.3（最良） | RX の VRAM は本体 `-ts 1,5` だと満杯でドラフトが溢れ、1 ステップ約 400 ms になった（`t3`。VRAM 溢れは未確認の推定）。`-ts 1,2` で空きを作った |
| RX 6400（本体 `-ts 5,1`、LM head は WX） | abort | `pre-allocated tensor (token_embd.weight) in a buffer (Vulkan1) that cannot run the operation`（`fail_dspark_rx`, `t1`） |
| CPU（`-devd none`） | abort | 同上（`fail_dspark_cpu`） |
| GT 730（CL11、mix4） | abort | 同上（`fail_dspark_gt730`）。CL11 から Vulkan のバッファの重みを読めない |
| GT 710 | 未実施 | GT 730 と同じプラグインで同じ原因のはず（未実測） |
| GT 430（RPC） | 未実施（環境不備） | 1 プロセス内で Fermi は Kepler と共存できず RPC0 が作られなかった（`-devd RPC0: invalid device`）。仮に作れても共有 `token_embd` の制約は同じはず（未実測） |

共通の原因: DSpark サイドカーは本体の `token_embd` / LM head を共有するため、ドラフトは本体のそれがあるバッファと同じデバイスで動かす必要がある。層分割だと LM head は最後のデバイスに置かれる。GT 系や CPU にドラフトを逃がす構成は、この llama.cpp では作れない。

### 5.3 ドラフト長 n-max の振り（tg t/s / 受理率 % / tokens/step）

WX 上、DSpark Q8_0、本体 `-ts 5,1`（`dspark_wx_q8_n*`）:

| n-max | explain | code | chat | 3 本平均 tg |
|---|---|---|---|---:|
| 3 | 22.99 / 40.0 / 2.19 | 31.65 / 66.5 / 2.98 | 29.04 / 57.7 / 2.72 | 27.9 |
| 5 | 18.17 / 29.8 / 2.47 | 27.79 / 56.3 / 3.80 | 24.40 / 46.6 / 3.33 | 23.5 |
| 7 | 15.49 / 22.6 / 2.55 | 25.77 / 49.2 / 4.41 | 20.73 / 35.4 / 3.45 | 20.7 |

RX 上、DSpark Q4_K_M、本体 `-dev Vulkan1,Vulkan0 -ts 1,2`（`dspark_rx_rev12_q4km_n*`）:

| n-max | explain | code | chat | 3 本平均 tg |
|---|---|---|---|---:|
| 3 | 29.01 / 44.8 / 2.34 | 40.77 / 76.4 / 3.27 | 27.68 / 41.1 / 2.22 | 32.5 |
| 5 | 22.72 / 28.8 / 2.42 | 40.80 / 67.4 / 4.36 | 23.29 / 29.0 / 2.41 | 28.9 |
| 7 | 18.71 / 21.1 / 2.46 | 40.21 / 62.1 / 5.30 | 19.70 / 21.9 / 2.47 | 26.2 |

本体のみの 3 本平均は 43.8（`-ts 5,1`）、44.7（`-ts 1,2`）。最良の投機（RX, n 3〜5, code）は本体のみ（同順序 44.21）の 0.92 倍、3 本平均では最良（RX, n=3）でも 0.73 倍。

その他の振り（WX, explain/code。`n=5` 基準、`dspark_wx_*`）: ドラフト量子化 Q4_K_M 17.49/28.50、F16 17.09/27.67、Q8_0 18.17/27.79（差は誤差程度）。`--no-spec-draft-backend-sampling` 17.31/26.50、`-fa on` 19.74/27.40。confidence head の `--spec-draft-p-min`（n=7）: 0.3 で 18.08/26.46、0.6 で 16.36/25.88（受理率は 67% に上がるが tokens/step が 1.95 に落ちる）。いずれも改善せず。draft の重みの大きさに速度が依存しない = 重みの読み出しが律速ではない。

**ステップ時間の分解（導出、概算）**: `decoded` 秒 ÷ ステップ数（= 生成数 − 受理数）をステップ時間、本体の `prompt eval time`（検証バッチの合計 + 約 0.4 s の最初の prompt）÷ ステップ数を検証時間、残りをドラフトほかとした。

| run | ステップ時間 | 検証 | ドラフトほか |
|---|---:|---:|---:|
| WX, n=3, code | 94 ms | 59 ms | 約 42 ms |
| WX, n=7, code | 171 ms | 105 ms | 約 75 ms |
| RX, n=3, code | 80 ms | 62 ms | 約 25 ms |
| RX, n=5, code | 107 ms | 82 ms | 約 33 ms |
| RX, n=7, code | 132 ms | 100 ms | 約 43 ms |

本体のみは 22〜23 ms/トークン。検証だけで n=3 でも 59〜62 ms（4 トークンを 1 バッチ）、受理を足して tokens/step が 3 前後に届いても 1 トークンあたり約 20〜30 ms になり、本体のみに勝てない。コードで 5.3 tokens/step（RX n=7）まで出ても検証 100 ms + ドラフト 43 ms ÷ 5.3 ≒ 25 ms/トークン。

### 5.4 検証バッチ時間（`llama-bench -p N -n 0 -r 3 -ub 128`、ms/バッチ。導出 = N / pp t/s）

| N | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 12 | 16 | 32 | 64 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `-ts 5/1` | 23.4 | 31.9 | 40.2 | 46.1 | 52.8 | 60.2 | 70.4 | 76.4 | 144.3 | 158.4 | 173.8 | 192.6 | 292.7 |
| `-dev Vulkan1/Vulkan0 -ts 1/5` | 20.3 | 26.7 | | 42.0 | 50.9 | | 65.2 | 75.4 | 172.8 | | | | |

9 トークンで急に倍になる。n-max 8 以上（検証バッチ 9 トークン以上）は最初の試行（`t2`: n-max 8）で 1 ステップ約 320 ms だった。以降 n-max ≤ 7 に限った。この段差は Vulkan の mul_mat_id の実装切り替えと推測するが、ソースは読んでいない（仮説）。注: ここの N=1 は prompt 1 トークンの評価であり tg ではない。

### 5.5 n-gram 自己投機（ドラフトモデルなし、本体 `-ts 5,1`）

lookup n=4、ドラフト m≤7（ngram-mod は match 3、n-max 7）。

| 方式 | explain | code | chat |
|---|---|---|---|
| ngram-simple | 40.69 (7/7 受理) | 40.00 (23/23) | 39.51 (4/4) |
| ngram-map-k | 41.78 (4/4) | 39.52 (21/21) | 39.92 (3/3) |
| ngram-mod | 39.73 (10/10) | 36.04 (47/47) | 37.83 (7/7) |

ドラフトが出る回数が 3〜47 トークンと少なく（192 トークン中）、効果なし。基準にすべきは `llama-speculative-simple` のドラフトなし速度（約 40）で、それとほぼ同じか悪化（ngram-mod code）。

## 6. 出力一致（`llama-server`、`/completion`、greedy、`return_tokens`、192 トークン）

構成: 本体 `-dev Vulkan1,Vulkan0 -ts 1,2`、ドラフト DSpark Q4_K_M を RX、n-max 5（`scripts/servertest.sh`, `client.py`）。

| プロンプト | 本体のみ tg | 投機 tg | 受理 | 最初に分岐したトークン位置 |
|---|---:|---:|---|---:|
| explain | 41.97 | 26.68 | 122/337 | 84 |
| code | 41.40 | 40.75 | 147/213 | 25 |
| chat | 41.65 | 23.02 | 111/388 | 76 |

**3 プロンプトとも本体のみの greedy 出力と一致しなかった。** 同じ現象は `llama-speculative-simple` 対 `llama-completion` のテキストでも見えている（`t2` の出力は本体のみと 3 段落目で異なる）。Liquid AI の README は「出力は本体のみと等しい」としているが、この環境では等しくない。原因は未特定。仮説: 検証が複数トークンのバッチで走る（別カーネル、別の加算順序）ため Vulkan 上の logits が 1 トークン decode と僅差で異なり、近い同点が反転する。検証していない（本体のみを 2 回流して決定性を見る、バッチ次元を変えた本体のみ出力を比べる、等が必要）。性能の数値には影響しないが、合格条件の正しさ側で「投機は出力を変えない」とは言えない。

## 7. 合意指標に相当する値（434 トークンの prompt、生成 128、`-b 512 -ub 128`）

| 構成 | pp（t/s） | tg（t/s） | pg（t/s、導出） |
|---|---:|---:|---:|
| 本体のみ `-ts 1,2`（`llama-completion`） | 170.3 | 44.20 | 約 103（561 トークン / 5.42 s） |
| DSpark RX n=5 | 164.8 | 27.11 | 約 76（564 / 7.43 s） |
| DSpark RX n=3 | 164.2 | 30.88 | 約 82（564 / 6.85 s） |

prompt 評価はドラフトで変わらない（−3%）。pg は tg に引きずられて 0.74〜0.80 倍。prompt に対する DSpark の受理: n=5 が 85/222、n=3 が 79/153（要約の続きで 38〜52%）。最初に 512 ちょうどで流した試行は `the prompt exceeds the batch size` で失敗し（`pg_*`、`-b 128`）、`-b 512` で取り直した（`pg2_*`）。

## 8. 失敗した構成と理由（まとめ）

- 小型 LFM2.5 dense をドラフトに: 語彙不一致（§3）。
- DSpark を RX（本体 `-ts 5,1`）/ CPU / GT 730 に: 共有 `token_embd` が別デバイスのバッファ（§5.2）。
- RX 満杯の状態でドラフトを RX に: 1 ステップ約 400 ms（`t3`）。
- n-max 8: 検証が 9 トークンバッチになり約 320 ms/ステップ（`t2`）。
- `GGML_VK_PERF_LOGGER=1` で内訳を取ろうとしたが何も出力されず（`t5_perf`、ログ削除済み）。内訳は §5.3 の導出に留まる。
- `llama-completion` に `-ngld` を渡して invalid argument（スクリプトの誤り、修正済み）。

## 9. 次に試すべき案（未実施。実測の代わりではない）

1. **ドラフト 1 回の固定費の原因を切り分ける。** DSpark は量子化でも `-fa`、バックエンドサンプリングの有無でも速度が変わらず、n-max に比例して増える（WX: n=3 で 42 ms、n=7 で 75 ms）。Markov head のブロック内逐次処理か、特徴の転送（5 層 × 2048 のホスト経由）が疑わしい。Vulkan のタイムスタンプで確かめる（`GGML_VK_PERF_LOGGER` がこのビルドで出力しなかった理由も要確認）。
2. **検証 9 トークンの段差を直す。** `MUL_MAT_ID` のベクトル版が 8 トークンまでしか使われない。これを 16 まで広げるか、検証を 8 トークンに収める運用（現状）に留まる。検証だけで 1 トークン decode の 3.3 倍（8 トークン）なので、これだけでは勝てない。
3. **tokens/step が 5 を超える場面に限る。** コード生成（RX n=7 で 5.3）のように受理が高い用途では、ドラフト 25〜43 ms と検証 100 ms でほぼ並ぶ。ドラフトの固定費を半分にできれば本体のみを超える可能性がある（導出の域を出ない）。
4. **ドラフトを GT 系に出すには `token_embd` / LM head の複製が要る。** llama.cpp のサイドカー読込みを変えるか、ドラフトに自前の埋め込みを持たせる実装が必要（今回は実装変更なし）。
5. **LM head を CPU に置く**（`-ot token_embd=CPU`）と CPU ドラフトは理論上可能だが、128000×2048 の Q4_0 行列を Celeron で毎トークン読むので逆効果になりそう（未測定）。
6. 自己投機（本体の浅い層で提案）、ドラフトを複数 GPU に分けた tree speculation は未着手。
7. 本体のみの副次的な観察: `-dev Vulkan1,Vulkan0 -ts 1,2` は `-ts 5,1` より tg が 0.3〜4%（0.1〜1.9 t/s）高い（3 プロンプトで同方向）。既存の最良構成の再検討に値する。ただし各 1 回の測定で、誤差の範囲かもしれない。

## 10. ファイル

- `scripts/`: `env.sh`（既存 env.sh を読んで別ビルドへ向ける）、`run.sh`（ロック・温度待ち・ログ）、`go.sh`、`bundle.sh`、`gen_jobs1.sh`〜`gen_jobs3c.sh`（ジョブ生成）、`servertest.sh` / `client.py`（出力一致）、`summarize.py`、`ggufmeta.py`（GGUF メタデータ）、`timed.sh`。
- `jobs/`: `bundle.sh` に渡した実行コマンド全文。
- `prompts/`: 固定プロンプト 4 種。
- `logs/`: 全 run の `.log`（コマンド・温度・stderr）と `.out`、サーバ出力、`id_server_*.json`（トークン列）、`summary_all.md`、`sha256.txt`、ビルドログ。`t1`〜`t4`, `t6` は最初の探索（`t6_time.log`: CPU 使用は user 9.7 s / 実時間 29.9 s で CPU 律速ではない）。
- tb250 側の実体: `~/moe-ep/spec/`（ビルド、スクリプト、`logs/runs`）、`~/moe-ep/models/`。
