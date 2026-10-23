// mk_tiny_moe: writes a small random qwen3moe GGUF (no tokenizer: "tokenizer.ggml.model" = "none"), to test the MoE path of SGLX0 through llama.cpp on a machine
// without the real model (docs/plans/HOT_EXPERT_CACHE.md). The experts are Q4_0 (the down projection of layer 0 is Q4_1, like the unsloth Qwen3-30B-A3B file), the rest
// F16 / F32. Shapes follow the 30B: gate/up [n_embd, n_ff], down [n_ff, n_embd], softmax router, 8 of E experts.
//
//   c++ -std=c++17 -O1 -I<llama.cpp>/ggml/include mk_tiny_moe.cpp -L<llama.cpp>/build-dl/bin -lggml -lggml-base -Wl,-rpath,<llama.cpp>/build-dl/bin -o mk_tiny_moe
//   ./mk_tiny_moe out.gguf [--layers 6] [--embd 256] [--experts 64] [--used 8] [--ff 128] [--vocab 512] [--seed 1]
#include "ggml.h"
#include "gguf.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

int main(int argc, char ** argv) {
    if (argc < 2) { fprintf(stderr, "usage: mk_tiny_moe out.gguf [--layers N] [--embd N] [--experts N] [--used N] [--ff N] [--vocab N] [--seed N]\n"); return 2; }
    std::string out = argv[1];
    int n_layer = 6, n_embd = 256, n_expert = 64, n_used = 8, n_ff = 128, n_vocab = 512, seed = 1;
    for (int i = 2; i + 1 < argc; i += 2) {
        std::string a = argv[i];
        int v = atoi(argv[i + 1]);
        if (a == "--layers") n_layer = v; else if (a == "--embd") n_embd = v; else if (a == "--experts") n_expert = v; else if (a == "--used") n_used = v;
        else if (a == "--ff") n_ff = v; else if (a == "--vocab") n_vocab = v; else if (a == "--seed") seed = v; else { fprintf(stderr, "unknown %s\n", a.c_str()); return 2; }
    }
    const int n_head = 4, n_head_kv = 2, head = 64;
    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.0f, 1.0f);

    size_t mem = 64u << 20;
    mem += (size_t) n_vocab * n_embd * 8 + (size_t) n_layer * ((size_t) n_expert * n_ff * n_embd * 3 * 2 + (size_t) n_embd * n_embd * 4 * 4 + (size_t) n_embd * n_expert * 4);
    ggml_init_params ip = { mem, nullptr, false };
    ggml_context * ctx = ggml_init(ip);
    gguf_context * g = gguf_init_empty();

    auto kv_u32 = [&](const char * k, uint32_t v) { gguf_set_val_u32(g, k, v); };
    gguf_set_val_str(g, "general.architecture", "qwen3moe");
    gguf_set_val_str(g, "general.name", "tiny-qwen3moe");
    kv_u32("qwen3moe.context_length", 2048);
    kv_u32("qwen3moe.embedding_length", n_embd);
    kv_u32("qwen3moe.block_count", n_layer);
    kv_u32("qwen3moe.feed_forward_length", n_ff * n_used);
    kv_u32("qwen3moe.attention.head_count", n_head);
    kv_u32("qwen3moe.attention.head_count_kv", n_head_kv);
    kv_u32("qwen3moe.attention.key_length", head);
    kv_u32("qwen3moe.attention.value_length", head);
    gguf_set_val_f32(g, "qwen3moe.attention.layer_norm_rms_epsilon", 1e-6f);
    gguf_set_val_f32(g, "qwen3moe.rope.freq_base", 10000.0f);
    kv_u32("qwen3moe.expert_count", n_expert);
    kv_u32("qwen3moe.expert_used_count", n_used);
    kv_u32("qwen3moe.expert_feed_forward_length", n_ff);
    kv_u32("qwen3moe.vocab_size", n_vocab);
    gguf_set_val_str(g, "tokenizer.ggml.model", "none");

    auto add = [&](ggml_tensor * t, const char * name, float scale, bool ones) {
        ggml_set_name(t, name);
        const int64_t n = ggml_nelements(t);
        std::vector<float> f((size_t) n);
        for (auto & v : f) v = ones ? 1.0f : nd(rng) * scale;
        if (t->type == GGML_TYPE_F32) {
            memcpy(t->data, f.data(), (size_t) n * 4);
        } else if (t->type == GGML_TYPE_F16) {
            ggml_fp32_to_fp16_row(f.data(), (ggml_fp16_t *) t->data, n);
        } else {
            const int64_t rows = n / t->ne[0];
            ggml_quantize_chunk(t->type, f.data(), t->data, 0, rows, t->ne[0], nullptr);
        }
        gguf_add_tensor(g, t);
    };
    add(ggml_new_tensor_2d(ctx, GGML_TYPE_F16, n_embd, n_vocab), "token_embd.weight", 0.5f, false);
    add(ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd), "output_norm.weight", 0, true);
    add(ggml_new_tensor_2d(ctx, GGML_TYPE_F16, n_embd, n_vocab), "output.weight", 0.5f, false);
    char nm[96];
    const float ws = 1.0f / sqrtf((float) n_embd);
    for (int l = 0; l < n_layer; ++l) {
        snprintf(nm, sizeof(nm), "blk.%d.attn_norm.weight", l);   add(ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd), nm, 0, true);
        snprintf(nm, sizeof(nm), "blk.%d.attn_q.weight", l);      add(ggml_new_tensor_2d(ctx, GGML_TYPE_F16, n_embd, n_head * head), nm, ws, false);
        snprintf(nm, sizeof(nm), "blk.%d.attn_k.weight", l);      add(ggml_new_tensor_2d(ctx, GGML_TYPE_F16, n_embd, n_head_kv * head), nm, ws, false);
        snprintf(nm, sizeof(nm), "blk.%d.attn_v.weight", l);      add(ggml_new_tensor_2d(ctx, GGML_TYPE_F16, n_embd, n_head_kv * head), nm, ws, false);
        snprintf(nm, sizeof(nm), "blk.%d.attn_output.weight", l); add(ggml_new_tensor_2d(ctx, GGML_TYPE_F16, n_head * head, n_embd), nm, ws, false);
        snprintf(nm, sizeof(nm), "blk.%d.attn_q_norm.weight", l); add(ggml_new_tensor_1d(ctx, GGML_TYPE_F32, head), nm, 0, true);
        snprintf(nm, sizeof(nm), "blk.%d.attn_k_norm.weight", l); add(ggml_new_tensor_1d(ctx, GGML_TYPE_F32, head), nm, 0, true);
        snprintf(nm, sizeof(nm), "blk.%d.ffn_norm.weight", l);    add(ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd), nm, 0, true);
        snprintf(nm, sizeof(nm), "blk.%d.ffn_gate_inp.weight", l);add(ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_expert), nm, 1.5f * ws, false);
        snprintf(nm, sizeof(nm), "blk.%d.ffn_gate_exps.weight", l); add(ggml_new_tensor_3d(ctx, GGML_TYPE_Q4_0, n_embd, n_ff, n_expert), nm, ws, false);
        snprintf(nm, sizeof(nm), "blk.%d.ffn_up_exps.weight", l);   add(ggml_new_tensor_3d(ctx, GGML_TYPE_Q4_0, n_embd, n_ff, n_expert), nm, ws, false);
        snprintf(nm, sizeof(nm), "blk.%d.ffn_down_exps.weight", l); add(ggml_new_tensor_3d(ctx, l == 0 ? GGML_TYPE_Q4_1 : GGML_TYPE_Q4_0, n_ff, n_embd, n_expert), nm, 1.0f / sqrtf((float) n_ff), false);
    }
    if (!gguf_write_to_file(g, out.c_str(), false)) { fprintf(stderr, "write failed\n"); return 1; }
    fprintf(stderr, "wrote %s: %d layers, embd %d, %d experts (%d used), ff %d, vocab %d\n", out.c_str(), n_layer, n_embd, n_expert, n_used, n_ff, n_vocab);
    return 0;
}
