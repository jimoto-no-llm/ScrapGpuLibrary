// hec_trace: MoE routing trace for the hot-expert-cache design (docs/plans/HOT_EXPERT_CACHE.md, unit H1 / U-trace).
//
// Records, for every token and every layer, the 8 selected expert ids (the `ffn_moe_topk-<il>` tensor, read by the scheduler eval callback
// `cb_eval`). llama.cpp is not modified; only the public llama.h / ggml-backend.h API is used.
//
// Output (PREFIX.bin + PREFIX.json):
//   PREFIX.bin   raw uint8 array [n_tokens][n_layer][n_used]  (expert id < 256; 255 = layer without a MoE router)
//   PREFIX.json  {"n_layer":..,"n_expert":..,"n_used":..,"model":..,"tokens":[..], "segments":[{"name","phase","start","n"}..], "ubatches":[{"start","n","phase"}..]}
//                phase = "prefill" | "decode"; one segment per measured run (llama-bench rep or natural-text prompt/generation)
//
// Modes
//   --bench "512,128,512:128" [--reps N]   the three llama-bench instances `-p 512 -n 128 -pg 512,128` in the order llama-bench runs them, with
//                                          the same token stream: std::rand() % n_vocab with the default seed (no srand), in the same order of
//                                          draws as llama-bench's test_prompt / test_gen including the warmup runs (the warmup draws are consumed
//                                          but the warmup is not computed, so rep r here has exactly the tokens of rep r in llama-bench).
//   --text FILE --n-prompt N --n-gen M     natural text: tokenize FILE, prefill the first N tokens (ubatch -ub), then sample M tokens from the
//                                          model itself (top-k 40, temp 0.8, fixed --seed) and decode them one by one (the generation).
//
// Build (tb250):  g++ -O2 -std=c++17 hec_trace.cpp -I$L/include -I$L/ggml/include -L$B -lllama -lggml -lggml-base -Wl,-rpath,$B -o hec_trace
//   L = ~/moe-ep/src/llama.cpp, B = ~/moe-ep/src/llama-build/bin  (the shared build, read only)
// Run:    GGML_BACKEND_PATH is not used; --cpu-lib PATH loads only that ggml-cpu .so and disables every other device (CPU-only trace).

#include "llama.h"
#include "ggml-backend.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <algorithm>

struct segment { std::string name, phase; size_t start, n; };
struct ubatch  { size_t start, n; std::string phase; };

struct tracer {
    int n_layer = 0, n_used = 8, n_expert = 128;
    std::vector<uint8_t>  data;          // [token][layer][n_used]
    std::vector<int32_t>  tokens;        // token ids in order
    std::vector<segment>  segs;
    std::vector<ubatch>   ubs;
    std::string           phase = "prefill";
    bool                  recording = true;
    // callback state
    int    last_il = -1;
    size_t blk_base = 0;                 // token index of the first token of the current ubatch
    size_t blk_n = 0;
    std::vector<int32_t> tmp;
    size_t ntok() const { return data.size() / (size_t) (n_layer * n_used); }
};

static bool cb_eval(struct ggml_tensor * t, bool ask, void * ud) {
    tracer * tr = (tracer *) ud;
    const char * name = ggml_get_name(t);
    if (strncmp(name, "ffn_moe_topk-", 13) != 0) return false;   // not interested: do not split the graph here
    if (ask) return true;
    if (!tr->recording) return true;
    const int il = atoi(name + 13);
    const int k = (int) t->ne[0], n = (int) t->ne[1];
    if (t->type != GGML_TYPE_I32 || k != tr->n_used || il < 0 || il >= tr->n_layer) {
        fprintf(stderr, "hec_trace: unexpected %s type=%d ne=%d,%d\n", name, (int) t->type, k, n);
        return true;
    }
    tr->tmp.resize((size_t) k * n);
    // the tensor may be a non-contiguous view: read row by row through nb[1]
    if (ggml_is_contiguous(t)) {
        ggml_backend_tensor_get(t, tr->tmp.data(), 0, sizeof(int32_t) * (size_t) k * n);
    } else {
        for (int j = 0; j < n; j++) ggml_backend_tensor_get(t, tr->tmp.data() + (size_t) j * k, (size_t) j * t->nb[1], sizeof(int32_t) * k);
    }
    if (il <= tr->last_il || tr->blk_n == 0) {   // first layer of a new ubatch
        tr->blk_base = tr->ntok();
        tr->blk_n = n;
        tr->data.resize(tr->data.size() + (size_t) n * tr->n_layer * tr->n_used, 255);
        tr->ubs.push_back({tr->blk_base, (size_t) n, tr->phase});
    }
    tr->last_il = il;
    // llama.cpp prunes the last layer's FFN input to the rows that need logits (inp_out_ids): n < blk_n there. Those rows are the LAST n tokens
    // of the ubatch (llama_batch_get_one requests logits for the last token only); the other tokens keep 255 (= not observed) in that layer.
    const size_t row0 = tr->blk_n - (size_t) n;
    for (int j = 0; j < n; j++) {
        uint8_t * dst = &tr->data[((tr->blk_base + row0 + j) * tr->n_layer + il) * tr->n_used];
        for (int e = 0; e < k; e++) dst[e] = (uint8_t) tr->tmp[(size_t) j * k + e];
    }
    return true;
}

static void begin_seg(tracer & tr, const std::string & name, const std::string & phase) {
    tr.phase = phase; tr.last_il = -1; tr.blk_n = 0;
    tr.segs.push_back({name, phase, tr.ntok(), 0});
}
static void end_seg(tracer & tr) { tr.segs.back().n = tr.ntok() - tr.segs.back().start; }

static bool decode(llama_context * ctx, tracer & tr, std::vector<llama_token> & toks, int off, int n) {
    llama_batch b = llama_batch_get_one(toks.data() + off, n);
    int rc = llama_decode(ctx, b);
    if (rc != 0) { fprintf(stderr, "llama_decode rc=%d\n", rc); return false; }
    if (tr.recording) for (int i = 0; i < n; i++) tr.tokens.push_back(toks[off + i]);
    return true;
}

static std::vector<std::string> split(const std::string & s, char c) {
    std::vector<std::string> r; std::stringstream ss(s); std::string x;
    while (std::getline(ss, x, c)) r.push_back(x);
    return r;
}

// same draw order as llama-bench test_prompt / test_gen. The first token of a prompt/gen is bos if the vocab adds one, else a draw.
static void draw_prompt(const llama_vocab * vocab, int n_vocab, int n_prompt, int n_batch, std::vector<std::vector<llama_token>> & chunks) {
    int done = 0;
    while (done < n_prompt) {
        int n = std::min(n_prompt - done, n_batch);
        std::vector<llama_token> tk(n);
        tk[0] = done == 0 && llama_vocab_get_add_bos(vocab) ? llama_vocab_bos(vocab) : std::rand() % n_vocab;
        for (int i = 1; i < n; i++) tk[i] = std::rand() % n_vocab;
        chunks.push_back(tk);
        done += n;
    }
}

static void quiet_log(enum ggml_log_level level, const char * text, void *) { if (level == GGML_LOG_LEVEL_ERROR) fputs(text, stderr); }

int main(int argc, char ** argv) {
    std::string model, out = "trace", bench, text, cpu_lib;
    int n_prompt = 512, n_gen = 128, reps = 1, n_ub = 512, n_thr = 2, ngl = 0, seed = 1234;
    bool cpu_only = false;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto nx = [&]() { if (i + 1 >= argc) { fprintf(stderr, "missing value for %s\n", a.c_str()); exit(2); } return std::string(argv[++i]); };
        if (a == "-m") model = nx();
        else if (a == "--out") out = nx();
        else if (a == "--bench") bench = nx();
        else if (a == "--reps") reps = atoi(nx().c_str());
        else if (a == "--text") text = nx();
        else if (a == "--n-prompt") n_prompt = atoi(nx().c_str());
        else if (a == "--n-gen") n_gen = atoi(nx().c_str());
        else if (a == "-ub") n_ub = atoi(nx().c_str());
        else if (a == "-t") n_thr = atoi(nx().c_str());
        else if (a == "-ngl") ngl = atoi(nx().c_str());
        else if (a == "--seed") seed = atoi(nx().c_str());
        else if (a == "--cpu-lib") { cpu_lib = nx(); cpu_only = true; }
        else { fprintf(stderr, "unknown arg %s\n", a.c_str()); return 2; }
    }
    if (model.empty() || (bench.empty() && text.empty())) { fprintf(stderr, "usage: hec_trace -m MODEL --out PREFIX (--bench | --text FILE) ...\n"); return 2; }

    if (cpu_only) ggml_backend_load(cpu_lib.c_str()); else ggml_backend_load_all();
    llama_log_set(quiet_log, nullptr);
    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = ngl;
    ggml_backend_dev_t none[1] = { nullptr };
    if (cpu_only) mp.devices = none;
    llama_model * lm = llama_model_load_from_file(model.c_str(), mp);
    if (!lm) { fprintf(stderr, "load failed\n"); return 1; }
    const llama_vocab * vocab = llama_model_get_vocab(lm);
    const int n_vocab = llama_vocab_n_tokens(vocab);

    tracer tr;
    tr.n_layer = llama_model_n_layer(lm);
    char buf[64];
    if (llama_model_meta_val_str(lm, "qwen3moe.expert_count", buf, sizeof(buf)) > 0) tr.n_expert = atoi(buf);
    if (llama_model_meta_val_str(lm, "qwen3moe.expert_used_count", buf, sizeof(buf)) > 0) tr.n_used = atoi(buf);

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 2048; cp.n_batch = n_ub; cp.n_ubatch = n_ub; cp.n_threads = n_thr; cp.n_threads_batch = n_thr;
    cp.cb_eval = cb_eval; cp.cb_eval_user_data = &tr;
    llama_context * ctx = llama_init_from_model(lm, cp);
    if (!ctx) { fprintf(stderr, "ctx failed\n"); return 1; }

    if (!bench.empty()) {
        // instances in llama-bench order: pp<P>, tg<G>, pg<P>,<G>. bench = "P,G,PP:GG"  e.g. "512,128,512:128"
        auto parts = split(bench, ',');
        int P = atoi(parts[0].c_str()), G = atoi(parts[1].c_str());
        int PP = 0, PG = 0;
        if (parts.size() > 2) { auto pg = split(parts[2], ':'); PP = atoi(pg[0].c_str()); PG = atoi(pg[1].c_str()); }
        std::srand(1);   // glibc default; llama-bench never calls srand
        auto reset = [&]() { llama_memory_clear(llama_get_memory(ctx), false); };
        auto run_prompt = [&](int np, bool record, const std::string & nm, bool is_pg) {
            std::vector<std::vector<llama_token>> ch;
            draw_prompt(vocab, n_vocab, np, n_ub, ch);
            if (!record) return;
            tr.recording = true;
            begin_seg(tr, nm, "prefill");
            for (auto & c : ch) if (!decode(ctx, tr, c, 0, (int) c.size())) exit(1);
            end_seg(tr);
            (void) is_pg;
        };
        auto run_gen = [&](int ng, bool record, const std::string & nm) {
            llama_token tok = llama_vocab_get_add_bos(vocab) ? llama_vocab_bos(vocab) : std::rand() % n_vocab;
            if (record) { begin_seg(tr, nm, "decode"); }
            for (int i = 0; i < ng; i++) {
                if (record) { std::vector<llama_token> one{tok}; if (!decode(ctx, tr, one, 0, 1)) exit(1); }
                tok = std::rand() % n_vocab;
            }
            if (record) end_seg(tr);
        };
        // an instance = warmup (draws only), then `reps` measured runs
        if (P > 0) {
            run_prompt(P, false, "", false);                                   // warmup draws
            for (int r = 0; r < reps; r++) { reset(); run_prompt(P, true, "pp" + std::to_string(P) + "_r" + std::to_string(r), false); fprintf(stderr, "pp%d rep %d done, %zu tokens\n", P, r, tr.ntok()); }
        }
        if (G > 0) {
            run_gen(1, false, "");                                             // warmup (1 token)
            for (int r = 0; r < reps; r++) { reset(); run_gen(G, true, "tg" + std::to_string(G) + "_r" + std::to_string(r)); fprintf(stderr, "tg%d rep %d done, %zu tokens\n", G, r, tr.ntok()); }
        }
        if (PP > 0) {
            run_prompt(PP, false, "", true); run_gen(1, false, "");            // warmup draws for the pg instance (prompt then gen)
            for (int r = 0; r < reps; r++) {
                reset();
                std::vector<std::vector<llama_token>> ch;                      // measured run: prompt draws, then gen draws
                draw_prompt(vocab, n_vocab, PP, n_ub, ch);
                tr.recording = true;
                begin_seg(tr, "pg" + std::to_string(PP) + "," + std::to_string(PG) + "_r" + std::to_string(r) + "_pp", "prefill");
                for (auto & c : ch) if (!decode(ctx, tr, c, 0, (int) c.size())) exit(1);
                end_seg(tr);
                run_gen(PG, true, "pg" + std::to_string(PP) + "," + std::to_string(PG) + "_r" + std::to_string(r) + "_tg");
                fprintf(stderr, "pg rep %d done, %zu tokens\n", r, tr.ntok());
            }
        }
    } else {
        std::ifstream f(text, std::ios::binary);
        std::stringstream ss; ss << f.rdbuf();
        std::string txt = ss.str();
        std::vector<llama_token> tk(txt.size() + 16);
        int nt = llama_tokenize(vocab, txt.c_str(), (int) txt.size(), tk.data(), (int) tk.size(), llama_vocab_get_add_bos(vocab), true);
        if (nt < 0) { fprintf(stderr, "tokenize: need %d\n", -nt); return 1; }
        tk.resize(nt);
        if (nt < n_prompt) { fprintf(stderr, "text has only %d tokens (< %d)\n", nt, n_prompt); n_prompt = nt; }
        tk.resize(n_prompt);
        fprintf(stderr, "prompt: %d tokens from %s\n", n_prompt, text.c_str());
        std::string nm = text.substr(text.find_last_of('/') + 1);
        tr.recording = true;
        begin_seg(tr, nm + "_prefill", "prefill");
        for (int o = 0; o < n_prompt; o += n_ub) if (!decode(ctx, tr, tk, o, std::min(n_ub, n_prompt - o))) return 1;
        end_seg(tr);
        llama_sampler * smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
        llama_sampler_chain_add(smpl, llama_sampler_init_top_k(40));
        llama_sampler_chain_add(smpl, llama_sampler_init_temp(0.8f));
        llama_sampler_chain_add(smpl, llama_sampler_init_dist((uint32_t) seed));
        begin_seg(tr, nm + "_decode", "decode");
        std::string gen;
        for (int i = 0; i < n_gen; i++) {
            llama_token id = llama_sampler_sample(smpl, ctx, -1);
            if (llama_vocab_is_eog(vocab, id)) id = tk[std::min<size_t>(tk.size() - 1, 1)];   // keep going: EOG is replaced by an arbitrary prompt token
            char pc[256]; int pl = llama_token_to_piece(vocab, id, pc, sizeof(pc), 0, false);
            if (pl > 0) gen.append(pc, pl);
            std::vector<llama_token> one{id};
            if (!decode(ctx, tr, one, 0, 1)) return 1;
        }
        end_seg(tr);
        std::ofstream(out + ".gen.txt") << gen;
        llama_sampler_free(smpl);
    }

    { std::ofstream f(out + ".bin", std::ios::binary); f.write((const char *) tr.data.data(), tr.data.size()); }
    {
        std::ofstream j(out + ".json");
        j << "{\"model\":\"" << model << "\",\"n_layer\":" << tr.n_layer << ",\"n_expert\":" << tr.n_expert << ",\"n_used\":" << tr.n_used
          << ",\"n_tokens\":" << tr.ntok() << ",\n\"segments\":[";
        for (size_t i = 0; i < tr.segs.size(); i++) j << (i ? "," : "") << "{\"name\":\"" << tr.segs[i].name << "\",\"phase\":\"" << tr.segs[i].phase << "\",\"start\":" << tr.segs[i].start << ",\"n\":" << tr.segs[i].n << "}";
        j << "],\n\"ubatches\":[";
        for (size_t i = 0; i < tr.ubs.size(); i++) j << (i ? "," : "") << "[" << tr.ubs[i].start << "," << tr.ubs[i].n << "," << (tr.ubs[i].phase == "prefill" ? 0 : 1) << "]";
        j << "],\n\"tokens\":[";
        for (size_t i = 0; i < tr.tokens.size(); i++) j << (i ? "," : "") << tr.tokens[i];
        j << "]}\n";
    }
    fprintf(stderr, "wrote %s.bin (%zu tokens x %d layers x %d)\n", out.c_str(), tr.ntok(), tr.n_layer, tr.n_used);
    llama_free(ctx);
    llama_model_free(lm);
    return 0;
}
