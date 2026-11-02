// eg2bench: EmbeddingGemma 2 embeddings per second and embedding dump for llama.cpp (ggml backends, CL11 plugin).
//
// Build (against a llama.cpp build with shared libs; see docs/benchmarks/eg2-20261102/README.md):
//   g++ -O2 -std=c++17 eg2bench.cpp -I<llama>/include -I<llama>/ggml/include -L<build>/bin -lllama -lggml -lggml-base \
//       -Wl,-rpath,<build>/bin -o eg2bench
//
// Modes:
//   --dump texts.txt out.bin   one text per line, sequentially, one text per llama_decode; writes N*dim float32
//                              (mean pooled, output projection, NOT yet L2-normalised: the comparer normalises)
//   --bench L B reps [--duration S [--start-at EPOCH]]  B sequences of exactly L tokens per decode (synthetic ids, fixed seed), prints embeddings/s
// Common: -m model.gguf  -ngl N  -t threads  -ub N (physical batch, default = needed)  --warm N  -c ctx
//         --ts a,b,c (tensor split)  --dev name,name (backend device names, comma list)  --sm layer|none  -v
#include "llama.h"
#include "ggml-backend.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>
#include <random>
#include <numeric>
#include <cmath>
#include <thread>

static double now_ms() {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

static void quiet_log(enum ggml_log_level lvl, const char * text, void *) {
    if (lvl >= GGML_LOG_LEVEL_ERROR || getenv("EG2_VERBOSE")) {
        fputs(text, stderr);
    }
}

int main(int argc, char ** argv) {
    std::string model_path, mode, texts_path, out_path, ts_str, dev_str, sm = "layer";
    double duration = 0, start_at = 0;
    int ngl = 99, nthreads = 2, ub = 0, warm = 2, reps = 10, L = 128, B = 1, ctx = 0, max_tokens = 0;
    bool fa = false;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto nx = [&]() { if (i + 1 >= argc) { fprintf(stderr, "missing arg for %s\n", a.c_str()); exit(2); } return std::string(argv[++i]); };
        if (a == "-m") model_path = nx();
        else if (a == "-ngl") ngl = atoi(nx().c_str());
        else if (a == "-t") nthreads = atoi(nx().c_str());
        else if (a == "-ub") ub = atoi(nx().c_str());
        else if (a == "-c") ctx = atoi(nx().c_str());
        else if (a == "--warm") warm = atoi(nx().c_str());
        else if (a == "--ts") ts_str = nx();
        else if (a == "--dev") dev_str = nx();
        else if (a == "--sm") sm = nx();
        else if (a == "--fa") fa = nx() == "on";
        else if (a == "--max-tokens") max_tokens = atoi(nx().c_str());
        else if (a == "--duration") duration = atof(nx().c_str());
        else if (a == "--start-at") start_at = atof(nx().c_str());
        else if (a == "--dump") { mode = "dump"; texts_path = nx(); out_path = nx(); }
        else if (a == "--bench") { mode = "bench"; L = atoi(nx().c_str()); B = atoi(nx().c_str()); reps = atoi(nx().c_str()); }
        else { fprintf(stderr, "unknown arg %s\n", a.c_str()); return 2; }
    }
    if (model_path.empty() || mode.empty()) { fprintf(stderr, "usage: see header\n"); return 2; }

    llama_log_set(quiet_log, nullptr);
    if (const char * bd = getenv("EG2_BACKEND_DIR")) ggml_backend_load_all_from_path(bd); else ggml_backend_load_all();
    llama_backend_init();

    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = ngl;
    mp.split_mode = sm == "none" ? LLAMA_SPLIT_MODE_NONE : LLAMA_SPLIT_MODE_LAYER;
    std::vector<ggml_backend_dev_t> devs;
    if (!dev_str.empty()) {
        size_t p = 0;
        while (p <= dev_str.size()) {
            size_t q = dev_str.find(',', p);
            if (q == std::string::npos) q = dev_str.size();
            std::string name = dev_str.substr(p, q - p);
            if (!name.empty()) {
                ggml_backend_dev_t d = ggml_backend_dev_by_name(name.c_str());
                if (!d) { fprintf(stderr, "no such device %s\n", name.c_str()); return 2; }
                devs.push_back(d);
            }
            p = q + 1;
        }
        devs.push_back(nullptr);
        mp.devices = devs.data();
    }
    std::vector<float> ts(llama_max_devices(), 0.0f);
    if (!ts_str.empty()) {
        size_t p = 0, k = 0;
        while (p < ts_str.size() && k < ts.size()) {
            size_t q = ts_str.find(',', p);
            if (q == std::string::npos) q = ts_str.size();
            ts[k++] = (float) atof(ts_str.substr(p, q - p).c_str());
            p = q + 1;
        }
        mp.tensor_split = ts.data();
    }
    double t0 = now_ms();
    llama_model * model = llama_model_load_from_file(model_path.c_str(), mp);
    if (!model) { fprintf(stderr, "load failed\n"); return 1; }
    const double t_load = now_ms() - t0;
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_out = llama_model_n_embd_out(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);

    auto make_ctx = [&](int n_tok_ub, int n_seq) {
        llama_context_params cp = llama_context_default_params();
        cp.n_ctx = ctx > 0 ? ctx : n_tok_ub;
        cp.n_batch = n_tok_ub;
        cp.n_ubatch = ub > 0 ? ub : n_tok_ub;
        if ((int) cp.n_ubatch < n_tok_ub) cp.n_ubatch = n_tok_ub; // non-causal: whole batch in one ubatch
        cp.n_seq_max = n_seq;
        cp.n_threads = nthreads;
        cp.n_threads_batch = nthreads;
        cp.embeddings = true;
        cp.flash_attn_type = fa ? LLAMA_FLASH_ATTN_TYPE_ENABLED : LLAMA_FLASH_ATTN_TYPE_DISABLED;
        cp.no_perf = true;
        return llama_init_from_model(model, cp);
    };

    if (mode == "dump") {
        std::ifstream f(texts_path);
        std::vector<std::string> texts;
        std::string line;
        while (std::getline(f, line)) texts.push_back(line);
        std::vector<std::vector<llama_token>> toks;
        int maxn = 0;
        for (auto & t : texts) {
            std::vector<llama_token> tk(t.size() + 16);
            int n = llama_tokenize(vocab, t.c_str(), (int) t.size(), tk.data(), (int) tk.size(), true, true);
            if (n < 0) { tk.resize(-n); n = llama_tokenize(vocab, t.c_str(), (int) t.size(), tk.data(), (int) tk.size(), true, true); }
            tk.resize(n);
            if (max_tokens > 0 && (int) tk.size() > max_tokens) tk.resize(max_tokens);
            maxn = std::max(maxn, (int) tk.size());
            toks.push_back(std::move(tk));
        }
        llama_context * lc = make_ctx(maxn, 1);
        if (!lc) { fprintf(stderr, "ctx failed\n"); return 1; }
        llama_batch batch = llama_batch_init(maxn, 0, 1);
        std::vector<float> all;
        for (size_t i = 0; i < toks.size(); i++) {
            batch.n_tokens = (int) toks[i].size();
            for (int j = 0; j < batch.n_tokens; j++) {
                batch.token[j] = toks[i][j]; batch.pos[j] = j; batch.n_seq_id[j] = 1; batch.seq_id[j][0] = 0; batch.logits[j] = 1;
            }
            if (llama_decode(lc, batch) != 0) { fprintf(stderr, "decode failed on text %zu (%d tokens)\n", i, batch.n_tokens); return 1; }
            const float * e = llama_get_embeddings_seq(lc, 0);
            if (!e) { fprintf(stderr, "no embeddings\n"); return 1; }
            all.insert(all.end(), e, e + n_out);
        }
        FILE * o = fopen(out_path.c_str(), "wb");
        int32_t hdr[2] = { (int32_t) toks.size(), n_out };
        fwrite(hdr, 4, 2, o);
        fwrite(all.data(), 4, all.size(), o);
        fclose(o);
        fprintf(stderr, "dumped %zu x %d, max tokens %d, load %.0f ms\n", toks.size(), n_out, maxn, t_load);
        llama_batch_free(batch);
        llama_free(lc);
    } else {
        const int T = L * B;
        llama_context * lc = make_ctx(T, B);
        if (!lc) { fprintf(stderr, "ctx failed\n"); return 1; }
        std::mt19937 rng(12345);
        std::vector<llama_token> seq(L);
        llama_batch batch = llama_batch_init(T, 0, 1);
        batch.n_tokens = T;
        for (int s = 0; s < B; s++) {
            seq[0] = llama_vocab_bos(vocab);
            for (int j = 1; j < L; j++) seq[j] = 1000 + (llama_token) (rng() % (uint32_t) std::min(n_vocab - 2000, 60000));
            for (int j = 0; j < L; j++) {
                int k = s * L + j;
                batch.token[k] = seq[j]; batch.pos[k] = j; batch.n_seq_id[k] = 1; batch.seq_id[k][0] = s; batch.logits[k] = 1;
            }
        }
        if (duration > 0) {
            // timed window: warm up, wait for the common absolute start (epoch seconds), then decode until the window ends
            for (int r = 0; r < warm; r++) {
                if (llama_decode(lc, batch) != 0) { fprintf(stderr, "decode failed\n"); return 1; }
                llama_get_embeddings_seq(lc, B - 1);
            }
            auto epoch = []() { return std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count(); };
            while (start_at > 0 && epoch() < start_at) { std::this_thread::sleep_for(std::chrono::milliseconds(2)); }
            const double w0 = epoch(); long done = 0; double last = w0;
            while (epoch() < w0 + duration) {
                if (llama_decode(lc, batch) != 0) { fprintf(stderr, "decode failed\n"); return 1; }
                llama_get_embeddings_seq(lc, B - 1);
                done += B; last = epoch();
            }
            printf("WINDOW L=%d B=%d start=%.3f dur=%.2f embeddings=%ld emb_per_s=%.3f load_ms=%.0f\n", L, B, w0, last - w0, done, done / (last - w0), t_load);
            llama_batch_free(batch); llama_free(lc); llama_model_free(model);
            return 0;
        }
        std::vector<double> times;
        for (int r = 0; r < warm + reps; r++) {
            double a = now_ms();
            if (llama_decode(lc, batch) != 0) { fprintf(stderr, "decode failed\n"); return 1; }
            const float * e = llama_get_embeddings_seq(lc, B - 1); // forces the sync
            if (!e) { fprintf(stderr, "no embeddings\n"); return 1; }
            double b = now_ms();
            if (r >= warm) times.push_back(b - a);
        }
        std::sort(times.begin(), times.end());
        double sum = std::accumulate(times.begin(), times.end(), 0.0);
        double mean = sum / times.size();
        double var = 0; for (double t : times) var += (t - mean) * (t - mean);
        double sd = times.size() > 1 ? std::sqrt(var / (times.size() - 1)) : 0;
        double med = times[times.size() / 2];
        printf("RESULT L=%d B=%d reps=%d mean_ms=%.2f sd_ms=%.2f med_ms=%.2f min_ms=%.2f emb_per_s_mean=%.2f emb_per_s_med=%.2f tok_per_s=%.1f load_ms=%.0f\n",
               L, B, reps, mean, sd, med, times.front(), B * 1000.0 / mean, B * 1000.0 / med, T * 1000.0 / mean, t_load);
        llama_batch_free(batch);
        llama_free(lc);
    }
    llama_model_free(model);
    return 0;
}
