// hec_bench: micro-benchmarks for the hot-expert-cache design (docs/plans/HOT_EXPERT_CACHE.md, unit H3: MB-H1 and MB-H2).
//
//   cpu1  MB-H1  time of the Qwen3-30B-A3B MoE block for n = 1..8 experts of one token on the ggml-cpu backend, with the real Q4_0 expert weights
//                (gate/up [2048,768,128], down [768,2048,128]) mmapped from the GGUF. Per call: MUL_MAT_ID gate + MUL_MAT_ID up + SWIGLU + MUL_MAT_ID down
//                (the weights, sum and router are not part of it). Random layer and random distinct expert ids per call, so no call finds its weights in a
//                cache. The ggml-cpu library is loaded from --cpu-lib, so one binary compares builds (default / SSE2-only / ...).
//   spin  MB-H2 diagnosis: submit a GPU graph, busy-loop D us on the host (optionally with N more busy threads), synchronize; residual wait vs D
//   conc  MB-H2  the same block split between the CPU (n_cpu experts, mmap) and a GPU backend (m_gpu experts in a slot array of --slots experts on
//                the device), run concurrently the way the design's decode path does: write x / ids, submit the GPU graph (async), compute the CPU graph,
//                synchronize the GPU, read the result. Reports GPU alone, CPU alone, both (same thread: submit async + CPU + sync) and both with the GPU
//                driven from a worker thread, so wall / max(alone) and the slowdown of the CPU part can be read off.
//
// Build (tb250, shared llama.cpp 23b0202 build, read only):
//   L=~/moe-ep/src/llama.cpp; B=~/moe-ep/src/llama-build/bin
//   g++ -O2 -std=c++17 hec_bench.cpp -I$L/ggml/include -L$B -lggml -lggml-base -lpthread -Wl,-rpath,$B -o hec_bench
// Run (use tools/hec_run.sh for the bench lock and the temperature wait):
//   hec_bench cpu1 --model Qwen3-30B-A3B-Q4_0.gguf --cpu-lib $B/libggml-cpu.so --threads 2 --iters 300
//   VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/radeon_icd.json hec_bench conc --model ... --cpu-lib ... --vk-lib $B/libggml-vulkan.so --vk-dev Vulkan0 --threads 2

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "gguf.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include <fcntl.h>
#include <pthread.h>
#include <sched.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

static double now_us() { return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

struct stats { double mean, med, p10, p90, mn; };
static stats summarize(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    double s = 0; for (double x : v) s += x;
    return { s / v.size(), v[v.size() / 2], v[v.size() / 10], v[v.size() * 9 / 10], v[0] };
}

struct model_map {
    void * base = nullptr; size_t size = 0;
    int n_layer = 0, n_embd = 0, n_ff = 0, n_expert = 0;
    std::vector<size_t> off[3];            // [gate, up, down][layer] byte offset in the file
    std::vector<size_t> eb[3];             // [gate, up, down][layer] bytes of one expert
    std::vector<enum ggml_type> ty[3];     // [gate, up, down][layer] tensor type (unsloth's Q4_0 file has ffn_down in Q4_1 for layers 0-5)
    double expert_mb(int layer) const { return (eb[0][layer] + eb[1][layer] + eb[2][layer]) / 1e6; }
};

static bool load_model(const char * path, model_map & m) {
    struct ggml_context * mc = nullptr;
    gguf_init_params gp = { /*no_alloc*/ true, &mc };
    gguf_context * g = gguf_init_from_file(path, gp);
    if (!g) { fprintf(stderr, "gguf_init_from_file failed\n"); return false; }
    const size_t data_off = gguf_get_data_offset(g);
    const char * kinds[3] = { "gate", "up", "down" };
    for (int L = 0;; L++) {
        bool any = false;
        for (int k = 0; k < 3; k++) {
            char nm[96]; snprintf(nm, sizeof(nm), "blk.%d.ffn_%s_exps.weight", L, kinds[k]);
            int64_t id = gguf_find_tensor(g, nm);
            if (id < 0) continue;
            any = true;
            ggml_tensor * t = ggml_get_tensor(mc, nm);
            m.n_expert = (int) t->ne[2];
            if (k == 0) { m.n_embd = (int) t->ne[0]; m.n_ff = (int) t->ne[1]; }
            m.ty[k].push_back(t->type);
            m.eb[k].push_back(ggml_nbytes(t) / t->ne[2]);
            if (L == 0 || t->type != m.ty[k][L - 1]) fprintf(stderr, "layer %d %s: type=%s ne=[%lld,%lld,%lld] expert=%zu B\n", L, nm, ggml_type_name(t->type), (long long) t->ne[0], (long long) t->ne[1], (long long) t->ne[2], m.eb[k].back());
            m.off[k].push_back(data_off + gguf_get_tensor_offset(g, id));
        }
        if (!any) break;
        m.n_layer = L + 1;
    }
    gguf_free(g);
    int fd = open(path, O_RDONLY);
    struct stat st; fstat(fd, &st);
    m.size = st.st_size;
    m.base = mmap(nullptr, m.size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (m.base == MAP_FAILED) { perror("mmap"); return false; }
    madvise(m.base, m.size, MADV_RANDOM);
    return true;
}

// one MoE block graph on one backend: gate/up/down expert tensors [n_embd,n_ff,S] / [n_ff,n_embd,S], n_used experts
struct moe_graph {
    ggml_context * ctx = nullptr;
    ggml_tensor *Wg, *Wu, *Wd, *x, *ids, *out;
    ggml_cgraph * gf = nullptr;
    ggml_gallocr_t ga = nullptr;
    ggml_backend_buffer_t wbuf = nullptr;   // weights buffer (device slot arrays), null for the mmap-backed CPU graph
    int n_used = 0, n_slots = 0;
    int n_embd = 0, n_ff = 0;
};

static bool build_graph(moe_graph & g, const model_map & m, ggml_backend_t be, int n_used, int n_slots, bool weights_on_backend, enum ggml_type tg, enum ggml_type tu, enum ggml_type td) {
    g.n_used = n_used; g.n_slots = n_slots; g.n_embd = m.n_embd; g.n_ff = m.n_ff;
    ggml_init_params ip = { ggml_tensor_overhead() * 32 + ggml_graph_overhead() + 4096, nullptr, true };
    g.ctx = ggml_init(ip);
    g.Wg = ggml_new_tensor_3d(g.ctx, tg, m.n_embd, m.n_ff, n_slots);
    g.Wu = ggml_new_tensor_3d(g.ctx, tu, m.n_embd, m.n_ff, n_slots);
    g.Wd = ggml_new_tensor_3d(g.ctx, td, m.n_ff, m.n_embd, n_slots);
    g.x   = ggml_new_tensor_3d(g.ctx, GGML_TYPE_F32, m.n_embd, 1, 1);
    g.ids = ggml_new_tensor_2d(g.ctx, GGML_TYPE_I32, n_used, 1);
    ggml_set_input(g.x); ggml_set_input(g.ids);
    if (weights_on_backend) {
        g.wbuf = ggml_backend_alloc_ctx_tensors_from_buft(g.ctx, ggml_backend_get_default_buffer_type(be));   // allocates Wg, Wu, Wd, x, ids on the device
        if (!g.wbuf) { fprintf(stderr, "weights alloc failed\n"); return false; }
    } else {
        // mmap-backed weights: data pointers are set per call; x / ids live in a host buffer
        g.Wg->data = g.Wu->data = g.Wd->data = m.base;
    }
    ggml_tensor * gate = ggml_mul_mat_id(g.ctx, g.Wg, g.x, g.ids);
    ggml_tensor * up   = ggml_mul_mat_id(g.ctx, g.Wu, g.x, g.ids);
    ggml_tensor * act  = ggml_swiglu_split(g.ctx, gate, up);
    g.out = ggml_mul_mat_id(g.ctx, g.Wd, act, g.ids);
    ggml_set_output(g.out);
    g.gf = ggml_new_graph(g.ctx);
    ggml_build_forward_expand(g.gf, g.out);
    g.ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(be));
    if (!ggml_gallocr_alloc_graph(g.ga, g.gf)) { fprintf(stderr, "gallocr failed\n"); return false; }
    return true;
}

struct cpu_ctx {
    ggml_backend_t be = nullptr;
    ggml_backend_dev_t dev = nullptr;
};

static cpu_ctx init_cpu(const std::string & lib, int threads, int poll) {
    cpu_ctx c;
    ggml_backend_reg_t r = ggml_backend_load(lib.c_str());
    if (!r) { fprintf(stderr, "cannot load %s\n", lib.c_str()); exit(1); }
    c.dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    c.be = ggml_backend_dev_init(c.dev, nullptr);
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(c.dev);
    typedef ggml_threadpool_t (*tp_new_t)(const ggml_threadpool_params *);
    typedef void (*set_tp_t)(ggml_backend_t, ggml_threadpool_t);
    auto tp_new = (tp_new_t) ggml_backend_reg_get_proc_address(reg, "ggml_threadpool_new");
    auto set_tp = (set_tp_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cpu_set_threadpool");
    auto set_nt = (void (*)(ggml_backend_t, int)) ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads");
    ggml_threadpool_params tpp = ggml_threadpool_params_default(threads);
    tpp.poll = poll;
    if (tp_new && set_tp) { set_tp(c.be, tp_new(&tpp)); }
    if (set_nt) set_nt(c.be, threads);
    return c;
}

static void set_layer(moe_graph & g, const model_map & m, int layer) {
    g.Wg->data = (char *) m.base + m.off[0][layer];
    g.Wu->data = (char *) m.base + m.off[1][layer];
    g.Wd->data = (char *) m.base + m.off[2][layer];
}

static void pick_ids(std::mt19937 & rng, int n, int n_expert, std::vector<int32_t> & ids) {
    ids.clear();
    while ((int) ids.size() < n) {
        int e = rng() % n_expert;
        if (std::find(ids.begin(), ids.end(), e) == ids.end()) ids.push_back(e);
    }
}

static void touch_all(const model_map & m) {
    volatile unsigned long s = 0;
    for (int k = 0; k < 3; k++) for (int L = 0; L < m.n_layer; L++) {
        const char * p = (const char *) m.base + m.off[k][L];
        for (size_t o = 0; o < m.eb[k][L] * m.n_expert; o += 4096) s += p[o];
    }
}

static std::string arg(int argc, char ** argv, const char * k, const char * def) {
    for (int i = 2; i + 1 < argc; i++) if (!strcmp(argv[i], k)) return argv[i + 1];
    return def;
}
static bool flag(int argc, char ** argv, const char * k) { for (int i = 2; i < argc; i++) if (!strcmp(argv[i], k)) return true; return false; }

static void fill_x(ggml_tensor * x, std::mt19937 & rng) {
    std::vector<float> v(ggml_nelements(x));
    for (auto & f : v) f = ((int) (rng() % 2000) - 1000) / 1000.0f;
    ggml_backend_tensor_set(x, v.data(), 0, v.size() * sizeof(float));
}

// ---------------------------------------------------------------------------------------------------------------------------------------------------
static int run_cpu1(int argc, char ** argv) {
    model_map m; if (!load_model(arg(argc, argv, "--model", "").c_str(), m)) return 1;
    int threads = atoi(arg(argc, argv, "--threads", "2").c_str()), iters = atoi(arg(argc, argv, "--iters", "300").c_str());
    int poll = atoi(arg(argc, argv, "--poll", "50").c_str());
    std::string lib = arg(argc, argv, "--cpu-lib", "");
    cpu_ctx c = init_cpu(lib, threads, poll);
    if (!flag(argc, argv, "--no-warm")) { double t0 = now_us(); touch_all(m); fprintf(stderr, "touched all expert pages: %.1f s\n", (now_us() - t0) / 1e6); }
    std::mt19937 rng(1);
    // layers whose gate / up / down are all Q4_0 (the shape the design assumes: 2.654 MB per expert) and the others, timed separately
    std::vector<int> lay_q40, lay_other;
    for (int L = 0; L < m.n_layer; L++) (m.ty[0][L] == GGML_TYPE_Q4_0 && m.ty[1][L] == GGML_TYPE_Q4_0 && m.ty[2][L] == GGML_TYPE_Q4_0 ? lay_q40 : lay_other).push_back(L);
    printf("# cpu1 lib=%s threads=%d poll=%d iters=%d layers: %zu all-Q4_0 (%.3f MB/expert), %zu with ffn_down != Q4_0 (%s, %.3f MB/expert)\n", lib.c_str(), threads, poll, iters,
           lay_q40.size(), lay_q40.empty() ? 0.0 : m.expert_mb(lay_q40[0]), lay_other.size(), lay_other.empty() ? "-" : ggml_type_name(m.ty[2][lay_other[0]]), lay_other.empty() ? 0.0 : m.expert_mb(lay_other[0]));
    printf("# group  n_used  mean_us  median_us  p10_us  p90_us  min_us  | per-expert(mean)_us  GB/s(mean)\n");
    for (int grp = 0; grp < 2; grp++) {
        const std::vector<int> & lay = grp == 0 ? lay_q40 : lay_other;
        if (lay.empty() || (grp == 1 && !flag(argc, argv, "--other"))) continue;
        const double emb = m.expert_mb(lay[0]);
        std::vector<double> n_axis, mean_axis;
        for (int n = 1; n <= 8; n++) {
            moe_graph g; if (!build_graph(g, m, c.be, n, m.n_expert, false, m.ty[0][lay[0]], m.ty[1][lay[0]], m.ty[2][lay[0]])) return 1;
            fill_x(g.x, rng);
            std::vector<int32_t> ids; std::vector<double> ts;
            for (int it = -20; it < iters; it++) {
                pick_ids(rng, n, m.n_expert, ids);
                ggml_backend_tensor_set(g.ids, ids.data(), 0, n * sizeof(int32_t));
                set_layer(g, m, lay[rng() % lay.size()]);
                double t0 = now_us();
                ggml_backend_graph_compute(c.be, g.gf);
                double t1 = now_us();
                if (it >= 0) ts.push_back(t1 - t0);
            }
            stats s = summarize(ts);
            printf("%s  %d  %.1f  %.1f  %.1f  %.1f  %.1f  | %.1f  %.2f\n", grp == 0 ? "q4_0" : "other", n, s.mean, s.med, s.p10, s.p90, s.mn, s.mean / n, n * emb * 1e6 / (s.mean * 1e-6) / 1e9);
            fflush(stdout);
            n_axis.push_back(n); mean_axis.push_back(s.mean);
            ggml_gallocr_free(g.ga); ggml_free(g.ctx);
        }
        double sx = 0, sy = 0, sxx = 0, sxy = 0; int N = (int) n_axis.size();     // least squares t = a + b * n
        for (int i = 0; i < N; i++) { sx += n_axis[i]; sy += mean_axis[i]; sxx += n_axis[i] * n_axis[i]; sxy += n_axis[i] * mean_axis[i]; }
        double b = (N * sxy - sx * sy) / (N * sxx - sx * sx), a = (sy - b * sx) / N;
        printf("# fit %s: t = %.1f us + n * %.1f us   (per-expert %.3f ms, %.2f GB/s)\n", grp == 0 ? "q4_0" : "other", a, b, b / 1000, emb * 1e6 / (b * 1e-6) / 1e9);
    }
    return 0;
}

// ---------------------------------------------------------------------------------------------------------------------------------------------------
struct gpu_worker {
    std::thread th; std::mutex mu; std::condition_variable cv;
    bool go = false, done = false, quit = false;
    ggml_backend_t be = nullptr; ggml_cgraph * gf = nullptr; double t_done = 0;
    int pin_cpu = -1;
    void start() {
        th = std::thread([this]() {
            if (pin_cpu >= 0) { cpu_set_t cs; CPU_ZERO(&cs); CPU_SET(pin_cpu, &cs); pthread_setaffinity_np(pthread_self(), sizeof(cs), &cs); }
            std::unique_lock<std::mutex> lk(mu);
            for (;;) {
                cv.wait(lk, [&] { return go || quit; });
                if (quit) return;
                go = false;
                lk.unlock();
                ggml_backend_graph_compute_async(be, gf);
                ggml_backend_synchronize(be);
                t_done = now_us();
                lk.lock();
                done = true; cv.notify_all();
            }
        });
    }
    void kick() { std::lock_guard<std::mutex> lk(mu); go = true; done = false; cv.notify_all(); }
    void wait() { std::unique_lock<std::mutex> lk(mu); cv.wait(lk, [&] { return done; }); }
    void stop() { { std::lock_guard<std::mutex> lk(mu); quit = true; cv.notify_all(); } th.join(); }
};

static int run_conc(int argc, char ** argv) {
    model_map m; if (!load_model(arg(argc, argv, "--model", "").c_str(), m)) return 1;
    int threads = atoi(arg(argc, argv, "--threads", "2").c_str()), iters = atoi(arg(argc, argv, "--iters", "200").c_str());
    int poll = atoi(arg(argc, argv, "--poll", "50").c_str()), slots = atoi(arg(argc, argv, "--slots", "24").c_str());
    std::string lib = arg(argc, argv, "--cpu-lib", ""), vklib = arg(argc, argv, "--vk-lib", ""), vkdev = arg(argc, argv, "--vk-dev", "Vulkan0");
    std::string pairs = arg(argc, argv, "--pairs", "8:0,0:8,6:2,4:4,2:6,5:3,3:5,1:7,7:1");   // m_gpu:n_cpu
    cpu_ctx c = init_cpu(lib, threads, poll);
    if (!ggml_backend_load(vklib.c_str())) { fprintf(stderr, "cannot load %s\n", vklib.c_str()); return 1; }
    ggml_backend_dev_t vd = ggml_backend_dev_by_name(vkdev.c_str());
    if (!vd) { fprintf(stderr, "no device %s\n", vkdev.c_str()); return 1; }
    ggml_backend_t vk = ggml_backend_dev_init(vd, nullptr);
    if (!flag(argc, argv, "--no-warm")) touch_all(m);
    const bool pin = flag(argc, argv, "--pin");     // main thread (submit + CPU part) on core 0, GPU worker thread on core 1 (use with --threads 1)
    if (pin) { cpu_set_t cs; CPU_ZERO(&cs); CPU_SET(0, &cs); pthread_setaffinity_np(pthread_self(), sizeof(cs), &cs); }
    std::mt19937 rng(7);
    std::vector<int> lay_q40;                                  // only layers with gate / up / down all Q4_0 are used on both sides
    for (int L = 0; L < m.n_layer; L++) if (m.ty[0][L] == GGML_TYPE_Q4_0 && m.ty[1][L] == GGML_TYPE_Q4_0 && m.ty[2][L] == GGML_TYPE_Q4_0) lay_q40.push_back(L);
    const int lay0 = lay_q40[0];

    // GPU slot array filled with the first `slots` experts of layer 0 (real weights; the values do not matter for timing)
    printf("# conc cpu_lib=%s threads=%d poll=%d iters=%d vk=%s (%s) slots=%d\n", lib.c_str(), threads, poll, iters, vkdev.c_str(), ggml_backend_dev_description(vd), slots);
    printf("# columns: m_gpu n_cpu | gpu_alone cpu_alone (mean us) | same-thread: wall cpu_part gpu_wait | worker-thread: wall cpu_part | wall/max(alone)  wall/sum(alone)  cpu slowdown(same)  cpu slowdown(worker)\n");

    std::vector<std::pair<int, int>> prs;
    { size_t p = 0; while (p < pairs.size()) { size_t q = pairs.find(',', p); if (q == std::string::npos) q = pairs.size(); std::string s = pairs.substr(p, q - p); prs.push_back({atoi(s.c_str()), atoi(s.substr(s.find(':') + 1).c_str())}); p = q + 1; } }

    for (auto pr : prs) {
        int mg = pr.first, nc = pr.second;
        moe_graph gg, gc;
        if (mg > 0) {
            if (!build_graph(gg, m, vk, mg, slots, true, GGML_TYPE_Q4_0, GGML_TYPE_Q4_0, GGML_TYPE_Q4_0)) return 1;
            // upload real weights into the slot array
            for (int k = 0; k < 3; k++) {
                ggml_tensor * W = k == 0 ? gg.Wg : k == 1 ? gg.Wu : gg.Wd;
                ggml_backend_tensor_set(W, (char *) m.base + m.off[k][lay0], 0, m.eb[k][lay0] * slots);
            }
            fill_x(gg.x, rng);
        }
        if (nc > 0) { if (!build_graph(gc, m, c.be, nc, m.n_expert, false, GGML_TYPE_Q4_0, GGML_TYPE_Q4_0, GGML_TYPE_Q4_0)) return 1; fill_x(gc.x, rng); }

        gpu_worker wk; if (mg > 0) { wk.be = vk; wk.gf = gg.gf; wk.pin_cpu = pin ? 1 : -1; wk.start(); }
        std::vector<double> t_g, t_c, t_s, t_sc, t_sg, t_ss, t_w, t_wc;
        std::vector<float> res;
        std::vector<int32_t> ids;
        auto prep_gpu = [&]() {
            pick_ids(rng, mg, slots, ids);
            ggml_backend_tensor_set(gg.ids, ids.data(), 0, mg * sizeof(int32_t));
        };
        auto prep_cpu = [&]() {
            pick_ids(rng, nc, m.n_expert, ids);
            ggml_backend_tensor_set(gc.ids, ids.data(), 0, nc * sizeof(int32_t));
            set_layer(gc, m, lay_q40[rng() % lay_q40.size()]);
        };
        if (mg > 0) res.resize((size_t) m.n_embd * mg);
        for (int phase = 0; phase < 4; phase++) {          // 0 gpu alone, 1 cpu alone, 2 same thread, 3 worker thread
            if (phase == 0 && mg == 0) continue;
            if (phase == 1 && nc == 0) continue;
            if ((phase == 2 || phase == 3) && (mg == 0 || nc == 0)) continue;
            for (int it = -10; it < iters; it++) {
                double t0 = now_us(), t1 = 0, t2 = 0, t3 = 0;
                if (phase == 0) {
                    prep_gpu();
                    ggml_backend_graph_compute_async(vk, gg.gf); ggml_backend_synchronize(vk);
                    ggml_backend_tensor_get(gg.out, res.data(), 0, res.size() * 4);
                    t1 = now_us(); if (it >= 0) t_g.push_back(t1 - t0);
                } else if (phase == 1) {
                    prep_cpu();
                    t0 = now_us(); ggml_backend_graph_compute(c.be, gc.gf); t1 = now_us();
                    if (it >= 0) t_c.push_back(t1 - t0);
                } else if (phase == 2) {
                    prep_gpu(); prep_cpu();
                    t0 = now_us();
                    ggml_backend_graph_compute_async(vk, gg.gf);           // submit
                    t1 = now_us();
                    ggml_backend_graph_compute(c.be, gc.gf);               // CPU part
                    t2 = now_us();
                    ggml_backend_synchronize(vk);                          // wait
                    ggml_backend_tensor_get(gg.out, res.data(), 0, res.size() * 4);
                    t3 = now_us();
                    if (it >= 0) { t_s.push_back(t3 - t0); t_sc.push_back(t2 - t1); t_sg.push_back(t3 - t2); t_ss.push_back(t1 - t0); }
                } else {
                    prep_gpu(); prep_cpu();
                    t0 = now_us();
                    wk.kick();
                    ggml_backend_graph_compute(c.be, gc.gf);
                    t2 = now_us();
                    wk.wait();
                    ggml_backend_tensor_get(gg.out, res.data(), 0, res.size() * 4);
                    t3 = now_us();
                    if (it >= 0) { t_w.push_back(t3 - t0); t_wc.push_back(t2 - t0); }
                }
            }
        }
        if (mg > 0) wk.stop();
        auto mean = [](const std::vector<double> & v) { if (v.empty()) return 0.0; double s = 0; for (double x : v) s += x; return s / v.size(); };
        double g = mean(t_g), cc = mean(t_c), s = mean(t_s), sc = mean(t_sc), w = mean(t_w), wc = mean(t_wc);
        double mx = std::max(g, cc), sum = g + cc;
        printf("%d %d | %.0f %.0f | %.0f %.0f %.0f (submit %.0f) | %.0f %.0f | %.3f %.3f %.3f %.3f\n", mg, nc, g, cc, s, sc, mean(t_sg), mean(t_ss), w, wc,
               s > 0 ? s / mx : 0, s > 0 ? s / sum : 0, (s > 0 && cc > 0) ? sc / cc : 0, (w > 0 && cc > 0) ? wc / cc : 0);
        fflush(stdout);
        if (mg > 0) { ggml_gallocr_free(gg.ga); ggml_backend_buffer_free(gg.wbuf); ggml_free(gg.ctx); }
        if (nc > 0) { ggml_gallocr_free(gc.ga); ggml_free(gc.ctx); }
    }
    return 0;
}

// ---------------------------------------------------------------------------------------------------------------------------------------------------
// spin: does the GPU make progress while the host is busy? submit the GPU graph (async), busy-loop D us on the calling thread (optionally with
// --busy N further spinning threads, to emulate the ggml-cpu threads of the 2-core Celeron), then synchronize. The residual wait after the spin shows how
// much of the GPU latency was hidden: if the GPU only starts once the cores are free (a driver submit thread starved by the busy cores), the wait stays constant.
static int run_spin(int argc, char ** argv) {
    model_map m; if (!load_model(arg(argc, argv, "--model", "").c_str(), m)) return 1;
    int iters = atoi(arg(argc, argv, "--iters", "200").c_str()), slots = atoi(arg(argc, argv, "--slots", "24").c_str());
    int mg = atoi(arg(argc, argv, "--m-gpu", "4").c_str()), busy = atoi(arg(argc, argv, "--busy", "0").c_str());
    std::string vklib = arg(argc, argv, "--vk-lib", ""), vkdev = arg(argc, argv, "--vk-dev", "Vulkan0");
    if (!ggml_backend_load(vklib.c_str())) return 1;
    ggml_backend_dev_t vd = ggml_backend_dev_by_name(vkdev.c_str());
    ggml_backend_t vk = ggml_backend_dev_init(vd, nullptr);
    std::mt19937 rng(3);
    int lay0 = 6;
    moe_graph gg; if (!build_graph(gg, m, vk, mg, slots, true, GGML_TYPE_Q4_0, GGML_TYPE_Q4_0, GGML_TYPE_Q4_0)) return 1;
    for (int k = 0; k < 3; k++) { ggml_tensor * W = k == 0 ? gg.Wg : k == 1 ? gg.Wu : gg.Wd; ggml_backend_tensor_set(W, (char *) m.base + m.off[k][lay0], 0, m.eb[k][lay0] * slots); }
    fill_x(gg.x, rng);
    std::atomic<bool> stop{false};
    std::vector<std::thread> bt;
    const bool mem = flag(argc, argv, "--mem");        // the busy loops stream through a 512 MB buffer (DRAM traffic like the expert weights) instead of an ALU spin
    std::vector<uint64_t> big(mem ? (512u << 20) / 8 : 1, 1);
    auto burn = [&](double until_us_from, double D) {   // run for D us
        if (!mem) { while (now_us() - until_us_from < D) {} return; }
        static thread_local size_t pos = 0; uint64_t acc = 0;
        while (now_us() - until_us_from < D) { for (int k = 0; k < 4096; k++) { acc += big[pos]; pos = (pos + 8) % big.size(); } }
        if (acc == 1) printf(" ");
    };
    for (int i = 0; i < busy; i++) bt.emplace_back([&]() { while (!stop.load(std::memory_order_relaxed)) burn(now_us(), 200); });
    std::vector<float> res((size_t) m.n_embd * mg);
    std::vector<int32_t> ids;
    printf("# spin vk=%s m_gpu=%d busy_threads=%d mem=%d iters=%d\n# D_us  wall_us (set..get)  residual_wait_us (sync)  submit_us\n", vkdev.c_str(), mg, busy, (int) mem, iters);
    for (int D : {0, 50, 100, 200, 300, 400, 600, 800, 1200, 2000}) {
        std::vector<double> tw, tr, ts;
        for (int it = -10; it < iters; it++) {
            pick_ids(rng, mg, slots, ids);
            ggml_backend_tensor_set(gg.ids, ids.data(), 0, mg * sizeof(int32_t));
            double t0 = now_us();
            ggml_backend_graph_compute_async(vk, gg.gf);
            double t1 = now_us();
            burn(t1, D);
            double t2 = now_us();
            ggml_backend_synchronize(vk);
            double t3 = now_us();
            ggml_backend_tensor_get(gg.out, res.data(), 0, res.size() * 4);
            double t4 = now_us();
            if (it >= 0) { tw.push_back(t4 - t0); tr.push_back(t3 - t2); ts.push_back(t1 - t0); }
        }
        auto mean = [](const std::vector<double> & v) { double s = 0; for (double x : v) s += x; return s / v.size(); };
        printf("%d  %.0f  %.0f  %.0f\n", D, mean(tw), mean(tr), mean(ts));
        fflush(stdout);
    }
    stop = true; for (auto & t : bt) t.join();
    return 0;
}

int main(int argc, char ** argv) {
    if (argc < 2) { fprintf(stderr, "usage: hec_bench cpu1|conc --model GGUF --cpu-lib LIB [--threads N --iters N ...]\n"); return 2; }
    if (!strcmp(argv[1], "cpu1")) return run_cpu1(argc, argv);
    if (!strcmp(argv[1], "conc")) return run_conc(argc, argv);
    if (!strcmp(argv[1], "spin")) return run_spin(argc, argv);
    fprintf(stderr, "unknown mode\n");
    return 2;
}
