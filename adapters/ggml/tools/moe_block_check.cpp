// SGLX0 MoE block (composite/moe_block.cpp) against the CPU backend, on the graph that llama.cpp builds for a MoE layer (build_moe_ffn: router, softmax, top-k,
// normalised weights, MUL_MAT_ID up / gate, SWIGLU, MUL_MAT_ID down, weighting and the ADD chain), several layers with a residual. The expert weights of the SGLX0 run
// live in a buffer made with buffer_from_host_ptr (the way llama.cpp places the mmap of a model file) and are named blk.N.ffn_{up,gate,down}_exps.weight.
//
//   c++ -std=c++17 -O1 -I<llama.cpp>/ggml/include adapters/ggml/tools/moe_block_check.cpp -L<llama.cpp>/build-dl/bin -lggml -lggml-base -Wl,-rpath,<llama.cpp>/build-dl/bin -o moe_block_check
//   GGML_BACKEND_PATH=<plugin>/libggml-cl11.so GGML_SGLX_MOE=MTL0:auto ./moe_block_check [--big] [--layers N] [--q41]   # Mac: slots on Metal
//   GGML_BACKEND_PATH=... GGML_SGLX_MOE=1 ./moe_block_check                                                          # no slot device: everything cold (cpu)
//
// A pass of N tokens (prefill) and then single tokens (decode) run on both sides; the slots are placed after the prefill. The result of the layer stack must agree to
// NMSE < 1e-6 (cold experts: identical operations, hot experts: the device's own rounding). GGML_LIB_DIR=<dir of libggml-cpu.so> if it is not next to the tool.
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <unistd.h>

static float frand() { return (float) rand() / (float) RAND_MAX - 0.5f; }

struct cfg { int K, F, E, U, L, N_pre, n_dec; bool q41; double tol; };

struct layer_w {
    std::vector<uint8_t> up, gate, down;
    std::vector<float>   router;
};

static std::vector<uint8_t> quant(ggml_type type, int64_t K, int64_t rows, float scale) {
    std::vector<float> f((size_t) K * rows);
    for (auto & v : f) v = frand() * scale;
    std::vector<uint8_t> q(ggml_row_size(type, K) * rows);
    ggml_quantize_chunk(type, f.data(), q.data(), 0, rows, K, nullptr);
    return q;
}

struct model {
    cfg c;
    std::vector<layer_w> lw;
    ggml_context * ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    std::vector<ggml_tensor *> up, gate, down, router;
    std::vector<uint8_t> hostmem;
};

// the weights of layer l in `m`: the expert tensors in `expert_buf` (placed by ggml_backend_tensor_alloc into the host memory block), the router in the default buffer of `be`
static void build_model(model & m, const cfg & c, ggml_backend_dev_t dev_exp, ggml_backend_t be_main, bool from_host_ptr) {
    m.c = c;
    ggml_init_params ip = { ggml_tensor_overhead() * (size_t) (c.L * 4 + 8), nullptr, true };
    m.ctx = ggml_init(ip);
    ggml_context * rctx = m.ctx;
    size_t total = 0;
    for (int l = 0; l < c.L; ++l) {
        const ggml_type tdown = (c.q41 && l == 0) ? GGML_TYPE_Q4_1 : GGML_TYPE_Q4_0;
        m.up.push_back(ggml_new_tensor_3d(rctx, GGML_TYPE_Q4_0, c.K, c.F, c.E));
        m.gate.push_back(ggml_new_tensor_3d(rctx, GGML_TYPE_Q4_0, c.K, c.F, c.E));
        m.down.push_back(ggml_new_tensor_3d(rctx, tdown, c.F, c.K, c.E));
        char nm[96];
        snprintf(nm, sizeof(nm), "blk.%d.ffn_up_exps.weight", l);   ggml_set_name(m.up.back(), nm);
        snprintf(nm, sizeof(nm), "blk.%d.ffn_gate_exps.weight", l); ggml_set_name(m.gate.back(), nm);
        snprintf(nm, sizeof(nm), "blk.%d.ffn_down_exps.weight", l); ggml_set_name(m.down.back(), nm);
        m.router.push_back(ggml_new_tensor_2d(rctx, GGML_TYPE_F32, c.K, c.E));
        for (ggml_tensor * t : { m.up.back(), m.gate.back(), m.down.back() }) total += (ggml_nbytes(t) + 63) & ~(size_t) 63;
    }
    if (from_host_ptr) {
        // the expert tensors in a buffer of the expert device wrapping our own host memory (like an mmap); the router in the main backend
        m.hostmem.assign(total + 4096, 0);
        char * base = (char *) (((uintptr_t) m.hostmem.data() + 63) & ~(uintptr_t) 63);
        ggml_backend_buffer_t eb = ggml_backend_dev_buffer_from_host_ptr(dev_exp, base, total, 0);
        if (!eb) { fprintf(stderr, "buffer_from_host_ptr failed\n"); exit(1); }
        ggml_backend_buffer_set_usage(eb, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        size_t off = 0;
        for (int l = 0; l < c.L; ++l) {
            for (ggml_tensor * t : { m.up[l], m.gate[l], m.down[l] }) {
                ggml_backend_tensor_alloc(eb, t, base + off);
                off += (ggml_nbytes(t) + 63) & ~(size_t) 63;
            }
        }
        m.buf = eb;
    } else {
        // plain CPU buffer for everything expert-related
        ggml_backend_buffer_t eb = ggml_backend_alloc_ctx_tensors_from_buft(m.ctx, ggml_backend_get_default_buffer_type(be_main));
        // routers are in the same context: they are allocated here too
        ggml_backend_buffer_set_usage(eb, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        m.buf = eb;
    }
}

// the router tensors of both models must exist in a buffer: allocate them separately in a second context
static ggml_backend_buffer_t alloc_routers(model & m, ggml_backend_t be, ggml_context ** pctx) {
    ggml_init_params ip = { ggml_tensor_overhead() * (size_t) (m.c.L + 4), nullptr, true };
    *pctx = ggml_init(ip);
    m.router.assign((size_t) m.c.L, nullptr);
    for (int l = 0; l < m.c.L; ++l) {
        m.router[l] = ggml_new_tensor_2d(*pctx, GGML_TYPE_F32, m.c.K, m.c.E);
    }
    ggml_backend_buffer_t b = ggml_backend_alloc_ctx_tensors(*pctx, be);
    return b;
}

// llama.cpp's build_moe_ffn (qwen3moe: softmax gating, norm_w, silu), residual after it
static ggml_tensor * moe_layer(ggml_context * g, ggml_cgraph * gf, const model & m, int l, ggml_tensor * x) {
    const int K = m.c.K, E = m.c.E, U = m.c.U;
    const int64_t N = x->ne[1];
    ggml_tensor * cur = ggml_rms_norm(g, x, 1e-6f);
    ggml_tensor * logits = ggml_mul_mat(g, m.router[l], cur);
    ggml_tensor * probs = ggml_soft_max(g, logits);
    ggml_tensor * sel = ggml_argsort_top_k(g, probs, U);
    ggml_tensor * p3 = ggml_reshape_3d(g, probs, 1, E, N);
    ggml_tensor * weights = ggml_get_rows(g, p3, sel);                      // [1, U, N]
    weights = ggml_reshape_2d(g, weights, U, N);
    ggml_tensor * wsum = ggml_clamp(g, ggml_sum_rows(g, weights), 6.103515625e-5f, INFINITY);
    weights = ggml_reshape_3d(g, ggml_div(g, weights, wsum), 1, U, N);
    ggml_build_forward_expand(gf, weights);
    ggml_tensor * c3 = ggml_reshape_3d(g, cur, K, 1, N);
    ggml_tensor * up = ggml_mul_mat_id(g, m.up[l], c3, sel);
    ggml_tensor * gt = ggml_mul_mat_id(g, m.gate[l], c3, sel);
    ggml_tensor * act = ggml_swiglu_split(g, gt, up);
    ggml_tensor * experts = ggml_mul_mat_id(g, m.down[l], act, sel);        // [K, U, N]
    experts = ggml_mul(g, experts, weights);
    ggml_build_forward_expand(gf, experts);
    ggml_tensor * views[16];
    for (int i = 0; i < U; ++i) {
        views[i] = ggml_view_2d(g, experts, K, N, experts->nb[2], (size_t) i * experts->nb[1]);
        ggml_build_forward_expand(gf, views[i]);
    }
    ggml_tensor * out = views[0];
    for (int i = 1; i < U; ++i) {
        out = ggml_add(g, out, views[i]);
        ggml_build_forward_expand(gf, out);
    }
    return ggml_add(g, x, out);
}

static ggml_tensor * run(ggml_backend_sched_t sched, const model & m, int N, const std::vector<float> & in, std::vector<float> & out, ggml_context * g, bool print_splits) {
    ggml_cgraph * gf = ggml_new_graph_custom(g, 4096, false);
    ggml_tensor * x = ggml_new_tensor_2d(g, GGML_TYPE_F32, m.c.K, N);
    ggml_set_input(x);
    ggml_tensor * h = x;
    for (int l = 0; l < m.c.L; ++l) {
        h = moe_layer(g, gf, m, l, h);
    }
    ggml_set_output(h);
    ggml_build_forward_expand(gf, h);
    ggml_backend_sched_reset(sched);
    if (!ggml_backend_sched_alloc_graph(sched, gf)) { fprintf(stderr, "alloc_graph failed\n"); exit(1); }
    if (print_splits) fprintf(stderr, "splits: %d\n", ggml_backend_sched_get_n_splits(sched));
    ggml_backend_tensor_set(x, in.data(), 0, in.size() * sizeof(float));
    if (ggml_backend_sched_graph_compute(sched, gf) != GGML_STATUS_SUCCESS) { fprintf(stderr, "compute failed\n"); exit(1); }
    out.resize((size_t) m.c.K * N);
    ggml_backend_tensor_get(h, out.data(), 0, out.size() * sizeof(float));
    return h;
}

static double nmse(const std::vector<float> & a, const std::vector<float> & b) {
    double num = 0, den = 0;
    for (size_t i = 0; i < a.size(); ++i) { const double d = (double) a[i] - b[i]; num += d * d; den += (double) b[i] * b[i]; }
    return den > 0 ? num / den : num;
}

int main(int argc, char ** argv) {
    cfg c = { 512, 256, 32, 4, 4, 48, 12, false, 1e-6 };
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--big") c = { 2048, 768, 128, 8, 3, 64, 12, false, c.tol };
        else if (a == "--tol" && i + 1 < argc) c.tol = atof(argv[++i]);
        else if (a == "--layers" && i + 1 < argc) c.L = atoi(argv[++i]);
        else if (a == "--q41") c.q41 = true;
        else if (a == "--prefill" && i + 1 < argc) c.N_pre = atoi(argv[++i]);
        else if (a == "--decode" && i + 1 < argc) c.n_dec = atoi(argv[++i]);
    }
    if (getenv("GGML_LIB_DIR") != nullptr) ggml_backend_load_all_from_path(getenv("GGML_LIB_DIR")); else ggml_backend_load_all();
    ggml_backend_dev_t sglx = ggml_backend_dev_by_name("SGLX0");
    ggml_backend_dev_t cpud = ggml_backend_dev_by_name("CPU");
    if (!sglx) { fprintf(stderr, "no SGLX0 device (GGML_BACKEND_PATH=.../libggml-cl11.so GGML_SGLX_MOE=...)\n"); return 2; }
    ggml_backend_t be_sglx = ggml_backend_dev_init(sglx, nullptr);
    ggml_backend_t be_cpu = ggml_backend_dev_init(cpud, nullptr);
    ggml_backend_t be_ref = ggml_backend_dev_init(cpud, nullptr);

    srand(7);
    model mt, mr; // test (SGLX0) and reference (CPU)
    std::vector<layer_w> lw((size_t) c.L);
    for (int l = 0; l < c.L; ++l) {
        const ggml_type tdown = (c.q41 && l == 0) ? GGML_TYPE_Q4_1 : GGML_TYPE_Q4_0;
        lw[l].up = quant(GGML_TYPE_Q4_0, c.K, (int64_t) c.F * c.E, 1.0f);
        lw[l].gate = quant(GGML_TYPE_Q4_0, c.K, (int64_t) c.F * c.E, 1.0f);
        lw[l].down = quant(tdown, c.F, (int64_t) c.K * c.E, 1.0f);
        lw[l].router.resize((size_t) c.K * c.E);
        for (auto & v : lw[l].router) v = frand() * 0.3f;
    }
    // SGLX0 side: experts from a host pointer, router in the cpu backend
    ggml_context * rctx_t = nullptr, * rctx_r = nullptr;
    build_model(mt, c, sglx, be_cpu, true);
    ggml_backend_buffer_t rb_t = alloc_routers(mt, be_cpu, &rctx_t);
    // reference: experts in a cpu buffer
    {
        mr.c = c;
        ggml_init_params ip = { ggml_tensor_overhead() * (size_t) (c.L * 4 + 8), nullptr, true };
        mr.ctx = ggml_init(ip);
        for (int l = 0; l < c.L; ++l) {
            const ggml_type tdown = (c.q41 && l == 0) ? GGML_TYPE_Q4_1 : GGML_TYPE_Q4_0;
            mr.up.push_back(ggml_new_tensor_3d(mr.ctx, GGML_TYPE_Q4_0, c.K, c.F, c.E));
            mr.gate.push_back(ggml_new_tensor_3d(mr.ctx, GGML_TYPE_Q4_0, c.K, c.F, c.E));
            mr.down.push_back(ggml_new_tensor_3d(mr.ctx, tdown, c.F, c.K, c.E));
        }
        mr.buf = ggml_backend_alloc_ctx_tensors(mr.ctx, be_ref);
        ggml_backend_buffer_set_usage(mr.buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    }
    ggml_backend_buffer_t rb_r = alloc_routers(mr, be_ref, &rctx_r);
    for (int l = 0; l < c.L; ++l) {
        memcpy(mt.up[l]->data, lw[l].up.data(), lw[l].up.size());
        memcpy(mt.gate[l]->data, lw[l].gate.data(), lw[l].gate.size());
        memcpy(mt.down[l]->data, lw[l].down.data(), lw[l].down.size());
        ggml_backend_tensor_set(mt.router[l], lw[l].router.data(), 0, lw[l].router.size() * sizeof(float));
        ggml_backend_tensor_set(mr.up[l], lw[l].up.data(), 0, lw[l].up.size());
        ggml_backend_tensor_set(mr.gate[l], lw[l].gate.data(), 0, lw[l].gate.size());
        ggml_backend_tensor_set(mr.down[l], lw[l].down.data(), 0, lw[l].down.size());
        ggml_backend_tensor_set(mr.router[l], lw[l].router.data(), 0, lw[l].router.size() * sizeof(float));
    }
    ggml_backend_buffer_set_usage(rb_t, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    ggml_backend_buffer_set_usage(rb_r, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    ggml_backend_t backs_t[2] = { be_sglx, be_cpu };
    ggml_backend_buffer_type_t bufts_t[2] = { ggml_backend_get_default_buffer_type(be_sglx), ggml_backend_get_default_buffer_type(be_cpu) };
    ggml_backend_sched_t sched_t = ggml_backend_sched_new(backs_t, bufts_t, 2, 8192, false, true);
    ggml_backend_t backs_r[1] = { be_ref };
    ggml_backend_buffer_type_t bufts_r[1] = { ggml_backend_get_default_buffer_type(be_ref) };
    ggml_backend_sched_t sched_r = ggml_backend_sched_new(backs_r, bufts_r, 1, 8192, false, true);

    bool all_ok = true;
    auto step = [&](int N, const char * what, bool first) {
        std::vector<float> in((size_t) c.K * N), o_t, o_r;
        for (auto & v : in) v = frand() * 2.0f;
        ggml_init_params gp = { 64u << 20, nullptr, true };
        ggml_context * g1 = ggml_init(gp);
        ggml_context * g2 = ggml_init(gp);
        run(sched_t, mt, N, in, o_t, g1, first);
        run(sched_r, mr, N, in, o_r, g2, false);
        const double e = nmse(o_t, o_r);
        printf("%-12s N=%-4d NMSE %.3e %s\n", what, N, e, e < c.tol ? "OK" : "FAIL");
        all_ok = all_ok && e < c.tol;
        ggml_free(g1);
        ggml_free(g2);
    };
    step(c.N_pre, "prefill", true);
    step(c.N_pre / 2 + 1, "prefill", false);
    for (int i = 0; i < c.n_dec; ++i) step(1, "decode", false);
    step(c.N_pre, "prefill", false);
    for (int i = 0; i < 3; ++i) step(1, "decode", false);
    step(3, "small", false);
    printf("%s\n", all_ok ? "all OK" : "FAILED");
    fflush(stdout);
    _exit(all_ok ? 0 : 1); // the Metal device teardown asserts on exit when resources are still alive
}
