// SGLX0 MoE block execution and the hot / medium / cold expert placement (see moe_block.h and docs/plans/HOT_EXPERT_CACHE.md).
//
// What llama.cpp hands to SGLX0 for one MoE layer (build_moe_ffn): up = MUL_MAT_ID(up_exps, x, ids), gate = MUL_MAT_ID(gate_exps, x, ids),
// act = SWIGLU(gate, up), y = MUL_MAT_ID(down_exps, act, ids) [n_embd, n_used, N], then y *= weights and the sum of the n_used slices (MUL, VIEW, ADD).
// The expert tensors are the mmap of the model file (SGLX0 buffer from host pointer), so the CPU reads them in place. This module
//   * runs the four nodes up / gate / SWIGLU / down as one job ("block") at the first of them. For every (token, slot) pair the expert is either
//     in a slot array of a GPU (hot / medium: the pdev computes up, gate, SWIGLU and down in one graph, a few KB in and out) or only in the mmap
//     (cold: ggml-cpu, 2 threads). The result of every pair goes unweighted to y; MUL and ADD then run on the host in the order of ggml's graph, so the
//     sum is the same whatever device computed a pair.
//   * counts how often every (layer, expert) is selected (prefill and decode), and places the most frequent experts into the slots of the GPUs in the
//     order of the devices in GGML_SGLX_MOE (static placement from the statistics of the prompt; GGML_SGLX_MOE_FREEZE=1 keeps the first placement).
//   * runs any other MoE node (a MUL_MAT_ID that is not part of a recognised block, ...) with ggml-cpu on the mmap'd weights: slow but correct.

#include "moe_block.h"
#include "moe_cpu.h"
#include "pdev.h"

#include "ggml-alloc.h"
#include "ggml-backend-impl.h"
#include "ggml-impl.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <functional>
#include <list>
#include <map>
#include <memory>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>

namespace sglx {
namespace moe {
namespace {

using clk = std::chrono::steady_clock;

double ms_since(clk::time_point t0) {
    return std::chrono::duration<double, std::milli>(clk::now() - t0).count();
}

struct acc {
    double   sum = 0.0;
    double   mx  = 0.0;
    uint64_t n   = 0;
    void add(double v) { sum += v; mx = std::max(mx, v); ++n; }
    double mean() const { return n != 0 ? sum / (double) n : 0.0; }
};

#define MOE_LOG(...) fprintf(stderr, "SGLX MoE: " __VA_ARGS__)

struct cls_rec {
    moe_shape sh;
    size_t    b_gate = 0, b_up = 0, b_down = 0; // bytes of one expert's matrices
    int       n_layers = 0;
    double    cpu_cost = 1.0;                   // time of one expert on the cpu relative to a Q4_0 one (the value of a slot is frequency * cpu_cost)
};

// ggml-cpu without AVX: a Q4_1 down projection is 2.2 times slower than Q4_0 (tb250, docs/benchmarks/moe-ep-20261023/hec-day1), the K-quants are slower still
double cpu_cost_of(ggml_type t) {
    switch (t) {
        case GGML_TYPE_Q4_0: case GGML_TYPE_Q8_0: return 1.0;
        case GGML_TYPE_Q4_1: case GGML_TYPE_Q5_0: return 2.2;
        default:                                  return 3.0;
    }
}

struct layer_rec {
    int           il = -1;
    ggml_tensor * up = nullptr;
    ggml_tensor * gate = nullptr;
    ggml_tensor * down = nullptr;
    int           cls = -1;
    int64_t       n_expert = 0;
    std::vector<int16_t> dev;    // per expert: index into devs, -1 = cold (mmap only)
    std::vector<int32_t> slot;   // per expert: slot in the array of (dev, cls)
    std::vector<double>  freq;   // weighted selection counts
    bool complete() const { return up != nullptr && gate != nullptr && down != nullptr; }
};

struct dev_cls {
    moe_slots *          s = nullptr;
    int                  n_slots = 0;     // slots for placed experts: 0 .. n_slots-1
    int                  n_ring = 0;      // streaming ring (prefill): the slots n_slots .. n_slots+n_ring-1
    std::vector<int32_t> owner;  // slot -> layer_index * n_expert_max + expert, -1 = free
};

struct dev_rec {
    std::unique_ptr<pdev> p;
    std::string           name;
    long                  want_main = -1;   // slots for the main class: -1 = as many as fit
    long                  want_other = 0;   // slots for each other class
    bool                  gated = false;    // medium device: used in a layer only if the cpu would take longer (see run_decode)
    double                lat_ms = 3.0;     // its latency for one job (EMA)
    uint64_t              gate_tick = 0;
    uint64_t              gate_skipped = 0, gate_used = 0;
    std::vector<dev_cls>  cls;
    std::vector<float>    out;              // result staging [n_embd, 16]
    std::vector<int32_t>  ps, pt;           // pair arrays handed to moe_submit (must outlive the call)
    acc                   t_submit, t_wait;
};

// a ggml-cpu graph for the cold experts of one class (weights are dummies whose data is set before every run)
struct cpu_graph {
    int            n = 0;
    ggml_context * ctx = nullptr;
    ggml_cgraph *  gf = nullptr;
    ggml_tensor *  wu = nullptr, * wg = nullptr, * wd = nullptr;
    ggml_tensor *  x = nullptr, * ids = nullptr, * tok = nullptr, * y = nullptr;
    ggml_gallocr_t ga = nullptr;
    ~cpu_graph() {
        if (ga != nullptr) {
            ggml_gallocr_free(ga);
        }
        if (ctx != nullptr) {
            ggml_free(ctx);
        }
    }
};

struct block {
    ggml_tensor * up = nullptr, * gate = nullptr, * glu = nullptr, * down = nullptr;
    // the weighting and the sum that follow (MUL(down, weights), then the ADD chain of the n_used slices): when they are found, a prefill is finished on the devices and the
    // host writes the sum (`fin`, [n_embd, N]) directly; the nodes are then skipped
    ggml_tensor *              mul = nullptr;
    std::vector<ggml_tensor *> adds;
    ggml_tensor *              fin = nullptr;
    bool                       tail_done = false;   // set by run_block when it has written `fin`
    int                        jdown = -1;          // index of `down` in the graph
};

struct state {
    bool                      ready = false;
    ggml_backend_buffer_type_t buft = nullptr;
    hooks                     h{};
    // configuration
    struct spec_t { std::string name; long main = -1; long other = 0; bool gated = false; };
    std::vector<spec_t>       specs;
    int                       n_threads = 2;
    bool                      use_kern = true;      // GGML_SGLX_MOE_CPUKERN=0: use a ggml graph for the cold experts of a token (OpenMP: see moe_cpu.h)
    int                       poll = 50;           // GGML_SGLX_MOE_POLL: how long the cpu threads spin for the next graph (0..100), as llama-bench -poll
    double                    halflife = 4096.0;   // tokens
    bool                      freeze = false;
    bool                      profile = false;
    long                      reserve_mib = 320;   // device memory left free when the slot count is automatic
    int                       plan_min_tokens = 16;
    std::string               stats_in, stats_out;  // GGML_SGLX_MOE_STATS_IN / _OUT: a prior for the frequencies (from an earlier run), and where to write them at exit
    double                    prior_tokens = 64.0;  // GGML_SGLX_MOE_PRIOR_TOKENS: the weight of the prior, in tokens of selections
    bool                      prior_applied = false;
    double                    min_mass = 0.0;       // GGML_SGLX_MOE_MIN_MASS: a device after the first keeps the experts of a layer only if they cover this fraction of the layer's selections
    int                       ring_main = 96;      // GGML_SGLX_MOE_RING: ring slots of the main class on the first device (0 = no streaming)
    int                       stream_tau = 3;
    bool                      fused_prefill = true;  // GGML_SGLX_MOE_FUSED=0: pair outputs to the host and the weighted sum there (the first implementation)      // experts with more than this many pairs in a prefill ubatch are streamed, the others run on the cpu
    int                       order = 0;            // 0: submit the GPUs, then the CPU, then wait; 1: CPU first
    int                       max_gpus_per_layer = 0; // 0 = no limit (not used yet)

    // model
    std::vector<cls_rec>      classes;
    std::vector<layer_rec>    layers;
    std::map<const ggml_tensor *, int> layer_of;    // an expert tensor -> index into layers
    std::map<int, int>        layer_by_il;
    int64_t                   n_expert_max = 0;
    int                       max_il = -1;            // the last layer with experts
    int                       nu_seen = 8;            // experts per token

    // devices
    bool                      devs_inited = false;
    bool                      cpu_inited = false;
    std::vector<dev_rec>      devs;

    // cpu
    ggml_backend_t            cpu = nullptr;
    cpu_kernel                kern;                 // decode on the cold experts without a ggml graph (moe_cpu.cpp)
    bool                      kern_ok = false;
    int                       kern_spin_us = 0;
    std::vector<std::vector<std::unique_ptr<cpu_graph>>> cold; // [class][n-1]
    std::vector<double>       cold_ms;              // per class: time of one cold expert on the cpu (EMA, ms)

    // statistics and placement
    double                    scale = 1.0;           // weight of the next selection (grows by 2^(1/halflife) per token)
    int64_t                   tokens_seen = 0;
    int64_t                   tokens_since_plan = 0;
    int                       plans = 0;
    bool                      replan_pending = false;
    int                       last_il = -1000;
    bool                      pass_started = false;
    int                       pass_n = 0;            // tokens of the current pass
    int                       passes = 0;

    std::unordered_set<const ggml_tensor *> done;

    // profile
    acc                       t_dec, t_dec_cold, t_dec_gpu_sub, t_dec_gpu_wait, t_dec_host;
    acc                       t_pre, t_pre_cold, t_pre_stream;
    acc                       t_copy;               // lazy input copies per graph_compute call
    acc                       t_cold_alloc, t_cold_set, t_cold_compute;  // inside cold_decode
    double                    n_stream_experts = 0;
    double                    n_pairs_gpu = 0, n_pairs_cold = 0;       // decode
    double                    n_pre_gpu = 0, n_pre_cold = 0;           // prefill
    uint64_t                  dec_blocks = 0;
    double                    gpu_used_sum = 0;
    double                    plan_ms = 0;
    std::vector<float>        xg;                    // gathered columns
};

state * S = nullptr;

// ---------------------------------------------------------------------------------------------------------------------------------------
// configuration

bool parse_env(state & st) {
    const char * env = getenv("GGML_SGLX_MOE");
    if (env == nullptr || *env == '\0' || strcmp(env, "0") == 0) {
        return false;
    }
    std::string s(env);
    size_t pos = 0;
    while (pos <= s.size()) {
        size_t end = s.find(',', pos);
        if (end == std::string::npos) {
            end = s.size();
        }
        std::string item = s.substr(pos, end - pos);
        pos = end + 1;
        if (item.empty() || item == "1") {
            continue;
        }
        state::spec_t sp;
        if (!item.empty() && item.back() == '!') {   // "RPC0:300!": a medium device that is used in a layer only when the cpu would take longer than the device's latency
            sp.gated = true;
            item.pop_back();
        }
        size_t c1 = item.find(':');
        sp.name = item.substr(0, c1);
        if (c1 != std::string::npos) {
            std::string rest = item.substr(c1 + 1);
            size_t c2 = rest.find(':');
            std::string a = rest.substr(0, c2);
            sp.main = a == "auto" ? -1 : atol(a.c_str());
            if (c2 != std::string::npos) {
                sp.other = atol(rest.c_str() + c2 + 1);
            }
        }
        st.specs.push_back(sp);
    }
    auto envi = [](const char * n, long dflt) { const char * v = getenv(n); return v != nullptr && *v != '\0' ? atol(v) : dflt; };
    st.n_threads = (int) envi("GGML_SGLX_MOE_THREADS", 2);
    st.poll = (int) envi("GGML_SGLX_MOE_POLL", 50);
    if (getenv("GGML_SGLX_MOE_STATS_IN") != nullptr) { st.stats_in = getenv("GGML_SGLX_MOE_STATS_IN"); }
    if (getenv("GGML_SGLX_MOE_STATS_OUT") != nullptr) { st.stats_out = getenv("GGML_SGLX_MOE_STATS_OUT"); }
    st.prior_tokens = (double) envi("GGML_SGLX_MOE_PRIOR_TOKENS", 64);
    if (getenv("GGML_SGLX_MOE_MIN_MASS") != nullptr) {
        st.min_mass = atof(getenv("GGML_SGLX_MOE_MIN_MASS"));
    }
    st.use_kern = envi("GGML_SGLX_MOE_CPUKERN", 1) != 0;
    st.kern_spin_us = (int) envi("GGML_SGLX_MOE_SPIN_US", 0);
    st.halflife = (double) envi("GGML_SGLX_MOE_HALFLIFE", 4096);
    st.freeze = envi("GGML_SGLX_MOE_FREEZE", 0) != 0;
    st.profile = envi("GGML_SGLX_PROFILE", 0) != 0;
    st.reserve_mib = envi("GGML_SGLX_MOE_RESERVE_MIB", 320);
    st.plan_min_tokens = (int) envi("GGML_SGLX_MOE_PLAN_TOKENS", 16);
    st.order = (int) envi("GGML_SGLX_MOE_ORDER", 0);
    st.ring_main = (int) envi("GGML_SGLX_MOE_RING", 96);
    st.stream_tau = (int) envi("GGML_SGLX_MOE_STREAM_TAU", 3);
    st.fused_prefill = envi("GGML_SGLX_MOE_FUSED", 1) != 0;
    return true;
}

// ---------------------------------------------------------------------------------------------------------------------------------------
// the model: layers and classes

int class_of(state & st, const moe_shape & sh, const ggml_tensor * g, const ggml_tensor * u, const ggml_tensor * d) {
    for (size_t i = 0; i < st.classes.size(); ++i) {
        if (st.classes[i].sh == sh) {
            return (int) i;
        }
    }
    cls_rec c;
    c.sh = sh;
    c.b_gate = g->nb[2];
    c.b_up = u->nb[2];
    c.b_down = d->nb[2];
    c.cpu_cost = (cpu_cost_of(sh.t_gate) + cpu_cost_of(sh.t_up) + cpu_cost_of(sh.t_down)) / 3.0;
    st.classes.push_back(c);
    return (int) st.classes.size() - 1;
}

// "blk.12.ffn_up_exps.weight" -> (12, 'u'); returns false for other names
bool parse_name(const char * name, int & il, char & kind) {
    if (strncmp(name, "blk.", 4) != 0) {
        return false;
    }
    char * e = nullptr;
    il = (int) strtol(name + 4, &e, 10);
    if (e == name + 4 || *e != '.') {
        return false;
    }
    const char * r = e + 1;
    if (strncmp(r, "ffn_up_exps", 11) == 0) { kind = 'u'; return true; }
    if (strncmp(r, "ffn_gate_exps", 13) == 0) { kind = 'g'; return true; }
    if (strncmp(r, "ffn_down_exps", 13) == 0) { kind = 'd'; return true; }
    return false;
}

void sync_classes(state & st);
void apply_prior(state & st, int nu);

// finish a layer record when its three tensors are known
void complete_layer(state & st, int idx) {
    layer_rec & L = st.layers[idx];
    if (!L.complete() || L.cls >= 0) {
        return;
    }
    if (L.up->ne[0] != L.gate->ne[0] || L.up->ne[1] != L.gate->ne[1] || L.up->ne[2] != L.gate->ne[2] || L.down->ne[0] != L.up->ne[1] ||
        L.down->ne[1] != L.up->ne[0] || L.down->ne[2] != L.up->ne[2]) {
        MOE_LOG("layer %d: the three expert tensors do not fit together, not cached\n", L.il);
        L.cls = -2;
        return;
    }
    moe_shape sh;
    sh.t_gate = L.gate->type;
    sh.t_up = L.up->type;
    sh.t_down = L.down->type;
    sh.n_embd = L.up->ne[0];
    sh.n_ff = L.up->ne[1];
    L.cls = class_of(st, sh, L.gate, L.up, L.down);
    st.classes[L.cls].n_layers++;
    st.max_il = std::max(st.max_il, L.il);
    sync_classes(st);
    L.n_expert = L.up->ne[2];
    L.dev.assign((size_t) L.n_expert, -1);
    L.slot.assign((size_t) L.n_expert, -1);
    L.freq.assign((size_t) L.n_expert, 0.0);
    st.n_expert_max = std::max(st.n_expert_max, L.n_expert);
}

void sync_classes(state & st) {
    if (st.cold_ms.size() < st.classes.size()) {
        const size_t old = st.cold_ms.size();
        st.cold_ms.resize(st.classes.size());
        for (size_t i = old; i < st.classes.size(); ++i) {
            st.cold_ms[i] = 0.7 * st.classes[i].cpu_cost;    // the first guess: a Q4_0 expert takes 0.28 ms on two threads, 0.55 on one
        }
    }
    for (auto & d : st.devs) {
        if (d.cls.size() < st.classes.size()) {
            d.cls.resize(st.classes.size());
        }
    }
    if (st.cold.size() < st.classes.size()) {
        st.cold.resize(st.classes.size());
        for (auto & v : st.cold) {
            if (v.size() < 16) {
                v.resize(16);
            }
        }
    }
}

int layer_index_for_tensor(state & st, ggml_tensor * t, char kind, int il_hint) {
    auto it = st.layer_of.find(t);
    if (it != st.layer_of.end()) {
        return it->second;
    }
    int idx = -1;
    if (il_hint >= 0) {
        auto jt = st.layer_by_il.find(il_hint);
        if (jt != st.layer_by_il.end()) {
            idx = jt->second;
        }
    }
    if (idx < 0) {
        idx = (int) st.layers.size();
        st.layers.emplace_back();
        st.layers[idx].il = il_hint >= 0 ? il_hint : 100000 + idx;
        if (il_hint >= 0) {
            st.layer_by_il[il_hint] = idx;
        }
    }
    layer_rec & L = st.layers[idx];
    if (kind == 'u') { L.up = t; }
    else if (kind == 'g') { L.gate = t; }
    else { L.down = t; }
    st.layer_of[t] = idx;
    complete_layer(st, idx);
    return idx;
}

// ---------------------------------------------------------------------------------------------------------------------------------------
// devices and slots

int main_class(const state & st) {
    int best = -1;
    for (size_t i = 0; i < st.classes.size(); ++i) {
        if (best < 0 || st.classes[i].n_layers > st.classes[best].n_layers) {
            best = (int) i;
        }
    }
    return best;
}

void ensure_cpu(state & st) {
    if (st.cpu_inited) {
        return;
    }
    st.cpu_inited = true;
    ggml_backend_dev_t cd = find_ggml_device("CPU");
    if (cd != nullptr) {
        st.cpu = ggml_backend_dev_init(cd, nullptr);
        if (st.cpu != nullptr) {
            ggml_backend_reg_t creg = ggml_backend_dev_backend_reg(cd);
            using fn_t = void (*)(ggml_backend_t, int);
            auto fn = (fn_t) ggml_backend_reg_get_proc_address(creg, "ggml_backend_set_n_threads");
            if (fn != nullptr) {
                fn(st.cpu, st.n_threads);
            }
            // a threadpool of our own: without one every graph_compute starts and joins its threads (about 10 ms on the TB250)
            using tp_new_t = ggml_threadpool_t (*)(const ggml_threadpool_params *);
            using tp_set_t = void (*)(ggml_backend_t, ggml_threadpool_t);
            auto tp_new = (tp_new_t) ggml_backend_reg_get_proc_address(creg, "ggml_threadpool_new");
            auto tp_set = (tp_set_t) ggml_backend_reg_get_proc_address(creg, "ggml_backend_cpu_set_threadpool");
            if (st.use_kern) {
                st.kern_ok = st.kern.init(creg, std::max(0, st.n_threads - 1), st.kern_spin_us);
                if (!st.kern_ok) {
                    MOE_LOG("the ggml-cpu kernels are not reachable: the cold experts run as a ggml graph\n");
                }
            }
            if (tp_new != nullptr && tp_set != nullptr && st.n_threads > 1) {
                ggml_threadpool_params tpp;
                memset(&tpp, 0, sizeof(tpp));
                tpp.n_threads = st.n_threads;
                tpp.prio = GGML_SCHED_PRIO_NORMAL;
                tpp.poll = st.poll;
                tpp.strict_cpu = false;
                tpp.paused = false;
                ggml_threadpool_t tp = tp_new(&tpp);
                if (tp != nullptr) {
                    tp_set(st.cpu, tp);
                }
            }
        }
    }
    sync_classes(st);
}

void init_devices(state & st, int nu) {
    if (st.devs_inited) {
        return;
    }
    st.devs_inited = true;
    const int mc = main_class(st);
    for (const auto & sp : st.specs) {
        ggml_backend_dev_t d = find_ggml_device(sp.name.c_str());
        if (d == nullptr) {
            MOE_LOG("no ggml device %s (is its backend loaded?), skipped\n", sp.name.c_str());
            continue;
        }
        std::unique_ptr<pdev> p = make_pdev_ggml(d);
        if (!p) {
            MOE_LOG("cannot use %s, skipped\n", sp.name.c_str());
            continue;
        }
        dev_rec dr;
        dr.name = sp.name;
        dr.want_main = sp.main;
        dr.want_other = sp.other;
        dr.gated = sp.gated;
        dr.cls.resize(st.classes.size());
        size_t freeb = 0, totalb = 0;
        const bool have_mem = p->memory(&freeb, &totalb);
        // MiB left for slots: the first device keeps room for the compute buffers of llama.cpp's own later contexts and for the prefill graph, the others less
        long budget = have_mem ? (long) (freeb >> 20) - (st.devs.empty() ? st.reserve_mib : std::min<long>(st.reserve_mib, 96)) : 0;
        std::string line;
        for (size_t c = 0; c < st.classes.size(); ++c) {
            const cls_rec & cr = st.classes[c];
            if (cr.n_layers == 0) {
                continue;
            }
            const size_t slot_bytes = cr.b_gate + cr.b_up + cr.b_down;
            long want = (int) c == mc ? dr.want_main : dr.want_other;
            if (!p->supports_moe(cr.sh)) {
                MOE_LOG("%s cannot compute the MoE block of class %zu (%s/%s/%s)\n", dr.name.c_str(), c, ggml_type_name(cr.sh.t_gate),
                        ggml_type_name(cr.sh.t_up), ggml_type_name(cr.sh.t_down));
                continue;
            }
            const long cap = (long) cr.n_layers * (long) st.n_expert_max;
            // the streaming ring lives on the first device only
            const long ring = (st.devs.empty() && st.ring_main >= 2) ? (long) st.ring_main & ~1L : 0;
            if (want < 0) {
                want = std::max<long>(0, (long) ((double) budget * 1048576.0 / (double) slot_bytes) - ring);
            }
            want = std::min(want, cap);
            if (want <= 0 && ring <= 0) {
                continue;
            }
            // allocate; on failure try fewer slots
            moe_slots * s = nullptr;
            long n = want;
            while (true) {
                s = p->moe_create(cr.sh, (int) (n + ring));
                if (s != nullptr && ring > 0 && nu > 0 && !p->moe_reserve(s, 512, nu)) {
                    p->moe_free(s);   // the compute buffer of the prefill graph did not fit next to the slots
                    s = nullptr;
                }
                if (s != nullptr || n <= 0) {
                    break;
                }
                n = n * 9 / 10;
                if (n < 8) {
                    n = 0;
                }
            }
            if (s == nullptr) {
                MOE_LOG("%s: no room for slots of class %zu\n", dr.name.c_str(), c);
                continue;
            }
            dr.cls[c].s = s;
            dr.cls[c].n_slots = (int) n;
            dr.cls[c].n_ring = (int) ring;
            dr.cls[c].owner.assign((size_t) (n + ring), -1);   // the ring slots hold experts during a generation too (evict_ring)
            budget -= (long) ((double) (n + ring) * (double) slot_bytes / 1048576.0);
            char buf[160];
            snprintf(buf, sizeof(buf), " class %zu: %ld slots + %ld ring (%.0f MiB)", c, n, ring, (double) (n + ring) * (double) slot_bytes / 1048576.0);
            line += buf;
        }
        MOE_LOG("device %s (%s):%s\n", dr.name.c_str(), p->description().c_str(), line.empty() ? " no slots" : line.c_str());
        dr.p = std::move(p);
        st.devs.push_back(std::move(dr));
    }
    sync_classes(st);
}

// ---------------------------------------------------------------------------------------------------------------------------------------
// placement

const char * expert_ptr(const layer_rec & L, char kind, int e) {
    const ggml_tensor * t = kind == 'u' ? L.up : (kind == 'g' ? L.gate : L.down);
    return (const char *) t->data + (size_t) e * t->nb[2];
}

// Static placement: the experts of all layers by decreasing frequency, each into the first device (in the order of GGML_SGLX_MOE) that still has a
// free slot of the layer's class. Slots keep their expert when it stays on the same device; the others are (re)uploaded.
void plan(state & st) {
    const auto t0 = clk::now();
    if (st.devs.empty()) {
        return;
    }
    sync_classes(st);
    apply_prior(st, st.nu_seen);
    struct cand { double f; int layer; int e; };
    std::vector<cand> cands;
    for (size_t l = 0; l < st.layers.size(); ++l) {
        const layer_rec & L = st.layers[l];
        if (L.cls < 0) {
            continue;
        }
        for (int e = 0; e < (int) L.n_expert; ++e) {
            if (L.freq[(size_t) e] > 0.0) {
                cands.push_back({ L.freq[(size_t) e] * st.classes[(size_t) L.cls].cpu_cost, (int) l, e });
            }
        }
    }
    std::stable_sort(cands.begin(), cands.end(), [](const cand & a, const cand & b) { return a.f > b.f; });
    const size_t nd = st.devs.size();
    const size_t nc = st.classes.size();
    std::vector<std::vector<int>> room(nd, std::vector<int>(nc, 0));
    for (size_t j = 0; j < nd; ++j) {
        for (size_t c = 0; c < nc; ++c) {
            room[j][c] = st.devs[j].cls[c].n_slots + st.devs[j].cls[c].n_ring;   // decode may use the ring as cache
        }
    }
    // wanted[l][e] = device. A device other than the first costs the host a fixed time in every layer where it is used (a submit, a thread hand-over, a wait): a (device, layer)
    // group of experts that covers less than min_mass of the selections of its layer is not worth it; its slots go to the next candidates (a few rounds).
    std::vector<std::vector<int16_t>> want(st.layers.size());
    std::vector<std::vector<char>> allowed(nd, std::vector<char>(st.layers.size(), 1));
    std::vector<double> layer_mass(st.layers.size(), 0.0);
    for (size_t l = 0; l < st.layers.size(); ++l) {
        for (double f : st.layers[l].freq) {
            layer_mass[l] += f;
        }
    }
    for (int round = 0; round < 4; ++round) {
        for (size_t j = 0; j < nd; ++j) {
            for (size_t c = 0; c < nc; ++c) {
                room[j][c] = st.devs[j].cls[c].n_slots + st.devs[j].cls[c].n_ring;
            }
        }
        for (size_t l = 0; l < st.layers.size(); ++l) {
            want[l].assign((size_t) st.layers[l].n_expert, -1);
        }
        for (const cand & c : cands) {
            const int cl = st.layers[(size_t) c.layer].cls;
            for (size_t j = 0; j < nd; ++j) {
                if (allowed[j][(size_t) c.layer] && room[j][(size_t) cl] > 0) {
                    room[j][(size_t) cl]--;
                    want[(size_t) c.layer][(size_t) c.e] = (int16_t) j;
                    break;
                }
            }
        }
        if (st.min_mass <= 0.0 || nd < 2) {
            break;
        }
        bool changed = false;
        for (size_t j = 1; j < nd; ++j) {
            std::vector<double> mass(st.layers.size(), 0.0);
            std::vector<int> cnt(st.layers.size(), 0);
            for (size_t l = 0; l < st.layers.size(); ++l) {
                for (int e = 0; e < (int) st.layers[l].n_expert; ++e) {
                    if (want[l][(size_t) e] == (int16_t) j) {
                        mass[l] += st.layers[l].freq[(size_t) e];
                        ++cnt[l];
                    }
                }
            }
            for (size_t l = 0; l < st.layers.size(); ++l) {
                if (cnt[l] > 0 && allowed[j][l] && layer_mass[l] > 0.0 && mass[l] / layer_mass[l] < st.min_mass) {
                    allowed[j][l] = 0;
                    changed = true;
                }
            }
        }
        if (!changed) {
            break;
        }
    }
    // free the slots whose expert moves away, then fill
    size_t n_keep = 0, n_up = 0, n_free = 0;
    for (size_t j = 0; j < nd; ++j) {
        for (size_t c = 0; c < nc; ++c) {
            dev_cls & dc = st.devs[j].cls[c];
            for (int s = 0; s < dc.n_slots + dc.n_ring; ++s) {
                const int32_t o = dc.owner[(size_t) s];
                if (o < 0) {
                    continue;
                }
                const int l = (int) (o / (int32_t) st.n_expert_max), e = (int) (o % (int32_t) st.n_expert_max);
                if (want[(size_t) l][(size_t) e] != (int16_t) j) {
                    dc.owner[(size_t) s] = -1;
                    st.layers[(size_t) l].dev[(size_t) e] = -1;
                    st.layers[(size_t) l].slot[(size_t) e] = -1;
                    ++n_free;
                } else {
                    ++n_keep;
                }
            }
        }
    }
    std::vector<std::vector<int>> next_free(nd, std::vector<int>(nc, 0));
    struct up_job { int layer, e, slot, cls; };
    std::vector<std::vector<up_job>> jobs(nd);
    for (const cand & c : cands) {
        layer_rec & L = st.layers[(size_t) c.layer];
        const int16_t j = want[(size_t) c.layer][(size_t) c.e];
        if (j < 0 || L.dev[(size_t) c.e] == j) {
            continue;
        }
        dev_cls & dc = st.devs[(size_t) j].cls[(size_t) L.cls];
        int & nf = next_free[(size_t) j][(size_t) L.cls];
        while (nf < dc.n_slots + dc.n_ring && dc.owner[(size_t) nf] >= 0) {
            ++nf;
        }
        if (nf >= dc.n_slots + dc.n_ring) {
            continue; // cannot happen: the room was counted
        }
        dc.owner[(size_t) nf] = (int32_t) (c.layer * (int32_t) st.n_expert_max + c.e);
        L.dev[(size_t) c.e] = j;
        L.slot[(size_t) c.e] = nf;
        jobs[(size_t) j].push_back({ c.layer, c.e, nf, L.cls });
        ++n_up;
    }
    // the uploads of different devices run at the same time (each is a blocking copy over its own PCIe link)
    std::vector<std::thread> th;
    std::vector<char> failed(nd, 0);
    for (size_t j = 0; j < nd; ++j) {
        if (jobs[j].empty()) {
            continue;
        }
        th.emplace_back([&st, &jobs, &failed, j] {
            for (const up_job & u : jobs[j]) {
                layer_rec & L = st.layers[(size_t) u.layer];
                dev_cls & dc = st.devs[j].cls[(size_t) u.cls];
                if (!st.devs[j].p->moe_upload(dc.s, u.slot, expert_ptr(L, 'g', u.e), expert_ptr(L, 'u', u.e), expert_ptr(L, 'd', u.e))) {
                    failed[j] = 1;
                    return;
                }
            }
        });
    }
    for (auto & t : th) {
        t.join();
    }
    for (size_t j = 0; j < nd; ++j) {
        if (failed[j]) {
            MOE_LOG("upload to %s failed: its experts are computed on the cpu\n", st.devs[j].name.c_str());
            for (const up_job & u : jobs[j]) {
                layer_rec & L = st.layers[(size_t) u.layer];
                L.dev[(size_t) u.e] = -1;
                L.slot[(size_t) u.e] = -1;
                st.devs[j].cls[(size_t) u.cls].owner[(size_t) u.slot] = -1;
            }
        }
    }
    ++st.plans;
    st.tokens_since_plan = 0;
    st.plan_ms += ms_since(t0);
    if (st.profile || getenv("GGML_SGLX_MOE_VERBOSE") != nullptr) {
        // hit mass of the placement
        double tot = 0, hit = 0;
        for (const cand & c : cands) {
            tot += c.f;
            hit += st.layers[(size_t) c.layer].dev[(size_t) c.e] >= 0 ? c.f : 0.0;
        }
        MOE_LOG("placement %d: kept %zu, uploaded %zu, freed %zu in %.0f ms; the placed experts cover %.1f%% of the selections seen\n", st.plans, n_keep, n_up,
                n_free, ms_since(t0), tot > 0 ? 100.0 * hit / tot : 0.0);
    }
}

// ---------------------------------------------------------------------------------------------------------------------------------------
// statistics

void count(state & st, layer_rec & L, const int32_t * ids, int nu, int N) {
    for (int t = 0; t < N; ++t) {
        for (int k = 0; k < nu; ++k) {
            const int32_t e = ids[(size_t) t * (size_t) nu + (size_t) k];
            if (e >= 0 && e < L.n_expert) {
                L.freq[(size_t) e] += st.scale;
            }
        }
    }
}


// frequencies from an earlier run ("layer expert share" lines, the shares of a layer add up to 1): added once, before the first placement, as `prior_tokens` tokens of selections
void apply_prior(state & st, int nu) {
    if (st.prior_applied || st.stats_in.empty()) {
        return;
    }
    st.prior_applied = true;
    FILE * f = fopen(st.stats_in.c_str(), "r");
    if (f == nullptr) {
        MOE_LOG("cannot read %s\n", st.stats_in.c_str());
        return;
    }
    int il, e;
    double v;
    size_t n = 0;
    while (fscanf(f, "%d %d %lf", &il, &e, &v) == 3) {
        auto it = st.layer_by_il.find(il);
        if (it == st.layer_by_il.end()) {
            continue;
        }
        layer_rec & L = st.layers[(size_t) it->second];
        if (e >= 0 && e < L.n_expert) {
            L.freq[(size_t) e] += v * st.prior_tokens * (double) nu * st.scale;
            ++n;
        }
    }
    fclose(f);
    MOE_LOG("prior from %s: %zu values, worth %.0f tokens\n", st.stats_in.c_str(), n, st.prior_tokens);
}

void write_stats(state & st) {
    if (st.stats_out.empty()) {
        return;
    }
    FILE * f = fopen(st.stats_out.c_str(), "w");
    if (f == nullptr) {
        return;
    }
    for (const auto & L : st.layers) {
        if (L.cls < 0) {
            continue;
        }
        double sum = 0;
        for (double v : L.freq) { sum += v; }
        for (int e = 0; e < (int) L.n_expert; ++e) {
            fprintf(f, "%d %d %.6g\n", L.il, e, sum > 0 ? L.freq[(size_t) e] / sum : 0.0);
        }
    }
    fclose(f);
}

// A prompt (more than one token) is going to use the ring of the streaming device: the experts that a generation kept in the ring slots go back to the host
void evict_ring(state & st) {
    for (auto & d : st.devs) {
        for (size_t c = 0; c < d.cls.size(); ++c) {
            dev_cls & dc = d.cls[c];
            for (int s = dc.n_slots; s < dc.n_slots + dc.n_ring; ++s) {
                const int32_t o = dc.owner[(size_t) s];
                if (o < 0) {
                    continue;
                }
                const int l = (int) (o / (int32_t) st.n_expert_max), e = (int) (o % (int32_t) st.n_expert_max);
                st.layers[(size_t) l].dev[(size_t) e] = -1;
                st.layers[(size_t) l].slot[(size_t) e] = -1;
                dc.owner[(size_t) s] = -1;
            }
        }
    }
}

// a new pass over the layers begins (the layer number did not increase): decay, and the placement decisions
void begin_pass(state & st, int N) {
    ++st.passes;
    st.tokens_seen += N;
    st.tokens_since_plan += N;
    st.scale *= std::pow(2.0, (double) N / st.halflife);
    if (st.scale > 1e100) {
        for (auto & L : st.layers) {
            for (auto & f : L.freq) {
                f /= st.scale;
            }
        }
        st.scale = 1.0;
    }
}

// ---------------------------------------------------------------------------------------------------------------------------------------
// cpu execution

cpu_graph * cold_graph(state & st, int cls, int n) {
    auto & slot = st.cold[(size_t) cls][(size_t) n - 1];
    if (slot) {
        return slot.get();
    }
    const moe_shape & sh = st.classes[(size_t) cls].sh;
    auto g = std::make_unique<cpu_graph>();
    g->n = n;
    ggml_init_params ip = { ggml_tensor_overhead() * 24 + ggml_graph_overhead_custom(24, false) + 1024, nullptr, true };
    g->ctx = ggml_init(ip);
    static char dummy[64];
    g->wu = ggml_new_tensor_3d(g->ctx, sh.t_up, sh.n_embd, sh.n_ff, 128);
    g->wg = ggml_new_tensor_3d(g->ctx, sh.t_gate, sh.n_embd, sh.n_ff, 128);
    g->wd = ggml_new_tensor_3d(g->ctx, sh.t_down, sh.n_ff, sh.n_embd, 128);
    g->wu->data = g->wg->data = g->wd->data = dummy; // not allocated by the graph allocator, set before every run
    g->x   = ggml_new_tensor_3d(g->ctx, GGML_TYPE_F32, sh.n_embd, 1, 1);
    g->ids = ggml_new_tensor_2d(g->ctx, GGML_TYPE_I32, n, 1);
    ggml_set_input(g->x);
    ggml_set_input(g->ids);
    ggml_tensor * up = ggml_mul_mat_id(g->ctx, g->wu, g->x, g->ids);
    ggml_tensor * gt = ggml_mul_mat_id(g->ctx, g->wg, g->x, g->ids);
    ggml_tensor * h = ggml_swiglu_split(g->ctx, gt, up);
    g->y = ggml_mul_mat_id(g->ctx, g->wd, h, g->ids);
    ggml_set_output(g->y);
    g->gf = ggml_new_graph_custom(g->ctx, 24, false);
    ggml_build_forward_expand(g->gf, g->y);
    g->ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(st.cpu));
    if (!ggml_gallocr_reserve(g->ga, g->gf)) {
        MOE_LOG("cannot reserve the cpu compute buffer\n");
        return nullptr;
    }
    slot = std::move(g);
    return slot.get();
}

// the n cold experts `eids` for one token column x: out[:, i] (unweighted)
bool cold_decode(state & st, const layer_rec & L, const float * x, const int32_t * eids, int n, float * out) {
    if (st.kern_ok && st.kern.supports(L.gate->type, L.up->type, L.down->type)) {
        const auto tk = clk::now();
        const bool ok = st.kern.run(L.gate->type, L.up->type, L.down->type, L.up->ne[0], L.up->ne[1], (const char *) L.gate->data, L.gate->nb[2],
                                    (const char *) L.up->data, L.up->nb[2], (const char *) L.down->data, L.down->nb[2], x, eids, n, out);
        if (ok) {
            const double per = ms_since(tk) / (double) n;
            st.cold_ms[(size_t) L.cls] = 0.95 * st.cold_ms[(size_t) L.cls] + 0.05 * per;
            if (st.profile) {
                st.t_cold_compute.add(ms_since(tk));
            }
            return true;
        }
    }
    cpu_graph * g = cold_graph(st, L.cls, n);
    if (g == nullptr) {
        return false;
    }
    const auto t0 = clk::now();
    g->wu->data = L.up->data;
    g->wg->data = L.gate->data;
    g->wd->data = L.down->data;
    if (!ggml_gallocr_alloc_graph(g->ga, g->gf)) {
        return false;
    }
    const auto t1 = clk::now();
    ggml_backend_tensor_set(g->x, x, 0, (size_t) L.up->ne[0] * sizeof(float));
    ggml_backend_tensor_set(g->ids, eids, 0, (size_t) n * sizeof(int32_t));
    const auto t2 = clk::now();
    if (ggml_backend_graph_compute(st.cpu, g->gf) != GGML_STATUS_SUCCESS) {
        return false;
    }
    const auto t3 = clk::now();
    ggml_backend_tensor_get(g->y, out, 0, (size_t) L.up->ne[0] * (size_t) n * sizeof(float));
    if (st.profile) {
        st.t_cold_alloc.add(std::chrono::duration<double, std::milli>(t1 - t0).count());
        st.t_cold_set.add(std::chrono::duration<double, std::milli>(t2 - t1).count());
        st.t_cold_compute.add(std::chrono::duration<double, std::milli>(t3 - t2).count());
    }
    return true;
}

// P (token, expert) pairs on the cpu: x is [n_embd, N_x]; out [n_embd, P]
bool cold_pairs(state & st, const layer_rec & L, const float * x, int64_t N_x, const int32_t * eids, const int32_t * toks, int64_t P, float * out) {
    const moe_shape & sh = st.classes[(size_t) L.cls].sh;
    ggml_init_params ip = { ggml_tensor_overhead() * 24 + ggml_graph_overhead_custom(24, false) + 1024, nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    static char dummy[64];
    ggml_tensor * wu = ggml_new_tensor_3d(ctx, sh.t_up, sh.n_embd, sh.n_ff, L.n_expert);
    ggml_tensor * wg = ggml_new_tensor_3d(ctx, sh.t_gate, sh.n_embd, sh.n_ff, L.n_expert);
    ggml_tensor * wd = ggml_new_tensor_3d(ctx, sh.t_down, sh.n_ff, sh.n_embd, L.n_expert);
    wu->data = L.up->data;
    wg->data = L.gate->data;
    wd->data = L.down->data;
    (void) dummy;
    ggml_tensor * xt  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, sh.n_embd, N_x);
    ggml_tensor * tok = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, P);
    ggml_tensor * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 1, P);
    ggml_set_input(xt);
    ggml_set_input(tok);
    ggml_set_input(ids);
    ggml_tensor * x3 = ggml_reshape_3d(ctx, ggml_get_rows(ctx, xt, tok), sh.n_embd, 1, P);
    ggml_tensor * up = ggml_mul_mat_id(ctx, wu, x3, ids);
    ggml_tensor * gt = ggml_mul_mat_id(ctx, wg, x3, ids);
    ggml_tensor * h = ggml_swiglu_split(ctx, gt, up);
    ggml_tensor * y = ggml_mul_mat_id(ctx, wd, h, ids);
    ggml_set_output(y);
    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 24, false);
    ggml_build_forward_expand(gf, y);
    ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(st.cpu));
    bool ok = ggml_gallocr_alloc_graph(ga, gf);
    if (ok) {
        ggml_backend_tensor_set(xt, x, 0, (size_t) sh.n_embd * (size_t) N_x * sizeof(float));
        ggml_backend_tensor_set(tok, toks, 0, (size_t) P * sizeof(int32_t));
        ggml_backend_tensor_set(ids, eids, 0, (size_t) P * sizeof(int32_t));
        ok = ggml_backend_graph_compute(st.cpu, gf) == GGML_STATUS_SUCCESS;
        if (ok) {
            ggml_backend_tensor_get(y, out, 0, (size_t) sh.n_embd * (size_t) P * sizeof(float));
        }
    }
    ggml_gallocr_free(ga);
    ggml_free(ctx);
    return ok;
}

// any node with ggml-cpu (the tensors are in host memory)
bool exec_cpu_node(state & st, ggml_cgraph * g, int i) {
    if (st.cpu == nullptr) {
        return false;
    }
    ggml_cgraph v = *g;
    v.size = 0;
    v.n_nodes = 1;
    v.n_leafs = 0;
    v.nodes = g->nodes + i;
    v.grads = nullptr;
    v.grad_accs = nullptr;
    v.leafs = nullptr;
    return ggml_backend_graph_compute(st.cpu, &v) == GGML_STATUS_SUCCESS;
}

// ---------------------------------------------------------------------------------------------------------------------------------------
// the block

bool is_mmid(const ggml_tensor * t) {
    return t != nullptr && t->op == GGML_OP_MUL_MAT_ID;
}

bool weight_ours(const ggml_tensor * w) {
    if (w == nullptr) {
        return false;
    }
    ggml_backend_buffer_t b = w->view_src != nullptr ? w->view_src->buffer : w->buffer;
    // no buffer yet: test-backend-ops asks before it allocates the tensors (the weights of a model always have one)
    return S != nullptr && (b == nullptr || S->h.owned(b));
}

bool mmid_ours(const ggml_tensor * t) {
    return is_mmid(t) && weight_ours(t->src[0]);
}

// find the block that starts at the MUL_MAT_ID node i: its SWIGLU and the down projection (all in this graph)
bool find_block(ggml_cgraph * g, int i, block & b) {
    ggml_tensor * a = g->nodes[i];
    ggml_tensor * glu = nullptr;
    int jglu = -1;
    for (int j = i + 1; j < g->n_nodes && j < i + 16; ++j) {
        ggml_tensor * n = g->nodes[j];
        if (n->op == GGML_OP_GLU && ggml_get_glu_op(n) == GGML_GLU_OP_SWIGLU && (n->src[0] == a || n->src[1] == a)) {
            glu = n;
            jglu = j;
            break;
        }
    }
    if (glu == nullptr) {
        return false;
    }
    ggml_tensor * gate = glu->src[0];
    ggml_tensor * up = glu->src[1];
    if (!mmid_ours(gate) || !mmid_ours(up) || gate->src[1] != up->src[1] || gate->src[2] != up->src[2]) {
        return false;
    }
    ggml_tensor * down = nullptr;
    for (int j = jglu + 1; j < g->n_nodes && j < jglu + 16; ++j) {
        ggml_tensor * n = g->nodes[j];
        if (is_mmid(n) && n->src[1] == glu && n->src[2] == up->src[2]) {
            down = n;
            b.jdown = j;
            break;
        }
    }
    if (down == nullptr || !weight_ours(down->src[0])) {
        return false;
    }
    // the partner must be a later node of this graph (it has not run yet)
    bool partner_later = false;
    for (int j = i + 1; j < jglu; ++j) {
        partner_later = partner_later || g->nodes[j] == (a == up ? gate : up);
    }
    if (!partner_later) {
        return false;
    }
    b.up = up; b.gate = gate; b.glu = glu; b.down = down;
    return true;
}


// the MUL and the ADD chain after the down projection at node index `jd` (see block::mul); false when the graph is shaped differently
bool find_tail(ggml_cgraph * g, int jd, block & b) {
    const int U = (int) b.down->ne[1];
    if (U < 2) {
        return false;
    }
    int jm = -1;
    for (int j = jd + 1; j < g->n_nodes && j < jd + 24; ++j) {
        ggml_tensor * n = g->nodes[j];
        if (n->op == GGML_OP_MUL && n->src[0] == b.down) {
            b.mul = n;
            jm = j;
            break;
        }
    }
    if (jm < 0 || b.mul->src[1] == nullptr || b.mul->src[1]->type != GGML_TYPE_F32 || b.mul->src[1]->ne[0] != 1 || b.mul->src[1]->ne[1] != U) {
        b.mul = nullptr;
        return false;
    }
    auto is_view = [&](const ggml_tensor * t) { return t != nullptr && t->op == GGML_OP_VIEW && (t->view_src == b.mul || t->src[0] == b.mul); };
    ggml_tensor * prev = nullptr;
    b.adds.clear();
    for (int j = jm + 1; j < g->n_nodes && (int) b.adds.size() < U - 1; ++j) {
        ggml_tensor * n = g->nodes[j];
        if (n->op != GGML_OP_ADD) {
            continue;
        }
        const bool ok = prev == nullptr ? (is_view(n->src[0]) && is_view(n->src[1])) : (n->src[0] == prev && is_view(n->src[1]));
        if (!ok) {
            continue;
        }
        b.adds.push_back(n);
        prev = n;
    }
    if ((int) b.adds.size() != U - 1) {
        b.mul = nullptr;
        b.adds.clear();
        return false;
    }
    b.fin = prev;
    const ggml_tensor * f = b.fin;
    if (f->type != GGML_TYPE_F32 || f->ne[0] != b.down->ne[0] || f->ne[1] != b.down->ne[2] || f->nb[0] != sizeof(float) || f->nb[1] != (size_t) f->ne[0] * sizeof(float)) {
        b.mul = nullptr; b.fin = nullptr; b.adds.clear();
        return false;
    }
    return true;
}

bool block_shapes_ok(const block & b) {
    const ggml_tensor * x = b.up->src[1];
    const ggml_tensor * ids = b.up->src[2];
    const ggml_tensor * y = b.down;
    const ggml_tensor * wu = b.up->src[0], * wg = b.gate->src[0], * wd = b.down->src[0];
    if (x->type != GGML_TYPE_F32 || x->ne[1] != 1 || x->ne[3] != 1 || x->nb[0] != sizeof(float)) {
        return false;
    }
    if (ids->type != GGML_TYPE_I32 || ids->nb[0] != sizeof(int32_t) || ids->ne[2] != 1 || ids->ne[1] != x->ne[2]) {
        return false;
    }
    if (y->type != GGML_TYPE_F32 || y->ne[0] != wd->ne[1] || y->ne[1] != ids->ne[0] || y->ne[2] != ids->ne[1] || y->nb[0] != sizeof(float) ||
        y->nb[1] != (size_t) y->ne[0] * sizeof(float) || y->nb[2] != y->nb[1] * (size_t) y->ne[1]) {
        return false;
    }
    if (wu->ne[0] != x->ne[0] || wg->ne[0] != x->ne[0] || wd->ne[0] != wu->ne[1] || wd->ne[1] != x->ne[0] || wu->ne[3] != 1) {
        return false;
    }
    return true;
}

// the layer record of a block (registered at load by name, or now)
int layer_of_block(state & st, const block & b) {
    ggml_tensor * wu = b.up->src[0], * wg = b.gate->src[0], * wd = b.down->src[0];
    auto it = st.layer_of.find(wu);
    if (it != st.layer_of.end() && st.layers[(size_t) it->second].down == wd && st.layers[(size_t) it->second].gate == wg) {
        return it->second;
    }
    // not registered by name (a test, or the tensors were not seen at load): one record per weight triple
    for (size_t l = 0; l < st.layers.size(); ++l) {
        if (st.layers[l].up == wu && st.layers[l].gate == wg && st.layers[l].down == wd) {
            return (int) l;
        }
    }
    const int idx = (int) st.layers.size();
    st.layers.emplace_back();
    layer_rec & L = st.layers.back();
    L.il = 100000 + idx;
    L.up = wu; L.gate = wg; L.down = wd;
    st.layer_of[wu] = idx;
    complete_layer(st, idx);
    return idx;
}

void read_ids(const ggml_tensor * ids, int nu, int N, std::vector<int32_t> & out) {
    out.resize((size_t) nu * (size_t) N);
    for (int t = 0; t < N; ++t) {
        memcpy(out.data() + (size_t) t * (size_t) nu, (const char *) ids->data + (size_t) t * ids->nb[1], (size_t) nu * sizeof(int32_t));
    }
}

std::vector<int32_t> g_ids;
std::vector<float>   g_cold_out;

// one token (N == 1): the experts of the slots go to the GPUs and the cpu at the same time
bool run_decode(state & st, layer_rec & L, const block & b, const int32_t * ids, int nu) {
    const auto t0 = clk::now();
    const int K = (int) L.up->ne[0];
    const float * x = (const float *) b.up->src[1]->data;
    float * y = (float *) b.down->data;
    const size_t nd = st.devs.size();
    std::vector<int> n_dev(nd, 0);
    int cold_k[16], cold_e[16], n_cold = 0;
    // per device: the slots of its experts of this layer
    for (size_t j = 0; j < nd; ++j) {
        st.devs[j].ps.clear();
    }
    std::vector<std::vector<int>> kslot(nd), kexp(nd);
    for (int k = 0; k < nu; ++k) {
        const int e = ids[k];
        if (e < 0 || e >= L.n_expert) {
            return false;
        }
        const int16_t j = L.dev[(size_t) e];
        if (j >= 0) {
            st.devs[(size_t) j].ps.push_back(L.slot[(size_t) e]);
            kslot[(size_t) j].push_back(k);
            kexp[(size_t) j].push_back(e);
        } else if (n_cold < 16) {
            cold_k[n_cold] = k;
            cold_e[n_cold] = e;
            ++n_cold;
        } else {
            return false;
        }
    }
    // a medium device is used only when the cpu would be busy for longer than the device needs (it would otherwise lengthen the layer)
    for (size_t j = 0; j < nd; ++j) {
        dev_rec & dr = st.devs[j];
        if (!dr.gated || dr.ps.empty()) {
            continue;
        }
        const double est_cpu = (double) dr.ps.size() * st.cold_ms[(size_t) L.cls];
        const bool explore = (++dr.gate_tick % 64) == 0;    // refresh the latency estimate now and then
        if (est_cpu > dr.lat_ms || explore) {
            ++dr.gate_used;
            continue;
        }
        ++dr.gate_skipped;
        for (size_t i = 0; i < kslot[j].size() && n_cold < 16; ++i) {
            cold_k[n_cold] = kslot[j][i];
            cold_e[n_cold] = kexp[j][i];
            ++n_cold;
        }
        dr.ps.clear();
        kslot[j].clear();
    }
    int used = 0;
    // GPUs first (they work while the cpu does the cold experts), the slowest first is not needed: one host thread submits all
    bool ok = true;
    const auto ts = clk::now();
    std::vector<size_t> submitted;
    for (size_t j = 0; j < nd; ++j) {
        dev_rec & dr = st.devs[j];
        const int n = (int) dr.ps.size();
        if (n == 0) {
            continue;
        }
        if (n > 8) {
            return false; // more than 8 ids for one token on one device: not supported by the pdev graphs
        }
        dr.pt.assign((size_t) n, 0);
        if (dr.out.size() < (size_t) K * 8) {
            dr.out.resize((size_t) K * 8);
        }
        const auto t1 = clk::now();
        ok = dr.p->moe_submit(dr.cls[(size_t) L.cls].s, x, 1, dr.ps.data(), dr.pt.data(), n, dr.out.data()) && ok;
        dr.t_submit.add(ms_since(t1));
        submitted.push_back(j);
        ++used;
    }
    const double t_sub = ms_since(ts);
    // the cold experts on the cpu
    const auto tc = clk::now();
    if (n_cold > 0) {
        if (g_cold_out.size() < (size_t) K * 16) {
            g_cold_out.resize((size_t) K * 16);
        }
        if (!cold_decode(st, L, x, cold_e, n_cold, g_cold_out.data())) {
            MOE_LOG("cold experts of layer %d failed\n", L.il);
            ok = false;
        }
    }
    const double t_cold = ms_since(tc);
    const auto tw = clk::now();
    for (size_t j : submitted) {
        const auto t1 = clk::now();
        ok = st.devs[j].p->wait() && ok;
        st.devs[j].t_wait.add(ms_since(t1));
    }
    const double t_wait = ms_since(tw);
    for (size_t j : submitted) {
        if (st.devs[j].gated) {
            const double l = st.devs[j].p->moe_last_ms();
            if (l > 0.0) {
                st.devs[j].lat_ms = 0.9 * st.devs[j].lat_ms + 0.1 * l;
            }
        }
    }
    if (!ok) {
        return false;
    }
    // scatter, unweighted, to y[:, k]
    for (size_t j : submitted) {
        const std::vector<int> & ks = kslot[j];
        for (size_t i = 0; i < ks.size(); ++i) {
            memcpy(y + (size_t) ks[i] * (size_t) K, st.devs[j].out.data() + i * (size_t) K, (size_t) K * sizeof(float));
        }
    }
    for (int i = 0; i < n_cold; ++i) {
        memcpy(y + (size_t) cold_k[i] * (size_t) K, g_cold_out.data() + (size_t) i * (size_t) K, (size_t) K * sizeof(float));
    }
    if (st.profile) {
        st.t_dec.add(ms_since(t0));
        st.t_dec_cold.add(t_cold);
        st.t_dec_gpu_sub.add(t_sub);
        st.t_dec_gpu_wait.add(t_wait);
        st.n_pairs_cold += n_cold;
        st.n_pairs_gpu += nu - n_cold;
        st.gpu_used_sum += used;
        ++st.dec_blocks;
    }
    return true;
}


// ---- prefill with the weighted sum on the streaming device (moe_prefill) ----

struct overlap_ctx {
    std::function<bool()> fn;
    bool ok = true;
};
void overlap_tramp(void * c) {
    auto * o = (overlap_ctx *) c;
    o->ok = o->fn();
}

// A prefill block when the MUL / ADD tail is known (block::fin): the device that streams (the first one) computes, for ONE graph, every pair that is on it or that it streams in,
// and returns the weighted sums [n_embd, tokens]; the other devices and the cpu return pair outputs, which are weighted and added here in the order of k. `fin` gets the result.
bool run_prefill_fused(state & st, layer_rec & L, block & b, const int32_t * ids, int nu, int N, int sdev) {
    const auto t0 = clk::now();
    const int K = (int) L.up->ne[0];
    const ggml_tensor * xt = b.up->src[1];
    const ggml_tensor * wt = b.mul->src[1];
    float * fin = (float *) b.fin->data;
    const size_t nd = st.devs.size();
    dev_rec & sd = st.devs[(size_t) sdev];
    dev_cls & dc = sd.cls[(size_t) L.cls];
    const int zero = sd.p->moe_zero_slot(dc.s);
    if (zero < 0) {
        return false;
    }
    auto w_at = [&](int k, int t) -> float { return *(const float *) ((const char *) wt->data + (size_t) k * wt->nb[1] + (size_t) t * wt->nb[2]); };
    const int NB = 512;
    for (int t0b = 0; t0b < N; t0b += NB) {
        const int nt = std::min(NB, N - t0b);
        const float * xs;
        if (xt->nb[2] == (size_t) K * sizeof(float)) {
            xs = (const float *) xt->data + (size_t) t0b * (size_t) K;
        } else {
            st.xg.resize((size_t) K * (size_t) nt);
            for (int t = 0; t < nt; ++t) {
                memcpy(st.xg.data() + (size_t) t * (size_t) K, (const char *) xt->data + (size_t) (t0b + t) * xt->nb[2], (size_t) K * sizeof(float));
            }
            xs = st.xg.data();
        }
        // classify the pairs
        std::vector<int32_t> ids_rx((size_t) nu * (size_t) nt, zero);
        std::vector<float>   w_rx((size_t) nu * (size_t) nt, 0.0f);
        struct work { std::vector<int32_t> slot, tok, k; };
        std::vector<work> w(nd);
        std::vector<int32_t> c_e, c_tok, c_k;
        for (int t = 0; t < nt; ++t) {
            for (int k = 0; k < nu; ++k) {
                const int e = ids[(size_t) (t0b + t) * (size_t) nu + (size_t) k];
                if (e < 0 || e >= L.n_expert) {
                    return false;
                }
                const int16_t j = L.dev[(size_t) e];
                if (j == (int16_t) sdev) {
                    ids_rx[(size_t) k + (size_t) nu * (size_t) t] = L.slot[(size_t) e];
                    w_rx[(size_t) k + (size_t) nu * (size_t) t] = w_at(k, t0b + t);
                } else if (j >= 0) {
                    w[(size_t) j].slot.push_back(L.slot[(size_t) e]);
                    w[(size_t) j].tok.push_back(t);
                    w[(size_t) j].k.push_back(k);
                } else {
                    c_e.push_back(e);
                    c_tok.push_back(t);
                    c_k.push_back(k);
                }
            }
        }
        // the unplaced experts: more than stream_tau pairs -> through the ring of the streaming device, the rest on the cpu
        std::vector<int> cnt((size_t) L.n_expert, 0), sidx((size_t) L.n_expert, -1);
        for (int32_t e : c_e) {
            ++cnt[(size_t) e];
        }
        std::vector<int32_t> s_exp;
        for (int e = 0; e < (int) L.n_expert; ++e) {
            if (cnt[(size_t) e] > st.stream_tau) {
                sidx[(size_t) e] = (int) s_exp.size();
                s_exp.push_back(e);
            }
        }
        std::vector<int32_t> u_e, u_tok, u_k;          // pairs for the cpu
        std::vector<int32_t> s_pair_t, s_pair_k, s_pair_i;   // pairs of the streamed experts
        for (size_t i = 0; i < c_e.size(); ++i) {
            const int ix = sidx[(size_t) c_e[i]];
            if (ix >= 0) {
                s_pair_t.push_back(c_tok[i]);
                s_pair_k.push_back(c_k[i]);
                s_pair_i.push_back(ix);
            } else {
                u_e.push_back(c_e[i]);
                u_tok.push_back(c_tok[i]);
                u_k.push_back(c_k[i]);
            }
        }
        // the host side: the other devices and the cpu (their rows are kept for the final sum)
        std::deque<std::vector<float>> keep;
        std::vector<const float *> hostp((size_t) nu * (size_t) nt, nullptr);
        auto host_work = [&]() -> bool {
            std::vector<size_t> pos(nd, 0);
            size_t cold_pos = 0;
            for (;;) {
                std::vector<size_t> sub, cnt_(nd, 0);
                std::vector<std::vector<float> *> outs(nd, nullptr);
                for (size_t j = 0; j < nd; ++j) {
                    if ((int) j == sdev) {
                        continue;
                    }
                    const size_t left = w[j].slot.size() - pos[j];
                    if (left == 0) {
                        continue;
                    }
                    const size_t n = std::min<size_t>(left, 1024);
                    keep.emplace_back((size_t) K * n);
                    outs[j] = &keep.back();
                    if (!st.devs[j].p->moe_submit(st.devs[j].cls[(size_t) L.cls].s, xs, nt, w[j].slot.data() + pos[j], w[j].tok.data() + pos[j], (int) n, outs[j]->data())) {
                        return false;
                    }
                    cnt_[j] = n;
                    sub.push_back(j);
                }
                size_t cold_n = 0;
                const float * cold_rows = nullptr;
                if (cold_pos < u_e.size()) {
                    cold_n = std::min<size_t>(u_e.size() - cold_pos, 1024);
                    keep.emplace_back((size_t) K * cold_n);
                    const auto tc = clk::now();
                    if (!cold_pairs(st, L, xs, nt, u_e.data() + cold_pos, u_tok.data() + cold_pos, (int64_t) cold_n, keep.back().data())) {
                        return false;
                    }
                    if (st.profile) {
                        st.t_pre_cold.add(ms_since(tc));
                    }
                    cold_rows = keep.back().data();
                }
                bool ok = true;
                for (size_t j : sub) {
                    ok = st.devs[j].p->wait() && ok;
                }
                if (!ok) {
                    return false;
                }
                for (size_t j : sub) {
                    for (size_t i = 0; i < cnt_[j]; ++i) {
                        hostp[(size_t) w[j].k[pos[j] + i] + (size_t) nu * (size_t) w[j].tok[pos[j] + i]] = outs[j]->data() + i * (size_t) K;
                    }
                    pos[j] += cnt_[j];
                }
                for (size_t i = 0; i < cold_n; ++i) {
                    hostp[(size_t) u_k[cold_pos + i] + (size_t) nu * (size_t) u_tok[cold_pos + i]] = cold_rows + i * (size_t) K;
                }
                cold_pos += cold_n;
                if (sub.empty() && cold_n == 0) {
                    break;
                }
            }
            return true;
        };
        // rounds on the streaming device: the resident pairs with the first group of streamed experts, then the others
        std::vector<float> acc((size_t) K * (size_t) nt, 0.0f), acc_r((size_t) K * (size_t) nt);
        const int R = dc.n_ring;
        const int n_rounds = std::max(1, (int) ((s_exp.size() + (size_t) R - 1) / (size_t) R));
        bool host_done = false;
        for (int r = 0; r < n_rounds; ++r) {
            const int e0 = r * R, e1 = std::min((int) s_exp.size(), e0 + R);
            std::vector<int32_t> ids_r;
            std::vector<float> w_r;
            if (r == 0) {
                ids_r = ids_rx;
                w_r = w_rx;
            } else {
                ids_r.assign((size_t) nu * (size_t) nt, zero);
                w_r.assign((size_t) nu * (size_t) nt, 0.0f);
            }
            std::vector<moe_src> src((size_t) std::max(0, e1 - e0));
            for (int i = e0; i < e1; ++i) {
                src[(size_t) (i - e0)].gate = expert_ptr(L, 'g', s_exp[(size_t) i]);
                src[(size_t) (i - e0)].up = expert_ptr(L, 'u', s_exp[(size_t) i]);
                src[(size_t) (i - e0)].down = expert_ptr(L, 'd', s_exp[(size_t) i]);
            }
            for (size_t q = 0; q < s_pair_i.size(); ++q) {
                const int i = s_pair_i[q];
                if (i >= e0 && i < e1) {
                    ids_r[(size_t) s_pair_k[q] + (size_t) nu * (size_t) s_pair_t[q]] = dc.n_slots + (i - e0);
                    w_r[(size_t) s_pair_k[q] + (size_t) nu * (size_t) s_pair_t[q]] = w_at(s_pair_k[q], t0b + s_pair_t[q]);
                }
            }
            overlap_ctx oc;
            oc.fn = [&]() { host_done = true; return host_work(); };
            const auto ts = clk::now();
            if (!sd.p->moe_prefill(dc.s, dc.n_slots, src.data(), (int) src.size(), xs, nt, nu, ids_r.data(), w_r.data(), acc_r.data(), r == 0 ? overlap_tramp : nullptr, &oc)) {
                MOE_LOG("prefill on %s failed\n", sd.name.c_str());
                return false;
            }
            if (st.profile) {
                st.t_pre_stream.add(ms_since(ts));
                st.n_stream_experts += (double) (e1 - e0);
            }
            if (r == 0 && !oc.ok) {
                return false;
            }
            for (size_t i = 0; i < acc.size(); ++i) {
                acc[i] += acc_r[i];
            }
        }
        if (!host_done && !host_work()) {
            return false;
        }
        if (getenv("GGML_SGLX_MOE_DEBUG") != nullptr) {
            for (int t = 0; t < nt; ++t) {
                double sum = 0;
                for (int i = 0; i < K; ++i) { sum += fabs(acc[(size_t) t * (size_t) K + (size_t) i]); }
                if (sum == 0.0) {
                    fprintf(stderr, "SGLX MoE debug: layer %d token %d has a zero sum; pairs:", L.il, t);
                    for (int k = 0; k < nu; ++k) {
                        const int e = ids[(size_t) (t0b + t) * (size_t) nu + (size_t) k];
                        fprintf(stderr, " [e%d dev%d slot%d sidx%d rx_id %d w %.3g]", e, (int) L.dev[(size_t) e], L.slot[(size_t) e], sidx[(size_t) e], ids_rx[(size_t) k + (size_t) nu * (size_t) t], w_rx[(size_t) k + (size_t) nu * (size_t) t]);
                    }
                    fprintf(stderr, " (R=%d rounds=%d s_exp=%zu)\n", R, n_rounds, s_exp.size());
                }
            }
        }
        // the sum: the streaming device's part, then the host pairs in the order of k
        for (int t = 0; t < nt; ++t) {
            float * o = fin + (size_t) (t0b + t) * (size_t) K;
            memcpy(o, acc.data() + (size_t) t * (size_t) K, (size_t) K * sizeof(float));
            for (int k = 0; k < nu; ++k) {
                const float * y = hostp[(size_t) k + (size_t) nu * (size_t) t];
                if (y != nullptr) {
                    const float wv = w_at(k, t0b + t);
                    for (int i = 0; i < K; ++i) {
                        const float v = wv * y[i];
                        o[i] += v;
                    }
                }
            }
        }
        st.n_pre_gpu += (double) (nu * nt) - (double) (u_e.size());
        st.n_pre_cold += (double) (u_e.size());
    }
    if (st.profile) {
        st.t_pre.add(ms_since(t0));
    }
    return true;
}

// N > 1 tokens: pairs (token, slot) are grouped by where the expert lives
bool run_prefill(state & st, layer_rec & L, block & b, const int32_t * ids, int nu, int N) {
    const auto t0 = clk::now();
    const int K = (int) L.up->ne[0];
    const ggml_tensor * xt = b.up->src[1];
    float * y = (float *) b.down->data;
    const size_t nd = st.devs.size();
    const int NB = 1024; // tokens per block (the pdev graphs take at most 1024 columns and 1024 pairs)
    // the device that streams the experts that are in no slot through its ring: the first one that has a ring for this class
    int sdev = -1;
    for (size_t j = 0; j < nd; ++j) {
        if (st.devs[j].cls[(size_t) L.cls].s != nullptr && st.devs[j].cls[(size_t) L.cls].n_ring >= 2) {
            sdev = (int) j;
            break;
        }
    }
    if (sdev >= 0 && b.fin != nullptr && st.fused_prefill) {
        if (run_prefill_fused(st, L, b, ids, nu, N, sdev)) {
            b.tail_done = true;
            if (getenv("GGML_SGLX_MOE_DEBUG") != nullptr) {
                // compare with the pair-output path (debug)
                st.fused_prefill = false;
                block b2 = b;
                b2.fin = nullptr;
                const bool ok2 = run_prefill(st, L, b2, ids, nu, N);
                st.fused_prefill = true;
                if (ok2) {
                    const ggml_tensor * wt = b.mul->src[1];
                    double worst = 0;
                    int bad = 0;
                    for (int t = 0; t < N; ++t) {
                        std::vector<float> ref((size_t) K, 0.0f);
                        for (int k = 0; k < nu; ++k) {
                            const float wv = *(const float *) ((const char *) wt->data + (size_t) k * wt->nb[1] + (size_t) t * wt->nb[2]);
                            const float * yk = (const float *) b.down->data + ((size_t) k + (size_t) nu * (size_t) t) * (size_t) K;
                            for (int i = 0; i < K; ++i) { ref[(size_t) i] += wv * yk[i]; }
                        }
                        double mx = 0, den = 0;
                        const float * f = (const float *) b.fin->data + (size_t) t * (size_t) K;
                        for (int i = 0; i < K; ++i) { mx = std::max(mx, (double) fabsf(f[i] - ref[(size_t) i])); den = std::max(den, (double) fabsf(ref[(size_t) i])); }
                        if (mx > 1e-2 * (den + 1e-6)) { ++bad; if (bad <= 3) {
                            fprintf(stderr, "SGLX MoE debug: layer %d token %d: max diff %.4g (ref max %.4g): fused %.4g %.4g %.4g, ref %.4g %.4g %.4g\n", L.il, t, mx, den, f[0], f[1], f[2], ref[0], ref[1], ref[2]);
                            {
                                // the truth: the cpu
                                std::vector<int32_t> ee, tt;
                                for (int k = 0; k < nu; ++k) { ee.push_back(ids[(size_t) t * (size_t) nu + (size_t) k]); tt.push_back(0); }
                                std::vector<float> cy((size_t) K * (size_t) nu), tsum((size_t) K, 0.0f);
                                const float * xcol = (const float *) ((const char *) xt->data + (size_t) t * xt->nb[2]);
                                cold_pairs(st, L, xcol, 1, ee.data(), tt.data(), nu, cy.data());
                                for (int k = 0; k < nu; ++k) {
                                    const float * yk = (const float *) b.down->data + ((size_t) k + (size_t) nu * (size_t) t) * (size_t) K;
                                    const float wv = *(const float *) ((const char *) wt->data + (size_t) k * wt->nb[1] + (size_t) t * wt->nb[2]);
                                    for (int i = 0; i < K; ++i) { tsum[(size_t) i] += wv * cy[(size_t) k * (size_t) K + (size_t) i]; }
                                    fprintf(stderr, "    k=%d e=%d w=%.4g old y0=%.4g cpu y0=%.4g\n", k, ee[(size_t) k], wv, yk[0], cy[(size_t) k * (size_t) K]);
                                }
                                fprintf(stderr, "    cpu sum %.4g %.4g %.4g\n", tsum[0], tsum[1], tsum[2]);
                            }
                        } }
                        worst = std::max(worst, mx / (den + 1e-6));
                    }
                    fprintf(stderr, "SGLX MoE debug: layer %d N=%d: %d tokens differ, worst relative %.3g\n", L.il, N, bad, worst);
                }
            }
            return true;
        }
        MOE_LOG("fused prefill failed, falling back\n");
        return false;
    }
    for (int t0b = 0; t0b < N; t0b += NB) {
        const int nt = std::min(NB, N - t0b);
        // x columns of the block, contiguous
        const float * xs;
        if (xt->nb[2] == (size_t) K * sizeof(float)) {
            xs = (const float *) xt->data + (size_t) t0b * (size_t) K;
        } else {
            st.xg.resize((size_t) K * (size_t) nt);
            for (int t = 0; t < nt; ++t) {
                memcpy(st.xg.data() + (size_t) t * (size_t) K, (const char *) xt->data + (size_t) (t0b + t) * xt->nb[2], (size_t) K * sizeof(float));
            }
            xs = st.xg.data();
        }
        struct work { std::vector<int32_t> slot, tok, k; };
        std::vector<work> w(nd);
        std::vector<int32_t> c_e, c_tok, c_k;     // pairs on experts that are in no slot
        for (int t = 0; t < nt; ++t) {
            for (int k = 0; k < nu; ++k) {
                const int e = ids[(size_t) (t0b + t) * (size_t) nu + (size_t) k];
                if (e < 0 || e >= L.n_expert) {
                    return false;
                }
                const int16_t j = L.dev[(size_t) e];
                if (j >= 0) {
                    w[(size_t) j].slot.push_back(L.slot[(size_t) e]);
                    w[(size_t) j].tok.push_back(t);
                    w[(size_t) j].k.push_back(k);
                } else {
                    c_e.push_back(e);
                    c_tok.push_back(t);
                    c_k.push_back(k);
                }
            }
        }
        auto put = [&](int t, int k, const float * src) {
            memcpy(y + (size_t) k * (size_t) K + (size_t) (t0b + t) * (size_t) nu * (size_t) K, src, (size_t) K * sizeof(float));
        };
        // split the unplaced pairs: experts with more than stream_tau pairs go through the ring, the rest to the cpu
        std::vector<int32_t> s_exp_id, s_pair_exp, s_pair_tok, s_pair_k;   // streamed experts, and their pairs (index into s_exp_id)
        std::vector<int32_t> u_e, u_tok, u_k;                              // pairs for the cpu
        if (sdev >= 0 && !c_e.empty()) {
            std::vector<int> cnt((size_t) L.n_expert, 0), idx((size_t) L.n_expert, -1);
            for (int32_t e : c_e) {
                ++cnt[(size_t) e];
            }
            for (int e = 0; e < (int) L.n_expert; ++e) {
                if (cnt[(size_t) e] > st.stream_tau) {
                    idx[(size_t) e] = (int) s_exp_id.size();
                    s_exp_id.push_back(e);
                }
            }
            for (size_t i = 0; i < c_e.size(); ++i) {
                const int ix = idx[(size_t) c_e[i]];
                if (ix >= 0) {
                    s_pair_exp.push_back(ix);
                    s_pair_tok.push_back(c_tok[i]);
                    s_pair_k.push_back(c_k[i]);
                } else {
                    u_e.push_back(c_e[i]);
                    u_tok.push_back(c_tok[i]);
                    u_k.push_back(c_k[i]);
                }
            }
        } else {
            u_e = c_e; u_tok = c_tok; u_k = c_k;
        }
        // rounds of at most 1024 pairs per device; the cpu takes the rest in the same rounds
        std::vector<size_t> pos(nd, 0);
        std::vector<std::vector<float>> outs(nd);
        size_t cold_pos = 0;
        for (;;) {
            std::vector<size_t> sub;
            std::vector<size_t> cnt(nd, 0);
            for (size_t j = 0; j < nd; ++j) {
                const size_t left = w[j].slot.size() - pos[j];
                if (left == 0) {
                    continue;
                }
                const size_t n = std::min<size_t>(left, 1024);
                outs[j].resize((size_t) K * n);
                if (!st.devs[j].p->moe_submit(st.devs[j].cls[(size_t) L.cls].s, xs, nt, w[j].slot.data() + pos[j], w[j].tok.data() + pos[j], (int) n, outs[j].data())) {
                    return false;
                }
                cnt[j] = n;
                sub.push_back(j);
            }
            size_t cold_n = 0;
            std::vector<float> cout_;
            if (cold_pos < u_e.size()) {
                cold_n = std::min<size_t>(u_e.size() - cold_pos, 1024);
                cout_.resize((size_t) K * cold_n);
                const auto tc = clk::now();
                if (!cold_pairs(st, L, xs, nt, u_e.data() + cold_pos, u_tok.data() + cold_pos, (int64_t) cold_n, cout_.data())) {
                    return false;
                }
                if (st.profile) {
                    st.t_pre_cold.add(ms_since(tc));
                }
            }
            bool ok = true;
            for (size_t j : sub) {
                ok = st.devs[j].p->wait() && ok;
            }
            if (!ok) {
                return false;
            }
            for (size_t j : sub) {
                for (size_t i = 0; i < cnt[j]; ++i) {
                    put(w[j].tok[pos[j] + i], w[j].k[pos[j] + i], outs[j].data() + i * (size_t) K);
                }
                pos[j] += cnt[j];
                st.n_pre_gpu += (double) cnt[j];
            }
            for (size_t i = 0; i < cold_n; ++i) {
                put(u_tok[cold_pos + i], u_k[cold_pos + i], cout_.data() + i * (size_t) K);
            }
            cold_pos += cold_n;
            st.n_pre_cold += (double) cold_n;
            if (sub.empty() && cold_n == 0) {
                break;
            }
        }
        // the streamed experts
        if (!s_exp_id.empty()) {
            dev_rec & dr = st.devs[(size_t) sdev];
            dev_cls & dc = dr.cls[(size_t) L.cls];
            std::vector<moe_src> src(s_exp_id.size());
            for (size_t i = 0; i < s_exp_id.size(); ++i) {
                src[i].gate = expert_ptr(L, 'g', s_exp_id[i]);
                src[i].up = expert_ptr(L, 'u', s_exp_id[i]);
                src[i].down = expert_ptr(L, 'd', s_exp_id[i]);
            }
            std::vector<float> sout((size_t) K * s_pair_exp.size());
            const auto ts = clk::now();
            if (!dr.p->moe_stream(dc.s, dc.n_slots, dc.n_ring, src.data(), (int) src.size(), xs, nt, s_pair_exp.data(), s_pair_tok.data(), (int) s_pair_exp.size(), sout.data())) {
                MOE_LOG("streaming through %s failed\n", dr.name.c_str());
                return false;
            }
            if (st.profile) {
                st.t_pre_stream.add(ms_since(ts));
                st.n_stream_experts += (double) s_exp_id.size();
            }
            for (size_t i = 0; i < s_pair_exp.size(); ++i) {
                put(s_pair_tok[i], s_pair_k[i], sout.data() + i * (size_t) K);
            }
            st.n_pre_gpu += (double) s_pair_exp.size();
        }
    }
    if (st.profile) {
        st.t_pre.add(ms_since(t0));
    }
    return true;
}

bool run_block(state & st, ggml_cgraph * g, int i, block & b) {
    (void) g; (void) i;
    const int li = layer_of_block(st, b);
    layer_rec & L = st.layers[(size_t) li];
    if (L.cls < 0) {
        return false;
    }
    const ggml_tensor * ids_t = b.up->src[2];
    const int nu = (int) ids_t->ne[0];
    const int N = (int) ids_t->ne[1];
    read_ids(ids_t, nu, N, g_ids);
    // a new pass over the layers?
    if (L.il <= st.last_il || !st.pass_started) {
        st.pass_started = true;
        begin_pass(st, N);
        st.pass_n = N;
        if (N > 1) {
            evict_ring(st);
        }
        if (N >= 8) {
            st.replan_pending = true;     // a prompt was processed: place the experts it used (at the end of the pass, or at the first decode)
        }
        if (N == 1) {
            const bool never = st.plans == 0;
            if ((st.replan_pending && (!st.freeze || never) && st.tokens_since_plan >= 1) || (never && st.tokens_seen >= st.plan_min_tokens)) {
                plan(st);
                st.replan_pending = false;
            }
        }
    }
    st.last_il = L.il;
    count(st, L, g_ids.data(), nu, N);
    const bool ok = N == 1 ? run_decode(st, L, b, g_ids.data(), nu) : run_prefill(st, L, b, g_ids.data(), nu, N);
    // the last layer of a prefill pass: the statistics of the whole prompt are in; place the experts now, so that the next pass (the generation, or the next
    // prompt of a benchmark) finds them in the slots. (Not for the first decode token of a pass whose prompt was a few tokens: handled above.)
    if (ok && st.pass_n >= 8 && L.il >= st.max_il && st.replan_pending && (!st.freeze || st.plans == 0)) {
        plan(st);
        st.replan_pending = false;
    }
    return ok;
}

// ---------------------------------------------------------------------------------------------------------------------------------------
// the elementwise ops of the block, on the host

bool host_binop(ggml_tensor * dst, bool mul) {
    const ggml_tensor * a = dst->src[0];
    const ggml_tensor * b = dst->src[1];
    if (a->type != GGML_TYPE_F32 || b->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32 || a->nb[0] != 4 || b->nb[0] != 4 || dst->nb[0] != 4) {
        return false;
    }
    for (int i3 = 0; i3 < dst->ne[3]; ++i3) {
        for (int i2 = 0; i2 < dst->ne[2]; ++i2) {
            for (int i1 = 0; i1 < dst->ne[1]; ++i1) {
                const float * pa = (const float *) ((const char *) a->data + i3 * a->nb[3] + i2 * a->nb[2] + i1 * a->nb[1]);
                const float * pb = (const float *) ((const char *) b->data + (i3 % b->ne[3]) * b->nb[3] + (i2 % b->ne[2]) * b->nb[2] + (i1 % b->ne[1]) * b->nb[1]);
                float * pd = (float *) ((char *) dst->data + i3 * dst->nb[3] + i2 * dst->nb[2] + i1 * dst->nb[1]);
                if (b->ne[0] == dst->ne[0]) {
                    if (mul) { for (int64_t i0 = 0; i0 < dst->ne[0]; ++i0) { pd[i0] = pa[i0] * pb[i0]; } }
                    else     { for (int64_t i0 = 0; i0 < dst->ne[0]; ++i0) { pd[i0] = pa[i0] + pb[i0]; } }
                } else {
                    for (int64_t i0 = 0; i0 < dst->ne[0]; ++i0) {
                        const float v = pb[i0 % b->ne[0]];
                        pd[i0] = mul ? pa[i0] * v : pa[i0] + v;
                    }
                }
            }
        }
    }
    return true;
}

bool is_weighted(const ggml_tensor * t) {
    return t != nullptr && t->op == GGML_OP_MUL && mmid_ours(t->src[0]);
}

bool is_part(const ggml_tensor * t) {
    if (t == nullptr) {
        return false;
    }
    if (t->op == GGML_OP_VIEW) {
        return is_weighted(t->view_src != nullptr ? t->view_src : t->src[0]);
    }
    if (t->op == GGML_OP_ADD) {
        return is_part(t->src[0]) && is_part(t->src[1]);
    }
    return false;
}

} // namespace

// ---------------------------------------------------------------------------------------------------------------------------------------
// the interface

bool enabled() {
    static int e = -1;
    if (e < 0) {
        state tmp;
        e = parse_env(tmp) ? 1 : 0;
    }
    return e == 1;
}

void init(ggml_backend_buffer_type_t buft, const hooks & h) {
    if (S != nullptr || !enabled()) {
        return;
    }
    S = new state();
    parse_env(*S);
    S->buft = buft;
    S->h = h;
    S->ready = true;
}

void note_tensor(ggml_tensor * t) {
    if (S == nullptr || t == nullptr || t->ne[2] <= 1 || t->view_src != nullptr) {
        return;
    }
    int il = -1;
    char kind = 0;
    if (!parse_name(t->name, il, kind)) {
        return;
    }
    layer_index_for_tensor(*S, t, kind, il);
}

bool supports_op(const ggml_tensor * op) {
    if (S == nullptr) {
        return false;
    }
    switch (op->op) {
        case GGML_OP_MUL_MAT_ID: {
            const ggml_tensor * w = op->src[0];
            const ggml_tensor * x = op->src[1];
            const ggml_tensor * ids = op->src[2];
            return weight_ours(w) && x != nullptr && ids != nullptr && x->type == GGML_TYPE_F32 && ids->type == GGML_TYPE_I32 && op->type == GGML_TYPE_F32 &&
                   w->ne[3] == 1 && w->nb[0] == ggml_type_size(w->type) && w->ne[0] % ggml_blck_size(w->type) == 0;
        }
        case GGML_OP_GLU:
            return ggml_get_glu_op(op) == GGML_GLU_OP_SWIGLU && mmid_ours(op->src[0]) && mmid_ours(op->src[1]) && op->src[0]->src[1] == op->src[1]->src[1];
        case GGML_OP_MUL:
            return op->type == GGML_TYPE_F32 && op->src[0]->type == GGML_TYPE_F32 && op->src[1]->type == GGML_TYPE_F32 && mmid_ours(op->src[0]) &&
                   ggml_can_repeat(op->src[1], op->src[0]);
        case GGML_OP_ADD:
            return op->type == GGML_TYPE_F32 && is_part(op->src[0]) && is_part(op->src[1]);
        default:
            return false;
    }
}

bool handles(const ggml_tensor * node) {
    if (S == nullptr) {
        return false;
    }
    switch (node->op) {
        case GGML_OP_MUL_MAT_ID:
        case GGML_OP_GLU:
            return true;
        case GGML_OP_MUL:
            return mmid_ours(node->src[0]);
        case GGML_OP_ADD:
            return is_part(node->src[0]) && is_part(node->src[1]);
        default:
            return false;
    }
}

bool exec(ggml_cgraph * g, int i) {
    state & st = *S;
    ggml_tensor * node = g->nodes[i];
    ensure_cpu(st);
    if (node->op == GGML_OP_MUL_MAT_ID && node->src[2] != nullptr) {
        st.nu_seen = (int) node->src[2]->ne[0];
    }
    if (!st.devs_inited && node->op == GGML_OP_MUL_MAT_ID && node->src[2] != nullptr) {
        init_devices(st, (int) node->src[2]->ne[0]);
    }
    auto it = st.done.find(node);
    if (it != st.done.end()) {
        st.done.erase(it);
        return true;
    }
    switch (node->op) {
        case GGML_OP_MUL_MAT_ID: {
            block b;
            if (find_block(g, i, b) && block_shapes_ok(b) && st.cpu != nullptr) {
                // a prefill can also take the weighting and the sum that follow (see block::mul)
                if (b.jdown >= 0 && b.down->ne[2] > 1) {
                    find_tail(g, b.jdown, b);
                }
                // the weights must be known as one layer (3 tensors of the same shape)
                if (run_block(st, g, i, b)) {
                    st.done.insert(b.gate == node ? b.up : b.gate);
                    st.done.insert(b.glu);
                    st.done.insert(b.down);
                    if (b.tail_done) {
                        st.done.insert(b.mul);
                        for (ggml_tensor * a : b.adds) {
                            st.done.insert(a);
                        }
                    }
                    return true;
                }
                MOE_LOG("block execution failed, falling back to ggml-cpu for node %s\n", node->name);
                // fall through to the generic path for this node only
            }
            return exec_cpu_node(st, g, i);
        }
        case GGML_OP_MUL:
            if (host_binop(node, true)) {
                return true;
            }
            return exec_cpu_node(st, g, i);
        case GGML_OP_ADD:
            if (host_binop(node, false)) {
                return true;
            }
            return exec_cpu_node(st, g, i);
        default:
            return exec_cpu_node(st, g, i);
    }
}

void shutdown() {
    if (S == nullptr) {
        return;
    }
    for (auto & d : S->devs) {
        for (auto & c : d.cls) {
            if (c.s != nullptr) {
                d.p->moe_free(c.s);
                c.s = nullptr;
            }
        }
    }
    // the placement tables point at freed slots now: nothing runs after this
    for (auto & L : S->layers) {
        std::fill(L.dev.begin(), L.dev.end(), (int16_t) -1);
    }
}

ggml_backend_buffer_t alloc_pinned(size_t size) {
    if (S == nullptr || S->specs.empty() || (getenv("GGML_SGLX_MOE_PINNED") != nullptr && atoi(getenv("GGML_SGLX_MOE_PINNED")) == 0)) {
        return nullptr;
    }
    ggml_backend_dev_t d = find_ggml_device(S->specs[0].name.c_str());
    if (d == nullptr) {
        return nullptr;
    }
    ggml_backend_buffer_type_t bt = ggml_backend_dev_host_buffer_type(d);
    return bt != nullptr ? ggml_backend_buft_alloc_buffer(bt, size) : nullptr;
}

void note_input_copy_ms(double ms) {
    if (S != nullptr && S->profile && ms > 0.0) {
        S->t_copy.add(ms);
    }
}

void print_profile() {
    if (S != nullptr) {
        write_stats(*S);
    }
    if (S == nullptr || !S->profile) {
        return;
    }
    state & st = *S;
    if (st.dec_blocks != 0) {
        fprintf(stderr, "SGLX MoE profile decode: %llu blocks, mean %.3f ms (max %.3f): gpu submit %.3f, cold cpu %.3f, gpu wait %.3f; cold %.2f / gpu %.2f experts per block, "
                        "%.2f gpus used\n",
                (unsigned long long) st.dec_blocks, st.t_dec.mean(), st.t_dec.mx, st.t_dec_gpu_sub.mean(), st.t_dec_cold.mean(), st.t_dec_gpu_wait.mean(),
                st.n_pairs_cold / (double) st.dec_blocks, st.n_pairs_gpu / (double) st.dec_blocks, st.gpu_used_sum / (double) st.dec_blocks);
        for (const auto & d : st.devs) {
            fprintf(stderr, "SGLX MoE profile   %s: submit %.3f ms, wait %.3f ms (%llu calls)\n", d.name.c_str(), d.t_submit.mean(), d.t_wait.mean(),
                    (unsigned long long) d.t_submit.n);
        }
    }
    for (const auto & d : st.devs) {
        if (d.gated) {
            fprintf(stderr, "SGLX MoE profile   %s (gated): used in %llu layers, skipped in %llu, latency EMA %.2f ms\n", d.name.c_str(), (unsigned long long) d.gate_used,
                    (unsigned long long) d.gate_skipped, d.lat_ms);
        }
        const std::string ds = d.p->stats();
        if (!ds.empty()) {
            fprintf(stderr, "SGLX MoE profile   %s %s\n", d.name.c_str(), ds.c_str());
        }
    }
    if (st.t_cold_compute.n != 0) {
        fprintf(stderr, "SGLX MoE profile cold decode: alloc %.3f, set %.3f, graph_compute %.3f ms (%llu calls)\n", st.t_cold_alloc.mean(), st.t_cold_set.mean(),
                st.t_cold_compute.mean(), (unsigned long long) st.t_cold_compute.n);
    }
    if (st.t_pre.n != 0) {
        fprintf(stderr, "SGLX MoE profile prefill: %llu blocks, mean %.1f ms (max %.1f), cold cpu rounds %.1f ms, streaming %.1f ms (%.1f experts per block); pairs: gpu %.0f, cold %.0f "
                        "per block\n", (unsigned long long) st.t_pre.n, st.t_pre.mean(), st.t_pre.mx, st.t_pre_cold.mean(), st.t_pre_stream.mean(),
                st.n_stream_experts / (double) st.t_pre.n, st.n_pre_gpu / (double) st.t_pre.n, st.n_pre_cold / (double) st.t_pre.n);
    }
    if (st.t_copy.n != 0) {
        fprintf(stderr, "SGLX MoE profile: lazy input copies %.3f ms per graph_compute that had them (%llu calls, max %.3f)\n", st.t_copy.mean(), (unsigned long long) st.t_copy.n, st.t_copy.mx);
    }
    fprintf(stderr, "SGLX MoE profile: %d placements, %.0f ms in total, %lld tokens\n", st.plans, st.plan_ms, (long long) st.tokens_seen);
}

} // namespace moe
} // namespace sglx
