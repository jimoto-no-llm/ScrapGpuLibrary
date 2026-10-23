// see moe_cpu.h
#include "moe_cpu.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>

#if defined(__x86_64__) || defined(_M_X64)
#  include <immintrin.h>
#  define SGLX_RELAX() _mm_pause()
#else
#  define SGLX_RELAX() do { } while (0)
#endif

#if !defined(_WIN32)
#  include <dlfcn.h>
#endif

namespace sglx {
namespace moe {

namespace {

// the layout of ggml_type_traits_cpu in ggml-cpu.h (ggml_from_float_t, ggml_vec_dot_t, vec_dot_type, nrows); checked against the library by the tests
struct traits_cpu_abi {
    void (*from_float)(const float *, void *, int64_t);
    void (*vec_dot)(int, float *, size_t, const void *, size_t, const void *, size_t, int);
    ggml_type vec_dot_type;
    int64_t   nrows;
};

} // namespace

cpu_kernel::~cpu_kernel() {
    quit_.store(true);
    {
        std::lock_guard<std::mutex> lk(mu_);
        gen_.fetch_add(1);
    }
    cv_.notify_all();
    for (auto & t : threads_) {
        if (t.joinable()) {
            t.join();
        }
    }
}

bool cpu_kernel::init(ggml_backend_reg_t cpu_reg, int helpers, int spin_us) {
#if defined(_WIN32)
    (void) cpu_reg; (void) helpers; (void) spin_us;
    return false;
#else
    // libggml-cpu is loaded privately by ggml_backend_load: find it through the address of one of its functions
    void * fp = ggml_backend_reg_get_proc_address(cpu_reg, "ggml_backend_set_n_threads");
    Dl_info info;
    if (fp == nullptr || dladdr(fp, &info) == 0 || info.dli_fname == nullptr) {
        return false;
    }
    void * h = dlopen(info.dli_fname, RTLD_NOW | RTLD_NOLOAD);
    if (h == nullptr) {
        return false;
    }
    get_traits_ = (const void * (*)(ggml_type)) dlsym(h, "ggml_get_type_traits_cpu");
    if (get_traits_ == nullptr) {
        return false;
    }
    spin_us_ = spin_us;
    for (int i = 0; i < helpers; ++i) {
        threads_.emplace_back([this] { helper_loop(); });
    }
    return true;
#endif
}

bool cpu_kernel::get(ggml_type t, traits & out) const {
    if (get_traits_ == nullptr) {
        return false;
    }
    const auto * tr = (const traits_cpu_abi *) get_traits_(t);
    if (tr == nullptr || tr->vec_dot == nullptr) {
        return false;
    }
    // the activations are quantized to vec_dot_type; F32 weights would need no conversion but are not worth a kernel here
    if (tr->vec_dot_type == GGML_TYPE_F32) {
        return false;
    }
    // the activations are quantized with the from_float of the type they are converted to
    const auto * qt = (const traits_cpu_abi *) get_traits_(tr->vec_dot_type);
    if (qt == nullptr || qt->from_float == nullptr) {
        return false;
    }
    out.from_float = qt->from_float;
    out.vec_dot = tr->vec_dot;
    out.vec_dot_type = tr->vec_dot_type;
    return true;
}

bool cpu_kernel::supports(ggml_type t_gate, ggml_type t_up, ggml_type t_down) const {
    traits a, b, c;
    return get(t_gate, a) && get(t_up, b) && get(t_down, c);
}

void cpu_kernel::execute(job & j) {
    for (;;) {
        const int t = j.next.fetch_add(1, std::memory_order_relaxed);
        if (t >= j.n_tasks) {
            break;
        }
        j.fn(j.ctx, t);
        j.done.fetch_add(1, std::memory_order_release);
    }
}

void cpu_kernel::helper_loop() {
    uint64_t seen = 0;
    for (;;) {
        // spin for a while, then sleep until the next job
        const auto t0 = std::chrono::steady_clock::now();
        for (;;) {
            if (quit_.load(std::memory_order_relaxed)) {
                return;
            }
            if (gen_.load(std::memory_order_acquire) != seen) {
                break;
            }
            for (int i = 0; i < 64; ++i) {
                SGLX_RELAX();
            }
            if (spin_us_ >= 0 && std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() > spin_us_) {
                std::unique_lock<std::mutex> lk(mu_);
                cv_.wait(lk, [&] { return quit_.load() || gen_.load() != seen; });
            }
        }
        seen = gen_.load(std::memory_order_acquire);
        job * j = cur_.load(std::memory_order_acquire);
        if (j != nullptr) {
            execute(*j);
        }
    }
}

namespace {

struct stage1_ctx {
    const char * wg, * wu;
    size_t       sg, su;
    const int32_t * e;
    int64_t      n_ff, n_embd;
    size_t       rg, ru;                 // bytes of a row
    const char * xq_g, * xq_u;
    void (*dot_g)(int, float *, size_t, const void *, size_t, const void *, size_t, int);
    void (*dot_u)(int, float *, size_t, const void *, size_t, const void *, size_t, int);
    float * g, * u;
    int     rows_per_task;
    int     tasks_per_mat;
};

void stage1_task(void * c, int t) {
    auto * s = (stage1_ctx *) c;
    // t = (expert i, matrix m, block b)
    const int per_expert = 2 * s->tasks_per_mat;
    const int i = t / per_expert;
    const int r = t % per_expert;
    const int m = r / s->tasks_per_mat;
    const int b = r % s->tasks_per_mat;
    const int64_t r0 = (int64_t) b * s->rows_per_task, r1 = std::min<int64_t>(s->n_ff, r0 + s->rows_per_task);
    const char * w = (m == 0 ? s->wg + (size_t) s->e[i] * s->sg : s->wu + (size_t) s->e[i] * s->su);
    const size_t rb = m == 0 ? s->rg : s->ru;
    const char * xq = m == 0 ? s->xq_g : s->xq_u;
    float * out = (m == 0 ? s->g : s->u) + (size_t) i * (size_t) s->n_ff;
    auto dot = m == 0 ? s->dot_g : s->dot_u;
    for (int64_t r_ = r0; r_ < r1; ++r_) {
        dot((int) s->n_embd, out + r_, 0, w + (size_t) r_ * rb, 0, xq, 0, 1);
    }
}

struct stage3_ctx {
    const char * wd;
    size_t       sd;
    const int32_t * e;
    int64_t      n_ff, n_embd;
    size_t       rd;
    const char * aq;                     // quantized activations, one per expert
    size_t       aq_bytes;
    void (*dot_d)(int, float *, size_t, const void *, size_t, const void *, size_t, int);
    float * out;
    int     rows_per_task;
    int     tasks_per_expert;
};

void stage3_task(void * c, int t) {
    auto * s = (stage3_ctx *) c;
    const int i = t / s->tasks_per_expert;
    const int b = t % s->tasks_per_expert;
    const int64_t r0 = (int64_t) b * s->rows_per_task, r1 = std::min<int64_t>(s->n_embd, r0 + s->rows_per_task);
    const char * w = s->wd + (size_t) s->e[i] * s->sd;
    float * out = s->out + (size_t) i * (size_t) s->n_embd;
    const char * aq = s->aq + (size_t) i * s->aq_bytes;
    for (int64_t r_ = r0; r_ < r1; ++r_) {
        s->dot_d((int) s->n_ff, out + r_, 0, w + (size_t) r_ * s->rd, 0, aq, 0, 1);
    }
}

} // namespace

bool cpu_kernel::run(ggml_type t_gate, ggml_type t_up, ggml_type t_down, int64_t n_embd, int64_t n_ff, const char * wg, size_t sg, const char * wu, size_t su,
                     const char * wd, size_t sd, const float * x, const int32_t * e, int n, float * out) {
    traits tg, tu, td;
    if (n <= 0 || n > 16 || !get(t_gate, tg) || !get(t_up, tu) || !get(t_down, td)) {
        return false;
    }
    const size_t xq_bytes = ggml_row_size(tg.vec_dot_type, n_embd);
    const size_t xq2_bytes = ggml_row_size(tu.vec_dot_type, n_embd);
    const size_t aq_bytes = ggml_row_size(td.vec_dot_type, n_ff);
    if (xq_.size() < xq_bytes) { xq_.resize(xq_bytes); }
    if (xq2_.size() < xq2_bytes) { xq2_.resize(xq2_bytes); }
    if (aq_.size() < aq_bytes * 16) { aq_.resize(aq_bytes * 16); }
    if (g_.size() < (size_t) n_ff * 16) { g_.resize((size_t) n_ff * 16); u_.resize((size_t) n_ff * 16); }
    tg.from_float(x, xq_.data(), n_embd);
    if (tu.vec_dot_type == tg.vec_dot_type) {
        memcpy(xq2_.data(), xq_.data(), xq_bytes);
    } else {
        tu.from_float(x, xq2_.data(), n_embd);
    }
    // stage 1: gate and up
    const int rows1 = 64;
    stage1_ctx s1{ wg, wu, sg, su, e, n_ff, n_embd, ggml_row_size(t_gate, n_embd), ggml_row_size(t_up, n_embd), xq_.data(), xq2_.data(), tg.vec_dot, tu.vec_dot,
                   g_.data(), u_.data(), rows1, (int) ((n_ff + rows1 - 1) / rows1) };
    auto next_job = [&](void (*fn)(void *, int), void * ctx, int n_tasks) -> job & {
        // a ring of jobs: a helper that is late may still look at the previous job
        job & j = jobs_[(jobs_next_++) & 3];
        j.fn = fn;
        j.ctx = ctx;
        j.n_tasks = n_tasks;
        j.done.store(0, std::memory_order_relaxed);
        j.next.store(0, std::memory_order_release);
        return j;
    };
    job & j1 = next_job(stage1_task, &s1, n * 2 * s1.tasks_per_mat);
    auto run_job = [&](job & j) {
        if (!threads_.empty()) {
            cur_.store(&j, std::memory_order_release);
            {
                std::lock_guard<std::mutex> lk(mu_);
                gen_.fetch_add(1, std::memory_order_acq_rel);
            }
            cv_.notify_all();
        }
        execute(j);
        // chunks that a helper holds: wait for them
        while (j.done.load(std::memory_order_acquire) < j.n_tasks) {
            SGLX_RELAX();
        }
    };
    run_job(j1);
    // activations: silu(gate) * up, quantized for the down projection
    for (int i = 0; i < n; ++i) {
        float * gi = g_.data() + (size_t) i * (size_t) n_ff;
        const float * ui = u_.data() + (size_t) i * (size_t) n_ff;
        for (int64_t k = 0; k < n_ff; ++k) {
            const float v = gi[k];
            gi[k] = v / (1.0f + expf(-v)) * ui[k];
        }
        td.from_float(gi, aq_.data() + (size_t) i * aq_bytes, n_ff);
    }
    // stage 3: down
    const int rows3 = 64;
    stage3_ctx s3{ wd, sd, e, n_ff, n_embd, ggml_row_size(t_down, n_ff), aq_.data(), aq_bytes, td.vec_dot, out, rows3, (int) ((n_embd + rows3 - 1) / rows3) };
    job & j3 = next_job(stage3_task, &s3, n * s3.tasks_per_expert);
    run_job(j3);
    return true;
}

} // namespace moe
} // namespace sglx
