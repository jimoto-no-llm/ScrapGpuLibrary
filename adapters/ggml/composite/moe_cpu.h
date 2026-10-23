// The cold experts of one token on the CPU, without a ggml graph: gate / up mat-vec, SWIGLU and down mat-vec with the vec_dot kernels of ggml-cpu (the same
// kernels and the same Q8 quantization of the activations that ggml-cpu's MUL_MAT_ID uses), split over the calling thread and one helper thread by dynamic
// chunks. Why: on the TB250 (2 cores) a ggml graph run is 10 ms when another thread of the process (llama.cpp's thread pool polls for 0.3 s after each graph) takes a
// core, because the graph's barriers wait for the descheduled thread of the OpenMP team. Here the calling thread takes every chunk that the helper has not
// taken, so a late helper only costs the chunk that it holds.
#ifndef SGLX_MOE_CPU_H
#define SGLX_MOE_CPU_H

#include "ggml-backend.h"
#include "ggml.h"

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <thread>
#include <vector>

namespace sglx {
namespace moe {

class cpu_kernel {
public:
    cpu_kernel() = default;
    ~cpu_kernel();
    // finds ggml-cpu's type traits through the loaded CPU backend; `helpers` threads in addition to the caller (0 = the caller does everything)
    bool init(ggml_backend_reg_t cpu_reg, int helpers, int spin_us);
    // can it compute these types?
    bool supports(ggml_type t_gate, ggml_type t_up, ggml_type t_down) const;
    // `n` (<= 16) experts `e[i]` of one token column x[n_embd]. wg / wu: [n_embd, n_ff] per expert (stride sg / su), wd: [n_ff, n_embd] (stride sd).
    // out[i * n_embd ...] = down_e (silu(gate_e x) * (up_e x)), no routing weight.
    bool run(ggml_type t_gate, ggml_type t_up, ggml_type t_down, int64_t n_embd, int64_t n_ff, const char * wg, size_t sg, const char * wu, size_t su,
             const char * wd, size_t sd, const float * x, const int32_t * e, int n, float * out);

private:
    struct traits {
        void (*from_float)(const float *, void *, int64_t) = nullptr;
        void (*vec_dot)(int, float *, size_t, const void *, size_t, const void *, size_t, int) = nullptr;
        ggml_type vec_dot_type = GGML_TYPE_F32;
    };
    bool get(ggml_type t, traits & out) const;

    struct job {
        void (*fn)(void *, int) = nullptr;
        void * ctx = nullptr;
        int    n_tasks = 0;
        std::atomic<int> next{0};
        std::atomic<int> done{0};
    };
    void execute(job & j);
    void helper_loop();

    const void * (*get_traits_)(ggml_type) = nullptr;
    std::vector<std::thread> threads_;
    std::atomic<bool> quit_{false};
    std::atomic<uint64_t> gen_{0};
    std::atomic<job *> cur_{nullptr};
    job jobs_[4];
    unsigned jobs_next_ = 0;
    std::mutex mu_;
    std::condition_variable cv_;
    int spin_us_ = 500;
    std::vector<char>  xq_, xq2_, aq_;
    std::vector<float> g_, u_;
};

} // namespace moe
} // namespace sglx

#endif
