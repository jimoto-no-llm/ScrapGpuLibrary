// Bandwidth and latency of the quantized MUL_MAT / MUL_MAT_ID through the public C API, on one device of one backend (cuda | opencl11 | vulkan | cpu),
// so that the CUDA backend and the CL11 (OpenCL 1.1) backend are measured with the same code. One GPU per process on NVIDIA 390 (Fermi and Kepler
// cannot share a process): pick the card with SCRAPGPU_CUDA_DEVICES="GT 730" or GGML_CL11_PLATFORMS=NVIDIA GGML_CL11_DEVICES="GT 730".
//
//   mm_bench BACKEND mv   [--type q6_K|q4_0] [--rows 128000] [--k 2048] [--n 1] [--iters 20] [--reps 7]
//       MUL_MAT of a [k, rows] weight (default: the LFM2.5-8B output projection, q6_K 2048 x 128000 = 205.1 MiB) with n F32 columns.
//   mm_bench BACKEND mid  [--experts 32] [--rows 1792] [--k 2048] [--used 4] [--tokens 1] [--iters 50] [--reps 7]
//       MUL_MAT_ID with Q4_0 experts (one gate or up projection of the model: 32 experts of 1792 x 2048, 4 used per token).
//
// Weights are random (a few distinct rows tiled, quantized by the reference); the correctness of the results is tested by the tests, not here.
// "throughput" enqueues `iters` ops and syncs once (the kernel time with the launch costs overlapped); "latency" syncs after every op.
// Built with SCRAPGPU_BUILD_BENCHMARKS (and the backends that are built).

#include "scrapgpu/backend.h"
#include "scrapgpu/buffer.h"
#include "scrapgpu/cpu.h"
#include "scrapgpu/device.h"
#include "scrapgpu/reference.h"
#include "scrapgpu/stream.h"
#include "scrapgpu/tensor.h"
#if defined(MMB_WITH_OPENCL11)
#include "scrapgpu/opencl11.h"
#endif
#if defined(MMB_WITH_VULKAN)
#include "scrapgpu/vulkan.h"
#endif
#if defined(MMB_WITH_CUDA)
#include "scrapgpu/cuda.h"
#endif

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace {

void check(scrapgpu_status s, const char * what) {
    if (s != SCRAPGPU_STATUS_OK) {
        std::fprintf(stderr, "%s: %s: %s\n", what, scrapgpu_status_string(s), scrapgpu_last_error());
        std::exit(1);
    }
}

scrapgpu_tensor_desc tdesc(scrapgpu_dtype dt, int64_t n0, int64_t n1 = 1, int64_t n2 = 1) {
    scrapgpu_tensor_desc t{};
    t.struct_size = sizeof(t);
    t.dtype = dt;
    t.ne[0] = n0; t.ne[1] = n1; t.ne[2] = n2; t.ne[3] = 1;
    t.nb[0] = scrapgpu_dtype_type_size(dt);
    t.nb[1] = scrapgpu_tensor_row_size(dt, n0);
    t.nb[2] = t.nb[1] * (uint64_t) n1;
    t.nb[3] = t.nb[2] * (uint64_t) n2;
    t.layout = SCRAPGPU_LAYOUT_NATURAL;
    return t;
}

uint64_t span(const scrapgpu_tensor_desc & t) {
    return t.nb[3] * (uint64_t) t.ne[3];
}

double now_ms() {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

struct Env {
    scrapgpu_device_t * dev = nullptr;
    scrapgpu_stream_t * stream = nullptr;
    scrapgpu_layout wlayout = SCRAPGPU_LAYOUT_NATURAL;
};

Env g;

scrapgpu_buffer_t * alloc(size_t bytes) {
    scrapgpu_buffer_t * b = nullptr;
    check(scrapgpu_buffer_alloc(g.dev, bytes, &b), "buffer alloc");
    return b;
}

// a quantized weight of `rows` rows built by tiling 64 random quantized rows (a 205 MiB random image would take the reference minutes)
void upload_weight(scrapgpu_tensor_desc & t, scrapgpu_dtype dt, int64_t K, int64_t rows, int64_t layers) {
    std::mt19937 rng(7);
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    const int64_t pat_rows = std::min<int64_t>(rows, 64);
    std::vector<float> f((size_t) (pat_rows * K));
    for (auto & v : f) v = u(rng);
    const size_t rb = scrapgpu_tensor_row_size(dt, K);
    std::vector<uint8_t> pat((size_t) pat_rows * rb);
    check(scrapgpu_ref_quantize(dt, f.data(), pat.data(), pat_rows * K), "quantize");
    std::vector<uint8_t> img((size_t) rows * (size_t) layers * rb);
    for (int64_t r = 0; r < rows * layers; ++r) {
        std::memcpy(img.data() + (size_t) r * rb, pat.data() + (size_t) (r % pat_rows) * rb, rb);
    }
    t = tdesc(dt, K, rows, layers);
    t.layout = g.wlayout;
    t.buffer = alloc((size_t) span(t));
    check(scrapgpu_tensor_upload(g.stream, &t, img.data(), img.size()), "weight upload");
    check(scrapgpu_stream_sync(g.stream), "sync");
}

scrapgpu_tensor_desc make_f32(int64_t n0, int64_t n1 = 1, int64_t n2 = 1, bool fill = true) {
    scrapgpu_tensor_desc t = tdesc(SCRAPGPU_DTYPE_F32, n0, n1, n2);
    t.buffer = alloc((size_t) span(t));
    if (fill) {
        std::mt19937 rng(11);
        std::uniform_real_distribution<float> u(-1.0f, 1.0f);
        std::vector<float> v((size_t) (n0 * n1 * n2));
        for (auto & x : v) x = u(rng);
        check(scrapgpu_tensor_upload(g.stream, &t, v.data(), v.size() * 4), "upload");
    }
    return t;
}

struct Timing {
    double best = 0, median = 0;
};

template <typename F>
Timing time_op(F && enqueue, int iters, int reps, bool sync_each) {
    for (int i = 0; i < 3; ++i) enqueue();
    check(scrapgpu_stream_sync(g.stream), "warmup sync");
    std::vector<double> t;
    for (int r = 0; r < reps; ++r) {
        const double t0 = now_ms();
        for (int i = 0; i < iters; ++i) {
            enqueue();
            if (sync_each) check(scrapgpu_stream_sync(g.stream), "sync");
        }
        check(scrapgpu_stream_sync(g.stream), "sync");
        t.push_back((now_ms() - t0) / iters);
    }
    std::sort(t.begin(), t.end());
    return { t.front(), t[t.size() / 2] };
}

const char * arg(int argc, char ** argv, const char * name, const char * def) {
    for (int i = 3; i + 1 < argc; ++i) {
        if (std::strcmp(argv[i], name) == 0) return argv[i + 1];
    }
    return def;
}

} // namespace

int main(int argc, char ** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s BACKEND mv|mid [options] (see the header of tools/mm_bench.cpp)\n", argv[0]);
        return 2;
    }
    const std::string be = argv[1], mode = argv[2];
    if (be == "cpu") {
        check(scrapgpu_cpu_register(), "register cpu");
#if defined(MMB_WITH_OPENCL11)
    } else if (be == "opencl11") {
        check(scrapgpu_opencl11_register(), "register opencl11");
        g.wlayout = SCRAPGPU_LAYOUT_R32;
#endif
#if defined(MMB_WITH_VULKAN)
    } else if (be == "vulkan") {
        check(scrapgpu_vulkan_register(), "register vulkan");
#endif
#if defined(MMB_WITH_CUDA)
    } else if (be == "cuda") {
        check(scrapgpu_cuda_register(), "register cuda");
#endif
    } else {
        std::fprintf(stderr, "backend not built: %s\n", be.c_str());
        return 2;
    }
    if (scrapgpu_get_device_count() == 0) {
        std::puts("SKIP: no device");
        return 77;
    }
    check(scrapgpu_get_device(0, &g.dev), "get device");
    check(scrapgpu_stream_create(g.dev, &g.stream), "stream create");
    scrapgpu_device_info info{};
    check(scrapgpu_device_info_init(&info, sizeof(info)), "info init");
    check(scrapgpu_device_get_info(g.dev, &info), "info");
    std::printf("device=%s backend=%s driver=%s tuning=%s\n", info.name, info.backend_name, info.driver_version ? info.driver_version : "",
                info.tuning_profile_name ? info.tuning_profile_name : "");

    const int iters = std::atoi(arg(argc, argv, "--iters", mode == "mv" ? "20" : "50"));
    const int reps = std::atoi(arg(argc, argv, "--reps", "7"));
    if (mode == "mv") {
        const std::string ty = arg(argc, argv, "--type", "q6_K");
        const scrapgpu_dtype dt = ty == "q4_0" ? SCRAPGPU_DTYPE_Q4_0 : SCRAPGPU_DTYPE_Q6_K;
        const int64_t rows = std::atoll(arg(argc, argv, "--rows", "128000"));
        const int64_t K = std::atoll(arg(argc, argv, "--k", "2048"));
        const int64_t N = std::atoll(arg(argc, argv, "--n", "1"));
        scrapgpu_tensor_desc w, x = make_f32(K, N), o = make_f32(rows, N, 1, false);
        upload_weight(w, dt, K, rows, 1);
        const double bytes = (double) scrapgpu_tensor_row_size(dt, K) * (double) rows;
        scrapgpu_mul_mat_params p{sizeof(p), 0};
        scrapgpu_tensor_desc ins[2] = { w, x };
        scrapgpu_op_desc d{};
        d.struct_size = sizeof(d);
        d.op = SCRAPGPU_OP_MUL_MAT;
        d.inputs = ins;
        d.input_count = 2;
        d.outputs = &o;
        d.output_count = 1;
        d.params = &p;
        d.params_size = sizeof(p);
        const char * why = nullptr;
        if (scrapgpu_op_supported(g.dev, &d, &why) != SCRAPGPU_STATUS_OK) {
            std::printf("UNSUPPORTED: %s\n", why ? why : "");
            return 1;
        }
        const auto run = [&] { check(scrapgpu_op_execute(g.stream, &d), "MUL_MAT"); };
        const Timing th = time_op(run, iters, reps, false);
        const Timing la = time_op(run, std::max(1, iters / 4), reps, true);
        std::printf("mv %s k=%lld rows=%lld n=%lld weight=%.1f MiB\n", ty.c_str(), (long long) K, (long long) rows, (long long) N, bytes / 1048576.0);
        std::printf("  throughput: best %.3f ms  median %.3f ms  = %.2f / %.2f GB/s (weights read once per column chunk)\n", th.best, th.median,
                    bytes / th.best / 1e6, bytes / th.median / 1e6);
        std::printf("  latency   : best %.3f ms  median %.3f ms\n", la.best, la.median);
    } else if (mode == "mid") {
        const int64_t E = std::atoll(arg(argc, argv, "--experts", "32"));
        const int64_t rows = std::atoll(arg(argc, argv, "--rows", "1792"));
        const int64_t K = std::atoll(arg(argc, argv, "--k", "2048"));
        const int64_t used = std::atoll(arg(argc, argv, "--used", "4"));
        const int64_t tok = std::atoll(arg(argc, argv, "--tokens", "1"));
        scrapgpu_tensor_desc w, x = make_f32(K, used, tok), o = make_f32(rows, used, tok, false);
        upload_weight(w, SCRAPGPU_DTYPE_Q4_0, K, rows, E);
        scrapgpu_tensor_desc ids = tdesc(SCRAPGPU_DTYPE_I32, used, tok);
        ids.buffer = alloc((size_t) span(ids));
        std::vector<int32_t> iv((size_t) (used * tok));
        for (size_t i = 0; i < iv.size(); ++i) iv[i] = (int32_t) ((i * 7 + 3) % (size_t) E);
        check(scrapgpu_tensor_upload(g.stream, &ids, iv.data(), iv.size() * 4), "ids upload");
        scrapgpu_mul_mat_id_params p{sizeof(p), (uint32_t) used, 0};
        scrapgpu_tensor_desc ins[3] = { w, x, ids };
        scrapgpu_op_desc d{};
        d.struct_size = sizeof(d);
        d.op = SCRAPGPU_OP_MUL_MAT_ID;
        d.inputs = ins;
        d.input_count = 3;
        d.outputs = &o;
        d.output_count = 1;
        d.params = &p;
        d.params_size = sizeof(p);
        const char * why = nullptr;
        if (scrapgpu_op_supported(g.dev, &d, &why) != SCRAPGPU_STATUS_OK) {
            std::printf("UNSUPPORTED: %s\n", why ? why : "");
            return 1;
        }
        const double bytes = (double) scrapgpu_tensor_row_size(SCRAPGPU_DTYPE_Q4_0, K) * (double) rows * (double) used * (double) tok; // weights read (a repeated expert counts again)
        const auto run = [&] { check(scrapgpu_op_execute(g.stream, &d), "MUL_MAT_ID"); };
        const Timing th = time_op(run, iters, reps, false);
        const Timing la = time_op(run, std::max(1, iters / 4), reps, true);
        std::printf("mid q4_0 experts=%lld rows=%lld k=%lld used=%lld tokens=%lld  weights read per op %.2f MiB\n", (long long) E, (long long) rows,
                    (long long) K, (long long) used, (long long) tok, bytes / 1048576.0);
        std::printf("  throughput: best %.3f ms  median %.3f ms  = %.2f / %.2f GB/s\n", th.best, th.median, bytes / th.best / 1e6, bytes / th.median / 1e6);
        std::printf("  latency   : best %.3f ms  median %.3f ms  = %.2f / %.2f GB/s\n", la.best, la.median, bytes / la.best / 1e6, bytes / la.median / 1e6);
    } else {
        std::fprintf(stderr, "unknown mode %s\n", mode.c_str());
        return 2;
    }
    return 0;
}
