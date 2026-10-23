// Where the time of a mat-vec on a device in another process goes (scrapgpu-device-server, the "remote" backend), for the GT 430 in the SGLX split
// (docs/benchmarks/moe-ep-20261022/gt430). The pieces of one job of adapters/ggml/composite/pdev_scrapgpu.cpp are timed alone and together:
//
//   scrapgpu_remote_bench [--rows 4608] [--k 2048] [--type q6_k|q4_0] [--iters 300] [--batch]   (--batch: scrapgpu_remote_set_batching)
//   environment: SCRAPGPU_REMOTE_BACKEND=opencl11|cuda and the device selection of the server as for scrapgpu_remote_spawn, e.g.
//   SCRAPGPU_DEVICE_SERVER=.../scrapgpu-device-server SCRAPGPU_REMOTE_BACKEND=cuda SCRAPGPU_CUDA_DEVICES="GT 430" scrapgpu_remote_bench
//   (or --socket PATH for a server that is already running)
//   --gap-us N   (instead of the pieces) a sequence of jobs with N us of sleep between them, as in token generation (about one job per token, the GPU
//                idle in between); --hog T: T threads that spin during it (llama-bench's -t 2 threads on the 2-core host); --spin-gap: spin instead
//                of sleeping in the gap; prints the percentiles of the job time
//
// Every number is the mean of --iters calls in ms (min in brackets).

#include "scrapgpu/api.h"
#include "scrapgpu/backend.h"
#include "scrapgpu/buffer.h"
#include "scrapgpu/device.h"
#include "scrapgpu/op.h"
#include "scrapgpu/remote.h"
#include "scrapgpu/stream.h"
#include "scrapgpu/tensor.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <atomic>
#include <string>
#include <thread>
#include <vector>
#include <unistd.h>

namespace {

using clk = std::chrono::steady_clock;

double ms_since(clk::time_point t0) {
    return std::chrono::duration<double, std::milli>(clk::now() - t0).count();
}

#define CHECK(x)                                                                                                         \
    do {                                                                                                                 \
        const scrapgpu_status st_ = (x);                                                                                 \
        if (st_ != SCRAPGPU_STATUS_OK) {                                                                                 \
            fprintf(stderr, "%s:%d: %s failed (%d): %s\n", __FILE__, __LINE__, #x, (int) st_, scrapgpu_last_error());    \
            return 1;                                                                                                    \
        }                                                                                                                \
    } while (0)

struct stat_t {
    double sum = 0, mn = 1e30;
    int    n = 0;
    void add(double v) { sum += v; mn = std::min(mn, v); ++n; }
};

void report(const char * what, const stat_t & s) {
    printf("%-52s %8.3f ms  (min %.3f)\n", what, s.sum / std::max(1, s.n), s.mn);
}

scrapgpu_tensor_desc make_desc(scrapgpu_dtype dtype, int64_t ne0, int64_t ne1, scrapgpu_buffer_t * buffer, scrapgpu_layout layout) {
    scrapgpu_tensor_desc d;
    memset(&d, 0, sizeof(d));
    d.struct_size = sizeof(d);
    d.dtype = dtype;
    d.ne[0] = ne0;
    d.ne[1] = ne1;
    d.ne[2] = 1;
    d.ne[3] = 1;
    d.nb[0] = scrapgpu_dtype_type_size(dtype);
    d.nb[1] = scrapgpu_tensor_row_size(dtype, ne0);
    d.nb[2] = d.nb[1] * (uint64_t) ne1;
    d.nb[3] = d.nb[2];
    d.buffer = buffer;
    d.layout = layout;
    return d;
}

} // namespace

int main(int argc, char ** argv) {
    int64_t rows = 4608, K = 2048;
    int iters = 300;
    scrapgpu_dtype dt = SCRAPGPU_DTYPE_Q6_K;
    const char * sock = nullptr;
    bool batch = false;
    int gap_us = -1, hog = 0;
    bool spin_gap = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--rows" && i + 1 < argc) rows = atoll(argv[++i]);
        else if (a == "--k" && i + 1 < argc) K = atoll(argv[++i]);
        else if (a == "--iters" && i + 1 < argc) iters = atoi(argv[++i]);
        else if (a == "--batch") batch = true;
        else if (a == "--gap-us" && i + 1 < argc) gap_us = atoi(argv[++i]);
        else if (a == "--hog" && i + 1 < argc) hog = atoi(argv[++i]);
        else if (a == "--spin-gap") spin_gap = true;
        else if (a == "--socket" && i + 1 < argc) sock = argv[++i];
        else if (a == "--type" && i + 1 < argc) dt = std::string(argv[++i]) == "q4_0" ? SCRAPGPU_DTYPE_Q4_0 : SCRAPGPU_DTYPE_Q6_K;
        else { fprintf(stderr, "unknown argument %s\n", a.c_str()); return 2; }
    }
    scrapgpu_remote_t * remote = nullptr;
    if (sock != nullptr) {
        CHECK(scrapgpu_remote_connect(sock, &remote));
    } else {
        const char * env[] = { nullptr };
        CHECK(scrapgpu_remote_spawn(env, &remote));
    }
    if (batch) {
        CHECK(scrapgpu_remote_set_batching(remote, 1));
    }
    scrapgpu_device_t * dev = nullptr;
    if (scrapgpu_get_device_count() == 0) {
        fprintf(stderr, "no device\n");
        return 1;
    }
    CHECK(scrapgpu_get_device(0, &dev));
    scrapgpu_device_info info;
    scrapgpu_device_info_init(&info, sizeof(info));
    CHECK(scrapgpu_device_get_info(dev, &info));
    printf("device %s, backend %s; %lld rows x %lld, %s\n", info.name, info.backend_name != nullptr ? info.backend_name : "?", (long long) rows, (long long) K,
           dt == SCRAPGPU_DTYPE_Q4_0 ? "q4_0" : "q6_K");
    scrapgpu_stream_t * s = nullptr;
    CHECK(scrapgpu_stream_create(dev, &s));

    scrapgpu_tensor_desc wd = make_desc(dt, K, rows, nullptr, SCRAPGPU_LAYOUT_NATURAL);
    const scrapgpu_layout layout = scrapgpu_tensor_r32_eligible(&wd) ? SCRAPGPU_LAYOUT_R32 : SCRAPGPU_LAYOUT_NATURAL;
    const size_t wbytes = (size_t) wd.nb[2];
    scrapgpu_buffer_t *wb = nullptr, *xb = nullptr, *ob = nullptr;
    CHECK(scrapgpu_buffer_alloc(dev, wbytes, &wb));
    CHECK(scrapgpu_buffer_alloc(dev, (size_t) K * 4, &xb));
    CHECK(scrapgpu_buffer_alloc(dev, (size_t) rows * 4, &ob));
    std::vector<uint8_t> w(wbytes);
    uint32_t seed = 12345;
    for (size_t i = 0; i < w.size(); ++i) {
        seed = seed * 1664525u + 1013904223u;
        w[i] = (uint8_t) (seed >> 24);
    }
    // fix the scale halves so that no value is NaN/Inf: not needed for timing, the kernels do not branch on values
    wd = make_desc(dt, K, rows, wb, layout);
    const auto tu = clk::now();
    CHECK(scrapgpu_tensor_upload(s, &wd, w.data(), wbytes));
    CHECK(scrapgpu_stream_sync(s));
    printf("upload of %.1f MiB: %.1f ms\n", (double) wbytes / 1048576.0, ms_since(tu));

    std::vector<float> x((size_t) K, 0.5f), out((size_t) rows);
    scrapgpu_tensor_desc ins[2] = { wd, make_desc(SCRAPGPU_DTYPE_F32, K, 1, xb, SCRAPGPU_LAYOUT_NATURAL) };
    scrapgpu_tensor_desc o = make_desc(SCRAPGPU_DTYPE_F32, rows, 1, ob, SCRAPGPU_LAYOUT_NATURAL);
    scrapgpu_mul_mat_params prm = { sizeof(prm), 0 };
    scrapgpu_op_desc od;
    memset(&od, 0, sizeof(od));
    od.struct_size = sizeof(od);
    od.op = SCRAPGPU_OP_MUL_MAT;
    od.inputs = ins;
    od.input_count = 2;
    od.outputs = &o;
    od.output_count = 1;
    od.params = &prm;
    od.params_size = sizeof(prm);

    // warm up
    for (int i = 0; i < 20; ++i) {
        CHECK(scrapgpu_buffer_write(s, xb, 0, x.data(), x.size() * 4, 0));
        CHECK(scrapgpu_op_execute(s, &od));
        CHECK(scrapgpu_buffer_read(s, ob, 0, out.data(), out.size() * 4, 1));
    }
    if (gap_us >= 0) {
        std::atomic<bool> stop{false};
        std::vector<std::thread> hogs;
        for (int h = 0; h < hog; ++h) {
            hogs.emplace_back([&] { volatile uint64_t v = 0; while (!stop.load(std::memory_order_relaxed)) { for (int k = 0; k < 1000; ++k) v = v + 1; } });
        }
        std::vector<double> t_job, t_wr, t_ex, t_rd;
        for (int i = 0; i < iters; ++i) {
            const auto g0 = clk::now();
            if (spin_gap) { while (ms_since(g0) * 1000.0 < gap_us) {} } else if (gap_us > 0) { usleep((useconds_t) gap_us); }
            auto t = clk::now();
            CHECK(scrapgpu_buffer_write(s, xb, 0, x.data(), x.size() * 4, 0));
            const double a = ms_since(t);
            CHECK(scrapgpu_op_execute(s, &od));
            const double b = ms_since(t);
            CHECK(scrapgpu_buffer_read(s, ob, 0, out.data(), out.size() * 4, 1));
            const double c = ms_since(t);
            t_wr.push_back(a); t_ex.push_back(b - a); t_rd.push_back(c - b); t_job.push_back(c);
        }
        stop = true;
        for (auto & th : hogs) th.join();
        auto pr = [&](const char * what, std::vector<double> v) {
            std::sort(v.begin(), v.end());
            double sum = 0; for (double d : v) sum += d;
            printf("%-34s mean %.3f  p10 %.3f  p50 %.3f  p90 %.3f  max %.3f ms\n", what, sum / v.size(), v[v.size() / 10], v[v.size() / 2], v[v.size() * 9 / 10], v.back());
        };
        printf("gap %d us (%s), hog %d threads, %d jobs\n", gap_us, spin_gap ? "spin" : "sleep", hog, iters);
        pr("write x", t_wr); pr("op_execute (enqueue)", t_ex); pr("read (kernel + read)", t_rd); pr("job", t_job);
        scrapgpu_buffer_free(wb); scrapgpu_buffer_free(xb); scrapgpu_buffer_free(ob); scrapgpu_stream_free(s); scrapgpu_remote_close(remote);
        return 0;
    }
    stat_t idle, wr, wr_b, ex, exsync, rd, job, kern_only;
    for (int i = 0; i < iters; ++i) {
        auto t = clk::now(); CHECK(scrapgpu_stream_sync(s)); idle.add(ms_since(t));
        t = clk::now(); CHECK(scrapgpu_buffer_write(s, xb, 0, x.data(), x.size() * 4, 0)); wr.add(ms_since(t));
        t = clk::now(); CHECK(scrapgpu_buffer_write(s, xb, 0, x.data(), x.size() * 4, 1)); wr_b.add(ms_since(t));
        t = clk::now(); CHECK(scrapgpu_op_execute(s, &od)); ex.add(ms_since(t));
        t = clk::now(); CHECK(scrapgpu_stream_sync(s)); exsync.add(ms_since(t));
        t = clk::now(); CHECK(scrapgpu_op_execute(s, &od)); CHECK(scrapgpu_stream_sync(s)); kern_only.add(ms_since(t));
        t = clk::now(); CHECK(scrapgpu_buffer_read(s, ob, 0, out.data(), out.size() * 4, 1)); rd.add(ms_since(t));
        t = clk::now();
        CHECK(scrapgpu_buffer_write(s, xb, 0, x.data(), x.size() * 4, 0));
        CHECK(scrapgpu_op_execute(s, &od));
        CHECK(scrapgpu_buffer_read(s, ob, 0, out.data(), out.size() * 4, 1));
        job.add(ms_since(t));
    }
    report("stream_sync, idle (one round trip)", idle);
    report("buffer_write x (non-blocking flag)", wr);
    report("buffer_write x (blocking flag)", wr_b);
    report("op_execute MUL_MAT, enqueue only", ex);
    report("stream_sync after it (the kernel + one round trip)", exsync);
    report("op_execute + stream_sync (kernel alone, device idle)", kern_only);
    report("buffer_read result, device idle", rd);
    report("job: write x + op_execute + read (blocking)", job);
    const double gb = (double) wbytes / 1e9;
    printf("weights %.2f MB: the kernel alone = %.1f GB/s\n", (double) wbytes / 1e6, gb / (kern_only.sum / kern_only.n / 1000.0));

    scrapgpu_buffer_free(wb);
    scrapgpu_buffer_free(xb);
    scrapgpu_buffer_free(ob);
    scrapgpu_stream_free(s);
    scrapgpu_remote_close(remote);
    return 0;
}
