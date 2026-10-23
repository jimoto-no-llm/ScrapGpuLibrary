// Micro benchmarks MB0-MB5 of docs/plans/MOE_COMPOSITE_DEVICE.md chapter 9 (the go / no-go numbers of the MoE composite device).
// One stand-alone tool; the sub command is the first argument:
//
//   mb0   CUDA Driver API (dlopen libcuda): can one process create a context on, and launch a kernel on, every NVIDIA GPU (Fermi + Kepler)?
//   mb1c  CUDA round trip (8 KiB write -> tiny kernel -> 8 KiB read) per device and variant, plus all devices at once (the CUDA MB1 / MB2)
//   mb1   OpenCL round trip per device: blocking / non-blocking+clFinish / event busy-poll / pinned host memory, and its pieces
//   mb2   OpenCL: MB1 on several devices at once (submit to all, then wait): sequential vs one thread vs one thread per device
//   mb3   OpenCL: one MoE layer of one token (LFM2.5 shape: 2048 x 1792 Q4_0, gate+up+swiglu+down, k=4, 32 experts), one device, and
//         expert-slice (ES) split over several devices, with the host sum
//   mb3x  mb3 (ii) with the WX 2100 slice computed by the official ggml-vulkan and the NVIDIA slices by OpenCL, in one process
//   mb3g  the same layer as a ggml graph (MUL_MAT_ID) on ggml backends (Vulkan RX 6400 / WX 2100, CPU)
//   mb4   ggml backend boundary cost: 8 KiB tensor_set / tensor_get, 1-node graph_compute + synchronize (what ggml_backend_sched does)
//   mb5   OpenCL: MB3 for a prompt batch (--ntok 128 / 512), single device, ES and EP (expert parallel) over several devices
//   mb5g  the same prompt batch as a ggml graph on ggml backends
//
// Build (ggml is needed by mb3g / mb4 / mb5g only; -DMRB_NO_GGML builds the rest without it):
//   Mac:    c++ -std=c++17 -O2 -Wall -Wextra -I<llama.cpp>/ggml/include adapters/ggml/tools/moe_rt_bench.cpp -framework OpenCL -L<llama.cpp>/build-dl/bin -lggml -lggml-base -Wl,-rpath,<llama.cpp>/build-dl/bin -o moe_rt_bench
//   tb250:  g++ -std=c++17 -O2 -Wall -Wextra -I<llama.cpp>/ggml/include adapters/ggml/tools/moe_rt_bench.cpp -lOpenCL -ldl -pthread -L<llama.cpp>/build/bin -lggml -lggml-base -Wl,-rpath,<llama.cpp>/build/bin -o moe_rt_bench
//   without ggml:  ... -DMRB_NO_GGML (no -lggml)
//
// Run (every GPU run on the TB250 under the bench lock, below 85 C; adapters/ggml/tools/moe_rt_bench_run.sh does that):
//   ./moe_rt_bench mb0 --libcuda /usr/lib/x86_64-linux-gnu/libcuda.so.1
//   NVIDIA only:   ./moe_rt_bench mb1 --platforms NVIDIA --devices "GT 730,GT 710"
//   NVIDIA+rusticl: RUSTICL_ENABLE=radeonsi ./moe_rt_bench mb2 --platforms NVIDIA,rusticl --devices "GT 730,GT 710,WX 2100,RX 6400"
//   ggml:          VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/radeon_icd.json ./moe_rt_bench mb4 --lib-dir <llama.cpp>/build/bin
// Common options: --iters N (timed repetitions), --warmup N, --csv FILE (appended: mb,variant,device,n,median_us,p99_us,min_us,mean_us,note),
//   --platforms a,b (substring of the OpenCL platform name), --devices "x,y" (substring of the device name, the order is kept),
//   --ggml-devs "Vulkan0,CPU" (ggml device names), --lib-dir DIR (where libggml-vulkan.so / libggml-cpu.so are), --ntok N, --ratios r1,r2,..
//   --spl N / --spd N (work-items per row in the gate-up / down kernels = 32 * N), --pin 0 (pageable instead of pinned host staging in mb3 / mb5; pinned is the default because non-blocking transfers from pageable memory take ~100 ms on NVIDIA 390), --seed N.
// OpenCL 1.1 host API and OpenCL C 1.1 only, as everywhere in this project. The kernels here are simple R32-layout Q4_0 kernels written for the
// bench (not the tuned CL11 kernels); every run prints the bandwidth it reached so that the reader can judge T_k + bytes / B.
// Exit code 0 when every correctness check passed (each MB checks its result against a CPU reference), 1 otherwise.
#define CL_TARGET_OPENCL_VERSION 110
#define CL_USE_DEPRECATED_OPENCL_1_1_APIS
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define CL_SILENCE_DEPRECATION
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif
#ifndef MRB_NO_GGML
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#endif
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <map>
#include <memory>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

// ---------------------------------------------------------------------------------------------------------------------------------
// common
// ---------------------------------------------------------------------------------------------------------------------------------
static double now_us() { return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

struct Stats { double med = 0, p99 = 0, mn = 0, mean = 0, mx = 0; size_t n = 0; };
static Stats stats_of(std::vector<double> v) {
    Stats s;
    if (v.empty()) return s;
    std::sort(v.begin(), v.end());
    s.n    = v.size();
    s.mn   = v.front();
    s.mx   = v.back();
    s.med  = (v.size() % 2) ? v[v.size() / 2] : 0.5 * (v[v.size() / 2 - 1] + v[v.size() / 2]);
    s.p99  = v[std::min(v.size() - 1, (size_t) std::ceil(0.99 * (double) v.size()) - 1)];
    s.mean = std::accumulate(v.begin(), v.end(), 0.0) / (double) v.size();
    return s;
}

static FILE * g_csv = nullptr;
static int    g_fail = 0;

static std::string sci(double v) { char b[32]; snprintf(b, sizeof b, "%.2e", v); return b; }

static std::string no_comma(std::string s) { for (auto & c : s) if (c == ',' || c == '\n') c = ';'; return s; }

static void report(const char * mb, const std::string & variant, const std::string & dev, const Stats & s, const std::string & note = "") {
    printf("%-5s %-34s %-24s n=%-6zu median=%10.1f p99=%10.1f min=%10.1f mean=%10.1f us  %s\n", mb, variant.c_str(), dev.c_str(), s.n, s.med, s.p99,
           s.mn, s.mean, note.c_str());
    fflush(stdout);
    if (g_csv) {
        fprintf(g_csv, "%s,%s,%s,%zu,%.2f,%.2f,%.2f,%.2f,%s\n", mb, no_comma(variant).c_str(), no_comma(dev).c_str(), s.n, s.med, s.p99, s.mn, s.mean,
                no_comma(note).c_str());
        fflush(g_csv);
    }
}

static void check(const char * mb, const std::string & what, bool ok, const std::string & detail = "") {
    printf("%-5s CHECK %-40s %s %s\n", mb, what.c_str(), ok ? "OK" : "FAIL", detail.c_str());
    fflush(stdout);
    if (g_csv) fprintf(g_csv, "%s,check:%s,,0,0,0,0,0,%s %s\n", mb, no_comma(what).c_str(), ok ? "OK" : "FAIL", no_comma(detail).c_str());
    if (!ok) g_fail++;
}

template <class F> static Stats bench(size_t warm, size_t iters, F && f) {
    for (size_t i = 0; i < warm; ++i) f();
    std::vector<double> t(iters);
    for (size_t i = 0; i < iters; ++i) { const double t0 = now_us(); f(); t[i] = now_us() - t0; }
    return stats_of(t);
}

struct Args {
    std::map<std::string, std::string> kv;
    std::string cmd;
    std::string get(const std::string & k, const std::string & d = "") const { auto it = kv.find(k); return it == kv.end() ? d : it->second; }
    long geti(const std::string & k, long d) const { auto it = kv.find(k); return it == kv.end() ? d : atol(it->second.c_str()); }
    bool has(const std::string & k) const { return kv.count(k) != 0; }
};

static std::vector<std::string> split(const std::string & s, char sep = ',') {
    std::vector<std::string> r;
    size_t p = 0;
    while (p <= s.size()) {
        size_t q = s.find(sep, p);
        if (q == std::string::npos) q = s.size();
        if (q > p) r.push_back(s.substr(p, q - p));
        p = q + 1;
    }
    return r;
}

static std::string lower(std::string s) { for (auto & c : s) c = (char) tolower((unsigned char) c); return s; }

static uint64_t g_rng = 88172645463325252ull;
static uint64_t rnd64() { g_rng ^= g_rng << 13; g_rng ^= g_rng >> 7; g_rng ^= g_rng << 17; return g_rng; }
static float    frand() { return (float) (rnd64() >> 40) / (float) (1 << 24) - 0.5f; }

static uint16_t f2h(float f) {
    uint32_t x; memcpy(&x, &f, 4);
    const uint32_t s = (x >> 16) & 0x8000u;
    const int      e = (int) ((x >> 23) & 0xff) - 127 + 15;
    if (e <= 0) return (uint16_t) s;
    if (e >= 31) return (uint16_t) (s | 0x7c00u);
    return (uint16_t) (s | ((uint32_t) e << 10) | ((x & 0x7fffffu) >> 13));
}
static float h2f(uint16_t h) {
    const uint32_t s = (uint32_t) (h & 0x8000u) << 16;
    const uint32_t e = (h >> 10) & 31, m = h & 1023;
    uint32_t x;
    if (e == 0) x = s;                       // zero (no sub normals are produced by f2h)
    else if (e == 31) x = s | 0x7f800000u | (m << 13);
    else x = s | ((e + 112) << 23) | (m << 13);
    float f; memcpy(&f, &x, 4);
    return f;
}

// ---------------------------------------------------------------------------------------------------------------------------------
// MoE layer data: LFM2.5-8B-A1B shapes, Q4_0 blocks in the ggml layout (2 byte half scale + 16 bytes of nibbles), random.
//   gate / up: [E][F rows][D/32 blocks]   down: [E][D rows][F/32 blocks]
// ---------------------------------------------------------------------------------------------------------------------------------
#pragma pack(push, 1)
struct Blk { uint16_t d; uint8_t qs[16]; };
#pragma pack(pop)
static_assert(sizeof(Blk) == 18, "Q4_0 block");

static const int DIM = 2048, FF = 1792, NEXP = 32, KUSE = 4;
static const int NBD = DIM / 32, NBF = FF / 32;

struct Canon {
    std::vector<Blk> g, u, d;
    void gen() {
        auto fill = [&](std::vector<Blk> & v, size_t n) {
            v.resize(n);
            for (size_t i = 0; i < n; ++i) {
                const uint64_t a = rnd64(), b = rnd64();
                memcpy(v[i].qs, &a, 8);
                memcpy(v[i].qs + 8, &b, 8);
                v[i].d = f2h(frand() * 0.02f);
            }
        };
        fill(g, (size_t) NEXP * FF * NBD);
        fill(u, (size_t) NEXP * FF * NBD);
        fill(d, (size_t) NEXP * DIM * NBF);
    }
    const Blk * gate(int e) const { return g.data() + (size_t) e * FF * NBD; }
    const Blk * up(int e) const { return u.data() + (size_t) e * FF * NBD; }
    const Blk * down(int e) const { return d.data() + (size_t) e * DIM * NBF; }
};

static inline float blk_dot_ref(const Blk & b, const float * x, double & acc) {
    double s = 0;
    for (int j = 0; j < 16; ++j) {
        s += (double) ((int) (b.qs[j] & 15) - 8) * x[j];
        s += (double) ((int) (b.qs[j] >> 4) - 8) * x[j + 16];
    }
    acc += (double) h2f(b.d) * s;
    return 0;
}

// double precision reference of one token: out = sum_s w_s * Down_e(silu(Gate_e x) * Up_e x)
static void ref_moe_token(const Canon & c, const float * x, const int * ids, const float * w, int k, std::vector<double> & out) {
    out.assign(DIM, 0.0);
    std::vector<double> h(FF);
    for (int s = 0; s < k; ++s) {
        const Blk * G = c.gate(ids[s]); const Blk * U = c.up(ids[s]); const Blk * D = c.down(ids[s]);
        for (int m = 0; m < FF; ++m) {
            double ag = 0, au = 0;
            for (int b = 0; b < NBD; ++b) { blk_dot_ref(G[(size_t) m * NBD + b], x + b * 32, ag); blk_dot_ref(U[(size_t) m * NBD + b], x + b * 32, au); }
            h[m] = ag / (1.0 + std::exp(-ag)) * au;
        }
        std::vector<float> hf(FF);
        for (int m = 0; m < FF; ++m) hf[m] = (float) h[m];
        for (int m = 0; m < DIM; ++m) {
            double a = 0;
            for (int b = 0; b < NBF; ++b) blk_dot_ref(D[(size_t) m * NBF + b], hf.data() + b * 32, a);
            out[m] += (double) w[s] * a;
        }
    }
}

static double nmse(const double * ref, const float * got, size_t n) {
    double num = 0, den = 0;
    for (size_t i = 0; i < n; ++i) { const double e = (double) got[i] - ref[i]; num += e * e; den += ref[i] * ref[i]; }
    return den > 0 ? num / den : num;
}

// R32 layout of one expert slice (rows [m0, m0 + M), blocks [b0, b0 + nb)), see src/backends/opencl11/kernels/mul_mv_r.cl:
//   quants: group g at g * 32 * nb * 16, 16 byte unit (b * 32 + lane);   scales (half): at M * nb * 16 + 2 * (g * 32 * nb + b * 32 + lane)
static void pack_r32(const Blk * base, int nbtot, int m0, int M, int b0, int nb, uint8_t * dst) {
    for (int g = 0; g < M / 32; ++g)
        for (int b = 0; b < nb; ++b)
            for (int lane = 0; lane < 32; ++lane) {
                const Blk & s = base[(size_t) (m0 + g * 32 + lane) * nbtot + b0 + b];
                memcpy(dst + (size_t) g * 32 * nb * 16 + ((size_t) b * 32 + lane) * 16, s.qs, 16);
                memcpy(dst + (size_t) M * nb * 16 + 2 * ((size_t) g * 32 * nb + (size_t) b * 32 + lane), &s.d, 2);
            }
}

// nominal bandwidth of the TB250 cards (docs/HARDWARE.md), used for the default ES / EP ratios
static double nominal_bw(const std::string & name) {
    const std::string n = lower(name);
    if (n.find("rx 6400") != std::string::npos) return 126.6;
    if (n.find("wx 2100") != std::string::npos) return 43.6;
    if (n.find("gt 730") != std::string::npos) return 37.9;
    if (n.find("gt 430") != std::string::npos) return 22.9;
    if (n.find("gt 710") != std::string::npos) return 11.7;
    return 20.0;
}

// largest remainder split of `total` units over the weights
static std::vector<int> split_units(int total, const std::vector<double> & wt) {
    double sum = 0; for (double w : wt) sum += w;
    std::vector<int> r(wt.size()); std::vector<double> rem(wt.size());
    int used = 0;
    for (size_t i = 0; i < wt.size(); ++i) { const double q = total * wt[i] / sum; r[i] = (int) q; rem[i] = q - r[i]; used += r[i]; }
    while (used < total) {
        size_t bi = 0; for (size_t i = 1; i < wt.size(); ++i) if (rem[i] > rem[bi]) bi = i;
        r[bi]++; rem[bi] = -1; used++;
    }
    return r;
}

// ---------------------------------------------------------------------------------------------------------------------------------
// OpenCL helpers
// ---------------------------------------------------------------------------------------------------------------------------------
#define CK(x) do { cl_int e_ = (x); if (e_ != CL_SUCCESS) { fprintf(stderr, "OpenCL error %d at line %d: %s\n", e_, __LINE__, #x); throw std::runtime_error("opencl"); } } while (0)

static cl_mem mkbuf(cl_context ctx, cl_mem_flags fl, size_t sz, void * host = nullptr) {
    cl_int e = 0;
    cl_mem m = clCreateBuffer(ctx, fl, sz, host, &e);
    if (e != CL_SUCCESS || !m) { fprintf(stderr, "clCreateBuffer(%zu bytes) failed: %d\n", sz, e); throw std::runtime_error("opencl"); }
    return m;
}

struct Dev {
    std::string plat, name, full_name, ver, drv;
    cl_platform_id pid = nullptr;
    cl_device_id   did = nullptr;
    cl_context     ctx = nullptr;
    cl_command_queue q = nullptr;
    size_t max_wg = 0;
    cl_ulong max_alloc = 0, gmem = 0;
    std::map<std::string, cl_program> progs;
    std::vector<cl_kernel> kernels;

    void open() {
        cl_int e;
        cl_context_properties pr[] = { CL_CONTEXT_PLATFORM, (cl_context_properties) pid, 0 };
        ctx = clCreateContext(pr, 1, &did, nullptr, nullptr, &e);
        if (e != CL_SUCCESS) { fprintf(stderr, "clCreateContext(%s) failed: %d\n", name.c_str(), e); throw std::runtime_error("opencl"); }
        q = clCreateCommandQueue(ctx, did, 0, &e);
        if (e != CL_SUCCESS) { fprintf(stderr, "clCreateCommandQueue(%s) failed: %d\n", name.c_str(), e); throw std::runtime_error("opencl"); }
    }
    void build(const std::string & pname, const std::string & src, const std::string & opts) {
        cl_int e;
        const char * s = src.c_str(); size_t len = src.size();
        cl_program p = clCreateProgramWithSource(ctx, 1, &s, &len, &e);
        CK(e);
        e = clBuildProgram(p, 1, &did, ("-cl-std=CL1.1 " + opts).c_str(), nullptr, nullptr);
        if (e != CL_SUCCESS) {
            size_t n = 0; clGetProgramBuildInfo(p, did, CL_PROGRAM_BUILD_LOG, 0, nullptr, &n);
            std::string log(n, 0); clGetProgramBuildInfo(p, did, CL_PROGRAM_BUILD_LOG, n, &log[0], nullptr);
            fprintf(stderr, "build of %s on %s failed (%d):\n%s\n", pname.c_str(), name.c_str(), e, log.c_str());
            throw std::runtime_error("build");
        }
        progs[pname] = p;
    }
    cl_kernel kern(const std::string & pname, const char * kname) {
        cl_int e; cl_kernel k = clCreateKernel(progs.at(pname), kname, &e); CK(e);
        kernels.push_back(k);
        return k;
    }
    ~Dev() {
        for (auto k : kernels) clReleaseKernel(k);
        for (auto & p : progs) clReleaseProgram(p.second);
        if (q) clReleaseCommandQueue(q);
        if (ctx) clReleaseContext(ctx);
    }
};
typedef std::shared_ptr<Dev> DevP;

static std::string clinfo_str(cl_device_id d, cl_device_info p) { size_t n = 0; clGetDeviceInfo(d, p, 0, nullptr, &n); std::string s(n, 0); clGetDeviceInfo(d, p, n, &s[0], nullptr); s.resize(strlen(s.c_str())); return s; }

static std::vector<DevP> cl_enumerate(const Args & a) {
    std::vector<DevP> all;
    cl_uint np = 0;
    clGetPlatformIDs(0, nullptr, &np);
    std::vector<cl_platform_id> pl(np);
    if (np) clGetPlatformIDs(np, pl.data(), nullptr);
    const std::vector<std::string> pf = split(lower(a.get("platforms")));
    for (cl_platform_id p : pl) {
        size_t n = 0; clGetPlatformInfo(p, CL_PLATFORM_NAME, 0, nullptr, &n);
        std::string pn(n, 0); clGetPlatformInfo(p, CL_PLATFORM_NAME, n, &pn[0], nullptr); pn.resize(strlen(pn.c_str()));
        if (!pf.empty()) {
            bool m = false; for (auto & f : pf) if (lower(pn).find(f) != std::string::npos) m = true;
            if (!m) continue;
        }
        cl_uint nd = 0;
        if (clGetDeviceIDs(p, CL_DEVICE_TYPE_GPU, 0, nullptr, &nd) != CL_SUCCESS || nd == 0) {
            if (!a.has("devices") && clGetDeviceIDs(p, CL_DEVICE_TYPE_ALL, 0, nullptr, &nd) != CL_SUCCESS) continue;   // Mac fallback
            if (nd == 0) continue;
        }
        std::vector<cl_device_id> ds(nd);
        if (clGetDeviceIDs(p, CL_DEVICE_TYPE_GPU, nd, ds.data(), nullptr) != CL_SUCCESS) clGetDeviceIDs(p, CL_DEVICE_TYPE_ALL, nd, ds.data(), nullptr);
        for (cl_device_id d : ds) {
            auto dv = std::make_shared<Dev>();
            dv->plat = pn; dv->pid = p; dv->did = d;
            dv->name = clinfo_str(d, CL_DEVICE_NAME);
            dv->full_name = dv->name;
            { const size_t q = dv->name.find(" ("); if (q != std::string::npos) dv->name.resize(q); }   // Mesa: "AMD Radeon RX 6400 (radeonsi, navi24, ...)" 
            dv->ver = clinfo_str(d, CL_DEVICE_VERSION);
            dv->drv = clinfo_str(d, CL_DRIVER_VERSION);
            clGetDeviceInfo(d, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof dv->max_wg, &dv->max_wg, nullptr);
            clGetDeviceInfo(d, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof dv->max_alloc, &dv->max_alloc, nullptr);
            clGetDeviceInfo(d, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof dv->gmem, &dv->gmem, nullptr);
            all.push_back(dv);
        }
    }
    const std::vector<std::string> df = split(a.get("devices"));
    std::vector<DevP> sel;
    if (df.empty()) sel = all;
    else
        for (auto & f : df)
            for (auto & d : all)
                if (d->name.find(f) != std::string::npos && std::find(sel.begin(), sel.end(), d) == sel.end()) { sel.push_back(d); break; }
    // --dup N: use every selected device N times (separate contexts), to try the multi device paths on a machine with one GPU
    const long dup = a.geti("dup", 1);
    if (dup > 1) {
        std::vector<DevP> r;
        for (auto & d : sel) for (long i = 0; i < dup; ++i) { auto c = std::make_shared<Dev>(*d); c->ctx = nullptr; c->q = nullptr; c->progs.clear(); c->kernels.clear(); r.push_back(c); }
        sel = r;
    }
    return sel;
}

static void print_dev(const DevP & d) {
    printf("# device: %-28s [%s] platform=%s version=%s driver=%s max_wg=%zu max_alloc=%.0fMiB gmem=%.0fMiB\n", d->name.c_str(), d->full_name.c_str(), d->plat.c_str(), d->ver.c_str(),
           d->drv.c_str(), d->max_wg, d->max_alloc / 1048576.0, d->gmem / 1048576.0);
}

// event busy poll
static void poll_event(cl_event ev) {
    cl_int st;
    do { clGetEventInfo(ev, CL_EVENT_COMMAND_EXECUTION_STATUS, sizeof st, &st, nullptr); } while (st > CL_COMPLETE);
    if (st < 0) { fprintf(stderr, "event error %d\n", st); throw std::runtime_error("event"); }
}

// ---------------------------------------------------------------------------------------------------------------------------------
// MB1 / MB2: round trip of 8 KiB
// ---------------------------------------------------------------------------------------------------------------------------------
static const char * RT_SRC = R"CLC(
kernel void rt_k(global const float * in, global float * out, int n, int iters) {
    const int i = (int) get_global_id(0);
    if (i >= n) return;
    uint s = (uint) i;
    for (int j = 0; j < iters; ++j) s = s * 1664525u + 1013904223u;
    float r = in[i] * 2.0f + 1.0f;
    if (s == 0x12345678u && i == 999999) r += 1.0f;
    out[i] = r;
}
)CLC";

static const int RT_N = 2048;   // floats = 8 KiB

struct RtDev {
    DevP d;
    cl_kernel k = nullptr;
    cl_mem din = nullptr, dout = nullptr, pin_in = nullptr, pin_out = nullptr;
    std::vector<float> src, dst;
    float * psrc = nullptr, * pdst = nullptr;     // pinned (mapped) host memory
    int iters = 0;
    cl_event ev = nullptr;

    void setup() {
        d->open();
        d->build("rt", RT_SRC, "");
        k = d->kern("rt", "rt_k");
        din  = mkbuf(d->ctx, CL_MEM_READ_ONLY, RT_N * 4);
        dout = mkbuf(d->ctx, CL_MEM_WRITE_ONLY, RT_N * 4);
        pin_in  = mkbuf(d->ctx, CL_MEM_ALLOC_HOST_PTR | CL_MEM_READ_WRITE, RT_N * 4);
        pin_out = mkbuf(d->ctx, CL_MEM_ALLOC_HOST_PTR | CL_MEM_READ_WRITE, RT_N * 4);
        cl_int e;
        psrc = (float *) clEnqueueMapBuffer(d->q, pin_in,  CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0, RT_N * 4, 0, nullptr, nullptr, &e); CK(e);
        pdst = (float *) clEnqueueMapBuffer(d->q, pin_out, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0, RT_N * 4, 0, nullptr, nullptr, &e); CK(e);
        src.resize(RT_N); dst.assign(RT_N, 0.f);
        for (auto & v : src) v = frand();
        memcpy(psrc, src.data(), RT_N * 4);
        int n = RT_N, it = 0;
        CK(clSetKernelArg(k, 0, sizeof(cl_mem), &din)); CK(clSetKernelArg(k, 1, sizeof(cl_mem), &dout));
        CK(clSetKernelArg(k, 2, sizeof n, &n)); CK(clSetKernelArg(k, 3, sizeof it, &it));
    }
    void set_iters(int it) { iters = it; CK(clSetKernelArg(k, 3, sizeof it, &it)); }
    void enq_kernel() { size_t gs = RT_N; CK(clEnqueueNDRangeKernel(d->q, k, 1, nullptr, &gs, nullptr, 0, nullptr, nullptr)); }
    // variants: 0 blocking, 1 non-blocking + clFinish, 2 non-blocking + event busy poll, 3 pinned + clFinish, 4 pinned + busy poll
    void cycle(int v) {
        const bool pin = v >= 3;
        const float * s = pin ? psrc : src.data();
        float * t = pin ? pdst : dst.data();
        if (v == 0) {
            CK(clEnqueueWriteBuffer(d->q, din, CL_TRUE, 0, RT_N * 4, s, 0, nullptr, nullptr));
            enq_kernel();
            CK(clEnqueueReadBuffer(d->q, dout, CL_TRUE, 0, RT_N * 4, t, 0, nullptr, nullptr));
        } else if (v == 1 || v == 3) {
            CK(clEnqueueWriteBuffer(d->q, din, CL_FALSE, 0, RT_N * 4, s, 0, nullptr, nullptr));
            enq_kernel();
            CK(clEnqueueReadBuffer(d->q, dout, CL_FALSE, 0, RT_N * 4, t, 0, nullptr, nullptr));
            CK(clFinish(d->q));
        } else {
            submit(pin);
            wait_poll();
        }
    }
    // submit without waiting (MB2); the read event is kept in ev
    void submit(bool pin = true) {
        const float * s = pin ? psrc : src.data();
        float * t = pin ? pdst : dst.data();
        CK(clEnqueueWriteBuffer(d->q, din, CL_FALSE, 0, RT_N * 4, s, 0, nullptr, nullptr));
        enq_kernel();
        CK(clEnqueueReadBuffer(d->q, dout, CL_FALSE, 0, RT_N * 4, t, 0, nullptr, &ev));
        CK(clFlush(d->q));
    }
    void wait_poll() { poll_event(ev); clReleaseEvent(ev); ev = nullptr; }
    void wait_finish() { CK(clFinish(d->q)); clReleaseEvent(ev); ev = nullptr; }
    // exact check: out = in * 2 + 1
    bool verify(bool pin = true) const {
        const float * t = pin ? pdst : dst.data();
        for (int i = 0; i < RT_N; ++i) if (t[i] != src[i] * 2.0f + 1.0f) return false;
        return true;
    }
    void clear_dst() { std::fill(dst.begin(), dst.end(), 0.f); memset(pdst, 0, RT_N * 4); }
    ~RtDev() {
        if (psrc) clEnqueueUnmapMemObject(d->q, pin_in, psrc, 0, nullptr, nullptr);
        if (pdst) clEnqueueUnmapMemObject(d->q, pin_out, pdst, 0, nullptr, nullptr);
        if (d->q) clFinish(d->q);
        for (cl_mem m : { din, dout, pin_in, pin_out }) if (m) clReleaseMemObject(m);
    }
};

static int mb1(const Args & a) {
    auto devs = cl_enumerate(a);
    if (devs.empty()) { fprintf(stderr, "mb1: no OpenCL device matched\n"); return 1; }
    const size_t iters = a.geti("iters", 10000), warm = a.geti("warmup", 200);
    for (auto & dv : devs) {
        print_dev(dv);
        try {
            RtDev r; r.d = dv; r.setup();
            const char * names[] = { "rt_blocking", "rt_nonblock_finish", "rt_event_busypoll", "rt_pinned_finish", "rt_pinned_busypoll" };
            for (int v = 0; v < 5; ++v) {
                r.clear_dst();
                r.cycle(v);
                bool ok = r.verify(v >= 3);
                // a variant that takes milliseconds per cycle (NVIDIA 390: non-blocking transfers from pageable memory wait ~100 ms) gets 100 repetitions
                const Stats probe = bench(1, 3, [&] { r.cycle(v); });
                const bool slow = probe.med > 2000.0;
                Stats s = bench(slow ? 2 : warm, slow ? std::min<size_t>(iters, 100) : iters, [&] { r.cycle(v); });
                r.clear_dst(); r.cycle(v); ok = ok && r.verify(v >= 3);
                report("MB1", names[v], dv->name, s, std::string(ok ? "check=OK" : "check=FAIL") + (slow ? " (slow variant: repetitions reduced)" : ""));
                if (!ok) g_fail++;
            }
            // the pieces
            Stats s;
            s = bench(warm, iters, [&] { CK(clFinish(r.d->q)); });
            report("MB1", "piece_clFinish_idle", dv->name, s);
            s = bench(warm, iters, [&] { r.enq_kernel(); CK(clFinish(r.d->q)); });
            report("MB1", "piece_kernel_finish", dv->name, s, "T_k: tiny kernel + clFinish");
            s = bench(warm, iters, [&] { CK(clEnqueueWriteBuffer(r.d->q, r.din, CL_TRUE, 0, RT_N * 4, r.src.data(), 0, nullptr, nullptr)); });
            report("MB1", "piece_write8k_blocking", dv->name, s);
            s = bench(warm, iters, [&] { CK(clEnqueueReadBuffer(r.d->q, r.dout, CL_TRUE, 0, RT_N * 4, r.dst.data(), 0, nullptr, nullptr)); });
            report("MB1", "piece_read8k_blocking", dv->name, s);
            s = bench(warm, iters, [&] { CK(clEnqueueWriteBuffer(r.d->q, r.din, CL_TRUE, 0, RT_N * 4, r.psrc, 0, nullptr, nullptr)); });
            report("MB1", "piece_write8k_pinned", dv->name, s);
            s = bench(warm, iters, [&] { CK(clEnqueueReadBuffer(r.d->q, r.dout, CL_TRUE, 0, RT_N * 4, r.pdst, 0, nullptr, nullptr)); });
            report("MB1", "piece_read8k_pinned", dv->name, s);
            // 3 input copies of 32 B and 8 KiB, the shape of the sched boundary of method E
            s = bench(warm, iters, [&] {
                CK(clEnqueueWriteBuffer(r.d->q, r.din, CL_TRUE, 0, 32, r.src.data(), 0, nullptr, nullptr));
            });
            report("MB1", "piece_write32B_blocking", dv->name, s);
        } catch (std::exception & e) { fprintf(stderr, "mb1 %s: %s\n", dv->name.c_str(), e.what()); g_fail++; }
    }
    return 0;
}

static int mb2(const Args & a) {
    auto devs = cl_enumerate(a);
    if (devs.size() < 2) { fprintf(stderr, "mb2: needs >= 2 matching devices (got %zu)\n", devs.size()); return 1; }
    const size_t iters = a.geti("iters", 10000), warm = a.geti("warmup", 200);
    std::vector<std::unique_ptr<RtDev>> rd;
    std::string names;
    for (auto & dv : devs) { print_dev(dv); rd.emplace_back(new RtDev); rd.back()->d = dv; rd.back()->setup(); names += (names.empty() ? "" : "+") + dv->name; }
    const size_t N = rd.size();
    for (int kind = 0; kind < 2; ++kind) {                         // 0: tiny kernel, 1: kernel of ~500 us on every device
        const size_t it = kind ? std::min<size_t>(iters, 3000) : iters;
        if (kind) {
            for (auto & r : rd) {
                int n = 2000;
                for (int c = 0; c < 8; ++c) {
                    r->set_iters(n);
                    Stats s = bench(3, 15, [&] { r->enq_kernel(); CK(clFinish(r->d->q)); });
                    const double f = 500.0 / std::max(1.0, s.med);
                    if (f > 0.85 && f < 1.15) break;
                    n = std::max(1, (int) (n * f));
                }
                printf("# %s: busy kernel iters=%d\n", r->d->name.c_str(), r->iters);
            }
        } else for (auto & r : rd) r->set_iters(0);
        const char * kn = kind ? "busy500us" : "tiny";
        for (int wm = 0; wm < 2; ++wm) {                           // 0: clFinish, 1: event busy poll
            const char * wn = wm ? "poll" : "finish";
            auto wait = [&](RtDev & r) { if (wm) r.wait_poll(); else r.wait_finish(); };
            std::vector<Stats> solo(N);
            double mx = 0, sum = 0;
            for (size_t i = 0; i < N; ++i) {
                RtDev & r = *rd[i];
                solo[i] = bench(warm, it, [&] { r.submit(); wait(r); });
                report("MB2", std::string("solo_") + kn + "_" + wn, r.d->name, solo[i]);
                mx = std::max(mx, solo[i].med); sum += solo[i].med;
            }
            bool ok = true;
            Stats seq = bench(warm, it, [&] { for (auto & r : rd) { r->submit(); wait(*r); } });
            report("MB2", std::string("sequential_") + kn + "_" + wn, names, seq, "one after the other");
            Stats p1 = bench(warm, it, [&] { for (auto & r : rd) r->submit(); for (auto & r : rd) wait(*r); });
            for (auto & r : rd) ok = ok && r->verify();
            report("MB2", std::string("parallel_1thread_") + kn + "_" + wn, names, p1, "submit all, then wait in order");
            // one thread per device
            std::mutex m; std::condition_variable cvg, cvd; long gen = 0; size_t done = 0; bool stop = false;
            std::vector<std::thread> th;
            for (size_t i = 0; i < N; ++i)
                th.emplace_back([&, i] {
                    long my = 0;
                    for (;;) {
                        { std::unique_lock<std::mutex> lk(m); cvg.wait(lk, [&] { return stop || gen > my; }); if (stop) return; my = gen; }
                        rd[i]->submit(); wait(*rd[i]);
                        { std::lock_guard<std::mutex> lk(m); if (++done == N) cvd.notify_one(); }
                    }
                });
            auto one = [&] {
                { std::lock_guard<std::mutex> lk(m); done = 0; gen++; }
                cvg.notify_all();
                std::unique_lock<std::mutex> lk(m); cvd.wait(lk, [&] { return done == N; });
            };
            Stats pt = bench(warm, it, one);
            { std::lock_guard<std::mutex> lk(m); stop = true; } cvg.notify_all();
            for (auto & t : th) t.join();
            for (auto & r : rd) ok = ok && r->verify();
            report("MB2", std::string("parallel_threads_") + kn + "_" + wn, names, pt, "one thread per device (wake-up included)");
            char buf[300];
            snprintf(buf, sizeof buf, "median max(solo)=%.1f sum(solo)=%.1f sequential=%.1f 1thread=%.1f threads=%.1f ; 1thread/max=%.2f 1thread/sum=%.2f threads/max=%.2f",
                     mx, sum, seq.med, p1.med, pt.med, p1.med / mx, p1.med / sum, pt.med / mx);
            printf("MB2   SUMMARY %s_%s: %s\n", kn, wn, buf);
            if (g_csv) fprintf(g_csv, "MB2,summary_%s_%s,%s,0,0,0,0,0,%s\n", kn, wn, no_comma(names).c_str(), no_comma(buf).c_str());
            check("MB2", std::string("results ") + kn + "_" + wn, ok);
        }
    }
    return 0;
}

// ---------------------------------------------------------------------------------------------------------------------------------
// MB3 / MB5: OpenCL kernels of the MoE layer
// ---------------------------------------------------------------------------------------------------------------------------------
static const char * MOE_SRC = R"CLC(
#ifndef SPL
#define SPL 4
#endif
#ifndef SPD
#define SPD 8
#endif
#define DIMK 2048
#define TN 16
#define LP 33

float dot_w(uint v, float4 xl, float4 xh) {
    const float4 lo = (float4)((float)(v & 15u), (float)((v >> 8) & 15u), (float)((v >> 16) & 15u), (float)((v >> 24) & 15u)) - (float4)(8.0f);
    const float4 hi = (float4)((float)((v >> 4) & 15u), (float)((v >> 12) & 15u), (float)((v >> 20) & 15u), (float)((v >> 28) & 15u)) - (float4)(8.0f);
    return dot(lo, xl) + dot(hi, xh);
}
float blk_dot(uint4 q, global const float4 * xb) {
    return dot_w(q.x, xb[0], xb[4]) + dot_w(q.y, xb[1], xb[5]) + dot_w(q.z, xb[2], xb[6]) + dot_w(q.w, xb[3], xb[7]);
}

// decode: h[slot][row] = silu(gate . x) * (up . x) ; one work-group = 32 rows (a group of the R32 layout) of one slot, SPL splits of the blocks
kernel void k_gu(global const uchar * wg, global const uchar * wu, global const float * x, global const int * idsw, global float * h,
                 int Mj, int nb, int estride) {
    const int lid = (int) get_local_id(0), lane = lid & 31, sp = lid >> 5;
    const int g = (int) get_group_id(0), slot = (int) get_group_id(1);
    const int e = idsw[slot];
    global const uchar * ag = wg + (size_t) e * estride;
    global const uchar * au = wu + (size_t) e * estride;
    global const uint4 * qg = (global const uint4 *)(ag + (size_t) g * 32 * nb * 16);
    global const uint4 * qu = (global const uint4 *)(au + (size_t) g * 32 * nb * 16);
    global const uchar * sg = ag + (size_t) Mj * nb * 16 + (size_t) 2 * g * 32 * nb;
    global const uchar * su = au + (size_t) Mj * nb * 16 + (size_t) 2 * g * 32 * nb;
    global const float4 * xp = (global const float4 *) x;
    float accg = 0.0f, accu = 0.0f;
    for (int b = sp; b < nb; b += SPL) {
        const float dg = vload_half(0, (global const half *)(sg + 2 * (b * 32 + lane)));
        const float du = vload_half(0, (global const half *)(su + 2 * (b * 32 + lane)));
        const uint4 a = qg[b * 32 + lane];
        const uint4 c = qu[b * 32 + lane];
        global const float4 * xb = xp + b * 8;
        accg += dg * blk_dot(a, xb);
        accu += du * blk_dot(c, xb);
    }
    local float red[2 * SPL * 32];
    red[(0 * SPL + sp) * 32 + lane] = accg;
    red[(1 * SPL + sp) * 32 + lane] = accu;
    barrier(CLK_LOCAL_MEM_FENCE);
    if (sp == 0) {
        for (int s = 1; s < SPL; ++s) { accg += red[(0 * SPL + s) * 32 + lane]; accu += red[(1 * SPL + s) * 32 + lane]; }
        h[slot * Mj + g * 32 + lane] = accg / (1.0f + exp(-accg)) * accu;
    }
}

// decode: out[row] = sum_slots w_slot * (down_e . h_slot) over the K slice of this device (nb blocks); fixed slot order. 2048 rows = 64 groups
kernel void k_down(global const uchar * wd, global const float * h, global const int * idsw, global float * out, int kslots, int nb, int estride) {
    const int lid = (int) get_local_id(0), lane = lid & 31, sp = lid >> 5;
    const int g = (int) get_group_id(0);
    global const float * wts = (global const float *)(idsw + kslots);
    local float red[SPD * 32];
    float total = 0.0f;
    for (int s = 0; s < kslots; ++s) {
        global const uchar * ae = wd + (size_t) idsw[s] * estride;
        global const uint4 * q = (global const uint4 *)(ae + (size_t) g * 32 * nb * 16);
        global const uchar * sc = ae + (size_t) DIMK * nb * 16 + (size_t) 2 * g * 32 * nb;
        global const float4 * xp = (global const float4 *)(h + (size_t) s * nb * 32);
        float acc = 0.0f;
        for (int b = sp; b < nb; b += SPD) {
            const float d = vload_half(0, (global const half *)(sc + 2 * (b * 32 + lane)));
            acc += d * blk_dot(q[b * 32 + lane], xp + b * 8);
        }
        red[sp * 32 + lane] = acc;
        barrier(CLK_LOCAL_MEM_FENCE);
        if (sp == 0) {
            float t = acc;
            for (int s2 = 1; s2 < SPD; ++s2) t += red[s2 * 32 + lane];
            total += wts[s] * t;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (sp == 0) out[g * 32 + lane] = total;
}

// ---- prompt batch: tiles of up to TN pairs (token, slot) of one expert; 64 work-items, each 4 rows x 2 pairs
// the dequantized 32 x 32 tile of the weights of block b, group g (M rows in total, nb blocks per row): row r = lid / 2, half hf = lid % 2
void load_wtile(local float * dst, global const uchar * wbase, int M, int nb, int g, int b, int lid) {
    const int r = lid >> 1, hf = lid & 1;
    const uint4 q = ((global const uint4 *)(wbase + (size_t) g * 32 * nb * 16))[b * 32 + r];
    const float d = vload_half(0, (global const half *)(wbase + (size_t) M * nb * 16 + 2 * ((size_t) g * 32 * nb + (size_t) b * 32 + r)));
    uint w[4]; w[0] = q.x; w[1] = q.y; w[2] = q.z; w[3] = q.w;
    for (int j = 0; j < 4; ++j)
        for (int c = 0; c < 4; ++c) {
            const uint n = hf ? ((w[j] >> (8 * c + 4)) & 15u) : ((w[j] >> (8 * c)) & 15u);
            dst[r * LP + hf * 16 + j * 4 + c] = ((float) n - 8.0f) * d;
        }
}

// meta (int words): tiles at o_tiles (4 ints per tile: expert, first pair, pairs, 0), pair token (compact index) at o_ptok, pair weight (float bits) at o_pw
kernel void k_pgu(global const uchar * wg, global const uchar * wu, global const float * X, global const int * meta, global float * H,
                  int Mj, int nb, int estride, int o_ptok, int o_tiles) {
    const int lid = (int) get_local_id(0);
    const int g = (int) get_group_id(0), tile = (int) get_group_id(1);
    const int4 tl = ((global const int4 *)(meta + o_tiles))[tile];
    const int e = tl.x, ps = tl.y, cnt = tl.z;
    global const uchar * ag = wg + (size_t) e * estride;
    global const uchar * au = wu + (size_t) e * estride;
    local float xt[TN * LP], wgt[32 * LP], wut[32 * LP];
    float cg[8], cu[8];
    for (int i = 0; i < 8; ++i) { cg[i] = 0.0f; cu[i] = 0.0f; }
    const int tx = lid & 7, ty = lid >> 3;
    for (int b = 0; b < nb; ++b) {
        for (int i = 0; i < 8; ++i) {
            const int idx = lid + 64 * i, t = idx >> 5, kk = idx & 31;
            float v = 0.0f;
            if (t < cnt) v = X[(size_t) meta[o_ptok + ps + t] * DIMK + b * 32 + kk];
            xt[t * LP + kk] = v;
        }
        load_wtile(wgt, ag, Mj, nb, g, b, lid);
        load_wtile(wut, au, Mj, nb, g, b, lid);
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int kk = 0; kk < 32; ++kk) {
            float x0 = xt[(ty * 2) * LP + kk], x1 = xt[(ty * 2 + 1) * LP + kk];
            for (int r = 0; r < 4; ++r) {
                const float a = wgt[(tx * 4 + r) * LP + kk], c = wut[(tx * 4 + r) * LP + kk];
                cg[r * 2] += a * x0; cg[r * 2 + 1] += a * x1;
                cu[r * 2] += c * x0; cu[r * 2 + 1] += c * x1;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    for (int r = 0; r < 4; ++r)
        for (int t = 0; t < 2; ++t) {
            const int tt = ty * 2 + t;
            if (tt < cnt) { const float gv = cg[r * 2 + t]; H[(size_t) (ps + tt) * Mj + g * 32 + tx * 4 + r] = gv / (1.0f + exp(-gv)) * cu[r * 2 + t]; }
        }
}

// Y[pair][row] = w_pair * (down_e . h_pair) over the K slice (nb blocks) ; 2048 rows
kernel void k_pdown(global const uchar * wd, global const float * H, global const int * meta, global float * Y, int nb, int estride, int o_tiles, int o_pw) {
    const int lid = (int) get_local_id(0);
    const int g = (int) get_group_id(0), tile = (int) get_group_id(1);
    const int4 tl = ((global const int4 *)(meta + o_tiles))[tile];
    const int e = tl.x, ps = tl.y, cnt = tl.z;
    global const uchar * ae = wd + (size_t) e * estride;
    global const float * pw = (global const float *)(meta + o_pw);
    local float xt[TN * LP], wt[32 * LP];
    float c[8];
    for (int i = 0; i < 8; ++i) c[i] = 0.0f;
    const int tx = lid & 7, ty = lid >> 3;
    for (int b = 0; b < nb; ++b) {
        for (int i = 0; i < 8; ++i) {
            const int idx = lid + 64 * i, t = idx >> 5, kk = idx & 31;
            xt[t * LP + kk] = (t < cnt) ? H[(size_t) (ps + t) * nb * 32 + b * 32 + kk] : 0.0f;
        }
        load_wtile(wt, ae, DIMK, nb, g, b, lid);
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int kk = 0; kk < 32; ++kk) {
            const float x0 = xt[(ty * 2) * LP + kk], x1 = xt[(ty * 2 + 1) * LP + kk];
            for (int r = 0; r < 4; ++r) { const float a = wt[(tx * 4 + r) * LP + kk]; c[r * 2] += a * x0; c[r * 2 + 1] += a * x1; }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    for (int r = 0; r < 4; ++r)
        for (int t = 0; t < 2; ++t) {
            const int tt = ty * 2 + t;
            if (tt < cnt) Y[(size_t) (ps + tt) * DIMK + g * 32 + tx * 4 + r] = c[r * 2 + t] * pw[ps + tt];
        }
}

// out[token][row] = sum of Y over the pairs of the token (fixed order)
kernel void k_scatter(global const float * Y, global const int * meta, global float * out, int ntok, int o_off, int o_tp) {
    const int i = (int) get_global_id(0);
    if (i >= ntok * DIMK) return;
    const int t = i >> 11, r = i & 2047;
    const int a = meta[o_off + t], e = meta[o_off + t + 1];
    float s = 0.0f;
    for (int j = a; j < e; ++j) s += Y[(size_t) meta[o_tp + j] * DIMK + r];
    out[i] = s;
}
)CLC";

struct HostBuf {   // plain or pinned host memory for the staging of one device
    DevP d; cl_mem m = nullptr; void * p = nullptr; std::vector<uint8_t> v; size_t sz = 0;
    void init(DevP dev, size_t bytes, bool pin) {
        d = dev; sz = bytes;
        if (pin) {
            m = mkbuf(d->ctx, CL_MEM_ALLOC_HOST_PTR | CL_MEM_READ_WRITE, bytes);
            cl_int e; p = clEnqueueMapBuffer(d->q, m, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0, bytes, 0, nullptr, nullptr, &e); CK(e);
        } else { v.assign(bytes, 0); p = v.data(); }
    }
    ~HostBuf() { if (m) { clEnqueueUnmapMemObject(d->q, m, p, 0, nullptr, nullptr); clFinish(d->q); clReleaseMemObject(m); } }
};

static std::string moe_opts(const Args & a, const Dev & d) {
    long spl = a.geti("spl", 4), spd = a.geti("spd", 8);
    while (spl > 1 && (size_t) (32 * spl) > d.max_wg) spl /= 2;
    while (spd > 1 && (size_t) (32 * spd) > d.max_wg) spd /= 2;
    return "-DSPL=" + std::to_string(spl) + " -DSPD=" + std::to_string(spd);
}

// ---- decode state of one device holding a row slice [row0, row0 + Mj) of the F dimension of all the 32 experts
struct Dec {
    DevP d; int Mj = 0, kb = 0, spl = 4, spd = 8;
    cl_mem wg = nullptr, wu = nullptr, wd = nullptr, dx = nullptr, dix = nullptr, dh = nullptr, dout = nullptr;
    cl_kernel kgu = nullptr, kdn = nullptr;
    HostBuf hx, hix, hout;
    cl_event ev = nullptr;
    double bytes_per_layer = 0;

    void setup(const Args & a, const Canon & c, int row0, int rows, bool pin) {
        Mj = rows; kb = rows / 32;
        d->open();
        d->build("moe", MOE_SRC, moe_opts(a, *d));
        { std::string o = moe_opts(a, *d); spl = atoi(o.c_str() + o.find("SPL=") + 4); spd = atoi(o.c_str() + o.find("SPD=") + 4); }
        kgu = d->kern("moe", "k_gu"); kdn = d->kern("moe", "k_down");
        const size_t esg = (size_t) Mj * NBD * 18, esd = (size_t) DIM * kb * 18;
        auto up = [&](cl_mem & m, size_t sz, const std::vector<uint8_t> & host) {
            m = mkbuf(d->ctx, CL_MEM_READ_ONLY, sz);
            CK(clEnqueueWriteBuffer(d->q, m, CL_TRUE, 0, sz, host.data(), 0, nullptr, nullptr));
        };
        std::vector<uint8_t> big;
        big.resize(esg * NEXP);
        for (int e = 0; e < NEXP; ++e) pack_r32(c.gate(e), NBD, row0, Mj, 0, NBD, big.data() + e * esg);
        up(wg, esg * NEXP, big);
        for (int e = 0; e < NEXP; ++e) pack_r32(c.up(e), NBD, row0, Mj, 0, NBD, big.data() + e * esg);
        up(wu, esg * NEXP, big);
        big.resize(esd * NEXP);
        for (int e = 0; e < NEXP; ++e) pack_r32(c.down(e), NBF, 0, DIM, row0 / 32, kb, big.data() + e * esd);
        up(wd, esd * NEXP, big);
        dx = mkbuf(d->ctx, CL_MEM_READ_ONLY, DIM * 4);
        dix = mkbuf(d->ctx, CL_MEM_READ_ONLY, 64);
        dh = mkbuf(d->ctx, CL_MEM_READ_WRITE, (size_t) KUSE * Mj * 4);
        dout = mkbuf(d->ctx, CL_MEM_READ_WRITE, DIM * 4);
        hx.init(d, DIM * 4, pin); hix.init(d, 64, pin); hout.init(d, DIM * 4, pin);
        int kk = KUSE, nbd = NBD, mj = Mj; int esgi = (int) esg, esdi = (int) esd;
        CK(clSetKernelArg(kgu, 0, sizeof(cl_mem), &wg)); CK(clSetKernelArg(kgu, 1, sizeof(cl_mem), &wu)); CK(clSetKernelArg(kgu, 2, sizeof(cl_mem), &dx));
        CK(clSetKernelArg(kgu, 3, sizeof(cl_mem), &dix)); CK(clSetKernelArg(kgu, 4, sizeof(cl_mem), &dh));
        CK(clSetKernelArg(kgu, 5, sizeof mj, &mj)); CK(clSetKernelArg(kgu, 6, sizeof nbd, &nbd)); CK(clSetKernelArg(kgu, 7, sizeof esgi, &esgi));
        CK(clSetKernelArg(kdn, 0, sizeof(cl_mem), &wd)); CK(clSetKernelArg(kdn, 1, sizeof(cl_mem), &dh)); CK(clSetKernelArg(kdn, 2, sizeof(cl_mem), &dix));
        CK(clSetKernelArg(kdn, 3, sizeof(cl_mem), &dout)); CK(clSetKernelArg(kdn, 4, sizeof kk, &kk)); CK(clSetKernelArg(kdn, 5, sizeof kb, &kb));
        CK(clSetKernelArg(kdn, 6, sizeof esdi, &esdi));
        bytes_per_layer = (double) KUSE * (2.0 * esg + (double) esd);
    }
    void set_x(const float * x) { memcpy(hx.p, x, DIM * 4); }
    void set_ids(const int * ids, const float * w) { memcpy(hix.p, ids, KUSE * 4); memcpy((char *) hix.p + KUSE * 4, w, KUSE * 4); }
    void enq_kernels() {
        size_t gs1[2] = { (size_t) kb * 32 * spl, (size_t) KUSE }, ls1[2] = { (size_t) 32 * spl, 1 };
        CK(clEnqueueNDRangeKernel(d->q, kgu, 2, nullptr, gs1, ls1, 0, nullptr, nullptr));
        size_t gs2 = (size_t) 64 * 32 * spd, ls2 = (size_t) 32 * spd;
        CK(clEnqueueNDRangeKernel(d->q, kdn, 1, nullptr, &gs2, &ls2, 0, nullptr, nullptr));
    }
    // compute only: ids (32 B) + the two kernels, then clFinish
    void compute_only(const int * ids, const float * w) {
        set_ids(ids, w);
        CK(clEnqueueWriteBuffer(d->q, dix, CL_FALSE, 0, 32, hix.p, 0, nullptr, nullptr));
        enq_kernels();
        CK(clFinish(d->q));
    }
    // the whole layer as method E would do it: write x and ids, kernels, read the partial output; no wait
    void submit(const float * x, const int * ids, const float * w) {
        set_x(x); set_ids(ids, w);
        CK(clEnqueueWriteBuffer(d->q, dx, CL_FALSE, 0, DIM * 4, hx.p, 0, nullptr, nullptr));
        CK(clEnqueueWriteBuffer(d->q, dix, CL_FALSE, 0, 32, hix.p, 0, nullptr, nullptr));
        enq_kernels();
        CK(clEnqueueReadBuffer(d->q, dout, CL_FALSE, 0, DIM * 4, hout.p, 0, nullptr, &ev));
        CK(clFlush(d->q));
    }
    void wait() { CK(clFinish(d->q)); clReleaseEvent(ev); ev = nullptr; }
    const float * out() const { return (const float *) hout.p; }
    ~Dec() { for (cl_mem m : { wg, wu, wd, dx, dix, dh, dout }) if (m) clReleaseMemObject(m); }
};

struct Route { int ids[KUSE]; float w[KUSE]; };
static Route make_route() {
    Route r; int n = 0;
    while (n < KUSE) { int e = (int) (rnd64() % NEXP); bool dup = false; for (int i = 0; i < n; ++i) if (r.ids[i] == e) dup = true; if (!dup) r.ids[n++] = e; }
    float s = 0; for (int i = 0; i < KUSE; ++i) { r.w[i] = 0.1f + 0.4f * (frand() + 0.5f); s += r.w[i]; }
    for (int i = 0; i < KUSE; ++i) r.w[i] /= s;
    return r;
}

static std::vector<float> make_x(int n) { std::vector<float> x((size_t) n * DIM); for (auto & v : x) v = frand() * 2.0f; return x; }

static int mb3(const Args & a) {
    auto devs = cl_enumerate(a);
    if (devs.empty()) { fprintf(stderr, "mb3: no OpenCL device matched\n"); return 1; }
    const size_t iters = a.geti("iters", 1000), warm = a.geti("warmup", 50);
    const bool pin = a.geti("pin", 1) != 0;
    printf("# generating the layer (32 experts, Q4_0, random)...\n");
    g_rng = 88172645463325252ull + a.geti("seed", 0);
    Canon canon; canon.gen();
    const std::vector<float> x = make_x(1);
    std::vector<Route> routes; for (int i = 0; i < 64; ++i) routes.push_back(make_route());
    const Route ck_route = routes[0];
    std::vector<double> ref; ref_moe_token(canon, x.data(), ck_route.ids, ck_route.w, KUSE, ref);
    const double tol = 1e-8;

    // (i) one device, all 1792 rows
    std::vector<std::unique_ptr<Dec>> single;
    for (auto & dv : devs) {
        print_dev(dv);
        try {
            auto s = std::unique_ptr<Dec>(new Dec); s->d = dv; s->setup(a, canon, 0, FF, pin);
            s->submit(x.data(), ck_route.ids, ck_route.w); s->wait();
            double e1 = nmse(ref.data(), s->out(), DIM);
            size_t ri = 0;
            Stats co = bench(warm, iters, [&] { const Route & r = routes[ri++ & 63]; s->compute_only(r.ids, r.w); });
            char note[200]; snprintf(note, sizeof note, "achieved %.2f GB/s (%.2f MB per layer) nmse=%.2e", s->bytes_per_layer / (co.med * 1e-6) / 1e9, s->bytes_per_layer / 1e6, e1);
            report("MB3", "i_single_compute_only", dv->name, co, note);
            ri = 0;
            Stats fu = bench(warm, iters, [&] { const Route & r = routes[ri++ & 63]; s->submit(x.data(), r.ids, r.w); s->wait(); });
            const Route & lr = routes[(ri - 1) & 63];
            std::vector<double> ref2; ref_moe_token(canon, x.data(), lr.ids, lr.w, KUSE, ref2);
            double e2 = nmse(ref2.data(), s->out(), DIM);
            report("MB3", "i_single_full_cycle", dv->name, fu, "x write + ids + kernels + out read");
            check("MB3", "single " + dv->name + " first/last", e1 < tol && e2 < tol, "nmse " + sci(e1) + " / " + sci(e2));
            single.push_back(std::move(s));
        } catch (std::exception & e) { fprintf(stderr, "mb3 %s: %s\n", dv->name.c_str(), e.what()); g_fail++; }
    }
    // (ii) ES over all the matching devices
    if (single.size() >= 2) {
        std::vector<double> wt; std::string names;
        for (auto & s : single) { wt.push_back(nominal_bw(s->d->name)); names += (names.empty() ? "" : "+") + s->d->name; }
        if (a.has("ratios")) { wt.clear(); for (auto & r : split(a.get("ratios"))) wt.push_back(atof(r.c_str())); }
        const std::vector<int> groups = split_units(FF / 32, wt);
        std::vector<std::unique_ptr<Dec>> es;
        int row0 = 0; std::string desc;
        for (size_t i = 0; i < single.size(); ++i) {
            if (groups[i] == 0) continue;
            auto s = std::unique_ptr<Dec>(new Dec);
            // a second context on the same device is fine in OpenCL, but re-use the device object: open a fresh Dev
            auto dv2 = std::make_shared<Dev>(*single[i]->d);
            dv2->ctx = nullptr; dv2->q = nullptr; dv2->progs.clear(); dv2->kernels.clear();
            s->d = dv2; s->setup(a, canon, row0, groups[i] * 32, pin);
            desc += single[i]->d->name + ":" + std::to_string(groups[i] * 32) + " rows ";
            row0 += groups[i] * 32;
            es.push_back(std::move(s));
        }
        printf("# ES split: %s\n", desc.c_str());
        std::vector<float> sum(DIM);
        auto run = [&](const Route & r) {
            for (auto & s : es) s->submit(x.data(), r.ids, r.w);
            for (auto & s : es) s->wait();
            const float * o0 = es[0]->out();
            for (int i = 0; i < DIM; ++i) sum[i] = o0[i];
            for (size_t j = 1; j < es.size(); ++j) { const float * o = es[j]->out(); for (int i = 0; i < DIM; ++i) sum[i] += o[i]; }
        };
        run(ck_route);
        const double e1 = nmse(ref.data(), sum.data(), DIM);
        size_t ri = 0;
        Stats st = bench(warm, iters, [&] { run(routes[ri++ & 63]); });
        const Route & lr = routes[(ri - 1) & 63];
        std::vector<double> ref2; ref_moe_token(canon, x.data(), lr.ids, lr.w, KUSE, ref2);
        const double e2 = nmse(ref2.data(), sum.data(), DIM);
        report("MB3", "ii_ES_full_cycle", names, st, "submit all, wait all, host sum; " + desc);
        // compute only (no x / out transfers)
        ri = 0;
        Stats sc = bench(warm, iters, [&] {
            const Route & r = routes[ri++ & 63];
            for (auto & s : es) { s->set_ids(r.ids, r.w); CK(clEnqueueWriteBuffer(s->d->q, s->dix, CL_FALSE, 0, 32, s->hix.p, 0, nullptr, nullptr)); s->enq_kernels(); CK(clFlush(s->d->q)); }
            for (auto & s : es) CK(clFinish(s->d->q));
        });
        report("MB3", "ii_ES_compute_only", names, sc, "ids + kernels on all devices, wait all");
        check("MB3", "ES " + names + " first/last", e1 < tol && e2 < tol, "nmse " + sci(e1) + " / " + sci(e2));
        for (auto & s : es) { /* per device time inside the ES run */
            size_t rj = 0;
            Stats sj = bench(warm, iters, [&] { const Route & r = routes[rj++ & 63]; s->compute_only(r.ids, r.w); });
            char note[160]; snprintf(note, sizeof note, "slice %d rows, achieved %.2f GB/s", s->Mj, s->bytes_per_layer / (sj.med * 1e-6) / 1e9);
            report("MB3", "ii_ES_slice_compute_only", s->d->name, sj, note);
        }
    }
    return 0;
}

// ---------------------------------------------------------------------------------------------------------------------------------
// MB5: prompt batch. ES: every device gets all the tokens and a row slice; EP: every device gets a subset of the experts and the tokens that hit it.
// ---------------------------------------------------------------------------------------------------------------------------------
struct Pre {
    DevP d; bool ep = false;
    int Mj = 0, kb = 0, nlocal = 0;            // rows of the slice, K blocks of the down slice, local experts
    std::vector<int> gexp;                     // local expert -> global expert
    int nmax = 0;
    cl_mem wg = nullptr, wu = nullptr, wd = nullptr, dX = nullptr, dmeta = nullptr, dH = nullptr, dY = nullptr, dout = nullptr;
    cl_kernel kgu = nullptr, kdn = nullptr, ksc = nullptr;
    HostBuf hX, hmeta, hout;
    int esg = 0, esd = 0;
    size_t meta_cap = 0;
    // per call
    int ntok = 0, npair = 0, ntile = 0;
    std::vector<int> compact_to_tok;
    std::vector<int> meta;
    cl_event ev = nullptr;
    double host_prep_us = 0;

    void setup(const Args & a, const Canon & c, int n_max, bool pin, bool ep_mode, int row0, int rows, const std::vector<int> & experts) {
        ep = ep_mode; nmax = n_max; Mj = rows; kb = rows / 32; gexp = experts; nlocal = (int) experts.size();
        d->open();
        d->build("moe", MOE_SRC, moe_opts(a, *d));
        kgu = d->kern("moe", "k_pgu"); kdn = d->kern("moe", "k_pdown"); ksc = d->kern("moe", "k_scatter");
        const size_t esgs = (size_t) Mj * NBD * 18, esds = (size_t) DIM * kb * 18;
        esg = (int) esgs; esd = (int) esds;
        auto up = [&](cl_mem & m, size_t sz, const std::vector<uint8_t> & host) {
            m = mkbuf(d->ctx, CL_MEM_READ_ONLY, sz);
            CK(clEnqueueWriteBuffer(d->q, m, CL_TRUE, 0, sz, host.data(), 0, nullptr, nullptr));
        };
        std::vector<uint8_t> big(esgs * nlocal);
        for (int l = 0; l < nlocal; ++l) pack_r32(c.gate(gexp[l]), NBD, row0, Mj, 0, NBD, big.data() + l * esgs);
        up(wg, big.size(), big);
        for (int l = 0; l < nlocal; ++l) pack_r32(c.up(gexp[l]), NBD, row0, Mj, 0, NBD, big.data() + l * esgs);
        up(wu, big.size(), big);
        big.resize(esds * nlocal);
        for (int l = 0; l < nlocal; ++l) pack_r32(c.down(gexp[l]), NBF, 0, DIM, row0 / 32, kb, big.data() + l * esds);
        up(wd, big.size(), big);
        const size_t maxpair = (size_t) n_max * KUSE;
        dX = mkbuf(d->ctx, CL_MEM_READ_ONLY, (size_t) n_max * DIM * 4);
        meta_cap = 4 * (maxpair + n_max + 64) + 4 * (maxpair + 2 * n_max + 16) * 2 + 4096;   // generous, in ints
        dmeta = mkbuf(d->ctx, CL_MEM_READ_ONLY, meta_cap * 4);
        dH = mkbuf(d->ctx, CL_MEM_READ_WRITE, maxpair * Mj * 4);
        dY = mkbuf(d->ctx, CL_MEM_READ_WRITE, maxpair * DIM * 4);
        dout = mkbuf(d->ctx, CL_MEM_READ_WRITE, (size_t) n_max * DIM * 4);
        hX.init(d, (size_t) n_max * DIM * 4, pin); hmeta.init(d, meta_cap * 4, pin); hout.init(d, (size_t) n_max * DIM * 4, pin);
        meta.resize(meta_cap);
    }
    // build the pair lists for the tokens of `rt` (n routes), gather the hidden states, enqueue everything, no wait
    void submit(const float * X, const Route * rt, int n) {
        const double t0 = now_us();
        std::vector<int> lmap(NEXP, -1); for (int l = 0; l < nlocal; ++l) lmap[gexp[l]] = l;
        // pairs by local expert
        std::vector<std::vector<std::pair<int, float>>> bucket(nlocal);      // (token, weight)
        for (int t = 0; t < n; ++t) for (int s = 0; s < KUSE; ++s) { const int l = lmap[rt[t].ids[s]]; if (l >= 0) bucket[l].push_back({ t, rt[t].w[s] }); }
        std::vector<int> cidx(n, -1); compact_to_tok.clear();
        npair = 0;
        for (auto & b : bucket) npair += (int) b.size();
        if (!ep) { for (int t = 0; t < n; ++t) { cidx[t] = t; compact_to_tok.push_back(t); } }
        else { for (int l = 0; l < nlocal; ++l) for (auto & pr : bucket[l]) if (cidx[pr.first] < 0) { cidx[pr.first] = (int) compact_to_tok.size(); compact_to_tok.push_back(pr.first); } }
        ntok = (int) compact_to_tok.size();
        // meta: [ptok P][pw P][tiles 4*T][tok_off ntok+1][tok_pairs P]
        std::vector<int4_t> tiles;
        std::vector<int> ptok(npair), pw(npair);
        std::vector<int> cnt_tok(ntok + 1, 0);
        int p = 0;
        for (int l = 0; l < nlocal; ++l) {
            const int p0 = p;
            for (auto & pr : bucket[l]) { ptok[p] = cidx[pr.first]; memcpy(&pw[p], &pr.second, 4); cnt_tok[cidx[pr.first]]++; ++p; }
            for (int o = 0; o < (int) bucket[l].size(); o += 16) tiles.push_back({ l, p0 + o, std::min(16, (int) bucket[l].size() - o), 0 });
        }
        ntile = (int) tiles.size();
        std::vector<int> off(ntok + 1, 0); for (int t = 0; t < ntok; ++t) off[t + 1] = off[t] + cnt_tok[t];
        std::vector<int> tp(npair), fill(ntok, 0);
        for (int pp = 0; pp < npair; ++pp) { const int t = ptok[pp]; tp[off[t] + fill[t]++] = pp; }
        const int o_ptok = 0, o_pw = npair, o_tiles = (npair * 2 + 3) & ~3, o_off = o_tiles + 4 * ntile, o_tp = o_off + ntok + 1;
        const int total = o_tp + npair;
        if ((size_t) total > meta_cap) { fprintf(stderr, "meta overflow\n"); throw std::runtime_error("meta"); }
        int * m = (int *) hmeta.p;
        if (npair) { memcpy(m + o_ptok, ptok.data(), npair * 4); memcpy(m + o_pw, pw.data(), npair * 4); memcpy(m + o_tp, tp.data(), npair * 4); }
        if (ntile) memcpy(m + o_tiles, tiles.data(), ntile * 16);
        memcpy(m + o_off, off.data(), (ntok + 1) * 4);
        // gather the hidden states
        float * xs = (float *) hX.p;
        for (int c = 0; c < ntok; ++c) memcpy(xs + (size_t) c * DIM, X + (size_t) compact_to_tok[c] * DIM, DIM * 4);
        host_prep_us = now_us() - t0;
        if (ntok == 0) return;
        CK(clEnqueueWriteBuffer(d->q, dX, CL_FALSE, 0, (size_t) ntok * DIM * 4, hX.p, 0, nullptr, nullptr));
        CK(clEnqueueWriteBuffer(d->q, dmeta, CL_FALSE, 0, (size_t) total * 4, hmeta.p, 0, nullptr, nullptr));
        int mj = Mj, nbd = NBD, esgi = esg, esdi = esd, kbi = kb;
        CK(clSetKernelArg(kgu, 0, sizeof(cl_mem), &wg)); CK(clSetKernelArg(kgu, 1, sizeof(cl_mem), &wu)); CK(clSetKernelArg(kgu, 2, sizeof(cl_mem), &dX));
        CK(clSetKernelArg(kgu, 3, sizeof(cl_mem), &dmeta)); CK(clSetKernelArg(kgu, 4, sizeof(cl_mem), &dH));
        CK(clSetKernelArg(kgu, 5, sizeof mj, &mj)); CK(clSetKernelArg(kgu, 6, sizeof nbd, &nbd)); CK(clSetKernelArg(kgu, 7, sizeof esgi, &esgi));
        CK(clSetKernelArg(kgu, 8, sizeof o_ptok, &o_ptok)); CK(clSetKernelArg(kgu, 9, sizeof o_tiles, &o_tiles));
        CK(clSetKernelArg(kdn, 0, sizeof(cl_mem), &wd)); CK(clSetKernelArg(kdn, 1, sizeof(cl_mem), &dH)); CK(clSetKernelArg(kdn, 2, sizeof(cl_mem), &dmeta));
        CK(clSetKernelArg(kdn, 3, sizeof(cl_mem), &dY)); CK(clSetKernelArg(kdn, 4, sizeof kbi, &kbi)); CK(clSetKernelArg(kdn, 5, sizeof esdi, &esdi));
        CK(clSetKernelArg(kdn, 6, sizeof o_tiles, &o_tiles)); CK(clSetKernelArg(kdn, 7, sizeof o_pw, &o_pw));
        CK(clSetKernelArg(ksc, 0, sizeof(cl_mem), &dY)); CK(clSetKernelArg(ksc, 1, sizeof(cl_mem), &dmeta)); CK(clSetKernelArg(ksc, 2, sizeof(cl_mem), &dout));
        CK(clSetKernelArg(ksc, 3, sizeof ntok, &ntok)); CK(clSetKernelArg(ksc, 4, sizeof o_off, &o_off)); CK(clSetKernelArg(ksc, 5, sizeof o_tp, &o_tp));
        size_t gs1[2] = { (size_t) (Mj / 32) * 64, (size_t) ntile }, ls1[2] = { 64, 1 };
        CK(clEnqueueNDRangeKernel(d->q, kgu, 2, nullptr, gs1, ls1, 0, nullptr, nullptr));
        size_t gs2[2] = { (size_t) 64 * 64, (size_t) ntile };
        CK(clEnqueueNDRangeKernel(d->q, kdn, 2, nullptr, gs2, ls1, 0, nullptr, nullptr));
        size_t gs3 = (size_t) ntok * DIM;
        const size_t ls3 = 64;
        gs3 = (gs3 + ls3 - 1) / ls3 * ls3;
        CK(clEnqueueNDRangeKernel(d->q, ksc, 1, nullptr, &gs3, &ls3, 0, nullptr, nullptr));
        CK(clEnqueueReadBuffer(d->q, dout, CL_FALSE, 0, (size_t) ntok * DIM * 4, hout.p, 0, nullptr, &ev));
        CK(clFlush(d->q));
    }
    void wait() { if (ntok == 0) return; CK(clFinish(d->q)); clReleaseEvent(ev); ev = nullptr; }
    ~Pre() { for (cl_mem m : { wg, wu, wd, dX, dmeta, dH, dY, dout }) if (m) clReleaseMemObject(m); }
    struct int4_t { int x, y, z, w; };
};

static int mb5(const Args & a) {
    auto devs = cl_enumerate(a);
    if (devs.empty()) { fprintf(stderr, "mb5: no OpenCL device matched\n"); return 1; }
    const int n = (int) a.geti("ntok", 128);
    const size_t iters = a.geti("iters", 20), warm = a.geti("warmup", 3);
    const bool pin = a.geti("pin", 1) != 0;
    g_rng = 88172645463325252ull + a.geti("seed", 0);
    printf("# generating the layer, ntok=%d...\n", n);
    Canon canon; canon.gen();
    std::vector<Route> rt(n * 4); for (auto & r : rt) r = make_route();
    std::vector<std::vector<Route>> sets; for (int i = 0; i < 4; ++i) { std::vector<Route> s(n); for (auto & r : s) r = make_route(); sets.push_back(s); }
    const std::vector<float> X = make_x(n);
    const int nsample = 3;
    const double flops = (double) n * KUSE * 3.0 * 2.0 * DIM * FF;
    std::vector<int> all_exp(NEXP); std::iota(all_exp.begin(), all_exp.end(), 0);
    const double tol = 1e-8;
    auto verify = [&](const std::string & what, const std::vector<float> & got, const std::vector<Route> & set) {
        double worst = 0;
        for (int t = 0; t < nsample; ++t) {
            std::vector<double> ref; ref_moe_token(canon, X.data() + (size_t) t * DIM, set[t].ids, set[t].w, KUSE, ref);
            worst = std::max(worst, nmse(ref.data(), got.data() + (size_t) t * DIM, DIM));
        }
        check("MB5", what, worst < tol, "nmse(" + std::to_string(nsample) + " tokens) " + sci(worst));
    };
    // single device
    for (auto & dv : devs) {
        print_dev(dv);
        try {
            Pre p; p.d = dv; p.setup(a, canon, n, pin, false, 0, FF, all_exp);
            std::vector<float> got((size_t) n * DIM);
            p.submit(X.data(), sets[0].data(), n); p.wait();
            memcpy(got.data(), p.hout.p, got.size() * 4);
            verify("single " + dv->name + " n=" + std::to_string(n), got, sets[0]);
            size_t si = 0;
            Stats s = bench(warm, iters, [&] { p.submit(X.data(), sets[si++ & 3].data(), n); p.wait(); });
            char note[200]; snprintf(note, sizeof note, "ntok=%d pairs=%d tiles=%d  %.1f GFLOPS effective, %.1f tok/s (layer)", n, p.npair, p.ntile, flops / (s.med * 1e-6) / 1e9, n / (s.med * 1e-6));
            report("MB5", "single_full_cycle_n" + std::to_string(n), dv->name, s, note);
        } catch (std::exception & e) { fprintf(stderr, "mb5 %s: %s\n", dv->name.c_str(), e.what()); g_fail++; }
    }
    if (devs.size() >= 2) {
        std::vector<double> wt; std::string names;
        for (auto & d : devs) { wt.push_back(nominal_bw(d->name)); names += (names.empty() ? "" : "+") + d->name; }
        if (a.has("ratios")) { wt.clear(); for (auto & r : split(a.get("ratios"))) wt.push_back(atof(r.c_str())); }
        for (int mode = 0; mode < 2; ++mode) {              // 0: ES, 1: EP
            const char * mn = mode ? "EP" : "ES";
            try {
                std::vector<std::unique_ptr<Pre>> ps;
                std::string desc;
                const std::vector<int> units = split_units(mode ? NEXP : FF / 32, wt);
                int pos = 0;
                for (size_t i = 0; i < devs.size(); ++i) {
                    if (units[i] == 0) continue;
                    auto p = std::unique_ptr<Pre>(new Pre);
                    auto dv2 = std::make_shared<Dev>(*devs[i]);
                    dv2->ctx = nullptr; dv2->q = nullptr; dv2->progs.clear(); dv2->kernels.clear();
                    p->d = dv2;
                    if (mode == 0) { p->setup(a, canon, n, pin, false, pos * 32, units[i] * 32, all_exp); desc += devs[i]->name + ":" + std::to_string(units[i] * 32) + " rows "; }
                    else { std::vector<int> ex; for (int e = pos; e < pos + units[i]; ++e) ex.push_back(e); p->setup(a, canon, n, pin, true, 0, FF, ex); desc += devs[i]->name + ":" + std::to_string(units[i]) + " experts "; }
                    pos += units[i];
                    ps.push_back(std::move(p));
                }
                printf("# %s split: %s\n", mn, desc.c_str());
                std::vector<float> sum((size_t) n * DIM);
                double prep = 0;
                auto run = [&](const std::vector<Route> & set) {
                    for (auto & p : ps) p->submit(X.data(), set.data(), n);
                    for (auto & p : ps) p->wait();
                    std::fill(sum.begin(), sum.end(), 0.f);
                    for (auto & p : ps) {
                        const float * o = (const float *) p->hout.p;
                        for (int c = 0; c < p->ntok; ++c) { float * dst = sum.data() + (size_t) p->compact_to_tok[c] * DIM; const float * s = o + (size_t) c * DIM; for (int i = 0; i < DIM; ++i) dst[i] += s[i]; }
                    }
                    prep = 0; for (auto & p : ps) prep = std::max(prep, p->host_prep_us);
                };
                run(sets[0]);
                verify(std::string(mn) + " " + names + " n=" + std::to_string(n), sum, sets[0]);
                size_t si = 0;
                Stats s = bench(warm, iters, [&] { run(sets[si++ & 3]); });
                int maxpair = 0; for (auto & p : ps) maxpair = std::max(maxpair, p->npair);
                double xfer = 0; for (auto & p : ps) xfer += (double) p->ntok * DIM * 4 * 2 / 1e6;
                char note[300]; snprintf(note, sizeof note, "%s ; %.1f tok/s (layer) ; max pairs/device=%d ; host prep(last,max)=%.0f us ; x+out transfer %.1f MB total", desc.c_str(), n / (s.med * 1e-6), maxpair, prep, xfer);
                report("MB5", std::string(mn) + "_full_cycle_n" + std::to_string(n), names, s, note);
                // per device alone inside the split (compute + own transfers), to see max_j
                for (auto & p : ps) {
                    size_t sj = 0;
                    Stats ss = bench(warm, iters, [&] { p->submit(X.data(), sets[sj++ & 3].data(), n); p->wait(); });
                    char nn[200]; snprintf(nn, sizeof nn, "pairs=%d tokens=%d", p->npair, p->ntok);
                    report("MB5", std::string(mn) + "_part_alone_n" + std::to_string(n), p->d->name, ss, nn);
                }
            } catch (std::exception & e) { fprintf(stderr, "mb5 %s: %s\n", mn, e.what()); g_fail++; }
        }
    }
    return 0;
}

// ---------------------------------------------------------------------------------------------------------------------------------
// MB0 / MB1c: CUDA Driver API through dlopen
// ---------------------------------------------------------------------------------------------------------------------------------
typedef int CUresult; typedef int CUdevice;
typedef struct CUctx_st * CUcontext; typedef struct CUmod_st * CUmodule; typedef struct CUfunc_st * CUfunction;
typedef struct CUstream_st * CUstream; typedef struct CUevent_st * CUevent; typedef unsigned long long CUdeviceptr;
struct CUApi {
    void * h = nullptr;
    CUresult (*cuInit)(unsigned) = nullptr;
    CUresult (*cuDriverGetVersion)(int *) = nullptr;
    CUresult (*cuDeviceGetCount)(int *) = nullptr;
    CUresult (*cuDeviceGet)(CUdevice *, int) = nullptr;
    CUresult (*cuDeviceGetName)(char *, int, CUdevice) = nullptr;
    CUresult (*cuDeviceComputeCapability)(int *, int *, CUdevice) = nullptr;
    CUresult (*cuDeviceTotalMem)(size_t *, CUdevice) = nullptr;
    CUresult (*cuDeviceGetAttribute)(int *, int, CUdevice) = nullptr;
    CUresult (*cuCtxCreate)(CUcontext *, unsigned, CUdevice) = nullptr;
    CUresult (*cuCtxDestroy)(CUcontext) = nullptr;
    CUresult (*cuCtxSetCurrent)(CUcontext) = nullptr;
    CUresult (*cuCtxSynchronize)(void) = nullptr;
    CUresult (*cuModuleLoadData)(CUmodule *, const void *) = nullptr;
    CUresult (*cuModuleGetFunction)(CUfunction *, CUmodule, const char *) = nullptr;
    CUresult (*cuMemAlloc)(CUdeviceptr *, size_t) = nullptr;
    CUresult (*cuMemFree)(CUdeviceptr) = nullptr;
    CUresult (*cuMemcpyHtoD)(CUdeviceptr, const void *, size_t) = nullptr;
    CUresult (*cuMemcpyDtoH)(void *, CUdeviceptr, size_t) = nullptr;
    CUresult (*cuMemcpyHtoDAsync)(CUdeviceptr, const void *, size_t, CUstream) = nullptr;
    CUresult (*cuMemcpyDtoHAsync)(void *, CUdeviceptr, size_t, CUstream) = nullptr;
    CUresult (*cuMemHostAlloc)(void **, size_t, unsigned) = nullptr;
    CUresult (*cuMemFreeHost)(void *) = nullptr;
    CUresult (*cuStreamCreate)(CUstream *, unsigned) = nullptr;
    CUresult (*cuStreamSynchronize)(CUstream) = nullptr;
    CUresult (*cuStreamDestroy)(CUstream) = nullptr;
    CUresult (*cuEventCreate)(CUevent *, unsigned) = nullptr;
    CUresult (*cuEventRecord)(CUevent, CUstream) = nullptr;
    CUresult (*cuEventQuery)(CUevent) = nullptr;
    CUresult (*cuEventSynchronize)(CUevent) = nullptr;
    CUresult (*cuEventDestroy)(CUevent) = nullptr;
    CUresult (*cuLaunchKernel)(CUfunction, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, CUstream, void **, void **) = nullptr;
    CUresult (*cuGetErrorName)(CUresult, const char **) = nullptr;
    CUresult (*cuGetErrorString)(CUresult, const char **) = nullptr;

    bool load(const std::string & path) {
        const char * cands[] = { path.c_str(), "/usr/lib/x86_64-linux-gnu/libcuda.so.1", "libcuda.so.1", "libcuda.so" };
        for (const char * c : cands) { if (!c || !*c) continue; h = dlopen(c, RTLD_NOW | RTLD_LOCAL); if (h) { printf("# dlopen(%s) ok\n", c); break; } }
        if (!h) { printf("# dlopen(libcuda) failed: %s\n", dlerror()); return false; }
#define LD(n, sym) do { *(void **) &n = dlsym(h, sym); if (!n) { printf("# missing symbol %s\n", sym); return false; } } while (0)
        LD(cuInit, "cuInit"); LD(cuDriverGetVersion, "cuDriverGetVersion"); LD(cuDeviceGetCount, "cuDeviceGetCount"); LD(cuDeviceGet, "cuDeviceGet");
        LD(cuDeviceGetName, "cuDeviceGetName"); LD(cuDeviceComputeCapability, "cuDeviceComputeCapability"); LD(cuDeviceTotalMem, "cuDeviceTotalMem_v2");
        LD(cuDeviceGetAttribute, "cuDeviceGetAttribute");
        LD(cuCtxCreate, "cuCtxCreate_v2"); LD(cuCtxDestroy, "cuCtxDestroy_v2"); LD(cuCtxSetCurrent, "cuCtxSetCurrent"); LD(cuCtxSynchronize, "cuCtxSynchronize");
        LD(cuModuleLoadData, "cuModuleLoadData"); LD(cuModuleGetFunction, "cuModuleGetFunction");
        LD(cuMemAlloc, "cuMemAlloc_v2"); LD(cuMemFree, "cuMemFree_v2"); LD(cuMemcpyHtoD, "cuMemcpyHtoD_v2"); LD(cuMemcpyDtoH, "cuMemcpyDtoH_v2");
        LD(cuMemcpyHtoDAsync, "cuMemcpyHtoDAsync_v2"); LD(cuMemcpyDtoHAsync, "cuMemcpyDtoHAsync_v2"); LD(cuMemHostAlloc, "cuMemHostAlloc"); LD(cuMemFreeHost, "cuMemFreeHost");
        LD(cuStreamCreate, "cuStreamCreate"); LD(cuStreamSynchronize, "cuStreamSynchronize"); LD(cuStreamDestroy, "cuStreamDestroy_v2");
        LD(cuEventCreate, "cuEventCreate"); LD(cuEventRecord, "cuEventRecord"); LD(cuEventQuery, "cuEventQuery"); LD(cuEventSynchronize, "cuEventSynchronize");
        LD(cuEventDestroy, "cuEventDestroy_v2"); LD(cuLaunchKernel, "cuLaunchKernel");
        LD(cuGetErrorName, "cuGetErrorName"); LD(cuGetErrorString, "cuGetErrorString");
#undef LD
        return true;
    }
    std::string err(CUresult r) const {
        const char * n = nullptr, * s = nullptr;
        if (cuGetErrorName) cuGetErrorName(r, &n);
        if (cuGetErrorString) cuGetErrorString(r, &s);
        return std::to_string(r) + " " + (n ? n : "?") + " (" + (s ? s : "?") + ")";
    }
};

// PTX ISA 4.0 / sm_20: accepted by the JIT of a 390 (CUDA 9.1) driver for Fermi and Kepler. in * 2 + 1, then an LCG loop of `iters` rounds (busy_k)
static const char * PTX_SRC = R"PTX(
.version 4.0
.target sm_20
.address_size 64

.visible .entry rt_k(.param .u64 p_in, .param .u64 p_out, .param .u32 p_n)
{
    .reg .pred %p<2>;
    .reg .s32 %r<6>;
    .reg .f32 %f<4>;
    .reg .s64 %rd<8>;
    ld.param.u64 %rd1, [p_in];
    ld.param.u64 %rd2, [p_out];
    ld.param.u32 %r1, [p_n];
    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, %ntid.x;
    mov.u32 %r4, %tid.x;
    mad.lo.s32 %r5, %r2, %r3, %r4;
    setp.ge.s32 %p1, %r5, %r1;
    @%p1 bra DONE;
    mul.wide.s32 %rd3, %r5, 4;
    add.s64 %rd4, %rd1, %rd3;
    add.s64 %rd5, %rd2, %rd3;
    ld.global.f32 %f1, [%rd4];
    fma.rn.f32 %f2, %f1, 0f40000000, 0f3F800000;
    st.global.f32 [%rd5], %f2;
DONE:
    ret;
}

.visible .entry busy_k(.param .u64 p_in, .param .u64 p_out, .param .u32 p_n, .param .u32 p_iters)
{
    .reg .pred %p<5>;
    .reg .s32 %r<10>;
    .reg .f32 %f<4>;
    .reg .s64 %rd<8>;
    ld.param.u64 %rd1, [p_in];
    ld.param.u64 %rd2, [p_out];
    ld.param.u32 %r1, [p_n];
    ld.param.u32 %r6, [p_iters];
    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, %ntid.x;
    mov.u32 %r4, %tid.x;
    mad.lo.s32 %r5, %r2, %r3, %r4;
    setp.ge.s32 %p1, %r5, %r1;
    @%p1 bra DONE;
    mov.s32 %r7, 0;
    mov.s32 %r8, %r5;
LOOP:
    setp.ge.s32 %p2, %r7, %r6;
    @%p2 bra END;
    mad.lo.s32 %r8, %r8, 1664525, 1013904223;
    add.s32 %r7, %r7, 1;
    bra LOOP;
END:
    mul.wide.s32 %rd3, %r5, 4;
    add.s64 %rd4, %rd1, %rd3;
    add.s64 %rd5, %rd2, %rd3;
    ld.global.f32 %f1, [%rd4];
    fma.rn.f32 %f2, %f1, 0f40000000, 0f3F800000;
    setp.eq.s32 %p3, %r8, 305419896;
    setp.eq.s32 %p4, %r5, 999999;
    and.pred %p3, %p3, %p4;
    @%p3 add.f32 %f2, %f2, 0f3F800000;
    st.global.f32 [%rd5], %f2;
DONE:
    ret;
}
)PTX";

struct CuDev {
    CUApi * api; int idx = -1; CUdevice dev = 0; std::string name; int ccmaj = 0, ccmin = 0; size_t mem = 0;
    CUcontext ctx = nullptr; CUmodule mod = nullptr; CUfunction f_rt = nullptr, f_busy = nullptr;
    CUdeviceptr din = 0, dout = 0; CUstream st = nullptr; CUevent ev = nullptr;
    void * pin_s = nullptr, * pin_d = nullptr;
    std::vector<float> src, dst;
    int iters = 0;
    bool ctx_ok = false, mod_ok = false;
    std::string ctx_err, mod_err;

    bool create() {
        CUresult r = api->cuCtxCreate(&ctx, 0, dev);
        if (r != 0) { ctx_err = api->err(r); return false; }
        ctx_ok = true;
        return true;
    }
    void cur() { api->cuCtxSetCurrent(ctx); }
    bool load() {
        cur();
        CUresult r = api->cuModuleLoadData(&mod, PTX_SRC);
        if (r != 0) { mod_err = "cuModuleLoadData: " + api->err(r); return false; }
        r = api->cuModuleGetFunction(&f_rt, mod, "rt_k"); if (r) { mod_err = "GetFunction rt_k: " + api->err(r); return false; }
        r = api->cuModuleGetFunction(&f_busy, mod, "busy_k"); if (r) { mod_err = "GetFunction busy_k: " + api->err(r); return false; }
        if (api->cuMemAlloc(&din, RT_N * 4) || api->cuMemAlloc(&dout, RT_N * 4)) { mod_err = "cuMemAlloc"; return false; }
        api->cuStreamCreate(&st, 0);
        api->cuEventCreate(&ev, 0);
        if (api->cuMemHostAlloc(&pin_s, RT_N * 4, 0) || api->cuMemHostAlloc(&pin_d, RT_N * 4, 0)) { pin_s = pin_d = nullptr; }
        src.resize(RT_N); dst.assign(RT_N, 0.f);
        for (auto & v : src) v = frand();
        if (pin_s) memcpy(pin_s, src.data(), RT_N * 4);
        mod_ok = true;
        return true;
    }
    CUresult launch(bool busy) {
        int n = RT_N; CUdeviceptr i = din, o = dout; int it = iters;
        void * args_rt[] = { &i, &o, &n };
        void * args_b[]  = { &i, &o, &n, &it };
        return api->cuLaunchKernel(busy ? f_busy : f_rt, RT_N / 256, 1, 1, 256, 1, 1, 0, st, busy ? args_b : args_rt, nullptr);
    }
    bool verify(const float * t) const { for (int i = 0; i < RT_N; ++i) if (t[i] != src[i] * 2.0f + 1.0f) return false; return true; }
    // variants: 0 sync copies + ctx sync, 1 async pageable + stream sync, 2 async pinned + stream sync, 3 async pinned + event busy poll
    void cycle(int v) {
        cur();
        const void * s = (v >= 2 && pin_s) ? pin_s : src.data();
        void * t = (v >= 2 && pin_d) ? pin_d : dst.data();
        if (v == 0) {
            api->cuMemcpyHtoD(din, s, RT_N * 4); launch(false); api->cuMemcpyDtoH(t, dout, RT_N * 4);
        } else {
            submit(s, t);
            if (v == 3) { while (api->cuEventQuery(ev) != 0) {} } else api->cuStreamSynchronize(st);
        }
    }
    void submit(const void * s, void * t, bool busy = false) {
        api->cuMemcpyHtoDAsync(din, s, RT_N * 4, st); launch(busy); api->cuMemcpyDtoHAsync(t, dout, RT_N * 4, st); api->cuEventRecord(ev, st);
    }
    void * hs(bool pin) { return pin && pin_s ? pin_s : (void *) src.data(); }
    void * hd(bool pin) { return pin && pin_d ? pin_d : (void *) dst.data(); }
};

static std::vector<std::unique_ptr<CuDev>> cu_open(CUApi & api, const Args & a, bool create_ctx, bool & init_ok) {
    std::vector<std::unique_ptr<CuDev>> v;
    init_ok = false;
    CUresult r = api.cuInit(0);
    printf("# cuInit: %s\n", api.err(r).c_str());
    if (r) return v;
    init_ok = true;
    int ver = 0; api.cuDriverGetVersion(&ver);
    int cnt = 0; api.cuDeviceGetCount(&cnt);
    printf("# CUDA driver version %d, %d device(s)\n", ver, cnt);
    std::vector<int> order;
    if (a.has("cu-devs")) for (auto & s : split(a.get("cu-devs"))) order.push_back(atoi(s.c_str()));
    else for (int i = 0; i < cnt; ++i) order.push_back(i);
    if (a.get("cu-order") == "rev" && !a.has("cu-devs")) std::reverse(order.begin(), order.end());
    for (int i : order) {
        auto d = std::unique_ptr<CuDev>(new CuDev); d->api = &api; d->idx = i;
        if (api.cuDeviceGet(&d->dev, i)) continue;
        char nm[128] = { 0 }; api.cuDeviceGetName(nm, sizeof nm, d->dev); d->name = nm;
        api.cuDeviceComputeCapability(&d->ccmaj, &d->ccmin, d->dev);
        api.cuDeviceTotalMem(&d->mem, d->dev);
        printf("# cuda device %d: %s  sm_%d%d  %.0f MiB\n", i, nm, d->ccmaj, d->ccmin, d->mem / 1048576.0);
        v.push_back(std::move(d));
    }
    if (create_ctx) {
        for (auto & d : v) {
            const double t0 = now_us();
            const bool ok = d->create();
            printf("MB0   cuCtxCreate(device %d %s sm_%d%d): %s  (%.0f us)\n", d->idx, d->name.c_str(), d->ccmaj, d->ccmin, ok ? "OK" : d->ctx_err.c_str(), now_us() - t0);
            if (g_csv) fprintf(g_csv, "MB0,cuCtxCreate,%s,1,%.1f,0,0,0,%s\n", no_comma(d->name).c_str(), now_us() - t0, ok ? "OK" : no_comma(d->ctx_err).c_str());
            if (ok) {
                const bool lo = d->load();
                printf("MB0   cuModuleLoadData (PTX .version 4.0 .target sm_20 JIT) + alloc on %s: %s\n", d->name.c_str(), lo ? "OK" : d->mod_err.c_str());
                if (lo) {
                    d->cur();
                    api.cuMemcpyHtoD(d->din, d->src.data(), RT_N * 4);
                    CUresult lr = d->launch(false);
                    CUresult sr = api.cuCtxSynchronize();
                    api.cuMemcpyDtoH(d->dst.data(), d->dout, RT_N * 4);
                    const bool vr = d->verify(d->dst.data());
                    printf("MB0   kernel launch on %s: launch=%s sync=%s result=%s\n", d->name.c_str(), api.err(lr).c_str(), api.err(sr).c_str(), vr ? "correct" : "WRONG");
                    check("MB0", "launch " + d->name, lr == 0 && sr == 0 && vr);
                }
            }
        }
    }
    return v;
}

static int mb0(const Args & a) {
    CUApi api;
    if (!api.load(a.get("libcuda", "/usr/lib/x86_64-linux-gnu/libcuda.so.1"))) { printf("MB0   SKIP: no CUDA driver library on this machine\n"); return 0; }
    bool ok; auto v = cu_open(api, a, true, ok);
    if (!ok) return 1;
    int made = 0, good = 0;
    for (auto & d : v) { made += d->ctx_ok; good += d->mod_ok; }
    printf("MB0   contexts created %d / %zu, modules + launches ok %d / %zu\n", made, v.size(), good, v.size());
    // all contexts alive in one process: interleaved rounds
    const long rounds = a.geti("iters", 1000);
    long bad = 0;
    for (long r = 0; r < rounds; ++r)
        for (auto & d : v) {
            if (!d->mod_ok) continue;
            d->cur();
            for (auto & x : d->dst) x = 0;
            api.cuMemcpyHtoD(d->din, d->src.data(), RT_N * 4);
            if (d->launch(false)) { bad++; continue; }
            api.cuCtxSynchronize();
            api.cuMemcpyDtoH(d->dst.data(), d->dout, RT_N * 4);
            if (!d->verify(d->dst.data())) bad++;
        }
    printf("MB0   interleaved rounds over all live contexts: %ld rounds x %d devices, %ld failures\n", rounds, good, bad);
    check("MB0", "interleaved launches on all contexts", bad == 0 && good > 0);
    check("MB0", "all devices usable in one process", good == (int) v.size() && !v.empty(), std::to_string(good) + " of " + std::to_string(v.size()));
    return 0;
}

static int mb1c(const Args & a) {
    CUApi api;
    if (!api.load(a.get("libcuda", "/usr/lib/x86_64-linux-gnu/libcuda.so.1"))) { printf("MB1C  SKIP: no CUDA driver library\n"); return 0; }
    bool ok; auto v = cu_open(api, a, true, ok);
    if (!ok) return 1;
    const size_t iters = a.geti("iters", 10000), warm = a.geti("warmup", 200);
    std::vector<CuDev *> live;
    for (auto & d : v) if (d->mod_ok) live.push_back(d.get());
    const char * vn[] = { "cuda_sync_copies", "cuda_async_pageable_streamsync", "cuda_async_pinned_streamsync", "cuda_async_pinned_eventpoll" };
    for (auto * d : live) {
        for (int var = 0; var < 4; ++var) {
            d->cur(); d->cycle(var); bool good = d->verify((var >= 2 && d->pin_d) ? (float *) d->pin_d : d->dst.data());
            Stats s = bench(warm, iters, [&] { d->cycle(var); });
            report("MB1C", vn[var], d->name, s, good ? "check=OK" : "check=FAIL");
            if (!good) g_fail++;
        }
        d->cur();
        Stats s = bench(warm, iters, [&] { d->launch(false); d->api->cuStreamSynchronize(d->st); });
        report("MB1C", "cuda_kernel_streamsync", d->name, s, "T_k: tiny kernel + sync");
        s = bench(warm, iters, [&] { d->api->cuMemcpyHtoD(d->din, d->src.data(), RT_N * 4); });
        report("MB1C", "cuda_write8k_sync", d->name, s);
        s = bench(warm, iters, [&] { d->api->cuMemcpyDtoH(d->dst.data(), d->dout, RT_N * 4); });
        report("MB1C", "cuda_read8k_sync", d->name, s);
    }
    // all devices at once from one thread (cuCtxSetCurrent per device)
    if (live.size() >= 2) {
        std::string names; for (auto * d : live) names += (names.empty() ? "" : "+") + d->name;
        for (int kind = 0; kind < 2; ++kind) {
            for (auto * d : live) {
                d->iters = 0;
                if (kind) {
                    int n = 2000;
                    for (int c = 0; c < 8; ++c) {
                        d->iters = n; d->cur();
                        Stats s = bench(3, 15, [&] { d->launch(true); d->api->cuStreamSynchronize(d->st); });
                        const double f = 500.0 / std::max(1.0, s.med);
                        if (f > 0.85 && f < 1.15) break;
                        n = std::max(1, (int) (n * f));
                    }
                    printf("# %s: busy kernel iters=%d\n", d->name.c_str(), d->iters);
                }
            }
            const char * kn = kind ? "busy500us" : "tiny";
            double mx = 0, sum = 0;
            for (auto * d : live) {
                Stats s = bench(warm, kind ? std::min<size_t>(iters, 3000) : iters, [&] { d->cur(); d->submit(d->hs(true), d->hd(true), kind); d->api->cuStreamSynchronize(d->st); });
                report("MB1C", std::string("solo_") + kn, d->name, s);
                mx = std::max(mx, s.med); sum += s.med;
            }
            bool allok = true;
            Stats p = bench(warm, kind ? std::min<size_t>(iters, 3000) : iters, [&] {
                for (auto * d : live) { d->cur(); d->submit(d->hs(true), d->hd(true), kind); }
                for (auto * d : live) { d->cur(); d->api->cuStreamSynchronize(d->st); }
            });
            for (auto * d : live) allok = allok && d->verify((float *) d->hd(true));
            char buf[200]; snprintf(buf, sizeof buf, "max(solo)=%.1f sum(solo)=%.1f parallel=%.1f ; parallel/max=%.2f parallel/sum=%.2f", mx, sum, p.med, p.med / mx, p.med / sum);
            report("MB1C", std::string("parallel_1thread_") + kn, names, p, buf);
            check("MB1C", std::string("parallel results ") + kn, allok);
        }
    }
    return 0;
}

// ---------------------------------------------------------------------------------------------------------------------------------
// ggml part: mb3g / mb4 / mb5g
// ---------------------------------------------------------------------------------------------------------------------------------
#ifndef MRB_NO_GGML
typedef void (*set_threads_t)(ggml_backend_t, int);

static std::vector<ggml_backend_dev_t> ggml_pick(const Args & a, bool gpu_default_only) {
    std::string lib = a.get("lib-dir"); if (lib.empty()) { const char * e = getenv("GGML_LIB_DIR"); lib = e ? e : "."; }
    ggml_backend_load_all_from_path(lib.c_str());
    std::vector<ggml_backend_dev_t> all;
    printf("# ggml devices (lib dir %s):\n", lib.c_str());
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t d = ggml_backend_dev_get(i);
        size_t fr = 0, tot = 0; ggml_backend_dev_memory(d, &fr, &tot);
        printf("#   %s : %s (type %d, %.0f / %.0f MiB free)\n", ggml_backend_dev_name(d), ggml_backend_dev_description(d), (int) ggml_backend_dev_type(d), fr / 1048576.0, tot / 1048576.0);
        all.push_back(d);
    }
    std::vector<ggml_backend_dev_t> sel;
    const std::vector<std::string> names = split(a.get("ggml-devs"));
    if (names.empty()) {
        for (auto d : all) {
            const int t = (int) ggml_backend_dev_type(d);
            if (t == (int) GGML_BACKEND_DEVICE_TYPE_ACCEL) continue;
            if (gpu_default_only && t == (int) GGML_BACKEND_DEVICE_TYPE_CPU) continue;
            sel.push_back(d);
        }
        if (sel.empty()) sel = all;
    } else {
        for (auto & n : names) for (auto d : all) if (n == ggml_backend_dev_name(d) || (n == "CPU" && ggml_backend_dev_type(d) == GGML_BACKEND_DEVICE_TYPE_CPU)) sel.push_back(d);
    }
    return sel;
}

static ggml_backend_t ggml_open(ggml_backend_dev_t dev, int threads) {
    ggml_backend_t be = ggml_backend_dev_init(dev, nullptr);
    if (be && ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU) {
        auto fn = (set_threads_t) ggml_backend_reg_get_proc_address(ggml_backend_dev_backend_reg(dev), "ggml_backend_set_n_threads");
        if (fn) fn(be, threads);
    }
    return be;
}

struct GMoe {
    ggml_backend_t be = nullptr; ggml_context * wctx = nullptr, * gctx = nullptr; ggml_backend_buffer_t wbuf = nullptr; ggml_gallocr_t ga = nullptr;
    ggml_tensor * Wg = nullptr, * Wu = nullptr, * Wd = nullptr, * x = nullptr, * ids = nullptr, * wts = nullptr, * out = nullptr;
    ggml_cgraph * gf = nullptr; int n = 1; int unsupported = 0;
    // rows != FF: the expert slice (ES) with the rows [row0, row0 + rows) of the F dimension (gate / up rows, down K blocks)
    bool init(ggml_backend_t b, const Canon & c, int n_tok, int row0 = 0, int rows = FF) {
        be = b; n = n_tok;
        wctx = ggml_init({ ggml_tensor_overhead() * 8, nullptr, true });
        Wg = ggml_new_tensor_3d(wctx, GGML_TYPE_Q4_0, DIM, rows, NEXP);
        Wu = ggml_new_tensor_3d(wctx, GGML_TYPE_Q4_0, DIM, rows, NEXP);
        Wd = ggml_new_tensor_3d(wctx, GGML_TYPE_Q4_0, rows, DIM, NEXP);
        wbuf = ggml_backend_alloc_ctx_tensors(wctx, be);
        if (!wbuf) return false;
        ggml_backend_buffer_set_usage(wbuf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        if (rows == FF) {
            ggml_backend_tensor_set(Wg, c.g.data(), 0, ggml_nbytes(Wg));
            ggml_backend_tensor_set(Wu, c.u.data(), 0, ggml_nbytes(Wu));
            ggml_backend_tensor_set(Wd, c.d.data(), 0, ggml_nbytes(Wd));
        } else {
            const int kb = rows / 32;
            std::vector<Blk> sg((size_t) NEXP * rows * NBD), su(sg.size()), sd((size_t) NEXP * DIM * kb);
            for (int e = 0; e < NEXP; ++e) {
                memcpy(&sg[(size_t) e * rows * NBD], c.g.data() + ((size_t) e * FF + row0) * NBD, (size_t) rows * NBD * sizeof(Blk));
                memcpy(&su[(size_t) e * rows * NBD], c.u.data() + ((size_t) e * FF + row0) * NBD, (size_t) rows * NBD * sizeof(Blk));
                for (int m = 0; m < DIM; ++m) memcpy(&sd[((size_t) e * DIM + m) * kb], c.d.data() + ((size_t) e * DIM + m) * NBF + row0 / 32, (size_t) kb * sizeof(Blk));
            }
            ggml_backend_tensor_set(Wg, sg.data(), 0, ggml_nbytes(Wg));
            ggml_backend_tensor_set(Wu, su.data(), 0, ggml_nbytes(Wu));
            ggml_backend_tensor_set(Wd, sd.data(), 0, ggml_nbytes(Wd));
        }
        gctx = ggml_init({ ggml_tensor_overhead() * 128 + ggml_graph_overhead(), nullptr, true });
        x   = ggml_new_tensor_3d(gctx, GGML_TYPE_F32, DIM, 1, n);
        ids = ggml_new_tensor_2d(gctx, GGML_TYPE_I32, KUSE, n);
        wts = ggml_new_tensor_3d(gctx, GGML_TYPE_F32, 1, KUSE, n);
        ggml_set_input(x); ggml_set_input(ids); ggml_set_input(wts);
        ggml_tensor * gate = ggml_mul_mat_id(gctx, Wg, x, ids);
        ggml_tensor * up   = ggml_mul_mat_id(gctx, Wu, x, ids);
        ggml_tensor * h    = ggml_swiglu_split(gctx, gate, up);
        ggml_tensor * ex   = ggml_mul_mat_id(gctx, Wd, h, ids);
        ex = ggml_mul(gctx, ex, wts);
        out = ggml_view_2d(gctx, ex, DIM, n, ex->nb[2], 0);
        for (int i = 1; i < KUSE; ++i) out = ggml_add(gctx, out, ggml_view_2d(gctx, ex, DIM, n, ex->nb[2], i * ex->nb[1]));
        ggml_set_output(out);
        gf = ggml_new_graph(gctx);
        ggml_build_forward_expand(gf, out);
        for (int i = 0; i < ggml_graph_n_nodes(gf); ++i) if (!ggml_backend_supports_op(be, ggml_graph_node(gf, i))) unsupported++;
        ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(be));
        return ggml_gallocr_alloc_graph(ga, gf);
    }
    void compute() { ggml_backend_graph_compute(be, gf); ggml_backend_synchronize(be); }
    ~GMoe() { if (ga) ggml_gallocr_free(ga); if (wbuf) ggml_backend_buffer_free(wbuf); if (gctx) ggml_free(gctx); if (wctx) ggml_free(wctx); }
};

static int mb3g_5g(const Args & a, int n) {   // n == 1: mb3g, otherwise mb5g
    const char * mb = n == 1 ? "MB3G" : "MB5G";
    auto devs = ggml_pick(a, false);
    if (devs.empty()) { fprintf(stderr, "%s: no ggml device\n", mb); return 1; }
    g_rng = 88172645463325252ull + a.geti("seed", 0);
    const size_t iters = a.geti("iters", n == 1 ? 1000 : 20), warm = a.geti("warmup", n == 1 ? 50 : 3);
    printf("# generating the layer...\n");
    Canon canon; canon.gen();
    const std::vector<float> X = make_x(n);
    std::vector<std::vector<Route>> sets; for (int i = 0; i < 4; ++i) { std::vector<Route> s(n); for (auto & r : s) r = make_route(); sets.push_back(s); }
    const double flops = (double) n * KUSE * 3.0 * 2.0 * DIM * FF;
    for (auto dev : devs) {
        const std::string dn = std::string(ggml_backend_dev_name(dev)) + " " + ggml_backend_dev_description(dev);
        ggml_backend_t be = ggml_open(dev, (int) a.geti("threads", 2));
        if (!be) { fprintf(stderr, "%s: cannot init %s\n", mb, dn.c_str()); g_fail++; continue; }
        GMoe g;
        if (!g.init(be, canon, n)) { fprintf(stderr, "%s: graph alloc failed on %s\n", mb, dn.c_str()); g_fail++; ggml_backend_free(be); continue; }
        auto setin = [&](const std::vector<Route> & set, bool with_x) {
            std::vector<int32_t> ids((size_t) n * KUSE); std::vector<float> w((size_t) n * KUSE);
            for (int t = 0; t < n; ++t) for (int s = 0; s < KUSE; ++s) { ids[(size_t) t * KUSE + s] = set[t].ids[s]; w[(size_t) t * KUSE + s] = set[t].w[s]; }
            ggml_backend_tensor_set(g.ids, ids.data(), 0, ids.size() * 4);
            ggml_backend_tensor_set(g.wts, w.data(), 0, w.size() * 4);
            if (with_x) ggml_backend_tensor_set(g.x, X.data(), 0, X.size() * 4);
        };
        std::vector<float> got((size_t) n * DIM);
        setin(sets[0], true); g.compute();
        ggml_backend_tensor_get(g.out, got.data(), 0, got.size() * 4);
        double worst = 0;
        for (int t = 0; t < std::min(n, 3); ++t) {
            std::vector<double> ref; ref_moe_token(canon, X.data() + (size_t) t * DIM, sets[0][t].ids, sets[0][t].w, KUSE, ref);
            worst = std::max(worst, nmse(ref.data(), got.data() + (size_t) t * DIM, DIM));
        }
        check(mb, "result " + dn, worst < 1e-3, "nmse " + sci(worst) + (g.unsupported ? "  (" + std::to_string(g.unsupported) + " nodes unsupported by the backend)" : ""));
        size_t si = 0;
        char note[200];
        Stats s1 = bench(warm, iters, [&] { setin(sets[si++ & 3], false); g.compute(); });
        snprintf(note, sizeof note, "ids+weights set, graph_compute, synchronize ; %.1f GFLOPS ; unsupported nodes %d", flops / (s1.med * 1e-6) / 1e9, g.unsupported);
        report(mb, "ggml_compute_only_n" + std::to_string(n), dn, s1, note);
        si = 0;
        Stats s2 = bench(warm, iters, [&] { setin(sets[si++ & 3], true); g.compute(); ggml_backend_tensor_get(g.out, got.data(), 0, got.size() * 4); });
        snprintf(note, sizeof note, "x + ids + weights set, compute, out get ; %.1f tok/s (layer)", n / (s2.med * 1e-6));
        report(mb, "ggml_full_cycle_n" + std::to_string(n), dn, s2, note);
        const double mbytes = (double) KUSE * 3 * DIM * FF / 32 * 18;
        if (n == 1) printf("%-5s       achieved weight bandwidth (compute only): %.2f GB/s\n", mb, mbytes / (s1.med * 1e-6) / 1e9);
    }
    return 0;
}


// MB3x: the ES of MB3 (ii) with a ggml backend (the official Vulkan of the WX 2100) holding one of the slices and the OpenCL NVIDIA cards the others,
// all in one process: NVIDIA kernels are submitted first (asynchronous), then the ggml graph runs (it blocks), then the NVIDIA devices are waited for.
static int mb3x(const Args & a) {
    auto cld = cl_enumerate(a);
    auto gd = ggml_pick(a, true);
    if (gd.size() != 1 || cld.empty()) { fprintf(stderr, "mb3x: needs one ggml device (--ggml-devs) and >= 1 OpenCL device (--devices)\n"); return 1; }
    const size_t iters = a.geti("iters", 1000), warm = a.geti("warmup", 50);
    const bool pin = a.geti("pin", 1) != 0;
    g_rng = 88172645463325252ull + a.geti("seed", 0);
    printf("# generating the layer...\n");
    Canon canon; canon.gen();
    const std::vector<float> x = make_x(1);
    std::vector<Route> routes; for (int i = 0; i < 64; ++i) routes.push_back(make_route());
    std::vector<double> ref; ref_moe_token(canon, x.data(), routes[0].ids, routes[0].w, KUSE, ref);
    const std::string gname = std::string(ggml_backend_dev_name(gd[0])) + " " + ggml_backend_dev_description(gd[0]);
    std::vector<double> wt = { nominal_bw(gname) };
    for (auto & d : cld) wt.push_back(nominal_bw(d->name));
    if (a.has("ratios")) { wt.clear(); for (auto & r : split(a.get("ratios"))) wt.push_back(atof(r.c_str())); }
    const std::vector<int> groups = split_units(FF / 32, wt);
    std::string desc = gname + ":" + std::to_string(groups[0] * 32) + " rows";
    ggml_backend_t be = ggml_open(gd[0], 2);
    GMoe g;
    int row0 = 0;
    if (groups[0] > 0) { if (!g.init(be, canon, 1, 0, groups[0] * 32)) { fprintf(stderr, "mb3x: ggml graph alloc failed\n"); return 1; } }
    row0 = groups[0] * 32;
    std::vector<std::unique_ptr<Dec>> cl;
    for (size_t i = 0; i < cld.size(); ++i) {
        if (groups[i + 1] == 0) continue;
        auto s = std::unique_ptr<Dec>(new Dec); s->d = cld[i];
        s->setup(a, canon, row0, groups[i + 1] * 32, pin);
        desc += " " + cld[i]->name + ":" + std::to_string(groups[i + 1] * 32) + " rows";
        row0 += groups[i + 1] * 32;
        cl.push_back(std::move(s));
    }
    printf("# ES split: %s\n", desc.c_str());
    std::vector<float> gout(DIM), sum(DIM);
    auto gset = [&](const Route & r, bool withx) {
        int32_t ids[KUSE]; float w[KUSE];
        for (int i = 0; i < KUSE; ++i) { ids[i] = r.ids[i]; w[i] = r.w[i]; }
        ggml_backend_tensor_set(g.ids, ids, 0, sizeof ids); ggml_backend_tensor_set(g.wts, w, 0, sizeof w);
        if (withx) ggml_backend_tensor_set(g.x, x.data(), 0, DIM * 4);
    };
    auto run = [&](const Route & r) {
        for (auto & s : cl) s->submit(x.data(), r.ids, r.w);
        if (groups[0] > 0) { gset(r, true); g.compute(); ggml_backend_tensor_get(g.out, gout.data(), 0, DIM * 4); }
        for (auto & s : cl) s->wait();
        if (groups[0] > 0) for (int i = 0; i < DIM; ++i) sum[i] = gout[i]; else for (int i = 0; i < DIM; ++i) sum[i] = 0.f;
        for (auto & s : cl) { const float * o = s->out(); for (int i = 0; i < DIM; ++i) sum[i] += o[i]; }
    };
    run(routes[0]);
    const double e1 = nmse(ref.data(), sum.data(), DIM);
    size_t ri = 0;
    Stats st = bench(warm, iters, [&] { run(routes[ri++ & 63]); });
    const Route & lr = routes[(ri - 1) & 63];
    std::vector<double> ref2; ref_moe_token(canon, x.data(), lr.ids, lr.w, KUSE, ref2);
    const double e2 = nmse(ref2.data(), sum.data(), DIM);
    report("MB3X", "ES_ggml+opencl_full_cycle", gname, st, "ggml slice + OpenCL slices, one thread: " + desc);
    check("MB3X", "ES result", e1 < 1e-3 && e2 < 1e-3, "nmse " + sci(e1) + " / " + sci(e2));
    ri = 0;
    if (groups[0] > 0) {
        Stats sg = bench(warm, iters, [&] { const Route & r = routes[ri++ & 63]; gset(r, true); g.compute(); ggml_backend_tensor_get(g.out, gout.data(), 0, DIM * 4); });
        report("MB3X", "slice_alone_ggml_full_cycle", gname, sg, std::to_string(groups[0] * 32) + " rows");
    }
    for (auto & s : cl) {
        size_t rj = 0;
        Stats sj = bench(warm, iters, [&] { const Route & r = routes[rj++ & 63]; s->submit(x.data(), r.ids, r.w); s->wait(); });
        report("MB3X", "slice_alone_cl_full_cycle", s->d->name, sj, std::to_string(s->Mj) + " rows");
    }
    ggml_backend_free(be);
    return 0;
}

static int mb4(const Args & a) {
    auto devs = ggml_pick(a, true);
    if (devs.empty()) { fprintf(stderr, "mb4: no ggml device\n"); return 1; }
    const size_t iters = a.geti("iters", 10000), warm = a.geti("warmup", 200);
    for (auto dev : devs) {
        const std::string dn = std::string(ggml_backend_dev_name(dev)) + " " + ggml_backend_dev_description(dev);
        ggml_backend_t be = ggml_open(dev, (int) a.geti("threads", 2));
        if (!be) { fprintf(stderr, "mb4: cannot init %s\n", dn.c_str()); g_fail++; continue; }
        ggml_context * wctx = ggml_init({ ggml_tensor_overhead() * 8, nullptr, true });
        ggml_tensor * t1 = ggml_new_tensor_1d(wctx, GGML_TYPE_F32, DIM);
        ggml_tensor * t2 = ggml_new_tensor_1d(wctx, GGML_TYPE_F32, DIM);
        ggml_tensor * t3 = ggml_new_tensor_1d(wctx, GGML_TYPE_F32, DIM);
        ggml_tensor * ids = ggml_new_tensor_1d(wctx, GGML_TYPE_I32, 4);
        ggml_tensor * wts = ggml_new_tensor_1d(wctx, GGML_TYPE_F32, 4);
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(wctx, be);
        ggml_context * gctx = ggml_init({ ggml_tensor_overhead() * 16 + ggml_graph_overhead(), nullptr, true });
        ggml_tensor * z = ggml_add(gctx, t1, t2);
        ggml_set_output(z);
        ggml_cgraph * gf = ggml_new_graph(gctx);
        ggml_build_forward_expand(gf, z);
        ggml_backend_buffer_t gbuf = ggml_backend_alloc_ctx_tensors(gctx, be);   // z lives here
        std::vector<float> a1(DIM), a2(DIM), r(DIM), r2(DIM);
        for (int i = 0; i < DIM; ++i) { a1[i] = frand(); a2[i] = frand(); }
        std::vector<int32_t> hid = { 1, 2, 3, 4 }; std::vector<float> hw = { .1f, .2f, .3f, .4f };
        ggml_backend_tensor_set(t1, a1.data(), 0, DIM * 4);
        ggml_backend_tensor_set(t2, a2.data(), 0, DIM * 4);
        ggml_backend_graph_compute(be, gf); ggml_backend_synchronize(be);
        ggml_backend_tensor_get(z, r.data(), 0, DIM * 4);
        bool ok = true; for (int i = 0; i < DIM; ++i) if (r[i] != a1[i] + a2[i]) ok = false;
        // set / get round trip of 8 KiB is bit exact
        ggml_backend_tensor_set(t3, a1.data(), 0, DIM * 4); ggml_backend_tensor_get(t3, r2.data(), 0, DIM * 4);
        for (int i = 0; i < DIM; ++i) if (r2[i] != a1[i]) ok = false;
        check("MB4", "1-node add and 8 KiB set/get on " + dn, ok);
        Stats s;
        s = bench(warm, iters, [&] { ggml_backend_tensor_set(t1, a1.data(), 0, DIM * 4); });
        report("MB4", "tensor_set_8KiB", dn, s, "RT_out half: Vulkan0 <- host");
        s = bench(warm, iters, [&] { ggml_backend_tensor_get(t3, r.data(), 0, DIM * 4); });
        report("MB4", "tensor_get_8KiB", dn, s, "RT_in half: host <- Vulkan0");
        s = bench(warm, iters, [&] { ggml_backend_tensor_set(ids, hid.data(), 0, 16); });
        report("MB4", "tensor_set_16B", dn, s);
        s = bench(warm, iters, [&] { ggml_backend_tensor_get(ids, hid.data(), 0, 16); });
        report("MB4", "tensor_get_16B", dn, s);
        s = bench(warm, iters, [&] { ggml_backend_tensor_get(t3, r.data(), 0, DIM * 4); ggml_backend_tensor_get(ids, hid.data(), 0, 16); ggml_backend_tensor_get(wts, hw.data(), 0, 16); });
        report("MB4", "tensor_get_8KiB+16B+16B", dn, s, "sched copies of hidden, ids, weights into the composite device");
        s = bench(warm, iters, [&] { ggml_backend_tensor_set(t1, a1.data(), 0, DIM * 4); ggml_backend_tensor_get(t3, r.data(), 0, DIM * 4); });
        report("MB4", "set_8KiB+get_8KiB", dn, s, "RT_out + RT_in with no compute between");
        s = bench(warm, iters, [&] { ggml_backend_graph_compute(be, gf); });
        report("MB4", "graph_compute_1node_nosync", dn, s, "submit only");
        ggml_backend_synchronize(be);
        s = bench(warm, iters, [&] { ggml_backend_graph_compute(be, gf); ggml_backend_synchronize(be); });
        report("MB4", "graph_compute_1node+synchronize", dn, s, "one op round trip without data transfer");
        s = bench(warm, iters, [&] { ggml_backend_synchronize(be); });
        report("MB4", "synchronize_idle", dn, s);
        s = bench(warm, iters, [&] {
            ggml_backend_tensor_set(t1, a1.data(), 0, DIM * 4);
            ggml_backend_graph_compute(be, gf); ggml_backend_synchronize(be);
            ggml_backend_tensor_get(z, r.data(), 0, DIM * 4);
        });
        report("MB4", "set8K+node+sync+get8K", dn, s, "one boundary + one op (the cheapest possible remote layer through sched)");
        ggml_backend_buffer_free(gbuf); ggml_free(gctx); ggml_backend_buffer_free(buf); ggml_free(wctx); ggml_backend_free(be);
    }
    return 0;
}
#endif   // MRB_NO_GGML

int main(int argc, char ** argv) {
    Args a;
    if (argc < 2) { fprintf(stderr, "usage: %s mb0|mb1c|mb1|mb2|mb3|mb3g|mb3x|mb4|mb5|mb5g [options]  (see the header of moe_rt_bench.cpp)\n", argv[0]); return 2; }
    a.cmd = argv[1];
    for (int i = 2; i < argc; ++i) {
        std::string k = argv[i];
        if (k.rfind("--", 0) == 0) { k = k.substr(2); a.kv[k] = (i + 1 < argc && std::string(argv[i + 1]).rfind("--", 0) != 0) ? argv[++i] : "1"; }
    }
    if (a.has("csv")) { g_csv = fopen(a.get("csv").c_str(), "a"); if (!g_csv) { perror("csv"); return 2; } }
    g_rng += a.geti("seed", 0);
    printf("# moe_rt_bench %s  args:", a.cmd.c_str());
    for (auto & kv : a.kv) printf(" --%s %s", kv.first.c_str(), kv.second.c_str());
    printf("\n");
    for (const char * e : { "RUSTICL_ENABLE", "VK_ICD_FILENAMES", "GGML_VK_VISIBLE_DEVICES", "GGML_LIB_DIR" }) if (getenv(e)) printf("# env %s=%s\n", e, getenv(e));
    int rc = 0;
    try {
        if      (a.cmd == "mb0")  rc = mb0(a);
        else if (a.cmd == "mb1c") rc = mb1c(a);
        else if (a.cmd == "mb1")  rc = mb1(a);
        else if (a.cmd == "mb2")  rc = mb2(a);
        else if (a.cmd == "mb3")  rc = mb3(a);
        else if (a.cmd == "mb5")  rc = mb5(a);
#ifndef MRB_NO_GGML
        else if (a.cmd == "mb3g") rc = mb3g_5g(a, 1);
        else if (a.cmd == "mb5g") rc = mb3g_5g(a, (int) a.geti("ntok", 128));
        else if (a.cmd == "mb4")  rc = mb4(a);
        else if (a.cmd == "mb3x") rc = mb3x(a);
#endif
        else { fprintf(stderr, "unknown sub command %s\n", a.cmd.c_str()); rc = 2; }
    } catch (std::exception & e) { fprintf(stderr, "fatal: %s\n", e.what()); rc = 1; }
    if (g_csv) fclose(g_csv);
    printf("# done %s: %s (%d failed check(s))\n", a.cmd.c_str(), (rc == 0 && g_fail == 0) ? "all OK" : "FAILED", g_fail);
    return (rc == 0 && g_fail == 0) ? 0 : 1;
}
