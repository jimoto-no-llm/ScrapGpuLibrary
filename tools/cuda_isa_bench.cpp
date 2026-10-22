// Instruction throughput of an NVIDIA GPU through the CUDA Driver API (libcuda loaded with dlopen like the backend does), for tuning the PTX of
// src/backends/cuda/kernels: warp instructions per clock per SM of prmt / and / shr / fma / cvt / ...
//   cuda_isa_bench isa.ptx [device ordinal]      (isa.ptx from kernels/isa_bench.c, see gen_ptx_to.sh)
// Built with the backend (SCRAPGPU_BUILD_CUDA + SCRAPGPU_BUILD_BENCHMARKS). One GPU per process.

#include "cuda/driver.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using namespace scrapgpu_cuda;

int main(int argc, char ** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s isa.ptx [ordinal]\n", argv[0]);
        return 2;
    }
    const Driver * d = driver();
    if (d == nullptr) {
        std::puts("SKIP: no CUDA driver");
        return 77;
    }
    FILE * f = std::fopen(argv[1], "rb");
    if (f == nullptr) {
        std::fprintf(stderr, "cannot read %s\n", argv[1]);
        return 2;
    }
    std::string ptx;
    char buf[65536];
    size_t n;
    while ((n = std::fread(buf, 1, sizeof(buf), f)) > 0) ptx.append(buf, n);
    std::fclose(f);
    int ord = argc > 2 ? std::atoi(argv[2]) : 0;
    CUdevice dev = 0;
    CUcontext ctx = nullptr;
    CUmodule mod = nullptr;
    if (d->cuDeviceGet(&dev, ord) != CUDA_SUCCESS || d->cuCtxCreate(&ctx, 0, dev) != CUDA_SUCCESS) {
        std::fprintf(stderr, "no device %d\n", ord);
        return 1;
    }
    char name[256] = {};
    d->cuDeviceGetName(name, 255, dev);
    int sms = 0, clk = 0;
    d->cuDeviceGetAttribute(&sms, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, dev);
    d->cuDeviceGetAttribute(&clk, CU_DEVICE_ATTRIBUTE_CLOCK_RATE, dev);
    char log[4096] = {};
    int opts[2] = { CU_JIT_ERROR_LOG_BUFFER, CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES };
    void * vals[2] = { log, (void *) (uintptr_t) sizeof(log) };
    if (d->cuModuleLoadDataEx(&mod, ptx.c_str(), 2, opts, vals) != CUDA_SUCCESS) {
        std::fprintf(stderr, "module load failed: %s\n", log);
        return 1;
    }
    std::printf("%s: %d SM, %.0f MHz (reported)\n", name, sms, clk / 1000.0);
    const unsigned blocks = (unsigned) sms * 8, threads = 1024;
    CUdeviceptr out = 0;
    d->cuMemAlloc(&out, (size_t) blocks * threads * 4);
    const char * names[] = { "k_prmt", "k_prmt_b", "k_and", "k_shr", "k_addi", "k_madi", "k_bfe", "k_selp", "k_fma", "k_cvt" };
    for (const char * kn : names) {
        CUfunction fn = nullptr;
        if (d->cuModuleGetFunction(&fn, mod, kn) != CUDA_SUCCESS) continue;
        for (unsigned iters : { 1000u, 20000u }) {
            unsigned it = iters;
            void * args[] = { &it, &out };
            d->cuLaunchKernel(fn, blocks, 1, 1, threads, 1, 1, 0, nullptr, args, nullptr);
            d->cuCtxSynchronize();
            if (iters == 1000u) continue;
            const auto t0 = std::chrono::steady_clock::now();
            d->cuLaunchKernel(fn, blocks, 1, 1, threads, 1, 1, 0, nullptr, args, nullptr);
            d->cuCtxSynchronize();
            const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            const double warp_instr = (double) it * 8 * (double) blocks * threads / 32.0; // (+ loop overhead not counted)
            const double per_sm_clk = warp_instr / sms / (sec * clk * 1e3);
            std::printf("  %-9s %8.3f ms   %.2f warp-instr / clk / SM   (%.1f lanes/clk/SM)\n", kn, sec * 1e3, per_sm_clk, per_sm_clk * 32.0);
        }
    }
    return 0;
}
