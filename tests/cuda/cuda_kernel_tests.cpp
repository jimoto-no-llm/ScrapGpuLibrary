// The kernels of the CUDA backend (src/backends/cuda) against a CPU reference: Q4_0 / Q6_K x F32 mat-vec and mat-mat, and MUL_MAT_ID (Q4_0).
// The weights are random floats quantized with scrapgpu_ref_quantize, repacked to the R32 layout on upload (r32_upload) and dequantized again for
// the reference (double precision dot products), so the only difference between the GPU and the reference is the F32 accumulation. Passes when
// the NMSE is below 1e-9.
// Exit code 77 (ctest SKIP_RETURN_CODE) when there is no CUDA device.
//
// A Fermi and a Kepler card cannot share a process (the second context fails), so every device whose init() fails after another one
// succeeded is reported as SKIP; run with SCRAPGPU_CUDA_DEVICES="GT 430" for the other generation.

#include "cuda/device.h"
#include "cuda/r32_transfer.h"
#include "scrapgpu/reference.h"
#include "scrapgpu/tensor.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

using namespace scrapgpu_cuda;

namespace {

std::mt19937 g_rng(4242);

float frand() { return std::uniform_real_distribution<float>(-1.0f, 1.0f)(g_rng); }

int g_pass = 0;
int g_fail = 0;

scrapgpu_dtype dtype_of(WType t) { return t == WType::Q4_0 ? SCRAPGPU_DTYPE_Q4_0 : SCRAPGPU_DTYPE_Q6_K; }

double nmse(const std::vector<float> & got, const std::vector<double> & ref) {
    double num = 0.0, den = 0.0;
    for (size_t i = 0; i < got.size(); ++i) {
        num += (got[i] - ref[i]) * (got[i] - ref[i]);
        den += ref[i] * ref[i];
    }
    return den > 0.0 ? num / den : num;
}

struct Quant {
    std::vector<uint8_t> bytes; // rows * row_bytes
    std::vector<float>   deq;   // rows * K
};

Quant make_weights(WType t, int64_t rows, int64_t K, float scale = 1.0f) {
    Quant q;
    std::vector<float> f((size_t) (rows * K));
    for (auto & v : f) v = frand() * scale;
    q.bytes.resize((size_t) (rows * (int64_t) wtype_row_bytes(t, K)));
    q.deq.resize(f.size());
    if (scrapgpu_ref_quantize(dtype_of(t), f.data(), q.bytes.data(), rows * K) != SCRAPGPU_STATUS_OK ||
        scrapgpu_ref_dequantize(dtype_of(t), q.bytes.data(), q.deq.data(), rows * K) != SCRAPGPU_STATUS_OK) {
        std::fprintf(stderr, "reference quantize failed\n");
        std::exit(1);
    }
    return q;
}

struct Dev {
    Device * d;
    CUstream s;
    template <typename T> uint64_t up(const std::vector<T> & v) {
        uint64_t p = 0;
        if (!d->alloc(v.size() * sizeof(T), p) || !d->h2d(p, v.data(), v.size() * sizeof(T), s)) {
            std::fprintf(stderr, "upload failed: %s\n", d->last_error().c_str());
            std::exit(1);
        }
        return p;
    }
    // a quantized weight of `rows` rows (natural bytes) in the R32 layout on the device
    uint64_t up_w(WType t, int64_t K, int64_t rows, const std::vector<uint8_t> & bytes) {
        uint64_t p = 0;
        if (!d->alloc(bytes.size(), p) || !r32_upload(*d, t, K, rows, bytes.data(), p, s)) {
            std::fprintf(stderr, "weight upload failed: %s\n", d->last_error().c_str());
            std::exit(1);
        }
        return p;
    }
    uint64_t scratch(size_t bytes) {
        uint64_t p = 0;
        if (!d->alloc(bytes, p)) {
            std::fprintf(stderr, "alloc failed: %s\n", d->last_error().c_str());
            std::exit(1);
        }
        return p;
    }
    std::vector<float> down(uint64_t p, size_t n) {
        std::vector<float> v(n);
        if (!d->d2h(v.data(), p, n * 4, s) || !d->stream_sync(s)) {
            std::fprintf(stderr, "download failed: %s\n", d->last_error().c_str());
            std::exit(1);
        }
        return v;
    }
};

void report(const char * name, double e, double tol) {
    const bool ok = e <= tol;
    std::printf("%s  %-44s NMSE %.3e\n", ok ? "PASS" : "FAIL", name, e);
    (ok ? g_pass : g_fail)++;
}

void test_mul_mat(Dev & dv, WType t, int64_t K, int64_t rows, int64_t N) {
    const Quant w = make_weights(t, rows, K);
    std::vector<float> x((size_t) (K * N));
    for (auto & v : x) v = frand();
    std::vector<double> ref((size_t) (rows * N));
    for (int64_t c = 0; c < N; ++c)
        for (int64_t r = 0; r < rows; ++r) {
            double s = 0.0;
            for (int64_t k = 0; k < K; ++k) s += (double) w.deq[(size_t) (r * K + k)] * (double) x[(size_t) (c * K + k)];
            ref[(size_t) (c * rows + r)] = s;
        }
    const uint64_t dw = dv.up_w(t, K, rows, w.bytes), dx = dv.up(x), dout = dv.scratch((size_t) (rows * N) * 4);
    dv.d->memset8(dout, 0xFF, (size_t) (rows * N) * 4, dv.s); // NaNs: an unwritten element fails
    if (!dv.d->mul_mat(dv.s, t, dw, (uint32_t) rows, (uint32_t) K, dx, (uint32_t) N, dout)) {
        std::printf("FAIL  mul_mat: %s\n", dv.d->last_error().c_str());
        ++g_fail;
        return;
    }
    const std::vector<float> got = dv.down(dout, (size_t) (rows * N));
    char name[96];
    std::snprintf(name, sizeof(name), "MUL_MAT %s K=%lld rows=%lld N=%lld", t == WType::Q4_0 ? "q4_0" : "q6_K", (long long) K, (long long) rows, (long long) N);
    const double e = nmse(got, ref);
    report(name, std::isnan(e) ? 1.0 : e, 1e-9);
    dv.d->free(dw);
    dv.d->free(dx);
    dv.d->free(dout);
}

void test_mul_mat_id(Dev & dv, int64_t K, int64_t rows, int64_t E, int64_t n_ids, int64_t n_tok, bool x_per_slot, bool bad_ids) {
    const Quant w = make_weights(WType::Q4_0, rows * E, K);
    std::vector<float> x((size_t) (K * (x_per_slot ? n_ids : 1) * n_tok));
    for (auto & v : x) v = frand();
    std::vector<int32_t> ids((size_t) (n_ids * n_tok));
    for (auto & i : ids) i = std::uniform_int_distribution<int>(0, (int) E - 1)(g_rng);
    if (bad_ids) {
        ids[0] = -1;
        ids[ids.size() - 1] = (int32_t) E;
    }
    std::vector<double> ref((size_t) (rows * n_ids * n_tok), 0.0);
    for (int64_t t = 0; t < n_tok; ++t)
        for (int64_t s = 0; s < n_ids; ++s) {
            const int32_t id = ids[(size_t) (t * n_ids + s)];
            if (id < 0 || id >= E) continue;
            const float * xc = x.data() + (size_t) ((t * (x_per_slot ? n_ids : 1) + (x_per_slot ? s : 0)) * K);
            for (int64_t r = 0; r < rows; ++r) {
                double a = 0.0;
                for (int64_t k = 0; k < K; ++k) a += (double) w.deq[(size_t) ((id * rows + r) * K + k)] * (double) xc[k];
                ref[(size_t) ((t * n_ids + s) * rows + r)] = a;
            }
        }
    const uint64_t rb = wtype_row_bytes(WType::Q4_0, K);
    const uint64_t slice = rb * (uint64_t) rows; // a multiple of 16 for the sizes tested
    uint64_t dw = 0;
    if (!dv.d->alloc((size_t) (slice * (uint64_t) E), dw)) {
        std::fprintf(stderr, "alloc failed\n");
        std::exit(1);
    }
    for (int64_t e = 0; e < E; ++e) {
        if (!r32_upload(*dv.d, WType::Q4_0, K, rows, w.bytes.data() + (size_t) (slice * (uint64_t) e), dw + slice * (uint64_t) e, dv.s)) {
            std::fprintf(stderr, "expert upload failed: %s\n", dv.d->last_error().c_str());
            std::exit(1);
        }
    }
    const uint64_t dx = dv.up(x), did = dv.up(ids), dout = dv.scratch(ref.size() * 4);
    dv.d->memset8(dout, 0xFF, ref.size() * 4, dv.s);
    if (!dv.d->mul_mat_id_q4_0(dv.s, dw, slice, (uint32_t) E, (uint32_t) rows, (uint32_t) K, did, (uint32_t) n_ids, (uint32_t) n_ids,
                               (uint32_t) n_tok, dx, x_per_slot ? (uint32_t) K : 0u, (uint32_t) (K * (x_per_slot ? n_ids : 1)), dout)) {
        std::printf("FAIL  mul_mat_id: %s\n", dv.d->last_error().c_str());
        ++g_fail;
        return;
    }
    const std::vector<float> got = dv.down(dout, ref.size());
    char name[96];
    std::snprintf(name, sizeof(name), "MUL_MAT_ID q4_0 K=%lld rows=%lld E=%lld ids=%lld tok=%lld%s%s", (long long) K, (long long) rows, (long long) E,
                  (long long) n_ids, (long long) n_tok, x_per_slot ? " x/slot" : " x/tok", bad_ids ? " bad-ids" : "");
    const double e = nmse(got, ref);
    report(name, std::isnan(e) ? 1.0 : e, 1e-9);
    dv.d->free(dw);
    dv.d->free(dx);
    dv.d->free(did);
    dv.d->free(dout);
}

} // namespace

int main() {
    const int n = device_count();
    if (n == 0) {
        std::printf("SKIP: no CUDA device (libcuda missing or no NVIDIA GPU)\n");
        return 77;
    }
    int tested = 0;
    for (int i = 0; i < n; ++i) {
        Device * d = Device::get(i);
        std::printf("device %d: %s sm_%d%d, %.0f MiB, %d SM, %.1f GB/s peak\n", i, d->info().name.c_str(), d->info().cc_major, d->info().cc_minor,
                    d->info().total_memory / 1048576.0, d->info().sm_count, d->info().peak_gbs());
        if (!d->init()) {
            std::printf("SKIP  device %d: %s\n", i, d->last_error().c_str());
            continue;
        }
        ++tested;
        Dev dv{d, d->stream_default()};
        for (WType t : {WType::Q4_0, WType::Q6_K}) {
            const int64_t Ks[] = { (int64_t) wtype_block_size(t), (int64_t) wtype_block_size(t) * 3, 2048 };
            for (int64_t K : Ks)
                for (int64_t rows : {1, 5, 130})
                    for (int64_t N : {1, 2, 3, 4, 5, 8, 9, 17})
                        test_mul_mat(dv, t, K, rows, N);
        }
        test_mul_mat(dv, WType::Q4_0, 2048, 6000, 1);
        test_mul_mat(dv, WType::Q6_K, 2048, 6000, 1);
        test_mul_mat(dv, WType::Q6_K, 2048, 6000, 8);
        for (bool per_slot : {true, false})
            for (int64_t K : {32, 96, 2048})
                for (int64_t tok : {1, 3}) test_mul_mat_id(dv, K, 40, 6, 4, tok, per_slot, false);
        test_mul_mat_id(dv, 2048, 64, 8, 4, 2, true, true);
        test_mul_mat_id(dv, 2048, 1792, 32, 4, 1, false, false);
    }
    if (tested == 0) {
        std::printf("SKIP: no CUDA device could be initialized\n");
        return 77;
    }
    std::printf("%d passed, %d failed\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
