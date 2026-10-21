/* CUDA kernels of the ScrapGpu CUDA backend (src/backends/cuda), written in freestanding C for clang's NVPTX target.
 *
 * There is no nvcc on the machines this backend is for (NVIDIA driver 390, CUDA 9.1 generation Driver API, Fermi and Kepler cards). The
 * kernels are compiled to PTX offline by gen_ptx.sh and the PTX text is embedded in the library (ptx_sm20.inc, ptx_sm35.inc); the driver's
 * JIT (cuModuleLoadDataEx) turns it into machine code at run time. The PTX is checked into the repository, so building the library needs no
 * compiler for it.
 *
 * Two builds of this file:
 *   sm20  (.target sm_20): Fermi and everything newer. The reduction over a warp goes through shared memory.
 *   sm35  (-DKEPLER, .target sm_35): Kepler GK110 / GK208 and newer. The weights and activations are read through the read-only data cache
 *         (ld.global.nc, from the const __restrict__ kernel arguments) and the reduction over a warp uses shfl.down.
 *
 * Data layout: the weights are read in the natural GGUF block layout, so a weight tensor needs no repacking (blocks are 2-byte aligned
 * only: Q4_0 is 18 bytes, Q6_K 210 bytes). All arithmetic is F32 (the activations are not quantized).
 *
 *   mq4_cN   N = 1, 2, 4, 8: out[c * rows + r] = sum_k W[r][k] * x[c * K + k] for c < N, W in Q4_0   (K % 32 == 0)
 *   mq6_cN   the same for Q6_K                                                                      (K % 256 == 0)
 *   mid4     MUL_MAT_ID (the experts of a MoE layer), Q4_0: one (slot, token) pair per blockIdx.y
 *
 * One warp computes one row (a block of 4 warps = 4 rows; grid-stride over the rows). x must be 16-byte aligned (float4 loads).
 */

typedef unsigned char u8;
typedef signed char s8;
typedef unsigned short u16;
typedef unsigned int u32;
typedef unsigned long long u64;
typedef float f4 __attribute__((ext_vector_type(4), aligned(16)));

#define SM __attribute__((address_space(3)))
#define KERNEL __attribute__((nvptx_kernel))

#define WPB 4 /* warps per block (128 threads) */

static inline u32 tid_x(void) { return __nvvm_read_ptx_sreg_tid_x(); }
static inline u32 bid_x(void) { return __nvvm_read_ptx_sreg_ctaid_x(); }
static inline u32 bid_y(void) { return __nvvm_read_ptx_sreg_ctaid_y(); }
static inline u32 nbid_x(void) { return __nvvm_read_ptx_sreg_nctaid_x(); }

static inline float h2f(u32 h) {
    float f;
    u16 hh = (u16) h;
    __asm__("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"(hh));
    return f;
}

static inline u32 ld16(const u8 * p) { return *(const u16 *) p; }
static inline u32 ld32(const u8 * p) { return ld16(p) | (ld16(p + 2) << 16); }

#ifdef KEPLER
static inline float shfl_down(float v, u32 o) {
    float r;
    __asm__ volatile("shfl.down.b32 %0, %1, %2, 0x1f;" : "=f"(r) : "f"(v), "r"(o));
    return r;
}
static inline float wsum(float v, volatile SM float * scr, u32 lane) {
    (void) scr;
    (void) lane;
    v += shfl_down(v, 16);
    v += shfl_down(v, 8);
    v += shfl_down(v, 4);
    v += shfl_down(v, 2);
    v += shfl_down(v, 1);
    return v; /* lane 0 holds the sum */
}
#else
/* warp-synchronous reduction through shared memory (Fermi / Kepler execute a warp in lockstep; volatile keeps the order) */
static inline float wsum(float v, volatile SM float * scr, u32 lane) {
    scr[lane] = v;
    if (lane < 16) scr[lane] += scr[lane + 16];
    if (lane < 8)  scr[lane] += scr[lane + 8];
    if (lane < 4)  scr[lane] += scr[lane + 4];
    if (lane < 2)  scr[lane] += scr[lane + 2];
    if (lane < 1)  scr[lane] += scr[lane + 1];
    return scr[0];
}
#endif

static SM volatile float g_red[WPB * 32];

/* ---- Q4_0: block = half d; u8 qs[16]; value j (< 16) = (qs[j] & 15) - 8, value j + 16 = (qs[j] >> 4) - 8; times d ---- */

/* dot of one 32-value Q4_0 slice: 4 bytes of qs (values 4s..4s+3 and 16+4s..16+4s+3) against x[..] (xl at +4s, xh at +16+4s) */
static inline float dot4_q4(u32 qs, f4 xl, f4 xh) {
    const u32 lo = qs & 0x0F0F0F0Fu;
    const u32 hi = (qs >> 4) & 0x0F0F0F0Fu;
    float s = ((float) (int) (lo & 0xFF) - 8.0f) * xl.x;
    s += ((float) (int) ((lo >> 8) & 0xFF) - 8.0f) * xl.y;
    s += ((float) (int) ((lo >> 16) & 0xFF) - 8.0f) * xl.z;
    s += ((float) (int) (lo >> 24) - 8.0f) * xl.w;
    s += ((float) (int) (hi & 0xFF) - 8.0f) * xh.x;
    s += ((float) (int) ((hi >> 8) & 0xFF) - 8.0f) * xh.y;
    s += ((float) (int) ((hi >> 16) & 0xFF) - 8.0f) * xh.z;
    s += ((float) (int) (hi >> 24) - 8.0f) * xh.w;
    return s;
}

#define MQ4_BODY(NC)                                                                                   \
    const u32 tid = tid_x();                                                                           \
    const u32 warp = tid >> 5, lane = tid & 31;                                                        \
    const u32 nb = K >> 5;                                                                             \
    const u32 bi = lane >> 2, s = lane & 3;                                                            \
    volatile SM float * scr = g_red + warp * 32;                                                       \
    for (u32 row = bid_x() * WPB + warp; row < rows; row += nbid_x() * WPB) {                         \
        const u8 * rp = w + (u64) row * row_bytes;                                                     \
        float acc[NC];                                                                                 \
        for (int c = 0; c < NC; ++c) acc[c] = 0.0f;                                                    \
        for (u32 b0 = 0; b0 < nb; b0 += 8) {                                                           \
            const u32 b = b0 + bi;                                                                     \
            if (b < nb) {                                                                              \
                const u8 * bp = rp + (u64) b * 18;                                                     \
                const u32 qs = ld32(bp + 2 + 4 * s);                                                   \
                const float d = h2f(ld16(bp));                                                         \
                for (int c = 0; c < NC; ++c) {                                                         \
                    const float * xc = x + (u64) c * K + b * 32 + 4 * s;                               \
                    acc[c] += d * dot4_q4(qs, *(const f4 *) xc, *(const f4 *) (xc + 16));              \
                }                                                                                      \
            }                                                                                          \
        }                                                                                              \
        for (int c = 0; c < NC; ++c) {                                                                 \
            const float t = wsum(acc[c], scr, lane);                                                   \
            if (lane == 0) out[(u64) c * rows + row] = t;                                              \
        }                                                                                              \
    }

#define MQ4(NC)                                                                                        \
    KERNEL void mq4_c##NC(const u8 * __restrict__ w, const float * __restrict__ x, float * __restrict__ out, \
                          u32 K, u32 rows, u64 row_bytes) {                                            \
        MQ4_BODY(NC)                                                                                   \
    }

MQ4(1)
MQ4(2)
MQ4(4)
MQ4(8)

/* ---- Q6_K: block = u8 ql[128]; u8 qh[64]; s8 scales[16]; half d (210 bytes, 256 values).
 * Value v = 128 * h + 32 * g + l (h: half, g: 0..3, l: 0..31) = d * scales[v / 16] * (((ql[64 h + 32 (g & 1) + l] >> 4 * (g >> 1)) & 15) |
 *                                                                          (((qh[32 h + l] >> 2 g) & 3) << 4)) - 32).
 * Lane i of a warp owns ql bytes 4i..4i+3 of the block: the low nibbles are the values vlo = 128 h + p .. +3 and the high nibbles the values
 * vhi = vlo + 64 .. +3 (p = (4 i) & 63); their high bits are in qh bytes 32 h + (p & 31) .. +3 at bit 2 (p >> 5) and 2 (p >> 5) + 4. ---- */

static inline float dot4_q6(u32 q, f4 xv) {
    float s = ((float) (int) (q & 0xFF) - 32.0f) * xv.x;
    s += ((float) (int) ((q >> 8) & 0xFF) - 32.0f) * xv.y;
    s += ((float) (int) ((q >> 16) & 0xFF) - 32.0f) * xv.z;
    s += ((float) (int) (q >> 24) - 32.0f) * xv.w;
    return s;
}

#define MQ6_BODY(NC)                                                                                   \
    const u32 tid = tid_x();                                                                           \
    const u32 warp = tid >> 5, lane = tid & 31;                                                        \
    const u32 nb = K >> 8;                                                                             \
    const u32 j = lane * 4, h = j >> 6, p = j & 63, g0 = p >> 5, l = p & 31;                           \
    const u32 vlo = h * 128 + p, vhi = vlo + 64;                                                       \
    const u32 slo = vlo >> 4, shi = vhi >> 4;                                                          \
    const u32 sh_lo = 2 * g0, sh_hi = 2 * g0 + 4;                                                      \
    const u32 qho = 128 + h * 32 + l;                                                                  \
    volatile SM float * scr = g_red + warp * 32;                                                       \
    for (u32 row = bid_x() * WPB + warp; row < rows; row += nbid_x() * WPB) {                         \
        const u8 * rp = w + (u64) row * row_bytes;                                                     \
        float acc[NC];                                                                                 \
        for (int c = 0; c < NC; ++c) acc[c] = 0.0f;                                                    \
        _Pragma("unroll 2")                                                                            \
        for (u32 b = 0; b < nb; ++b) {                                                                 \
            const u8 * bp = rp + (u64) b * 210;                                                        \
            const u32 ql = ld32(bp + j);                                                               \
            const u32 qh = ld32(bp + qho);                                                             \
            const float d = h2f(ld16(bp + 208));                                                       \
            const float sc_lo = (float) (int) (s8) bp[192 + slo] * d;                                  \
            const float sc_hi = (float) (int) (s8) bp[192 + shi] * d;                                  \
            const u32 qlo = (ql & 0x0F0F0F0Fu) | (((qh >> sh_lo) & 0x03030303u) << 4);                 \
            const u32 qhi = ((ql >> 4) & 0x0F0F0F0Fu) | (((qh >> sh_hi) & 0x03030303u) << 4);          \
            for (int c = 0; c < NC; ++c) {                                                             \
                const float * xc = x + (u64) c * K + b * 256;                                          \
                acc[c] += sc_lo * dot4_q6(qlo, *(const f4 *) (xc + vlo)) +                             \
                          sc_hi * dot4_q6(qhi, *(const f4 *) (xc + vhi));                              \
            }                                                                                          \
        }                                                                                              \
        for (int c = 0; c < NC; ++c) {                                                                 \
            const float t = wsum(acc[c], scr, lane);                                                   \
            if (lane == 0) out[(u64) c * rows + row] = t;                                              \
        }                                                                                              \
    }

#define MQ6(NC)                                                                                        \
    KERNEL void mq6_c##NC(const u8 * __restrict__ w, const float * __restrict__ x, float * __restrict__ out, \
                          u32 K, u32 rows, u64 row_bytes) {                                            \
        MQ6_BODY(NC)                                                                                   \
    }

MQ6(1)
MQ6(2)
MQ6(4)
MQ6(8)

/* ---- MUL_MAT_ID, Q4_0: dst[r, s, t] = sum_k W[ids[s, t]][r][k] * x[k, s or 0, t]  (ggml's ne = [rows, n_ids, n_tokens]).
 * blockIdx.y = t * n_ids + s (a Fermi grid has no third dimension). A negative or too large expert id writes zeros.
 *   ids          int32 [n_ids, n_tokens], the row of token t starts at ids + t * ids_tok_stride
 *   x            the column of (s, t) starts at x + t * x_tok_stride + s * x_slot_stride (x_slot_stride is 0 when one column serves every slot)
 *   expert_bytes the bytes of one expert (rows * row_bytes) ---- */
KERNEL void mid4(const u8 * __restrict__ w, const int * __restrict__ ids, const float * __restrict__ x, float * __restrict__ out,
                 u32 K, u32 rows, u64 row_bytes, u64 expert_bytes, u32 n_ids, u32 n_expert, u32 ids_tok_stride,
                 u32 x_slot_stride, u32 x_tok_stride) {
    const u32 st = bid_y();
    const u32 t = st / n_ids, sl = st - t * n_ids;
    const int id = ids[(u64) t * ids_tok_stride + sl];
    const u32 tid = tid_x();
    const u32 warp = tid >> 5, lane = tid & 31;
    const u32 nb = K >> 5;
    const u32 bi = lane >> 2, s = lane & 3;
    volatile SM float * scr = g_red + warp * 32;
    const int valid = id >= 0 && (u32) id < n_expert;
    const u8 * wb = w + (u64) (valid ? (u32) id : 0u) * expert_bytes;
    const float * xv = x + (u64) t * x_tok_stride + (u64) sl * x_slot_stride;
    float * ov = out + (u64) st * rows;
    for (u32 row = bid_x() * WPB + warp; row < rows; row += nbid_x() * WPB) {
        const u8 * rp = wb + (u64) row * row_bytes;
        float acc = 0.0f;
        if (valid) {
            for (u32 b0 = 0; b0 < nb; b0 += 8) {
                const u32 b = b0 + bi;
                if (b < nb) {
                    const u8 * bp = rp + (u64) b * 18;
                    const u32 qs = ld32(bp + 2 + 4 * s);
                    const float d = h2f(ld16(bp));
                    const float * xc = xv + b * 32 + 4 * s;
                    acc += d * dot4_q4(qs, *(const f4 *) xc, *(const f4 *) (xc + 16));
                }
            }
        }
        const float tt = wsum(acc, scr, lane);
        if (lane == 0) ov[row] = tt;
    }
}
