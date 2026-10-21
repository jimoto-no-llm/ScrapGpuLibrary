/* CUDA kernels of the ScrapGpu CUDA backend (src/backends/cuda), written in freestanding C for clang's NVPTX target.
 *
 * There is no nvcc on the machines this backend is for (NVIDIA driver 390, CUDA 9.1 generation Driver API, Fermi and Kepler cards). The
 * kernels are compiled to PTX offline by gen_ptx.sh and the PTX text is embedded in the library (ptx_sm20.inc, ptx_sm35.inc); the driver's
 * JIT (cuModuleLoadDataEx) turns it into machine code at run time. The PTX is checked into the repository, so building the library needs no
 * compiler for it.
 *
 * Two builds of this file: sm20 (.target sm_20, Fermi and everything newer) and sm35 (-DKEPLER, .target sm_35: the const __restrict__ arguments are
 * read through the read-only data cache, ld.global.nc).
 *
 * Weights are in the R32 layout (src/tensor/tensor.h, r32_geometry; the layout of the OpenCL 1.1 backend): the rows are grouped by 32, and inside
 * a group the 16-byte unit j of the block b of the row `lane` is the unit (b * U + j) * rs + lane (rs = rows of the group, 32 except for the last
 * group), so a warp whose lane is a row reads 512 contiguous bytes per 128-bit load and the 32 rows need no reduction over the lanes. The
 * activations are F32 and the same for every lane (broadcast loads). All arithmetic is F32.
 *   Q4_0: area 0 = nibbles (U = 1), area 1 = d (half, at b * rs + lane);                             area k starts at M * nb * (bytes of the
 *   Q6_K: area 0 = ql + qh (U = 12: ql pieces 0..7, qh pieces 8..11), area 1 = scales (U = 1, xor 0x80), areas before it per block)
 *         area 2 = d.
 *
 * The 4 weights of a 32-bit word are converted without the slow integer-to-float conversion: a byte b is the mantissa of the float 2^23 + b
 * (prmt builds it in one instruction), the offset is subtracted afterwards (exact). The high nibble is not shifted down: (w & 0xF0F0F0F0) holds
 * 16 * nibble, the factor 1/16 is applied to the sum.
 *
 *   mq4_cN   N = 1, 2, 4, 8: out[c * M + r] = sum_k W[r][k] * x[c * K + k] for c < N, W in Q4_0   (K % 32 == 0)
 *   mq6_cN   the same for Q6_K                                                                   (K % 256 == 0)
 *   mid4     MUL_MAT_ID (the experts of a MoE layer; every expert is a slice), Q4_0: one (slot, token) pair per blockIdx.y
 *
 * One warp computes one group of 32 rows (a block of 4 warps = 4 groups; grid-stride over the groups). x must be 16-byte aligned (float4 loads).
 */

typedef unsigned char u8;
typedef signed char s8;
typedef unsigned short u16;
typedef unsigned int u32;
typedef unsigned long long u64;
typedef float f4 __attribute__((ext_vector_type(4), aligned(16)));
typedef unsigned int u4 __attribute__((ext_vector_type(4), aligned(16)));

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

static inline float as_f(u32 v) { return __builtin_bit_cast(float, v); }

/* the float 2^23 + byte K of w (K is a compile-time constant: the selector 0x7540 | K picks byte K of w, three zero bytes and 0x4B) */
#define MAG(W, K)                                                                  \
    ({                                                                             \
        u32 r_;                                                                    \
        __asm__("prmt.b32 %0, %1, %2, %3;" : "=r"(r_) : "r"(W), "r"(0x4B000000u), "n"(0x7540 | (K))); \
        as_f(r_);                                                                  \
    })

static inline f4 f4_mag(u32 w) {
    f4 r = { MAG(w, 0), MAG(w, 1), MAG(w, 2), MAG(w, 3) };
    return r;
}

static inline f4 splat(float v) {
    f4 r = { v, v, v, v };
    return r;
}

static inline float dot4(f4 a, f4 b) {
    return __builtin_fmaf(a.x, b.x, __builtin_fmaf(a.y, b.y, __builtin_fmaf(a.z, b.z, a.w * b.w)));
}

/* ---- Q4_0: block = half d; u8 qs[16]; value j (< 16) = (qs[j] & 15) - 8, value j + 16 = (qs[j] >> 4) - 8; times d.
 * Word i (0..3) of the unit holds the bytes 4 i .. 4 i + 3: the low nibbles are the values 4 i .. 4 i + 3 (float4 8 b + i of x), the high nibbles
 * the values 16 + 4 i .. (float4 8 b + 4 + i). ---- */

#define MQ4(NC)                                                                                                         \
    KERNEL void mq4_c##NC(const u8 * __restrict__ w, const float * __restrict__ x, float * __restrict__ out, u32 K, u32 M) { \
        const u32 tid = tid_x();                                                                                        \
        const u32 warp = tid >> 5, lane = tid & 31;                                                                     \
        const u32 nb = K >> 5;                                                                                          \
        const u32 ngrp = (M + 31) >> 5;                                                                                 \
        for (u32 g = bid_x() * WPB + warp; g < ngrp; g += nbid_x() * WPB) {                                            \
            const u32 rs = M - g * 32 < 32 ? M - g * 32 : 32;                                                           \
            const u32 lc = lane < rs ? lane : rs - 1;                                                                   \
            const u4 * qp = (const u4 *) w + (u64) g * 32 * nb + lc;                                                    \
            const u16 * dp = (const u16 *) (w + (u64) M * nb * 16) + (u64) g * 32 * nb + lc;                            \
            float acc[NC];                                                                                              \
            for (int c = 0; c < NC; ++c) acc[c] = 0.0f;                                                                 \
            for (u32 b = 0; b < nb; ++b) {                                                                              \
                const u4 q = qp[(u64) b * rs];                                                                          \
                const float d = h2f(dp[(u64) b * rs]);                                                                  \
                float sl[NC], sh[NC];                                                                                   \
                for (int c = 0; c < NC; ++c) { sl[c] = 0.0f; sh[c] = 0.0f; }                                            \
                _Pragma("unroll")                                                                                       \
                for (int i = 0; i < 4; ++i) {                                                                           \
                    const u32 wd = i == 0 ? q.x : i == 1 ? q.y : i == 2 ? q.z : q.w;                                    \
                    const f4 vl = f4_mag(wd & 0x0F0F0F0Fu) - splat(8388616.0f);                                         \
                    const f4 vh = f4_mag(wd & 0xF0F0F0F0u) - splat(8388736.0f);                                         \
                    for (int c = 0; c < NC; ++c) {                                                                      \
                        const f4 * xc = (const f4 *) (x + (u64) c * K + (u64) b * 32);                                  \
                        sl[c] += dot4(vl, xc[i]);                                                                       \
                        sh[c] += dot4(vh, xc[4 + i]);                                                                   \
                    }                                                                                                   \
                }                                                                                                       \
                for (int c = 0; c < NC; ++c) acc[c] = __builtin_fmaf(d, __builtin_fmaf(sh[c], 0.0625f, sl[c]), acc[c]); \
            }                                                                                                           \
            if (lane < rs) {                                                                                            \
                for (int c = 0; c < NC; ++c) out[(u64) c * M + (u64) g * 32 + lane] = acc[c];                          \
            }                                                                                                           \
        }                                                                                                               \
    }

MQ4(1)
MQ4(2)
MQ4(4)
MQ4(8)

/* ---- Q6_K: block = u8 ql[128]; u8 qh[64]; s8 scales[16]; half d (256 values).
 * Value v = 128 h + 32 qd + l (h: half, qd: quadrant 0..3, l: 0..31) = d * scales[8 h + 2 qd + l / 16] * (q - 32), q = nibble | field << 4:
 * the nibble is the low (qd 0, 1) or high (qd 2, 3) half of ql[64 h + 32 (qd & 1) + l], the field the bits 2 qd of qh[32 h + l].
 * A lane handles the 16 consecutive l (lh = 0, 1) of the 4 quadrants of a half h: the pieces pa = ql[64 h + 16 lh] (unit 4 h + lh),
 * pb = ql[64 h + 32 + 16 lh] (unit 4 h + 2 + lh), pc = qh[32 h + 16 lh] (unit 8 + 2 h + lh), 4 words each (l = 16 lh + 4 k .. + 3); the
 * x of (h, qd, l = 16 lh + 4 k) is the float4 32 h + 8 qd + 4 lh + k of the block. ---- */

#define V6_0(PA, PB, PC) (((PA) & 0x0F0F0F0Fu) | (((PC) << 4) & 0x30303030u))
#define V6_1(PA, PB, PC) (((PB) & 0x0F0F0F0Fu) | (((PC) << 2) & 0x30303030u))
#define V6_2(PA, PB, PC) ((((PA) >> 4) & 0x0F0F0F0Fu) | ((PC) & 0x30303030u))
#define V6_3(PA, PB, PC) ((((PB) >> 4) & 0x0F0F0F0Fu) | (((PC) >> 2) & 0x30303030u))

/* the 16 values of one (h, lh, qd) against x: sum over k of dot(q - 32, x) */
#define Q6_QUAD(NC, H, LH, QD, VF)                                                                                      \
    {                                                                                                                   \
        const int xo = (H) * 32 + (QD) * 8 + 4 * (LH);                                                                  \
        const f4 v0 = f4_mag(VF(pa.x, pb.x, pc.x)) - splat(8388640.0f);                                                 \
        const f4 v1 = f4_mag(VF(pa.y, pb.y, pc.y)) - splat(8388640.0f);                                                 \
        const f4 v2 = f4_mag(VF(pa.z, pb.z, pc.z)) - splat(8388640.0f);                                                 \
        const f4 v3 = f4_mag(VF(pa.w, pb.w, pc.w)) - splat(8388640.0f);                                                 \
        const float sc = sw[(H) * 8 + (LH) + 2 * (QD)];                                                                 \
        for (int c = 0; c < NC; ++c) {                                                                                  \
            const f4 * xc = (const f4 *) (x + (u64) c * K + (u64) b * 256);                                             \
            const float s = (dot4(v0, xc[xo]) + dot4(v1, xc[xo + 1])) + (dot4(v2, xc[xo + 2]) + dot4(v3, xc[xo + 3])); \
            tt[c] = __builtin_fmaf(sc, s, tt[c]);                                                                       \
        }                                                                                                               \
    }

#define Q6_PIECE(NC, H, LH)                                                                                             \
    {                                                                                                                   \
        const u4 pa = qp[(u64) (b * 12 + 4 * (H) + (LH)) * rs];                                                         \
        const u4 pb = qp[(u64) (b * 12 + 4 * (H) + 2 + (LH)) * rs];                                                     \
        const u4 pc = qp[(u64) (b * 12 + 8 + 2 * (H) + (LH)) * rs];                                                     \
        Q6_QUAD(NC, H, LH, 0, V6_0)                                                                                     \
        Q6_QUAD(NC, H, LH, 1, V6_1)                                                                                     \
        Q6_QUAD(NC, H, LH, 2, V6_2)                                                                                     \
        Q6_QUAD(NC, H, LH, 3, V6_3)                                                                                     \
    }

#define MQ6(NC)                                                                                                         \
    KERNEL void mq6_c##NC(const u8 * __restrict__ w, const float * __restrict__ x, float * __restrict__ out, u32 K, u32 M) { \
        const u32 tid = tid_x();                                                                                        \
        const u32 warp = tid >> 5, lane = tid & 31;                                                                     \
        const u32 nb = K >> 8;                                                                                          \
        const u32 ngrp = (M + 31) >> 5;                                                                                 \
        for (u32 g = bid_x() * WPB + warp; g < ngrp; g += nbid_x() * WPB) {                                            \
            const u32 rs = M - g * 32 < 32 ? M - g * 32 : 32;                                                           \
            const u32 lc = lane < rs ? lane : rs - 1;                                                                   \
            const u4 * qp = (const u4 *) w + (u64) g * 32 * nb * 12 + lc;                                               \
            const u4 * sp = (const u4 *) (w + (u64) M * nb * 192) + (u64) g * 32 * nb + lc;                             \
            const u16 * dp = (const u16 *) (w + (u64) M * nb * 208) + (u64) g * 32 * nb + lc;                           \
            float acc[NC];                                                                                              \
            for (int c = 0; c < NC; ++c) acc[c] = 0.0f;                                                                 \
            for (u32 b = 0; b < nb; ++b) {                                                                              \
                float sw[16]; /* the 16 int8 scales (stored xor 0x80: the byte is scale + 128) */                      \
                {                                                                                                       \
                    const u4 s4 = sp[(u64) b * rs];                                                                     \
                    for (int i = 0; i < 4; ++i) {                                                                       \
                        const u32 wd = i == 0 ? s4.x : i == 1 ? s4.y : i == 2 ? s4.z : s4.w;                            \
                        const f4 v = f4_mag(wd) - splat(8388736.0f);                                                    \
                        sw[4 * i] = v.x; sw[4 * i + 1] = v.y; sw[4 * i + 2] = v.z; sw[4 * i + 3] = v.w;                 \
                    }                                                                                                   \
                }                                                                                                       \
                float tt[NC];                                                                                           \
                for (int c = 0; c < NC; ++c) tt[c] = 0.0f;                                                              \
                Q6_PIECE(NC, 0, 0)                                                                                      \
                Q6_PIECE(NC, 0, 1)                                                                                      \
                Q6_PIECE(NC, 1, 0)                                                                                      \
                Q6_PIECE(NC, 1, 1)                                                                                      \
                const float d = h2f(dp[(u64) b * rs]);                                                                  \
                for (int c = 0; c < NC; ++c) acc[c] = __builtin_fmaf(d, tt[c], acc[c]);                                 \
            }                                                                                                           \
            if (lane < rs) {                                                                                            \
                for (int c = 0; c < NC; ++c) out[(u64) c * M + (u64) g * 32 + lane] = acc[c];                          \
            }                                                                                                           \
        }                                                                                                               \
    }

MQ6(1)
MQ6(2)
MQ6(4)
MQ6(8)

/* ---- MUL_MAT_ID, Q4_0: dst[r, s, t] = sum_k W[ids[s, t]][r][k] * x[k, s or 0, t]  (ggml's ne = [rows, n_ids, n_tokens]); the expert e is the slice
 * at w + e * slice_bytes (M rows in the R32 layout). blockIdx.y = t * n_ids + s (a Fermi grid has no third dimension). A negative or too large
 * expert id writes zeros.
 *   ids          int32 [n_ids, n_tokens], the row of token t starts at ids + t * ids_tok_stride
 *   x            the column of (s, t) starts at x + t * x_tok_stride + s * x_slot_stride (x_slot_stride is 0 when one column serves every slot) ---- */
KERNEL void mid4(const u8 * __restrict__ w, const int * __restrict__ ids, const float * __restrict__ x, float * __restrict__ out,
                 u32 K, u32 M, u64 slice_bytes, u32 n_ids, u32 n_expert, u32 ids_tok_stride, u32 x_slot_stride, u32 x_tok_stride) {
    const u32 st = bid_y();
    const u32 t = st / n_ids, sl = st - t * n_ids;
    const int id = ids[(u64) t * ids_tok_stride + sl];
    const u32 tid = tid_x();
    const u32 warp = tid >> 5, lane = tid & 31;
    const u32 nb = K >> 5;
    const u32 ngrp = (M + 31) >> 5;
    const int valid = id >= 0 && (u32) id < n_expert;
    const u8 * wb = w + (u64) (valid ? (u32) id : 0u) * slice_bytes;
    const float * xv = x + (u64) t * x_tok_stride + (u64) sl * x_slot_stride;
    float * ov = out + (u64) st * M;
    for (u32 g = bid_x() * WPB + warp; g < ngrp; g += nbid_x() * WPB) {
        const u32 rs = M - g * 32 < 32 ? M - g * 32 : 32;
        const u32 lc = lane < rs ? lane : rs - 1;
        const u4 * qp = (const u4 *) wb + (u64) g * 32 * nb + lc;
        const u16 * dp = (const u16 *) (wb + (u64) M * nb * 16) + (u64) g * 32 * nb + lc;
        float acc = 0.0f;
        if (valid) {
            for (u32 b = 0; b < nb; ++b) {
                const u4 q = qp[(u64) b * rs];
                const float d = h2f(dp[(u64) b * rs]);
                const f4 * xc = (const f4 *) (xv + (u64) b * 32);
                float sl_ = 0.0f, sh_ = 0.0f;
                _Pragma("unroll")
                for (int i = 0; i < 4; ++i) {
                    const u32 wd = i == 0 ? q.x : i == 1 ? q.y : i == 2 ? q.z : q.w;
                    sl_ += dot4(f4_mag(wd & 0x0F0F0F0Fu) - splat(8388616.0f), xc[i]);
                    sh_ += dot4(f4_mag(wd & 0xF0F0F0F0u) - splat(8388736.0f), xc[4 + i]);
                }
                acc = __builtin_fmaf(d, __builtin_fmaf(sh_, 0.0625f, sl_), acc);
            }
        }
        if (lane < rs) {
            ov[(u64) g * 32 + lane] = acc;
        }
    }
}
