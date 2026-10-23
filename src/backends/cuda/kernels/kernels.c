/* CUDA kernels of the ScrapGpu CUDA backend (src/backends/cuda), written in freestanding C for clang's NVPTX target.
 *
 * There is no nvcc on the machines this backend is for (NVIDIA driver 390, CUDA 9.1 generation Driver API, Fermi and Kepler cards). The
 * kernels are compiled to PTX offline by gen_ptx.sh and the PTX text is embedded in the library (ptx_sm20.inc, ptx_sm35.inc); the driver's
 * JIT (cuModuleLoadDataEx) turns it into machine code at run time. The PTX is checked into the repository, so building the library needs no
 * compiler for it.
 *
 * Two builds of this file: sm20 (.target sm_20, Fermi and everything newer) and sm35 (-DKEPLER, .target sm_35: the const RESTRICT arguments are
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
 * The 4 weights of a 32-bit word are converted without the slow integer-to-float conversion: a byte b < 128 placed in byte 2 of the float 0x43000000
 * is exactly 128 + b (prmt builds it in one instruction). The offset is not subtracted per weight: with m = 128 + q,
 *     sum_i (q_i - z) x_i = sum_i m_i x_i - (128 + z) * sum_i x_i,
 * and the sums of x over the groups of 16 values (the same for every row) are computed once per block into shared memory. The error of the
 * single-precision products with the small offset 128 is about 1e-5 of the group sum of x (NMSE ~ 1e-12, checked by the tests).
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

/* const __restrict__ kernel arguments make the sm_35 backend use ld.global.nc (the read-only data cache); on the GT 730 plain loads are faster
 * (docs/benchmarks/moe-ep-20261022/cuda), so it is off unless -DUSE_NC */
#ifdef USE_NC
#define RESTRICT __restrict__
#else
#define RESTRICT
#endif

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

/* the float 128 + byte K of w for a byte < 128 (K is a compile-time constant: the selector 0x7044 | K << 8 builds [0, 0, byte K, 0x43]) */
#define MAG(W, K)                                                                  \
    ({                                                                             \
        u32 r_;                                                                    \
        __asm__("prmt.b32 %0, %1, %2, %3;" : "=r"(r_) : "r"(W), "r"(0x43000000u), "n"(0x7044 | ((K) << 8))); \
        as_f(r_);                                                                  \
    })

static inline f4 f4_mag(u32 w) {
    f4 r = { MAG(w, 0), MAG(w, 1), MAG(w, 2), MAG(w, 3) };
    return r;
}

/* sums of x over the groups of 16 values: g_xs[c * (K / 16) + group] for the NC columns of x (K / 16 * NC <= XS_MAX); every warp of the block
 * must see them: call at the start of the kernel, before any divergence */
#define XS_MAX 2048
static __attribute__((address_space(3))) float g_xs[XS_MAX];

static inline void xsum_prologue(const float * x, u32 K, int nc, u32 tid) {
    const u32 ng = K >> 4;
    _Pragma("unroll") for (int c = 0; c < nc; ++c) {
        for (u32 gi = tid; gi < ng; gi += WPB * 32) {
            const f4 * p = (const f4 *) (x + (u64) c * K + (u64) gi * 16);
            const f4 a = p[0], b = p[1], d = p[2], e = p[3];
            g_xs[c * ng + gi] = (((a.x + a.y) + (a.z + a.w)) + ((b.x + b.y) + (b.z + b.w))) + (((d.x + d.y) + (d.z + d.w)) + ((e.x + e.y) + (e.z + e.w)));
        }
    }
    __asm__ volatile("bar.sync 0;" ::: "memory");
}

/* 4 bytes b (0..255) -> b - 128 as floats (the int8 scales stored xor 0x80): the float 2^23 + b, minus 2^23 + 128 */
static inline f4 f4_s8(u32 w) {
    u32 r0, r1, r2, r3;
    __asm__("prmt.b32 %0, %1, %2, %3;" : "=r"(r0) : "r"(w), "r"(0x4B000000u), "n"(0x7540));
    __asm__("prmt.b32 %0, %1, %2, %3;" : "=r"(r1) : "r"(w), "r"(0x4B000000u), "n"(0x7541));
    __asm__("prmt.b32 %0, %1, %2, %3;" : "=r"(r2) : "r"(w), "r"(0x4B000000u), "n"(0x7542));
    __asm__("prmt.b32 %0, %1, %2, %3;" : "=r"(r3) : "r"(w), "r"(0x4B000000u), "n"(0x7543));
    f4 r = { as_f(r0) - 8388736.0f, as_f(r1) - 8388736.0f, as_f(r2) - 8388736.0f, as_f(r3) - 8388736.0f };
    return r;
}

static inline f4 splat(float v) {
    f4 r = { v, v, v, v };
    return r;
}

static inline float dot4(f4 a, f4 b) {
    return __builtin_fmaf(a.x, b.x, __builtin_fmaf(a.y, b.y, __builtin_fmaf(a.z, b.z, a.w * b.w)));
}

/* s + a . b as a chain of fmas */
static inline float fma4(f4 a, f4 b, float s) {
    return __builtin_fmaf(a.w, b.w, __builtin_fmaf(a.z, b.z, __builtin_fmaf(a.y, b.y, __builtin_fmaf(a.x, b.x, s))));
}

/* ---- Q4_0: block = half d; u8 qs[16]; value j (< 16) = (qs[j] & 15) - 8, value j + 16 = (qs[j] >> 4) - 8; times d.
 * Word i (0..3) of the unit holds the bytes 4 i .. 4 i + 3: the low nibbles are the values 4 i .. 4 i + 3 (float4 8 b + i of x), the high nibbles
 * the values 16 + 4 i .. (float4 8 b + 4 + i). ---- */

/* R = groups of 32 rows per warp (a lane owns R rows): the loads of x are shared by the R rows */
#define MQ4(NC, R)                                                                                                      \
    KERNEL void mq4_c##NC##r##R(const u8 * RESTRICT w, const float * RESTRICT x, float * RESTRICT out, u32 K, u32 M) { \
        const u32 tid = tid_x();                                                                                        \
        xsum_prologue(x, K, NC, tid);                                                                                   \
        const u32 warp = tid >> 5, lane = tid & 31;                                                                     \
        const u32 nb = K >> 5;                                                                                          \
        const u32 ng = K >> 4;                                                                                          \
        const u32 ngrp = (M + 31) >> 5;                                                                                 \
        for (u32 g0 = (bid_x() * WPB + warp) * R; g0 < ngrp; g0 += nbid_x() * WPB * R) {                               \
            const u4 * qp[R];                                                                                           \
            const u16 * dp[R];                                                                                          \
            u32 rs[R];                                                                                                  \
            _Pragma("unroll") for (int r = 0; r < R; ++r) {                                                                               \
                const u32 g = g0 + r < ngrp ? g0 + r : ngrp - 1;                                                        \
                rs[r] = M - g * 32 < 32 ? M - g * 32 : 32;                                                              \
                const u32 lc = lane < rs[r] ? lane : rs[r] - 1;                                                         \
                qp[r] = (const u4 *) w + (u64) g * 32 * nb + lc;                                                        \
                dp[r] = (const u16 *) (w + (u64) M * nb * 16) + (u64) g * 32 * nb + lc;                                 \
            }                                                                                                           \
            float acc[R][NC];                                                                                           \
            _Pragma("unroll") for (int r = 0; r < R; ++r) _Pragma("unroll") for (int c = 0; c < NC; ++c) acc[r][c] = 0.0f;                                  \
            for (u32 b = 0; b < nb; ++b) {                                                                              \
                u4 q[R];                                                                                                \
                float d[R];                                                                                             \
                _Pragma("unroll") for (int r = 0; r < R; ++r) {                                                                           \
                    q[r] = qp[r][(u64) b * rs[r]];                                                                      \
                    d[r] = h2f(dp[r][(u64) b * rs[r]]);                                                                 \
                }                                                                                                       \
                float sm[R][NC];                                                                                        \
                _Pragma("unroll") for (int r = 0; r < R; ++r) _Pragma("unroll") for (int c = 0; c < NC; ++c) sm[r][c] = 0.0f;                               \
                _Pragma("unroll")                                                                                       \
                for (int i = 0; i < 4; ++i) {                                                                           \
                    f4 vl[R], vh[R];                                                                                    \
                    _Pragma("unroll") for (int r = 0; r < R; ++r) {                                                                       \
                        const u32 wd = i == 0 ? q[r].x : i == 1 ? q[r].y : i == 2 ? q[r].z : q[r].w;                    \
                        vl[r] = f4_mag(wd & 0x0F0F0F0Fu);                                                               \
                        vh[r] = f4_mag((wd >> 4) & 0x0F0F0F0Fu);                                                        \
                    }                                                                                                   \
                    _Pragma("unroll") for (int c = 0; c < NC; ++c) {                                                                      \
                        const f4 * xc = (const f4 *) (x + (u64) c * K + (u64) b * 32);                                  \
                        const f4 xl = xc[i], xh = xc[4 + i];                                                            \
                        _Pragma("unroll") for (int r = 0; r < R; ++r) {                                                                   \
                            sm[r][c] = fma4(vl[r], xl, sm[r][c]);                                                       \
                            sm[r][c] = fma4(vh[r], xh, sm[r][c]);                                                       \
                        }                                                                                               \
                    }                                                                                                   \
                }                                                                                                       \
                _Pragma("unroll") for (int c = 0; c < NC; ++c) {                                                                          \
                    const float xs = g_xs[c * ng + 2 * b] + g_xs[c * ng + 2 * b + 1];                                  \
                    _Pragma("unroll") for (int r = 0; r < R; ++r) acc[r][c] = __builtin_fmaf(d[r], __builtin_fmaf(-136.0f, xs, sm[r][c]), acc[r][c]); \
                }                                                                                                       \
            }                                                                                                           \
            _Pragma("unroll") for (int r = 0; r < R; ++r) {                                                                               \
                if (g0 + r < ngrp && lane < rs[r]) {                                                                    \
                    _Pragma("unroll") for (int c = 0; c < NC; ++c) out[(u64) c * M + (u64) (g0 + r) * 32 + lane] = acc[r][c];             \
                }                                                                                                       \
            }                                                                                                           \
        }                                                                                                               \
    }

MQ4(1, 1)
MQ4(1, 2)
MQ4(1, 4)
MQ4(2, 1)
MQ4(2, 2)
MQ4(4, 1)
MQ4(8, 1)

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

/* the 16 values of one (h, lh, qd) of the R rows against x: sum over k of m * x, and the x sum of the group, weighted by the scale */
#define Q6_QUAD(NC, R, H, LH, QD, VF)                                                                                   \
    {                                                                                                                   \
        const int xo = (H) * 32 + (QD) * 8 + 4 * (LH);                                                                  \
        const int gi = (H) * 8 + (LH) + 2 * (QD);                                                                       \
        f4 v0[R], v1[R], v2[R], v3[R];                                                                                  \
        _Pragma("unroll") for (int r = 0; r < R; ++r) {                                                                                   \
            v0[r] = f4_mag(VF(pa[r].x, pb[r].x, pc[r].x));                                                              \
            v1[r] = f4_mag(VF(pa[r].y, pb[r].y, pc[r].y));                                                              \
            v2[r] = f4_mag(VF(pa[r].z, pb[r].z, pc[r].z));                                                              \
            v3[r] = f4_mag(VF(pa[r].w, pb[r].w, pc[r].w));                                                              \
        }                                                                                                               \
        _Pragma("unroll") for (int c = 0; c < NC; ++c) {                                                                                  \
            const f4 * xc = (const f4 *) (x + (u64) c * K + (u64) b * 256);                                             \
            const f4 x0 = xc[xo], x1 = xc[xo + 1], x2 = xc[xo + 2], x3 = xc[xo + 3];                                    \
            const float xg = g_xs[c * ng + b * 16 + gi];                                                                \
            _Pragma("unroll") for (int r = 0; r < R; ++r) {                                                                               \
                float s = fma4(v0[r], x0, 0.0f);                                                                        \
                s = fma4(v1[r], x1, s);                                                                                 \
                s = fma4(v2[r], x2, s);                                                                                 \
                s = fma4(v3[r], x3, s);                                                                                 \
                tt[r][c] = __builtin_fmaf(sw[r][gi], s, tt[r][c]);                                                      \
                tx[r][c] = __builtin_fmaf(sw[r][gi], xg, tx[r][c]);                                                     \
            }                                                                                                           \
        }                                                                                                               \
    }

#define Q6_PIECE(NC, R, H, LH)                                                                                          \
    {                                                                                                                   \
        u4 pa[R], pb[R], pc[R];                                                                                         \
        _Pragma("unroll") for (int r = 0; r < R; ++r) {                                                                                   \
            pa[r] = qp[r][(u64) (b * 12 + 4 * (H) + (LH)) * rs[r]];                                                     \
            pb[r] = qp[r][(u64) (b * 12 + 4 * (H) + 2 + (LH)) * rs[r]];                                                 \
            pc[r] = qp[r][(u64) (b * 12 + 8 + 2 * (H) + (LH)) * rs[r]];                                                 \
        }                                                                                                               \
        Q6_QUAD(NC, R, H, LH, 0, V6_0)                                                                                  \
        Q6_QUAD(NC, R, H, LH, 1, V6_1)                                                                                  \
        Q6_QUAD(NC, R, H, LH, 2, V6_2)                                                                                  \
        Q6_QUAD(NC, R, H, LH, 3, V6_3)                                                                                  \
    }

#define MQ6(NC, R)                                                                                                      \
    KERNEL void mq6_c##NC##r##R(const u8 * RESTRICT w, const float * RESTRICT x, float * RESTRICT out, u32 K, u32 M) { \
        const u32 tid = tid_x();                                                                                        \
        xsum_prologue(x, K, NC, tid);                                                                                   \
        const u32 warp = tid >> 5, lane = tid & 31;                                                                     \
        const u32 nb = K >> 8;                                                                                          \
        const u32 ng = K >> 4;                                                                                          \
        const u32 ngrp = (M + 31) >> 5;                                                                                 \
        for (u32 g0 = (bid_x() * WPB + warp) * R; g0 < ngrp; g0 += nbid_x() * WPB * R) {                               \
            const u4 * qp[R];                                                                                           \
            const u4 * sp[R];                                                                                           \
            const u16 * dp[R];                                                                                          \
            u32 rs[R];                                                                                                  \
            _Pragma("unroll") for (int r = 0; r < R; ++r) {                                                                               \
                const u32 g = g0 + r < ngrp ? g0 + r : ngrp - 1;                                                        \
                rs[r] = M - g * 32 < 32 ? M - g * 32 : 32;                                                              \
                const u32 lc = lane < rs[r] ? lane : rs[r] - 1;                                                         \
                qp[r] = (const u4 *) w + (u64) g * 32 * nb * 12 + lc;                                                   \
                sp[r] = (const u4 *) (w + (u64) M * nb * 192) + (u64) g * 32 * nb + lc;                                 \
                dp[r] = (const u16 *) (w + (u64) M * nb * 208) + (u64) g * 32 * nb + lc;                                \
            }                                                                                                           \
            float acc[R][NC];                                                                                           \
            _Pragma("unroll") for (int r = 0; r < R; ++r) _Pragma("unroll") for (int c = 0; c < NC; ++c) acc[r][c] = 0.0f;                                  \
            for (u32 b = 0; b < nb; ++b) {                                                                              \
                float sw[R][16]; /* the 16 int8 scales (stored xor 0x80) */                                             \
                _Pragma("unroll") for (int r = 0; r < R; ++r) {                                                                           \
                    const u4 s4 = sp[r][(u64) b * rs[r]];                                                               \
                    _Pragma("unroll") for (int i = 0; i < 4; ++i) {                                                                       \
                        const u32 wd = i == 0 ? s4.x : i == 1 ? s4.y : i == 2 ? s4.z : s4.w;                            \
                        const f4 v = f4_s8(wd);                                                                         \
                        sw[r][4 * i] = v.x; sw[r][4 * i + 1] = v.y; sw[r][4 * i + 2] = v.z; sw[r][4 * i + 3] = v.w;     \
                    }                                                                                                   \
                }                                                                                                       \
                float tt[R][NC], tx[R][NC];                                                                             \
                _Pragma("unroll") for (int r = 0; r < R; ++r) _Pragma("unroll") for (int c = 0; c < NC; ++c) { tt[r][c] = 0.0f; tx[r][c] = 0.0f; }          \
                Q6_PIECE(NC, R, 0, 0)                                                                                   \
                Q6_PIECE(NC, R, 0, 1)                                                                                   \
                Q6_PIECE(NC, R, 1, 0)                                                                                   \
                Q6_PIECE(NC, R, 1, 1)                                                                                   \
                _Pragma("unroll") for (int r = 0; r < R; ++r) {                                                                           \
                    const float d = h2f(dp[r][(u64) b * rs[r]]);                                                        \
                    _Pragma("unroll") for (int c = 0; c < NC; ++c) acc[r][c] = __builtin_fmaf(d, __builtin_fmaf(-160.0f, tx[r][c], tt[r][c]), acc[r][c]); \
                }                                                                                                       \
            }                                                                                                           \
            _Pragma("unroll") for (int r = 0; r < R; ++r) {                                                                               \
                if (g0 + r < ngrp && lane < rs[r]) {                                                                    \
                    _Pragma("unroll") for (int c = 0; c < NC; ++c) out[(u64) c * M + (u64) (g0 + r) * 32 + lane] = acc[r][c];             \
                }                                                                                                       \
            }                                                                                                           \
        }                                                                                                               \
    }

MQ6(1, 1)
MQ6(2, 1)
MQ6(4, 1)
MQ6(8, 1)

/* ---- MUL_MAT_ID, Q4_0: dst[r, s, t] = sum_k W[ids[s, t]][r][k] * x[k, s or 0, t]  (ggml's ne = [rows, n_ids, n_tokens]); the expert e is the slice
 * at w + e * slice_bytes (M rows in the R32 layout). blockIdx.y = t * n_ids + s (a Fermi grid has no third dimension). A negative or too large
 * expert id writes zeros.
 *   ids          int32 [n_ids, n_tokens], the row of token t starts at ids + t * ids_tok_stride
 *   x            the column of (s, t) starts at x + t * x_tok_stride + s * x_slot_stride (x_slot_stride is 0 when one column serves every slot) ---- */
KERNEL void mid4(const u8 * RESTRICT w, const int * RESTRICT ids, const float * RESTRICT x, float * RESTRICT out,
                 u32 K, u32 M, u64 slice_bytes, u32 n_ids, u32 n_expert, u32 ids_tok_stride, u32 x_slot_stride, u32 x_tok_stride) {
    const u32 st = bid_y();
    const u32 t = st / n_ids, sl = st - t * n_ids;
    const float * xv = x + (u64) t * x_tok_stride + (u64) sl * x_slot_stride;
    const u32 tid = tid_x();
    xsum_prologue(xv, K, 1, tid);
    const int id = ids[(u64) t * ids_tok_stride + sl];
    const u32 warp = tid >> 5, lane = tid & 31;
    const u32 nb = K >> 5;
    const u32 ngrp = (M + 31) >> 5;
    const int valid = id >= 0 && (u32) id < n_expert;
    const u8 * wb = w + (u64) (valid ? (u32) id : 0u) * slice_bytes;
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
                float sm = 0.0f;
                _Pragma("unroll")
                for (int i = 0; i < 4; ++i) {
                    const u32 wd = i == 0 ? q.x : i == 1 ? q.y : i == 2 ? q.z : q.w;
                    sm = fma4(f4_mag(wd & 0x0F0F0F0Fu), xc[i], sm);
                    sm = fma4(f4_mag((wd >> 4) & 0x0F0F0F0Fu), xc[4 + i], sm);
                }
                acc = __builtin_fmaf(d, __builtin_fmaf(-136.0f, g_xs[2 * b] + g_xs[2 * b + 1], sm), acc);
            }
        }
        if (lane < rs) {
            ov[(u64) g * 32 + lane] = acc;
        }
    }
}
