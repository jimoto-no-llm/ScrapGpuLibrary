/* Instruction throughput microbenchmark kernels for tools/cuda_isa_bench.cpp (compiled with gen_ptx_to.sh, not part of the library):
 *   KSRC=.../isa_bench.c gen_ptx_to.sh sm_35 isa35.ptx
 * Each kernel runs N iterations of 8 independent instances of one instruction (asm volatile, so nothing is folded). */
typedef unsigned int u32;
typedef unsigned long long u64;
#define KERNEL __attribute__((nvptx_kernel))

#define BODY(NAME, DECL, OP)                                                                       \
    KERNEL void NAME(u32 n, u32 * out) {                                                           \
        u32 t = __nvvm_read_ptx_sreg_tid_x() + __nvvm_read_ptx_sreg_ctaid_x() * 1024u;             \
        u32 a0 = t, a1 = t + 1, a2 = t + 2, a3 = t + 3, a4 = t + 4, a5 = t + 5, a6 = t + 6, a7 = t + 7; \
        DECL                                                                                       \
        for (u32 i = 0; i < n; ++i) {                                                              \
            OP(a0) OP(a1) OP(a2) OP(a3) OP(a4) OP(a5) OP(a6) OP(a7)                                \
        }                                                                                          \
        out[t] = a0 ^ a1 ^ a2 ^ a3 ^ a4 ^ a5 ^ a6 ^ a7;                                            \
    }

#define OP_PRMT(A) __asm__ volatile("prmt.b32 %0, %0, %1, 0x7044;" : "+r"(A) : "r"(0x43000000u));
#define OP_PRMT_B(A) __asm__ volatile("prmt.b32 %0, %0, %0, 0x3210;" : "+r"(A));
#define OP_AND(A) __asm__ volatile("and.b32 %0, %0, %1;" : "+r"(A) : "r"(0x0F0F0F0Fu));
#define OP_SHR(A) __asm__ volatile("shr.u32 %0, %0, 3;" : "+r"(A));
#define OP_ADDI(A) __asm__ volatile("add.u32 %0, %0, 7;" : "+r"(A));
#define OP_FMA(A) __asm__ volatile("fma.rn.f32 %0, %0, %1, %2;" : "+f"(*(float *) &A) : "f"(1.0000001f), "f"(1e-30f));
#define OP_CVT(A) __asm__ volatile("{ .reg .f32 f; cvt.rn.f32.u32 f, %0; cvt.rzi.u32.f32 %0, f; }" : "+r"(A));
#define OP_BFE(A) __asm__ volatile("bfe.u32 %0, %0, 8, 8;" : "+r"(A));
#define OP_MADI(A) __asm__ volatile("mad.lo.u32 %0, %0, 3, 1;" : "+r"(A));
#define OP_SELP(A) __asm__ volatile("{ .reg .pred p; setp.lt.u32 p, %0, 5; selp.u32 %0, %0, 9, p; }" : "+r"(A));

BODY(k_prmt, , OP_PRMT)
BODY(k_prmt_b, , OP_PRMT_B)
BODY(k_and, , OP_AND)
BODY(k_shr, , OP_SHR)
BODY(k_addi, , OP_ADDI)
BODY(k_fma, , OP_FMA)
BODY(k_cvt, , OP_CVT)
BODY(k_bfe, , OP_BFE)
BODY(k_madi, , OP_MADI)
BODY(k_selp, , OP_SELP)
