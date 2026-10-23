#!/bin/bash
# bench_cuda470.sh : the repo's own CUDA backend (libcuda via dlopen, PTX sm_35, no nvcc) on 470, Kepler cards only.
# The 390 numbers of the same binaries are in docs/benchmarks/moe-ep-20261022/cuda/README.md (kernel tests 161/161, GT 730 q6_K 26.4 GB/s, q4_0 32.7 GB/s).
# Order: kernel tests (correctness) -> mm_bench (bandwidth) -> SGLX LFM config with CUDA devices. Needs state nv470 or nv470-nouveau430.
. "$(dirname "$0")/env_drv.sh"
need_state nv390
clean_env
for C in "GT 730" "GT 710"; do
  T=$(echo $C | tr -d ' ')
  $RUN drv_cuda390_ktest_$T env SCRAPGPU_CUDA_DEVICES="$C" "$CUDA_BIN/scrapgpu_cuda_kernel_tests"
  $RUN drv_cuda390_ops_$T env SCRAPGPU_TEST_BACKEND=cuda SCRAPGPU_CUDA_DEVICES="$C" "$CUDA_BIN/scrapgpu_ops_tests"
  for K in q4_0 q6_K; do   # same arguments as the 10/22 logs (cuda_final_cuda_GT730_mv_q6K)
    $RUN drv_cuda390_mv_${K}_$T env SCRAPGPU_CUDA_DEVICES="$C" "$CUDA_BIN/scrapgpu_mm_bench" cuda mv --type "$K"
  done
  $RUN drv_cuda390_mid32_$T env SCRAPGPU_CUDA_DEVICES="$C" "$CUDA_BIN/scrapgpu_mm_bench" cuda mid --tokens 32
done
# SGLX C (CUDA) like docs/benchmarks/moe-ep-20261022/cuda/scripts/sglxcuda.sh, one run each for C1 and CUDA (use REPS=10 and alternate twice for a real comparison)
. "$ROOT/sglx/env.sh"
sglx_env "Vulkan0:0.5,Vulkan1:0.2,CUDA0:0.2,CUDA1:0.1"
export GGML_BACKEND_PATH=$CUDA_PLUG SCRAPGPU_CUDA_DEVICES="GT 730,GT 710"
$RUN drv_cuda390_sglx_C "$LB/llama-bench" -m "$M_LFM" -dev Vulkan0/Vulkan1/SGLX0 -ngl 99 -fa 0 -t 2 -b 512 -ub 512 -r ${REPS:-5} -sm layer -ncmoe 0 -ts 5/1/0 -ot 'token_embd\.weight=SGLX0' -o jsonl -p 512 -n 128 -pg 512,128
