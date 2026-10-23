#!/bin/bash
# bench_cl11_cards.sh PLATFORM CARD... : CL11 plugin on single cards through one OpenCL platform, small model.
#   PLATFORM = NVIDIA (390 / 470 proprietary OpenCL, the baseline), rusticl (RUSTICL_ENABLE=nouveau), Clover (Mesa 25.0.7's old front end)
#   CARD     = gt430 gt730 gt710
# Per card: llama-bench pp128 / tg64 (-r 5) plus a short logit check against the CPU reference if KLD=1 (needs $ROOT/kl files, see baseline/scripts/kld.sh).
# Compare with the same script on NVIDIA/390 taken in the same session (docs/HARDWARE.md has older numbers for other models, do not mix them).
. "$(dirname "$0")/env_drv.sh"
P=${1:?NVIDIA|rusticl|Clover}; shift
STATE=$(cat /run/sgl-driver.state 2>/dev/null || echo unknown)
case $P in NVIDIA) need_state nv390 nv470 nv470-nouveau430;; *) need_state nouveau nouveau-kepler nv470-nouveau430;; esac
for C in "$@"; do
  clean_env
  I=$(card_idx $C) || { echo "bad card $C" >&2; continue; }
  export VK_ICD_FILENAMES=$ICD_AMD GGML_BACKEND_PATH=$PLUG GGML_CL11_AUTO_RPC=0 GGML_CL11_PLATFORMS=$P
  DEV=CL11_0
  case $P in
    NVIDIA)  case $C in gt430) export GGML_CL11_DEVICES="GT 430";; gt730) export GGML_CL11_DEVICES="GT 730";; gt710) export GGML_CL11_DEVICES="GT 710";; esac ;;
    rusticl) export RUSTICL_ENABLE=nouveau:$I ;;                  # one device visible -> CL11_0
    Clover)  DEV=CL11_$I ;;                                       # Clover has no index filter: all nouveau cards are listed in probe order
  esac
  TAG=drv_${STATE}_${P}_${C}
  # bandwidth first (cheap, same code as the CUDA numbers of 10/22): q6_K 2048 x 128000 mat-vec, q4_0, and 32-token MUL_MAT_ID
  $RUN ${TAG}_mv_q6K "$CUDA_BIN/scrapgpu_mm_bench" opencl11 mv --type q6_K
  $RUN ${TAG}_mv_q4_0 "$CUDA_BIN/scrapgpu_mm_bench" opencl11 mv --type q4_0
  $RUN ${TAG}_mid32 "$CUDA_BIN/scrapgpu_mm_bench" opencl11 mid --tokens 32
  $RUN $TAG "$LB/llama-bench" -m "$M_SMALL" -dev $DEV -ngl 99 -fa 0 -t 2 -b 512 -ub 320 -r 5 -sm layer -o jsonl -p 128 -n 64
done
