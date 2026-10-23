#!/bin/bash
# bench_cl11_470.sh : the existing winning configurations with the 470 OpenCL (libnvidia-opencl 470) in place of 390's. Same command lines as
# hec/scripts/q_lfm.sh (config H, LFM) and hec/scripts/cfg_n.sh (config N, 30B), so the 10/23 numbers (tg 69.0 / pg 240.4 for H with perf pin) are the reference.
# State nv470 (GT 430 has no driver) or nv470-nouveau430 (GT 430 on nouveau; use bench_cl11_cards.sh rusticl gt430 for it).
. "$(dirname "$0")/env_drv.sh"
need_state nv470 nv470-nouveau430
clean_env
export VK_ICD_FILENAMES=$ICD_AMD
sudo -n $SW perf pin; trap 'sudo -n $SW perf auto' EXIT
H_ENV=(GGML_SGLX_SPLIT=Vulkan0:0.78,CL11_0:0.17,CL11_1:0.05 GGML_BACKEND_PATH=$PLUG GGML_CL11_PLATFORMS=NVIDIA "GGML_CL11_DEVICES=GT 730,GT 710" GGML_CL11_AUTO_RPC=0 GGML_SGLX_PROFILE=0)
for round in 1 2; do
  $RUN drv_470_lfm_c1_$round taskset -c 0 "$LB/llama-bench" -m "$M_LFM" -dev Vulkan0/Vulkan1 -ngl 99 -fa 0 -t 2 -b 512 -ub 320 -r 5 -sm layer -ncmoe 0 -ts 5/1 -o jsonl -p 512 -n 128 -pg 512,128
  env "${H_ENV[@]}" $RUN drv_470_lfm_H_$round taskset -c 0 "$LB/llama-bench" -m "$M_LFM" -dev Vulkan0/Vulkan1/SGLX0 -ngl 99 -fa 0 -t 2 -b 512 -ub 320 -r 5 -sm layer -ncmoe 0 -ts 5/1/0 -ot 'token_embd\.weight=SGLX0' -o jsonl -p 512 -n 128 -pg 512,128
done
if [ "${WITH_30B:-0}" = 1 ]; then   # config N needs the hec tooling (hec_gen / hec_run.sh); run hec/scripts/q_final.sh with this state instead, and compare with its 390 logs
  echo "30B: run ~/moe-ep/hec/scripts/q_final.sh unchanged under this state; only the OpenCL library differs" >&2
fi
