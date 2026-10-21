#!/bin/bash
# 1-token profiles (tg only, -r 1) and graph splits. usage: profile.sh
. "$(dirname "$0")/env.sh"
S="$(dirname "$0")"
bench() { # TAG env-setup(function) extra-env...  -- args
  :
}
# --- C0: RX 6400 official Vulkan, --n-cpu-moe 5
use_env rx-vk
GGML_VK_PERF_LOGGER=1 "$S/run.sh" prof_c0_vk_perf "$LB/llama-bench" -m "$MODEL" -dev Vulkan0 -ngl 99 -fa 0 -t 2 -b 128 -ub 128 -ncmoe 5 -p 0 -n 32 -r 1 -o jsonl
GGML_SCHED_DEBUG=1 "$S/run.sh" prof_c0_vk_splits "$LB/llama-completion" -m "$MODEL" -dev Vulkan0 -ngl 99 -fa off -t 2 -b 128 -ub 128 -ncmoe 5 -c 256 -n 3 -p "Hello" --temp 0 -no-cnv
# --- C1: two AMD cards
use_mg amdvk
# GGML_VK_PERF_LOGGER=1 with two Vulkan devices aborts (GGML_ASSERT(ctx->compute_ctx.expired())), see logs/prof_c1_vk_perf.log
GGML_SCHED_DEBUG=1 "$S/run.sh" prof_c1_vk_splits "$LB/llama-completion" -m "$MODEL" -dev "${DEV//\//,}" -ngl 99 -fa off -t 2 -b 128 -ub 128 -ts 5,1 -sm layer -c 256 -n 3 -p "Hello" --temp 0 -no-cnv
# --- five cards CL11
use_mg cl5
GGML_CL11_LAUNCH_LOG=1 "$S/run.sh" prof_cl5_launch "$LB/llama-bench" -m "$MODEL" -dev "$DEV" -ngl 99 -fa 0 -t 2 -b 128 -ub 128 -ts 8/8/18/40/18 -sm layer -p 0 -n 16 -r 1 -o jsonl
GGML_CL11_PROFILE=1 "$S/run.sh" prof_cl5_kernels "$LB/llama-bench" -m "$MODEL" -dev "$DEV" -ngl 99 -fa 0 -t 2 -b 128 -ub 128 -ts 8/8/18/40/18 -sm layer -p 0 -n 16 -r 1 -o jsonl
GGML_SCHED_DEBUG=1 "$S/run.sh" prof_cl5_splits "$LB/llama-completion" -m "$MODEL" -dev "${DEV//\//,}" -ngl 99 -fa off -t 2 -b 128 -ub 128 -ts 8,8,18,40,18 -sm layer -c 256 -n 3 -p "Hello" --temp 0 -no-cnv
# --- GT 730 (CL11, experts of 2 layers on the card)
use_env gt730
GGML_CL11_LAUNCH_LOG=1 "$S/run.sh" prof_gt730_launch "$LB/llama-bench" -m "$MODEL" -dev CL11_0 -ngl 99 -fa 0 -t 2 -b 128 -ub 128 -ncmoe 22 -p 0 -n 16 -r 1 -o jsonl
GGML_CL11_PROFILE=1 "$S/run.sh" prof_gt730_kernels "$LB/llama-bench" -m "$MODEL" -dev CL11_0 -ngl 99 -fa 0 -t 2 -b 128 -ub 128 -ncmoe 22 -p 0 -n 16 -r 1 -o jsonl
GGML_SCHED_DEBUG=1 "$S/run.sh" prof_gt730_splits "$LB/llama-completion" -m "$MODEL" -dev CL11_0 -ngl 99 -fa off -t 2 -b 128 -ub 128 -ncmoe 22 -c 256 -n 3 -p "Hello" --temp 0 -no-cnv
echo PROFILE_DONE
