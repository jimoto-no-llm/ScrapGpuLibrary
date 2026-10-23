#!/bin/bash
# (sourced, not executed) sourced by the driver-stage scripts on the TB250 (work directory ~/moe-ep/drivers/). Nothing here touches a GPU.
# These scripts are written for docs/plans/DRIVER_STAGES.md and were NOT run when committed (no measurement exists yet).
export PATH=$HOME/.local/bin:$PATH
ROOT=$HOME/moe-ep
DRV=$ROOT/drivers
LB=${LB:-$ROOT/src/llama-build/bin}                              # llama.cpp build with Vulkan + the plugin loader (same as the earlier stages)
PLUG=${PLUG:-$ROOT/hec/plug/libggml-cl11.so}                     # CL11 plugin (the build the current N / H configurations use)
CUDA_PLUG=${CUDA_PLUG:-$ROOT/fix/build-cuda/libggml-cl11.so}     # same plugin with CL11_WITH_CUDA=ON (pdev_cuda)
CUDA_BIN=${CUDA_BIN:-$ROOT/cuda/bB}                              # scrapgpu_cuda_kernel_tests / scrapgpu_ops_tests / scrapgpu_mm_bench (the 10/22 final logs used this directory)
M_SMALL=$ROOT/models/LFM2.5-1.2B-Instruct-Q4_0.gguf              # small model: fits GT 710 (1.9 GiB) and GT 430
M_LFM=$ROOT/models/LFM2.5-8B-A1B-Q4_0.gguf
M_30B=$ROOT/models/Qwen3-30B-A3B-Q4_0.gguf
RUN=$ROOT/sglx/run.sh                                            # flock ~/bench/gpu-bench.lock + temperature wait; TAG cmd...; LOGS comes from sglx/env.sh
SW=/usr/local/sbin/sgl-driver-switch
ICD_AMD=/usr/share/vulkan/icd.d/radeon_icd.json
ICD_NVK=/usr/share/vulkan/icd.d/nouveau_icd.json
ICD_NV=/etc/vulkan/icd.d/nvidia_icd.json                         # installed by the nvidia .run (390: api 1.0.65, 470: api 1.2.175)
OUT=${OUT:-$DRV/logs}
mkdir -p "$OUT"

# Card -> index among the cards of one driver, in PCI order (GT 430 04:00.0, GT 730 07:00.0, GT 710 08:00.0).
# Under nouveau the order follows probe order; probe_state.sh prints the real mapping, override with IDX_GT430 etc. if it differs (ESTIMATE).
IDX_GT430=${IDX_GT430:-0}; IDX_GT730=${IDX_GT730:-1}; IDX_GT710=${IDX_GT710:-2}
card_idx() { case "$1" in gt430) echo $IDX_GT430;; gt730) echo $IDX_GT730;; gt710) echo $IDX_GT710;; *) return 1;; esac; }

# clean_env: forget every backend selector of earlier runs
clean_env() {
  unset GGML_BACKEND_PATH GGML_CL11_PLATFORMS GGML_CL11_DEVICES GGML_CL11_AUTO_RPC RUSTICL_ENABLE RUSTICL_FEATURES RUSTICL_DEBUG \
        VK_ICD_FILENAMES GGML_VK_VISIBLE_DEVICES NVK_I_WANT_A_BROKEN_VULKAN_DRIVER GGML_SGLX_SPLIT GGML_SGLX_MOE SCRAPGPU_CUDA_DEVICES
}

# need_state WANT...: stop unless the switch script reports one of the wanted states (state file is /run/sgl-driver.state, readable)
need_state() {
  local cur; cur=$(cat /run/sgl-driver.state 2>/dev/null || echo unknown)
  for w in "$@"; do [ "$cur" = "$w" ] && return 0; done
  echo "state is '$cur', need one of: $* (switch with: sudo $SW <state>; the user does that)" >&2; exit 3
}
