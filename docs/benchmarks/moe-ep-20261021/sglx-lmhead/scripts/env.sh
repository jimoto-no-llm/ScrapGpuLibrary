# sourced by the SGLX measurement scripts on the TB250 (work directory ~/moe-ep/sglx/)
export PATH=$HOME/.local/bin:$PATH
ROOT=$HOME/moe-ep
SG=$ROOT/sglx
LB=$ROOT/src/llama-build/bin                 # llama.cpp 23b0202 (shared build of the earlier measurements, read only)
PLUG=$SG/build-cl11/libggml-cl11.so          # the plugin with the SGLX0 device
MODEL=$ROOT/models/LFM2.5-8B-A1B-Q4_0.gguf
TEXT=$ROOT/kl/text.txt
CPU_BASE=$ROOT/kl/base.kld                   # CPU reference logits (llama-perplexity --kl-divergence-base), made by the baseline measurements
LOGS=$SG/logs
VKICD=/usr/share/vulkan/icd.d/radeon_icd.json # NVIDIA 390's Vulkan ICD crashes: the loader only sees the AMD cards
RUN=$SG/run.sh
mkdir -p "$LOGS"

# sglx_env SPLIT: environment of one run. SPLIT = GGML_SGLX_SPLIT ("" = plain C1 without the plugin)
sglx_env() {
  unset GGML_BACKEND_PATH GGML_CL11_PLATFORMS GGML_CL11_DEVICES RUSTICL_ENABLE GGML_SGLX_SPLIT GGML_SGLX_PROFILE GGML_SGLX_SERIAL GGML_VK_FORCE_MMVQ GGML_CL11_AUTO_RPC
  export VK_ICD_FILENAMES=$VKICD
  if [ -n "$1" ]; then
    export GGML_BACKEND_PATH=$PLUG GGML_CL11_PLATFORMS=NVIDIA GGML_CL11_DEVICES="GT 730,GT 710" GGML_CL11_AUTO_RPC=0
    export GGML_SGLX_SPLIT="$1"
    export GGML_SGLX_PROFILE=${PROFILE:-1}
    [ -n "$SERIAL" ] && export GGML_SGLX_SERIAL=1
  fi
}
