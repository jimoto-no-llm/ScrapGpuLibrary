# sourced by the GT 430 measurement scripts on the TB250 (work directory ~/moe-ep/g430/)
export PATH=$HOME/.local/bin:$PATH
ROOT=$HOME/moe-ep
SG=$ROOT/g430/scripts
LB=$ROOT/src/llama-build/bin                 # llama.cpp 23b0202 (shared build of the earlier measurements, read only)
PLUG=${PLUG:-$ROOT/g430/plug2/libggml-cl11.so}
SERVER=${SERVER:-$ROOT/g430/srv2/scrapgpu-device-server}
MODEL=$ROOT/models/LFM2.5-8B-A1B-Q4_0.gguf
TEXT=$ROOT/kl/text.txt
CPU_BASE=$ROOT/kl/base.kld                   # CPU reference logits (llama-perplexity --kl-divergence-base)
LOGS=$ROOT/g430/logs
VKICD=/usr/share/vulkan/icd.d/radeon_icd.json # NVIDIA 390's Vulkan ICD crashes: the loader only sees the AMD cards
RUN=$SG/run.sh
mkdir -p "$LOGS"

# sglx_env SPLIT: environment of one run. SPLIT = GGML_SGLX_SPLIT ("" = plain C1 without the plugin)
# REMOTE_BE=opencl11|cuda: the backend of the device server that serves REMOTE0 (the GT 430, alone in its process)
sglx_env() {
  unset GGML_BACKEND_PATH GGML_CL11_PLATFORMS GGML_CL11_DEVICES RUSTICL_ENABLE GGML_SGLX_SPLIT GGML_SGLX_PROFILE GGML_SGLX_SERIAL GGML_VK_FORCE_MMVQ GGML_CL11_AUTO_RPC \
        GGML_SGLX_REMOTE_BACKEND GGML_SGLX_REMOTE_ENV GGML_SGLX_REMOTE_SOCKET SCRAPGPU_DEVICE_SERVER SCRAPGPU_CUDA_DEVICES GGML_SGLX_AUTOTUNE GGML_SGLX_REMOTE_BATCH
  export VK_ICD_FILENAMES=$VKICD
  if [ -n "$1" ]; then
    export GGML_BACKEND_PATH=$PLUG GGML_CL11_PLATFORMS=NVIDIA GGML_CL11_DEVICES="GT 730,GT 710" GGML_CL11_AUTO_RPC=0
    export GGML_SGLX_SPLIT="$1"
    export GGML_SGLX_PROFILE=${PROFILE:-1}
    [ -n "$SERIAL" ] && export GGML_SGLX_SERIAL=1
    case "$1" in *REMOTE*)
      export SCRAPGPU_DEVICE_SERVER=$SERVER GGML_SGLX_REMOTE_BACKEND=${REMOTE_BE:-opencl11}
      export GGML_SGLX_REMOTE_ENV="GGML_CL11_PLATFORMS=NVIDIA;GGML_CL11_DEVICES=GT 430;SCRAPGPU_CUDA_DEVICES=GT 430" ;;
    esac
    [ -n "$CUDA_SCHED" ] && export GGML_SGLX_REMOTE_ENV="$GGML_SGLX_REMOTE_ENV;SCRAPGPU_CUDA_SCHED=$CUDA_SCHED"
    case "$1" in *CUDA*) export SCRAPGPU_CUDA_DEVICES="GT 730,GT 710" ;; esac
    [ -n "$RBATCH" ] && export GGML_SGLX_REMOTE_BATCH=$RBATCH
    [ -n "$AUTOTUNE" ] && export GGML_SGLX_AUTOTUNE=$AUTOTUNE
  fi
}
