# sourced by the Qwen3-30B-A3B measurement scripts (work directory ~/moe-ep/q30/)
export PATH=$HOME/.local/bin:$PATH
ROOT=$HOME/moe-ep
Q=$ROOT/q30
LB=$ROOT/src/llama-build/bin
PLUG=$ROOT/src/build-cl11/libggml-cl11.so   # baseline plugin (no SGLX0 device needed here)
MODELDIR=$ROOT/models
MODEL=${MODEL:-$MODELDIR/Qwen3-30B-A3B-Q4_0.gguf}
TEXT=$ROOT/kl/text.txt
LOGS=$Q/logs
VKICD=/usr/share/vulkan/icd.d/radeon_icd.json
mkdir -p "$LOGS"
# cfg NAME: sets DEV (llama-bench -dev) and the environment
cfg() {
  unset GGML_BACKEND_PATH GGML_CL11_PLATFORMS GGML_CL11_DEVICES RUSTICL_ENABLE GGML_CL11_AUTO_RPC GGML_VK_VISIBLE_DEVICES
  export VK_ICD_FILENAMES=$VKICD
  case "$1" in
    cpu)    DEV=none ;;
    rx)     DEV=Vulkan0 ;;
    wx)     DEV=Vulkan1 ;;
    amd)    DEV=Vulkan0/Vulkan1 ;;
    # AMD official Vulkan x2 + CL11 on GT 730 / GT 710 (GT 430 not used)
    mix4)   export GGML_BACKEND_PATH=$PLUG GGML_CL11_PLATFORMS=NVIDIA GGML_CL11_DEVICES="GT 730,GT 710" GGML_CL11_AUTO_RPC=0; DEV=Vulkan0/Vulkan1/CL11_0/CL11_1 ;;
    mix5)   export GGML_BACKEND_PATH=$PLUG GGML_CL11_PLATFORMS=NVIDIA; DEV=RPC0/CL11_0/CL11_1/Vulkan0/Vulkan1 ;;
    cl5)    export GGML_BACKEND_PATH=$PLUG RUSTICL_ENABLE=radeonsi GGML_CL11_PLATFORMS=NVIDIA,rusticl; DEV=RPC0/CL11_0/CL11_1/CL11_2/CL11_3 ;;
    # same devices, ordered so the late layers (which keep their experts on the GPU) land on the cards with VRAM: RX, WX, GT 710, GT 730, GT 430 (RPC0)
    cl5b)   export GGML_BACKEND_PATH=$PLUG RUSTICL_ENABLE=radeonsi GGML_CL11_PLATFORMS=NVIDIA,rusticl; DEV=CL11_2/CL11_3/CL11_1/CL11_0/RPC0 ;;
    mix5b)  export GGML_BACKEND_PATH=$PLUG GGML_CL11_PLATFORMS=NVIDIA; DEV=Vulkan0/Vulkan1/CL11_1/CL11_0/RPC0 ;;
    mix4b)  export GGML_BACKEND_PATH=$PLUG GGML_CL11_PLATFORMS=NVIDIA GGML_CL11_DEVICES="GT 730,GT 710" GGML_CL11_AUTO_RPC=0; DEV=Vulkan0/Vulkan1/CL11_1/CL11_0 ;;
    gt730)  export GGML_BACKEND_PATH=$PLUG GGML_CL11_PLATFORMS=NVIDIA GGML_CL11_DEVICES="GT 730" GGML_CL11_AUTO_RPC=0; DEV=CL11_0 ;;
    *) echo "cfg: unknown $1" >&2; return 1 ;;
  esac
}
