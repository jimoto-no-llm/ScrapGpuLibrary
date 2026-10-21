# sourced by the measurement scripts on the TB250
export PATH=$HOME/.local/bin:$PATH
ROOT=$HOME/moe-ep
LB=$ROOT/src/llama-build/bin
PLUG=$ROOT/src/build-cl11/libggml-cl11.so
MODEL=$ROOT/models/LFM2.5-8B-A1B-Q4_0.gguf
RUNS=$ROOT/logs/runs
TEXT=$ROOT/kl/text.txt
VKICD=/usr/share/vulkan/icd.d/radeon_icd.json
mkdir -p "$RUNS"
# use_env NAME: export the environment of one backend / multi-GPU configuration; sets DEV (the -dev argument)
use_env() {
  unset GGML_BACKEND_PATH GGML_CL11_PLATFORMS GGML_CL11_DEVICES RUSTICL_ENABLE VK_ICD_FILENAMES GGML_VK_VISIBLE_DEVICES
  export VK_ICD_FILENAMES=$VKICD   # NVIDIA 390 Vulkan ICD crashes: the Vulkan loader only sees the AMD cards
  case "$1" in
    cpu)       DEV=none ;;
    gt430)     export GGML_BACKEND_PATH=$PLUG GGML_CL11_PLATFORMS=NVIDIA GGML_CL11_DEVICES="GT 430"; DEV=CL11_0 ;;
    gt730)     export GGML_BACKEND_PATH=$PLUG GGML_CL11_PLATFORMS=NVIDIA GGML_CL11_DEVICES="GT 730"; DEV=CL11_0 ;;
    gt710)     export GGML_BACKEND_PATH=$PLUG GGML_CL11_PLATFORMS=NVIDIA GGML_CL11_DEVICES="GT 710"; DEV=CL11_0 ;;
    rx-cl11)   export GGML_BACKEND_PATH=$PLUG RUSTICL_ENABLE=radeonsi GGML_CL11_PLATFORMS=rusticl GGML_CL11_DEVICES="RX 6400"; DEV=CL11_0 ;;
    wx-cl11)   export GGML_BACKEND_PATH=$PLUG GGML_CL11_PLATFORMS=Clover GGML_CL11_DEVICES="WX 2100"; DEV=CL11_0 ;;
    rx-vk)     DEV=Vulkan0 ;;
    wx-vk)     DEV=Vulkan1 ;;
    *) echo "use_env: unknown $1" >&2; return 1 ;;
  esac
}

# use_mg NAME: multi-GPU configurations (one process). Sets DEV. The tensor split (-ts) follows the order of DEV.
# Vulkan is always loaded by llama.cpp's backend directory, so -dev lists the devices explicitly.
use_mg() {
  unset GGML_BACKEND_PATH GGML_CL11_PLATFORMS GGML_CL11_DEVICES RUSTICL_ENABLE VK_ICD_FILENAMES GGML_VK_VISIBLE_DEVICES GGML_CL11_AUTO_RPC
  export VK_ICD_FILENAMES=$VKICD
  case "$1" in
    # (a) docs/MULTI_GPU.md: 5 cards, CL11 only, GT 430 over the automatic RPC (RPC0 comes first)
    cl5)       export GGML_BACKEND_PATH=$PLUG RUSTICL_ENABLE=radeonsi GGML_CL11_PLATFORMS=NVIDIA,rusticl; DEV=RPC0/CL11_0/CL11_1/CL11_2/CL11_3 ;;
    # 4 cards in one process, CL11 only (no GT 430)
    cl4)       export GGML_BACKEND_PATH=$PLUG RUSTICL_ENABLE=radeonsi GGML_CL11_PLATFORMS=NVIDIA,rusticl GGML_CL11_DEVICES="GT 730,GT 710,RX 6400,WX 2100"; DEV=CL11_0/CL11_1/CL11_2/CL11_3 ;;
    # (b) the two AMD cards, official Vulkan
    amdvk)     DEV=Vulkan0/Vulkan1 ;;
    # (c) CL11 on the two Kepler cards + official Vulkan on the two AMD cards, one process
    mix4)      export GGML_BACKEND_PATH=$PLUG GGML_CL11_PLATFORMS=NVIDIA GGML_CL11_DEVICES="GT 730,GT 710"; DEV=CL11_0/CL11_1/Vulkan0/Vulkan1 ;;
    # same plus GT 430 over the automatic RPC
    mix5)      export GGML_BACKEND_PATH=$PLUG GGML_CL11_PLATFORMS=NVIDIA; DEV=RPC0/CL11_0/CL11_1/Vulkan0/Vulkan1 ;;
    *) echo "use_mg: unknown $1" >&2; return 1 ;;
  esac
}
