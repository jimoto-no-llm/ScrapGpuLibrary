# card_env CARD  -> exports the environment of one card / path and sets DEVNAME (the llama.cpp device name) and the label.
# CARDs: gt430 gt730 gt710 (CL11, NVIDIA 390), rx6400-vk wx2100-vk (official Vulkan), rx6400-cl wx2100-cl (CL11 rusticl), wx2100-clover (CL11 Clover), cpu
card_env() {
  unset GGML_BACKEND_PATH GGML_CL11_PLATFORMS GGML_CL11_DEVICES RUSTICL_ENABLE GGML_VK_VISIBLE_DEVICES GGML_CL11_AUTO_RPC
  export VK_ICD_FILENAMES=$VKICD EG2_BACKEND_DIR=$LB
  case "$1" in
    gt430|gt730|gt710)
      export GGML_BACKEND_PATH=$PLUG GGML_CL11_PLATFORMS=NVIDIA GGML_CL11_AUTO_RPC=0
      case "$1" in gt430) export GGML_CL11_DEVICES="GT 430";; gt730) export GGML_CL11_DEVICES="GT 730";; gt710) export GGML_CL11_DEVICES="GT 710";; esac
      DEVNAME=CL11_0 ;;
    rx6400-vk) export GGML_VK_VISIBLE_DEVICES=0; DEVNAME=Vulkan0 ;;
    wx2100-vk) export GGML_VK_VISIBLE_DEVICES=1; DEVNAME=Vulkan0 ;;
    rx6400-cl) export GGML_BACKEND_PATH=$PLUG GGML_CL11_PLATFORMS=rusticl GGML_CL11_DEVICES="RX 6400" GGML_CL11_AUTO_RPC=0 RUSTICL_ENABLE=radeonsi GGML_VK_VISIBLE_DEVICES=; DEVNAME=CL11_0 ;;
    wx2100-cl) export GGML_BACKEND_PATH=$PLUG GGML_CL11_PLATFORMS=rusticl GGML_CL11_DEVICES="WX 2100" GGML_CL11_AUTO_RPC=0 RUSTICL_ENABLE=radeonsi; DEVNAME=CL11_0 ;;
    wx2100-clover) export GGML_BACKEND_PATH=$PLUG GGML_CL11_PLATFORMS=Clover GGML_CL11_DEVICES="WX 2100" GGML_CL11_AUTO_RPC=0; DEVNAME=CL11_0 ;;
    vk-pair) DEVNAME=Vulkan0,Vulkan1 ;;   # layer split of one process over RX 6400 (Vulkan0) and WX 2100 (Vulkan1)
    kepler-pair) export GGML_BACKEND_PATH=$PLUG GGML_CL11_PLATFORMS=NVIDIA GGML_CL11_DEVICES="GT 730,GT 710" GGML_CL11_AUTO_RPC=0; DEVNAME=CL11_0,CL11_1 ;;
    cpu) DEVNAME=CPU ;;
    *) echo "unknown card $1" >&2; return 1 ;;
  esac
}
