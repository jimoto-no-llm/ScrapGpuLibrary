#!/bin/bash
# The moe_rt_bench runs of MB0-MB5 on the TB250 (docs/plans/MOE_COMPOSITE_DEVICE.md chapter 9).
#   usage: suite.sh GROUP...      GROUP: mb0 mb1c mb1 mb2 mb3 mb3g mb3x mb4 mb5 mb5r mb5g (DRY_RUN=1 only prints the commands)
# Every command goes through moe_rt_bench_run.sh (bench lock, below 85 C, log header with the temperatures and the environment).
H=$HOME/moe-ep/rt-bench; BIN=$H/moe_rt_bench; RUN=$H/moe_rt_bench_run.sh; LOGS=${LOGS:-$H/logs}; CSV=$LOGS/all.csv
B=$HOME/moe-ep/src/llama-build/bin                       # llama.cpp build of ~/moe-ep (read only): libggml-vulkan.so, libggml-cpu.so
VK=/usr/share/vulkan/icd.d/radeon_icd.json               # the NVIDIA 390 Vulkan ICD crashes: only the AMD cards are visible to Vulkan
RC=radeonsi
mkdir -p "$LOGS"
r() { local tag=$1; shift; "$RUN" "$LOGS" "$tag" "$BIN" "$@" --csv "$CSV"; }
for g in "$@"; do
  case $g in
  mb0)
    r mb0_asc mb0 --cu-order asc
    r mb0_rev mb0 --cu-order rev
    r mb0_fermi_first mb0 --cu-devs 1,0,2          # CUDA indices on the TB250: 0 = GT 730, 1 = GT 430, 2 = GT 710
    r mb0_fermi_only mb0 --cu-devs 1
    ;;
  mb1c)
    r mb1c mb1c                                    # GT 730 + GT 710 (a Fermi context cannot join them)
    r mb1c_fermi mb1c --cu-devs 1                 # GT 430 alone
    ;;
  mb1)
    r mb1_gt430 mb1 --platforms NVIDIA --devices "GT 430"
    r mb1_gt730 mb1 --platforms NVIDIA --devices "GT 730"
    r mb1_gt710 mb1 --platforms NVIDIA --devices "GT 710"
    r mb1_wx_clover mb1 --platforms Clover --devices "WX 2100"
    RUSTICL_ENABLE=$RC r mb1_wx_rusticl mb1 --platforms rusticl --devices "WX 2100"
    RUSTICL_ENABLE=$RC r mb1_rx_rusticl mb1 --platforms rusticl --devices "RX 6400"
    ;;
  mb2)
    RUSTICL_ENABLE=$RC r mb2_4dev_nv_rusticl mb2 --platforms NVIDIA,rusticl --devices "GT 730,GT 710,WX 2100,RX 6400"
    r mb2_kepler_pair mb2 --platforms NVIDIA --devices "GT 730,GT 710"
    r mb2_3dev_nv_clover mb2 --platforms NVIDIA,Clover --devices "GT 730,GT 710,WX 2100"
    RUSTICL_ENABLE=$RC r mb2_3dev_nv_rusticl mb2 --platforms NVIDIA,rusticl --devices "GT 730,GT 710,WX 2100"
    ;;
  mb3)
    r mb3_gt430 mb3 --platforms NVIDIA --devices "GT 430"
    RUSTICL_ENABLE=$RC r mb3_es_rusticl mb3 --platforms NVIDIA,rusticl --devices "WX 2100,GT 730,GT 710"
    RUSTICL_ENABLE=$RC r mb3_rx_rusticl mb3 --platforms rusticl --devices "RX 6400"
    r mb3_es_clover mb3 --platforms NVIDIA,Clover --devices "WX 2100,GT 730,GT 710"
    ;;
  mb3g) VK_ICD_FILENAMES=$VK GGML_LIB_DIR=$B r mb3g mb3g --lib-dir "$B" --threads 2 ;;
  mb3x)    # ES with the WX 2100 slice on the official ggml-vulkan (Vulkan1) and the NVIDIA slices on OpenCL, one process
    VK_ICD_FILENAMES=$VK GGML_LIB_DIR=$B r mb3x_wxvk_gt730_gt710 mb3x --lib-dir "$B" --ggml-devs Vulkan1 --platforms NVIDIA --devices "GT 730,GT 710"
    VK_ICD_FILENAMES=$VK GGML_LIB_DIR=$B r mb3x_wxvk_gt730 mb3x --lib-dir "$B" --ggml-devs Vulkan1 --platforms NVIDIA --devices "GT 730"
    ;;
  mb4)  VK_ICD_FILENAMES=$VK GGML_LIB_DIR=$B r mb4 mb4 --lib-dir "$B" ;;
  mb5)
    for n in 128 512; do
      RUSTICL_ENABLE=$RC r mb5_es_ep_n$n mb5 --ntok $n --iters 10 --platforms NVIDIA,rusticl --devices "WX 2100,GT 730,GT 710"
      RUSTICL_ENABLE=$RC r mb5_rx_n$n mb5 --ntok $n --iters 10 --platforms rusticl --devices "RX 6400"
    done
    ;;
  mb5r)    # prompt batch with the ES / EP split proportional to the measured GFLOPS of mb5 (WX 2100 : GT 730 : GT 710 = 176 : 55 : 27), not to the bandwidth
    for n in 128 512; do
      RUSTICL_ENABLE=$RC r mb5r_3dev_n$n mb5 --ntok $n --iters 10 --platforms NVIDIA,rusticl --devices "WX 2100,GT 730,GT 710" --ratios 176,55,27
      RUSTICL_ENABLE=$RC r mb5r_2dev_n$n mb5 --ntok $n --iters 10 --platforms NVIDIA,rusticl --devices "WX 2100,GT 730" --ratios 176,55
    done
    ;;
  mb5g)
    for n in 128 512; do
      VK_ICD_FILENAMES=$VK GGML_LIB_DIR=$B r mb5g_n$n mb5g --ntok $n --iters 10 --lib-dir "$B" --ggml-devs Vulkan0,Vulkan1
    done
    ;;
  *) echo "unknown group $g" >&2; exit 2 ;;
  esac
done
