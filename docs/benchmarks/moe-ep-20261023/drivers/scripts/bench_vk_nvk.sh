#!/bin/bash
# bench_vk_nvk.sh CARD... : ggml-vulkan (the official backend) on a nouveau card through NVK. Needs Mesa that advertises Kepler:
# 25.0.7 (installed) only with NVK_I_WANT_A_BROKEN_VULKAN_DRIVER=1 (set here), 25.2.1+ without it. Fermi (GT 430) is not an NVK target.
# Success ladder: vulkaninfo lists the card -> llama-bench pp/tg runs -> test-backend-ops MUL_MAT passes. Stop at the first failure and keep the log.
. "$(dirname "$0")/env_drv.sh"
need_state nouveau nv470-nouveau430
for C in "$@"; do
  clean_env; I=$(card_idx $C) || continue
  export VK_ICD_FILENAMES=$ICD_NVK NVK_I_WANT_A_BROKEN_VULKAN_DRIVER=1 GGML_VK_VISIBLE_DEVICES=$I
  # the visible-device index is the Vulkan enumeration order; probe_state.sh vk-nvk-broken.txt shows which index is which card
  TMO=300 $RUN drv_nvk_${C}_ops "$LB/test-backend-ops" test -b Vulkan0 -o MUL_MAT -p "type_a=q4_0"
  $RUN drv_nvk_${C}_bench "$LB/llama-bench" -m "$M_SMALL" -dev Vulkan0 -ngl 99 -fa 0 -t 2 -b 512 -ub 320 -r 5 -o jsonl -p 128 -n 64
done
