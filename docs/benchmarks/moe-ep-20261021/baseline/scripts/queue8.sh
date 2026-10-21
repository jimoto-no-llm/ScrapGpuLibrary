#!/bin/bash
# KL divergence against the CPU logits (kld.sh base was run first): C0, C1, five cards, then the other single bests and mixes
S="$(dirname "$0")"
"$S/kld.sh" kld_C0_rx-vk_ncmoe5 single rx-vk 5
"$S/kld.sh" kld_C1_amdvk_ts5-1 mg amdvk 5,1
"$S/kld.sh" kld_cl5_ts8-8-18-40-18 mg cl5 8,8,18,40,18
"$S/kld.sh" kld_gt730_ncmoe22 single gt730 22
"$S/kld.sh" kld_gt430_ncmoe24 single gt430 24
"$S/kld.sh" kld_gt710_ncmoe20 single gt710 20
"$S/kld.sh" kld_wx-vk_ncmoe17 single wx-vk 17
"$S/kld.sh" kld_rx-cl11_ncmoe14 single rx-cl11 14
"$S/kld.sh" kld_wx-cl11_ncmoe18 single wx-cl11 18
"$S/kld.sh" kld_mix4_ts1-4-60-20 mg mix4 1,4,60,20
"$S/kld.sh" kld_mix5_ts2-2-8-50-20 mg mix5 2,2,8,50,20
echo QUEUE8_DONE
