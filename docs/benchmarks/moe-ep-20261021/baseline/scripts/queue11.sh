#!/bin/bash
S="$(dirname "$0")"
"$S/kld.sh" kld_C1_amdvk_ts5-1 mg amdvk 5,1
"$S/kld.sh" kld_cl5_ts8-8-18-40-18 mg cl5 8,8,18,40,18
"$S/kld.sh" kld_mix4_ts1-4-60-20 mg mix4 1,4,60,20
"$S/kld.sh" kld_mix5_ts2-2-8-50-20 mg mix5 2,2,8,50,20
"$S/profile2.sh"
echo QUEUE11_DONE
