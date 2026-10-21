#!/bin/bash
# C1 / C0 under power_dpm_force_performance_level=high vs auto; always restores auto.
source ~/moe-ep/scripts/env.sh
O=~/moe-ep/perf
trap "sudo -n /usr/local/sbin/sgl-driver-switch perf auto >>$O/perf.log 2>&1" EXIT
cool() { for h in /sys/class/drm/card*/device/hwmon/hwmon*/temp1_input; do while [ "$(cat $h)" -ge 85000 ]; do sleep 5; done; done; }
B="$LB/llama-bench -m $MODEL -ngl 99 -fa 0 -t 2 -o jsonl"
for mode in high auto; do
  sudo -n /usr/local/sbin/sgl-driver-switch perf $mode >>$O/perf.log 2>&1
  for extra in "" "GGML_VK_FORCE_MMVQ=1"; do
    tag=${mode}${extra:+_mmvq}
    use_mg amdvk; cool
    env $extra $B -dev $DEV -sm layer -ts 5/1 -b 512 -ub 512 -r 5 -p 512 -n 128 -pg 512,128 > $O/c1_$tag.jsonl 2>$O/c1_$tag.err
    use_env rx-vk; cool
    env $extra $B -dev $DEV -ncmoe 5 -b 128 -ub 128 -r 5 -p 512 -n 128 -pg 512,128 > $O/c0_$tag.jsonl 2>$O/c0_$tag.err
    cat /sys/class/drm/card*/device/pp_dpm_sclk > $O/sclk_$tag.txt 2>/dev/null
  done
done
echo DONE >> $O/perf.log
