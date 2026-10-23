#!/bin/bash
# final matrix, power level auto: pp512 / tg128 / pg512,128, -r 5 (bench.sh defaults)
cd "$(dirname "$0")"
rm -f logs/queue5.done
for mm in 0 1; do
  M=""; [ $mm = 1 ] && M=1
  sfx=$([ $mm = 1 ] && echo _mmvq)
  FORCE_MMVQ=$M ./bench.sh "f_c1_plain${sfx}" ""
  FORCE_MMVQ=$M ./bench.sh "f_c1_noop${sfx}" noop
  FORCE_MMVQ=$M TS=4/1 ./bench.sh "f_c1_plain_ts41${sfx}" ""
  for cfg in A:Vulkan0:0.5,Vulkan1:0.5 B:Vulkan0:0.7,Vulkan1:0.3 C:Vulkan0:0.5,Vulkan1:0.2,CL11_0:0.2,CL11_1:0.1 D:Vulkan0:0.4,Vulkan1:0.3,CL11_0:0.2,CL11_1:0.1; do
    n=${cfg%%:*}; s=${cfg#*:}
    FORCE_MMVQ=$M ./bench.sh "f_sglx_${n}_ts51${sfx}" "$s"
    case $n in A|C) FORCE_MMVQ=$M TS=4/1 ./bench.sh "f_sglx_${n}_ts41${sfx}" "$s";; esac
  done
done
SERIAL=1 ./bench.sh f_sglx_A_ts51_serial "Vulkan0:0.5,Vulkan1:0.5"
echo done > logs/queue5.done
