. ./env.sh
sglx_env ""
SH=$HOME/moe-ep/c1p/vkshim.so
for ub in 512 320; do
  ./run.sh q8_tl_ub$ub env LD_PRELOAD=$SH VKSHIM_OUT=$LOGS/q8_tl_ub$ub.txt VKSHIM_EVENTS=1 $LB/llama-bench -m $MODEL -dev Vulkan0/Vulkan1 -ngl 99 -fa 0 -t 2 -b 512 -ub $ub -sm layer -ncmoe 0 -ts 5/1 -o jsonl -p 512 -n 0 -r 2
done
