. ./env.sh
sglx_env ""
SP=$HOME/moe-ep/c1p/sprof.so
./run.sh q7_sprof_ub256 env LD_PRELOAD=$SP SPROF_OUT=$LOGS/q7_sprof_ub256.txt SPROF_HZ=1000 $LB/llama-bench -m $MODEL -dev Vulkan0/Vulkan1 -ngl 99 -fa 0 -t 2 -b 512 -ub 256 -sm layer -ncmoe 0 -ts 5/1 -o jsonl -p 512 -n 0 -r 3
