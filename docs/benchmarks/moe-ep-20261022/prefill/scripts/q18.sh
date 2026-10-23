. ./env.sh
SH=$HOME/moe-ep/c1p/vkshim.so
C="Vulkan0:0.5,Vulkan1:0.2,CL11_0:0.2,CL11_1:0.1"
sglx_env "$C"
./run.sh q22_tl_C_ub256 env LD_PRELOAD=$SH VKSHIM_OUT=$LOGS/q22_tl_C_ub256.txt VKSHIM_EVENTS=1 $LB/llama-bench -m $MODEL -dev Vulkan0/Vulkan1/SGLX0 -ngl 99 -fa 0 -t 2 -b 512 -ub 256 -sm layer -ncmoe 0 -ts 5/1/0 -ot "token_embd\.weight=SGLX0" -o jsonl -p 512 -n 0 -r 2
sglx_env ""
./run.sh q22_tl_noop_ub256 env LD_PRELOAD=$SH VKSHIM_OUT=$LOGS/q22_tl_noop_ub256.txt VKSHIM_EVENTS=1 $LB/llama-bench -m $MODEL -dev Vulkan0/Vulkan1 -ngl 99 -fa 0 -t 2 -b 512 -ub 256 -sm layer -ncmoe 0 -ts 5/1 -ot no_such_tensor_zz=CPU -o jsonl -p 512 -n 0 -r 2
