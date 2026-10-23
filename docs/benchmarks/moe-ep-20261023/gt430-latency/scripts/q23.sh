# q23 (under perf pin): per-thread run / run-queue wait / cpu of llama-bench and the device server, G and H, token generation
export UB=320 REPS=3 PROFILE=1 REMOTE_BE=opencl11 SAMP_SECONDS=100
. ./env.sh
G="Vulkan0:0.6,CL11_0:0.25,CL11_1:0.05,REMOTE0:0.1"
H="Vulkan0:0.65,CL11_0:0.28,CL11_1:0.07"
for v in "G $G" "H $H"; do
  set -- $v
  sglx_env "$2"
  TMO=900 ./run.sh q23_$1 ./withsched.sh q23_$1 $LB/llama-bench -m $MODEL -dev Vulkan0/Vulkan1/SGLX0 -ngl 99 -fa 0 -t 2 -b 512 -ub 320 -r 3 -sm layer -ncmoe 0 -ts 5/1/0 -ot "token_embd\.weight=SGLX0" -o jsonl -p 0 -n 256
done
