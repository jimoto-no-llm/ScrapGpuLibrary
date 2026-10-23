# q11 (under perf pin): where the host CPU goes (cpusamp.py: % of one core of llama-bench, its threads and the device server) in token generation
export UB=320 REPS=2 PROFILE=0 REMOTE_BE=opencl11 SAMP_SECONDS=120
. ./env.sh
A="Vulkan0:0.45,Vulkan1:0.2,CL11_0:0.2,CL11_1:0.08,REMOTE0:0.07"
H="Vulkan0:0.65,CL11_0:0.28,CL11_1:0.07"
C="Vulkan0:0.5,Vulkan1:0.2,CL11_0:0.2,CL11_1:0.1"
for v in "A $A" "H $H" "C $C"; do
  set -- $v
  sglx_env "$2"
  [ $1 = A ] && export REMOTE_BE=opencl11
  TMO=900 ./run.sh q13_$1 ./withsamp.sh q13_$1 $LB/llama-bench -m $MODEL -dev Vulkan0/Vulkan1/SGLX0 -ngl 99 -fa 0 -t 2 -b 512 -ub 320 -r 3 -sm layer -ncmoe 0 -ts 5/1/0 -ot 'token_embd\.weight=SGLX0' -o jsonl -p 0 -n 256
done
