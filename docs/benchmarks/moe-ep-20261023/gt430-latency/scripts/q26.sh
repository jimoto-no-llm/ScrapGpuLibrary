# q26 (under perf pin): split grid for G (with the GT 430) and H (without), host pinned to cpu 0, server + worker on cpu 1, tg only
export UB=320 REPS=3 PROFILE=1 REMOTE_BE=opencl11
A="-p 0 -n 128"
export WRAP="taskset -c 0"
g() { SCRAPGPU_SERVER_CPU=1 GGML_SGLX_REMOTE_CPU=1 ./bench.sh q26_G_$1 "Vulkan0:$2,CL11_0:$3,CL11_1:$4,REMOTE0:$5" $A; }
h() { ./bench.sh q26_H_$1 "Vulkan0:$2,CL11_0:$3,CL11_1:$4" $A; }
g g1 0.67 0.17 0.06 0.10
g g2 0.70 0.16 0.06 0.08
g g3 0.66 0.18 0.06 0.10
g g4 0.68 0.15 0.06 0.11
g g5 0.64 0.17 0.06 0.13
h h1 0.74 0.19 0.07
h h2 0.72 0.21 0.07
h h3 0.76 0.18 0.06
h h4 0.70 0.23 0.07
