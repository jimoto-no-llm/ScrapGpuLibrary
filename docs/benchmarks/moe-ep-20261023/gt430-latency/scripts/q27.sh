# q27 (under perf pin): finer split grid for G (with the GT 430) and H (without), also without the GT 710; host on cpu 0, server + worker on cpu 1, tg only
export UB=320 REPS=3 PROFILE=1 REMOTE_BE=opencl11
A="-p 0 -n 128"
export WRAP="taskset -c 0"
g() { SCRAPGPU_SERVER_CPU=1 GGML_SGLX_REMOTE_CPU=1 ./bench.sh q27_G_$1 "$2" $A; }
h() { ./bench.sh q27_H_$1 "$2" $A; }
g g6 "Vulkan0:0.72,CL11_0:0.15,CL11_1:0.06,REMOTE0:0.07"
g g7 "Vulkan0:0.70,CL11_0:0.17,CL11_1:0.05,REMOTE0:0.08"
g g8 "Vulkan0:0.71,CL11_0:0.16,CL11_1:0.05,REMOTE0:0.08"
g g9 "Vulkan0:0.68,CL11_0:0.16,CL11_1:0.06,REMOTE0:0.10"
g g10 "Vulkan0:0.72,CL11_0:0.17,REMOTE0:0.11"
g g11 "Vulkan0:0.74,CL11_0:0.16,REMOTE0:0.10"
h h5 "Vulkan0:0.78,CL11_0:0.17,CL11_1:0.05"
h h6 "Vulkan0:0.77,CL11_0:0.18,CL11_1:0.05"
h h7 "Vulkan0:0.75,CL11_0:0.19,CL11_1:0.06"
h h8 "Vulkan0:0.78,CL11_0:0.16,CL11_1:0.06"
h h10 "Vulkan0:0.78,CL11_0:0.22"
h h11 "Vulkan0:0.80,CL11_0:0.20"
