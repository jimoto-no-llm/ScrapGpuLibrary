# q25 (under perf pin): GGML_SGLX_AUTOTUNE from G and from H (3 rounds), scheduling fixes on, tg only
export UB=320 REPS=3 PROFILE=1 REMOTE_BE=opencl11 AUTOTUNE=3,16
G="Vulkan0:0.6,CL11_0:0.25,CL11_1:0.05,REMOTE0:0.1"
H="Vulkan0:0.65,CL11_0:0.28,CL11_1:0.07"
A="-p 0 -n 256"
WRAP="taskset -c 0" SCRAPGPU_SERVER_CPU=1 GGML_SGLX_REMOTE_CPU=1 ./bench.sh q25_Gauto "$G" $A
WRAP="taskset -c 0" ./bench.sh q25_Hauto "$H" $A
