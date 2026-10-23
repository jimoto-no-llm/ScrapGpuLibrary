# q24 (under perf pin): scheduling variants of G (host thread vs worker / server on 2 cores), tg only
export UB=320 REPS=3 PROFILE=1 REMOTE_BE=opencl11
G="Vulkan0:0.6,CL11_0:0.25,CL11_1:0.05,REMOTE0:0.1"
H="Vulkan0:0.65,CL11_0:0.28,CL11_1:0.07"
A="-p 0 -n 128"
./bench.sh q24_G0 "$G" $A
WRAP="taskset -c 0" SCRAPGPU_SERVER_CPU=1 GGML_SGLX_REMOTE_CPU=1 ./bench.sh q24_G1pin "$G" $A
SCRAPGPU_SERVER_SCHED=batch GGML_SGLX_REMOTE_SCHED=batch ./bench.sh q24_G2batch "$G" $A
GGML_SGLX_THREADED_LAST=1 ./bench.sh q24_G3last "$G" $A
./bench.sh q24_H0 "$H" $A
WRAP="taskset -c 0" ./bench.sh q24_H1host0 "$H" $A
