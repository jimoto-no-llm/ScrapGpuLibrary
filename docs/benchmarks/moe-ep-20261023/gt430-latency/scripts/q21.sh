# q21 (under perf pin): G with the timestamp trace, tg only
export UB=320 REPS=2 PROFILE=1 REMOTE_BE=opencl11
G="Vulkan0:0.6,CL11_0:0.25,CL11_1:0.05,REMOTE0:0.1"
export SGL_TRACE=$HOME/moe-ep/g430b/logs/q21_G.trace
./bench.sh q21_G "$G" -p 0 -n 128
