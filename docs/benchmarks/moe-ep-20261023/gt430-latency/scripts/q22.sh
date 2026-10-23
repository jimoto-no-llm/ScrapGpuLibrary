# q22 (under perf pin): G / H with the timestamp trace (device completion times), and G serial (each device alone)
export UB=320 REPS=2 PROFILE=1 REMOTE_BE=opencl11
G="Vulkan0:0.6,CL11_0:0.25,CL11_1:0.05,REMOTE0:0.1"
H="Vulkan0:0.65,CL11_0:0.28,CL11_1:0.07"
export SGL_TRACE=$HOME/moe-ep/g430b/logs/q22_G.trace
./bench.sh q22_G "$G" -p 0 -n 128
export SGL_TRACE=$HOME/moe-ep/g430b/logs/q22_H.trace
./bench.sh q22_H "$H" -p 0 -n 128
unset SGL_TRACE
SERIAL=1 ./bench.sh q22_Gserial "$G" -p 0 -n 128
SERIAL=1 ./bench.sh q22_Hserial "$H" -p 0 -n 128
