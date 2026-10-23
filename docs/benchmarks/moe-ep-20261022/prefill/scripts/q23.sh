export REPS=1
C="Vulkan0:0.5,Vulkan1:0.2,CL11_0:0.2,CL11_1:0.1"
UB=256 ./bench.sh q23_C "$C" -p 128 -n 0 -v
UB=256 ./bench.sh q23_noop noop -p 128 -n 0 -v
UB=256 ./bench.sh q23_c1 "" -p 128 -n 0 -v
