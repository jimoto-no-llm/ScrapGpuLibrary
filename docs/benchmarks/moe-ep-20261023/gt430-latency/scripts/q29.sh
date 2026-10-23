# q29 (under perf pin): C1 / G / H alternating, 3 rounds, REPS=10, -ub 320, PROFILE=0.
#   G = V0 .70 C0 .17 C1 .05 R0 .08 ; H = V0 .78 C0 .17 C1 .05 (G without the GT 430, its rows to Vulkan0)
#   "pin": host (taskset -c 0), server + REMOTE0 worker on cpu 1.  "batch": no taskset, server + worker SCHED_BATCH.  "plain": nothing.
export UB=320 REPS=10 PROFILE=0 REMOTE_BE=opencl11
G="Vulkan0:0.70,CL11_0:0.17,CL11_1:0.05,REMOTE0:0.08"
H="Vulkan0:0.78,CL11_0:0.17,CL11_1:0.05"
for i in 1 2 3; do
  ./bench.sh q29_c1_$i ""
  WRAP="taskset -c 0" SCRAPGPU_SERVER_CPU=1 GGML_SGLX_REMOTE_CPU=1 ./bench.sh q29_Gpin_$i "$G"
  WRAP="taskset -c 0" ./bench.sh q29_Hpin_$i "$H"
  SCRAPGPU_SERVER_SCHED=batch GGML_SGLX_REMOTE_SCHED=batch ./bench.sh q29_Gbatch_$i "$G"
  ./bench.sh q29_Hplain_$i "$H"
done
