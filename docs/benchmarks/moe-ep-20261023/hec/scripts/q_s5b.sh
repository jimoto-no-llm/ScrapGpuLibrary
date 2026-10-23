. $HOME/moe-ep/hec/env_hec.sh
B=$LB
for sl in 12 24 100 1000; do
  $RUN $LOGD s5b_pairs_s$sl $H/bin/hec_bench pairs --cpu-lib $B/libggml-cpu.so --vk-lib $B/libggml-vulkan.so --vk-dev Vulkan0 --tokens 64 --slots $sl --iters 6
done
