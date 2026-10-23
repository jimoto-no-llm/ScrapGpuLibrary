. $HOME/moe-ep/hec/env_hec.sh
B=$LB
for n in 128 512; do
  $RUN $LOGD s5_pairs_n$n $H/bin/hec_bench pairs --cpu-lib $B/libggml-cpu.so --vk-lib $B/libggml-vulkan.so --vk-dev Vulkan0 --tokens $n --slots 100 --iters 6
done
