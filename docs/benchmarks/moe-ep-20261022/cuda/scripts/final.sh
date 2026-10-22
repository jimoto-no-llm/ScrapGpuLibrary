#!/bin/sh
R=$HOME/moe-ep/sglx/run.sh
D=$HOME/moe-ep/cuda/bB
for card in "GT 730" "GT 710" "GT 430"; do
  c=$(echo "$card" | tr -d ' ')
  $R cuda_final_ktest_$c env SCRAPGPU_CUDA_DEVICES="$card" $D/scrapgpu_cuda_kernel_tests
  $R cuda_final_ops_$c env SCRAPGPU_TEST_BACKEND=cuda SCRAPGPU_CUDA_DEVICES="$card" $D/scrapgpu_ops_tests
done
sed -i "s#cuda/b/#cuda/bB/#" $HOME/moe-ep/cuda/bw2.sh
TAGP=cuda_final ../cuda/bw2.sh
