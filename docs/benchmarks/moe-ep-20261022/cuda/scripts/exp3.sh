#!/bin/sh
R=$HOME/moe-ep/sglx/run.sh
B=$HOME/moe-ep/cuda/bB/scrapgpu_mm_bench
for nrr in 1 2 4; do
  $R cuda_e3_nrr${nrr}_q6K env SCRAPGPU_CUDA_NRR=$nrr SCRAPGPU_CUDA_PTX=$HOME/moe-ep/cuda/k35c.ptx SCRAPGPU_CUDA_DEVICES="GT 730" $B cuda mv --type q6_K --iters 10 --reps 5
  $R cuda_e3_nrr${nrr}_q4_0 env SCRAPGPU_CUDA_NRR=$nrr SCRAPGPU_CUDA_PTX=$HOME/moe-ep/cuda/k35c.ptx SCRAPGPU_CUDA_DEVICES="GT 730" $B cuda mv --type q4_0 --rows 65536 --iters 10 --reps 5
done
