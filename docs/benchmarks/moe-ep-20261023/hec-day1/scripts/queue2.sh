#!/bin/bash
cd $HOME/moe-ep/hec
B=$HOME/moe-ep/src/llama-build/bin; M=$HOME/moe-ep/models/Qwen3-30B-A3B-Q4_0.gguf
D=$B/libggml-cpu.so; S=$HOME/moe-ep/hec/build-sse2/bin/libggml-cpu.so
O=$HOME/moe-ep/hec/build-omp0/bin/libggml-cpu.so; U=$HOME/moe-ep/hec/build-tune/bin/libggml-cpu.so
R="bash src/hec_run.sh logs"
TMO=900 $R cpu1_check bin/hec_bench cpu1 --model $M --cpu-lib $D --threads 2 --iters 40 --other
TMO=900 $R build_omp0 bash build_variant.sh omp0 -DGGML_OPENMP=OFF
TMO=900 $R build_tune bash build_variant.sh tune -DCMAKE_C_FLAGS=-mtune=skylake -DCMAKE_CXX_FLAGS=-mtune=skylake
for rep in a b; do
  for v in default sse2 omp0 tune; do
    case $v in default) L=$D;; sse2) L=$S;; omp0) L=$O;; tune) L=$U;; esac
    TMO=900 $R cpu1_${v}_t2_$rep bin/hec_bench cpu1 --model $M --cpu-lib $L --threads 2 --iters 300 --other
  done
done
for v in default sse2; do
  case $v in default) L=$D;; sse2) L=$S;; esac
  TMO=900 $R cpu1_${v}_t1 bin/hec_bench cpu1 --model $M --cpu-lib $L --threads 1 --iters 200
done
echo QUEUE2 DONE > logs/queue2.done
