#!/bin/bash
cd $HOME/moe-ep/hec
B=$HOME/moe-ep/src/llama-build/bin; M=$HOME/moe-ep/models/Qwen3-30B-A3B-Q4_0.gguf
R="bash src/hec_run.sh logs"
TMO=3000 $R build_sse2 bash build_sse2.sh
TMO=3000 $R trace_bench bin/hec_trace -m $M --cpu-lib $B/libggml-cpu.so --out trace/bench --bench 512,128,512:128 --reps 2
for c in en_doc py_code ja_tech en_legal ja_design cpp_code; do
  TMO=3000 $R trace_$c bin/hec_trace -m $M --cpu-lib $B/libggml-cpu.so --out trace/$c --text corpus/$c.txt --n-prompt 512 --n-gen 256 --seed 1234
done
echo QUEUE1 DONE > logs/queue1.done
