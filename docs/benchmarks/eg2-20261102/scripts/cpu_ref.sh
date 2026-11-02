#!/bin/bash
# CPU reference dumps on the TB250 (2 threads, SSE4.2 build) of the correctness set, for every weight type
. "$(dirname "$0")/env.sh"
unset GGML_BACKEND_PATH
for t in ${TYPES:-bf16 F16 Q8_0 Q4_0 Q4_K_M}; do
  [ -f $DUMPS/cpu_${t}_corr.bin ] && continue
  TMO=7200 $RUN cpu_${t}_corr $EG2 -m $MODELS/eg2-$t.gguf -ngl 0 -t 2 --max-tokens 1500 --dump $TASK/texts_corr.txt $DUMPS/cpu_${t}_corr.bin
done
