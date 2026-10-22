#!/bin/bash
cd "$(dirname "$0")"
. ./env.sh; cfg rx
for c in readme:$TEXT en_legal:corpus/en_legal.txt py_code:corpus/py_code.txt; do
  n=${c%%:*}; f=${c#*:}
  ./run.sh imat_$n $LB/llama-imatrix -m $MODEL -f $f -o imat/$n.gguf --no-ppl -c 512 --chunks 8 -t 2 -b 512 -ub 512 -dev Vulkan0 -ngl 99 -ncmoe 39 -fa off >/dev/null 2>&1
  echo "imat_$n rc=$?"
done
echo Q9DONE
