#!/bin/bash
cd "$(dirname "$0")"
until grep -q Q18DONE logs/queue18.out; do sleep 10; done
. ./env.sh; cfg rx
# held-out check: the first 4 chunks and the next 4 chunks of the same corpus (512 tokens each)
for c in readme:$TEXT py_code:corpus/py_code.txt; do
  n=${c%%:*}; f=${c#*:}
  ./run.sh imat_${n}_h1 $LB/llama-imatrix -m $MODEL -f $f -o imat/${n}_h1.gguf --no-ppl -c 512 --chunks 4 -t 2 -b 512 -ub 512 -dev Vulkan0 -ngl 99 -ncmoe 39 -fa off >/dev/null 2>&1; echo "imat_${n}_h1 rc=$?"
  ./run.sh imat_${n}_h2 $LB/llama-imatrix -m $MODEL -f $f -o imat/${n}_h2.gguf --no-ppl -c 512 --chunk 4 --chunks 4 -t 2 -b 512 -ub 512 -dev Vulkan0 -ngl 99 -ncmoe 39 -fa off >/dev/null 2>&1; echo "imat_${n}_h2 rc=$?"
done
echo Q19DONE
