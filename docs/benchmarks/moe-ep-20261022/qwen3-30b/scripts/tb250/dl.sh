#!/bin/bash
cd /mnt/hdd/models
for f in Qwen3-30B-A3B-Q4_0 Qwen3-30B-A3B-Q3_K_M; do
  curl -L -sS --fail -C - -o $f.gguf https://huggingface.co/unsloth/Qwen3-30B-A3B-GGUF/resolve/main/$f.gguf >> ~/moe-ep/q30/logs/dl.log 2>&1
  echo "$f rc=$? $(date)" >> ~/moe-ep/q30/logs/dl.log
done
echo ALLDONE >> ~/moe-ep/q30/logs/dl.log
