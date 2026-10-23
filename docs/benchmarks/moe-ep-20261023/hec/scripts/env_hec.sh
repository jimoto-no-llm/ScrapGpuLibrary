# sourced by the hec queues on tb250
export PATH=$HOME/.local/bin:$PATH
H=$HOME/moe-ep/hec
LB=$HOME/moe-ep/src/llama-build/bin
MODEL=$HOME/moe-ep/models/Qwen3-30B-A3B-Q4_0.gguf
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/radeon_icd.json
RUN=$H/src/hec_run.sh
LOGD=${LOGD:-$H/run}
GEN=$H/bin/hec_gen
mkdir -p $LOGD
export HEC_BACKEND_DIR=$LB
