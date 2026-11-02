# sourced by the EmbeddingGemma 2 scripts on the TB250 (work directory ~/moe-ep/eg2/)
export PATH=$HOME/.local/bin:$PATH
ROOT=$HOME/moe-ep/eg2
SG=$ROOT/scripts
LB=$ROOT/llama-build/bin                     # llama.cpp 23b0202, own build (llama-embedding, llama-bench, test-backend-ops, llama-quantize)
PLUG=${PLUG:-$ROOT/build-cl11/libggml-cl11.so}
EG2=$ROOT/bin/eg2bench
MODELS=$ROOT/models
TASK=$ROOT/task
LOGS=${LOGS:-$ROOT/logs}
DUMPS=$ROOT/dumps
VKICD=/usr/share/vulkan/icd.d/radeon_icd.json # NVIDIA 390's Vulkan ICD crashes: the loader only sees the AMD cards
RUN=$SG/run.sh
mkdir -p "$LOGS" "$DUMPS"
export EG2_BACKEND_DIR=$LB
export VK_ICD_FILENAMES=$VKICD   # without it the Vulkan backend load crashes ("stack smashing") on the NVIDIA 390 Vulkan ICD
