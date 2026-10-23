#!/bin/bash
# ggml-cpu only, same options as ~/moe-ep/build-llama.sh but GGML_SSE42=OFF (the shared build has SSE42=ON by default: -msse4.2 on every CPU file)
export PATH=$HOME/.local/bin:$PATH
cmake -S $HOME/moe-ep/src/llama.cpp -B $HOME/moe-ep/hec/build-sse2 -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_BACKEND_DL=ON -DBUILD_SHARED_LIBS=ON \
  -DGGML_NATIVE=OFF -DGGML_AVX=OFF -DGGML_AVX2=OFF -DGGML_AVX512=OFF -DGGML_FMA=OFF -DGGML_F16C=OFF -DGGML_BMI2=OFF -DGGML_SSE42=OFF -DGGML_BLAS=OFF \
  -DGGML_VULKAN=OFF -DGGML_RPC=OFF -DCMAKE_PREFIX_PATH=$HOME/moe-ep/deps -DCMAKE_CXX_FLAGS=-I$HOME/moe-ep/deps/include \
  -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_TOOLS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_CURL=OFF -DLLAMA_OPENSSL=OFF || exit 1
nice cmake --build $HOME/moe-ep/hec/build-sse2 -j2 --target ggml-cpu
