#!/bin/bash
export PATH=$HOME/.local/bin:$PATH
cd $HOME/moe-ep/eg2
cmake -S llama.cpp -B llama-build -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_BACKEND_DL=ON -DBUILD_SHARED_LIBS=ON \
  -DGGML_NATIVE=OFF -DGGML_AVX=OFF -DGGML_AVX2=OFF -DGGML_AVX512=OFF -DGGML_FMA=OFF -DGGML_F16C=OFF -DGGML_BMI2=OFF -DGGML_BLAS=OFF \
  -DGGML_VULKAN=ON -DGGML_RPC=ON -DCMAKE_CXX_FLAGS=-I$HOME/moe-ep/deps/include -DCMAKE_PREFIX_PATH=$HOME/moe-ep/deps -DVulkan_LIBRARY=/usr/lib/x86_64-linux-gnu/libvulkan.so.1 -DVulkan_INCLUDE_DIR=$HOME/moe-ep/src/Vulkan-Headers/include \
  -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_SERVER=OFF -DLLAMA_CURL=OFF -DLLAMA_OPENSSL=OFF > logs/llama-cfg.log 2>&1 || { echo cfg failed > logs/llama-build.log; exit 1; }
nice cmake --build llama-build -j2 --target ggml-cpu ggml-vulkan ggml-rpc ggml-rpc-server llama-embedding llama-bench llama-quantize test-backend-ops > logs/llama-build.log 2>&1
echo "exit $?" >> logs/llama-build.log
