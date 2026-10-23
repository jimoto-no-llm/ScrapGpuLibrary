#!/bin/bash
# usage: build_variant.sh NAME [extra cmake args...]   -> ~/moe-ep/hec/build-NAME/bin/libggml-cpu.so (ggml-cpu only; options of ~/moe-ep/build-llama.sh)
export PATH=$HOME/.local/bin:$PATH
N=$1; shift
cmake -S $HOME/moe-ep/src/llama.cpp -B $HOME/moe-ep/hec/build-$N -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_BACKEND_DL=ON -DBUILD_SHARED_LIBS=ON \
  -DGGML_NATIVE=OFF -DGGML_AVX=OFF -DGGML_AVX2=OFF -DGGML_AVX512=OFF -DGGML_FMA=OFF -DGGML_F16C=OFF -DGGML_BMI2=OFF -DGGML_BLAS=OFF \
  -DGGML_VULKAN=OFF -DGGML_RPC=OFF -DCMAKE_PREFIX_PATH=$HOME/moe-ep/deps \
  -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_TOOLS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_CURL=OFF -DLLAMA_OPENSSL=OFF "$@" || exit 1
nice cmake --build $HOME/moe-ep/hec/build-$N -j2 --target ggml-cpu
