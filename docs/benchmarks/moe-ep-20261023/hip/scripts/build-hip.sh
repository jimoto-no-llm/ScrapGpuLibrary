#!/bin/bash
# usage: [BDIR=llama-build-mmq HIP_EXTRA=-DGGML_CUDA_FORCE_MMQ=ON] build-hip.sh
# llama.cpp (the source of the other measurements, 23b0202) with ggml-hip, ggml-vulkan and ggml-cpu as loadable backends, in ~/moe-ep/hip/${BDIR:-llama-build}.
# CPU flags as in ~/moe-ep/build-llama.sh (Celeron G3930: no AVX). ROCm 7.13.0 from env.sh.
. "$HOME/moe-ep/hip/env.sh"
export PATH=$HOME/.local/bin:$PATH
export HIPCXX=$ROCM_PATH/lib/llvm/bin/clang
cd $HOME/moe-ep/src
cmake -S llama.cpp -B $HOME/moe-ep/hip/${BDIR:-llama-build} -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_BACKEND_DL=ON -DBUILD_SHARED_LIBS=ON \
  -DGGML_NATIVE=OFF -DGGML_AVX=OFF -DGGML_AVX2=OFF -DGGML_AVX512=OFF -DGGML_FMA=OFF -DGGML_F16C=OFF -DGGML_BMI2=OFF -DGGML_BLAS=OFF \
  -DGGML_HIP=ON -DGPU_TARGETS=gfx1034 -DAMDGPU_TARGETS=gfx1034 -DGGML_CUDA_FA=${HIP_FA:-OFF} ${HIP_EXTRA} -DCMAKE_PREFIX_PATH="$ROCM_PATH;$HOME/moe-ep/deps" \
  -DCMAKE_C_COMPILER=$ROCM_PATH/lib/llvm/bin/clang -DCMAKE_CXX_COMPILER=$ROCM_PATH/lib/llvm/bin/clang++ \
  -DGGML_VULKAN=ON -DCMAKE_CXX_FLAGS=-I$HOME/moe-ep/deps/include -DVulkan_LIBRARY=/usr/lib/x86_64-linux-gnu/libvulkan.so.1 -DVulkan_INCLUDE_DIR=$HOME/moe-ep/src/Vulkan-Headers/include \
  -DGGML_RPC=ON -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_SERVER=OFF -DLLAMA_CURL=OFF -DLLAMA_OPENSSL=OFF > $HOME/moe-ep/hip/logs/${BDIR:-llama-build}-cfg.log 2>&1 || { echo cfg failed > $HOME/moe-ep/hip/logs/${BDIR:-llama-build}.log; exit 1; }
nice cmake --build $HOME/moe-ep/hip/${BDIR:-llama-build} -j2 --target ggml-cpu ggml-hip ggml-vulkan ggml-rpc llama-bench llama-completion llama-perplexity test-backend-ops > $HOME/moe-ep/hip/logs/${BDIR:-llama-build}.log 2>&1
echo "exit $?" >> $HOME/moe-ep/hip/logs/${BDIR:-llama-build}.log
