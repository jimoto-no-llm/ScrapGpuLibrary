#!/bin/sh
export PATH=$HOME/.local/bin:$PATH
G=$HOME/moe-ep/g430; S=$HOME/moe-ep/src
cd $G
cmake -S tree -B srv -G Ninja -DCMAKE_BUILD_TYPE=Release -DSCRAPGPU_BUILD_OPENCL11=ON -DSCRAPGPU_BUILD_CUDA=ON -DSCRAPGPU_BUILD_REMOTE=ON -DSCRAPGPU_BUILD_TESTS=OFF -DSCRAPGPU_BUILD_EXAMPLES=OFF > logs/srv-cfg.log 2>&1 || { echo cfg-srv-failed > logs/build.status; exit 1; }
nice -n 19 cmake --build srv --target scrapgpu-device-server -j2 > logs/srv-build.log 2>&1 || { echo build-srv-failed > logs/build.status; exit 1; }
cmake -S tree/adapters/ggml -B plug -G Ninja -DCMAKE_BUILD_TYPE=Release -DCL11_WITH_CUDA=ON -DCL11_WITH_REMOTE=ON -DLLAMA_CPP_SOURCE_DIR=$S/llama.cpp -DLLAMA_CPP_BUILD_DIR=$S/llama-build > logs/plug-cfg.log 2>&1 || { echo cfg-plug-failed > logs/build.status; exit 1; }
nice -n 19 cmake --build plug -j2 > logs/plug-build.log 2>&1 || { echo build-plug-failed > logs/build.status; exit 1; }
echo ok > logs/build.status
