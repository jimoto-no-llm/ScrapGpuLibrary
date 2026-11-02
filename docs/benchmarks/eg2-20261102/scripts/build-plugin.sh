#!/bin/bash
export PATH=$HOME/.local/bin:$PATH
cd $HOME/moe-ep/eg2
cmake -S ScrapGpuLibrary/adapters/ggml -B build-cl11 -G Ninja -DCMAKE_BUILD_TYPE=Release -DLLAMA_CPP_SOURCE_DIR=$PWD/llama.cpp -DLLAMA_CPP_BUILD_DIR=$PWD/llama-build "$@" > logs/plugin-cfg.log 2>&1 || { echo cfg failed > logs/plugin-build.log; exit 1; }
nice cmake --build build-cl11 -j2 > logs/plugin-build.log 2>&1
echo "exit $?" >> logs/plugin-build.log
