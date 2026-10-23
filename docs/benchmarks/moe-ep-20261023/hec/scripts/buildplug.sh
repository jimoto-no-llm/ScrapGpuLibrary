#!/bin/bash
# builds the SGLX plugin from ~/moe-ep/hec/tree against the shared llama.cpp 23b0202 build (read only), and the tools
export PATH=$HOME/.local/bin:$PATH
cd $HOME/moe-ep/hec
S=$HOME/moe-ep/src
cmake -S tree/adapters/ggml -B plug -G Ninja -DCMAKE_BUILD_TYPE=Release -DCL11_WITH_CUDA=ON -DCL11_WITH_REMOTE=ON -DLLAMA_CPP_SOURCE_DIR=$S/llama.cpp -DLLAMA_CPP_BUILD_DIR=$S/llama-build > logs/plug-cfg.log 2>&1 || { echo "cfg failed" > logs/plug-build.log; exit 1; }
nice cmake --build plug -j2 > logs/plug-build.log 2>&1
echo "exit $?" >> logs/plug-build.log
L=$S/llama.cpp; B=$S/llama-build/bin
for t in hec_gen; do nice g++ -O2 -std=c++17 tree/adapters/ggml/tools/$t.cpp -I$L/include -I$L/ggml/include -L$B -lllama -lggml -lggml-base -Wl,-rpath,$B -o bin/$t >> logs/plug-build.log 2>&1; done
nice g++ -O2 -std=c++17 tree/adapters/ggml/tools/hec_bench.cpp -I$L/ggml/include -L$B -lggml -lggml-base -lpthread -Wl,-rpath,$B -o bin/hec_bench >> logs/plug-build.log 2>&1
echo "tools done" >> logs/plug-build.log
nice g++ -O1 -std=c++17 tree/adapters/ggml/tools/moe_block_check.cpp -I$L/ggml/include -L$B -lggml -lggml-base -Wl,-rpath,$B -o bin/moe_block_check >> logs/plug-build.log 2>&1
nice g++ -O1 -std=c++17 tree/adapters/ggml/tools/mk_tiny_moe.cpp -I$L/ggml/include -L$B -lggml -lggml-base -Wl,-rpath,$B -o bin/mk_tiny_moe >> logs/plug-build.log 2>&1
echo "tools2 done" >> logs/plug-build.log
