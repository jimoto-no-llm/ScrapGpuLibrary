#!/bin/bash
# The correctness checks that need no tb250 (Mac, ggml-cpu + Metal + the CL11 devices of the Apple GPU), as run for the commits of this stage.
#   usage: LLAMA=<llama.cpp 23b0202 source> BUILD=<its build with -DGGML_BACKEND_DL=ON -DBUILD_SHARED_LIBS=ON>/bin PLUG=<libggml-cl11.so> TOOLS=<dir for the tool binaries> mac_check.sh
set -e
R=$(cd "$(dirname "$0")/../../../../.." && pwd)
: "${LLAMA:?}" "${BUILD:?}" "${PLUG:?}" "${TOOLS:?}"
mkdir -p "$TOOLS"
T=$R/adapters/ggml/tools
c++ -std=c++17 -O1 -I$LLAMA/ggml/include $T/moe_block_check.cpp -L$BUILD -lggml -lggml-base -Wl,-rpath,$BUILD -o $TOOLS/moe_block_check
c++ -std=c++17 -O1 -I$LLAMA/ggml/include $T/mk_tiny_moe.cpp -L$BUILD -lggml -lggml-base -Wl,-rpath,$BUILD -o $TOOLS/mk_tiny_moe
c++ -std=c++17 -O2 -I$LLAMA/include -I$LLAMA/ggml/include $T/hec_gen.cpp -L$BUILD -lllama -lggml -lggml-base -Wl,-rpath,$BUILD -o $TOOLS/hec_gen
export GGML_LIB_DIR=$BUILD HEC_BACKEND_DIR=$BUILD GGML_BACKEND_PATH=$PLUG

echo "== 1. test-backend-ops: MUL_MAT_ID with no slot device (everything on ggml-cpu), MUL_MAT and GET_ROWS of the SGLX split"
GGML_SGLX_MOE=1 $BUILD/test-backend-ops test -b SGLX0 -o MUL_MAT_ID | grep -E "tests passed|FAIL"
GGML_SGLX_SPLIT=CPU:0.5,MTL0:0.5 GGML_SGLX_MOE=1 $BUILD/test-backend-ops test -b SGLX0 -o MUL_MAT,GET_ROWS | grep -E "tests passed|FAIL"

echo "== 2. the MoE block of llama.cpp against ggml-cpu through ggml_backend_sched: cold (NMSE 0), Metal slots (rounding of Metal), 30B shapes with the Q4_1 class"
GGML_SGLX_MOE=1 $TOOLS/moe_block_check --layers 3 2>&1 | tail -1
GGML_SGLX_MOE=MTL0:20 GGML_SGLX_MOE_RING=24 $TOOLS/moe_block_check --layers 1 --tol 2e-3 2>&1 | tail -1
GGML_SGLX_MOE=MTL0:100:20 GGML_SGLX_MOE_RING=24 $TOOLS/moe_block_check --big --q41 --tol 3e-3 2>&1 | tail -1

echo "== 3. llama.cpp end to end on a tiny random qwen3moe (-ot ...=SGLX0): logits against the same model with the experts on ggml-cpu"
$TOOLS/mk_tiny_moe $TOOLS/tiny.gguf --layers 6 2>&1 | tail -1
OT='blk\.[0-9]+\.ffn_(up|gate|down)_exps\.weight=SGLX0'
$TOOLS/hec_gen -m $TOOLS/tiny.gguf -dev MTL0 -ngl 99 -ncmoe 6 --dump $TOOLS/ref.bin --n-tokens 60 --n-prefill 24 2>/dev/null
for spec in 1 MTL0:60:10 MTL0:60:10,CL11_0:40 'MTL0:60:10,CL11_0:40!'; do
  GGML_SGLX_MOE_RING=24 GGML_SGLX_MOE=$spec $TOOLS/hec_gen -m $TOOLS/tiny.gguf -dev MTL0/SGLX0 -ts 1/0 -ngl 99 -ot "$OT" --dump $TOOLS/out.bin --n-tokens 60 --n-prefill 24 2>/dev/null
  echo "GGML_SGLX_MOE=$spec: $($TOOLS/hec_gen --compare $TOOLS/ref.bin $TOOLS/out.bin)"
done
