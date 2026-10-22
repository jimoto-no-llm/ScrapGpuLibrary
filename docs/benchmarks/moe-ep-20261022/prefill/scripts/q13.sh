. ./env.sh
sglx_env ""
export GGML_BACKEND_PATH=$PLUG GGML_CL11_PLATFORMS=NVIDIA GGML_CL11_DEVICES="GT 730,GT 710" GGML_CL11_AUTO_RPC=0
for d in CL11_0 CL11_1; do
 ./run.sh q13_mmid_$d $LB/test-backend-ops perf -b $d -o MUL_MAT_ID
done
