. $HOME/moe-ep/hec/env_hec.sh
. $HOME/moe-ep/hec/cfg_n.sh
export HEC_GEN_VERBOSE=1
# GT 430 (Fermi, served by a ggml-rpc-server child process: RPC0) as a medium device, ungated and gated; devices: CL11_0 = GT 730, CL11_1 = GT 710, RPC0 = GT 430
g430() { ( n_set_env; export GGML_CL11_DEVICES=; unset GGML_CL11_DEVICES GGML_CL11_AUTO_RPC; export GGML_SGLX_MOE=$1; $RUN "${@:2}" ); }
g430 Vulkan0:auto:250 $LOGD g430_none $GEN -m $MODEL -ub 512 "${N_ARGS[@]}" -r 2 --bench pp256,tg128
g430 Vulkan0:auto:250,RPC0:300 $LOGD g430_rpc0 $GEN -m $MODEL -ub 512 "${N_ARGS[@]}" -r 2 --bench pp256,tg128
g430 'Vulkan0:auto:250,RPC0:300!' $LOGD g430_rpc0_gated $GEN -m $MODEL -ub 512 "${N_ARGS[@]}" -r 2 --bench pp256,tg128
