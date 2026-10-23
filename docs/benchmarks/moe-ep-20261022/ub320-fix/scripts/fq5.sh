. ./env.sh
Cs="Vulkan0:0.5,Vulkan1:0.2,CL11_0:0.2,CL11_1:0.1"
S0="Vulkan1:0.5,CL11_0:0.3,CL11_1:0.2"
S1="Vulkan0:0.1,Vulkan1:0.4,CL11_0:0.3,CL11_1:0.2"
DN="blk\.21\.ffn_down_exps\.weight=Vulkan1"
UPDN="blk\.21\.ffn_down_exps\.weight=Vulkan1;blk\.21\.ffn_up_exps\.weight=Vulkan1"
memrun() { # TAG SPLIT TS OTX
  python3 peak.py ../logs/$1.peak & local PK=$!
  OTX="$4" GGML_VK_MEMORY_LOGGER=1 TS=$3 UB=${UB:-320} REPS=3 ./bench.sh $1 "$2"
  kill $PK; wait $PK 2>/dev/null
}
memrun fD_ref_ts51 "$Cs" 5/1 ""
memrun fD_S0_dn "$S0" 6/1 "$DN"
memrun fD_S1_dn "$S1" 6/1 "$DN"
memrun fD_C_updn "$Cs" 6/1 "$UPDN"
UB=256 memrun fD_S0_dn_ub256 "$S0" 6/1 "$DN"
