#!/bin/bash
# test-backend-ops for the ops the EmbeddingGemma 2 graph adds (GELU, NORM) on every device, in the default / kepler / fermi tunings
. "$(dirname "$0")/env.sh"
export VK_ICD_FILENAMES=$VKICD GGML_BACKEND_PATH=$PLUG GGML_CL11_AUTO_RPC=0 RUSTICL_ENABLE=radeonsi
OPS=${OPS:-GELU,NORM,GELU_ERF,GELU_QUICK,UNARY}
dev() { # tag platforms devfilter
  for g in "" kepler fermi; do
    GGML_CL11_PLATFORMS=$2 GGML_CL11_DEVICES="$3" GGML_CL11_FORCE_GEN=$g $RUN tbo_$1_${g:-def} $LB/test-backend-ops test -b CL11_0 -o $OPS
  done
}
dev gt430 NVIDIA "GT 430"
dev gt730 NVIDIA "GT 730"
dev gt710 NVIDIA "GT 710"
for g in ""; do
  GGML_CL11_PLATFORMS=rusticl GGML_CL11_DEVICES="RX 6400" $RUN tbo_rx6400_def $LB/test-backend-ops test -b CL11_0 -o $OPS
  GGML_CL11_PLATFORMS=Clover GGML_CL11_DEVICES="WX 2100" $RUN tbo_wx2100clover_def $LB/test-backend-ops test -b CL11_0 -o $OPS
  GGML_CL11_PLATFORMS=rusticl GGML_CL11_DEVICES="WX 2100" $RUN tbo_wx2100rusticl_def $LB/test-backend-ops test -b CL11_0 -o $OPS
done
