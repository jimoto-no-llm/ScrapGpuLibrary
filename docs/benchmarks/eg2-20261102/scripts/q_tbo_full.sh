# full test-backend-ops of the CL11 plugin on the devices (after the GEGLU / GELU / NORM change); default tuning, GT 730 also kepler
. "$HOME/moe-ep/eg2/scripts/env.sh"
export VK_ICD_FILENAMES=$VKICD GGML_BACKEND_PATH=$PLUG GGML_CL11_AUTO_RPC=0 RUSTICL_ENABLE=radeonsi
GGML_CL11_PLATFORMS=NVIDIA GGML_CL11_DEVICES="GT 730" TMO=5400 $RUN tbofull_gt730_def $LB/test-backend-ops test -b CL11_0
GGML_CL11_PLATFORMS=NVIDIA GGML_CL11_DEVICES="GT 430" TMO=5400 $RUN tbofull_gt430_def $LB/test-backend-ops test -b CL11_0
GGML_CL11_PLATFORMS=rusticl GGML_CL11_DEVICES="RX 6400" TMO=5400 $RUN tbofull_rx6400_def $LB/test-backend-ops test -b CL11_0
echo QUEUE_DONE
