#!/bin/bash
cd ~/moe-ep/fix/scripts
. ./env.sh
sglx_env "Vulkan0:0.5,Vulkan1:0.2,CL11_0:0.2,CL11_1:0.1"
TMO=900 ./run.sh f4_tbo_C $LB/test-backend-ops test -b SGLX0 -o MUL_MAT,GET_ROWS
