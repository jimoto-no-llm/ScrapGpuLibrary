# q30 (no perf pin): the tbo failures with the split without the GT 430 (H): are they Vulkan0 only?
cd ~/moe-ep/g430b/scripts
. ./env.sh
sglx_env "Vulkan0:0.78,CL11_0:0.17,CL11_1:0.05"
TMO=1500 ./run.sh q30_tbo_H $LB/test-backend-ops test -b SGLX0 -o MUL_MAT,GET_ROWS
sglx_env "Vulkan0:1"
TMO=1500 ./run.sh q30_tbo_V0 $LB/test-backend-ops test -b SGLX0 -o MUL_MAT,GET_ROWS
echo Q30_DONE >> ../logs/queue.log
