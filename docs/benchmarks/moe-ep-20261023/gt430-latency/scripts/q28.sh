# q28 (no perf pin): correctness of the traced / scheduled build: test-backend-ops (REMOTE0 alone, G), KLD vs CPU and self-consistency (G vs one device), -ub 320
cd ~/moe-ep/g430b/scripts
. ./env.sh
export REMOTE_BE=opencl11
G="Vulkan0:0.70,CL11_0:0.17,CL11_1:0.05,REMOTE0:0.08"
H="Vulkan0:0.78,CL11_0:0.17,CL11_1:0.05"
sglx_env "REMOTE0:1"
TMO=1500 ./run.sh q28_tbo_remote0 $LB/test-backend-ops test -b SGLX0 -o MUL_MAT,GET_ROWS
sglx_env "$G"
TMO=1500 ./run.sh q28_tbo_G $LB/test-backend-ops test -b SGLX0 -o MUL_MAT,GET_ROWS
export KB=512 KUB=320
./kld.sh q28_G_cpu "$G" cpu
./kld.sh q28_H_cpu "$H" cpu
SCRAPGPU_SERVER_CPU=1 GGML_SGLX_REMOTE_CPU=1 GGML_SGLX_THREADED_LAST=1 ./kld.sh q28_Gsched_cpu "$G" cpu
./kld.sh q28_one320 "Vulkan0:1" - save
./kld.sh q28_G_self "$G" $SG/kl/q28_one320.kld
echo Q28_DONE >> ../logs/queue.log
