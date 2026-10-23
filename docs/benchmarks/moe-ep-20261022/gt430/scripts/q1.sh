#!/bin/bash
# q1: functional test of REMOTE0 (GT 430) through test-backend-ops, CL11 server and CUDA server
cd ~/moe-ep/g430/scripts
. ./env.sh
for be in opencl11 cuda; do
  REMOTE_BE=$be; export REMOTE_BE
  sglx_env "REMOTE0:1"
  TMO=1500 ./run.sh q1_tbo_remote0_$be $LB/test-backend-ops test -b SGLX0 -o MUL_MAT,GET_ROWS
done
echo Q1_DONE >> ../logs/queue.log
