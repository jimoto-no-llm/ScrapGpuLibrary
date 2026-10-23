#!/bin/bash
cd ~/moe-ep/drivers/scripts
./bench_cuda390.sh
./bench_cl11_390.sh
./q_n_drv.sh drv390
echo CHAIN_DONE
