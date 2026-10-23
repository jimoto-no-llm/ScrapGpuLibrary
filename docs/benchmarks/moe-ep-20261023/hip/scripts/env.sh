# sourced: ROCm 7.13.0 (TheRock, gfx103X-all wheels) in /mnt/hdd/rocm/v713, merged tree = _rocm_sdk_devel
export ROCM_PATH=/mnt/hdd/rocm/v713/lib/python3.13/site-packages/_rocm_sdk_devel
export HIP_PATH=$ROCM_PATH
export PATH=$ROCM_PATH/bin:$PATH
export LD_LIBRARY_PATH=$ROCM_PATH/lib:$ROCM_PATH/lib/llvm/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export HIP_DEVICE_LIB_PATH=$ROCM_PATH/lib/llvm/amdgcn/bitcode
