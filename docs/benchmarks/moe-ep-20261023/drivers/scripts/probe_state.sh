#!/bin/bash
# probe_state.sh LABEL : read-only inventory of what the current driver state exposes. Every command has a timeout; no model is loaded.
# Output: $OUT/LABEL/*.txt. Run once right after each switch, before any bench.
. "$(dirname "$0")/env_drv.sh"
L=${1:?label, e.g. nouveau_all}; D=$OUT/$L; mkdir -p "$D"
t() { local f=$1; shift; { echo "# $*"; timeout 90 "$@" 2>&1; echo "# rc=$?"; } > "$D/$f.txt"; }
t status        sudo -n $SW status
t lspci         bash -c 'lspci -nnk | grep -A3 -E "VGA|3D"'
t dri-nodes     bash -c 'for d in 0000:04:00.0 0000:07:00.0 0000:08:00.0 0000:03:00.0 0000:06:00.0; do echo $d $(ls /sys/bus/pci/devices/$d/drm 2>/dev/null | tr "\n" " "); done'
t lsmod         bash -c 'lsmod | grep -E "nouveau|nvidia|drm_gpuvm|ttm"'
t kernel-log    bash -c 'dmesg 2>/dev/null | grep -iE "nouveau|nvidia|NVRM" | tail -80 || journalctl -k --no-pager -n 400 2>/dev/null | grep -iE "nouveau|nvidia|NVRM" | tail -80'
t nouveau-parm  bash -c 'grep . /sys/module/nouveau/parameters/{modeset,config,debug,runpm,noaccel} 2>&1'
t libs          bash -c 'ls -l /usr/lib/x86_64-linux-gnu/libcuda.so* /usr/lib/x86_64-linux-gnu/libnvidia-opencl.so* /etc/OpenCL/vendors/ /etc/vulkan/icd.d/ /usr/share/vulkan/icd.d/'
# OpenCL: the three ICDs separately (nvidia.icd, Clover = mesa.icd, rusticl.icd)
t cl-nvidia     env OCL_ICD_VENDORS=/dev/null OCL_ICD_FILENAMES=libnvidia-opencl.so.1 clinfo -l
t cl-clover     env OCL_ICD_VENDORS=/dev/null OCL_ICD_FILENAMES=libMesaOpenCL.so.1 clinfo -l
t cl-rusticl    env OCL_ICD_VENDORS=/dev/null OCL_ICD_FILENAMES=libRusticlOpenCL.so.1 RUSTICL_ENABLE=nouveau clinfo -l
t cl-rusticl-full env OCL_ICD_VENDORS=/dev/null OCL_ICD_FILENAMES=libRusticlOpenCL.so.1 RUSTICL_ENABLE=nouveau clinfo
# Vulkan: each ICD alone. NVK on Mesa 25.0.7 needs the env var; without it a Kepler card is expected to be absent.
t vk-nvk-plain  env VK_ICD_FILENAMES=$ICD_NVK vulkaninfo --summary
t vk-nvk-broken env VK_ICD_FILENAMES=$ICD_NVK NVK_I_WANT_A_BROKEN_VULKAN_DRIVER=1 vulkaninfo --summary
t vk-nvidia     env VK_ICD_FILENAMES=$ICD_NV vulkaninfo --summary
t vk-amd        env VK_ICD_FILENAMES=$ICD_AMD vulkaninfo --summary
t nvidia-smi    nvidia-smi
t nvidia-smi-q  nvidia-smi -q -d CLOCK,PERFORMANCE,MEMORY
t pstate        sudo -n $SW pstate show     # fails unless sudo allows it (the user can run it by hand; see DRIVER_STAGES.md)
grep -H . "$D"/cl-*.txt "$D"/vk-*.txt 2>/dev/null | grep -iE "platform name|device name|deviceName|apiVersion|Device Version|driverName|rc=" | sed "s#$D/##" > "$D/SUMMARY.txt"
echo "wrote $D (see SUMMARY.txt)"
