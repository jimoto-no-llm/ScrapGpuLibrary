#!/bin/sh
# Writes the plain PTX text of kernels.c to a file, for SCRAPGPU_CUDA_PTX=<file> (kernel experiments without rebuilding the library):
#   gen_ptx_to.sh sm_20|sm_35 OUT.ptx [extra clang flags, e.g. -DEXPERIMENT]  [source file, default kernels.c via KSRC=...]
set -e
here=$(cd "$(dirname "$0")" && pwd)
CC=${CLANG:-}
if [ -z "$CC" ]; then
    if [ -x /opt/homebrew/opt/llvm/bin/clang ]; then CC=/opt/homebrew/opt/llvm/bin/clang; else CC=clang; fi
fi
arch=$1; out=$2; shift 2
extra=""
[ "$arch" = sm_35 ] && extra="-DKEPLER"
"$CC" --target=nvptx64-nvidia-cuda -march="$arch" -O2 -S -x c -std=gnu11 -ffreestanding -nostdlib -fno-math-errno $extra "$@" \
    -Xclang -target-feature -Xclang +ptx32 "${KSRC:-$here/kernels.c}" -o "$out"
sed -i.bak -e 's/^\.version .*/.version 3.2/' "$out" && rm -f "$out.bak"
