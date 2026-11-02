#!/bin/bash
# push the worktree to tb250 (no rsync there)
cd ~/dev/tools/ScrapGpuLibrary-eg2
COPYFILE_DISABLE=1 tar --no-xattrs --exclude='build*' --exclude='.git' --exclude='_w' --exclude='*.gguf' -cf - . | ssh -o HostName=100.98.122.14 -o HostKeyAlias=tb250.local tb250 'mkdir -p ~/moe-ep/eg2/ScrapGpuLibrary && cd ~/moe-ep/eg2/ScrapGpuLibrary && tar xf - 2>/dev/null'
