#!/bin/bash
# sync the SGL worktree (without docs, .git, third_party) to tb250:~/moe-ep/hec/tree
R=/Users/eightman/dev/tools/ScrapGpuLibrary-moe-ep
cd $R
COPYFILE_DISABLE=1 tar -cf - --exclude=.git --exclude=third_party --exclude=docs --exclude=build --exclude='._*' CMakeLists.txt cmake include src tests tools examples adapters 2>/dev/null | ssh -o HostName=100.98.122.14 -o HostKeyAlias=tb250.local tb250 'mkdir -p ~/moe-ep/hec/tree && cd ~/moe-ep/hec/tree && tar -xf - 2>/dev/null; find . -name "._*" -delete; echo synced'
