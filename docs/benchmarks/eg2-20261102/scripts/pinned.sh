#!/bin/bash
# usage: pinned.sh QUEUEFILE  -- runs the queue under perf pin, always restoring perf auto
trap "sudo -n /usr/local/sbin/sgl-driver-switch perf auto >> $HOME/moe-ep/eg2/logs/perf.log 2>&1; echo restored >> $HOME/moe-ep/eg2/logs/perf.log" EXIT
sudo -n /usr/local/sbin/sgl-driver-switch perf pin >> $HOME/moe-ep/eg2/logs/perf.log 2>&1
bash "$1"
