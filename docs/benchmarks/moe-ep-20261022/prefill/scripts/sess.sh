#!/bin/bash
# usage: sess.sh QUEUEFILE   -- runs the queue (a bash script that calls ./bench.sh) under perf pin, always restoring perf auto
cd ~/moe-ep/pp
trap "sudo -n /usr/local/sbin/sgl-driver-switch perf auto >> logs/perf.log 2>&1; echo restored >> logs/perf.log" EXIT
sudo -n /usr/local/sbin/sgl-driver-switch perf pin >> logs/perf.log 2>&1
export PROFILE=0
bash "$1"
echo "QUEUE_DONE $1" >> logs/queue.log
