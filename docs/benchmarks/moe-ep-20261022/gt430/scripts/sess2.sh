#!/bin/bash
# usage: sess2.sh QUEUEFILE   -- runs the queue under perf pin (PROFILE as the queue sets it), always restoring perf auto
cd ~/moe-ep/g430/scripts
trap "sudo -n /usr/local/sbin/sgl-driver-switch perf auto >> ../logs/perf.log 2>&1; echo restored >> ../logs/perf.log" EXIT
sudo -n /usr/local/sbin/sgl-driver-switch perf pin >> ../logs/perf.log 2>&1
bash "$1"
echo "QUEUE_DONE $1" >> ../logs/queue.log
