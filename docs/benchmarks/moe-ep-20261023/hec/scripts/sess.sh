#!/bin/bash
# usage: sess.sh QUEUE.sh   -- perf pin for the whole queue, perf auto at exit (the only place that touches perf)
sudo -n /usr/local/sbin/sgl-driver-switch perf pin > /dev/null 2>&1
trap 'sudo -n /usr/local/sbin/sgl-driver-switch perf auto > /dev/null 2>&1' EXIT
bash "$1"
