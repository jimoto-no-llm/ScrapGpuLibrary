#!/bin/bash
# usage: wait_then.sh PID queue.sh   -- waits for process PID (a sess.sh) to end, then runs the queue with the perf pin
cd $HOME/moe-ep/hec
while kill -0 $1 2>/dev/null; do sleep 10; done
exec ./sess.sh $2
