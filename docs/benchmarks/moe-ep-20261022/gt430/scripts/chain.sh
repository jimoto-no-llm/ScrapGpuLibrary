#!/bin/bash
# chain.sh DONE_MARK QUEUE...: wait for "QUEUE_DONE <DONE_MARK>" in queue.log, then run each queue under perf pin, one session after the other
cd ~/moe-ep/g430/scripts
mark=$1; shift
until grep -q "QUEUE_DONE $mark" ../logs/queue.log 2>/dev/null; do sleep 5; done
for q in "$@"; do ./sess2.sh $q; done
