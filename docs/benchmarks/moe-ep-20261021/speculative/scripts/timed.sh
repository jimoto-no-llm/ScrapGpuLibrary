#!/bin/bash
# prints wall/user/sys of the wrapped command to stderr
TIMEFORMAT="# timed: real=%R user=%U sys=%S"
time "$@"
