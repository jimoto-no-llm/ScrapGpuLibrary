#!/bin/bash
# usage: go.sh TAG CONFIG cmd...   CONFIG is a use_mg/use_env name from ~/moe-ep/scripts/env.sh (amdvk, mix4, mix5, ...) or "none"
# Exports that configuration's environment, then runs cmd under spec/scripts/run.sh (flock ~/bench/gpu-bench.lock, <85 C wait, logs in spec/logs/runs).
. "$(dirname "$0")/env.sh"
TAG=$1; CF=$2; shift 2
case "$CF" in
  none) ;;
  amdvk|mix4|mix5|cl4|cl5) use_mg "$CF" || exit 2 ;;
  *) use_env "$CF" || exit 2 ;;
esac
export GGML_SPEC_CONFIG=$CF
exec "$(dirname "$0")/run.sh" "$TAG" "$@"
