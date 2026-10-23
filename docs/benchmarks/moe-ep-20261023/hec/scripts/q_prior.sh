. $HOME/moe-ep/hec/env_hec.sh
. $HOME/moe-ep/hec/cfg_n.sh
export HEC_GEN_VERBOSE=1
C=$H/corpus
# the frequencies of the experts on six other texts (English docs / legal, Python / C++ code, Japanese technical / design text): 512 tokens of prompt and 64 generated each
for c in en_doc en_legal py_code cpp_code ja_tech ja_design; do
  ( N_STATS=$LOGD/stats_$c.txt; export GGML_SGLX_MOE_STATS_OUT=$N_STATS; run_n $LOGD prior_$c $GEN -m $MODEL -ub 512 "${N_ARGS[@]}" --text $C/$c.txt --n-prompt 512 --n-gen 64 )
done
LOGD=$LOGD python3 - <<'PY'
import glob, os, collections
d = os.environ.get('LOGD', '.')
acc = collections.defaultdict(float); n = 0
for f in sorted(glob.glob(d + '/stats_*.txt')):
    n += 1
    for l in open(f):
        il, e, v = l.split()
        acc[(int(il), int(e))] += float(v)
with open(d + '/prior_nat.txt', 'w') as o:
    for (il, e), v in sorted(acc.items()):
        o.write('%d %d %.6g\n' % (il, e, v / max(n, 1)))
print('prior from', n, 'texts')
PY
