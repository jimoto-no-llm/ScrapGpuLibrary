#!/usr/bin/env python3
"""markdown table of the llama-bench jsonl runs of LOGDIR whose tag starts with PREFIX: summarize.py LOGDIR PREFIX"""
import glob, json, os, re, sys

print("| tag | pp512 | tg128 | pg512,128 | SGLX MUL_MAT ms (N=1) | layers phase ms | rc |")
print("| --- | --- | --- | --- | --- | --- | --- |")
for f in sorted(glob.glob(os.path.join(sys.argv[1], sys.argv[2] + "*.out"))):
    tag = os.path.basename(f)[:-4]
    r = {}
    for line in open(f):
        line = line.strip()
        if not line.startswith("{"):
            continue
        j = json.loads(line)
        if j["n_prompt"] == 512 and j["n_gen"] == 0:
            k = "pp512"
        elif j["n_gen"] == 128 and j["n_prompt"] == 0:
            k = "tg128"
        elif j["n_prompt"] == 512:
            k = "pg"
        else:
            continue
        r[k] = (j["avg_ts"], j["stddev_ts"])
    log = open(f[:-4] + ".log", errors="replace").read()
    rc = re.search(r"# rc=(\d+)", log)

    def fmt(k):
        return "%.2f ± %.2f" % r[k] if k in r else "-"

    prof = re.findall(r"SGLX profile MUL_MAT N=1: calls (\d+), mean ([\d.]+) ms", log)
    ph = re.findall(r"layers on the other devices \(GET_ROWS end -> MUL_MAT start\) ([\d.]+) ms", log)
    print("| %s | %s | %s | %s | %s | %s | %s |" % (tag, fmt("pp512"), fmt("tg128"), fmt("pg"), prof[-1][1] if prof else "-", ph[-1] if ph else "-", rc.group(1) if rc else "?"))
