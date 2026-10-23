#!/usr/bin/env python3
"""markdown table of llama-bench jsonl runs: sumrz.py LOGDIR PREFIX [PREFIX...]  (tag, pp512, tg128, pg512,128, rc)"""
import glob, json, os, re, sys
print("| tag | pp512 | tg128 | pg512,128 | rc |")
print("| --- | --- | --- | --- | --- |")
for pre in sys.argv[2:]:
    for f in sorted(glob.glob(os.path.join(sys.argv[1], pre + "*.out"))):
        tag = os.path.basename(f)[:-4]
        r = {}
        for line in open(f):
            line = line.strip()
            if not line.startswith("{"):
                continue
            j = json.loads(line)
            if j["n_prompt"] == 512 and j["n_gen"] == 0: k = "pp512"
            elif j["n_gen"] == 128 and j["n_prompt"] == 0: k = "tg128"
            elif j["n_prompt"] == 512: k = "pg"
            else: continue
            r[k] = (j["avg_ts"], j["stddev_ts"])
        lg = f[:-4] + ".log"
        rc = re.search(r"# rc=(\d+)", open(lg).read()) if os.path.exists(lg) else None
        fm = lambda k: "%.2f ± %.2f" % r[k] if k in r else "-"
        print("| %s | %s | %s | %s | %s |" % (tag, fm("pp512"), fm("tg128"), fm("pg"), rc.group(1) if rc else "?"))
