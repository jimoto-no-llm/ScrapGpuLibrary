#!/usr/bin/env python3
"""Summarize llama-bench jsonl outputs (logs/runs/*.out): one line per tag with pp128 / pp512 / tg64 / pp512+tg64 (mean +- sd, t/s).
usage: summarize.py DIR [glob]"""
import glob, json, os, sys

d = sys.argv[1]
pat = sys.argv[2] if len(sys.argv) > 2 else "*.out"
rows = []
for f in sorted(glob.glob(os.path.join(d, pat))):
    tag = os.path.basename(f)[:-4]
    tests = {}
    meta = {}
    for line in open(f):
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            j = json.loads(line)
        except Exception:
            continue
        if "avg_ts" not in j:
            continue
        key = (j["n_prompt"], j["n_gen"])
        tests[key] = (j["avg_ts"], j["stddev_ts"], len(j.get("samples_ts", [])))
        meta = j
    if not tests:
        rows.append((tag, None, None))
        continue
    rows.append((tag, tests, meta))

def fmt(t):
    return "-" if t is None else f"{t[0]:.2f} ± {t[1]:.2f}"

print("| tag | devices | -ncmoe | -ub | pp512 | tg128 | pp512+tg128 (t/s overall) | pp128 | tg64 | reps |")
print("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
for tag, tests, meta in rows:
    if tests is None:
        print(f"| {tag} | no result | | | | | | | | |")
        continue
    reps = max(v[2] for v in tests.values())
    print(f"| {tag} | {meta.get('devices')} | {meta.get('n_cpu_moe')} | {meta.get('n_ubatch')} | {fmt(tests.get((512,0)))} | {fmt(tests.get((0,128)))} | {fmt(tests.get((512,128)))} | {fmt(tests.get((128,0)))} | {fmt(tests.get((0,64)))} | {reps} |")
