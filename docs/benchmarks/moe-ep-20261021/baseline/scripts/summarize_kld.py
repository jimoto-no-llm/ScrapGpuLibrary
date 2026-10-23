#!/usr/bin/env python3
"""KL divergence table from llama-perplexity --kl-divergence outputs. usage: summarize_kld.py DIR"""
import glob, os, re, sys
print("| config | mean KLD ± SE | 99% KLD | Same top p (top-1 agreement) ± SE | PPL(Q) / PPL(CPU) | rc |")
print("| --- | --- | --- | --- | --- | --- |")
for f in sorted(glob.glob(os.path.join(sys.argv[1], "kld_*.out"))):
    tag = os.path.basename(f)[:-4]
    if tag == "kld_base_cpu":
        continue
    t = open(f).read()
    lg = open(f[:-4] + ".log").read()
    rc = re.search(r"# rc=(\d+)", lg)
    m = re.search(r"Mean\s+KLD:\s+([\d.]+) ±\s+([\d.]+)", t)
    p99 = re.search(r"99.0%\s+KLD:\s+([\d.]+)", t)
    tp = re.search(r"Same top p:\s+([\d.]+) ±\s+([\d.]+)", t)
    pr = re.search(r"Mean PPL\(Q\)/PPL\(base\)\s+:\s+([\d.]+)", t)
    if not m:
        print(f"| {tag} | no result | | | | {rc.group(1) if rc else '?'} |")
        continue
    print(f"| {tag[4:]} | {m.group(1)} ± {m.group(2)} | {p99.group(1)} | {tp.group(1)} % ± {tp.group(2)} | {pr.group(1)} | {rc.group(1) if rc else '?'} |")
