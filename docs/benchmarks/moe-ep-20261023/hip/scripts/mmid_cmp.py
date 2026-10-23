#!/usr/bin/env python3
"""compare two test-backend-ops perf outputs (HIP vs Vulkan): mmid_cmp.py A.out B.out  -> markdown for the cases present in both"""
import re, sys
def load(p):
    d = {}
    for l in open(p, errors="replace"):
        l = re.sub(r"\x1b\[[0-9;]*m", "", l)
        m = re.match(r"\s+(MUL_MAT_ID\(.*?\)):\s+(\d+) runs -\s+([\d.]+) us/run -\s+([\d.]+) (\w+)FLOP/run -\s+([\d.]+) (\w+)FLOPS", l)
        if m:
            d[m.group(1)] = (float(m.group(3)), float(m.group(6)), m.group(7))
    return d
a, b = load(sys.argv[1]), load(sys.argv[2])
print("| MUL_MAT_ID case | %s us | %s us | ratio %s/%s |" % ("HIP", "Vulkan", "HIP", "Vulkan"))
print("| --- | --- | --- | --- |")
n = 0
for k in a:
    if k in b:
        short = re.sub(r",amax=[\d.]+|,b=0|,m_v=0", "", k)
        print("| %s | %.1f | %.1f | %.2f |" % (short, a[k][0], b[k][0], a[k][0] / b[k][0])); n += 1
print("\ncommon cases: %d (HIP %d, Vulkan %d)" % (n, len(a), len(b)))
