#!/usr/bin/env python3
"""Markdown tables from hec sim JSON files: make_tables.py sim_design.json [sim_other.json ...]"""
import json, sys
def grp(label): return "random" if label.startswith("random") else "text"
for path in sys.argv[1:]:
    d = json.load(open(path)); p = d["params"]
    print(f"\n### {path}  (cpu={p['cpu']}, q41={p['q41']}, serial_host={p.get('serial_host')})\n")
    pols = ["static-oracle", "static-prefill", "static-cross", "lru", "hyst"]
    cfgs = ["A:RX+WX", "C:+GT710"]
    print("| scenario | " + " | ".join(f"{po} {c}" for po in pols[:3] for c in cfgs) + " | lru C | hyst C |")
    print("|" + "---|" * (2 + 6 + 1))
    for lab, sc in d["scenarios"].items():
        cells = []
        for po in pols[:3]:
            for c in cfgs:
                r = sc.get(f"{po}|{c}")
                cells.append(f"{r['h']:.2f} / {r['tg']:.1f}" if r else "-")
        for po in ("lru", "hyst"):
            r = sc.get(f"{po}|C:+GT710")
            cells.append((f"{r['h']:.2f} / {r['tg']:.1f}" if r else "-"))
        print(f"| {lab} | " + " | ".join(cells) + " |")
