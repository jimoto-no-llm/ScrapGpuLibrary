#!/usr/bin/env python3
"""Build the small retrieval task (queries.txt / docs.txt / texts_*.txt) from the Markdown docs of this repository.
Passage matching inside one section: doc = characters 0-500 of the cleaned section body, query = characters 500-800 of the same body
(sections shorter than 900 characters are skipped).
Line i of queries.txt is the right answer for line i of docs.txt. Run with a repo root and an output directory."""
import re, sys, pathlib
root, out = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
Q, D = [], []
for p in sorted(list(root.glob("docs/*.md")) + [root / "README.md", root / "README.en.md", root / "ROADMAP.md"]):
    t = p.read_text(encoding="utf-8", errors="ignore")
    title = re.search(r"^# (.+)$", t, re.M)
    title = title.group(1).strip() if title else p.stem
    parts = re.split(r"^#{2,3} (.+)$", t, flags=re.M)
    for h, body in zip(parts[1::2], parts[2::2]):
        body = re.sub(r"```.*?```", " ", body, flags=re.S)
        body = re.sub(r"\|[-| :]+\|", " ", body)
        body = re.sub(r"[\s|*`>#]+", " ", body).strip()
        if len(body) < 900:
            continue
        Q.append(body[500:800])
        D.append(body[:500])
        if len(Q) >= 400:
            break
n = len(Q)
pre_q, pre_d = "task: search result | query: ", "title: none | text: "
(out / "queries.txt").write_text("\n".join(pre_q + q for q in Q) + "\n", encoding="utf-8")
(out / "docs.txt").write_text("\n".join(pre_d + d for d in D) + "\n", encoding="utf-8")
# one combined file (queries then docs) for the single-pass dump
(out / "texts_retrieval.txt").write_text("\n".join([pre_q + q for q in Q] + [pre_d + d for d in D]) + "\n", encoding="utf-8")
print("pairs", n)
