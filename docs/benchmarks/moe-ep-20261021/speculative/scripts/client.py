#!/usr/bin/env python3
"""client.py PORT OUT.json [N_PREDICT]: sends the three fixed prompts to a llama-server /completion (greedy, no prompt cache),
saves token ids, text and the server timings (predicted_per_second, draft_n, draft_n_accepted)."""
import json, sys, time, urllib.request, os
port, out = sys.argv[1], sys.argv[2]
n = int(sys.argv[3]) if len(sys.argv) > 3 else 192
here = os.path.dirname(os.path.abspath(__file__))
res = {}
def post(prompt):
    body = json.dumps({"prompt": prompt, "n_predict": n, "temperature": 0, "top_k": 1, "seed": 1, "cache_prompt": False,
                       "return_tokens": True, "stream": False}).encode()
    req = urllib.request.Request("http://127.0.0.1:%s/completion" % port, body, {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=600))
for name in ("explain", "code", "chat"):
    prompt = open(os.path.join(here, "..", "prompts", name + ".txt")).read()
    post(prompt[:60]) if name == "explain" else None   # warm-up request (not recorded)
    r = post(prompt)
    t = r.get("timings", {})
    res[name] = {"tokens": r.get("tokens"), "n": len(r.get("tokens") or []), "content": r.get("content"),
                 "predicted_per_second": t.get("predicted_per_second"), "predicted_n": t.get("predicted_n"),
                 "prompt_per_second": t.get("prompt_per_second"), "prompt_n": t.get("prompt_n"),
                 "draft_n": t.get("draft_n"), "draft_n_accepted": t.get("draft_n_accepted")}
json.dump(res, open(out, "w"))
print(json.dumps({k: {x: v[x] for x in v if x not in ("tokens", "content")} for k, v in res.items()}))
