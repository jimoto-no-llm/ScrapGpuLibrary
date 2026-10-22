import json,sys,re
for f in sys.argv[1:]:
    r=[]
    for l in open(f):
        try: d=json.loads(l)
        except: continue
        r.append("%s%d:%.1f(%.1f)"%("pp" if d["n_gen"]==0 else ("tg" if d["n_prompt"]==0 else "pg"), d["n_prompt"] or d["n_gen"], d["avg_ts"], d["stddev_ts"]))
    print(f.split("/")[-1], " ".join(r))
