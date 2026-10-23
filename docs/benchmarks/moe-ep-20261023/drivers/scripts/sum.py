import sys,json
for f in sys.argv[1:]:
    for l in open(f):
        if not l.startswith("{"): continue
        j=json.loads(l)
        print(f.split("/")[-1][:-4], "pp%d tg%d"%(j["n_prompt"],j["n_gen"]), round(j["avg_ts"],2), "+-", round(j["stddev_ts"],2))
