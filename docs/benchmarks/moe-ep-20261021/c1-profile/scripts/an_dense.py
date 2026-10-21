import json,glob,re,sys
def fit(pts):
    n=len(pts); mx=sum(p[0] for p in pts)/n; my=sum(p[1] for p in pts)/n
    b=sum((p[0]-mx)*(p[1]-my) for p in pts)/sum((p[0]-mx)**2 for p in pts); return my-b*mx,b
rows=[]
for f in sorted(glob.glob(sys.argv[1]+'/dense_*.json')):
    tag=f.split('/')[-1][:-5]; out=open(f.replace('.json','.out')).read()
    ts=float(re.search(r'"avg_ts": ([\d.]+)',out)[1]); sz=int(re.search(r'"model_size": (\d+)',out)[1])
    q='Q8' if 'Q8_0' in tag else 'Q4'; dev='RX' if tag.endswith('v0') else 'WX'
    rows.append((dev,q,re.search(r'dense_LFM2.5-(.+?)-Q',tag)[1],sz/1e6,1000/ts,ts))
for dev in('RX','WX'):
    for q in ('Q4','Q8',None):
        pts=[(r[3],r[4]) for r in rows if r[0]==dev and (q is None or r[1]==q)]
        a,b=fit(pts); print(f'{dev} {q or "all"}: ms/token = {a:.2f} + {b*1000:.2f} ms/GB x size  (slope = {1/(b*1000)*1000:.0f} MB/ms = {1/(b*1000):.0f} GB/s), n={len(pts)}')
for r in rows: print(f'{r[0]} {r[1]} {r[2]:14s} {r[3]:7.0f} MB  {r[5]:7.2f} t/s  {r[4]:6.2f} ms/tok')
