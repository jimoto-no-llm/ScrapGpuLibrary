#!/usr/bin/python3
"""sprof_an.py samples.txt [filter_substr]: resolves module+off with nm, prints self and inclusive tops (only samples containing filter, default llama_decode)."""
import sys,subprocess,bisect,collections,re
fn=sys.argv[1]; flt=sys.argv[2] if len(sys.argv)>2 else 'llama_decode'
cache={}
def syms(path):
    if path in cache: return cache[path]
    try: out=subprocess.run(['nm','--defined-only','-n',path],capture_output=True,text=True).stdout
    except Exception: out=''
    a=[];n=[]
    for l in out.splitlines():
        p=l.split(None,2)
        if len(p)==3 and p[1] in 'tTwWiI':
            a.append(int(p[0],16)); n.append(p[2])
    cache[path]=(a,n); return cache[path]
dem={}
def demangle(names):
    todo=[x for x in names if x not in dem]
    if todo:
        r=subprocess.run(['c++filt'],input='\n'.join(todo),capture_output=True,text=True).stdout.splitlines()
        for x,y in zip(todo,r): dem[x]=y
def res(fr):
    path,off=fr.rsplit('+0x',1); off=int(off,16)
    a,n=syms(path)
    if not a: return path.split('/')[-1]+'+0x%x'%off
    i=bisect.bisect_right(a,off)-1
    return (path.split('/')[-1].split('.so')[0]+':'+n[i]) if i>=0 else path+'?'
S=[l.split() for l in open(fn)]
raw=set()
res_c={}
def R(fr):
    if fr not in res_c: res_c[fr]=res(fr)
    return res_c[fr]
stacks=[]
for s in S:
    st=[R(x) for x in s[1:]]
    while st and (st[0].startswith('sprof:') or st[0].startswith('libc.so.6+0x3fd70')): st=st[1:]
    stacks.append(st)
allnames=set(x for st in stacks for x in st); demangle([x.split(':',1)[1] for x in allnames if ':' in x])
def D(x):
    if ':' in x:
        m,n=x.split(':',1); return m+':'+dem.get(n,n)[:110]
    return x
sel=[st for st in stacks if any(flt in D(x) for x in st)]
import os
EX=os.environ.get('EXCL_LEAF')
if EX: sel=[st for st in sel if EX not in D(st[0])]
print(f'samples total {len(stacks)}, with "{flt}": {len(sel)} (1 sample = 1 ms CPU at 1 kHz)')
self_=collections.Counter(D(st[0]) for st in sel)
incl=collections.Counter()
for st in sel:
    for x in set(D(x) for x in st): incl[x]+=1
print('--- self top 40'); [print(f'{c:6d} {100*c/len(sel):5.1f}%  {k}') for k,c in self_.most_common(40)]
print('--- inclusive top 50'); [print(f'{c:6d} {100*c/len(sel):5.1f}%  {k}') for k,c in incl.most_common(50)]
