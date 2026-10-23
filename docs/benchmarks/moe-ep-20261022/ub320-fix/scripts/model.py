# 2 ubatch の重なりモデル（README 4.2 / 4.5）: T = R(a) + max(W(a), R(b)) + W(b)。fA の T(n) 実測を使い、前回の -ub 掃引に rho を当てる。推定であり実測ではない。python3 model.py
import bisect
n=[64,96,128,160,192,224,256,288,320,352,384,448,512]
T=[270.5,313.5,339.4,386.2,404.5,453.9,481.2,529.3,556.0,600.6,621.4,698.5,761.9]
def f(x):
    if x<=n[0]: return T[0]*x/n[0]
    i=bisect.bisect_left(n,x)
    if n[i]==x: return T[i]
    x0,x1=n[i-1],n[i]; return T[i-1]+(T[i]-T[i-1])*(x-x0)/(x1-x0)
meas={192:652,224:644,256:694,288:689,304:694,320:717,336:693,352:686,384:682,416:651,448:652,512:676}
def total(ub,rho,ovh=0):
    parts=[];rem=512
    while rem>0:
        a=min(ub,rem);parts.append(a);rem-=a
    rx=0;wx=0
    for a in parts:
        rx+=rho*f(a); wx=max(wx,rx)+(1-rho)*f(a)
    return wx+ovh*len(parts),parts
for rho in [0.3,0.4,0.45,0.5,0.55,0.6,0.65,0.7]:
    err=0;out=[]
    for ub,v in meas.items():
        t,p=total(ub,rho); pred=512/t*1000; err+=(pred-v)**2; out.append((ub,round(pred)))
    print(rho,round((err/len(meas))**.5,1),out)

print("--- prediction: scale R by 22/21 and W by 2/3 (one layer moved WX->RX), rho=0.57")
def total2(ub,rho,sr,sw):
    parts=[];rem=512
    while rem>0:
        a=min(ub,rem);parts.append(a);rem-=a
    rx=0;wx=0
    for a in parts:
        rx+=sr*rho*f(a); wx=max(wx,rx)+sw*(1-rho)*f(a)
    return wx
for ub in [128,192,224,256,288,320,352,384,416,448,512]:
    b=512/total2(ub,0.57,1,1)*1000; m=512/total2(ub,0.57,22/21,2/3)*1000
    print(ub,"5/1 model %.0f   6/1 model %.0f"%(b,m))
