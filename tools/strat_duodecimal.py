"""S273: are 12 and 24 special units (calendar / duodecimal) or ordinary counts?
Numeral glyph values (GRAMMAR.md): W3-7 = 3-7 (short, one row), W12-20 = 2-10 (two tiers),
W25-29 = 5-9, W32-39 = tall 2-9, W55 = 12, W56 = 24; W1/W2 excluded (markers, S234).
Counts per value in distinct texts (site+text dedup). Control: fit a smooth decline (log-linear in
value) to values 3-10 and ask how 11, 12 and 24 compare with it."""
import json,collections,math
import numpy as np
m=json.load(open('data/derived/merged-corpus-canonical.json'))
VAL={}
for w in range(3,8): VAL[w]=w
for w,v in zip(range(12,21),range(2,11)): VAL[w]=v
for w,v in zip(range(25,30),range(5,10)): VAL[w]=v
for w,v in zip(range(32,40),range(2,10)): VAL[w]=v
VAL[55]=12; VAL[56]=24
seen=set(); c=collections.Counter()
for r in m:
    s=tuple(r.get('seq_raw') or [])
    if not s or (r['site'],s) in seen: continue
    seen.add((r['site'],s))
    for v in {VAL[x] for x in s if x in VAL}: c[v]+=1
print('distinct texts per value:',dict(sorted(c.items())))
xs=np.array([v for v in range(3,11)]); ys=np.array([math.log(c[v]) for v in xs])
b,a=np.polyfit(xs,ys,1)
for v in (11,12,24): print(f'value {v}: observed {c.get(v,0)}, smooth expectation {math.exp(a+b*v):.1f}')
# what does 12 count, versus 3-7?
nx12=collections.Counter(); nxs=collections.Counter(); seen=set()
for r in m:
    s=r.get('seq_raw') or []
    if not s or (r['site'],tuple(s)) in seen: continue
    seen.add((r['site'],tuple(s)))
    for i,x in enumerate(s):
        if x in VAL and i+1<len(s):
            (nx12 if VAL[x]==12 else nxs if 3<=VAL[x]<=7 else collections.Counter())[s[i+1]]+=1
t12=sum(nx12.values()); ts=sum(nxs.values())
print('after 12:',[(k,v,round(v/t12,2)) for k,v in nx12.most_common(8)])
print('after 3-7:',[(k,v,round(v/ts,2)) for k,v in nxs.most_common(8)])
from scipy.stats import fisher_exact
for k in (740,220,255,705):
    a=nx12[k]; b=t12-a; c_=nxs[k]; d=ts-c_
    print(f'W{k}: 12 -> {a}/{t12}, 3-7 -> {c_}/{ts}, Fisher p={fisher_exact([[a,b],[c_,d]])[1]:.3g}')
# site profile of 12 vs other numerals
sites12=collections.Counter(); sitesN=collections.Counter(); seen=set()
for r in m:
    s=r.get('seq_raw') or []
    if not s or (r['site'],tuple(s)) in seen: continue
    seen.add((r['site'],tuple(s)))
    v={VAL[x] for x in s if x in VAL}
    if 12 in v: sites12[r['site']]+=1
    elif v: sitesN[r['site']]+=1
print('sites with 12:',sites12.most_common(6)); print('sites, other numerals:',sitesN.most_common(6))
