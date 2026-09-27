"""S231: Are the frozen pre-jar titles (S229) regional? Site distribution (Mohenjo-daro / Harappa / other) of each title vs all other
'A X + jar' texts. Control: permute title labels among jar texts 5000x; statistic = summed TVD of each title's site mix from baseline."""
import json,collections as C,random
d=json.load(open('data/derived/merged-corpus-reading-order.json'))
T={(590,390):'590-390',(590,405):'590-405',(435,690):'435-690',(840,32):'840-||',(17,585):'17-585'}
seen=set(); rows=[]
for x in d:
    s=tuple(x['seq'])
    if (s,x['site']) in seen: continue
    seen.add((s,x['site']))
    site=x['site'] if x['site'] in ('Mohenjo-daro','Harappa') else 'other'
    lab=None
    for i in range(2,len(s)):
        if s[i]==740:
            lab=T.get((s[i-2],s[i-1]),lab or 'rest')
    if lab: rows.append((lab,site,x['type'].split(':')[0]))
base=C.Counter(s for l,s,t in rows); N=len(rows)
def tvd(rows):
    tot=0
    for l in T.values():
        c=C.Counter(s for ll,s,t in rows if ll==l); n=sum(c.values())
        if n: tot+=0.5*sum(abs(c[k]/n-base[k]/N) for k in base)
    return tot
o=tvd(rows); labs=[r[0] for r in rows]; random.seed(103); nl=[]
for _ in range(5000):
    random.shuffle(labs); nl.append(tvd([(l,s,t) for l,(x,s,t) in zip(labs,rows)]))
print('baseline',{k:round(v/N,2) for k,v in base.items()})
for l in T.values(): print(l,dict(C.Counter(s for ll,s,t in rows if ll==l)),dict(C.Counter(t for ll,s,t in rows if ll==l)))
print('TVD %.2f null %.2f P=%.4f'%(o,sum(nl)/5000,sum(v>=o for v in nl)/5000))
