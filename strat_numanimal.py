"""S236: Ice-Age analogy test (Bacon et al. 2023 claim: number of marks beside an animal = months, species-specific).
On Indus seals: does the VALUE of a count (short or tall stroke numeral, W1/W2/W31 excluded) depend on the animal? Seals with exactly one numeral.
Control: permute animal labels within site 5000x; statistic = summed TVD of each animal's value distribution from the overall one."""
import json,collections as C,random
d=json.load(open('data/derived/merged-corpus-reading-order.json'))
VAL={**{i:i for i in range(3,8)},**{i:i-10 for i in range(12,21)},**{i:i-20 for i in range(25,30)},**{i:i-30 for i in range(32,38)},39:9,55:12,56:24}
rows=[]
for x in d:
    if not x['type'].startswith('SEAL') or x['symbol'] in ('','-','Unknown'): continue
    v=[VAL[w] for w in x['seq'] if w in VAL]
    if len(v)==1:
        a=x['symbol'].split(':')[0]
        rows.append((x['site'],a if a in ('Bull1','Gaur','Zebu','Elep','Rhin','Tigr','Buff','Bull') else 'other',min(v[0],8)))
base=C.Counter(v for s,a,v in rows); N=len(rows)
def tvd(labs):
    t=0
    for a in set(labs):
        c=C.Counter(v for (s,_,v),l in zip(rows,labs) if l==a); n=sum(c.values())
        if n>=8: t+=n/N*0.5*sum(abs(c[k]/n-base[k]/N) for k in base)
    return t
labs=[a for s,a,v in rows]; o=tvd(labs)
for a in sorted(set(labs)):
    c=C.Counter(v for s,aa,v in rows if aa==a); print(a,sum(c.values()),dict(sorted(c.items())))
by=C.defaultdict(list)
for i,r in enumerate(rows): by[r[0]].append(i)
random.seed(109); nl=[]
for _ in range(5000):
    l2=list(labs)
    for s,ix in by.items():
        p=[labs[i] for i in ix]; random.shuffle(p)
        for i,v in zip(ix,p): l2[i]=v
    nl.append(tvd(l2))
print('weighted TVD %.3f null %.3f P=%.4f'%(o,sum(nl)/5000,sum(v>=o for v in nl)/5000))
