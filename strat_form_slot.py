"""S211: Control for S210. Is the front position due to the numeral's form or to the sign after it?
All numeral values; tokens = (series, following sign, front?) where front = text-initial or right after the opener (W817/861/820/W2).
Permute series labels WITHIN each following sign (5000x); statistic = front-rate(short) - front-rate(tall). Only signs taking both series."""
import json,collections as C,random
d=json.load(open('data/derived/merged-corpus-reading-order.json'))
SHORT=(set(range(1,8))|set(range(12,21))|set(range(25,30)))-{2}; TALL=set(range(32,38))|{39}
seen=set(); t=[]
for x in d:
    s=tuple(x['seq']); key=(s,x['site'])
    if key in seen: continue
    seen.add(key)
    for i in range(1,len(s)):
        n=s[i-1]
        if n in SHORT|TALL and s[i]>=60:
            j=i-1
            front= j==0 or s[j-1] in (817,861,820,2)
            t.append(('T' if n in TALL else 'S',s[i],front))
g=C.defaultdict(list)
for k in t: g[k[1]].append(k)
both=[w for w,v in g.items() if len({k[0] for k in v})==2 and len(v)>=5]
z=[k for w in both for k in g[w]]
def stat(z):
    S=[k[2] for k in z if k[0]=='S']; T=[k[2] for k in z if k[0]=='T']
    return sum(S)/len(S)-sum(T)/len(T)
r=stat(z); random.seed(29); nl=[]
idx=C.defaultdict(list)
for i,k in enumerate(z): idx[k[1]].append(i)
for _ in range(5000):
    lab=[k[0] for k in z]
    for w,ix in idx.items():
        p=[lab[i] for i in ix]; random.shuffle(p)
        for i,l in zip(ix,p): lab[i]=l
    nl.append(stat([(l,)+k[1:] for k,l in zip(z,lab)]))
print('signs with both series',len(both),'tokens',len(z))
print('front-rate short minus tall %.3f, within-sign null %.3f, P=%.4f'%(r,sum(nl)/5000,sum(v>=r for v in nl)/5000))
for w in sorted(both,key=lambda w:-len(g[w]))[:10]:
    v=g[w]; S=[k[2] for k in v if k[0]=='S']; T=[k[2] for k in v if k[0]=='T']
    print(' W%d short front %d/%d  tall front %d/%d'%(w,sum(S),len(S),sum(T),len(T)))
