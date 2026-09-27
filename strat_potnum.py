"""S206: Numerals on pottery vs seals/tablets: are pot numerals more often text-final or stand-alone (a label of quantity or capacity),
and larger? Control: Fisher/permutation of object labels among numeral tokens 5000x."""
import json,collections as C,random
d=json.load(open('data/derived/merged-corpus-reading-order.json'))
SHORT=(set(range(1,8))|set(range(12,21))|set(range(25,30)))-{2}; TALL=set(range(32,38))|{39}
VAL={**{i:i for i in range(1,8)},**{i:i-10 for i in range(12,21)},**{i:i-20 for i in range(25,30)},**{i:i-30 for i in range(32,38)},39:9}
seen=set(); tok=[]
for x in d:
    s=tuple(x['seq']); key=(s,x['site'])
    if key in seen: continue
    seen.add(key)
    t=x['type'].split(':')[0]
    for i,w in enumerate(s):
        if w in VAL and w!=2: tok.append((t,i==len(s)-1,len(s)==1,VAL[w],'T' if w in TALL else 'S'))
for t in ['POT','SEAL','TAB','TAG']:
    z=[k for k in tok if k[0]==t]; n=len(z)
    print(t,n,'final %.2f alone %.2f mean value %.2f tall %.2f'%(sum(k[1] for k in z)/n,sum(k[2] for k in z)/n,sum(k[3] for k in z)/n,sum(k[4]=='T' for k in z)/n))
random.seed(13)
for j,name in [(1,'final'),(3,'value')]:
    isp=[k[0]=='POT' for k in tok]
    vals=[float(k[j]) for k in tok]
    obs=sum(v for v,p in zip(vals,isp) if p)/sum(isp)-sum(v for v,p in zip(vals,isp) if not p)/(len(isp)-sum(isp))
    null=[]
    for _ in range(5000):
        random.shuffle(isp); null.append(sum(v for v,p in zip(vals,isp) if p)/sum(isp)-sum(v for v,p in zip(vals,isp) if not p)/(len(isp)-sum(isp)))
    print(name,'pot minus rest %+.3f P=%.4f'%(obs,sum(v>=obs for v in null)/5000))
