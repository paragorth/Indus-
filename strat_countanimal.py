"""S200: Do seals whose text contains a counted title (numeral + W220/390/900/405, S198) carry particular animals?
Control: permute animal labels among seals within each site 5000x; statistic = max standardized excess over animals."""
import json,random,collections as C,math
d=json.load(open('data/derived/merged-corpus-reading-order.json'))
b=json.load(open('data/derived/bridge_extended.json'))
NUM=(set(range(1,8))|set(range(12,21))|set(range(25,30))|set(range(31,38))|{39,42,43,55,56})-{2,31}
CT={220,390,900,405}
seals=[x for x in d if x['type'].startswith('SEAL') and x.get('symbol')]
def has(s): return any(s[i] in CT and s[i-1] in NUM for i in range(1,len(s)))
rows=[(x['site'],x['symbol'],has(x['seq'])) for x in seals]
top=[a for a,n in C.Counter(r[1] for r in rows).most_common() if n>=15]
print('animals',[(a,n) for a,n in C.Counter(r[1] for r in rows).most_common(10)])
def table(rows):
    out={}
    tot=sum(r[2] for r in rows)/len(rows)
    for a in top:
        g=[r[2] for r in rows if r[1]==a]; n=len(g); k=sum(g); e=n*tot
        out[a]=(k,n,(k-e)/math.sqrt(e*(1-tot)+1e-9))
    return out
real=table(rows)
for a,(k,n,z) in real.items(): print(a,'%d/%d z=%.2f'%(k,n,z))
stat=max(abs(z) for k,n,z in real.values())
bysite=C.defaultdict(list)
for i,r in enumerate(rows): bysite[r[0]].append(i)
random.seed(3); null=[]
for _ in range(5000):
    labs=[r[1] for r in rows]
    for s,ix in bysite.items():
        v=[labs[i] for i in ix]; random.shuffle(v)
        for i,a in zip(ix,v): labs[i]=a
    null.append(max(abs(z) for k,n,z in table([(r[0],l,r[2]) for r,l in zip(rows,labs)]).values()))
print('max|z| %.2f  P=%.4f'%(stat,sum(v>=stat for v in null)/5000))
