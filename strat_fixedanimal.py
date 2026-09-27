"""S212: Do the fixed-numeral terms (S198/S201) go with particular seal animals? For each term with 5+ seals carrying an animal,
the share on its most common animal vs the same share after permuting animal labels within site 5000x (max-over-terms statistic)."""
import json,collections as C,random
d=json.load(open('data/derived/merged-corpus-reading-order.json'))
FX=[(32,877),(32,632),(17,585),(17,575),(33,520),(33,923),(32,226),(3,'comb')]
def has(s,f):
    a,b=f
    return any(s[i-1]==a and (s[i]==b or (b=='comb' and 422<=s[i]<=426)) for i in range(1,len(s)))
seals=[x for x in d if x['type'].startswith('SEAL') and x.get('symbol') and x['symbol'] not in ('','-','Unknown')]
def an(a): return a.split(':')[0]
rows=[(x['site'],an(x['symbol']),x['seq']) for x in seals]
base=C.Counter(r[1] for r in rows); N=len(rows)
def stat(labs):
    out={}
    for f in FX:
        z=[l for (site,_,s),l in zip(rows,labs) if has(s,f)]
        if len(z)>=5:
            a,k=C.Counter(z).most_common(1)[0]; out[f]=(a,k,len(z),k/len(z)-base[a]/N)
    return out
real=stat([r[1] for r in rows])
for f,v in real.items(): print(f,v[:3],'excess %.2f'%v[3])
bysite=C.defaultdict(list)
for i,r in enumerate(rows): bysite[r[0]].append(i)
random.seed(31); null=[]
obs=max(v[3] for v in real.values())
for _ in range(2000):
    labs=[r[1] for r in rows]
    for s,ix in bysite.items():
        p=[labs[i] for i in ix]; random.shuffle(p)
        for i,l in zip(ix,p): labs[i]=l
    null.append(max(v[3] for v in stat(labs).values()))
print('base unicorn share %.2f; max excess %.2f P=%.4f'%(base['Bull1']/N,obs,sum(v>=obs for v in null)/2000))
