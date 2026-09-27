"""S202: Are fixed-numeral terms (S198/S201) pan-Indus titles? Number of sites per term vs all other bigrams with the same token count
(deduplicated by text+site). Control: rank-percentile of each term's site count among bigrams with equal frequency (+-2)."""
import json,collections as C
d=json.load(open('data/derived/merged-corpus-reading-order.json'))
seen=set(); bg=C.defaultdict(list)
for x in d:
    s=tuple(x['seq']); key=(s,x['site'])
    if key in seen: continue
    seen.add(key)
    for i in range(1,len(s)):
        b=(s[i-1], 'comb' if 422<=s[i]<=426 else s[i]); bg[b].append(x['site'])
FX=[(32,877),(32,632),(17,585),(17,575),(33,520),(33,923),(32,226),(3,'comb')]
pct=[]
for f in FX:
    n=len(bg[f]); k=len(set(bg[f]))
    peers=[len(set(v)) for b,v in bg.items() if abs(len(v)-n)<=2 and b not in FX]
    p=sum(q<k for q in peers)/len(peers)+0.5*sum(q==k for q in peers)/len(peers)
    pct.append(p); print(f,'tokens',n,'sites',k,'peer median',sorted(peers)[len(peers)//2],'n peers',len(peers),'percentile %.2f'%p)
print('mean percentile %.2f (0.5 = typical)'%(sum(pct)/len(pct)))
import random
random.seed(7); m=sum(pct)/len(pct)
null=[sum(random.random() for _ in pct)/len(pct) for _ in range(10000)]
print('P(mean>=obs) under uniform %.4f'%(sum(v>=m for v in null)/10000))
