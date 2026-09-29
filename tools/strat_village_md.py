"""S308b: the same village/kin-name test inside Mohenjo-daro, with find-spot quarters (area-section) as the unit:
recurring non-frame 2-3 sign groups (>= 3 seal texts) whose texts all come from ONE quarter; control = quarter
labels permuted among the seal texts (1,000x)."""
import json,collections,random
exec(open('tools/strat_village.py').read().split("seen=set(); T=[]")[0])
seen=set(); T=[]
for r in C:
    s=r['seq_raw']; q=r.get('area-section')
    if r['site']!='Mohenjo-daro' or not r['type'].startswith('SEAL') or not s or len(s)<3 or q in (None,'','-','--'): continue
    k=tuple(s)
    if k in seen: continue
    seen.add(k); T.append((q,s))
print('MD seal texts with a quarter',len(T),collections.Counter(q for q,_ in T).most_common(8))
def grams(s):
    return {tuple(s[i:i+n]) for n in (2,3) for i in range(len(s)-n+1) if not any(a in FRAME or a in NUM for a in s[i:i+n])}
occ=collections.defaultdict(set)
for i,(_,s) in enumerate(T):
    for g in grams(s): occ[g].add(i)
rec={g:v for g,v in occ.items() if len(v)>=3}
lab=[q for q,_ in T]
cnt=lambda L:sum(1 for v in rec.values() if len({L[i] for i in v})==1)
obs=cnt(lab); rnd=random.Random(18); null=[]
for _ in range(1000):
    p=lab[:]; rnd.shuffle(p); null.append(cnt(p))
null.sort(); print(f'recurring groups {len(rec)}; restricted to one quarter {obs}; null median {null[500]}, max {null[-1]}; P={(sum(x>=obs for x in null)+1)/1001:.4f}')
for g,v in sorted(rec.items(),key=lambda x:-len(x[1])):
    if len({lab[i] for i in v})==1: print('  ',g,lab[next(iter(v))],len(v),[T[i][1] for i in list(v)[:3]])
