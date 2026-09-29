"""S324 (audit idea 4): seal-to-sealing flow. Every clay sealing (TAG*) whose text (>= 3 signs) matches a seal text
exactly: same site or another site? Also same find-quarter at Mohenjo-daro? Control: match sealings to seals with the
text labels permuted among seals of the same length (1,000x): expected number of exact matches and cross-site share."""
import json,collections,random
C=json.load(open('data/derived/merged-corpus-canonical.json'))
seals=[(r['site'],r.get('area-section'),tuple(r['seq_raw'])) for r in C if r['type'].startswith('SEAL') and r['seq_raw'] and len(r['seq_raw'])>=3]
tags=[(r['site'],r.get('area-section'),tuple(r['seq_raw']),r['cisi']) for r in C if r['type'].startswith('TAG') and r['seq_raw'] and len(r['seq_raw'])>=3]
by=collections.defaultdict(list)
for st,q,s in seals: by[s].append((st,q))
m=[(t,by[t[2]]) for t in tags if t[2] in by]
print('sealings (3+ signs)',len(tags),'with an exact seal match',len(m))
cross=0;sameq=0;mdq=0
for (st,q,s,c),L in m:
    sites={x for x,_ in L}
    if st not in sites: cross+=1
    print('  ',c,st,q,s,'seals at',sorted(sites))
    if st=='Mohenjo-daro' and q not in (None,'-','--'):
        qs=[y for x,y in L if x=='Mohenjo-daro' and y not in (None,'-','--')]
        if qs: mdq+=1; sameq+= q in qs
print(f'cross-site matches {cross}/{len(m)}; Mohenjo-daro same-quarter {sameq}/{mdq}')
lens=collections.defaultdict(list)
for st,q,s in seals: lens[len(s)].append(s)
rnd=random.Random(32); null=[]
for _ in range(1000):
    pool={L:set(rnd.sample(v,len(v))) for L,v in lens.items()}
    # permute: shuffle sign order within seal texts to break identity but keep lengths
    fake=set(tuple(rnd.sample(s,len(s))) for s in (x for v in lens.values() for x in v))
    null.append(sum(1 for t in tags if t[2] in fake))
print('matches expected if seal texts were scrambled: median',sorted(null)[500])
