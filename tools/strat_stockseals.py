"""S352: are 'stock' texts (numeral + tree family, S351) physically different from 'office' texts (ending in the jar W740)?
Compare object type and emblem distributions (objects, not deduped texts). Control: permute the stock/office label
across the pooled objects 2000x; report categories with p<0.01."""
import json,collections,random
C=json.load(open('data/derived/merged-corpus-canonical.json'))
NUM={1,3,4,5,16,17,18,31,32,33,34}; TREE={390,405,407}
emb=lambda r:((r.get('symbol') or '').split(':')[0] or 'none').strip()
def lab(s):
    if any(s[i] in TREE and s[i-1] in NUM for i in range(1,len(s))): return 'stock'
    if s[-1]==740: return 'office'
R=[(lab(r['seq_raw']),r) for r in C if r.get('seq_raw') and len(r['seq_raw'])>=2]
R=[(l,r) for l,r in R if l]
print('objects',collections.Counter(l for l,_ in R))
for fld,f in (('type',lambda r:r['type']),('emblem',emb),('site',lambda r:r['site'])):
    labs=[l for l,_ in R]; vals=[f(r) for _,r in R]
    obs=collections.Counter(v for l,v in zip(labs,vals) if l=='stock'); n=labs.count('stock')
    top=[v for v,_ in collections.Counter(vals).most_common(8)]
    ge=collections.Counter(); le=collections.Counter(); random.seed(3)
    for _ in range(2000):
        sh=random.sample(labs,len(labs)); c=collections.Counter(v for l,v in zip(sh,vals) if l=='stock')
        for v in top: ge[v]+=c[v]>=obs[v]; le[v]+=c[v]<=obs[v]
    tot=collections.Counter(vals)
    print('==',fld)
    for v in top:
        e=tot[v]*n/len(R); fl=' <-- over' if ge[v]<20 else (' <-- under' if le[v]<20 else '')
        print(f'  {v}: stock {obs[v]} (exp {e:.1f}), office {tot[v]-obs[v]}{fl}')
