"""S336-S338 batch.
S336: the nameless quantity seals 'opener-2-N-item' (all items): list complete texts of exactly 4 signs; find-spots
      (area-section) and emblems; do same-N seals cluster in one quarter (vs quarter labels permuted)?
S337: 'of N fish': texts opener-2-N-220: what follows 220 (name? closer?).
S338: tablets: the SAME text with DIFFERENT numbers (real counts) vs seals (fixed): for texts differing only in a
      numeral sign, count pairs on tablets vs seals, normalised by number of texts of each type."""
import json,collections,random
C=json.load(open('data/derived/merged-corpus-canonical.json'))
NUM={1:1,3:3,4:4,5:5,16:6,17:7,18:8,31:1,32:2,33:3,34:4}
emb=lambda r:(r.get('symbol') or '').split(':')[0]
print('== S336 complete 4-sign quantity seals')
Q=[r for r in C if r['seq_raw'] and len(r['seq_raw'])==4 and r['seq_raw'][0] in (817,861,820) and r['seq_raw'][1]==2 and r['seq_raw'][2] in NUM]
for r in Q: print('  ',r['cisi'],r['site'],r['type'],emb(r),r.get('area-section'),r['seq_raw'])
items=collections.Counter(r['seq_raw'][3] for r in Q); print('  items',items)
md=[r for r in Q if r['site']=='Mohenjo-daro' and r.get('area-section') not in (None,'-','--')]
print('  Mohenjo-daro with quarter',len(md),collections.Counter(r['area-section'] for r in md))
print('== S337 of N fish')
seen=set(); F=collections.Counter()
for r in C:
    s=r['seq_raw']
    if not s or tuple(s) in seen: continue
    seen.add(tuple(s))
    for i in range(len(s)-3):
        if i==0 and s[0] in (817,861,820) and s[1]==2 and s[2] in NUM and s[3]==220:
            F[tuple(s[4:])]+=1
print('  continuations after opener-2-N-fish:',F.most_common(12))
print('== S338 same text, different number: tablets vs seals')
def pairs(typ):
    T=list({tuple(r['seq_raw']) for r in C if r['type'].startswith(typ) and r['seq_raw'] and len(r['seq_raw'])>=3})
    key=collections.defaultdict(set)
    for s in T:
        for i,a in enumerate(s):
            if a in NUM: key[(s[:i],s[i+1:])].add(a)
    return sum(len(v)*(len(v)-1)//2 for v in key.values()), len(T), [ (k,v) for k,v in key.items() if len(v)>1][:6]
for typ in ('TAB','SEAL'):
    p,n,ex=pairs(typ); print(f'  {typ}: {p} number-variant pairs among {n} distinct texts ({1000*p/n:.1f} per 1000)'); 
    for k,v in ex: print('     ',k,sorted(v))
