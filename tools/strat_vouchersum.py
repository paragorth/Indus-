"""S317 (audit idea 5): do Harappa voucher counts add up to round totals by deposit? Tablets carrying tall 2-4 + W700
(TAB types, Harappa; NOT deduplicated: each tablet is one voucher). Group by deposit = (area-section, block-house,
phase). Per deposit with >= 3 vouchers: total count; statistic = number of deposits whose total is a multiple of 12,
and separately of 6 and 4. Control: counts shuffled across all vouchers keeping deposit sizes (10,000x)."""
import json,collections,random
C=json.load(open('data/derived/merged-corpus-canonical.json'))
TALL={31:1,32:2,33:3,34:4}
V=[]
for r in C:
    s=r['seq_raw']
    if r['site']!='Harappa' or not r['type'].startswith('TAB') or not s: continue
    for i in range(len(s)-1):
        if s[i] in TALL and s[i+1]==700:
            V.append(((r.get('area-section'),r.get('block-house'),r.get('phase')),TALL[s[i]])); break
dep=collections.defaultdict(list)
for d,c in V: dep[d].append(c)
dep={d:v for d,v in dep.items() if len(v)>=3 and d[0] not in (None,'-','--')}
print('vouchers',len(V),'deposits with >=3',len(dep),'sizes',sorted(len(v) for v in dep.values())[-10:])
print('count distribution',collections.Counter(c for _,c in V))
def stat(D,m): return sum(sum(v)%m==0 for v in D)
D=list(dep.values()); allc=[c for v in D for c in v]; sizes=[len(v) for v in D]
rnd=random.Random(27)
for m in (12,6,4,5,10):
    obs=stat(D,m); ge=0
    for _ in range(10000):
        rnd.shuffle(allc); it=iter(allc); S=[[next(it) for _ in range(k)] for k in sizes]
        ge+=stat(S,m)>=obs
    print(f'  totals divisible by {m}: {obs}/{len(D)}; P={(ge+1)/10001:.4f}')
for d,v in sorted(dep.items(),key=lambda x:-len(x[1]))[:8]: print('  ',d,len(v),sum(v),collections.Counter(v))
