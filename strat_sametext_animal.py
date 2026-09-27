"""S228: When the same full text (3+ signs) is carved on two or more different SEALS (not impressions of one seal), do the seals carry the
same animal? If text = holder's identity and animal = holder's group, same text should mean same animal. Control: permute animals among
the seals with duplicated texts (within site) 5000x; statistic = pairs with same animal."""
import json,collections as C,random,itertools
d=json.load(open('data/derived/merged-corpus-reading-order.json'))
S=[x for x in d if x['type'].startswith('SEAL') and len(x['seq'])>=3 and x['symbol'] not in ('','Unknown')]
g=C.defaultdict(list)
for x in S: g[tuple(x['seq'])].append((x['site'],x['symbol'].split(':')[0],x['cisi']))
dup={t:v for t,v in g.items() if len(v)>=2}
rows=[(t,s,a) for t,v in dup.items() for s,a,c in v]
print('texts on 2+ seals',len(dup),'seals',len(rows))
for t,v in sorted(dup.items(),key=lambda kv:-len(kv[1]))[:15]: print(t,[(s[:4],a) for s,a,c in v])
def stat(rows):
    by=C.defaultdict(list)
    for t,s,a in rows: by[t].append(a)
    return sum(1 for v in by.values() for a,b in itertools.combinations(v,2) if a==b), sum(1 for v in by.values() for _ in itertools.combinations(v,2))
o,npairs=stat(rows); random.seed(89); bys=C.defaultdict(list)
for i,r in enumerate(rows): bys[r[1]].append(i)
nl=[]
for _ in range(5000):
    rr=list(rows)
    for s,ix in bys.items():
        p=[rows[i][2] for i in ix]; random.shuffle(p)
        for i,a in zip(ix,p): rr[i]=(rows[i][0],rows[i][1],a)
    nl.append(stat(rr)[0])
print('same-animal pairs %d of %d; null %.1f; P=%.4f'%(o,npairs,sum(nl)/5000,sum(v>=o for v in nl)/5000))
