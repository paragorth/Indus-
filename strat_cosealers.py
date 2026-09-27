"""S227: Lothal tags stamped by 2+ different seals (co-sealers; Frenez & Tosi 2005: joint responsibility). Are co-sealers' texts more alike
than random pairs of Lothal seal texts? Similarity = shares the final sign (closer), shares the opener, Jaccard of sign sets.
Control: 10000 random pairs of distinct Lothal tag texts."""
import json,collections as C,random,itertools
d=json.load(open('data/derived/merged-corpus-reading-order.json'))
byc=C.defaultdict(set); alltx=set()
for x in d:
    if x['site']=='Lothal' and x['type'].startswith('TAG') and len(x['seq'])>=2:
        byc[x['cisi']].add(tuple(x['seq'])); alltx.add(tuple(x['seq']))
multi=[v for v in byc.values() if len(v)>=2]
pairs=[p for v in multi for p in itertools.combinations(sorted(v),2)]
print('multi-seal tags',len(multi),'co-sealer pairs',len(pairs))
for v in multi: print(sorted(v))
OP={817,861,820}
def sim(a,b): return (a[-1]==b[-1], bool(set(a)&OP) and bool(set(b)&OP), len(set(a)&set(b))/len(set(a)|set(b)))
def agg(ps):
    s=[sim(a,b) for a,b in ps]; n=len(s)
    return [sum(x[0] for x in s)/n, sum(x[1] for x in s)/n, sum(x[2] for x in s)/n]
o=agg(pairs); alltx=sorted(alltx); random.seed(83)
null=[agg([tuple(random.sample(alltx,2)) for _ in range(len(pairs))]) for _ in range(5000)]
for j,name in enumerate(['same closer','both opener','jaccard']):
    print(name,'co-sealers %.3f random %.3f P=%.4f'%(o[j],sum(v[j] for v in null)/5000,sum(v[j]>=o[j] for v in null)/5000))
