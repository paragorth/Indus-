"""S308 (user hypothesis): a seal reads animal (owner mark) + title + NAME + VILLAGE/HOME NAME. A home-place name
predicts sign groups (2-3 signs, excluding frame signs, numerals and closers) that (a) recur on >= 3 different
seal texts, (b) all at ONE site, (c) sit in a consistent position (same distance from the end). Personal names do
not recur; titles recur across sites. Statistic: number of site-restricted recurring n-grams. Control: site labels
permuted among seal texts, 1,000x (keeps every n-gram's count, breaks any site tie). Wells seq_raw, square and
round seals, site+text dedup; sites with >= 30 seal texts."""
import json,collections,random
C=json.load(open('data/derived/merged-corpus-canonical.json'))
FRAME={817,861,820,2,60,740,400,90,520,151,156,527,226,617,154,158,236,700}
NUM={1,3,4,5,16,17,18,31,32,33,34,55,56}
seen=set(); T=[]
for r in C:
    s=r['seq_raw']
    if not r['type'].startswith('SEAL') or not s or len(s)<3: continue
    k=(r['site'],tuple(s))
    if k in seen: continue
    seen.add(k); T.append((r['site'],s))
sc=collections.Counter(st for st,_ in T); T=[(st,s) for st,s in T if sc[st]>=30]
print('seal texts',len(T),dict(collections.Counter(st for st,_ in T)))
def grams(s):
    out=set()
    for n in (2,3):
        for i in range(len(s)-n+1):
            g=tuple(s[i:i+n])
            if not any(a in FRAME or a in NUM for a in g): out.add((g,len(s)-i-n))
    return out
G=[grams(s) for _,s in T]
occ=collections.defaultdict(list)
for i,gs in enumerate(G):
    for g,d in gs: occ[g].append((i,d))
rec={g:v for g,v in occ.items() if len({i for i,_ in v})>=3}
def count(sites):
    res=[]
    for g,v in rec.items():
        ss={sites[i] for i,_ in v}
        if len(ss)==1: res.append(g)
    return res
sites=[st for st,_ in T]; obs=count(sites)
rnd=random.Random(17); null=[]
for _ in range(1000):
    p=sites[:]; rnd.shuffle(p); null.append(len(count(p)))
null.sort()
print(f'recurring n-grams (>=3 texts): {len(rec)}; site-restricted: {len(obs)}; null median {null[500]}, 95% {null[950]}, max {null[-1]}; P={(sum(x>=len(obs) for x in null)+1)/1001:.4f}')
for g in sorted(obs,key=lambda g:-len(rec[g])):
    v=rec[g]; ds=collections.Counter(d for _,d in v)
    print(f'  {g} at {sites[v[0][0]]}: {len(v)} texts; distance-from-end {dict(ds)}; e.g. {[T[i][1] for i,_ in v[:3]]}')
