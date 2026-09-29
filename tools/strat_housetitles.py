"""S329: do houses (emblems) hold different titles? Square seals at Mohenjo-daro + Harappa with an emblem and a
paradigm closer. (a) emblem x closer: mutual information vs emblem labels permuted within site (5,000x).
(b) emblem x title phrase (qualifier + closer) likewise. (c) per emblem: top titles; zebu in particular."""
import json,collections,random,math
C=json.load(open('data/derived/merged-corpus-canonical.json'))
CL={740,520,151,156,527,226,617,154,158,236,700}; SUF={400,90}
emb=lambda r:(r.get('symbol') or '').split(':')[0]
rows=[]; seen=set()
for r in C:
    s=r['seq_raw']; e=emb(r)
    if r['type']!='SEAL:S' or r['site'] not in ('Mohenjo-daro','Harappa') or not s or len(s)<2 or e in ('','-'): continue
    if (r['site'],tuple(s)) in seen: continue
    seen.add((r['site'],tuple(s)))
    core=list(s)
    while len(core)>1 and core[-1] in SUF: core.pop()
    if core[-1] in CL: rows.append((r['site'],e,core[-1],(core[-2] if len(core)>=2 else None,core[-1])))
def mi(xs,ys):
    n=len(xs); cx=collections.Counter(xs); cy=collections.Counter(ys); cxy=collections.Counter(zip(xs,ys))
    return sum(c/n*math.log(c*n/(cx[x]*cy[y])) for (x,y),c in cxy.items())
rnd=random.Random(35)
for name,idx in (('closer',2),('title phrase',3)):
    E=[r[1] for r in rows]; Y=[r[idx] for r in rows]; S=[r[0] for r in rows]
    # merge rare emblems
    ce=collections.Counter(E); E=[e if ce[e]>=15 else 'other' for e in E]
    cy=collections.Counter(Y); Y=[y if cy[y]>=5 else 'rare' for y in Y]
    o=mi(E,Y); by=collections.defaultdict(list)
    for i,s in enumerate(S): by[s].append(i)
    ge=0
    for _ in range(5000):
        P=E[:]
        for idxs in by.values():
            sub=[E[i] for i in idxs]; rnd.shuffle(sub)
            for i,v in zip(idxs,sub): P[i]=v
        ge+=mi(P,Y)>=o
    print(f'emblem x {name}: n={len(rows)} MI={o:.4f} P={(ge+1)/5001:.4f}')
tab=collections.defaultdict(collections.Counter)
for r in rows: tab[r[1]][r[3]]+=1
for e in ('Zebu','Elep','Gaur','Rhin','Tigr','Bull1'):
    n=sum(tab[e].values()); print(f'  {e} n={n}: jar {sum(v for k,v in tab[e].items() if k[1]==740)/max(1,n):.2f}; top phrases {tab[e].most_common(4)}')
