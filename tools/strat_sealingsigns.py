"""S318 (user question: which sign means 'sealed'?). Which signs are over-represented on clay SEALINGS (TAG*, the
actual acts of sealing) vs on SEALS, within the same sites? Per-sign Fisher (greater) with BH-FDR 5%, sites pooled
with a site-stratified permutation null for the number of hits (500x). Mohenjo-daro + Harappa + all others; site+text
dedup within object type. Also the reverse (seal-only signs)."""
import json,collections,random
from scipy.stats import fisher_exact
C=json.load(open('data/derived/merged-corpus-canonical.json'))
seen=set(); R=[]
for r in C:
    s=r['seq_raw']; t='TAG' if r['type'].startswith('TAG') else 'SEAL' if r['type'].startswith('SEAL') else None
    if not s or not t or (r['site'],t,tuple(s)) in seen: continue
    seen.add((r['site'],t,tuple(s))); R.append((t,r['site'],set(s)))
print('sealings',sum(t=='TAG' for t,_,_ in R),'seals',sum(t=='SEAL' for t,_,_ in R))
def scan(labels,want):
    inn=[s for (_,_,s),l in zip(R,labels) if l==want]; out=[s for (_,_,s),l in zip(R,labels) if l!=want]
    ci=collections.Counter(a for s in inn for a in s); co=collections.Counter(a for s in out for a in s)
    K=[];P=[]
    for a in set(ci)|set(co):
        if ci[a]+co[a]<8: continue
        p=fisher_exact([[ci[a],len(inn)-ci[a]],[co[a],len(out)-co[a]]],alternative='greater')[1]; K.append((p,a,ci[a],len(inn),co[a],len(out))); P.append(p)
    o=sorted(range(len(P)),key=lambda i:P[i]); k=0
    for rk,i in enumerate(o,1):
        if P[i]<=0.05*rk/len(P): k=rk
    return [K[i] for i in o[:k]]
labs=[t for t,_,_ in R]
for want in ('TAG','SEAL'):
    H=scan(labs,want)
    print(f'signs enriched on {want}: {len(H)}')
    for p,a,x,n,y,m in H[:10]: print(f'   W{a}: {x}/{n} vs {y}/{m}  p={p:.1e}')
rnd=random.Random(28); by=collections.defaultdict(list)
for i,(_,st,_) in enumerate(R): by[st].append(i)
null=[]
for _ in range(200):
    L=labs[:]
    for idx in by.values():
        sub=[labs[i] for i in idx]; rnd.shuffle(sub)
        for i,v in zip(idx,sub): L[i]=v
    null.append(len(scan(L,'TAG')))
null.sort(); print('null (TAG hits) median',null[100],'max',null[-1])
