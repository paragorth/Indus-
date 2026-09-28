"""S294: do pictographic glosses match the object carrying the text?
G6 'jar = vessel / measure': the jar sign W740 as a COUNTED item (a numeral right before it), or the jar sign at all,
   should be commoner on pottery inscriptions (POT*) than on seals and tablets.
G7 'arrow W520 = weapon': W520 should be commoner on implements (IMPL, copper tools/weapons) than elsewhere.
Plus an open scan: for each object class (POT, IMPL, BNGL, TAG, ROD), which signs are most enriched (Fisher,
BH-FDR), with a within-site permutation null for the number of FDR hits.
Canonical corpus, all three sequence levels; site+text dedup."""
import json,random,collections,math
from scipy.stats import fisher_exact
C=json.load(open('data/derived/merged-corpus-canonical.json'))
NUMS={1,3,4,5,16,17,18,31,32,33,34}
def cls(t):
    for k in ('POT','IMPL','BNGL','TAG','ROD','SEAL','TAB'):
        if t.startswith(k): return k
    return 'OTHER'
def bh(ps,q=0.05):
    m=len(ps); o=sorted(range(m),key=lambda i:ps[i]); k=0
    for r,i in enumerate(o,1):
        if ps[i]<=q*r/m: k=r
    return set(o[:k])
for key in ['seq_raw','seq_strong','seq_all']:
    seen=set(); R=[]
    for r in C:
        s=r.get(key)
        if not s or (r['site'],tuple(s)) in seen: continue
        seen.add((r['site'],tuple(s))); R.append((cls(r['type']),r['site'],s))
    print('==',key,collections.Counter(c for c,_,_ in R))
    def rate(pred,cl): 
        g=[pred(s) for c,_,s in R if c==cl]; return sum(g),len(g)
    for cl in ('POT','SEAL','TAB','TAG'):
        a,n=rate(lambda s:740 in s,cl); b,_=rate(lambda s:any(s[i]==740 and s[i-1] in NUMS for i in range(1,len(s))),cl)
        print(f'  G6 {cl}: jar in {a}/{n} ({a/max(1,n):.2f}); counted jar {b}/{n} ({b/max(1,n):.3f})')
    for cl in ('IMPL','SEAL','TAB','POT'):
        a,n=rate(lambda s:520 in s,cl); print(f'  G7 {cl}: W520 in {a}/{n} ({a/max(1,n):.3f})')
    # open scan
    def scan(labels):
        hits=[]
        for cl in ('POT','IMPL','BNGL','TAG','ROD'):
            inn=[s for (c,_,s),l in zip(R,labels) if l==cl]; out=[s for (c,_,s),l in zip(R,labels) if l!=cl]
            if len(inn)<10: continue
            cnt=collections.Counter(a for s in inn for a in set(s)); cout=collections.Counter(a for s in out for a in set(s))
            ps=[];keys=[]
            for a in set(cnt)|set(cout):
                if cnt[a]+cout[a]<5: continue
                p=fisher_exact([[cnt[a],len(inn)-cnt[a]],[cout[a],len(out)-cout[a]]],alternative='greater')[1]
                ps.append(p); keys.append((cl,a,cnt[a],len(inn),cout[a],len(out),p))
            for i in bh(ps): hits.append(keys[i])
        return hits
    labels=[c for c,_,_ in R]; H=scan(labels)
    for h in sorted(H,key=lambda h:h[-1]): print('   hit',h[0],'W%d'%h[1],f'{h[2]}/{h[3]} vs {h[4]}/{h[5]} p={h[6]:.1e}')
    if key=='seq_raw':
        rnd=random.Random(7); by=collections.defaultdict(list)
        for i,(c,site,s) in enumerate(R): by[site].append(i)
        null=[]
        for _ in range(100):
            L=labels[:]
            for idx in by.values():
                sub=[labels[i] for i in idx]; rnd.shuffle(sub)
                for i,v in zip(idx,sub): L[i]=v
            null.append(len(scan(L)))
        null.sort(); print(f'  FDR hits {len(H)}; within-site shuffled null median {null[50]}, max {null[-1]}')
