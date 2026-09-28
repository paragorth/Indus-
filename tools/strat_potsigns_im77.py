"""S295: replication of S294 (pots have their own sign set; bare tall numerals and 'N X N') in IM77.
Pottery graffiti vs all other objects: per-sign Fisher (greater), BH-FDR 5%; null = pot label shuffled over texts
within site (200x). Also: single-sign pot texts, and symmetric 'a X a' triples on pots vs elsewhere."""
import csv,collections,random
from scipy.stats import fisher_exact
rows=list(csv.DictReader(open('data/im77/im77_corpus_lines.csv')))
sides=collections.defaultdict(list)
for r in rows: sides[(r['text_no'],r['side'])].append(r)
seen=set(); T=[]
for k,ls in sides.items():
    ls=sorted(ls,key=lambda r:int(r['line'])); s=[]
    for l in ls: s+=l['signs_clean'].split()
    s=[x for x in s if x.isdigit() and x not in ('0','000')]
    if not s: continue
    key=(ls[0]['site'],tuple(s))
    if key in seen: continue
    seen.add(key); T.append((ls[0]['object_type']=='pottery graffito',ls[0]['site'],s))
def bh(ps,q=.05):
    o=sorted(range(len(ps)),key=lambda i:ps[i]); k=0
    for r,i in enumerate(o,1):
        if ps[i]<=q*r/len(ps): k=r
    return o[:k]
def scan(lab):
    inn=[s for (_,_,s),l in zip(T,lab) if l]; out=[s for (_,_,s),l in zip(T,lab) if not l]
    ci=collections.Counter(a for s in inn for a in set(s)); co=collections.Counter(a for s in out for a in set(s))
    K=[];P=[]
    for a in set(ci)|set(co):
        if ci[a]+co[a]<5: continue
        p=fisher_exact([[ci[a],len(inn)-ci[a]],[co[a],len(out)-co[a]]],alternative='greater')[1]
        K.append((a,ci[a],len(inn),co[a],len(out),p)); P.append(p)
    return [K[i] for i in bh(P)]
lab=[p for p,_,_ in T]; H=scan(lab)
print('pot texts',sum(lab),'others',len(lab)-sum(lab))
for h in sorted(H,key=lambda h:h[-1]): print('  hit M%s %d/%d vs %d/%d p=%.1e'%h)
rnd=random.Random(8); by=collections.defaultdict(list)
for i,(_,site,_) in enumerate(T): by[site].append(i)
null=[]
for _ in range(200):
    L=lab[:]
    for idx in by.values():
        sub=[lab[i] for i in idx]; rnd.shuffle(sub)
        for i,v in zip(idx,sub): L[i]=v
    null.append(len(scan(L)))
null.sort(); print('FDR hits',len(H),'null median',null[100],'max',null[-1])
single=collections.Counter(s[0] for p,_,s in T if p and len(s)==1); print('single-sign pot texts',single.most_common(10))
def sym(s): return sum(1 for i in range(len(s)-2) if s[i]==s[i+2] and s[i]!=s[i+1])
ps=[s for p,_,s in T if p and len(s)>=3]; os_=[s for p,_,s in T if not p and len(s)>=3]
a=sum(sym(s)>0 for s in ps); b=sum(sym(s)>0 for s in os_)
print(f'a-X-a triples: pots {a}/{len(ps)} vs others {b}/{len(os_)}; p={fisher_exact([[a,len(ps)-a],[b,len(os_)-b]],alternative="greater")[1]:.3g}')
for s in ps:
    if sym(s): print('   ',s)
