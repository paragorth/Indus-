"""S274: did moulded tablets circulate (tokens) or stay as batches?
For each site, pairs of same-type tablets with identical text: share of pairs found in the same
excavation area, and in the same period, vs all pairs of tablets of that type (baseline)."""
import json,collections,itertools,random
m=json.load(open('data/derived/merged-corpus-canonical.json'))
def area(r): a=r.get('area-section'); return a if a not in ('--','-',None,'') else None
def per(r): t=r.get('time'); return t if t not in ('-',None,'') else None
for site in ('Harappa','Mohenjo-daro'):
    for ty in ('TAB:B','TAB:I','TAB:C'):
        R=[r for r in m if r['site']==site and r['type']==ty and r.get('seq') and area(r)]
        by=collections.defaultdict(list)
        for r in R: by[tuple(r['seq'])].append(r)
        same=[(a,b) for g in by.values() if len(g)>1 for a,b in itertools.combinations(g,2)]
        if len(same)<5: continue
        rnd=random.Random(1); allp=[tuple(rnd.sample(R,2)) for _ in range(20000)]
        sa=sum(area(a)==area(b) for a,b in same)/len(same); ba=sum(area(a)==area(b) for a,b in allp)/len(allp)
        sp=[(a,b) for a,b in same if per(a) and per(b)]; bp=[(a,b) for a,b in allp if per(a) and per(b)]
        spp=sum(per(a)==per(b) for a,b in sp)/max(1,len(sp)); bpp=sum(per(a)==per(b) for a,b in bp)/max(1,len(bp))
        print(f'{site:12s} {ty}: texts with copies {sum(1 for g in by.values() if len(g)>1)}, same-text pairs {len(same)} | same area {sa:.2f} vs baseline {ba:.2f} | same period {spp:.2f} (n={len(sp)}) vs {bpp:.2f}')
print('per-text check (each text with >=3 copies counts once): distinct areas among its copies vs random groups of the same size')
for site in ('Harappa','Mohenjo-daro'):
    for ty in ('TAB:B','TAB:I','TAB:C'):
        R=[r for r in m if r['site']==site and r['type']==ty and r.get('seq') and area(r)]
        by=collections.defaultdict(list)
        for r in R: by[tuple(r['seq'])].append(r)
        G=[g for g in by.values() if len(g)>=3]
        if len(G)<3: continue
        rnd=random.Random(2); obs=[]; exp=[]
        for g in G:
            obs.append(len({area(r) for r in g})/len(g))
            exp.append(sum(len({area(r) for r in rnd.sample(R,len(g))}) for _ in range(200))/200/len(g))
        lower=sum(o<e for o,e in zip(obs,exp))
        print(f'  {site:12s} {ty}: {len(G)} texts | distinct-area ratio {sum(obs)/len(G):.2f} vs random {sum(exp)/len(G):.2f} | texts more clustered than random {lower}/{len(G)}')
