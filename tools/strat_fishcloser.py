"""S299: which closing word goes with the fish words? For texts ending in a paradigm closer (S289), compare
closer shares in texts that contain a modified fish (W240/235/233/231) vs texts without. Also where the fish
sits (distance from the closer). Control: fish label permuted within site x text length (5,000x); statistic =
largest standardized enrichment over closers. Replication in IM77 (M67/65/72/70; closers via bridge)."""
import json,collections,random,math,csv
C=json.load(open('data/derived/merged-corpus-canonical.json'))
SUF={400,90}; CL=[740,520,151,156,527,226,617,154,158,236,700]; MF={240,235,233,231}
def run(T,CL,MF,SUF,label):
    rows=[]
    for site,s in T:
        core=list(s)
        while len(core)>1 and core[-1] in SUF: core.pop()
        if core[-1] in CL and len(core)>=2: rows.append((site,len(core),core[-1],any(a in MF for a in core[:-1]),core))
    def stat(lab):
        n1=sum(lab); tot=collections.Counter(r[2] for r in rows); f=collections.Counter(r[2] for r,l in zip(rows,lab) if l)
        best=(-9,None)
        for c in CL:
            e=n1*tot[c]/len(rows)
            if e>=3: z=(f[c]-e)/math.sqrt(e); best=max(best,(z,c))
        return best
    lab=[r[3] for r in rows]; obs=stat(lab)
    strata=collections.defaultdict(list)
    for i,r in enumerate(rows): strata[(r[0],min(r[1],6))].append(i)
    rnd=random.Random(12); ge=0
    for _ in range(5000):
        L=lab[:]
        for idx in strata.values():
            sub=[lab[i] for i in idx]; rnd.shuffle(sub)
            for i,v in zip(idx,sub): L[i]=v
        ge+=stat(L)[0]>=obs[0]
    tot=collections.Counter(r[2] for r in rows); f=collections.Counter(r[2] for r in rows if r[3]); n1=sum(lab)
    print(f'{label}: {len(rows)} closer texts, {n1} with a modified fish')
    for c in CL:
        if tot[c]: print(f'   closer {c}: with fish {f[c]} ({f[c]/n1:.2f}) vs all {tot[c]/len(rows):.2f}')
    print(f'   max enrichment z={obs[0]:.2f} for closer {obs[1]}; P={(ge+1)/5001:.4f}')
    d=collections.Counter(len(r[4])-1-max(i for i,a in enumerate(r[4]) if a in MF) for r in rows if r[3])
    print('   distance of last fish from closer',sorted(d.items()))
seen=set(); T=[]
for r in C:
    s=r['seq_raw']
    if s and (r['site'],tuple(s)) not in seen: seen.add((r['site'],tuple(s))); T.append((r['site'],s))
run(T,CL,MF,SUF,'Wells seq_raw')
rows=list(csv.DictReader(open('data/im77/im77_corpus_lines.csv'))); sides=collections.defaultdict(list)
for r in rows: sides[(r['text_no'],r['side'])].append(r)
seen=set(); T2=[]
for k,ls in sides.items():
    ls=sorted(ls,key=lambda r:int(r['line'])); s=[]
    for l in ls: s+=l['signs_clean'].split()
    s=[int(x) for x in s if x.isdigit() and x not in ('0','000')]
    if s and (ls[0]['site'],tuple(s)) not in seen: seen.add((ls[0]['site'],tuple(s))); T2.append((ls[0]['site'],s))
run(T2,[342,211,12,15,254,60,328],{67,65,72,70},{176,1},'IM77')
