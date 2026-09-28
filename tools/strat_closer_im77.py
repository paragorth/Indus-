"""S291: replication of S289 (closer paradigm) in IM77. Predicted closers, fixed in advance from
Wells via the bridge: M211 (W520), M12 (W151), M254 (W527), M15 (W156), M60 (W226), M328 (W700).
Same cuts as S289: >= 15 tokens, final rate >= 0.4 (last before optional suffix M176/M1),
jar (M342) co-occurrence <= 0.5x expected by text length. Sides with lines joined, site+text dedup,
len >= 3, texts with damaged signs (0/000) dropped. Controls: (1) shuffled-within-text null count
of passing signs; (2) hypergeometric overlap of the predicted set with the passing set."""
import csv,collections,random,math
rows=list(csv.DictReader(open('data/im77/im77_corpus_lines.csv')))
sides=collections.defaultdict(list)
for r in rows: sides[(r['text_no'],r['side'])].append(r)
seen=set(); T=[]
for k,ls in sides.items():
    ls=sorted(ls,key=lambda r:int(r['line'])); s=[]
    for l in ls: s+=l['signs_clean'].split()
    if len(s)<3 or any(x in('0','000') or not x.isdigit() for x in s): continue
    key=(ls[0]['site'],tuple(s))
    if key in seen: continue
    seen.add(key); T.append([int(x) for x in s])
JAR=342; SUF={176,1}; PRED={211,12,254,15,60,328}; random.seed(4)
def stats(T):
    bylen=collections.defaultdict(list)
    for s in T: bylen[len(s)].append(JAR in s)
    pj={L:sum(v)/len(v) for L,v in bylen.items()}
    tok=collections.Counter(); fin=collections.Counter(); o=collections.Counter(); e=collections.Counter()
    for s in T:
        core=s[:]
        while len(core)>1 and core[-1] in SUF: core=core[:-1]
        for i,a in enumerate(s):
            tok[a]+=1; fin[a]+= i==len(core)-1
        for a in set(s): o[a]+=JAR in s; e[a]+=pj[len(s)]
    return {a:(fin[a]/tok[a],o[a]/e[a],tok[a]) for a in tok if tok[a]>=15 and a!=JAR and a not in SUF and e[a]>0}
P=lambda st:sorted(a for a,(f,r,n) in st.items() if f>=0.4 and r<=0.5)
st=stats(T); ps=P(st)
null=sorted(len(P(stats([random.sample(s,len(s)) for s in T]))) for _ in range(500))
print(f'IM77 texts {len(T)}; eligible signs {len(st)}; passing {len(ps)}; null median {null[250]}, 95% {null[475]}, max {null[-1]}')
for a in sorted(ps,key=lambda a:-st[a][0]): print('  M%d n=%d final=%.2f jar=%.2f %s'%(a,st[a][2],st[a][0],st[a][1],'PRED' if a in PRED else ''))
for a in PRED: print('  predicted M%d:'%a, tuple(round(x,2) for x in st[a]) if a in st else 'too rare')
N=len(st); K=len(ps); n=len([a for a in PRED if a in st]); k=len(set(ps)&PRED)
p=sum(math.comb(K,i)*math.comb(N-K,n-i) for i in range(k,min(K,n)+1))/math.comb(N,n)
print(f'overlap {k}/{n} predicted in passing set; hypergeometric P = {p:.2e}')
