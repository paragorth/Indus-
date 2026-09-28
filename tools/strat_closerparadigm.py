"""S289: the closer paradigm. Which signs replace the jar (W740) ending? For each sign with
>= 15 tokens: final rate (share of its tokens that are text-final, or followed only by the
suffix W400/W90) and jar ratio (texts with the sign that also contain W740, observed/expected
by text length). A jar substitute should be final-heavy AND avoid the jar. Control: 1,000
shuffles of signs within texts (keeps text lengths and sign counts, breaks position), giving a
null count of signs that pass both cuts. Runs on seq_raw / seq_strong / seq_all."""
import json,collections,random
C=json.load(open('data/derived/merged-corpus-canonical.json'))
JAR=740; SUF={400,90}; random.seed(2)
def texts(key):
    seen=set(); T=[]
    for r in C:
        s=r.get(key)
        if not s or len(s)<3 or (r['site'],tuple(s)) in seen: continue
        seen.add((r['site'],tuple(s))); T.append(list(s))
    return T
def stats(T):
    bylen=collections.defaultdict(list)
    for s in T: bylen[len(s)].append(JAR in s)
    pj={L:sum(v)/len(v) for L,v in bylen.items()}
    tok=collections.Counter(); fin=collections.Counter(); o=collections.Counter(); e=collections.Counter()
    for s in T:
        core=s[:]
        while len(core)>1 and core[-1] in SUF: core=core[:-1]
        for i,a in enumerate(s):
            tok[a]+=1
            if i==len(core)-1: fin[a]+=1
        for a in set(s):
            o[a]+=JAR in s; e[a]+=pj[len(s)]
    out={}
    for a in tok:
        if tok[a]>=15 and a not in {JAR}|SUF and e[a]>0: out[a]=(fin[a]/tok[a],o[a]/e[a],tok[a])
    return out
def passing(st): return sorted(a for a,(f,r,n) in st.items() if f>=0.4 and r<=0.5)
for key in ['seq_raw','seq_strong','seq_all']:
    T=texts(key); st=stats(T); P=passing(st)
    null=[]
    for _ in range(1000 if key=='seq_raw' else 200):
        S=[random.sample(s,len(s)) for s in T]; null.append(len(passing(stats(S))))
    null.sort()
    print(f'== {key}: {len(T)} texts; signs passing (final>=0.4, jar<=0.5): {len(P)}; null median {null[len(null)//2]}, 95% {null[int(.95*len(null))]}, max {null[-1]}')
    for a in sorted(P,key=lambda a:-st[a][0]): print('  W%d n=%d final=%.2f jar=%.2f'%(a,st[a][2],st[a][0],st[a][1]))
