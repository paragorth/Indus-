"""S360: whole-corpus mutual-exclusion graph. For every pair of signs with >=25 texts each, expected co-occurrence under
independence (hypergeometric mean); edge if observed co-occurrence is <= 10% of expected with expected >= 4. Cliques of
mutually exclusive signs = paradigm classes (one slot, alternative values). Control: same on texts with signs shuffled
across texts (keeps sign frequencies and text lengths), 200x: number of edges and the largest clique."""
import json,collections,random,itertools,sys
C=json.load(open('data/derived/merged-corpus-canonical.json'))
key=sys.argv[1] if len(sys.argv)>1 else 'seq_raw'
T=list({tuple(r[key]) for r in C if r.get(key) and len(r[key])>=2})
N=len(T); cnt=collections.Counter(s for t in T for s in set(t))
S=[s for s,c in cnt.items() if c>=25]
def edges(texts):
    cnt=collections.Counter(s for t in texts for s in set(t)); co=collections.Counter()
    for t in texts:
        u=sorted(set(t)&set(S))
        for a,b in itertools.combinations(u,2): co[(a,b)]+=1
    E=[]
    for a,b in itertools.combinations(sorted(S),2):
        e=cnt[a]*cnt[b]/len(texts)
        if e>=4 and co[(a,b)]<=0.1*e: E.append((a,b,co[(a,b)],round(e,1)))
    return E
def cliques(E):
    adj=collections.defaultdict(set)
    for a,b,_,_ in E: adj[a].add(b); adj[b].add(a)
    out=[]
    def bk(R,P,X):
        if not P and not X: out.append(R); return
        for v in list(P):
            bk(R|{v},P&adj[v],X&adj[v]); P=P-{v}; X=X|{v}
    bk(set(),set(adj),set()); return sorted(out,key=len,reverse=True)
E=edges(T); CL=cliques(E)
print(f'[{key}] texts {N}, signs tested {len(S)}, exclusion edges {len(E)}, largest clique {len(CL[0]) if CL else 0}')
for c in CL[:12]: print('  clique',sorted(c))
rng=random.Random(1); ne=[]; nc=[]
pool=[s for t in T for s in t]
for _ in range(200):
    rng.shuffle(pool); k=0; sh=[]
    for t in T: sh.append(tuple(pool[k:k+len(t)])); k+=len(t)
    e=edges(sh); c=cliques(e); ne.append(len(e)); nc.append(len(c[0]) if c else 0)
print('control: edges mean',sum(ne)/200,'max',max(ne),'| largest clique mean',sum(nc)/200,'max',max(nc))
