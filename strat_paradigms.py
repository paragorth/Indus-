"""Strategy 37: slot paradigms from complementary distribution + shared contexts.
For frequent signs a,b: co-occurrence ratio = texts containing both / expected under independence;
context similarity = cosine of (left-neighbour, right-neighbour, position) vectors.
Paradigm pairs: similarity high AND co-occurrence ratio low. Null: the same statistics after
shuffling signs within texts (keeps co-occurrence, destroys contexts)."""
import csv, collections, math, itertools, random
T=set()
for r in csv.DictReader(open('data/im77/im77_corpus_lines.csv')):
    t=tuple(r['signs_clean'].split())
    if '0' not in t and len(t)>=2: T.add(t)
T=list(T)
def ctx(texts):
    v=collections.defaultdict(collections.Counter); df=collections.Counter()
    for t in texts:
        for x in set(t): df[x]+=1
        for i,x in enumerate(t):
            v[x]['L'+(t[i-1] if i else '^')]+=1; v[x]['R'+(t[i+1] if i<len(t)-1 else '$')]+=1
    return v,df
def cos(a,b):
    num=sum(a[k]*b[k] for k in a if k in b); return num/math.sqrt(sum(x*x for x in a.values())*sum(x*x for x in b.values()))
v,df=ctx(T); N=len(T)
top=[x for x,_ in df.most_common(60)]
co=collections.Counter()
for t in T:
    s=[x for x in set(t) if x in top]
    for a,b in itertools.combinations(sorted(s),2): co[(a,b)]+=1
pairs=[]
for a,b in itertools.combinations(sorted(top),2):
    exp=df[a]*df[b]/N; r=co[(a,b)]/exp if exp else 1
    pairs.append((cos(v[a],v[b]),r,a,b,co[(a,b)],exp))
# null for similarity threshold
rng=random.Random(0); Ts=[tuple(rng.sample(t,len(t))) for t in T]; vs,_=ctx(Ts)
nulls=sorted(cos(vs[a],vs[b]) for a,b in itertools.combinations(sorted(top),2))
thr=nulls[int(.99*len(nulls))]
print('similarity 99th pct under shuffled-context null',round(thr,3))
sel=[p for p in pairs if p[0]>thr and p[1]<0.25 and p[5]>=3]
sel.sort(reverse=True)
for s,r,a,b,c,e in sel[:40]: print(f"M{a:>4} ~ M{b:>4}  context sim {s:.2f}  co-occur {c} vs exp {e:.1f}")
# connected components
g=collections.defaultdict(set)
for s,r,a,b,c,e in sel: g[a].add(b); g[b].add(a)
seen=set(); comps=[]
for x in g:
    if x in seen: continue
    st=[x]; comp=set()
    while st:
        y=st.pop()
        if y in comp: continue
        comp.add(y); st+=list(g[y])
    seen|=comp; comps.append(sorted(comp,key=int))
print('paradigm classes:',comps)
