"""S264: find sign pairs that fill the same slots but split by medium (seal vs tablet), like
W390/W405 (S262) and W154/W158 (S263): candidate carving variants of one sign.
For signs with >=10 occurrences: context vector = counts of prev and next signs.
Score = context cosine x medium divergence (share on seals). Control: same statistic with
the medium labels permuted across texts (100x) -> null distribution of the top score."""
import json,collections,random,math
m=json.load(open('data/derived/merged-corpus-reading-order.json'))
m=m if isinstance(m,list) else list(m.values())[0]
T=[(str(r['type']),r['seq']) for r in m if r.get('seq')]
def med(t): return 'seal' if t.startswith('SEAL') else 'tab' if t.startswith('TAB') else None
freq=collections.Counter(x for t,s in T for x in s)
S=[x for x,c in freq.items() if c>=10]
ctx={x:collections.Counter() for x in S}
for t,s in T:
    for i,x in enumerate(s):
        if x in ctx:
            (i and ctx[x].update(['p'+str(s[i-1])])); (i+1<len(s) and ctx[x].update(['n'+str(s[i+1])]))
def cos(a,b):
    if not a or not b: return 0
    num=sum(a[k]*b[k] for k in a if k in b); return num/math.sqrt(sum(v*v for v in a.values())*sum(v*v for v in b.values()))
def seal_share(labels):
    c=collections.defaultdict(lambda:[0,0])
    for (t,s),l in zip(T,labels):
        if l is None: continue
        for x in set(s):
            if x in ctx: c[x][0]+= l=='seal'; c[x][1]+=1
    return {x:(a/n if n>=5 else None) for x,(a,n) in c.items()}
lab=[med(t) for t,s in T]
def scores(labels):
    sh=seal_share(labels); out=[]
    for i,a in enumerate(S):
        for b in S[i+1:]:
            if sh.get(a) is None or sh.get(b) is None: continue
            c=cos(ctx[a],ctx[b])
            if c<0.5: continue
            out.append((c*abs(sh[a]-sh[b]),c,round(sh[a],2),round(sh[b],2),a,b))
    return sorted(out,reverse=True)
obs=scores(lab)
random.seed(3); null=[]
for _ in range(100):
    l=lab[:]; random.shuffle(l); sc=scores(l); null.append(sc[0][0] if sc else 0)
thr=sorted(null)[95]
print('null 95th pct of top score',round(thr,3))
for sc in obs[:20]: print(('*' if sc[0]>thr else ' '),f"W{sc[4]}~W{sc[5]} ctxcos={sc[1]:.2f} seal-share {sc[2]} vs {sc[3]} score={sc[0]:.3f}")
