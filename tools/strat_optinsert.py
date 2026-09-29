"""S301: are there other 'optional insert' phrases like the arrow complex (S300)? Pattern A (X) B: A directly
before B occurs >= 10 times; X directly before B occurs >= 10 times; texts with X-B carry A earlier at >= 2x the
base rate of A (binomial p < 0.001); texts with A-B almost never carry X (observed <= 0.25 x expected).
Signs A, X, B distinct. Control: the same search on texts whose signs are shuffled WITHIN each text
(20 shuffles), which keeps text composition but breaks order. seq_raw; results printed with Mahadevan numbers."""
import json,collections,random
from scipy.stats import binomtest
C=json.load(open('data/derived/merged-corpus-canonical.json'))
b=json.load(open('data/derived/bridge_extended.json'))
M=lambda w:'W%d/M%s'%(w,','.join(map(str,b.get(str(w),['?']))))
seen=set(); T=[]
for r in C:
    s=r['seq_raw']
    if s and len(s)>=3 and (r['site'],tuple(s)) not in seen: seen.add((r['site'],tuple(s))); T.append(s)
def search(T):
    N=len(T); has=collections.defaultdict(set)
    for i,s in enumerate(T):
        for a in s: has[a].add(i)
    pair=collections.defaultdict(list)
    for i,s in enumerate(T):
        for j in range(len(s)-1): pair[(s[j],s[j+1])].append((i,j))
    byB=collections.defaultdict(list)
    for (x,y),occ in pair.items():
        if len(occ)>=10 and x!=y: byB[y].append(x)
    out=[]
    for B,lefts in byB.items():
        for X in lefts:
            xb=pair[(X,B)]
            for A in lefts:
                if A==X: continue
                base=len(has[A])/N
                k=sum(1 for i,j in xb if A in T[i][:j]); n=len(xb)
                if k<5 or k/n<2*base: continue
                if binomtest(k,n,base,alternative='greater').pvalue>=0.001: continue
                ab=pair[(A,B)]; baseX=len(has[X])/N
                kx=sum(1 for i,j in ab if X in T[i]); exp=baseX*len(ab)
                if exp>=2 and kx<=0.25*exp: out.append((A,X,B,k,n,kx,round(exp,1)))
    return out
R=search(T)
print('patterns found',len(R))
for A,X,B,k,n,kx,e in sorted(R,key=lambda r:-r[4]): print(f'  {M(A)} ({M(X)}) {M(B)}: X-B texts with A earlier {k}/{n}; A-B texts with X {kx} vs {e} expected')
rnd=random.Random(13); null=[]
for _ in range(20):
    S=[rnd.sample(s,len(s)) for s in T]; null.append(len(search(S)))
print('shuffled-within-text null:',sorted(null))
