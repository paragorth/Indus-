"""Strategy 3: is a stroke-modified sign M = base B + fused following sign X?
Test: left context of M ~ left context of B (same stem), and right context of M ~
right context of the bigram (B, X) for the best X. Compare with random sign pairs."""
import math, random, collections as C
import indus_core as IC
texts=IC.unique_texts(IC.load_corpus()); segs=[s for t in texts for s in t]
def ctx(target):
    L=C.Counter(); R=C.Counter()
    for s in segs:
        s=['^']+list(s)+['$']
        for i in range(1,len(s)-1):
            if s[i]==target: L[s[i-1]]+=1; R[s[i+1]]+=1
    return L,R
def bictx(a,b):
    L=C.Counter(); R=C.Counter()
    for s in segs:
        s=['^']+list(s)+['$']
        for i in range(1,len(s)-2):
            if s[i]==a and s[i+1]==b: L[s[i-1]]+=1; R[s[i+2]]+=1
    return L,R
def jsd(p,q):
    keys=set(p)|set(q); sp=sum(p.values()) or 1; sq=sum(q.values()) or 1
    d=0
    for k in keys:
        a=p[k]/sp; b=q[k]/sq; m=(a+b)/2
        if a: d+=a*math.log2(a/m)/2
        if b: d+=b*math.log2(b/m)/2
    return d
freq=C.Counter(x for s in segs for x in s)
common=[x for x,c in freq.most_common(60)]
def best_fusion(M,B):
    LM,RM=ctx(M); LB,_=ctx(B)
    best=None
    for X in common:
        if X in (M,B): continue
        Lb,Rb=bictx(B,X)
        if sum(Rb.values())<5: continue
        d=jsd(LM,Lb)+jsd(RM,Rb)
        if best is None or d<best[0]: best=(d,X,sum(Rb.values()))
    return best, jsd(LM,LB)
PAIRS=[('741','740'),('226','220'),('233','220'),('235','220'),('240','220'),('405','390'),('407','390')]
print('modified  base  best-X  fusion-distance  (n bigram)  left(M)~left(B)')
real=[]
for M,B in PAIRS:
    if freq[M]<5: continue
    (d,X,n),lb=best_fusion(M,B); real.append(d)
    print(f'  {M}  {B}  {X}  {d:.3f}  ({n})  {lb:.3f}')
rng=random.Random(0); null=[]
pool=[x for x,c in freq.most_common(120) if c>=10]
for _ in range(40):
    M,B=rng.sample(pool,2)
    r=best_fusion(M,B)
    if r[0]: null.append(r[0][0])
import statistics
print('random pairs: fusion distance mean %.3f sd %.3f; real mean %.3f'%(statistics.mean(null),statistics.pstdev(null),statistics.mean(real)))
print('share of random pairs at least as good as each real pair:',[round(sum(n<=d for n in null)/len(null),2) for d in real])
