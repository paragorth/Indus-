"""S302: replication of S301 (seated person W176/M48 (+ three-headed W100/M8) + jar M342) in IM77.
Predictions: texts with M8-M342 carry M48 earlier above M48's base rate; texts with M48-M342 never carry M8;
M48-M8 order fixed. Binomial tests against base rates."""
import csv,collections
from scipy.stats import binomtest
rows=list(csv.DictReader(open('data/im77/im77_corpus_lines.csv'))); sides=collections.defaultdict(list)
for r in rows: sides[(r['text_no'],r['side'])].append(r)
seen=set(); T=[]
for k,ls in sides.items():
    ls=sorted(ls,key=lambda r:int(r['line'])); s=[]
    for l in ls: s+=l['signs_clean'].split()
    s=[int(x) for x in s if x.isdigit() and x not in ('0','000')]
    if len(s)>=3 and (ls[0]['site'],tuple(s)) not in seen: seen.add((ls[0]['site'],tuple(s))); T.append(s)
A,X,B=48,8,342; N=len(T)
bA=sum(A in s for s in T)/N; bX=sum(X in s for s in T)/N
xb=[(s,s.index(X)) for s in T if any(s[i]==X and s[i+1]==B for i in range(len(s)-1))]
k=sum(A in s[:i] for s,i in xb)
ab=[s for s in T if any(s[i]==A and s[i+1]==B for i in range(len(s)-1))]; kx=sum(X in s for s in ab)
adj=collections.Counter((s[i],s[i+1]) for s in T for i in range(len(s)-1))
print(f'texts {N}; base M48 {bA:.3f}, base M8 {bX:.3f}')
print(f'M8-M342 texts {len(xb)}: M48 earlier {k} (expected {bA*len(xb):.1f}); p={binomtest(k,len(xb),bA,alternative="greater").pvalue:.2g}')
print(f'M48-M342 texts {len(ab)}: with M8 {kx} (expected {bX*len(ab):.1f}); p_low={binomtest(kx,len(ab),bX,alternative="less").pvalue:.2g}')
print(f'order M48-M8 {adj[(48,8)]} vs M8-M48 {adj[(8,48)]}')
