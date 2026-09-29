"""S297: replication of S296 in IM77. Fish family via bridge: M59 (W220 plain), M67 (W240), M65 (W235), M72 (W233),
M70 (W231), M60 (W226). Same statistic (mean pairwise context cosine) and control (5,000 frequency-matched 6-sets)."""
import csv,collections,random,math,itertools
rows=list(csv.DictReader(open('data/im77/im77_corpus_lines.csv')))
sides=collections.defaultdict(list)
for r in rows: sides[(r['text_no'],r['side'])].append(r)
seen=set(); T=[]
for k,ls in sides.items():
    ls=sorted(ls,key=lambda r:int(r['line'])); s=[]
    for l in ls: s+=l['signs_clean'].split()
    s=[int(x) for x in s if x.isdigit() and x not in ('0','000')]
    if not s or (ls[0]['site'],tuple(s)) in seen: continue
    seen.add((ls[0]['site'],tuple(s))); T.append(s)
FISH=[59,67,65,72,70,60]; rnd=random.Random(10)
f=collections.Counter(a for s in T for a in s)
ctx=collections.defaultdict(collections.Counter)
for s in T:
    for i,a in enumerate(s):
        ctx[a]['L%s'%(s[i-1] if i else '^')]+=1; ctx[a]['R%s'%(s[i+1] if i+1<len(s) else '$')]+=1
def cos(a,b):
    A,B=ctx[a],ctx[b]; return sum(A[k]*B[k] for k in A)/math.sqrt(sum(v*v for v in A.values())*sum(v*v for v in B.values()))
mc=lambda S:sum(cos(a,b) for a,b in itertools.combinations(S,2))/15
obs=mc(FISH); pool=sorted(f,key=lambda a:f[a]); rank={a:i for i,a in enumerate(pool)}; ge=0
for _ in range(5000):
    S=[]
    for a in FISH:
        i=rank[a]; S.append(rnd.choice([b for b in pool[max(0,i-15):i+16] if b not in S and b not in FISH]))
    ge+=mc(S)>=obs
adj=collections.Counter((s[i],s[i+1]) for s in T for i in range(len(s)-1) if s[i] in FISH and s[i+1] in FISH)
print(f'IM77 texts {len(T)}; counts',{a:f[a] for a in FISH})
print(f'fish-family mean cosine {obs:.3f}; P={(ge+1)/5001:.4f}; adjacencies {sum(adj.values())} {adj.most_common(5)}')
for a,b in itertools.combinations(FISH,2): print(f'  M{a}-M{b} {cos(a,b):.2f}')
