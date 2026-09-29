"""S298: generalise S296/S297. Group signs into shape families by keyword in the Wells shape description
(top-200 dossiers; >= 15 tokens): fish, jar, tree, leaf, person, U-shape, triangle, rectangle/box, pitchfork.
For each family: (a) mean pairwise context cosine vs 2,000 frequency-matched random sets of the same size;
(b) adjacent stacking within the family and whether order is fixed (pairs with >= 6 adjacencies, share in the
majority direction). Families are defined by shape words only, before looking at contexts. seq_raw."""
import json,collections,random,math,itertools
C=json.load(open('data/derived/merged-corpus-canonical.json'))
dos=json.load(open('data/derived/sign-dossiers-top200.json'))
KW={'fish':['fish'],'jar':['jar'],'tree':['tree'],'leaf':['leaf'],'person':['person'],'U-shape':['u with','u '],
    'triangle':['triangle'],'box':['rectangle','square','box'],'pitchfork':['pitchfork','pitckfork']}
seen=set(); T=[]
for r in C:
    s=r['seq_raw']
    if not s or (r['site'],tuple(s)) in seen: continue
    seen.add((r['site'],tuple(s))); T.append(s)
f=collections.Counter(a for s in T for a in s)
ctx=collections.defaultdict(collections.Counter)
for s in T:
    for i,a in enumerate(s):
        ctx[a]['L%s'%(s[i-1] if i else '^')]+=1; ctx[a]['R%s'%(s[i+1] if i+1<len(s) else '$')]+=1
def cos(a,b):
    A,B=ctx[a],ctx[b]; return sum(A[k]*B[k] for k in A)/math.sqrt(sum(v*v for v in A.values())*sum(v*v for v in B.values()))
def mc(S): P=list(itertools.combinations(S,2)); return sum(cos(a,b) for a,b in P)/len(P)
pool=sorted(f,key=lambda a:f[a]); rank={a:i for i,a in enumerate(pool)}; rnd=random.Random(11)
adj=collections.Counter((s[i],s[i+1]) for s in T for i in range(len(s)-1))
for fam,kws in KW.items():
    S=sorted({x['glyph'] for x in dos if x['shape'] and any(k in x['shape'].lower() for k in kws) and f[x['glyph']]>=15})
    if fam=='triangle': S=[a for a in S if a not in (520,)]  # arrow counted separately? keep simple
    if len(S)<3: print(fam,'too few',S); continue
    obs=mc(S); ge=0
    for _ in range(2000):
        R=[]
        for a in S:
            i=rank[a]; R.append(rnd.choice([b for b in pool[max(0,i-15):i+16] if b not in R and b not in S]))
        ge+=mc(R)>=obs
    pairs=[(a,b,adj[(a,b)],adj[(b,a)]) for a,b in itertools.combinations(S,2) if adj[(a,b)]+adj[(b,a)]>=6]
    fixed=[p for p in pairs if max(p[2],p[3])/(p[2]+p[3])>=0.8]
    print(f'{fam:9s} n={len(S)} {S}\n   cosine {obs:.3f} P={(ge+1)/2001:.4f}; stacked pairs {len(pairs)}, fixed-order (>=80%) {len(fixed)}: {[(a,b,x,y) for a,b,x,y in pairs]}')
