"""S296: is the fish family (W220 plain, W240 whiskers, W235 hat, W233 bar, W231 stroke, W226 four strokes)
one word family ('fish' + qualifier), as WORKING-DICTIONARY guesses? Prediction: the variants share
neighbours (left and right context vectors) more than random sets of 6 signs matched in frequency.
Statistic: mean pairwise cosine of context vectors. Control: 5,000 frequency-matched random 6-sets.
Also: do fish variants ever sit next to each other (compounds)? All three sequence levels."""
import json,collections,random,math,itertools
C=json.load(open('data/derived/merged-corpus-canonical.json'))
FISH=[220,240,235,233,231,226]; rnd=random.Random(9)
for key in ['seq_raw','seq_strong','seq_all']:
    seen=set(); T=[]
    for r in C:
        s=r.get(key)
        if not s or (r['site'],tuple(s)) in seen: continue
        seen.add((r['site'],tuple(s))); T.append(s)
    f=collections.Counter(a for s in T for a in s)
    ctx=collections.defaultdict(collections.Counter)
    for s in T:
        for i,a in enumerate(s):
            ctx[a]['L%s'%(s[i-1] if i else '^')]+=1; ctx[a]['R%s'%(s[i+1] if i+1<len(s) else '$')]+=1
    def cos(a,b):
        A,B=ctx[a],ctx[b]; num=sum(A[k]*B[k] for k in A)
        return num/math.sqrt(sum(v*v for v in A.values())*sum(v*v for v in B.values()))
    def mc(S): return sum(cos(a,b) for a,b in itertools.combinations(S,2))/15
    obs=mc(FISH)
    pool=sorted(f,key=lambda a:f[a]); rank={a:i for i,a in enumerate(pool)}
    ge=0
    for _ in range(5000):
        S=[]
        for a in FISH:
            i=rank[a]; cand=[b for b in pool[max(0,i-15):i+16] if b not in S and b not in FISH]
            S.append(rnd.choice(cand))
        ge+=mc(S)>=obs
    adj=collections.Counter((s[i],s[i+1]) for s in T for i in range(len(s)-1) if s[i] in FISH and s[i+1] in FISH)
    print(f'{key}: fish-family mean context cosine {obs:.3f}; P={(ge+1)/5001:.4f}; fish-fish adjacencies {sum(adj.values())} {adj.most_common(5)}')
    for a,b in itertools.combinations(FISH,2): print('   ',a,b,round(cos(a,b),2),end=';')
    print()
