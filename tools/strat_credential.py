"""S347 (user hypothesis: the whole seal text is a title/credential, 'like a degree': shield (opener) + what it
commands/owns). Predictions:
(1) Texts are assembled from standard parts: a seal text can be segmented completely into chunks (length 1-3) that
    each occur in >= 3 OTHER texts in the same order; single signs count only if they occur in >= 10 other texts.
    Statistic: share of seal texts fully segmentable. Control: the same on texts shuffled within each text (20x),
    which keeps sign frequencies but breaks the building blocks.
(2) More parts = bigger seal: seal size vs text length, within Mohenjo-daro square seals (Spearman), and vs number of
    'standard parts' used."""
import json,csv,collections,random
from scipy.stats import spearmanr
C=json.load(open('data/derived/merged-corpus-canonical.json'))
seen=set(); T=[]
for r in C:
    s=r['seq_raw']
    if s and len(s)>=3 and (r['site'],tuple(s)) not in seen: seen.add((r['site'],tuple(s))); T.append((r,tuple(s)))
def counts(texts):
    c=collections.Counter()
    for s in texts:
        for n in (1,2,3):
            for g in {s[i:i+n] for i in range(len(s)-n+1)}: c[g]+=1
    return c
def segmentable(s,c):
    ok=[False]*(len(s)+1); ok[0]=True
    for i in range(len(s)):
        if not ok[i]: continue
        for n in (1,2,3):
            g=s[i:i+n]
            if len(g)<n: continue
            need=10 if n==1 else 3
            if c[g]-1>=need: ok[i+n]=True   # minus itself
    return ok[-1]
seals=[s for r,s in T if r['type'].startswith('SEAL')]
c=counts([s for _,s in T]); obs=sum(segmentable(s,c) for s in seals)/len(seals)
rnd=random.Random(38); null=[]
for _ in range(20):
    TT=[tuple(rnd.sample(s,len(s))) for _,s in T]; cc=counts(TT)
    SS=[tt for (r,_),tt in zip(T,TT) if r['type'].startswith('SEAL')]
    null.append(sum(segmentable(s,cc) for s in SS)/len(SS))
print(f'(1) seal texts fully built from standard parts: {obs:.3f}; shuffled null median {sorted(null)[10]:.3f}, max {max(null):.3f}')
raw={x['cisi']:x for x in csv.DictReader(open('data/raw/inscriptions.csv')) if x['cisi'] not in ('-','')}
L=[];S=[]
for r,s in T:
    x=raw.get(r['cisi'])
    if r['type']=='SEAL:S' and r['site']=='Mohenjo-daro' and x:
        try: m=max(float(x['horizontal(mm)']),float(x['vertical(mm)']))
        except: continue
        if m>0: L.append(len(s)); S.append(m)
print('(2) Mohenjo-daro square seals: n=%d; text length vs size Spearman rho=%.3f p=%.2g'%(len(L),*spearmanr(L,S)))
by=collections.defaultdict(list)
for l,m in zip(L,S): by[min(l,8)].append(m)
print('    mean size by length:',{k:round(sum(v)/len(v),1) for k,v in sorted(by.items())})
print('(3) nesting: share of texts (3-5 signs) that appear whole inside a LONGER text (ranks built by adding parts)')
def nest(texts):
    S=set(texts); longer=[t for t in S if len(t)>=4]
    subs=set()
    for t in longer:
        for n in (3,4,5):
            for i in range(len(t)-n+1):
                if n<len(t): subs.add(t[i:i+n])
    small=[t for t in S if 3<=len(t)<=5]
    return sum(t in subs for t in small)/len(small)
o=nest([s for _,s in T]); nn=[]
for _ in range(20): nn.append(nest([tuple(rnd.sample(s,len(s))) for _,s in T]))
print(f'    observed {o:.3f}; shuffled median {sorted(nn)[10]:.3f}, max {max(nn):.3f}')
print('(2b) size vs length with opener/title content held apart: partial check at fixed length 4-5')
for Lx in (4,5):
    a=[m for (r,s),m in zip([(r,s) for r,s in T if r['type']=='SEAL:S' and r['site']=='Mohenjo-daro' and raw.get(r['cisi'])],[None]*0)]
