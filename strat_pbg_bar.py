"""S160: Were pottery stamps (POT:T:s) made with bar seals? Kenoyer & Meadow (CISI 3.1) state that PBG imprints at
Harappa come from long rectangular seals (archaeological observation). Test from texts alone: stamped-pottery texts
share more sign bigrams with bar-seal texts than with square-seal texts. Control: square pool subsampled to bar-pool
size (2,000 draws), same site."""
import json,random,collections as C
d=json.load(open('data/derived/merged-corpus-reading-order.json'))
def bg(s): return {(s[i],s[i+1]) for i in range(len(s)-1)}
def uniq(xs):
    out,seen=[],set()
    for x in xs:
        t=tuple(x['seq'])
        if t not in seen: seen.add(t); out.append(x['seq'])
    return out
random.seed(2)
for site in ('Harappa','Mohenjo-daro',None):
    f=lambda x: site is None or x['site']==site
    pot=uniq([x for x in d if f(x) and x['type'].upper()=='POT:T:S' and len(x['seq'])>=2])
    bar=uniq([x for x in d if f(x) and x['type']=='SEAL:R'])
    sq=uniq([x for x in d if f(x) and x['type']=='SEAL:S'])
    PB=set().union(*map(bg,pot)) if pot else set()
    def share(pool): 
        B=set().union(*map(bg,pool)); return len(PB&B)
    ob=share(bar); sims=[share(random.sample(sq,len(bar))) for _ in range(2000)]
    p=sum(s>=ob for s in sims)/2000
    # exact whole-text match
    tb=sum(tuple(t) in {tuple(b) for b in bar} for t in pot); ts=sum(tuple(t) in {tuple(b) for b in sq} for t in pot)
    print(site,'pot',len(pot),'bigrams',len(PB),'bar',len(bar),'sq',len(sq),'| shared with bar',ob,'with equal-size square mean',round(sum(sims)/2000,1),'P',p,'| whole-text matches bar',tb,'square',ts)
