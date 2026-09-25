"""Strategy 1: Kober-style alternations (data only, Wells IDs, reading order).
A 'stem' is a sequence of >=2 signs that recurs with >=2 different next signs
at the end of a segment. Two final signs 'co-alternate' if they follow the same
stem. Compare co-alternation structure against a within-segment shuffled control."""
import random, itertools, collections as C
import indus_core as IC

def stems_endings(segs, L=2):
    d=C.defaultdict(C.Counter)
    for s in segs:
        if len(s)>=L+1:
            d[tuple(s[-L-1:-1])][s[-1]]+=1
    return {k:v for k,v in d.items() if len(v)>=2}

def coalt(se):
    m=C.Counter()
    for v in se.values():
        for a,b in itertools.combinations(sorted(v),2): m[(a,b)]+=1
    return m

def stats(segs):
    se=stems_endings(segs); m=coalt(se)
    trip=sum(1 for v in se.values() if len(v)>=3)
    strong=sum(1 for c in m.values() if c>=3)
    return len(se),trip,strong,m,se

texts=IC.unique_texts(IC.load_corpus()); segs=[s for t in texts for s in t]
n,trip,strong,m,se=stats(segs)
rng=random.Random(0); ctrl=[]
for i in range(20):
    sh=[]
    for s in segs:
        s=list(s); rng.shuffle(s); sh.append(s)
    ctrl.append(stats(sh)[:3])
mean=lambda k: sum(c[k] for c in ctrl)/len(ctrl)
print(f"stems with alternating endings: real {n}  shuffled {mean(0):.0f}")
print(f"'triplets' (stem with >=3 endings): real {trip}  shuffled {mean(1):.0f}")
print(f"ending pairs alternating after >=3 shared stems: real {strong}  shuffled {mean(2):.1f}")
print("top alternating ending pairs:", m.most_common(15))
print("top triplets:", sorted(((k,dict(v)) for k,v in se.items() if len(v)>=3), key=lambda x:-sum(x[1].values()))[:10])
