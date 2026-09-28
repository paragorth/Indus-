"""S266: validate the S262-S264 seal/tablet variant pairs. If W405=W390, W158=W154, W318=W320,
W525/526=W527 are one sign each, merging them should create extra texts shared exactly by a
seal and a tablet. Control: merge 4 random pairs of signs matched on frequency (1000x)."""
import json,collections,random
m=json.load(open('data/derived/merged-corpus-reading-order.json'))
m=m if isinstance(m,list) else list(m.values())[0]
T=[(str(r['type']),tuple(r['seq'])) for r in m if r.get('seq')]
PAIRS={405:390,158:154,318:320,525:527,526:527}
def shared(mp):
    se=set(); tb=set()
    for t,s in T:
        s2=tuple(mp.get(x,x) for x in s)
        if t.startswith('SEAL'): se.add(s2)
        elif t.startswith('TAB'): tb.add(s2)
    return len(se&tb)
base=shared({}); obs=shared(PAIRS)
freq=collections.Counter(x for t,s in T for x in s)
signs=list(freq)
def matched(x):
    f=freq[x]; c=[y for y in signs if 0.5*f<=freq[y]<=2*f and y!=x]; return random.choice(c)
random.seed(5); null=[]
for _ in range(1000):
    mp={}
    for a,b in PAIRS.items():
        mp[matched(a)]=matched(b)
    null.append(shared(mp)-base)
print('shared seal/tablet texts: base',base,'after merging variant pairs',obs,'gain',obs-base)
print('null gain mean',sum(null)/len(null),'P(gain>=obs)',sum(g>=obs-base for g in null)/len(null))
