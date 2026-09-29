"""S323 (audit idea 1): modifier algebra. Does adding an inner mark (a stroke or dot inside the sign) shift a sign's
use the same way whatever the base? Base -> marked pairs from the shape descriptions: fish W220 -> W231 (stroke in
body), jar W740 -> W741 (small mark inside), leaf W790 -> W832 (dot inside). Features per sign: position profile
(initial / medial / pre-closer / final shares) and slot classes of neighbours; offset = marked minus base.
Statistic: mean pairwise cosine of the 3 offsets. Control: 5,000 random triples of (base, other sign) pairs matched
on frequency."""
import json,collections,random,math,itertools
import numpy as np
C=json.load(open('data/derived/merged-corpus-canonical.json'))
PAIRS=[(220,231),(740,741),(790,832)]
NUM={1,2,3,4,5,16,17,18,31,32,33,34,55,56}; OPEN={817,861,820}; CL={740,520,151,156,527,226,617,154,158,236,700}
seen=set(); T=[]
for r in C:
    s=r['seq_raw']
    if s and (r['site'],tuple(s)) not in seen: seen.add((r['site'],tuple(s))); T.append(s)
def cls(a): return 'N' if a in NUM else 'O' if a in OPEN else 'C' if a in CL else 'X'
feat=collections.defaultdict(lambda:np.zeros(14)); cnt=collections.Counter()
for s in T:
    n=len(s)
    for i,a in enumerate(s):
        v=feat[a]; cnt[a]+=1
        v[0 if i==0 else 3 if i==n-1 else 2 if i+1<n and s[i+1] in CL else 1]+=1
        l=cls(s[i-1]) if i else 'B'; r=cls(s[i+1]) if i+1<n else 'E'
        v[4+'BNOCX'.index(l)]+=1; v[9+'ENOCX'.index(r)]+=1
def prof(a): v=feat[a]; return np.concatenate([v[:4]/max(1,v[:4].sum()), v[4:9]/max(1,v[4:9].sum()), v[9:]/max(1,v[9:].sum())])
def off(b,m): return prof(m)-prof(b)
def mc(O): return np.mean([np.dot(x,y)/np.linalg.norm(x)/np.linalg.norm(y) for x,y in itertools.combinations(O,2)])
obs=mc([off(b,m) for b,m in PAIRS])
pool=sorted([a for a in cnt if cnt[a]>=10],key=lambda a:cnt[a]); rk={a:i for i,a in enumerate(pool)}
rnd=random.Random(31); null=[]
for _ in range(5000):
    O=[]
    for b,m in PAIRS:
        i=rk[m]; m2=rnd.choice([x for x in pool[max(0,i-15):i+16] if x!=b and x!=m]); O.append(off(b,m2))
    null.append(mc(O))
null.sort(); print(f'offset cosine (inner mark) {obs:.3f}; null median {null[2500]:.3f}, 95% {null[4750]:.3f}; P={(sum(x>=obs for x in null)+1)/5001:.4f}')
for b,m in PAIRS: print(b,m,'pos base',np.round(prof(b)[:4],2),'marked',np.round(prof(m)[:4],2))
