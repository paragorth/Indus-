"""Cycle 5: transcription robustness in IM77 (Mahadevan 1977), which names 5 sites (MD, Harappa, Lothal, Kalibangan,
Chanhu-daro). (a) Vocabulary JSD between the 5 sites (seals only and all objects; equal 60-text subsamples) vs km and coast
difference, exact permutation (120). (b) texts shared between sites: distance of linked vs unlinked pairs. (c) the cycle 1/2 best
arrows (W125 = M?, leaf family, tall numerals) cannot be tested on 5 sites with power, so only direction is reported for the
frame features already in cycle 2."""
import sys, csv, collections, itertools, math, numpy as np
sys.path.insert(0,'/home/user/Indus-/data/derived/dark'); from loop6_engine import *
from scipy.stats import rankdata
out=open(ROOT+'data/derived/dark/loop6_c5_im77.txt','w')
def P(*a):
    print(*a); print(*a,file=out)
def hav(a,b,c,d):
    R=6371.0; p1,p2=math.radians(a),math.radians(c); dp=p2-p1; dl=math.radians(d-b)
    h=math.sin(dp/2)**2+math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return 2*R*math.asin(math.sqrt(h))
def jsd(p,q):
    m=(p+q)/2
    def kl(a,b):
        nz=a>0; return float((a[nz]*np.log2(a[nz]/b[nz])).sum())
    return 0.5*kl(p,m)+0.5*kl(q,m)
def spear(a,b): return float(np.corrcoef(rankdata(a),rankdata(b))[0,1])
def mantel_exact(A,B):
    n=len(A); iu=np.triu_indices(n,1); a=A[iu]; obs=spear(a,B[iu]); vals=[]
    for p in itertools.permutations(range(n)):
        p=np.array(p); vals.append(spear(a,B[p][:,p][iu]))
    vals=np.array(vals); return obs,float((np.abs(vals)>=abs(obs)-1e-12).mean())

rows=list(csv.DictReader(open(ROOT+'data/im77/im77_corpus_lines.csv')))
MAP={'Mohenjodaro':'Mohenjo-daro','Harappa':'Harappa','Lothal':'Lothal','Kalibangan':'Kalibangan','Chanhudaro':'Chanhu-daro'}
sides=collections.defaultdict(list)
for r in rows: sides[(r['text_no'],r['side'])].append(r)
T=[]; seen=set()
for k,ls in sides.items():
    ls=sorted(ls,key=lambda r:int(r['line'])); s=[]
    for l in ls: s+=l['signs_clean'].split()
    s=[int(x) for x in s if x.isdigit() and x not in ('0','000')]
    site=ls[0]['site']
    if not s or site not in MAP or (site,tuple(s)) in seen: continue
    seen.add((site,tuple(s))); T.append(dict(site=MAP[site],seq=s,type=ls[0]['object_type']))
P('IM77 object types: '+str(collections.Counter(t['type'] for t in T).most_common(8)))
rnd=np.random.default_rng(7)
for label,filt in [('all objects',lambda t:True),('seals only',lambda t:t['type'].lower().startswith('seal'))]:
    TT=[t for t in T if filt(t)]; by=collections.defaultdict(list)
    for t in TT: by[t['site']].append(t)
    sites=sorted(by); n=len(sites); k=min(60,min(len(by[s]) for s in sites))
    tok=collections.Counter(a for t in TT for a in t['seq']); V=[a for a,c in tok.items() if c>=10]
    J=np.zeros((n,n))
    for _ in range(200):
        prof=[]
        for s in sites:
            idx=rnd.choice(len(by[s]),k,replace=False); c=collections.Counter(a for i in idx for a in by[s][i]['seq'])
            v=np.array([c[a] for a in V],float); prof.append(v/v.sum())
        for i,j in itertools.combinations(range(n),2): J[i,j]+=jsd(prof[i],prof[j])/200; J[j,i]=J[i,j]
    D=np.array([[hav(COORDS[a]['lat'],COORDS[a]['lon'],COORDS[b]['lat'],COORDS[b]['lon']) for b in sites] for a in sites])
    DC=np.array([[abs(COORDS[a]['dist_coast_harappan_km']-COORDS[b]['dist_coast_harappan_km']) for b in sites] for a in sites])
    P(f'\n=== IM77 {label}: {len(TT)} texts; sites {dict((s,len(by[s])) for s in sites)}; {k}-text subsamples; vocab {len(V)}')
    for vn,M in [('km',D),('coast diff',DC)]:
        obs,p=mantel_exact(J,M); P(f'   JSD ~ {vn:<10} rho={obs:+.3f} exact p={p:.3f} (120 perms)')
    for i,s in enumerate(sites): P('     '+f'{s:<14}'+' '.join(f'{J[i,j]:.3f}' for j in range(n)))
    # shared texts (>=3 signs) between sites
    by_text=collections.defaultdict(set)
    for t in TT:
        if len(t['seq'])>=3: by_text[tuple(t['seq'])].add(t['site'])
    pairs=collections.Counter()
    for s,ss in by_text.items():
        if len(ss)>=2:
            for a,b in itertools.combinations(sorted(ss),2): pairs[(a,b)]+=1
    P(f'   shared texts (>=3 signs) between sites: {sum(pairs.values())} links; '+'; '.join(f'{a}-{b}:{c} ({D[sites.index(a),sites.index(b)]:.0f} km)' for (a,b),c in pairs.most_common()))
out.close()
