"""Cycle 3: the shared-text network. Pairs of sites that carry the same text (>= 3 signs, exact match; also the S324
seal -> sealing subset): does the number of shared texts between two sites fall with geographic distance, with coast
distance difference, or follow river order? Mantel-type test: Spearman between the pairwise shared-text matrix
(normalised by expected sharing n_i * n_j) and the pairwise distance matrix; null = site labels permuted (Mantel permutation,
20,000x), so the unit is the site. Also: same-river vs different-river pairs. Power by simulation."""
import sys, collections, itertools, numpy as np
sys.path.insert(0,'/home/user/Indus-/data/derived/dark'); from loop6_engine import *

out=open(ROOT+'data/derived/dark/loop6_c3_network.txt','w')
def P(*a):
    print(*a); print(*a,file=out)

def hav(a,b,c,d):
    import math
    R=6371.0; p1,p2=math.radians(a),math.radians(c); dp=p2-p1; dl=math.radians(d-b)
    h=math.sin(dp/2)**2+math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return 2*R*math.asin(math.sqrt(h))

def mantel(A,B,n_mc=20000,seed=3):
    """Spearman between upper triangles of A and B; null permutes site labels of B."""
    n=len(A); iu=np.triu_indices(n,1)
    a=A[iu]; obs=spearmanr(a,B[iu]).statistic
    rnd=np.random.default_rng(seed); ge=0
    for _ in range(n_mc):
        p=rnd.permutation(n); v=spearmanr(a,B[p][:,p][iu]).statistic
        if abs(v)>=abs(obs)-1e-12: ge+=1
    return obs,(ge+1)/(n_mc+1)

for level in ['seq_raw','seq_strong']:
    T=load_texts(level,minlen=3)
    # texts here are site+text deduplicated, so each (site,text) counts once
    by_text=collections.defaultdict(set)
    for t in T: by_text[tuple(t['seq'])].add(t['site'])
    shared=[(s,sites) for s,sites in by_text.items() if len(sites)>=2]
    sites=sorted({t['site'] for t in T}); idx={s:i for i,s in enumerate(sites)}; n=len(sites)
    N=np.array([sum(1 for t in T if t['site']==s) for s in sites],float)
    S=np.zeros((n,n))
    for s,ss in shared:
        for a,b in itertools.combinations(sorted(ss),2): S[idx[a],idx[b]]+=1; S[idx[b],idx[a]]+=1
    P(f'\n=== {level}: {len(T)} texts (>=3 signs) at {n} sites; {len(shared)} texts shared by >=2 sites; {int(S.sum()/2)} site-pair links')
    pairs=collections.Counter()
    for s,ss in shared:
        for a,b in itertools.combinations(sorted(ss),2): pairs[(a,b)]+=1
    for (a,b),k in pairs.most_common(12): P(f'   {a} - {b}: {k} shared texts; {hav(COORDS[a]["lat"],COORDS[a]["lon"],COORDS[b]["lat"],COORDS[b]["lon"]):.0f} km; rivers {COORDS[a]["river"]}/{COORDS[b]["river"]}')
    # distance matrices
    D=np.array([[hav(COORDS[a]['lat'],COORDS[a]['lon'],COORDS[b]['lat'],COORDS[b]['lon']) for b in sites] for a in sites])
    DC=np.array([[abs(COORDS[a]['dist_coast_harappan_km']-COORDS[b]['dist_coast_harappan_km']) for b in sites] for a in sites])
    RIV=np.array([[0.0 if COORDS[a]['river']==COORDS[b]['river'] else 1.0 for b in sites] for a in sites])
    # expected sharing under independence ~ n_i n_j ; normalised sharing
    E=np.outer(N,N); Sn=S/E
    # restrict to sites with >=1 link (otherwise rows of zeros): report both
    for name,M in [('all sites',None),('sites with >=1 link',S.sum(1)>0)]:
        keep=np.arange(n) if M is None else np.where(M)[0]
        if len(keep)<4: continue
        Sk=Sn[keep][:,keep]
        P(f'  [{name}: {len(keep)} sites]')
        for vname,V in [('great-circle distance',D),('coast-distance difference',DC),('different river system',RIV)]:
            Vk=V[keep][:,keep]; obs,p=mantel(Sk,Vk)
            P(f'     normalised sharing ~ {vname:<26} Mantel rho={obs:+.3f} p={p:.4f}')
        # raw counts too
        obs,p=mantel(S[keep][:,keep],D[keep][:,keep]); P(f'     raw shared count ~ great-circle distance        Mantel rho={obs:+.3f} p={p:.4f}')
    # without MD and HP
    keep=np.array([i for i,s in enumerate(sites) if s not in ('Mohenjo-daro','Harappa')])
    if (S[keep][:,keep].sum())>0:
        obs,p=mantel(Sn[keep][:,keep],D[keep][:,keep]); P(f'  without MD+HP ({len(keep)} sites, {int(S[keep][:,keep].sum()/2)} links): normalised sharing ~ distance rho={obs:+.3f} p={p:.4f}')
    else: P('  without MD+HP: no links remain')
    # link-level view: are linked pairs closer than unlinked pairs? (site-permutation null)
    iu=np.triu_indices(n,1); linked=S[iu]>0
    d_link=D[iu][linked].mean(); d_un=D[iu][~linked].mean()
    rnd=np.random.default_rng(5); ge=0
    for _ in range(20000):
        p=rnd.permutation(n); Dp=D[p][:,p][iu]
        if Dp[linked].mean()-Dp[~linked].mean() <= d_link-d_un: ge+=1
    P(f'  linked pairs mean distance {d_link:.0f} km vs unlinked {d_un:.0f} km ({linked.sum()} linked of {len(linked)} pairs); site-perm P(closer)={(ge+1)/20001:.4f}')
    # same river?
    same_link=(RIV[iu][linked]==0).mean(); same_un=(RIV[iu][~linked]==0).mean()
    P(f'  same-river share: linked {same_link:.2f} vs unlinked {same_un:.2f}')
    if level=='seq_raw':
        # power: simulate sharing where link probability decays with distance (halves every 300 km), same number of links
        L=int(linked.sum()); hits=0; rnd=np.random.default_rng(9)
        for _ in range(200):
            w=np.exp(-D[iu]/300.0)*(E[iu]); w=w/w.sum()
            pick=rnd.choice(len(w),size=L,replace=False,p=w); Sim=np.zeros((n,n)); Sim[iu[0][pick],iu[1][pick]]=1; Sim=Sim+Sim.T
            obs,p=mantel(Sim/E,D,n_mc=2000,seed=int(rnd.integers(1e9))); hits+=p<0.05
        P(f'  power (links placed with probability ~ n_i n_j exp(-d/300 km), {L} links, alpha 0.05): {hits/200:.2f}')
        # power for a strong effect: links only within 400 km
        hits=0
        for _ in range(200):
            w=(D[iu]<400)*(E[iu]); w=w/w.sum()
            pick=rnd.choice(len(w),size=L,replace=False,p=w); Sim=np.zeros((n,n)); Sim[iu[0][pick],iu[1][pick]]=1; Sim=Sim+Sim.T
            obs,p=mantel(Sim/E,D,n_mc=2000,seed=int(rnd.integers(1e9))); hits+=p<0.05
        P(f'  power (links only within 400 km): {hits/200:.2f}')

# S324 subset: seal -> sealing flow (TAG texts matching SEAL texts at another site)
T=load_texts('seq_raw',minlen=3)
seals=collections.defaultdict(set); tags=[]
for t in T:
    if t['type'].startswith('SEAL'): seals[tuple(t['seq'])].add(t['site'])
    if t['type'].startswith('TAG'): tags.append(t)
P('\nS324 flow pairs (sealing site <- seal site):')
flow=collections.Counter()
for t in tags:
    for s in seals.get(tuple(t['seq']),()):
        if s!=t['site']: flow[(t['site'],s)]+=1
for (a,b),k in flow.most_common(): P(f'   {a} <- {b}: {k}; {hav(COORDS[a]["lat"],COORDS[a]["lon"],COORDS[b]["lat"],COORDS[b]["lon"]):.0f} km; {COORDS[a]["river"]}/{COORDS[b]["river"]}; coast dist {COORDS[a]["dist_coast_harappan_km"]:.0f} <- {COORDS[b]["dist_coast_harappan_km"]:.0f}')
down=sum(k for (a,b),k in flow.items() if COORDS[a]['dist_coast_harappan_km']<COORDS[b]['dist_coast_harappan_km'])
P(f'  flows towards the coast (sealing nearer the sea than the seal): {down} of {sum(flow.values())} sealings, {sum(1 for (a,b) in flow if COORDS[a]["dist_coast_harappan_km"]<COORDS[b]["dist_coast_harappan_km"])} of {len(flow)} site pairs')
out.close()
