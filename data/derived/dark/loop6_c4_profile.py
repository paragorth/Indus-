"""Cycle 4: the whole vocabulary at once. Does the distance between two sites' sign-frequency profiles (Jensen-Shannon
divergence, sample-size corrected) grow with geographic distance, coast-distance difference, or river change?
(a) 6 big sites (>= 50 texts): equal subsamples of 50 texts, 200 draws, EXACT permutation of site labels (720).
(b) all 20 sites, JSD minus its within-site bootstrap expectation, Mantel with 20,000 label permutations.
(c) seals only. (d) without MD + HP. Also: nearest-neighbour test - is each site's vocabulary closest to its geographic
neighbour more often than chance?"""
import sys, collections, itertools, math, numpy as np
sys.path.insert(0,'/home/user/Indus-/data/derived/dark'); from loop6_engine import *
from scipy.stats import rankdata

out=open(ROOT+'data/derived/dark/loop6_c4_profile.txt','w')
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
def profile(texts,V):
    c=collections.Counter(a for t in texts for a in t['seq']); v=np.array([c[a] for a in V],float); return v/v.sum()
def spear(a,b):
    ra=rankdata(a); rb=rankdata(b); return float(np.corrcoef(ra,rb)[0,1])
def mantel_exact(A,B):
    n=len(A); iu=np.triu_indices(n,1); a=A[iu]; obs=spear(a,B[iu]); vals=[]
    for p in itertools.permutations(range(n)):
        p=np.array(p); vals.append(spear(a,B[p][:,p][iu]))
    vals=np.array(vals); return obs,float((np.abs(vals)>=abs(obs)-1e-12).mean()),len(vals)
def mantel_mc(A,B,n_mc=20000,seed=4):
    n=len(A); iu=np.triu_indices(n,1); a=A[iu]; obs=spear(a,B[iu]); rnd=np.random.default_rng(seed); ge=0
    for _ in range(n_mc):
        p=rnd.permutation(n);
        if abs(spear(a,B[p][:,p][iu]))>=abs(obs)-1e-12: ge+=1
    return obs,(ge+1)/(n_mc+1),n_mc

def geo_mats(sites):
    D=np.array([[hav(COORDS[a]['lat'],COORDS[a]['lon'],COORDS[b]['lat'],COORDS[b]['lon']) for b in sites] for a in sites])
    DC=np.array([[abs(COORDS[a]['dist_coast_harappan_km']-COORDS[b]['dist_coast_harappan_km']) for b in sites] for a in sites])
    RIV=np.array([[0.0 if COORDS[a]['river']==COORDS[b]['river'] else 1.0 for b in sites] for a in sites])
    return D,DC,RIV

rnd=np.random.default_rng(11)
for label,types in [('all objects',None),('seals only',('SEAL',))]:
    T=load_texts('seq_raw',types=types); by=site_table(T,5)
    tok=collections.Counter(a for t in T for a in t['seq']); V=[a for a,n in tok.items() if n>=10]
    P(f'\n=== {label}: {len(T)} texts, {len(by)} sites, vocabulary {len(V)} signs (>=10 tokens)')
    # (a) big sites, equal subsamples, exact permutation
    big=sorted([s for s in by if len(by[s])>=50]); k=50
    if len(big)>=4:
        n=len(big); J=np.zeros((n,n))
        for _ in range(200):
            prof=[profile([by[s][i] for i in rnd.choice(len(by[s]),k,replace=False)],V) for s in big]
            for i,j in itertools.combinations(range(n),2): J[i,j]+=jsd(prof[i],prof[j])/200; J[j,i]=J[i,j]
        D,DC,RIV=geo_mats(big)
        P(f'  (a) {n} sites with >=50 texts {big}, {k}-text subsamples x200; exact permutation ({math.factorial(n)}):')
        for vn,M in [('great-circle km',D),('coast-dist diff',DC),('river change',RIV)]:
            obs,p,kk=mantel_exact(J,M); P(f'      JSD ~ {vn:<16} Mantel rho={obs:+.3f} exact p={p:.4f}')
        P('      JSD matrix (bits):');
        for i,s in enumerate(big): P('        '+f'{s:<14}'+' '.join(f'{J[i,j]:.3f}' for j in range(n)))
        if n>=5:
            keep=[i for i,s in enumerate(big) if s not in ('Mohenjo-daro','Harappa')]
            if len(keep)>=4:
                obs,p,kk=mantel_exact(J[keep][:,keep],D[keep][:,keep]); P(f'      without MD+HP ({len(keep)} sites): JSD ~ km rho={obs:+.3f} exact p={p:.4f}')
    # (b) all sites, bootstrap-corrected JSD
    sites=sorted(by); n=len(sites); prof={s:profile(by[s],V) for s in sites}
    # within-site expectation: split-half JSD scaled to the two sample sizes is messy; use the simpler correction:
    # JSD_corr(i,j) = JSD(i,j) - 0.5*(E_i + E_j) where E_i = expected JSD between two independent samples of size n_i from site i's own profile
    def self_jsd(s,reps=100):
        m=len(by[s]); vals=[]
        for _ in range(reps):
            a=rnd.integers(0,m,m); b=rnd.integers(0,m,m)
            vals.append(jsd(profile([by[s][i] for i in a],V),profile([by[s][i] for i in b],V)))
        return float(np.mean(vals))/2  # half: one sample vs the true profile
    E={s:self_jsd(s) for s in sites}
    J=np.zeros((n,n))
    for i,j in itertools.combinations(range(n),2):
        J[i,j]=J[j,i]=jsd(prof[sites[i]],prof[sites[j]])-(E[sites[i]]+E[sites[j]])
    D,DC,RIV=geo_mats(sites)
    P(f'  (b) all {n} sites, bootstrap-corrected JSD, Mantel 20,000 label permutations:')
    for vn,M in [('great-circle km',D),('coast-dist diff',DC),('river change',RIV)]:
        obs,p,kk=mantel_mc(J,M); P(f'      JSD ~ {vn:<16} rho={obs:+.3f} p={p:.4f}')
    keep=[i for i,s in enumerate(sites) if s not in ('Mohenjo-daro','Harappa')]
    obs,p,kk=mantel_mc(J[keep][:,keep],D[keep][:,keep]); P(f'      without MD+HP ({len(keep)} sites): JSD ~ km rho={obs:+.3f} p={p:.4f}')
    # nearest neighbour: vocabulary-nearest site == geographically nearest site?
    hits=0
    for i in range(n):
        jv=min((j for j in range(n) if j!=i),key=lambda j:J[i,j]); jg=min((j for j in range(n) if j!=i),key=lambda j:D[i,j]); hits+= jv==jg
    # null: permute labels
    null=[]
    for _ in range(5000):
        p=rnd.permutation(n); Dp=D[p][:,p]; h=0
        for i in range(n):
            jv=min((j for j in range(n) if j!=i),key=lambda j:J[i,j]); jg=min((j for j in range(n) if j!=i),key=lambda j:Dp[i,j]); h+= jv==jg
        null.append(h)
    null=np.array(null); P(f'      vocabulary-nearest == geographically-nearest for {hits} of {n} sites; null mean {null.mean():.2f}, P(>=) = {(null>=hits).mean():.3f}')
    # which site is each site's vocabulary neighbour?
    P('      vocabulary neighbours: '+'; '.join(f'{sites[i]}->{sites[min((j for j in range(n) if j!=i),key=lambda j:J[i,j])]}' for i in range(n)))
out.close()
