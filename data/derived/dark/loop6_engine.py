"""Loop 6 shared engine: site-level geographic gradient tests with site-label permutation nulls.
Unit = SITE (texts deduplicated by site+text). Statistic = Spearman rho between per-site feature rate and a geographic
variable; null = geographic values permuted among sites (exact enumeration when n <= 9, else Monte Carlo 20,000x).
Categorical variables (river system): Kruskal-Wallis H, same permutation null. BH-FDR over all arrows fired.
"""
import json, csv, math, random, itertools, collections
import numpy as np
from scipy.stats import spearmanr, kruskal, rankdata

ROOT='/home/user/Indus-/'
COORDS={r['site']:r for r in csv.DictReader(open(ROOT+'data/derived/dark/site_coords.csv'))}
GEOVARS=['lat','lon','dist_coast_harappan_km','dist_coast_modern_km','dist_md_km','dist_hp_km','elev_m']
for s,r in COORDS.items():
    for k in GEOVARS: r[k]=float(r[k])

def load_texts(level='seq_raw', types=None, minlen=1):
    C=json.load(open(ROOT+'data/derived/merged-corpus-canonical.json'))
    seen=set(); out=[]
    for r in C:
        s=r.get(level) or []
        if len(s)<minlen or r['site'] in ('Unknown','Ur') or r['site'] not in COORDS: continue
        if types and not any(r['type'].startswith(t) for t in types): continue
        key=(r['site'],tuple(s))
        if key in seen: continue
        seen.add(key); out.append(dict(site=r['site'],seq=list(s),type=r['type'],rec=r))
    return out

def site_table(texts, min_texts=5):
    by=collections.defaultdict(list)
    for t in texts: by[t['site']].append(t)
    return {s:v for s,v in by.items() if len(v)>=min_texts}

def site_rates(by, feature):
    """feature: function(text)->bool/number. Returns dict site -> mean."""
    return {s:float(np.mean([feature(t) for t in v])) for s,v in by.items()}

_PERM_CACHE={}
def perm_matrix(n, n_mc=20000, seed=6):
    """Rows = permutations of range(n): exact (all n!) when n<=8, else n_mc Monte Carlo permutations."""
    key=(n,n_mc)
    if key not in _PERM_CACHE:
        if n<=8: M=np.array(list(itertools.permutations(range(n))),dtype=int)
        else:
            rnd=np.random.default_rng(seed); M=np.array([rnd.permutation(n) for _ in range(n_mc)])
        _PERM_CACHE[key]=M
    return _PERM_CACHE[key]

def _zrank(v):
    r=rankdata(v); r=r-r.mean(); s=np.sqrt((r**2).sum()); return r/s if s>0 else r

def exact_or_mc_p(x, y, stat, n_mc=20000, seed=6):
    """Permutation p-value (two-sided) for stat(x,y) with y permuted among sites. Exact enumeration if n<=8, else
    Monte Carlo. Spearman is vectorised (rho = dot of centred, normalised ranks)."""
    x=np.asarray(x,float); y=np.asarray(y,float); n=len(x)
    if stat is rho:
        if np.std(x)==0 or np.std(y)==0: return float('nan'),float('nan'),0
        rx=_zrank(x); ry=_zrank(y); obs=float(rx@ry)
        M=perm_matrix(n,n_mc,seed); vals=ry[M]@rx
        k=len(vals); ge=int((np.abs(vals)>=abs(obs)-1e-12).sum())
        return obs, (ge/k if n<=8 else (ge+1)/(k+1)), k
    if stat is KW:
        labs=sorted(set(y.tolist()));
        if len(labs)<2: return float('nan'),float('nan'),0
        G=np.array([[1.0 if yi==l else 0.0 for l in labs] for yi in y]); ng=G.sum(0)
        if (ng<2).any(): return float('nan'),float('nan'),0
        r=rankdata(x); N=n
        def H(R): S=R@G; return 12.0/(N*(N+1))*((S**2)/ng).sum(-1)-3*(N+1)
        obs=float(H(r)); M=perm_matrix(n,n_mc,seed); vals=H(r[M])
        k=len(vals); ge=int((vals>=obs-1e-12).sum())
        return obs, (ge/k if n<=8 else (ge+1)/(k+1)), k
    obs=stat(x,y)
    if np.isnan(obs): return obs, float('nan'), 0
    M=perm_matrix(n,min(n_mc,5000),seed); vals=[stat(x,y[p]) for p in M]
    k=len(vals); ge=sum(1 for v in vals if abs(v)>=abs(obs)-1e-12)
    return obs, (ge/k if n<=8 else (ge+1)/(k+1)), k

def rho(x,y):
    if np.std(x)==0 or np.std(y)==0: return float('nan')
    return spearmanr(x,y).statistic

def KW(x,y):
    g=collections.defaultdict(list)
    for xi,yi in zip(x,y): g[yi].append(xi)
    if len(g)<2 or any(len(v)<2 for v in g.values()): return float('nan')
    try: return kruskal(*g.values()).statistic
    except ValueError: return float('nan')
def kw_stat(_): return KW

def bh(pvals):
    p=np.asarray(pvals,float); n=len(p); order=np.argsort(p); q=np.empty(n); prev=1.0
    for rank,i in enumerate(order[::-1]):
        k=n-rank; prev=min(prev, p[i]*n/k); q[i]=prev
    return q

def run_arrows(by, features, geovars=GEOVARS, categorical=('river',), drop=()):
    """features: dict name -> function(text). Returns list of result dicts (one per arrow)."""
    sites=[s for s in sorted(by) if s not in drop]
    res=[]
    for fname,f in features.items():
        rates=site_rates({s:by[s] for s in sites}, f)
        x=[rates[s] for s in sites]
        if np.std(x)==0: continue
        for g in geovars:
            y=[COORDS[s][g] for s in sites]
            obs,p,k=exact_or_mc_p(x,y,rho)
            res.append(dict(feature=fname,var=g,n_sites=len(sites),stat=obs,p=p,perms=k))
        for c in categorical:
            labs=[COORDS[s][c] for s in sites]
            codes={l:i for i,l in enumerate(sorted(set(labs)))}
            y=[codes[l] for l in labs]
            obs,p,k=exact_or_mc_p(x,y,kw_stat(None))
            res.append(dict(feature=fname,var=c,n_sites=len(sites),stat=obs,p=p,perms=k))
    ps=[r['p'] for r in res]
    for r,q in zip(res,bh(ps)): r['q']=q
    return res

def power_sim(by, geovar, base=0.10, top=0.30, n_sim=400, alpha=0.05, drop=(), seed=1):
    """Power of the site-level Spearman test for a true linear gradient in a binary feature from `base` at the
    minimum of geovar to `top` at its maximum, with the observed per-site text counts (binomial sampling)."""
    sites=[s for s in sorted(by) if s not in drop]
    ns=np.array([len(by[s]) for s in sites]); y=np.array([COORDS[s][geovar] for s in sites])
    z=(y-y.min())/(y.max()-y.min()); ptrue=base+(top-base)*z
    rnd=np.random.default_rng(seed); hits=0
    for _ in range(n_sim):
        x=rnd.binomial(ns,ptrue)/ns
        if np.std(x)==0: continue
        obs,p,k=exact_or_mc_p(x,y,rho,n_mc=2000,seed=int(rnd.integers(1e9)))
        hits+= p<alpha
    return hits/n_sim

def fmt(r): return f"{r['feature']:>28} ~ {r['var']:<24} n={r['n_sites']:>2} stat={r['stat']:+.3f} p={r['p']:.4f} q={r['q']:.3f}"
