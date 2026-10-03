"""Cycle 1: every sign with >= 30 tokens; per-site share of texts containing it vs 7 geographic variables + river class.
Site-label permutation null; BH-FDR over all arrows; re-test with Mohenjo-daro and Harappa removed; power."""
import sys, collections, numpy as np
sys.path.insert(0,'/home/user/Indus-/data/derived/dark'); from loop6_engine import *

out=open(ROOT+'data/derived/dark/loop6_c1_signs.txt','w')
def P(*a):
    print(*a); print(*a,file=out)

for level in ['seq_raw','seq_strong','seq_all']:
    T=load_texts(level); by=site_table(T,5)
    tok=collections.Counter(a for t in T for a in t['seq'])
    signs=[a for a,n in tok.items() if n>=30]
    feats={f'W{a}':(lambda t,a=a: a in t['seq']) for a in signs}
    P(f'\n=== level {level}: {len(T)} distinct site+text pairs, {len(by)} sites with >=5 texts, {len(signs)} signs >=30 tokens, '
      f'{len(signs)*(len(GEOVARS)+1)} arrows')
    res=run_arrows(by,feats)
    res.sort(key=lambda r:r['p'])
    P(f'arrows fired {len(res)}; min p {res[0]["p"]:.4f}; min q {min(r["q"] for r in res):.3f}; '
      f'arrows with q<0.10: {sum(r["q"]<0.10 for r in res)}; p<0.01: {sum(r["p"]<0.01 for r in res)} (expected {0.01*len(res):.1f})')
    P('top 15 arrows:')
    for r in res[:15]: P('  '+fmt(r))
    # re-test the top 15 without MD/HP
    P('re-test of top 15 without Mohenjo-daro and Harappa:')
    top=res[:15]
    res2=run_arrows(by,{r['feature']:feats[r['feature']] for r in top},drop=('Mohenjo-daro','Harappa'))
    d2={(r['feature'],r['var']):r for r in res2}
    for r in top:
        r2=d2.get((r['feature'],r['var']))
        if r2: P(f"  {r['feature']:>8} ~ {r['var']:<24} full rho={r['stat']:+.3f} p={r['p']:.4f} | drop MD+HP rho={r2['stat']:+.3f} p={r2['p']:.4f}")
    # top hits: per-site rates for inspection
    for r in res[:3]:
        a=int(r['feature'][1:]); rates=site_rates(by,feats[r['feature']])
        P(f"  per-site rates for {r['feature']} (site, n, rate, {r['var']}):")
        for s in sorted(by,key=lambda s:COORDS[s][r['var']] if r['var']!='river' else 0):
            P(f"     {s:<22} n={len(by[s]):>4} rate={rates[s]:.3f} {r['var']}={COORDS[s][r['var']]}")
    if level=='seq_raw':
        P('power (site-level Spearman, alpha=0.05, binary feature rising linearly from 10% to 30% across the variable range):')
        for g in ['dist_coast_harappan_km','dist_md_km','lat']:
            P(f'   {g}: all sites {power_sim(by,g):.2f}; without MD+HP {power_sim(by,g,drop=("Mohenjo-daro","Harappa")):.2f}; 10%->50%: {power_sim(by,g,top=0.5):.2f}')
out.close()
