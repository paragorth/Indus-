"""pe78 cycle 3: freeze, then compare with outside numbers.

1. Freeze (sha256) the PE Taylor signatures (b, a) of every key and of the pre-registered HERD run, before any
   outside data is touched in this cycle: data/pe78_frozen.json.
2. Outside A (Ur III livestock and people vs grain and rations, CDLI; never read for PE): is the PE HERD run's
   (b, a) nearer the Ur III LIVING key cloud than the GOODS cloud?  Same for every other PE key (is HERD special?).
3. Outside B (Tal-e Malyan Banesh bones, pe37 targets: 0.74 births per female-year, fused 0.90 / 0.70 / 0.38):
   the herd ABM run with bone-fixed biology (nuisance parameters random) on the HERD keys' real group structure
   predicts a (b, a) cloud.  Does PE HERD fall inside it better than random biology does (500 random-biology
   clouds)?  And better than the GRAIN office and other keys?
4. Planted control: a planted herd key made with bone biology, embedded in PE group structure, must land in the
   bone cloud (power), and an ALLOC key must not.
"""
import sys, os, json, time, math
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pe78_common as C
from pe78_cycle1 import real_sigs

HERD = {'M362', 'M367', 'M346', 'M006'}
GRAIN = {'M010', 'M106', 'M002', 'M243', 'M075', 'M081', 'M265', 'M266', 'M296', 'M112'}
BONE = dict(f=0.74, mj=0.15, m=0.35, o=0.0)


def ba(s):
    return np.array([s['b'], s['a']])


def cloud_score(x, cloud):
    """log density of x under a Gaussian fitted to the cloud (2-d)."""
    mu = cloud.mean(0); S = np.cov(cloud.T) + 1e-6 * np.eye(2)
    d = x - mu
    return float(-0.5 * d @ np.linalg.solve(S, d) - 0.5 * np.log(np.linalg.det(S)))


def living_cloud(sizes_list, rng, n, fixed=None):
    out = []
    for i in range(n):
        sz = sizes_list[int(rng.integers(len(sizes_list)))]
        th = dict(f=rng.uniform(0.4, 1.3), m=rng.uniform(0.03, 0.3), mj=rng.uniform(0.1, 0.6),
                  o=rng.uniform(0, 0.35), se=rng.uniform(0, 0.6), Y=int(rng.integers(1, 16)),
                  sw=rng.uniform(0.2, 1.5), sh=rng.uniform(0, 1.0),
                  mu0=math.exp(rng.uniform(math.log(2), math.log(300))), cls=int(rng.integers(0, 4)))
        if fixed:
            th.update(fixed)
        s, _ = C.sim_signature('LIVING', sz, rng, th)
        if s:
            out.append(ba(s))
    return np.array(out)


if __name__ == '__main__':
    t0 = time.time()
    rows = []
    E = C.pe_entries()
    S = real_sigs(E, np.random.default_rng(3), 4)
    G = C.groups(E)
    herd = [k for k in S if k in HERD]
    frozen = dict(note='pe78 frozen Taylor signatures of PE keys (groups = tablets with >= 2 entries of the key; '
                       'min 4 groups; damage-aware corpus). HERD = pe20 run, GRAIN = pe25 office.',
                  keys={k: {f: round(v, 4) for f, v in S[k].items()} for k in sorted(S)},
                  herd=herd, herd_mean_ba=np.mean([ba(S[k]) for k in herd], 0).round(4).tolist())
    h = C.sha(frozen)
    frozen['sha256'] = h
    json.dump(frozen, open(os.path.join(C.DATA, 'pe78_frozen.json'), 'w'), indent=1)
    print('frozen', h[:16], 'herd keys', herd)
    # ---- outside A: Ur III
    U = C.ur_entries()
    SU = real_sigs(U, np.random.default_rng(3), 5)
    lab = {e[1]: e[3] for e in U}
    L = np.array([ba(SU[k]) for k in SU if lab[k] == 'LIVING'])
    Gd = np.array([ba(SU[k]) for k in SU if lab[k] == 'GOODS'])
    llr = {k: cloud_score(ba(S[k]), L) - cloud_score(ba(S[k]), Gd) for k in S}
    hv = np.mean([llr[k] for k in herd]); gk = [k for k in S if k in GRAIN]
    gv = np.mean([llr[k] for k in gk]) if gk else float('nan')
    allv = np.array(list(llr.values()))
    rng = np.random.default_rng(5)
    null = np.array([allv[rng.choice(len(allv), len(herd), replace=False)].mean() for _ in range(5000)])
    pA = float((null >= hv).mean())
    # calibration of outside A on proto-cuneiform (independent of Ur III): PC living keys vs PC goods keys
    P = [e for e in C.pc_entries() if e[3]]
    SP = real_sigs(P, np.random.default_rng(3), 4)
    labp = {e[1]: e[3] for e in P}
    pcl = [cloud_score(ba(SP[k]), L) - cloud_score(ba(SP[k]), Gd) for k in SP if labp[k] == 'LIVING']
    pcg = [cloud_score(ba(SP[k]), L) - cloud_score(ba(SP[k]), Gd) for k in SP if labp[k] == 'GOODS']
    from pe78_cycle1 import auc
    aPC = auc(pcl, pcg)
    print('outside A: herd LLR %.2f grain %.2f all mean %.2f p %.3f ; PC calibration AUC %.2f' % (hv, gv, allv.mean(), pA, aPC))
    for k in sorted(llr, key=llr.get):
        print('   %-18s LLR(UR living/goods) %.2f b %.2f a %.2f' % (k, llr[k], S[k]['b'], S[k]['a']))
    rows.append('| PE-78.3.1 | Freeze (sha256 %s) PE Taylor signatures; OUTSIDE A: Ur III LIVESTOCK+PEOPLE vs GRAIN+RATION (b, a) clouds (CDLI, %d+%d keys); log-likelihood ratio living/goods for the pre-registered HERD run vs 5,000 random PE key sets; calibration = same ratio on proto-cuneiform keys of known kind | HERD (%s) LLR %.2f, GRAIN office %.2f, all PE keys %.2f; HERD vs random sets p %.3f. PC calibration AUC %.2f | %s |' % (
        h[:16], len(L), len(Gd), ','.join(herd), hv, gv, allv.mean(), pA, aPC,
        'outside ruler does not transfer to proto-cuneiform, so it cannot type PE' if aPC < 0.65 else ('HERD reads as living' if pA < 0.05 else 'HERD not distinguished')))
    # ---- outside B: bones
    hs = [[len(g) for _, g in G[k]] for k in herd]
    rng = np.random.default_rng(21)
    bone = living_cloud(hs, rng, 1500, BONE)
    x = np.mean([ba(S[k]) for k in herd], 0)
    sb = cloud_score(x, bone)
    rnd = []
    for r in range(300):
        rf = dict(f=rng.uniform(0.4, 1.3), mj=rng.uniform(0.1, 0.6), m=rng.uniform(0.03, 0.5), o=rng.uniform(0, 0.35))
        rnd.append(cloud_score(x, living_cloud(hs, rng, 80, rf)))
    rnd = np.array(rnd)
    pB = float((rnd >= sb).mean())
    others = {k: cloud_score(ba(S[k]), bone) for k in S}
    oth = np.array([v for k, v in others.items() if k not in HERD])
    pBk = float((oth >= np.mean([others[k] for k in herd])).mean())
    # planted: bone herd key vs alloc key on herd structure
    pl_in, pl_alloc = [], []
    for r in range(100):
        sz = hs[r % len(hs)]
        sL, _ = C.sim_signature('LIVING', sz, rng, dict(BONE, se=rng.uniform(0, 0.6), Y=int(rng.integers(1, 16)),
                                sw=rng.uniform(0.2, 1.5), sh=rng.uniform(0, 1.0), mu0=math.exp(rng.uniform(0.7, 5.7)), cls=int(rng.integers(0, 4))))
        sA, _ = C.sim_signature('ALLOC', sz, rng)
        if sL:
            pl_in.append(cloud_score(ba(sL), bone))
        if sA:
            pl_alloc.append(cloud_score(ba(sA), bone))
    aPl = auc(pl_in, pl_alloc)
    print('outside B: herd score in bone cloud %.2f; random-biology clouds p %.3f; other PE keys ranked >= herd %.3f; planted AUC %.2f' % (sb, pB, pBk, aPl))
    rows.append('| PE-78.3.2 | OUTSIDE B: Tal-e Malyan Banesh bones (pe37 targets; 0.74 births per female-year, first-year survival ~0.85, adult/cull mortality ~0.35) fix the herd ABM biology; 1,500 runs on the HERD keys\' real group sizes give a (b, a) cloud. Nulls: 300 random-biology clouds; every non-HERD PE key; planted bone-herd vs ALLOC keys (power) | HERD log-density in bone cloud %.2f; random biology fits as well or better in %.0f%% of 300 (p %.2f); non-HERD PE keys fit as well or better: %.0f%%; planted bone herds vs ALLOC AUC %.2f | %s |' % (
        sb, 100 * pB, pB, 100 * pBk, aPl,
        'bone biology adds nothing over random biology' if pB > 0.1 else 'bone biology fits better than random biology'))
    open(os.path.join(C.CK, 'c3_rows.txt'), 'w').write('\n'.join(rows) + '\n')
    json.dump(dict(llr=llr, pA=pA, aPC=aPC, sb=sb, pB=pB, pBk=pBk, aPl=aPl, herd=herd, sha=h), open(os.path.join(C.CK, 'c3_res.json'), 'w'))
    print('\n'.join(rows))
    print('done %.0fs' % (time.time() - t0))
