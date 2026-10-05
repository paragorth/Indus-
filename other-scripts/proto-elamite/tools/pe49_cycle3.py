"""pe49 cycle 3: read the job descriptions of the hands.

Takes a saved consensus matrix (cycle 1 or 2), cuts it into k hands and asks:
  1. which content signs each hand over-uses (log-odds vs the rest; p from 1,000
     permutations of hand labels within length x dominant-unit strata; BH FDR);
  2. does a hand's profile replicate across a random split of its tablets
     (20 splits; null = strata-permuted labels);
  3. system vocabulary: frequent signs (>= 10% of tablets) with the same rate in
     every hand (chi-square p > 0.5);
  4. confounds: NMI of hands with publication series, vs strata permutation;
  5. Ur III / planted: do recovered-hand profiles match the true scribes' profiles
     (correlation of log-odds with the majority scribe's, vs shuffled mapping).
Usage: python3 pe49_cycle3.py {ur3|plant|pe} A_file k
"""
import os, sys, json, collections
os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np
from scipy.stats import chi2_contingency
import pe49_common as P
from pe49_cycle1 import plant


def logodds(Xc, m):
    a = Xc[m].sum(0) + 0.5
    b = m.sum() - Xc[m].sum(0) + 0.5
    c = Xc[~m].sum(0) + 0.5
    d = (~m).sum() - Xc[~m].sum(0) + 0.5
    return np.log(a / b) - np.log(c / d)


def main(which, Afile, k):
    rng = np.random.default_rng(31)
    if which == 'ur3':
        T, ch = P.load_ur3()
        truth = np.unique([t['group'] for t in T], return_inverse=True)[1]
    elif which == 'plant':
        T, truth, _ = plant(P.load_pe(), np.random.default_rng(5))
    else:
        T = P.load_pe()
        truth = None
    A = np.load(os.path.join(P.CK, Afile))
    lab = P.spectral_cut(A, k)
    Xc, voc = P.content_matrix(T)
    C, strata = P.coarse_covariates(T)
    out = {'which': which, 'A': Afile, 'k': k, 'sizes': np.bincount(lab).tolist()}
    # 1. enrichment with permutation p
    LO = np.array([logodds(Xc, lab == h) for h in range(k)])
    NP = 1000
    ge = np.zeros_like(LO)
    for _ in range(NP):
        pl = P.strata_perm(lab, strata, rng)
        LOp = np.array([logodds(Xc, pl == h) for h in range(k)])
        ge += LOp >= LO
    pv = (ge + 1) / (NP + 1)
    flat = np.sort(pv.ravel())
    m = len(flat)
    thr = 0.0
    for i, p in enumerate(flat):
        if p <= 0.10 * (i + 1) / m:
            thr = p
    sig = pv <= thr
    hands = []
    for h in range(k):
        idx = np.where(sig[h])[0]
        idx = idx[np.argsort(-LO[h, idx])]
        hands.append({'hand': h, 'n': int((lab == h).sum()),
                      'enriched': [(voc[i], round(float(LO[h, i]), 2), int(Xc[lab == h, i].sum()), float(pv[h, i])) for i in idx[:12]]})
    out['hands'] = hands
    out['n_sig'] = int(sig.sum())
    # null count of significant cells: same FDR on one permuted labelling
    pl = P.strata_perm(lab, strata, rng)
    LOq = np.array([logodds(Xc, pl == h) for h in range(k)])
    ge2 = np.zeros_like(LOq)
    for _ in range(300):
        pl2 = P.strata_perm(pl, strata, rng)
        ge2 += np.array([logodds(Xc, pl2 == h) for h in range(k)]) >= LOq
    pv2 = np.sort(((ge2 + 1) / 301).ravel())
    thr2 = 0.0
    for i, p in enumerate(pv2):
        if p <= 0.10 * (i + 1) / len(pv2):
            thr2 = p
    out['n_sig_null'] = int((pv2 <= thr2).sum()) if thr2 > 0 else 0
    # 2. split-half replication of hand profiles
    common_ = Xc.mean(0) >= 0.03

    def split_r(lb):
        rs = []
        for s in range(20):
            half = rng.random(len(lb)) < 0.5
            r = []
            for h in np.unique(lb):
                m1, m2 = (lb == h) & half, (lb == h) & ~half
                if m1.sum() < 8 or m2.sum() < 8:
                    continue
                a = logodds(Xc[half], (lb == h)[half])[common_]
                b = logodds(Xc[~half], (lb == h)[~half])[common_]
                r.append(np.corrcoef(a, b)[0, 1])
            rs.append(np.mean(r))
        return float(np.mean(rs))
    out['split_r'] = split_r(lab)
    out['split_r_null'] = [split_r(P.strata_perm(lab, strata, rng)) for _ in range(10)]
    # 3. system vocabulary
    sysv, offv = [], []
    for i, v in enumerate(voc):
        if Xc[:, i].mean() < 0.10:
            continue
        tab = np.array([[Xc[lab == h, i].sum(), (lab == h).sum() - Xc[lab == h, i].sum()] for h in range(k)])
        tab = tab[tab.sum(1) > 0]
        p = chi2_contingency(tab + 0.5)[1]
        (sysv if p > 0.5 else offv if p < 0.001 else []).append((v, round(float(Xc[:, i].mean()), 2), float(p)))
    out['system_vocab'] = sorted(sysv, key=lambda x: -x[1])
    out['office_vocab'] = sorted(offv, key=lambda x: x[2])
    # 4. confounds
    if which != 'ur3':
        pub = []
        for t in T:
            d = t.get('desig', '')
            pub.append(d.split(',')[0] if d else '?')
        out['nmi_pub'] = P.nmi(pub, lab)
        out['nmi_pub_null'] = float(np.mean([P.nmi(pub, P.strata_perm(lab, strata, rng)) for _ in range(100)]))
        out['nmi_strata'] = P.nmi(strata, lab)
        out['pub_by_hand'] = [collections.Counter(np.array(pub)[lab == h]).most_common(3) for h in range(k)]
    # 5. truth comparison
    if truth is not None:
        truth = np.asarray(truth)
        out['nmi_truth'] = P.nmi(truth, lab)
        rs, rnull = [], []
        TL = {g: logodds(Xc, truth == g)[common_] for g in np.unique(truth)}
        for h in range(k):
            m = lab == h
            if m.sum() < 10:
                continue
            g = collections.Counter(truth[m]).most_common(1)[0][0]
            a = logodds(Xc, m)[common_]
            rs.append(np.corrcoef(a, TL[g])[0, 1])
            others = [x for x in TL if x != g]
            rnull.append(np.mean([np.corrcoef(a, TL[x])[0, 1] for x in others]))
        out['profile_r_majority'] = float(np.mean(rs))
        out['profile_r_other'] = float(np.mean(rnull))
        out['profile_pairs'] = [(round(float(a), 3), round(float(b), 3)) for a, b in zip(rs, rnull)]
    print(json.dumps({a: b for a, b in out.items() if a not in ('hands', 'system_vocab', 'office_vocab', 'pub_by_hand')}, default=float), flush=True)
    for hnd in hands:
        print(hnd['hand'], hnd['n'], hnd['enriched'][:8], flush=True)
    print('system vocab', out['system_vocab'][:20])
    print('office vocab', out['office_vocab'][:20])
    json.dump(out, open(os.path.join(P.CK, f'c3_{which}_{Afile[:-6]}_k{k}.json'), 'w'), default=float)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]))
