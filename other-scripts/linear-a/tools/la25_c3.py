#!/usr/bin/env python3
"""la25 cycle 3: one fire or many? Same-season test, split-half reliability.

(1) ABC model choice 'all LA archives burned in the same month' vs 'independent months' and its
    calibration: AUC of P(same) on planted same / different datasets (free and oracle base rates).
(2) Split-half reliability: Hagia Triada documents split at random into two pseudo-archives that
    certainly burned together. Posterior P(month equal) for HTa-HTb must exceed cross-site pairs
    if the clock reads anything.
"""
import json, sys
import numpy as np
from la25_common import *
from la25_c2 import sim2, bank2, post

NB = int(sys.argv[1]) if len(sys.argv) > 1 else 1_000_000
out = open(os.path.join(CK, 'c3.log'), 'w')
def P(*a):
    s = ' '.join(str(x) for x in a); print(s); out.write(s + '\n'); out.flush()


def auc(pos, neg):
    pos = np.asarray(pos); neg = np.asarray(neg)
    return float(((pos[:, None] > neg[None]).mean() + 0.5 * (pos[:, None] == neg[None]).mean()))


def pair_eq(B, idx, i, j):
    m = B['m'][idx]
    return float((m[:, i] == m[:, j]).mean())


if __name__ == '__main__':
    la = la_entries()
    A4 = ['HT', 'KH', 'ZA', 'TY']
    X = profile(la, A4, CORE); N = X.sum(1).tolist()
    res = {}
    # (1) model choice calibration
    rng0 = np.random.default_rng(21)
    beta0 = rng0.normal(0, 1.5, len(CORE)); beta0[-1] = 0
    for tag, kw in (('free', dict(N=N, cats=CORE, groups=[1] * 4, fixed={}, s_off=0.0)),
                    ('oracle', dict(N=N, cats=CORE, groups=[1] * 4, fixed={}, s_off=0.0, beta0=beta0))):
        B = bank2(60 + (tag == 'oracle'), n=NB, **kw)
        Pm, ps, idx = post(B, X)
        if tag == 'free':
            P(f'LA real: P(all four burned in one month) = {ps:.3f} (prior 0.50)')
            res['P_same_real'] = ps
            for i in range(4):
                for j in range(i + 1, 4):
                    P(f'   pair {A4[i]}-{A4[j]}: P(same month) {pair_eq(B, idx, i, j):.3f} (prior {0.5 + 0.5 / 12:.3f})')
        rng = np.random.default_rng(70 + (tag == 'oracle'))
        for strong in (False, True):
            r = sim2(rng, 20000, **kw)
            ok = (r['gam'] > 1.2) & (r['sig'] < 0.4) if strong else np.ones(20000, bool)
            si = np.where(ok & r['same'])[0][:150]; di = np.where(ok & ~r['same'])[0][:150]
            ps_s = [post(B, r['X'][i])[1] for i in si]; ps_d = [post(B, r['X'][i])[1] for i in di]
            a = auc(ps_s, ps_d)
            P(f'CALIBRATION {tag} base rates, strong={strong}: AUC of P(same) planted same vs different {a:.3f} (chance 0.5); '
              f'mean P(same) {np.mean(ps_s):.2f} vs {np.mean(ps_d):.2f}')
            res[f'auc_{tag}_{strong}'] = a
    # (2) split-half reliability on HT
    docs = sorted({e[1] for e in la if e[0] == 'HT'})
    rng = np.random.default_rng(80)
    hh, hk, hz, kz = [], [], [], []
    Bs = None
    for rep in range(20):
        perm = rng.permutation(docs); half = set(perm[: len(docs) // 2])
        la2 = [(('HTa' if e[1] in half else 'HTb') if e[0] == 'HT' else e[0],) + e[1:] for e in la]
        X4 = profile(la2, ['HTa', 'HTb', 'KH', 'ZA'], CORE)
        if Bs is None:
            Bs = bank2(90, n=NB, N=X4.sum(1).tolist(), cats=CORE, groups=[1] * 4, fixed={}, s_off=0.0)
        idx, _ = abc(Bs['S'], summ(X4[None])[0], 500, Bs['scale'])
        hh.append(pair_eq(Bs, idx, 0, 1)); hk.append(pair_eq(Bs, idx, 0, 2)); hz.append(pair_eq(Bs, idx, 0, 3)); kz.append(pair_eq(Bs, idx, 2, 3))
    P(f'SPLIT-HALF (20 random HT halves): P(equal month) HTa-HTb {np.mean(hh):.3f}, HTa-KH {np.mean(hk):.3f}, HTa-ZA {np.mean(hz):.3f}, KH-ZA {np.mean(kz):.3f} (prior 0.542)')
    P(f'   HTa-HTb above every cross-site pair in {np.mean([h > max(a, b) for h, a, b in zip(hh, hk, hz)]):.2f} of splits')
    res['split'] = [np.mean(hh), np.mean(hk), np.mean(hz), np.mean(kz)]
    # planted reliability: two halves of one planted archive vs another archive, LA size
    r = sim2(np.random.default_rng(91), 60000, N=X4.sum(1).tolist(), cats=CORE, groups=[1] * 4, fixed={}, s_off=0.0)
    ok = np.where((r['gam'] > 1.2) & (r['sig'] < 0.4) & ~r['same'] & (r['m'][:, 0] == r['m'][:, 1]) & (r['m'][:, 0] != r['m'][:, 2]))[0][:200]
    gap = []
    for i in ok:
        # force halves to share month by construction: resimulate is costly; use pairs where m0==m1 by chance
        if r['m'][i, 0] != r['m'][i, 1] or r['m'][i, 0] == r['m'][i, 2]: continue
        idx, _ = abc(Bs['S'], summ(r['X'][i][None])[0], 500, Bs['scale'])
        gap.append(pair_eq(Bs, idx, 0, 1) - pair_eq(Bs, idx, 0, 2))
    P(f'PLANTED reliability (strong clock, halves same month, third archive different): mean P(eq) gap {np.mean(gap) if gap else float("nan"):.3f}, n={len(gap)}')
    json.dump(res, open(os.path.join(CK, 'c3.json'), 'w'), default=float)
