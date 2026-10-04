#!/usr/bin/env python3
"""la25 cycle 1: free-base-rate ABC clock.

Bank: 10^6 simulations per archive set (random agronomic calendar x destruction month per
archive x seasonal strength gamma x terroir noise sigma x free commodity base rates beta).
Outputs month posteriors for LA (HT, KH, ZA, TY) and LB (KN, PY, TH, MY); planted recovery
(same generator, held-out seed); commodity-shuffled null; random-calendar null.
"""
import json, sys
import numpy as np
from la25_common import *

NB = int(sys.argv[1]) if len(sys.argv) > 1 else 1_000_000
K = 500
out = open(os.path.join(CK, 'c1.log'), 'w')
def P(*a):
    s = ' '.join(str(x) for x in a); print(s); out.write(s + '\n'); out.flush()


def bank(seed, N, cats, mode='agro', n=NB, **kw):
    rng = np.random.default_rng(seed)
    S, M, SAME, G, SG = [], [], [], [], []
    for i in range(0, n, 100_000):
        r = simulate(rng, min(100_000, n - i), N, cats, mode=mode, **kw)
        S.append(summ(r['X']).astype(np.float32)); M.append(r['m'].astype(np.int8))
        SAME.append(r['same']); G.append(r['gam']); SG.append(r['sig'])
    return dict(S=np.concatenate(S), m=np.concatenate(M), same=np.concatenate(SAME),
                gam=np.concatenate(G), sig=np.concatenate(SG))


def infer(B, X, k=K):
    s = summ(X[None])[0]
    idx, sc = abc(B['S'], s, k)
    Pm = month_post(B['m'][idx].astype(int))
    return dict(P=Pm, same=float(B['same'][idx].mean()), gam=float(np.median(B['gam'][idx])),
                gam_q=np.quantile(B['gam'][idx], [0.1, 0.9]).tolist(), idx=idx)


def report(name, archs, res):
    P(f'{name}: P(same month)={res["same"]:.2f} gamma med {res["gam"]:.2f} [{res["gam_q"][0]:.2f},{res["gam_q"][1]:.2f}]')
    for a, nm in enumerate(archs):
        p = res['P'][a]
        P(f'   {nm}: MAP {MONTHS[int(p.argmax())]} p={p.max():.3f}  80% set [{fmt_months(cred_set(p))}] ({len(cred_set(p))} mo)  KL {kl_unif(p):.3f}  '
          + ' '.join(f'{x:.2f}' for x in p))


def planted(B, N, cats, seed, n=300, strong=False):
    rng = np.random.default_rng(seed)
    r = simulate(rng, 20000, N, cats)
    sel = np.arange(20000)
    if strong:
        sel = np.where((r['gam'] > 1.2) & (r['sig'] < 0.4))[0]
    sel = sel[:n]
    mass, near, cov = [], [], []
    for i in sel:
        res = infer(B, r['X'][i])
        for a in range(len(N)):
            t = r['m'][i, a]; p = res['P'][a]
            mass.append(p[t]); mm = int(p.argmax())
            near.append(min((mm - t) % 12, (t - mm) % 12) <= 1)
            cov.append(t in cred_set(p))
    return dict(mass=float(np.mean(mass)), near=float(np.mean(near)), cov80=float(np.mean(cov)), n=len(sel))


if __name__ == '__main__':
    la = la_entries(); lb = lb_entries()
    LA_A = ['HT', 'KH', 'ZA', 'TY']; LB_A = ['KN', 'PY', 'TH', 'MY']
    XLA = profile(la, LA_A, CORE); XLB = profile(lb, LB_A, CORE); XLBe = profile(lb, LB_A, EXT)
    NLA = XLA.sum(1).tolist(); NLB = XLB.sum(1).tolist()
    P('LA core doc-level profile', LA_A, CORE, XLA.tolist())
    P('LB core', LB_A, XLB.tolist()); P('LB ext', EXT, XLBe.tolist())
    res = {}
    BLA = bank(1, NLA, CORE)
    r = infer(BLA, XLA); report('LA agro', LA_A, r); res['LA'] = r['P'].tolist(); res['LA_same'] = r['same']
    # planted (same generator, different seed)
    for strong in (False, True):
        pl = planted(BLA, NLA, CORE, 99 + strong, strong=strong)
        P(f'PLANTED LA-size strong={strong}: mass on true month {pl["mass"]:.3f} (chance 0.083), MAP within +-1 {pl["near"]:.3f} (chance 0.25), 80% cov {pl["cov80"]:.2f}, n={pl["n"]}')
        res[f'plant_{strong}'] = pl
    # commodity-shuffled null
    rng = np.random.default_rng(7)
    kl_real = [kl_unif(p) for p in r['P']]
    kls = []
    for s in range(200):
        Xs = np.array([rng.permutation(x) for x in XLA])
        rs = infer(BLA, Xs); kls.append([kl_unif(p) for p in rs['P']])
    kls = np.array(kls)
    for a, nm in enumerate(LA_A):
        P(f'SHUFFLE null {nm}: real KL {kl_real[a]:.3f}, shuffled median {np.median(kls[:, a]):.3f}, P(shuf>=real) {(kls[:, a] >= kl_real[a]).mean():.3f}')
    tot = kls.sum(1); P(f'SHUFFLE null summed KL: real {sum(kl_real):.3f} P {(tot >= sum(kl_real)).mean():.3f}')
    res['shuffle_P'] = [(kls[:, a] >= kl_real[a]).mean() for a in range(4)]
    # random calendars: ABC model choice agro vs random (equal prior; pooled bank nearest-k)
    BLR = bank(2, NLA, CORE, mode='random')
    Sall = np.concatenate([BLA['S'], BLR['S']])
    idx, _ = abc(Sall, summ(XLA[None])[0], 2 * K)
    pa = float((idx < len(BLA['S'])).mean())
    P(f'MODEL CHOICE LA agro vs random calendar: P(agro)={pa:.3f} (prior 0.5)')
    res['LA_Pagro'] = pa
    # LB positive control
    BLB = bank(3, NLB, CORE)
    rb = infer(BLB, XLB); report('LB core', LB_A, rb)
    spring = [1, 2, 3, 4]
    for a in (0, 1):
        P(f'   LB control {LB_A[a]}: posterior mass in argued season Feb-May {rb["P"][a][spring].sum():.3f} (prior 0.333)')
    res['LB'] = rb['P'].tolist()
    BLBe = bank(4, XLBe.sum(1).tolist(), EXT)
    rbe = infer(BLBe, XLBe); report('LB ext', LB_A, rbe)
    for a in (0, 1):
        P(f'   LB ext control {LB_A[a]}: mass Feb-May {rbe["P"][a][spring].sum():.3f} (prior 0.333)')
    res['LBe'] = rbe['P'].tolist()
    ple = planted(BLBe, XLBe.sum(1).tolist(), EXT, 98, strong=True)
    P(f'PLANTED LB-ext-size strong: mass {ple["mass"]:.3f}, near {ple["near"]:.3f}, cov {ple["cov80"]:.2f}')
    plb = planted(BLB, NLB, CORE, 97, strong=True)
    P(f'PLANTED LB-core-size strong: mass {plb["mass"]:.3f}, near {plb["near"]:.3f}, cov {plb["cov80"]:.2f}')
    json.dump(res, open(os.path.join(CK, 'c1.json'), 'w'), default=float)
