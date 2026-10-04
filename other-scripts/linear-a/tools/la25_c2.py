#!/usr/bin/env python3
"""la25 cycle 2: anchor the clock.

Cycle 1 showed the free-base-rate clock is blind (base rates absorb season). Two anchors:
 (a) ORACLE: commodity base rates known (prior sd 0.1 around the truth). Upper bound on power at
     LA and LB size. If this fails, no commodity-profile clock can work on these archives.
 (b) TRANSFER: Linear B archives with an argued season (KN, PY: Feb-May) train the shared base
     rates and calendar; LA base rates = LB + offset ~ N(0, s). Leave-one-out LB control:
     KN fixed -> infer PY, and PY fixed -> infer KN. Then LA months.
"""
import json, sys
import numpy as np
from la25_common import *

NB = int(sys.argv[1]) if len(sys.argv) > 1 else 1_000_000
K = 500
out = open(os.path.join(CK, 'c2.log'), 'w')
def P(*a):
    s = ' '.join(str(x) for x in a); print(s); out.write(s + '\n'); out.flush()

SPRING = [1, 2, 3, 4]


def sim2(rng, n, N, cats, groups, fixed, s_off=0.5, mode='agro', beta0=None, beta_sd0=0.1,
         same_frac=0.5, gamma_lo=0.0):
    """groups[a] = 0 (LB) or 1 (LA); fixed = {archive index: allowed months}.
    Free archives share one month with prob same_frac."""
    A, C = len(N), len(cats)
    h, tau, fl = draw_calendar(rng, n, cats, mode)
    oth = cats.index('OTHER')
    if beta0 is None:
        b = rng.normal(0, 1.5, (n, C))
    else:
        b = beta0[None] + rng.normal(0, beta_sd0, (n, C))
    off = rng.normal(0, s_off, (n, C))
    B = np.stack([b + (off if g == 1 else 0) for g in groups], 1)   # n, A, C
    B[:, :, oth] = 0
    m = rng.integers(0, 12, (n, A))
    free = [a for a in range(A) if a not in fixed]
    same = rng.random(n) < same_frac
    if free:
        m[np.ix_(same, free)] = m[same][:, free[:1]]
    for a, allowed in fixed.items():
        m[:, a] = rng.choice(allowed, n)
    gam = rng.uniform(gamma_lo, 2, n); sig = rng.uniform(0, 1, n)
    r = B + gam[:, None, None] * activity(m, h, tau, fl, cats) + rng.normal(0, 1, (n, A, C)) * sig[:, None, None]
    r -= r.max(-1, keepdims=True); p = np.exp(r); p /= p.sum(-1, keepdims=True)
    X = np.zeros((n, A, C), np.int32)
    for a in range(A):
        X[:, a] = rng.multinomial(N[a], p[:, a])
    return dict(X=X, m=m, same=same, gam=gam, sig=sig)


def bank2(seed, n=NB, **kw):
    rng = np.random.default_rng(seed)
    S, M, SA, G = [], [], [], []
    for i in range(0, n, 100_000):
        r = sim2(rng, min(100_000, n - i), **kw)
        S.append(summ(r['X']).astype(np.float32)); M.append(r['m'].astype(np.int8))
        SA.append(r['same']); G.append(r['gam'])
    B = dict(S=np.concatenate(S), m=np.concatenate(M), same=np.concatenate(SA), gam=np.concatenate(G))
    _, B['scale'] = abc(B['S'][:60000], B['S'][0], 10)
    return B


def post(B, X, k=K):
    idx, _ = abc(B['S'], summ(X[None])[0], k, B['scale'])
    return month_post(B['m'][idx].astype(int)), float(B['same'][idx].mean()), idx


def power(B, rng_seed, N, cats, groups, fixed, n=200, strong=True, **kw):
    rng = np.random.default_rng(rng_seed)
    r = sim2(rng, 20000, N, cats, groups, fixed, **kw)
    sel = np.where((r['gam'] > 1.2) & (r['sig'] < 0.4))[0] if strong else np.arange(20000)
    sel = sel[:n]
    free = [a for a in range(len(N)) if a not in fixed]
    mass, near, cov = [], [], []
    for i in sel:
        Pm, _, _ = post(B, r['X'][i])
        for a in free:
            t = r['m'][i, a]; p = Pm[a]; mm = int(p.argmax())
            mass.append(p[t]); near.append(min((mm - t) % 12, (t - mm) % 12) <= 1); cov.append(t in cred_set(p))
    return float(np.mean(mass)), float(np.mean(near)), float(np.mean(cov))


if __name__ == '__main__':
    la = la_entries(); lb = lb_entries()
    XLA = profile(la, ['HT', 'KH', 'ZA', 'TY'], CORE); XLB = profile(lb, ['KN', 'PY'], CORE)
    res = {}
    # (a) oracle power: base rates known
    rng0 = np.random.default_rng(11)
    for size, N in (('LA', XLA.sum(1).tolist()), ('LB', XLB.sum(1).tolist())):
        for rep in range(3):
            beta0 = rng0.normal(0, 1.5, len(CORE)); beta0[-1] = 0
            kw = dict(N=N, cats=CORE, groups=[1] * len(N), fixed={}, beta0=beta0, s_off=0.0)
            B = bank2(100 + rep, n=NB // 2, **kw)
            pw = power(B, 200 + rep, **kw, n=100)
            P(f'ORACLE base rates {size}-size rep{rep}: mass on true {pw[0]:.3f} (chance .083) MAP+-1 {pw[1]:.3f} (chance .25) cov80 {pw[2]:.2f}')
            pw2 = power(B, 300 + rep, **kw, n=100, strong=False)
            P(f'   same, prior-random gamma/sigma: mass {pw2[0]:.3f} MAP+-1 {pw2[1]:.3f} cov80 {pw2[2]:.2f}')
            res[f'oracle_{size}_{rep}'] = [pw, pw2]
    # (b) LB leave-one-out: one archive fixed to the argued season, the other inferred
    NLB = XLB.sum(1).tolist()
    for fix, free in ((0, 1), (1, 0)):
        kw = dict(N=NLB, cats=CORE, groups=[0, 0], fixed={fix: SPRING})
        B = bank2(20 + fix, **kw)
        Pm, _, _ = post(B, XLB)
        p = Pm[free]
        P(f'LB LOO: {["KN","PY"][fix]} fixed Feb-May -> {["KN","PY"][free]}: MAP {MONTHS[int(p.argmax())]} mass Feb-May {p[SPRING].sum():.3f} (prior .333) 80% [{fmt_months(cred_set(p))}] '
          + ' '.join(f'{x:.2f}' for x in p))
        res[f'LOO_{free}'] = p.tolist()
        pw = power(B, 40 + fix, **kw, n=150)
        P(f'   planted power for this setup (strong): mass {pw[0]:.3f} MAP+-1 {pw[1]:.3f} cov80 {pw[2]:.2f}')
    # (c) transfer to LA
    X6 = np.concatenate([XLB, XLA]); N6 = X6.sum(1).tolist(); A6 = ['KN', 'PY', 'HT', 'KH', 'ZA', 'TY']
    for s_off in (0.3, 1.0):
        for mode in ('agro', 'random'):
            kw = dict(N=N6, cats=CORE, groups=[0, 0, 1, 1, 1, 1], fixed={0: SPRING, 1: SPRING}, s_off=s_off, mode=mode)
            B = bank2(30 + int(s_off * 10) + (mode == 'random'), **kw)
            Pm, ps, idx = post(B, X6)
            P(f'TRANSFER s_off={s_off} calendar={mode}: P(LA same month)={ps:.2f}')
            for a in range(2, 6):
                p = Pm[a]
                P(f'   {A6[a]}: MAP {MONTHS[int(p.argmax())]} p={p.max():.3f} 80% [{fmt_months(cred_set(p))}] KL {kl_unif(p):.3f}  ' + ' '.join(f'{x:.2f}' for x in p))
            res[f'transfer_{s_off}_{mode}'] = Pm.tolist()
            if mode == 'agro':
                pw = power(B, 50 + int(s_off * 10), **kw, n=150)
                P(f'   planted power (strong): mass {pw[0]:.3f} MAP+-1 {pw[1]:.3f} cov80 {pw[2]:.2f}')
                # commodity-shuffled LA null (LB kept)
                rng = np.random.default_rng(5)
                klr = sum(kl_unif(Pm[a]) for a in range(2, 6)); ks = []
                for _ in range(200):
                    Xs = X6.copy()
                    for a in range(2, 6): Xs[a] = rng.permutation(Xs[a])
                    Ps, _, _ = post(B, Xs); ks.append(sum(kl_unif(Ps[a]) for a in range(2, 6)))
                P(f'   SHUFFLE null (LA commodity labels permuted per site): summed KL real {klr:.3f} vs median {np.median(ks):.3f}, P {(np.array(ks) >= klr).mean():.3f}')
                res[f'transfer_{s_off}_shufP'] = float((np.array(ks) >= klr).mean())
    json.dump(res, open(os.path.join(CK, 'c2.json'), 'w'), default=float)
