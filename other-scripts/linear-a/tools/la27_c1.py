#!/usr/bin/env python3
"""LA-27 cycle 1: the physical weight ladder from published masses.

(a) Kendall cosine quantogram phi(q) = sqrt(2/N) sum cos(2 pi m_i / q), q 8-130 g, on masses
    5-1600 g (larger masses swamp a gram-level quantum). Nulls: (i) every mass jittered by
    uniform +-15 % (Kendall's Monte Carlo), (ii) log-uniform random masses over the same range.
    Significance on the max over the whole q grid. Planted: masses = k * 61 g * (1 + N(0, s))
    with k from {1/4,1/3,1/2,2/3,1,4/3,2,3,4,8,12,16,24}, same N; recovered if the top q is
    within 5 % of 61 g (or of 61/2, 61/3, 61*2: harmonics, reported separately).
(b) Denominations: each mass / q* -> nearest p/r (r <= 4) and nearest integer multiple; share
    within 4 %. Null: masses jittered 15 %.
(c) Upper steps: do masses > 300 g sit nearer multiples of 8 q* (a 'mina') than of q*? Ratio
    test of the 8 q* fit vs fits for every other step k q* (k = 5..16): which k is best.
Writes data/la27_ckpt/c1.json and loops/la27_cycle1.txt rows are written by hand from it.
"""
import json, math, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, '..', 'data')
rng = np.random.default_rng(27)
W = json.load(open(os.path.join(D, 'la27_weights.json')))
Q = np.linspace(8, 130, 1221)

def quant(m):
    m = np.asarray(m)[:, None]
    return np.sqrt(2.0 / m.shape[0]) * np.cos(2 * np.pi * m / Q[None, :]).sum(0)

def run(masses, label, nsim=2000):
    m = np.array([x for x in masses if 5 <= x <= 1600])
    phi = quant(m); i = phi.argmax()
    # local peaks
    pk = [(round(Q[j], 1), round(phi[j], 2)) for j in range(1, len(Q) - 1)
          if phi[j] > phi[j - 1] and phi[j] >= phi[j + 1]]
    pk.sort(key=lambda t: -t[1])
    jit = np.array([quant(m * rng.uniform(0.85, 1.15, m.size)).max() for _ in range(nsim)])
    lo, hi = np.log(m.min()), np.log(m.max())
    lu = np.array([quant(np.exp(rng.uniform(lo, hi, m.size))).max() for _ in range(nsim)])
    # value at 61 g specifically (pre-registered from CONTEXT.md)
    j61 = np.abs(Q - 61).argmin()
    jit61 = np.array([quant(m * rng.uniform(0.85, 1.15, m.size))[j61] for _ in range(nsim)])
    return dict(label=label, n=int(m.size), qbest=round(float(Q[i]), 1), phimax=round(float(phi[i]), 3),
                p_jitter=float((jit >= phi[i]).mean()), p_logunif=float((lu >= phi[i]).mean()),
                phi61=round(float(phi[j61]), 3), p61_jitter=float((jit61 >= phi[j61]).mean()),
                peaks=pk[:6])

def planted(n, s, reps=200):
    K = np.array([1/4, 1/3, 1/2, 2/3, 1, 4/3, 2, 3, 4, 8, 12, 16, 24])
    hit = harm = 0
    for _ in range(reps):
        m = rng.choice(K, n) * 61 * (1 + rng.normal(0, s, n))
        q = Q[quant(m).argmax()]
        if abs(q / 61 - 1) < .05: hit += 1
        elif any(abs(q / h - 1) < .05 for h in (61 / 2, 61 / 3, 61 / 4, 122)): harm += 1
    return dict(n=n, sd=s, recovered=hit / reps, harmonic=harm / reps)

def denoms(masses, q, nsim=1000):
    m = np.array([x for x in masses if 3 <= x <= 1600])
    fr = sorted({p / r for r in (1, 2, 3, 4) for p in range(1, 4 * 26) if p / r <= 26})
    fr = np.array(fr)
    def share(mm):
        k = mm / q
        d = np.abs(k[:, None] / fr[None, :] - 1).min(1)
        return (d < .04).mean()
    obs = share(m)
    null = np.array([share(m * rng.uniform(.85, 1.15, m.size)) for _ in range(nsim)])
    k = m / q
    near = [(round(float(x), 1), str(min(((abs(x / f - 1), f) for f in fr))[1])) for x in m]
    # tally which fraction denominators are used below 1 unit
    sub = {}
    for x in m:
        kk = x / q
        if kk < .95:
            for lab, f in [('1/4', .25), ('1/3', 1/3), ('1/2', .5), ('2/3', 2/3), ('3/4', .75),
                           ('1/5', .2), ('1/6', 1/6), ('1/8', .125), ('1/16', 1/16), ('1/12', 1/12)]:
                if abs(kk / f - 1) < .06: sub[lab] = sub.get(lab, 0) + 1
    return dict(q=q, share_on_grid=round(float(obs), 3), null_mean=round(float(null.mean()), 3),
                p=float((null >= obs).mean()), subunits=sub, n=int(m.size))

def upper(masses, q, nsim=1000):
    m = np.array([x for x in masses if x >= 300])
    out = {}
    for k in range(4, 17):
        step = k * q
        r = m / step
        err = np.abs(r - np.round(r)) * step / m      # relative error to nearest multiple
        out[k] = float(np.mean(err < .05))
    # null: jitter
    best = max(out, key=out.get)
    nul = []
    for _ in range(nsim):
        mm = m * rng.uniform(.85, 1.15, m.size)
        r = mm / (best * q); err = np.abs(r - np.round(r)) * best * q / mm
        nul.append(np.mean(err < .05))
    return dict(n=int(m.size), share_by_k={k: round(v, 3) for k, v in out.items()}, best_k=best,
                p_best_jitter=float((np.array(nul) >= out[best]).mean()))

if __name__ == '__main__':
    core = [w['g'] for w in W if w['core']]
    allm = [w['g'] for w in W]
    lead = [w['g'] for w in W if w['core'] and 'lead' in w['mat']]
    res = dict(core=run(core, 'core'), all=run(allm, 'all'), lead=run(lead, 'lead only'))
    res['planted'] = [planted(res['core']['n'], s) for s in (.02, .04, .06)]
    res['denoms61'] = denoms(core, 61.0)
    res['denoms_best'] = denoms(core, res['core']['qbest'])
    res['upper61'] = upper(core, 61.0)
    ing = 29000
    res['ingot_in_units'] = dict(units61=round(ing / 61, 1), minas488=round(ing / 488, 1))
    json.dump(res, open(os.path.join(D, 'la27_ckpt', 'c1.json'), 'w'), indent=1)
    print(json.dumps(res, indent=1))
