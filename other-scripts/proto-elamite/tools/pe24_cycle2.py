"""pe24 cycle 2: bigger hypothesis spaces.

  ratio : markers x {+1, -1, 0, 2, 1/2}, exhaustive over the 7 commonest markers
          (5^7 = 78,125 assignments); same splits / nulls as cycle 1.
  sign  : every sign token (plain and variant, on >= 2 tablets) gets {+1, -1, 0};
          simulated annealing (fit half, 4 restarts x 4,000 steps, penalty 0.3
          closures per non-+1 sign), 20 splits; nulls: markers re-dealt (10),
          random totals (10), 10 splits each; planted control: one plain sign and
          one variant sign set to -1 on tablets that close at baseline.
Usage: python3 pe24_cycle2.py ratio|sign
"""
import sys, time, random
from multiprocessing import Pool
from pe24_common import *
from pe24_cycle1 import load_pe, rand_totals, pvals, plant
VALS5 = (1.0, -1.0, 0.0, 2.0, 0.5)


def run_ratio(C, nsplit=50, seed=0, K=7):
    P = Problem(C, 'mk')
    order = np.argsort(-P.ntab_feat, kind='stable')
    feats = [P.feats[i] for i in order if P.ntab_feat[i] >= 2][:K]
    tabs = tables(P, feats, VALS5)
    A = all_assign(len(feats), 5)
    S = score_all(A, tabs, range(len(C)), 5).sum(1)
    nonp = (A != 0).sum(1)
    best = S.max()
    cand = np.where(S == best)[0]; cand = cand[nonp[cand] == nonp[cand].min()]
    rng = np.random.default_rng(seed)
    gains, ins = [], []
    for s in range(nsplit):
        perm = rng.permutation(len(C)); a, b = perm[:len(C) // 2], perm[len(C) // 2:]
        Sa = score_all(A, tabs, a, 5).sum(1); Sb = score_all(A, tabs, b, 5).sum(1)
        c = np.where(Sa == Sa.max())[0]; c = c[nonp[c] == nonp[c].min()]
        ins.append(Sa.max() - Sa[0]); gains.append(Sb[c].mean() - Sb[0])
    share = {f: {v: float(np.mean(A[cand, j] == i)) for i, v in enumerate(['+1', '-1', '0', 'x2', 'x1/2'])}
             for j, f in enumerate(feats)}
    return {'feats': feats, 'base': int(S[0]), 'best': int(best), 'gain_full': int(best - S[0]),
            'n_tied': int(len(cand)), 'share_full': share, 'heldout_gain': float(np.mean(gains)),
            'insample_gain_half': float(np.mean(ins))}


class SignSearch:
    def __init__(self, C, minfeat=2):
        self.P = P = Problem(C, 'sg', min_tab=minfeat)
        self.rel = []
        for k in range(len(P.cases)):
            self.rel.append(sorted({f for H in P.cases[k] for Pp in H for fl in Pp[2] for f in fl}))
        self.cache = [dict() for _ in P.cases]

    def closes(self, k, c):
        key = tuple(c[f] for f in self.rel[k])
        d = self.cache[k]
        if key not in d:
            d[key] = self.P.closes(k, c)
        return d[key]

    def anneal(self, idx, rng, steps=4000, restarts=4, lam=0.3):
        P = self.P
        nf = len(P.feats)
        feats_in = sorted({f for k in idx for f in self.rel[k]})
        tabs_of = {f: [k for k in idx if f in set(self.rel[k])] for f in feats_in}
        best_c, best_s = np.ones(nf), None
        for r in range(restarts):
            c = np.ones(nf)
            cl = {k: self.closes(k, c) for k in idx}
            cur = sum(cl.values()) - lam * 0
            for t in range(steps):
                if not feats_in:
                    break
                f = feats_in[rng.integers(len(feats_in))]
                old = c[f]
                new = (1.0, -1.0, 0.0)[rng.integers(3)]
                if new == old:
                    continue
                c[f] = new
                d = 0
                newcl = {}
                for k in tabs_of[f]:
                    x = self.closes(k, c); newcl[k] = x; d += int(x) - int(cl[k])
                dpen = lam * (int(new != 1.0) - int(old != 1.0))
                delta = d - dpen
                T = max(0.05, 1.5 * (1 - t / steps))
                if delta >= 0 or rng.random() < np.exp(delta / T):
                    cl.update(newcl); cur += delta
                else:
                    c[f] = old
            s = sum(cl.values()) - lam * (c != 1).sum()
            if best_s is None or s > best_s:
                best_s, best_c = s, c.copy()
        return best_c

    def score(self, c, idx):
        return sum(self.closes(k, c) for k in idx)


def run_sign(C, nsplit=20, seed=0):
    S = SignSearch(C)
    rng = np.random.default_rng(seed)
    n = len(C); base = np.ones(len(S.P.feats))
    full = S.anneal(range(n), rng)
    gains, ins, picks = [], [], collections.Counter()
    for s in range(nsplit):
        perm = rng.permutation(n); a, b = list(perm[:n // 2]), list(perm[n // 2:])
        c = S.anneal(a, rng)
        ins.append(S.score(c, a) - S.score(base, a)); gains.append(S.score(c, b) - S.score(base, b))
        for j in np.where(c != 1)[0]:
            picks[(S.P.feats[j], c[j])] += 1
    return {'n_feats': len(S.P.feats), 'base': S.score(base, range(n)), 'best_full': S.score(full, range(n)),
            'gain_full': S.score(full, range(n)) - S.score(base, range(n)),
            'full_fit': {S.P.feats[j]: full[j] for j in np.where(full != 1)[0]},
            'heldout_gain': float(np.mean(gains)), 'heldout_sd': float(np.std(gains)),
            'insample_gain_half': float(np.mean(ins)),
            'stable_picks': [(f, v, n) for (f, v), n in picks.most_common(12)]}


def null_job(args):
    part, which, s = args
    C = load_pe()
    rng = np.random.default_rng(2000 + s)
    C2 = shuffle_markers(C, random.Random(s)) if which == 'shuf' else rand_totals(C, rng)
    r = run_ratio(C2, nsplit=20, seed=s) if part == 'ratio' else run_sign(C2, nsplit=10, seed=s)
    return which, r['gain_full'], r['heldout_gain']


def plant_sign(C, sign):
    P = Problem(C, 'mk'); base = P.baseline()
    out = []
    for k, c in enumerate(C):
        if not P.closes(k, base):
            out.append(c); continue
        hs = []
        for h in c['hyps']:
            nh = []
            for p in h:
                sgn = [(-1 if sign in m else 1) for m in p['sg']]
                T = [sum(s * e[v] for s, e in zip(sgn, p['E'])) for v in range(len(p['T']))]
                nh.append(dict(p, T=T))
            hs.append(nh)
        out.append(dict(c, hyps=hs))
    return out


if __name__ == '__main__':
    part = sys.argv[1]
    t0 = time.time()
    C = load_pe()
    out = {}
    if part == 'ratio':
        for m in ('~a', '~c'):
            r = run_ratio(plant(C, m, None), nsplit=20)
            out['plant' + m] = {'heldout': r['heldout_gain'], 'share': r['share_full'][m], 'gain_full': r['gain_full']}
            print('plant', m, out['plant' + m], flush=True)
        r = run_ratio(C)
        n = 20
    else:
        # sign-level plants: a plain sign and a variant sign present on several tablets
        cnt = collections.Counter(s for c in C for p in c['hyps'][0] for m in p['sg'] for s in set(m))
        tc = collections.Counter()
        for c in C:
            tc.update({s for p in c['hyps'][0] for m in p['sg'] for s in m})
        plain = [s for s, n in tc.most_common() if '~' not in s and '@' not in s and n >= 5][3]
        var = [s for s, n in tc.most_common() if ('~' in s or '@' in s) and n >= 3][0]
        for sg in (plain, var):
            r = run_sign(plant_sign(C, sg), nsplit=10)
            out['plant_' + sg] = {'heldout': r['heldout_gain'], 'full_fit': r['full_fit'],
                                  'stable': r['stable_picks'][:4], 'gain_full': r['gain_full']}
            print('plant', sg, out['plant_' + sg], flush=True)
        r = run_sign(C)
        n = 10
    print(json.dumps(r, default=str)[:2000], flush=True)
    jobs = [(part, w, s) for w in ('shuf', 'rand') for s in range(n)]
    with Pool(2) as pool:
        res = pool.map(null_job, jobs)
    nl = {}
    for w in ('shuf', 'rand'):
        nl[w] = {'gain_full': [x[1] for x in res if x[0] == w], 'heldout': [x[2] for x in res if x[0] == w]}
    r['p'] = pvals(r, nl)
    out['real'] = r; out['nulls'] = nl; out['seconds'] = time.time() - t0
    json.dump(out, open(os.path.join(DATA, 'pe24_cycle2_%s.json' % part), 'w'), indent=1, default=str)
    print(json.dumps(r['p'], indent=1))
