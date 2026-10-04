"""pe24 cycle 1: marker-level red ink, exhaustive.

Every assignment of {+1, -1, 0} to the K commonest variant markers (3^K, K <= 10)
is scored by the number of tablets whose written total closes exactly. Full-data
best gain over all-+1 (search-corrected by nulls) and held-out gain (fit on a
random half, max closures then fewest non-+1 markers; mean over tied fits;
50 splits).
  ur3   : Ur III balanced accounts (truth: SEC_C -1, SUM 0); null = section/word
          features re-dealt among members; random totals
  plant : PE cases with a planted negative marker (~a or ~b) on the tablets that
          close at baseline
  pe    : real PE; nulls = markers re-dealt among sign tokens (30), totals
          replaced by random numbers of the same size (30)
Usage: python3 pe24_cycle1.py ur3|plant|pe
"""
import sys, time
from multiprocessing import Pool
from pe24_common import *
from pe24_ur3 import build_ur3

K = 10


def load_pe():
    C = json.load(open(os.path.join(CK, 'cases_pe.json')))
    for c in C:
        for h in c['hyps']:
            for p in h:
                p['T'] = [Fr(x) for x in p['T']]
                p['E'] = [[Fr(x) for x in e] for e in p['E']]
    return C


def run(C, kind='mk', K=K, nsplit=50, seed=0, minfeat=2):
    P = Problem(C, kind)
    order = np.argsort(-P.ntab_feat, kind='stable')
    feats = [P.feats[i] for i in order if P.ntab_feat[i] >= minfeat][:K]
    tabs = tables(P, feats)
    A = all_assign(len(feats))
    S = score_all(A, tabs, range(len(C))).sum(1)
    nonp = (A != 0).sum(1)
    best = S.max()
    cand = np.where(S == best)[0]
    cand = cand[nonp[cand] == nonp[cand].min()]
    rng = np.random.default_rng(seed)
    ins, gain, picks = split_eval(tabs, A, len(C), rng, nsplit)
    # per-marker share of tied full-data fits: -1 / 0
    share = {f: {'neg': float(np.mean(A[cand, j] == 1)), 'zero': float(np.mean(A[cand, j] == 2))}
             for j, f in enumerate(feats)}
    top = sorted(picks.items(), key=lambda x: -x[1])[:5]
    return {'feats': feats, 'base': int(S[0]), 'best': int(best), 'gain_full': int(best - S[0]),
            'n_tied': int(len(cand)), 'share_full': share,
            'heldout_gain': float(gain.mean()), 'heldout_sd': float(gain.std()),
            'insample_gain_half': float(ins.mean()),
            'top_split_fits': [({f: '+-0'[v] for f, v in zip(feats, a) if v}, round(w, 2)) for a, w in top]}


def rand_totals(C, rng):
    def f(k, p):
        r = rng.uniform(0.5, 1.5)
        out = []
        for t in p['T']:
            u = Fr(1)
            v = Fr(round(float(t) * r)) if t >= 1 else t * Fr(round(r * 4), 4)
            out.append(max(v, Fr(1)))
        return out
    return with_totals(C, f)


def null_job(args):
    which, s, base = args
    rng = np.random.default_rng(1000 + s)
    import random
    C = load_pe() if base == 'pe' else build_ur3()
    if which == 'shuf':
        C2 = shuffle_markers(C, random.Random(s)) if base == 'pe' else shuffle_ur3(C, rng)
    else:
        C2 = rand_totals(C, rng)
    r = run(C2, nsplit=20, seed=s)
    return which, r['gain_full'], r['heldout_gain']


def shuffle_ur3(C, rng):
    allf = [f for c in C for p in c['hyps'][0] for f in p['mk']]
    perm = [allf[i] for i in rng.permutation(len(allf))]
    it = iter(perm)
    out = []
    for c in C:
        p = c['hyps'][0][0]
        mk = [next(it) for _ in p['mk']]
        out.append(dict(c, hyps=[[dict(p, mk=mk, sg=mk)]]))
    return out


def plant(C, marker, P0):
    """Totals on tablets that close at baseline are recomputed with marker -> -1."""
    P = Problem(C, 'mk')
    base = P.baseline()
    out = []
    for k, c in enumerate(C):
        if not P.closes(k, base):
            out.append(c); continue
        hs = []
        for h in c['hyps']:
            nh = []
            for p in h:
                sgn = [(-1 if marker in m else 1) for m in p['mk']]
                T = [sum(s * e[v] for s, e in zip(sgn, p['E'])) for v in range(len(p['T']))]
                nh.append(dict(p, T=T))
            hs.append(nh)
        out.append(dict(c, hyps=hs))
    return out


def nulls(base, n):
    jobs = [('shuf', s, base) for s in range(n)] + [('rand', s, base) for s in range(n)]
    with Pool(2) as pool:
        res = pool.map(null_job, jobs)
    out = {}
    for w in ('shuf', 'rand'):
        g = np.array([r[1] for r in res if r[0] == w]); h = np.array([r[2] for r in res if r[0] == w])
        out[w] = {'gain_full': g.tolist(), 'heldout': h.tolist()}
    return out


def pvals(r, nl):
    out = {}
    for w, d in nl.items():
        g = np.array(d['gain_full']); h = np.array(d['heldout'])
        out[w] = {'p_full': float((1 + (g >= r['gain_full']).sum()) / (1 + len(g))),
                  'p_heldout': float((1 + (h >= r['heldout_gain']).sum()) / (1 + len(h))),
                  'null_full_mean': float(g.mean()), 'null_heldout_mean': float(h.mean()),
                  'null_heldout_q95': float(np.quantile(h, 0.95))}
    return out


if __name__ == '__main__':
    part = sys.argv[1]
    t0 = time.time()
    if part == 'ur3':
        C = build_ur3()
        r = run(C)
        nl = nulls('ur3', 30)
        r['p'] = pvals(r, nl)
        out = {'real': r, 'nulls': nl}
    elif part == 'plant':
        C = load_pe()
        out = {}
        for m in ('~a', '~b', '~d'):
            Cp = plant(C, m, None)
            r = run(Cp)
            out[m] = r
            print(m, r['base'], r['best'], r['share_full'].get(m), r['heldout_gain'], r['top_split_fits'][:2], flush=True)
    else:
        C = load_pe()
        r = run(C)
        print(json.dumps(r, indent=1), flush=True)
        nl = nulls('pe', 30)
        r['p'] = pvals(r, nl)
        out = {'real': r, 'nulls': nl}
    out['seconds'] = time.time() - t0
    json.dump(out, open(os.path.join(DATA, 'pe24_cycle1_%s.json' % part), 'w'), indent=1, default=str)
    print(json.dumps(out if part == 'plant' else out['real'], indent=1, default=str)[:4000])
