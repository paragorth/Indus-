#!/usr/bin/env python3
"""LA-30 cycle 3: hidden totals in a hidden currency, without the word KU-RO.

Every inscription with >= 3 numbers (212). A 'hidden total' = a number q_j that equals the
rate-weighted sum of a run of 2-7 numbers directly before it, where the run mixes >= 2
types (base commodity, or 'bare') and at least one type is a commodity. Rates are on the
simple-rational grid (p/q <= 12; 'bare' is fixed at 1 so the unit is that of the persons'
lists). Site fraction values. Score = inscriptions with >= 1 exact mixed-run hit.
Fit on half the inscriptions (random 2x10^4 + annealed, then the Occam reset of cycle 1d), count hits on the other half,
minus hits under the plain sum. Nulls: N1 commodity labels shuffled across all numbers of
all inscriptions (keeps every number in place); N2 numbers shuffled within each inscription
(keeps labels and positions of labels). Planted: a random rate vector, and in 12 % of the
inscriptions with a mixed run one number is replaced by the weighted sum of the run before it.
Usage: la30_c3.py NREP [MINT MAXRUN]  (variant: targets >= MINT, runs <= MAXRUN; planted targets
below MINT are not scored)
"""
import sys, json, os, time
from multiprocessing import Pool
from la30_common import *

NREP = int(sys.argv[1]) if len(sys.argv) > 1 else 20
NSPLIT = 6
MINT = float(sys.argv[2]) if len(sys.argv) > 2 else 0   # smallest target number considered
MAXRUN = int(sys.argv[3]) if len(sys.argv) > 3 else 7  # longest run
TAG = f'_t{int(MINT)}r{MAXRUN}' if len(sys.argv) > 2 else ''
TYPES = ['bare', 'GRA', 'OLE', 'OLIV', 'VIN', 'NI', 'CYP', 'VIR', '*304', '*308', 'AROM', '*86', '*305', 'OTHER']
TI = {t: i for i, t in enumerate(TYPES)}


def seqs():
    C = json.load(open(os.path.join(D, 'corpus.json')))
    out = []
    for ins in C:
        cur = 'bare'; seq = []
        for k, x in enumerate(ins['tokens']):
            if x['t'] == 'word':
                if x['s'] == ['NI']:
                    cur = 'NI'
                elif len(x['s']) == 1 and x['s'][0].startswith('*') and k + 1 < len(ins['tokens']) and ins['tokens'][k + 1]['t'] == 'num':
                    cur = x['s'][0]
                else:
                    cur = 'bare'
            elif x['t'] == 'logo':
                cur = base_of(x['v'])
            elif x['t'] == 'num':
                v = x['v'] + sum(SITE.get(f, 1 / 16) for f in x['frac'])
                seq.append([cur if cur in TI else 'OTHER', v])
        if len(seq) >= 3:
            out.append({'id': ins['id'], 'seq': seq})
    return out


SEQ = seqs()


def runs_matrix(seqs):
    """A (nruns, K) type sums, b (nruns) target, owner (nruns) inscription index."""
    A = []; b = []; own = []
    for n, s in enumerate(seqs):
        q = s['seq']
        for j in range(2, len(q)):
            if q[j][1] < MINT:
                continue
            for i in range(max(0, j - MAXRUN), j - 1):
                run = q[i:j]
                ts = {t for t, _ in run}
                if len(ts) < 2 or ts <= {'bare'}:
                    continue
                row = np.zeros(len(TYPES))
                for t, v in run:
                    row[TI[t]] += v
                A.append(row); b.append(q[j][1]); own.append(n)
    return np.array(A).reshape(-1, len(TYPES)), np.array(b), np.array(own)


def hits(R, A, b, own, mask, nins):
    """R (m,K): number of inscriptions (in mask) with >= 1 exact hit."""
    sel = mask[own]
    if not sel.any():
        return np.zeros(len(R))
    S = R @ A[sel].T
    h = np.abs(S - b[sel][None]) < 1e-6
    o = own[sel]
    out = np.zeros((len(R), nins), bool)
    for c in range(len(R)):
        out[c, o[h[c]]] = True
    return out.sum(1)


def fit(A, b, own, mask, nins, rng, n_random=20000, n_chain=12, n_steps=250):
    g = np.array(RGRID)
    K = len(TYPES)
    best = []
    for a in range(0, n_random, 2000):
        R = rng_rates(rng, 2000, K, 0.5); R[:, 0] = 1
        sc = hits(R, A, b, own, mask, nins)
        for t in np.argsort(-sc)[:n_chain]:
            best.append((sc[t], R[t].copy()))
    best.sort(key=lambda x: -x[0]); best = best[:n_chain]
    R = np.array([x[1] for x in best]); key = np.array([x[0] for x in best], float)
    top = best[0]
    selr = np.nonzero(mask[own])[0]
    for step in range(n_steps):
        R2 = R.copy()
        for c in range(len(R)):
            if rng.random() < 0.5 and len(selr):
                r = rng.choice(selr); ty = np.nonzero(A[r] > 0)[0]; ty = ty[ty > 0]
                if len(ty) == 0:
                    continue
                t = rng.choice(ty)
                need = R2[c, t] + (b[r] - A[r] @ R2[c]) / A[r, t]
                if need <= 0:
                    continue
                R2[c, t] = g[np.argmin(np.abs(np.log(g) - np.log(need)))]
            else:
                t = rng.integers(1, K)
                R2[c, t] = 1.0 if rng.random() < 0.3 else g[rng.integers(len(g))]
        k2 = hits(R2, A, b, own, mask, nins).astype(float)
        T = max(0.05, 1 - step / n_steps)
        acc = (k2 >= key) | (rng.random(len(key)) < np.exp((k2 - key) / T))
        R[acc] = R2[acc]; key[acc] = k2[acc]
        j = np.argmax(k2)
        if k2[j] > top[0]:
            top = (k2[j], R2[j].copy())
    return top


def prune3(R, A, b, own, mask, nins):
    """Occam step: reset each rate to 1 when that does not lower the train count."""
    R = R.copy(); base = hits(R[None], A, b, own, mask, nins)[0]
    for t in range(1, len(R)):
        if R[t] != 1:
            R2 = R.copy(); R2[t] = 1
            if hits(R2[None], A, b, own, mask, nins)[0] >= base:
                R = R2
    return R


def make(kind, seed):
    rng = np.random.default_rng(seed)
    S = json.loads(json.dumps(SEQ)); truth = None
    if kind == 'N1':
        pool = [x[0] for s in S for x in s['seq']]; rng.shuffle(pool); k = 0
        for s in S:
            for x in s['seq']:
                x[0] = pool[k]; k += 1
    elif kind == 'N2':
        for s in S:
            v = [x[1] for x in s['seq']]; rng.shuffle(v)
            for x, y in zip(s['seq'], v):
                x[1] = y
    elif kind == 'P':
        R = rng_rates(rng, 1, len(TYPES), 0.5)[0]; R[0] = 1; truth = R
        for s in S:
            q = s['seq']
            cand = [(i, j) for j in range(2, len(q)) for i in range(max(0, j - MAXRUN), j - 1)
                    if len({t for t, _ in q[i:j]}) >= 2 and not {t for t, _ in q[i:j]} <= {'bare'}]
            if cand and rng.random() < 0.12:
                i, j = cand[rng.integers(len(cand))]
                q[j][1] = float(sum(R[TI[t]] * v for t, v in q[i:j]))
    return S, truth


def run(job):
    kind, seed = job
    out = os.path.join(CK, f'c3{TAG}_{kind}_{seed}.json')
    if os.path.exists(out):
        return json.load(open(out))
    S, truth = make(kind, seed)
    A, b, own = runs_matrix(S)
    n = len(S); rng = np.random.default_rng(900 + seed)
    gains = []; base = []
    for sp in range(NSPLIT):
        perm = rng.permutation(n); tr = np.zeros(n, bool); tr[perm[:n // 2]] = True
        k, R = fit(A, b, own, tr, n, rng)
        R = prune3(R, A, b, own, tr, n)
        h = hits(R[None], A, b, own, ~tr, n)[0]
        h0 = hits(np.ones((1, len(TYPES))), A, b, own, ~tr, n)[0]
        gains.append(int(h - h0)); base.append(int(h0))
    res = {'kind': kind, 'seed': seed, 'gain': float(np.mean(gains)), 'base': float(np.mean(base)), 'nruns': int(len(b))}
    if truth is not None:
        k, R = fit(A, b, own, np.ones(n, bool), n, rng)
        nz = [i for i in range(1, len(TYPES)) if truth[i] != 1]
        res['recovered'] = int(sum(abs(R[i] - truth[i]) < 1e-9 for i in nz)); res['planted'] = len(nz)
    json.dump(res, open(out, 'w'))
    return res


if __name__ == '__main__':
    A, b, own = runs_matrix(SEQ)
    print('inscriptions', len(SEQ), 'mixed runs', len(b),
          'plain-sum hits (inscriptions)', int(hits(np.ones((1, len(TYPES))), A, b, own, np.ones(len(SEQ), bool), len(SEQ))[0]))
    jobs = [('real', s) for s in range(4)] + [(k, s) for s in range(NREP) for k in ('N1', 'N2')] + [('P', s) for s in range(NREP // 2)]
    t = time.time()
    with Pool(2) as p:
        Rs = p.map(run, jobs, chunksize=1)
    by = {}
    for r in Rs:
        by.setdefault(r['kind'], []).append(r)
    real = np.mean([r['gain'] for r in by['real']])
    summ = {}
    for k, L in by.items():
        g = np.array([r['gain'] for r in L])
        summ[k] = {'n': len(L), 'gain_mean': float(g.mean()), 'gain_sd': float(g.std()),
                   'base_mean': float(np.mean([r['base'] for r in L])), 'P_ge_real': float((1 + (g >= real).sum()) / (1 + len(g)))}
        if k == 'P':
            summ[k]['recovered'] = sum(r['recovered'] for r in L); summ[k]['planted'] = sum(r['planted'] for r in L)
            summ[k]['frac_gt_N1_95'] = float(np.mean(g > np.percentile([r['gain'] for r in by['N1']], 95)))
    # full fit on real data: which rates and which hits
    rng = np.random.default_rng(5)
    k, R = fit(A, b, own, np.ones(len(SEQ), bool), len(SEQ), rng, n_random=100000, n_steps=400)
    S = R @ A.T
    h = np.nonzero(np.abs(S - b) < 1e-6)[0]
    summ['fullfit'] = {'inscriptions_hit': float(k), 'rates': {t: float(R[i]) for i, t in enumerate(TYPES) if R[i] != 1},
                       'hit_docs': sorted({SEQ[own[i]]['id'] for i in h})}
    summ['time_s'] = time.time() - t
    json.dump(summ, open(os.path.join(CK, f'c3{TAG}_summary.json'), 'w'), indent=1)
    print(json.dumps(summ, indent=1))
