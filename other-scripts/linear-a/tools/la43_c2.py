#!/usr/bin/env python3
"""LA-43 cycle 2: leave-one-site-out prediction and which consonant rows carry it.

(a) LOSO: six folds (Hagia Triada, Khania, Phaistos, Zakros, Knossos, all other sites). Each fold is predicted from
    the other five; scores are pooled over folds (symbol-weighted bits, mask). True map vs 2,000 relabelings per
    tier (R2a, R2b, R3). LB control: LB drawn at the LA total size from KN+PY (5 draws), folds = six groups of
    (site, series letter) archives, balanced by size. In-site control: random fifths of Hagia Triada documents.
(b) Rows: for each consonant row of the grid with >= 2 signs, swap the values of that row's signs with as many
    random valued signs from other rows (1,000 swaps); P = share of swaps predicting as well as the true map.
    LB calibration: the same test on the LB LOSO draws (300 swaps per row) gives the pass rate of TRUE rows at LA size.
Freeze: true-map and sound-free predictions of every fold hashed before scoring.
"""
import sys, os, json, time, hashlib, collections
os.environ.setdefault('OMP_NUM_THREADS', '1'); os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import la43_common as M
import la32_common as C32
from multiprocessing import Pool

TIERS = ['R2a', 'R2b', 'R3']
SITES = ['Haghia Triada', 'Khania', 'Phaistos', 'Zakros', 'Knossos']


def folds_la(valued=False):
    U = M.la_units()
    if valued:   # only words written entirely with valued signs (as in la38)
        U = [u for u in U if len(u['w']) >= 2 and all(M.L38.cv_of(s) is not None for s in u['w'])]
    g = lambda u: u['site'] if u['site'] in SITES else 'other'
    return [[u for u in U if g(u) == f] for f in SITES + ['other']]


def folds_insite(seed=5):
    U = [u for u in M.la_units() if u['site'] == 'Haghia Triada']
    rng = np.random.default_rng(seed)
    docs = sorted({u['doc'] for u in U}); rng.shuffle(docs)
    f = {d: i % 5 for i, d in enumerate(docs)}
    return [[u for u in U if f[u['doc']] == k] for k in range(5)]


def folds_lb(d, ntok=5558):
    L = [dict(w=tuple(s.upper() for s in r['w']), doc=r['doc'], site=r['site'], series=r['series']) for r in C32.lb_words()]
    D = M.draw_docs(L, ntok, 300 + d)
    grp = collections.Counter()
    for u in D:
        grp[u['series'][:3]] += len(u['w'])
    folds = [[] for _ in range(6)]; size = np.zeros(6); asg = {}
    for k, n in grp.most_common():
        i = int(np.argmin(size)); asg[k] = i; size[i] += n
    for u in D:
        folds[asg[u['series'][:3]]].append(u)
    return folds


class Loso:
    def __init__(self, folds):
        allu = [u for f in folds for u in f]
        signs = set(s for u in allu for s in u['w'])
        self.E = []
        for i, f in enumerate(folds):
            tr = [u for j, g in enumerate(folds) if j != i for u in g]
            self.E.append(M.Experiment(tr, f, seed=i, grid_signs=signs))
        self.grid = self.E[0].grid
        self.n = np.array([e.test.nsym for e in self.E]); self.nt = np.array([len(e.test.tri) for e in self.E])

    def score(self, Cv, Vv, sound=True):
        bits = mask = 0.0
        for e, n, nt in zip(self.E, self.n, self.nt):
            P, _ = e.run_map(Cv, Vv, sound)
            s = e.test.score(P)
            bits += s['bits'] * n; mask += (s['mask'] * nt if nt else 0)
        return dict(bits=bits / self.n.sum(), mask=mask / self.nt.sum())

    def frozen(self):
        G = self.grid
        hs = hashlib.sha256()
        for e in self.E:
            hs.update(e.frozen()[4].encode()); hs.update(e.test.digest().encode())
        return hs.hexdigest()[:16]


def build(tag):
    if tag == 'LA':
        return Loso(folds_la())
    if tag == 'LAV':
        return Loso(folds_la(valued=True))
    if tag == 'LAIN':
        return Loso(folds_insite())
    if tag.startswith('LB'):
        return Loso(folds_lb(int(tag[2:])))
    raise ValueError(tag)


def tier_job(args):
    tag, tier, N, seed = args
    X = build(tag); G = X.grid
    st = X.score(G.C0, G.V0); sn = X.score(G.C0, G.V0, sound=False)
    Cn, Vn = G.relabels(tier, N, seed)
    null = {k: np.empty(N) for k in ('bits', 'mask')}
    for i in range(N):
        s = X.score(Cn[i], Vn[i])
        for k in null:
            null[k][i] = s[k]
    res = {}
    for k in null:
        x = null[k]
        res[k] = dict(true=st[k], none=sn[k], gain=sn[k] - st[k], null_gain=float(np.mean(sn[k] - x)),
                      z=float((x.mean() - st[k]) / (x.std() + 1e-12)), p=float(((x <= st[k] + 1e-12).sum() + 1) / (N + 1)))
    return dict(kind='tier', tag=tag, tier=tier, N=N, nsym=int(X.n.sum()), res=res)


def row_job(args):
    tag, row, N, seed = args
    X = build(tag); G = X.grid
    rng = np.random.default_rng(seed)
    st = X.score(G.C0, G.V0)
    ci = G.Clab.index(row)
    mine = np.where(G.C0 == ci)[0]; other = np.where(G.C0 != ci)[0]
    xb = np.empty(N); xm = np.empty(N)
    for i in range(N):
        Cv = G.C0.copy(); Vv = G.V0.copy()
        pick = rng.choice(other, len(mine), replace=False)
        Cv[mine], Cv[pick] = G.C0[pick], G.C0[mine]
        Vv[mine], Vv[pick] = G.V0[pick], G.V0[mine]
        s = X.score(Cv, Vv); xb[i] = s['bits']; xm[i] = s['mask']
    return dict(kind='row', tag=tag, row=row, nsign=int(len(mine)), N=N,
                bits=dict(z=float((xb.mean() - st['bits']) / (xb.std() + 1e-12)), p=float(((xb <= st['bits'] + 1e-12).sum() + 1) / (N + 1)),
                          drop=float(xb.mean() - st['bits'])),
                mask=dict(z=float((xm.mean() - st['mask']) / (xm.std() + 1e-12)), p=float(((xm <= st['mask'] + 1e-12).sum() + 1) / (N + 1))))


def job(a):
    return tier_job(a[1:]) if a[0] == 'tier' else row_job(a[1:])


def main():
    out_p = os.path.join(M.CK, 'c2.json')
    done = json.load(open(out_p)) if os.path.exists(out_p) else []
    have = {(d['kind'], d['tag'], d.get('tier') or d.get('row')) for d in done}
    jobs = []
    tags = ['LA', 'LAV', 'LAIN'] + [f'LB{d}' for d in range(5)]
    fz = os.path.join(M.CK, 'c2_freeze.json')
    if not os.path.exists(fz):
        F = {t: build(t).frozen() for t in tags}
        json.dump(F, open(fz, 'w'), indent=1)
        print('frozen', hashlib.sha256(open(fz, 'rb').read()).hexdigest()[:16], flush=True)
    for k, t in enumerate(TIERS):
        jobs.append(('tier', 'LA', t, 2000, 3000 + k))
        jobs.append(('tier', 'LAIN', t, 1000, 3100 + k))
        jobs.append(('tier', 'LAV', t, 2000, 3150 + k))
        for d in range(5):
            jobs.append(('tier', f'LB{d}', t, 300, 3200 + 10 * d + k))
    for tag in tags:
        if tag in ('LAIN', 'LAV', 'LB3', 'LB4'):
            continue
        G = build(tag).grid
        cnt = collections.Counter(G.C0.tolist())
        for ci, n in sorted(cnt.items()):
            if n >= 2:
                jobs.append(('row', tag, G.Clab[ci], 500 if tag == 'LA' else 150, 4000 + ci + 100 * tags.index(tag)))
    jobs = [j for j in jobs if (j[0], j[1], j[2]) not in have]
    jobs.sort(key=lambda j: (j[0] == 'row', j[1].startswith('LA'), j[3]))
    with Pool(2) as pool:
        for r in pool.imap_unordered(job, jobs):
            done.append(r)
            json.dump(done, open(out_p, 'w'))
            if r['kind'] == 'tier':
                print('tier', r['tag'], r['tier'], r['nsym'], '; '.join(f"{k} gain {v['gain']:+.3f} null {v['null_gain']:+.3f} z{v['z']:+.1f} P{v['p']:.3g}" for k, v in r['res'].items()), flush=True)
            else:
                print('row', r['tag'], repr(r['row']), r['nsign'], f"bits z{r['bits']['z']:+.1f} P{r['bits']['p']:.3g} mask z{r['mask']['z']:+.1f} P{r['mask']['p']:.3g}", flush=True)


if __name__ == '__main__':
    main()
