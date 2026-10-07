"""pe78 cycle 1: simulated worlds -> Taylor signatures -> can a simulator-trained reader type KNOWN keys?

1. template pool: the group sizes of every real key with a signature (PE, PC, Ur III; Ur III capped at 60 groups).
2. N random worlds per kind (LIVING stage-herd ABM with random biology; ALLOC rate x heads; MEASURED lognormal;
   POISSON no biology), each laid on a random real template.
3. kNN reader trained on the simulations (standardised features).  Planted check: held-out simulations.
4. Calibration: P(LIVING) for Ur III and proto-cuneiform keys of known kind (AUC LIVING vs GOODS).
5. Kill: values permuted across keys within each tablet (20x); 'Poisson must lose' = share of real keys whose
   nearest simulations are POISSON.
"""
import sys, os, json, time, math
import numpy as np
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pe78_common as C

N_PER = int(os.environ.get('PE78_N', 30000))


def templates():
    rng = np.random.default_rng(1)
    T = {}
    for name, f in (('PE', C.pe_entries), ('PC', C.pc_entries), ('UR', C.ur_entries)):
        G = C.groups(f())
        T[name] = []
        for k, L in G.items():
            if len(L) >= 5:
                L = C.cap_groups(L, rng)
                T[name].append([len(g) for _, g in L])
    return T


def work(args):
    kind, seed, n, pool = args
    rng = np.random.default_rng(seed)
    X, TH = [], []
    for i in range(n):
        sz = pool[int(rng.integers(len(pool)))]
        s, th = C.sim_signature(kind, sz, rng)
        if s:
            X.append(C.vec(s)); TH.append(th)
    return kind, np.array(X), TH


def simulate(pool, n_per, tag, seed0=0):
    fn = os.path.join(C.CK, 'sims_%s.npz' % tag)
    if os.path.exists(fn):
        d = np.load(fn, allow_pickle=True)
        return d['X'], d['y'], list(d['th'])
    jobs = []
    for ki, k in enumerate(C.KINDS):
        for c in range(8):
            jobs.append((k, seed0 + 1000 * ki + c, n_per // 8, pool))
    with Pool(2) as p:
        res = p.map(work, jobs)
    X = np.concatenate([r[1] for r in res if len(r[1])])
    y = np.concatenate([[C.KINDS.index(r[0])] * len(r[1]) for r in res if len(r[1])])
    th = sum([r[2] for r in res], [])
    np.savez(fn, X=X, y=y, th=np.array(th, dtype=object))
    return X, y, th


class Reader:
    def __init__(self, X, y, k=50):
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-9
        self.Z = (X - self.mu) / self.sd
        self.y = y
        self.k = k
        self.prior = np.bincount(y, minlength=4) / len(y)

    def post(self, x):
        z = (np.atleast_2d(x) - self.mu) / self.sd
        out = []
        for r in z:
            d = ((self.Z - r) ** 2).sum(1)
            nn = np.argpartition(d, self.k)[:self.k]
            c = np.bincount(self.y[nn], minlength=4) / self.k / self.prior
            out.append(c / c.sum())
        return np.array(out)


def auc(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if not len(a) or not len(b):
        return float('nan')
    return float(((a[:, None] > b[None]).mean() + 0.5 * (a[:, None] == b[None]).mean()))


def labelled(name):
    f = C.pc_entries if name == 'PC' else C.ur_entries
    return [e for e in f() if e[3]]


def real_sigs(E, rng, min_g):
    G = C.groups(E)
    S = {}
    for k, L in G.items():
        s = C.signature(C.cap_groups(L, rng), min_g)
        if s:
            S[k] = s
    return S


def calib(R, E, rng, min_g):
    lab = {e[1]: e[3] for e in E}
    S = real_sigs(E, rng, min_g)
    keys = sorted(S)
    P = R.post(np.array([C.vec(S[k]) for k in keys]))
    pl = {k: P[i, 0] for i, k in enumerate(keys)}
    a = auc([pl[k] for k in keys if lab[k] == 'LIVING'], [pl[k] for k in keys if lab[k] == 'GOODS'])
    return a, keys, P, S


if __name__ == '__main__':
    t0 = time.time()
    out = []
    T = templates()
    pool = T['PE'] + T['PC'] + T['UR']
    print('templates', {k: len(v) for k, v in T.items()}, flush=True)
    X, y, th = simulate(pool, N_PER, 'c1')
    print('sims', X.shape, np.bincount(y), '%.0fs' % (time.time() - t0), flush=True)
    rng = np.random.default_rng(7)
    idx = rng.permutation(len(y))
    tr, te = idx[:len(y) - 2000], idx[len(y) - 2000:]
    R = Reader(X[tr], y[tr])
    P = R.post(X[te])
    acc = (P.argmax(1) == y[te]).mean()
    conf = np.zeros((4, 4), int)
    for a_, b_ in zip(y[te], P.argmax(1)):
        conf[a_, b_] += 1
    aucL = auc(P[y[te] == 0, 0], P[y[te] != 0, 0])
    print('planted (held-out sims): acc %.3f (chance 0.25) AUC living-vs-rest %.3f' % (acc, aucL))
    print(conf)
    out.append('| PE-78.1.1 | Planted worlds: %d random simulated keys (4 kinds x random biology/rules on real group-size templates), kNN reader on Taylor signatures (b, a, scatter, zero-variance share, overall CV, mean span); 2,000 held-out simulations | 4-way accuracy %.3f (chance 0.25); LIVING vs rest AUC %.3f; confusion rows L/A/M/P = %s | %s |' % (
        len(y), acc, aucL, conf.tolist(), 'the simulator kinds are only partly separable from fluctuation scaling' if aucL < 0.8 else 'separable in principle'))
    R = Reader(X, y)
    res = {}
    for name, mg in (('UR', 5), ('PC', 4)):
        E = labelled(name)
        a, keys, Pr, S = calib(R, E, np.random.default_rng(3), mg)
        lab = {e[1]: e[3] for e in E}
        pois = float((Pr.argmax(1) == 3).mean())
        sh = []
        for r in range(20):
            Es = C.shuffle_within_tablet(E, np.random.default_rng(100 + r))
            sh.append(calib(R, Es, np.random.default_rng(3), mg)[0])
        sh = np.array(sh)
        res[name] = dict(auc=a, shuffle=sh.tolist(), pois=pois, n=len(keys),
                         nL=sum(lab[k] == 'LIVING' for k in keys))
        print(name, 'AUC P(living) LIVING vs GOODS %.3f  shuffled %.3f +- %.3f  share nearest POISSON %.2f  keys %d (%d living)' % (
            a, sh.mean(), sh.std(), pois, len(keys), res[name]['nL']), flush=True)
        for i, k in enumerate(keys):
            print('   %-22s %-7s P=%s b=%.2f' % (k, lab[k], np.round(Pr[i], 2), S[k]['b']))
    out.append('| PE-78.1.2 | Calibration: simulator-trained reader on Ur III keys of known kind (LIVESTOCK, PEOPLE vs GRAIN, RATION; %d keys) and proto-cuneiform keys (conventional animal/human vs goods signs; %d keys). Kill: values permuted across keys within tablet (20x) | AUC of P(LIVING): Ur III %.2f (within-tablet shuffle %.2f +- %.2f); PC %.2f (shuffle %.2f +- %.2f). Keys read as POISSON: Ur III %.2f, PC %.2f | %s |' % (
        res['UR']['n'], res['PC']['n'], res['UR']['auc'], np.mean(res['UR']['shuffle']), np.std(res['UR']['shuffle']),
        res['PC']['auc'], np.mean(res['PC']['shuffle']), np.std(res['PC']['shuffle']), res['UR']['pois'], res['PC']['pois'],
        'reader types known keys' if min(res['UR']['auc'], res['PC']['auc']) > 0.7 else 'reader does NOT type known keys in both calibration corpora'))
    json.dump(res, open(os.path.join(C.CK, 'c1_res.json'), 'w'))
    open(os.path.join(C.CK, 'c1_rows.txt'), 'w').write('\n'.join(out) + '\n')
    print('\n'.join(out))
    print('done %.0fs' % (time.time() - t0))
