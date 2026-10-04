"""v20 cycle 4: is the held-out line-level conserved combination (cycle 3c) a QUOTA, or just LINE FLAVOUR?
Kill control: fit a mixture of K multinomials over word types to the Voynich lines (EM, K = 1, 2, 4, 8), then
synthesise a corpus in the same layout where every line draws one flavour (its posterior draw) and fills its
token count with i.i.d. words of that flavour. Such a text has line flavours but no quota. If the synthetic text
reproduces the held-out ratio (~0.72) the 'conserved quantity' is flavour mixing; if it stays near 1, flavour
does not explain it.
Also (b) relative-dispersion outliers from cycle 1 (soft-quota detector) and the cycle-2 rerun for Manzoni and
Gadsby with the degeneracy-safe eigen solver."""
import sys, os, json, time
import numpy as np
import scipy.sparse as sp
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v20_lib as L
import v20_cycle3 as C3


def fit_mixture(C, K, iters=60, seed=0):
    rng = np.random.default_rng(seed)
    U = C.nunits['line']
    X = sp.csr_matrix((np.ones(C.N), (C.unit['line'], C.tid)), shape=(U, len(C.types)))
    R = rng.dirichlet(np.ones(K), U)
    for it in range(iters):
        pi = R.mean(0) + 1e-9
        phi = np.asarray((X.T @ R).T) + 0.01; phi /= phi.sum(1, keepdims=True)
        ll = np.asarray(X @ np.log(phi).T) + np.log(pi)
        ll -= ll.max(1, keepdims=True); R = np.exp(ll); R /= R.sum(1, keepdims=True)
    return R, phi


class _Spec(dict):
    pass


def synth(C, K, seed=1):
    rng = np.random.default_rng(seed)
    R, phi = fit_mixture(C, K, seed=seed)
    lines = []
    for i, Ln in enumerate(C.lines):
        k = rng.choice(len(phi), p=R[i])
        ws = rng.choice(len(C.types), size=len(Ln['words']), p=phi[k])
        lines.append(dict(Ln, words=[C.types[j] for j in ws]))
    return L.Corpus(lines, C.name + '-mix%d' % K, True)


def job_mix(K):
    out = os.path.join(L.CK, 'c4_mix%d.json' % K)
    if os.path.exists(out): return json.load(open(out))
    C = L.voynich_corpus()
    S = synth(C, K)
    orig = C3.load
    C3.load = lambda spec: S
    try:
        r = C3.part_c(dict(name='mix%d' % K, src='voy'))
    finally:
        C3.load = orig
    json.dump(r, open(out, 'w'), indent=1)
    return r


def job_c2(name):
    import v20_cycle2 as C2
    out = os.path.join(L.CK, 'c2_%s.json' % name)
    if os.path.exists(out): os.remove(out)
    spec = [j for j in C2.JOBS if j['name'] == name][0]
    return C2.job(spec)


def run(arg):
    kind, x = arg
    return (kind, x, job_mix(x) if kind == 'mix' else job_c2(x))


if __name__ == '__main__':
    tasks = [('mix', 1), ('mix', 2), ('mix', 4), ('mix', 8), ('c2', 'Italian-Manzoni'), ('c2', 'Gadsby')]
    with Pool(2) as P:
        for r in P.imap_unordered(run, tasks):
            print(r, flush=True)
