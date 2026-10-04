"""v20 cycle 2: CONSERVED QUANTITIES. Is any linear combination of per-unit counts (glyphs, initial-glyph and
final-glyph word classes) constant or nearly constant per line / page, beyond the redeal null?

(a) Generalized eigen search: residuals Y = X - E (E = exact null expectation). Observed covariance S, null
    covariance S0 (mean over redeals). Smallest eigenvalue lam of S a = lam S0 a is the most under-dispersed
    combination (variance ratio vs null). Calibrated: the same smallest eigenvalue for 100 held-out redeals
    (treated as if observed, with S0 from the other redeals).
(b) Same after partialling out the unit's total glyph count (removes the trivial 'line filled to a width').
(c) Integer-relation search: 20,000 random integer combinations (2-6 features, coefficients -3..3), variance
    ratio vs null; family-wise by the min over combos in held-out redeals. Also exact-constancy: share of units
    on the modal value.
Corpora: Voynich ZL / IT, planted (balance_hard = fin:l minus ini:q kept ~0 per page; cap2), Markov-2,
Latin Isidore, Italian Dante (verse: syllable count per line fixed), Italian Manzoni, Gadsby."""
import sys, os, json, time
import numpy as np
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v20_lib as L
import v20_cycle1 as C1
from scipy.linalg import eigh

REPS, HOLD = 200, 100
NCOMB = 20000


def base_features(names):
    keep = [i for i, n in enumerate(names) if n.startswith(('g:', 'ini:', 'fin:'))]
    return keep


def eig_min(S, S0):
    w, V = eigh(S, S0 + 1e-6 * np.eye(len(S)) * np.trace(S0) / len(S))
    return w[0], V[:, 0], w


def analyse(C, names, T, lev, tid, seed=9):
    rng = np.random.default_rng(seed)
    keep = base_features(names)
    gi = names.index('glyphs')
    Tb = T[:, keep + [gi]]
    E = C.expected(lev, Tb, tid)
    def resid(t): return C.counts(lev, t, Tb) - E
    Yo = resid(tid)
    Yn = [resid(C.perm_tid(lev, rng, tid)) for _ in range(REPS + HOLD)]
    U = Yo.shape[0]
    def cov(Y): return Y.T @ Y / U
    def partial(Y):  # remove total-glyph residual component (regression on last column)
        g = Y[:, -1:]; b = (g.T @ Y[:, :-1]) / max((g.T @ g).item(), 1e-9)
        return Y[:, :-1] - g @ b
    out = {}
    for mode in ('full', 'partial'):
        f = (lambda Y: Y[:, :-1]) if mode == 'full' else partial
        S0 = np.mean([cov(f(Y)) for Y in Yn[:REPS]], 0)
        lo, v, w = eig_min(cov(f(Yo)), S0)
        held = [eig_min(cov(f(Y)), S0)[0] for Y in Yn[REPS:]]
        # top loadings, scaled by null sd of each feature
        sd = np.sqrt(np.diag(S0)); vs = v * sd; vs = vs / np.abs(vs).max()
        fn = [names[i] for i in keep]
        top = sorted(zip(fn, vs), key=lambda x: -abs(x[1]))[:8]
        out[mode] = dict(lam=float(lo), held_med=float(np.median(held)), held_min=float(np.min(held)),
                         p=float(np.mean(np.array(held) <= lo)), top=[(a, round(float(b), 2)) for a, b in top],
                         lam2=float(w[1]))
    # integer relations on the full residuals (exclude total glyph col)
    k = len(keep)
    S0 = np.mean([cov(Y[:, :-1]) for Y in Yn[:REPS]], 0); So = cov(Yo[:, :-1])
    A = np.zeros((NCOMB, k))
    for i in range(NCOMB):
        m = rng.integers(2, 7); idx = rng.choice(k, m, replace=False)
        A[i, idx] = rng.choice([-3, -2, -1, 1, 2, 3], m)
    def ratios(S): return np.einsum('ij,jk,ik->i', A, S, A) / np.einsum('ij,jk,ik->i', A, S0, A)
    ro = ratios(So)
    hmin = np.array([ratios(cov(Y[:, :-1])).min() for Y in Yn[REPS:]])
    best = np.argsort(ro)[:5]
    fn = [names[i] for i in keep]
    out['intrel'] = dict(min_ratio=float(ro.min()), held_min_med=float(np.median(hmin)),
                         p=float(np.mean(hmin <= ro.min())),
                         best=[' '.join('%+d*%s' % (A[b, j], fn[j]) for j in np.nonzero(A[b])[0]) + ' ratio %.3f' % ro[b] for b in best])
    # exact constancy: share of units at modal value for the best integer combos vs null
    Xo = C.counts(lev, tid, Tb)[:, :-1]
    cons = []
    for b in best[:3]:
        val = Xo @ A[b]; mo = np.bincount((val - val.min()).astype(int)).max() / len(val)
        nv = []
        for _ in range(50):
            Xn = C.counts(lev, C.perm_tid(lev, rng, tid), Tb)[:, :-1] @ A[b]
            nv.append(np.bincount((Xn - Xn.min()).astype(int)).max() / len(Xn))
        cons.append((round(mo, 3), round(float(np.mean(nv)), 3), float(np.mean(np.array(nv) >= mo))))
    out['intrel']['modal'] = cons
    return out


def job(spec):
    name = spec['name']; out = os.path.join(L.CK, 'c2_%s.json' % name)
    if os.path.exists(out): return name, 'cached'
    t0 = time.time()
    if spec['src'] == 'voy':
        C = L.voynich_corpus(spec.get('tr', 'ZL3b'))
    else:
        C = L.ref_corpus(spec['key'], verse=spec.get('verse', False))
    names, T, _ = L.build_features(C, n_rand_glyph=0, n_rand_word=0)
    tid = C.tid
    if spec.get('plant'): tid = C1.plant(C, names, T, spec['plant'])
    if spec.get('markov'):
        C = L.markov_resynth(C); names, T, _ = L.build_features(C, n_rand_glyph=0, n_rand_word=0); tid = C.tid
    res = {lev: analyse(C, names, T, lev, tid) for lev in ('line', 'page')}
    json.dump(res, open(out, 'w'), indent=1)
    return name, '%.0fs' % (time.time() - t0)


JOBS = [dict(name='ZL', src='voy'), dict(name='IT', src='voy', tr='IT2a'),
        dict(name='plant_balance_hard', src='voy', plant='balance_hard'),
        dict(name='plant_cap2', src='voy', plant='cap2'),
        dict(name='markov', src='voy', markov=True),
        dict(name='Latin-Isidore', src='ref', key='Latin-Isidore'),
        dict(name='Italian-Dante', src='ref', key='Italian-Dante', verse=True),
        dict(name='Italian-Manzoni', src='ref', key='Italian-Manzoni'),
        dict(name='Gadsby', src='ref', key='English-Gadsby-lipogram')]

if __name__ == '__main__':
    only = sys.argv[1:]
    js = [j for j in JOBS if not only or j['name'] in only]
    with Pool(2) as P:
        for r in P.imap_unordered(job, js):
            print(*r, flush=True)
