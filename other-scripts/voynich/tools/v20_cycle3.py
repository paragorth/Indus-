"""v20 cycle 3: (a) re-run the line / paragraph search with a FIRST-LINE-AWARE null (slot classes x 'line is the
paragraph's first line'), so the known gallows-p/f first-line effect cannot pose as a quota; ZL, IT, Markov, Latin
Isidore, Italian Dante. (b) RELATIVE-dispersion detector for soft quotas: a quota that does not beat the natural
over-dispersion still makes its feature LESS over-dispersed than features of the same family and frequency.
Outlier score = (log R - median log R of same family and frequency bin) / robust sd; calibrated on the planted
soft quota (must be found) and on Markov / references (false-positive baseline); Voynich survivors must replicate in
IT2a and in both halves of the pages (odd / even page index)."""
import sys, os, json, time, glob
import numpy as np
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v20_lib as L

REPS = 100


def firstline_slots(C):
    fl = []
    prev = None
    for Ln in C.lines:
        isfirst = Ln['para'] != prev; prev = Ln['para']
        fl += [isfirst] * len(Ln['words'])
    C.slot = C.slot + 4 * np.array(fl, dtype=np.int64)
    # domain*4 in perm/expected must become domain*8: rescale domains
    for k in C.domain: C.domain[k] = C.domain[k] * 2
    return C


def load(spec):
    if spec['src'] == 'voy': C = L.voynich_corpus(spec.get('tr', 'ZL3b'))
    else: C = L.ref_corpus(spec['key'], verse=spec.get('verse', False))
    if spec.get('markov'): C = L.markov_resynth(C)
    return C


def job_a(spec):
    name = spec['name']; out = os.path.join(L.CK, 'c3a_%s.npz' % name)
    if os.path.exists(out): return name, 'cached'
    t0 = time.time()
    C = firstline_slots(load(spec))
    names, T, _ = L.build_features(C, n_rand_glyph=1200, n_rand_word=500)
    res = {}
    for lev in ('line', 'para'):
        obs, nulls = L.run_level(C, lev, T, reps=REPS, seed=13)
        Z = L.zscores(obs, nulls)
        res[lev + '__zD'] = Z[0][0]; res[lev + '__RD'] = Z[0][2]
        res[lev + '__fwD_low'] = L.fw_p(Z[0][0], Z[0][1], 'low')
        res[lev + '__fwD_high'] = L.fw_p(Z[0][0], Z[0][1], 'high')
    np.savez_compressed(out, names=np.array(names), **res)
    return name, '%.0fs' % (time.time() - t0)


def family(n):
    return n.split(':')[0] if ':' in n else 'other'


def outlier_scores(names, R, freq, nbins=8):
    lr = np.log(np.maximum(R, 1e-6)); lf = np.log(np.maximum(freq, 1e-6)); sc = np.zeros(len(R))
    fam = np.array([family(n) for n in names])
    for f in np.unique(fam):
        idx = np.where(fam == f)[0]
        if len(idx) < 10: sc[idx] = 0; continue
        q = np.quantile(lf[idx], np.linspace(0, 1, min(nbins, len(idx) // 5) + 1))
        b = np.clip(np.searchsorted(q, lf[idx], side='right') - 1, 0, len(q) - 2)
        for k in np.unique(b):
            j = idx[b == k]
            med = np.median(lr[j]); mad = np.median(np.abs(lr[j] - med)) * 1.4826 + 1e-6
            sc[j] = (lr[j] - med) / mad
    return sc


def part_b():
    """relative-dispersion outliers from the cycle-1 checkpoints."""
    rows = {}
    specs = {'ZL': dict(src='voy'), 'IT': dict(src='voy', tr='IT2a'), 'plant_quota_soft': dict(src='voy'),
             'plant_cap3': dict(src='voy'), 'plant_quota_hard': dict(src='voy'), 'markov': dict(src='voy', markov=True),
             'Latin-Isidore': dict(src='ref', key='Latin-Isidore'), 'Latin-Caesar': dict(src='ref', key='Latin-Caesar'),
             'Italian-Manzoni': dict(src='ref', key='Italian-Manzoni'),
             'Italian-Dante': dict(src='ref', key='Italian-Dante', verse=True),
             'Gadsby': dict(src='ref', key='English-Gadsby-lipogram')}
    cache = {}
    for name, sp in specs.items():
        d = np.load(os.path.join(L.CK, 'c1_%s.npz' % name)); names = [str(x) for x in d['names']]
        key = json.dumps(sp, sort_keys=True)
        if key not in cache:
            C = load(sp); _, T, _ = L.build_features(C, n_rand_glyph=1200, n_rand_word=500)
            cache[key] = np.asarray(T[C.tid].sum(0)).ravel()
        tot = cache[key]
        rows[name] = {}
        for lev in L.LEVELS:
            R = d[lev + '__RD']; sc = outlier_scores(names, R, tot)
            o = np.argsort(sc)[:5]
            rows[name][lev] = dict(min=float(sc.min()), n_below6=int((sc < -6).sum()),
                                   top=[(names[i], round(float(sc[i]), 1), round(float(R[i]), 2)) for i in o],
                                   q_rank=(int(np.where(np.argsort(sc) == names.index('ini:q'))[0][0]) if 'ini:q' in names else None),
                                   t_rank=(int(np.where(np.argsort(sc) == names.index('g:t'))[0][0]) if 'g:t' in names else None))
    json.dump(rows, open(os.path.join(L.CK, 'c3b.json'), 'w'), indent=1)
    return rows


JOBS = [dict(name='ZL', src='voy'), dict(name='IT', src='voy', tr='IT2a'), dict(name='markov', src='voy', markov=True),
        dict(name='Latin-Isidore', src='ref', key='Latin-Isidore'),
        dict(name='Italian-Dante', src='ref', key='Italian-Dante', verse=True)]

if __name__ == '__main__':
    if sys.argv[1:] == ['c']:
        pass
    elif sys.argv[1:] == ['b']:
        r = part_b()
        for n, d in r.items():
            for lev, x in d.items():
                print(n, lev, 'min %.1f n<-6 %d q_rank %s t_rank %s' % (x['min'], x['n_below6'], x['q_rank'], x['t_rank']), x['top'][:3])
    else:
        with Pool(2) as P:
            for r in P.imap_unordered(job_a, JOBS):
                print(*r, flush=True)


def part_c(spec, nrep=300, seed=21):
    """Held-out conserved combination at line level: learn the most under-dispersed combination (generalized
    eigenvector, total-glyph partialled out) on even-indexed pages, score it on odd-indexed pages against redeals."""
    import v20_cycle2 as C2
    from scipy.linalg import eigh
    rng = np.random.default_rng(seed)
    C = firstline_slots(load(spec)) if spec.get('fl') else load(spec)
    names, T, _ = L.build_features(C, n_rand_glyph=0, n_rand_word=0)
    keep = C2.base_features(names); Tb = T[:, keep + [names.index('glyphs')]]
    lev = 'line'
    E = C.expected(lev, Tb)
    lp = np.zeros(C.nunits[lev], dtype=np.int64); lp[C.unit[lev]] = C.unit['page']
    tr = lp % 2 == 0; te = ~tr
    def resid(t):
        Y = C.counts(lev, t, Tb) - E
        g = Y[:, -1:]; b = (g.T @ Y[:, :-1]) / max((g.T @ g).item(), 1e-9)
        return Y[:, :-1] - g @ b
    Yo = resid(C.tid); Yn = [resid(C.perm_tid(lev, rng)) for _ in range(nrep)]
    cov = lambda Y: Y.T @ Y / len(Y)
    S0tr = np.mean([cov(Y[tr]) for Y in Yn[:150]], 0); S0te = np.mean([cov(Y[te]) for Y in Yn[:150]], 0)
    lam0, v, w = C2.eig_min(cov(Yo[tr]), S0tr)
    r_te = float(v @ cov(Yo[te]) @ v / (v @ S0te @ v))
    r_null = np.array([v @ cov(Y[te]) @ v / (v @ S0te @ v) for Y in Yn[150:]])
    sd = np.sqrt(np.diag(S0tr)); vs = v * sd; vs = vs / np.abs(vs).max()
    fn = [names[i] for i in keep]
    top = sorted(zip(fn, vs), key=lambda x: -abs(x[1]))[:8]
    return dict(name=spec['name'], lam_train=float(w[0]), ratio_test=r_te, null_med=float(np.median(r_null)),
                null_q01=float(np.quantile(r_null, 0.01)), p=float(np.mean(r_null <= r_te)),
                top=[(a, round(float(b), 2)) for a, b in top])


CJOBS = [dict(name='ZL', src='voy'), dict(name='IT', src='voy', tr='IT2a'), dict(name='ZL-fl', src='voy', fl=True),
         dict(name='markov', src='voy', markov=True), dict(name='Latin-Isidore', src='ref', key='Latin-Isidore'),
         dict(name='Italian-Dante', src='ref', key='Italian-Dante', verse=True),
         dict(name='Italian-Manzoni', src='ref', key='Italian-Manzoni'), dict(name='Gadsby', src='ref', key='English-Gadsby-lipogram')]

if __name__ == '__main__' and sys.argv[1:] == ['c']:
    with Pool(2) as P:
        res = list(P.imap_unordered(part_c, CJOBS))
    json.dump(res, open(os.path.join(L.CK, 'c3c.json'), 'w'), indent=1)
    for r in res: print(r)
