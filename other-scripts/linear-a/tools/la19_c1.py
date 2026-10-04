#!/usr/bin/env python3
"""LA-19 cycle 1: choose K for Linear A and calibrate the choice.
Jobs (each checkpointed in data/la19_ckpt/c1_<name>.json):
  LA, LA_entry                     real Linear A word types (all / entry words followed by a number or logogram)
  LA_K1res_<i>                     words resampled from the K=1 model fitted to LA (must give K=1)
  LBpers, LBpers_<i>               LB personnel names (full / LA-size draws): positive control (Greek + non-Greek)
  LBKN, LBPY                       single-site LB names
  UR3, OB                          Ur III seal names (Sumerian + Akkadian), Old Babylonian seal names (LA size)
  L1_<lang>                        36 single real languages (la11 held-out word lists, LA size): negative controls
  PM_<a>_<b>_<p>                   planted 2-language mixtures in a shared syllable space (la11 bigram generators)
  PSAME_<a>                        the same generator twice (planted negative)
  PLA_<c>                          two Dirichlet(c * row) perturbations of LA's own K=1 bigram, 50/50, LA size
usage: la19_c1.py <worker> <nworkers>
"""
import sys, time, zlib
from la19_common import *

N_LA = 952


def la_k1_generator(words, beta, rnd, n, perturb=None, nrs=None):
    enc = Enc(words); g = enc.gvec(words); V = enc.V; W = V + 1
    C = np.zeros((W, W))
    for w in words:
        p = V
        for s in list(w) + [None]:
            x = V if s is None else enc.ix[s]; C[p, x] += 1; p = x if s is not None else V
    P = (C + beta * g[None, :]) / (C.sum(1, keepdims=True) + beta)
    if perturb is not None:
        P = np.array([nrs.dirichlet(perturb * row + 1e-3) for row in P])
    out, seen, tries = [], set(), 0
    cum = np.cumsum(P, 1)
    while len(out) < n and tries < n * 300:
        tries += 1; p = V; w = []
        while len(w) < 10:
            x = int(np.searchsorted(cum[p], rnd.random() * cum[p, -1], side='right')); x = min(x, V)
            if x == V: break
            w.append(enc.signs[x]); p = x
        w = tuple(w)
        if len(w) >= 2 and w not in seen: seen.add(w); out.append(w)
    return out


def jobs():
    J = []
    J.append(('LA', None)); J.append(('LA_entry', None))
    for i in range(3): J.append(('LA_K1res_%d' % i, i))
    J.append(('LBpers', None))
    for i in range(3): J.append(('LBpers_%d' % i, i))
    J += [('LBKN', None), ('LBPY', None), ('UR3', None), ('OB', None)]
    for a, b in [('fin', 'arb'), ('jpn', 'tur'), ('ell', 'eus'), ('sux', 'akk'), ('haw', 'kat')]:
        for p in (50, 80): J.append(('PM_%s_%s_%d' % (a, b, p), (a, b, p)))
    for a in ('fin', 'jpn', 'sux'): J.append(('PSAME_%s' % a, a))
    for c in (2, 5, 20): J.append(('PLA_%d' % c, c))
    for c in la11_codes():
        if c in ('amh', 'hit', 'yor'): continue
        J.append(('L1_' + c, c))
    return J


def build(name, arg):
    rnd = random.Random(zlib.crc32(name.encode()) % 100000); nrs = np.random.RandomState(zlib.crc32(name.encode()) % 100000)
    meta = {}
    if name == 'LA': W = la_types()
    elif name == 'LA_entry': W = la_types(True)
    elif name.startswith('LA_K1res'):
        W = la_k1_generator(la_types(), 10.0, random.Random(arg), N_LA)
    elif name == 'LBpers':
        L = lb_names(); W = [w for w, _ in L]
    elif name.startswith('LBpers_'):
        L = lb_names(); W = sorted(random.Random(arg).sample([w for w, _ in L], N_LA))
    elif name == 'LBKN': W = sorted(w for w, s in lb_names() if s in ('KN', 'BOTH'))
    elif name == 'LBPY': W = sorted(w for w, s in lb_names() if s in ('PY', 'BOTH'))
    elif name == 'UR3': W = sorted(rnd.sample(ur3_names(), N_LA))
    elif name == 'OB': W = sorted(rnd.sample(ob_names(), N_LA))
    elif name.startswith('PM_'):
        a, b, p = arg
        na = int(round(N_LA * p / 100)); nb = N_LA - na
        A = bigram_gen(la11_lang(a)[1]['bi'], na, rnd)
        B = bigram_gen(la11_lang(b)[1]['bi'], nb, rnd, exclude=set(A))
        W = A + B; meta['truth'] = [0] * len(A) + [1] * len(B)
        o = sorted(range(len(W)), key=lambda i: W[i]); W = [W[i] for i in o]; meta['truth'] = [meta['truth'][i] for i in o]
    elif name.startswith('PSAME_'):
        A = bigram_gen(la11_lang(arg)[1]['bi'], N_LA, rnd); W = sorted(A)
    elif name.startswith('PLA_'):
        base = la_types()
        A = la_k1_generator(base, 10.0, rnd, N_LA // 2, perturb=float(arg), nrs=nrs)
        B = [w for w in la_k1_generator(base, 10.0, rnd, N_LA, perturb=float(arg), nrs=nrs) if w not in set(A)][:N_LA - len(A)]
        W = A + B; t = [0] * len(A) + [1] * len(B)
        o = sorted(range(len(W)), key=lambda i: W[i]); W = [W[i] for i in o]; meta['truth'] = [t[i] for i in o]
    elif name.startswith('L1_'):
        W = la11_lang(arg)[2]; W = sorted(rnd.sample(W, min(N_LA, len(W))))
    return W, meta


def run(name, arg):
    path = os.path.join(CK, 'c1_%s.json' % name)
    if os.path.exists(path): return
    t0 = time.time()
    W, meta = build(name, arg)
    res = cv_mixture(W, seed=1)
    tri = cv_trigram(W, seed=1)
    row = summary_row(name, res, tri)
    row['mean_len'] = round(float(np.mean([len(w) for w in W])), 2); row['V'] = Enc(W).V
    K = row['K_sel']
    Kst = sorted(set([K, 2, row['K_best']]))
    row['stab'] = {}
    nre = 10 if not name.startswith('L1_') else 5
    for k in Kst:
        if k == 1: continue
        st = stability(W, k, res[k]['beta'], restarts=nre, seed=3)
        e = dict(ari_mean=round(st['ari_mean'], 3), ari_min=round(st['ari_min'], 3))
        if 'truth' in meta:
            e['ari_truth'] = round(float(np.mean([ari(m, meta['truth']) for m in st['maps']])), 3)
        if k == K or k == 2:
            P = consensus(st, k); e['map'] = P.argmax(1).tolist(); e['pmax_mean'] = round(float(P.max(1).mean()), 3)
            e['sizes'] = np.bincount(P.argmax(1), minlength=k).tolist()
        row['stab'][k] = e
    row['words'] = ['-'.join(w) for w in W]
    if 'truth' in meta: row['truth'] = meta['truth']
    row['lpw'] = {k: np.round(res[k]['lpw'], 4).tolist() for k in res}
    row['secs'] = round(time.time() - t0, 1)
    dump(path, row)
    print(name, 'K_sel', K, 'K_best', row['K_best'], 'gain', row['gain_per_word'][K], 'tri z', row['tri_vs_mix_z'],
          'stab', {k: v['ari_mean'] for k, v in row['stab'].items()}, row['secs'], 's', flush=True)


if __name__ == '__main__':
    wk, nw = int(sys.argv[1]), int(sys.argv[2])
    for i, (n, a) in enumerate(jobs()):
        if i % nw == wk: run(n, a)
