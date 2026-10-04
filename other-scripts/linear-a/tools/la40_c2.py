#!/usr/bin/env python3
"""LA-40 cycle 2: learn the type system from the control, by massive random guessing.

Each hypothesis is a random 'type signature' table w[role][feature value] (sparse, Gaussian),
a homogeneity weight and a DM on/off switch. viol[t][r] = -sum over t's occurrences of
w[r][features]. Each hypothesis is run (1 chain, short) on a Linear B Knossos sample at Linear A
size and scored by macro recall of the known word classes (P, L, C, H, T) on types with >= 2
tokens. The top systems are re-tested on Pylos samples (other site, mostly other words).
Search correction: the same search with the Knossos labels permuted among labelled types;
its best system is re-tested on Pylos the same way.
"""
import collections, json, os, random, sys, time
import numpy as np
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import la40_common as C

NH = int(os.environ.get('NH', 2000))
STEPS = int(os.environ.get('STEPS', 120000))
OFF = np.cumsum([0] + C.CARD)[:-1]
NV = sum(C.CARD)
LABS = ['P', 'L', 'C', 'H', 'T']
G = {}


def occ_onehot(B):
    X = np.zeros((len(B['occ']), NV))
    f = B['ofeat'].reshape(-1, len(C.CARD))
    for k in range(len(C.CARD)):
        X[np.arange(len(f)), OFF[k] + f[:, k]] = 1
    T = np.zeros((len(B['types']), NV))
    np.add.at(T, B['otype'], X)
    return T  # types x feature values (counts)


def draw_sys(rng):
    W = rng.normal(0, 2.0, (C.R, NV)) * (rng.random((C.R, NV)) < 0.3)
    return dict(W=W, lam_h=float(rng.choice([0, 1, 3, 6])), dm=bool(rng.random() < 0.5))


def run_sys(B, T, sysd, seed, steps=STEPS, nch=1):
    viol = -(T @ sysd['W'].T)                    # types x roles
    viol = np.ascontiguousarray(viol.ravel(), dtype=np.float64)
    alpha = 0.5 if sysd['dm'] else 1e6
    return C.run_chains(B, nchains=nch, steps=steps, thin=max(1000, steps // 100), seed0=seed, alpha=alpha,
                        lam_h=sysd['lam_h'], viol=viol)


def macro(B, S, lab):
    mode, freq, agree, _ = C.summarize(S)
    rec = collections.defaultdict(list)
    for t, w in enumerate(B['types']):
        if w in lab and B['nocc'][t] >= 2:
            rec[lab[w]].append(C.ROLES[mode[t]] == lab[w])
    return float(np.mean([np.mean(rec[l]) for l in LABS if rec[l]])), {l: (int(np.sum(rec[l])), len(rec[l])) for l in LABS}


def lb_site_sample(site, k):
    D = [d for d in C.lb_docs_all() if d['site'] == site]
    LA = C.la_docs(); ntok = sum(1 for d in LA for ln in d['lines'] for t in ln if C.is_word(t))
    return C.lb_sample(D, ntok, random.Random(C.seed('la40c2-%s-%d' % (site, k))))


def setup():
    lab = C.lb_labels()
    G['lab'] = lab
    G['train'] = []
    for k in range(2):
        B = C.build(lb_site_sample('KN', k)); G['train'].append((B, occ_onehot(B)))
    G['test'] = []
    for k in range(3):
        B = C.build(lb_site_sample('PY', k)); G['test'].append((B, occ_onehot(B)))
    # permuted-label control for the search (labels shuffled among labelled KN types)
    rng = random.Random(7)
    keys = sorted(w for w in lab)
    vals = [lab[w] for w in keys]; rng.shuffle(vals)
    G['labperm'] = dict(zip(keys, vals))


def eval_h(args):
    h, which = args
    rng = np.random.default_rng(C.seed('la40c2-sys-%d' % h))
    s = draw_sys(rng)
    labd = G['lab'] if which == 'real' else G['labperm']
    sc = [macro(B, run_sys(B, T, s, seed=h + 11 * i), labd)[0] for i, (B, T) in enumerate(G['train'])]
    return h, which, float(np.mean(sc))


def test_h(h):
    rng = np.random.default_rng(C.seed('la40c2-sys-%d' % h))
    s = draw_sys(rng)
    out = []
    for i, (B, T) in enumerate(G['test']):
        m, det = macro(B, run_sys(B, T, s, seed=h + 101 * i, nch=2), G['lab'])
        out.append((m, det))
    return h, out


def main():
    setup()
    t = time.time()
    jobs = [(h, 'real') for h in range(NH)] + [(h, 'perm') for h in range(NH // 2)]
    with Pool(2, initializer=setup) as p:
        res = p.map(eval_h, jobs, chunksize=20)
    real = sorted([(s, h) for h, w, s in res if w == 'real'], reverse=True)
    perm = sorted([(s, h) for h, w, s in res if w == 'perm'], reverse=True)
    print('search %.0fs; train macro recall real top5 %s; perm-label top5 %s; median %.3f' % (
        time.time() - t, [round(s, 3) for s, _ in real[:5]], [round(s, 3) for s, _ in perm[:5]], np.median([s for s, _ in real])), flush=True)
    top = [h for _, h in real[:20]]
    ptop = [h for _, h in perm[:5]]
    rnd = [h for _, h in real[NH // 2: NH // 2 + 20]]  # median-ranked systems as the test baseline
    with Pool(2, initializer=setup) as p:
        tt = dict(p.map(test_h, top + ptop + rnd, chunksize=1))
    summ = lambda hs: [float(np.mean([m for m, _ in tt[h]])) for h in hs]
    out = dict(real=real, perm=perm, top=top, ptop=ptop, rnd=rnd,
               test={h: [(m, d) for m, d in tt[h]] for h in tt},
               test_top=summ(top), test_ptop=summ(ptop), test_rnd=summ(rnd))
    json.dump(out, open(os.path.join(C.CK, 'c2.json'), 'w'))
    print('PY held-out macro recall: top20 mean %.3f (%s); perm-label top5 %.3f; median-ranked 20 %.3f' % (
        np.mean(out['test_top']), [round(x, 3) for x in out['test_top'][:10]], np.mean(out['test_ptop']), np.mean(out['test_rnd'])))
    for h in top[:5]:
        print(h, tt[h][0][1])


if __name__ == '__main__':
    main()
