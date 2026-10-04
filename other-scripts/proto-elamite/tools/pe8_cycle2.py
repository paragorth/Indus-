"""pe8 cycle 2.  (a) Are the clusters discrete forms or graded variation?  (b) Test 1: does the form predict an
entry's number system better than the counted good (the entry's final sign) does?

(a) 5-fold held-out log-likelihood per tablet (complete rows only) of: independence; Chow-Liu tree (best
    graded pairwise-dependence model, no classes); latent class model with K forms.  A real form system must
    beat the tree.  Run on REAL, PLANT_1..3 (must: LCM > tree) and NULL_1..3 (must: neither beats independence
    by much).
(b) 5-fold x 4 repeats by tablet.  Forms are fitted on training tablets WITHOUT any number information
    (SYS and TOT removed; strict variant also removes CLS/CLSR, the class-sign role features).  Predict each
    held-out entry's system (S counted / C capacity / O other) from:
      global marginal; GOOD = P(sys | entry's final sign); HDR = P(sys | header type);
      FORM = sum_k P(form k | tablet's non-number skeleton) P(sys | k); FORM+GOOD (product of experts);
      TABLET ceiling = the other entries on the same tablet (leave-one-entry-out, add-1).
    Control: the non-number skeleton block is permuted across tablets before fitting (forms intact, tied to
    the wrong tablet): FORM must fall to the marginal.
Output: data/pe8_cycle2.json
"""
import json, os, sys
from multiprocessing import Pool
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pe8_common import *

SYSC = ['S', 'C', 'O']


def discreteness(args):
    name, rows, K = args
    X, lev = encode(rows)
    card = [max(1, len(lev[k])) for k in FEATS]
    rng = np.random.default_rng(7)
    idx = rng.permutation(len(X)); parts = np.array_split(idx, 5)
    out = {'ind': [], 'tree': [], 'lcm': [], 'n_complete': 0}
    for f in range(5):
        te = parts[f]; tr = np.concatenate([parts[g] for g in range(5) if g != f])
        te = te[(X[te] >= 0).all(1)]
        out['n_complete'] += len(te)
        out['ind'] += indep_ll(X[tr], X[te], card).tolist()
        out['tree'] += chow_liu_ll(chow_liu(X[tr], card), X[te]).tolist()
        m = best_em(X[tr], card, K, 10, 31 + f)
        out['lcm'] += lcm_loglik(m, X[te])[0].tolist()
    r = {k: float(np.mean(v)) for k, v in out.items() if k != 'n_complete'}
    d = np.array(out['lcm']) - np.array(out['tree'])
    r['lcm_minus_tree'] = float(d.mean()); r['se'] = float(d.std() / np.sqrt(len(d)))
    r['tree_minus_ind'] = r['tree'] - r['ind']
    r['n_complete'] = out['n_complete']; r['name'] = name; r['K'] = K
    print('disc', r, flush=True)
    return r


def predict(args):
    variant, K, permute, rep = args
    R = load_skeletons(min_ent=2, clean=False)
    drop = {'SYS', 'TOT'} | ({'CLS', 'CLSR'} if variant == 'strict' else set())
    feats = [k for k in FEATS if k not in drop]
    X, lev = encode([r['f'] for r in R], feats)
    card = [max(1, len(lev[k])) for k in feats]
    HD = [r['f']['HDR'] for r in R]
    rng = np.random.default_rng(1000 + rep)
    if permute:
        X = X[rng.permutation(len(X))]
    idx = rng.permutation(len(R)); parts = np.array_split(idx, 5)
    acc = {k: [] for k in ('marg', 'good', 'hdr', 'form', 'formgood', 'tablet')}
    hit = {k: [] for k in acc}
    for f in range(5):
        te = parts[f]; tr = np.concatenate([parts[g] for g in range(5) if g != f])
        m = best_em(X[tr], card, K, 8, 77 + f + 10 * rep)
        Rtr = m['R']
        # P(sys | form)
        PS = np.full((K, 3), 0.5)
        for a, i in enumerate(tr):
            for e in R[i]['ents']:
                PS[:, SYSC.index(e['sys'])] += Rtr[a]
        PS /= PS.sum(1, keepdims=True)
        glob = np.full(3, 0.5); good = defaultdict(lambda: np.zeros(3)); hdr = defaultdict(lambda: np.zeros(3))
        for i in tr:
            for e in R[i]['ents']:
                j = SYSC.index(e['sys'])
                glob[j] += 1; good[e['final']][j] += 1; hdr[HD[i]][j] += 1
        glob /= glob.sum()
        def smooth(c, w=2.0):
            return (c + w * glob) / (c.sum() + w)
        _, post = lcm_loglik(m, X[te])
        for a, i in enumerate(te):
            pf = post[a] @ PS
            ents = R[i]['ents']
            ys = [SYSC.index(e['sys']) for e in ents]
            for b, e in enumerate(ents):
                y = ys[b]
                pg = smooth(good[e['final']])
                ph = smooth(hdr[HD[i]])
                pfg = pf * pg / glob; pfg /= pfg.sum()
                oth = np.bincount([ys[c] for c in range(len(ys)) if c != b], minlength=3) + glob
                pt = oth / oth.sum()
                for k, p in (('marg', glob), ('good', pg), ('hdr', ph), ('form', pf), ('formgood', pfg),
                             ('tablet', pt)):
                    acc[k].append(-np.log2(p[y])); hit[k].append(int(np.argmax(p) == y))
    res = {'variant': variant, 'K': K, 'permute': permute, 'rep': rep}
    for k in acc:
        res[k] = {'bits': float(np.mean(acc[k])), 'acc': float(np.mean(hit[k]))}
    print('pred', variant, K, permute, rep, {k: round(v['bits'], 4) for k, v in res.items() if isinstance(v, dict)},
          flush=True)
    return res


def main():
    c1 = {r['name']: r for r in json.load(open(os.path.join(DATA, 'pe8_cycle1.json')))}
    Kreal = c1['REAL']['K_cv']
    R = load_skeletons(min_ent=2, clean=False)
    A = [r['A'] for r in R]
    jobs = [('REAL', [r['f'] for r in R], Kreal)]
    for s in (1, 2, 3):
        P, lab, _ = planted_corpus(A, len(A), seed=s)
        jobs.append(('PLANT_%d' % s, [features(a) for a in P], c1['PLANT_%d' % s]['K_cv']))
    for s in (1, 2, 3):
        jobs.append(('NULL_%d' % s, [features(a) for a in null_corpus(A, len(A), seed=s)],
                     max(2, c1['NULL_%d' % s]['K_cv'])))
    pj = []
    for variant in ('main', 'strict'):
        for K in sorted({Kreal, 2, 4, 8}):
            for rep in range(4):
                pj.append((variant, K, False, rep))
        for rep in range(4):
            pj.append((variant, Kreal, True, rep))
    with Pool(2) as p:
        disc = p.map(discreteness, jobs, chunksize=1)
        pred = p.map(predict, pj, chunksize=1)
    json.dump({'disc': disc, 'pred': pred, 'Kreal': Kreal}, open(os.path.join(DATA, 'pe8_cycle2.json'), 'w'),
              indent=1)


if __name__ == '__main__':
    main()
