import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
"""v59 cycle 3c: (a) is the B-herbal ~ A-herbal agreement carried by line-interior words (content)
or by line/paragraph-edge words (layout)? (b) 1,500 random B->A transducers scored on the subject
AUC in ZL3b, top 20 re-tested in IT2a; null: the same search with A's H/P labels permuted.
"""
import sys, json, random
import numpy as np
from v59_lib import *
from v59_c3b import run, auc, vecs
from v59_c2 import side_stats, consensus


def interior(pages):
    out = []
    for p in pages:
        ls = [l[1:-1] for i, l in enumerate(p['lines']) if i > 0 and len(l) > 2]
        out.append(dict(p, lines=ls))
    return out


def edges(pages):
    out = []
    for p in pages:
        ls = [[l[0], l[-1]] if len(l) > 1 else l for l in p['lines']]
        out.append(dict(p, lines=ls))
    return out


def fast_auc(s1, X, Y, s2, POS, NEG, mapper, labs=None):
    import collections
    cnt = collections.Counter(w for p in s1 for l in p['lines'] for w in l)
    voc = [w for w, _ in cnt.most_common(500)]
    M1 = vecs(s1, voc, lambda w: w); df = (M1 > 0).mean(0) + 1e-3; idf = np.log(1 / df)
    nm = lambda M: (lambda Z: Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-9))(np.log1p(M) * idf)
    N1 = nm(M1); N2 = nm(vecs(s2, voc, mapper))
    lab = np.array([p['sec'] for p in s1]) if labs is None else labs
    cx = N1[lab == X].mean(0); cy = N1[lab == Y].mean(0)
    s = N2 @ cx / np.linalg.norm(cx) - N2 @ cy / np.linalg.norm(cy)
    pos = np.array([p['sec'] in POS for p in s2]); neg = np.array([p['sec'] in NEG for p in s2])
    return auc(s[pos], s[neg])


def main():
    rng = np.random.default_rng(77)
    out = []
    for tr in ('ZL3b', 'IT2a'):
        A = voynich(tr, 'A'); B = voynich(tr, 'B')
        out.append(run(f'V_{tr}_interior', interior(A), 'H', 'P', interior(B), {'H'}, {'B', 'S', 'C'}, lambda w: w, rng))
        out.append(run(f'V_{tr}_edges', edges(A), 'H', 'P', edges(B), {'H'}, {'B', 'S', 'C'}, lambda w: w, rng))
    I = isidore()
    s1 = [p for p in I if p['chap'] % 2 == 1]; s2 = [p for p in I if p['chap'] % 2 == 0]
    out.append(run('I_interior', interior(s1), '14', '3', interior(s2), {'14'}, {'9', '10', '11'}, lambda w: w, rng))
    out.append(run('I_edges', edges(s1), '14', '3', edges(s2), {'14'}, {'9', '10', '11'}, lambda w: w, rng))
    # (b) random transducer search on the subject AUC
    AZ, BZ = voynich('ZL3b', 'A'), voynich('ZL3b', 'B')
    AI, BI = voynich('IT2a', 'A'), voynich('IT2a', 'B')
    RS = RuleSampler(BZ, AZ, seed=591, n_top=60, kmax=4)
    Ts = [RS.T(3) for _ in range(1500)]
    labsZ = np.array([p['sec'] for p in AZ]); labsI = np.array([p['sec'] for p in AI])
    res = {}
    for mode in ('real', 'permuted'):
        if mode == 'real':
            lz, li = labsZ, labsI
        else:
            # permute H/P labels jointly by folio so ZL and IT2a share the same fake split
            fol = sorted({p['id'] for p in AZ if p['sec'] in 'HP'})
            r = random.Random(9); hp = {f: s for f, s in zip(fol, [p['sec'] for p in AZ if p['sec'] in 'HP'])}
            vals = list(hp.values()); r.shuffle(vals); fake = dict(zip(sorted(hp), vals))
            lz = np.array([fake.get(p['id'], p['sec']) for p in AZ]); li = np.array([fake.get(p['id'], p['sec']) for p in AI])
        b0z = fast_auc(AZ, 'H', 'P', BZ, {'H'}, {'B', 'S', 'C'}, lambda w: w, lz)
        b0i = fast_auc(AI, 'H', 'P', BI, {'H'}, {'B', 'S', 'C'}, lambda w: w, li)
        dz = np.array([fast_auc(AZ, 'H', 'P', BZ, {'H'}, {'B', 'S', 'C'}, (lambda T: lambda w: apply_T(w, T))(T), lz) - b0z for T in Ts])
        top = np.argsort(-dz)[:20]
        di = np.array([fast_auc(AI, 'H', 'P', BI, {'H'}, {'B', 'S', 'C'}, (lambda T: lambda w: apply_T(w, T))(Ts[j]), li) - b0i for j in top])
        ri = np.array([fast_auc(AI, 'H', 'P', BI, {'H'}, {'B', 'S', 'C'}, (lambda T: lambda w: apply_T(w, T))(Ts[j]), li) - b0i
                       for j in rng.choice(len(Ts), 40, replace=False)])
        res[mode] = {'base_zl': b0z, 'base_it': b0i, 'n_pos_zl': int((dz > 0).sum()), 'top20_dz': float(dz[top].mean()),
                     'top20_di': float(di.mean()), 'n_top_pos_it': int((di > 0).sum()), 'rand_di': float(ri.mean()),
                     'top': [(float(dz[j]), float(d), Ts[j]) for j, d in zip(top[:10], di[:10])]}
        print(mode, {k: v for k, v in res[mode].items() if k != 'top'}, res[mode]['top'][:5], flush=True)
    out.append({'tag': 'random_T_subject', **res})
    json.dump(out, open(os.path.join(CK, 'c3c.json'), 'w'))


if __name__ == '__main__':
    main()
