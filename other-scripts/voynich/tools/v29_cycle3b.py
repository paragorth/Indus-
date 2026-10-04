"""v29 cycle 3b: SIMILARITY ATTRACTION.  Feature-free version of the spread test.
For glyph pairs (a, b) in a relation (word junction: last glyph of a word -> first glyph of the next word;
within word: glyphs 2 or 3 apart), compute the log ratio of observed count to null-expected count
(junction: within-line word shuffle x20; within word: Markov-2 resynthesis x10).  If features spread,
pairs of glyphs that SHARE features should be over-represented: weighted correlation between the log ratio
and the feature similarity of a and b (weighted Jaccard of stroke / phonological feature vectors).
Null: the same glyph permutation applied to rows and columns of the similarity matrix (2,000x), free or
within frequency bins of 4.  Self-pairs (a = a) are excluded so that plain repetition does not count.
usage: python3 v29_cycle3b.py -> data/v29_ckpt/c3_sim.json"""
import os, json, random
import numpy as np
from collections import Counter
import v29_lib as L, v25_shapes as S
from v29_cycle1 import corpus


def jacc(a, b):
    ks = set(a) | set(b)
    return sum(min(a.get(k, 0), b.get(k, 0)) for k in ks) / max(sum(max(a.get(k, 0), b.get(k, 0)) for k in ks), 1e-9)


def pairmat(lines, alph, rel):
    idx = {g: i for i, g in enumerate(alph)}; n = len(alph)
    M = np.zeros((n, n))
    for Ln in lines:
        if rel == 'junc':
            for a, b in zip(Ln, Ln[1:]):
                if a[-1] in idx and b[0] in idx:
                    M[idx[a[-1]], idx[b[0]]] += 1
        else:
            d = int(rel[-1])
            for w in Ln:
                for i in range(len(w) - d):
                    if w[i] in idx and w[i + d] in idx:
                        M[idx[w[i]], idx[w[i + d]]] += 1
    return M


def wcorr(x, y, w):
    mx = (w * x).sum() / w.sum(); my = (w * y).sum() / w.sum()
    return (w * (x - mx) * (y - my)).sum() / np.sqrt((w * (x - mx) ** 2).sum() * (w * (y - my) ** 2).sum())


def test(lines, shapes, rel, R=20, P=2000, seed=0):
    alph, cnt = L.alphabet(lines, shapes)
    O = pairmat(lines, alph, rel)
    if rel == 'junc':
        E = np.mean([pairmat(L.wshuffle(lines, seed + q), alph, rel) for q in range(R)], 0)
    else:
        E = np.mean([pairmat(L.markov2(lines, seed + q), alph, rel) for q in range(R // 2)], 0)
    n = len(alph)
    Sim = np.array([[jacc(shapes[a], shapes[b]) for b in alph] for a in alph])
    mask = (E >= 5) & ~np.eye(n, dtype=bool)
    lr = np.log((O + 0.5) / (E + 0.5))
    w = E[mask]; x = lr[mask]
    obs = wcorr(x, Sim[mask], w)
    rng = np.random.default_rng(seed)
    bins = L.freq_bins(alph, cnt, 4); prng = random.Random(seed)
    nf, nb = [], []
    for _ in range(P):
        p = rng.permutation(n); nf.append(wcorr(x, Sim[np.ix_(p, p)][mask], w))
        q = L.relabel(bins, prng, n); nb.append(wcorr(x, Sim[np.ix_(q, q)][mask], w))
    nf = np.array(nf); nb = np.array(nb)
    # top attracting distinct-glyph pairs
    cells = [(lr[i, j], alph[i], alph[j], int(O[i, j]), float(E[i, j])) for i in range(n) for j in range(n) if mask[i, j]]
    cells.sort(reverse=True)
    return dict(r=float(obs), p_free=float((nf >= obs).mean()), z_free=float((obs - nf.mean()) / nf.std()),
                p_bin=float((nb >= obs).mean()), z_bin=float((obs - nb.mean()) / nb.std()), ncells=int(mask.sum()),
                top=[(round(float(a), 2), b, c, d, round(e, 1)) for a, b, c, d, e in cells[:6]])


if __name__ == '__main__':
    out = {}
    for name in ['tr', 'hu', 'fi', 'cs', 'la', 'ko', 'plant7', 'ZL', 'IT', 'ZL_A', 'ZL_B', 'copy5', 'copy8']:
        lines, sh = corpus(name)
        out[name] = {rel: test(lines, sh, rel) for rel in ('junc', 'raw2', 'raw3')}
        print(name, json.dumps({k: {kk: v[kk] for kk in ('r', 'z_free', 'p_free', 'z_bin', 'p_bin')} for k, v in out[name].items()}), flush=True)
    json.dump(out, open(os.path.join(L.CK, 'c3_sim.json'), 'w'), indent=1)
