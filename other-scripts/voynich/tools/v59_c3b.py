import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
"""v59 cycle 3b: does a learned A<->B mapping tell which B pages share A's subject?

Side 1 has two labelled groups X and Y (Voynich A: herbal H vs pharma P). Side-2 pages are
translated into side-1 words by a mapping (identity; the spelling-blind dictionary of cycle 2;
the v30 rewrite) and scored s = cos(page, X centroid) - cos(page, Y centroid) over tf-idf of the
500 most frequent side-1 types. AUC: side-2 X-subject pages (B herbal) vs side-2 pages of other
subjects (B biological, stars, cosmological). Null: side-1 labels permuted among X+Y pages.
Control: Isidore, odd chapters vs even chapters; X = book XVI (stones, metals), Y = book IV
(medicine); side-2 'other' = books XI-XIII (man, animals, world). Side 2 plain, scribal, or
opaque-encoded and translated back through the spelling-blind dictionary.
"""
import sys, json, collections, random
import numpy as np
from v59_lib import *
from v59_c2 import side_stats, consensus
import v30_lib

NPERM = 2000


def vecs(pages, voc, mapper):
    idx = {w: i for i, w in enumerate(voc)}
    M = np.zeros((len(pages), len(voc)))
    for r, p in enumerate(pages):
        for l in p['lines']:
            for w in l:
                j = idx.get(mapper(w))
                if j is not None:
                    M[r, j] += 1
    return M


def auc(pos, neg):
    pos, neg = np.asarray(pos), np.asarray(neg)
    return float(((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean()))


def run(tag, s1, X, Y, s2, POS, NEG, mapper, rng):
    cnt = collections.Counter(w for p in s1 for l in p['lines'] for w in l)
    voc = [w for w, _ in cnt.most_common(500)]
    M1 = vecs(s1, voc, lambda w: w)
    df = (M1 > 0).mean(0) + 1e-3
    idf = np.log(1 / df)
    def norm(M):
        M = np.log1p(M) * idf
        return M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-9)
    N1 = norm(M1)
    M2 = vecs(s2, voc, mapper)
    cover = float(M2.sum() / sum(len(l) for p in s2 for l in p['lines']))
    N2 = norm(M2)
    lab = np.array([p['sec'] for p in s1])
    xy = np.where(np.isin(lab, [X, Y]))[0]
    pos = np.array([p['sec'] in POS for p in s2]); neg = np.array([p['sec'] in NEG for p in s2])
    def score(labs):
        cx = N1[xy[labs == X]].mean(0); cy = N1[xy[labs == Y]].mean(0)
        s = N2 @ cx / (np.linalg.norm(cx) + 1e-9) - N2 @ cy / (np.linalg.norm(cy) + 1e-9)
        return auc(s[pos], s[neg])
    real = score(lab[xy])
    null = np.array([score(rng.permutation(lab[xy])) for _ in range(NPERM)])
    res = {'tag': tag, 'auc': real, 'null_mean': float(null.mean()), 'null_sd': float(null.std()),
           'z': float((real - null.mean()) / (null.std() + 1e-12)), 'p': float((null >= real).mean()),
           'coverage': cover, 'n_pos': int(pos.sum()), 'n_neg': int(neg.sum())}
    print(res, flush=True)
    return res


def dict_from_alignment(s_from, s_to, seeds=40):
    al = consensus(side_stats(s_from), side_stats(s_to), seeds, 7)
    return {a: b for a, (b, k) in al.items()}


def main():
    rng = np.random.default_rng(5959)
    out = []
    # ---------------- Isidore control
    I = isidore()
    s1 = [p for p in I if p['chap'] % 2 == 1]
    s2 = [p for p in I if p['chap'] % 2 == 0]
    X, Y, NEG = '14', '3', {'9', '10', '11'}
    out.append(run('I_plain_identity', s1, X, Y, s2, {X}, NEG, lambda w: w, rng))
    s2s = map_pages(s2, LAT_SCRIBAL)
    out.append(run('I_scribal_identity', s1, X, Y, s2s, {X}, NEG, lambda w: w, rng))
    s2o, key = opaque_verbose(s2s, 41)
    out.append(run('I_opaque_identity', s1, X, Y, s2o, {X}, NEG, lambda w: w, rng))
    # spelling-blind dictionary learned at Voynich-like size (11k vs 23k tokens) from other pages
    D = dict_from_alignment(s2o, s1)
    out.append(run('I_opaque_aligner_dict', s1, X, Y, s2o, {X}, NEG, lambda w: D.get(w, ''), rng))
    rnd = list(D.values()); random.Random(1).shuffle(rnd); Dr = dict(zip(D.keys(), rnd))
    out.append(run('I_opaque_shuffled_dict', s1, X, Y, s2o, {X}, NEG, lambda w: Dr.get(w, ''), rng))
    # ---------------- Voynich
    for tr in ('ZL3b', 'IT2a'):
        A = voynich(tr, 'A'); B = voynich(tr, 'B')
        out.append(run(f'V_{tr}_identity', A, 'H', 'P', B, {'H'}, {'B', 'S', 'C'}, lambda w: w, rng))
        D = dict_from_alignment(B, A)
        out.append(run(f'V_{tr}_aligner_dict', A, 'H', 'P', B, {'H'}, {'B', 'S', 'C'}, lambda w: D.get(w, ''), rng))
        rnd = list(D.values()); random.Random(2).shuffle(rnd); Dr = dict(zip(D.keys(), rnd))
        out.append(run(f'V_{tr}_shuffled_dict', A, 'H', 'P', B, {'H'}, {'B', 'S', 'C'}, lambda w: Dr.get(w, ''), rng))
        r = json.load(open(os.path.join(ROOT, 'data', 'v30_ckpt', 'g3_V_BA_s0.json')))['path'][-1]['rules'][:10]
        rb = v30_lib.rules_by_first([tuple(x) for x in r])
        out.append(run(f'V_{tr}_v30_BA10', A, 'H', 'P', B, {'H'}, {'B', 'S', 'C'}, lambda w: v30_lib.apply_word(w, rb), rng))
        # glyph-only view: identity on words reduced to their glyph skeleton (endings stripped) to gauge spelling-drift
        sk = lambda w: w[:3]
        A3 = [dict(p, lines=[[sk(w) for w in l] for l in p['lines']]) for p in A]
        out.append(run(f'V_{tr}_first3glyphs', A3, 'H', 'P', B, {'H'}, {'B', 'S', 'C'}, sk, rng))
        sk2 = lambda w: w[-3:]
        A4 = [dict(p, lines=[[sk2(w) for w in l] for l in p['lines']]) for p in A]
        out.append(run(f'V_{tr}_last3glyphs', A4, 'H', 'P', B, {'H'}, {'B', 'S', 'C'}, sk2, rng))
    json.dump(out, open(os.path.join(CK, 'c3b.json'), 'w'))


if __name__ == '__main__':
    main()
