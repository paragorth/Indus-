import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
"""v59 cycle 2: spelling-blind unsupervised word alignment between two corpus sides.

Each frequent word type is described only by how it is used (frequency, line position,
burstiness over pages, neighbour frequency, repetition, length percentile, the shape of its
co-occurrence row). A self-learning matcher (Hungarian + anchor context vectors, many random
seeds) pairs side-1 types with side-2 types. Controls with a known answer key: Czech modern
vs pre-reform spelling and Isidore vs scribal Latin, both passed through an opaque verbose
encoding (letters -> 1-3 private symbols). Unrelated pairs and Voynich nulls give the floor.
"""
import sys, json, time, random, collections, zlib
import numpy as np
from scipy.optimize import linear_sum_assignment
from multiprocessing import Pool
from v59_lib import *
from v59_c1 import take_tokens

NTOP = 150
SEEDS = int(sys.argv[1]) if len(sys.argv) > 1 else 40


def lev(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def side_stats(pages, ntop=NTOP):
    cnt = collections.Counter(w for p in pages for l in p['lines'] for w in l)
    voc = [w for w, c in cnt.most_common(ntop) if c >= 5]
    idx = {w: i for i, w in enumerate(voc)}
    n = len(voc)
    N = sum(cnt.values())
    first = np.zeros(n); last = np.zeros(n); relpos = np.zeros(n); lnb = np.zeros(n); rnb = np.zeros(n)
    rep = np.zeros(n); pfl = np.zeros(n); npg = np.zeros(n)
    co = np.zeros((n, n))
    lf = {w: np.log(c / N) for w, c in cnt.items()}
    for p in pages:
        seen = set()
        for li, l in enumerate(p['lines']):
            L = len(l)
            ids = [idx.get(w, -1) for w in l]
            for k, (w, i) in enumerate(zip(l, ids)):
                if i < 0:
                    continue
                seen.add(i)
                first[i] += k == 0; last[i] += k == L - 1
                relpos[i] += k / max(L - 1, 1)
                lnb[i] += lf[l[k - 1]] if k > 0 else 0
                rnb[i] += lf[l[k + 1]] if k < L - 1 else 0
                rep[i] += w in l[:k] or w in l[k + 1:]
                pfl[i] += li == 0
            u = [i for i in set(ids) if i >= 0]
            for a in u:
                for b in u:
                    if a != b:
                        co[a, b] += 1
        for i in seen:
            npg[i] += 1
    c = np.array([cnt[w] for w in voc], float)
    npages = len(pages)
    exp_pg = npages * (1 - np.exp(-c / npages))
    lens = np.array([len(w) for w in voc], float)
    lenpct = np.argsort(np.argsort(lens + 1e-3 * np.arange(n))) / max(n - 1, 1)
    tot = co.sum() + 1e-9
    rs = co.sum(1, keepdims=True) + 1e-9
    P = np.maximum(np.log((co / tot) / (rs / tot) / (rs.T / tot) + 1e-12), 0) * (co > 0)
    sig = -np.sort(-P, axis=1)[:, :5]
    F = np.column_stack([np.log(c / N), first / c, last / c, relpos / c, npg / exp_pg, lnb / c, rnb / c,
                         rep / c, pfl / c, lenpct, sig])
    F = (F - F.mean(0)) / (F.std(0) + 1e-9)
    return {'voc': voc, 'F': F, 'co': co, 'cnt': c}


def zrow(M):
    return (M - M.mean(1, keepdims=True)) / (M.std(1, keepdims=True) + 1e-9)


def align_once(S1, S2, seed, iters=8):
    rng = np.random.default_rng(seed)
    d = S1['F'].shape[1]
    wts = rng.uniform(0.2, 1.0, d) * (rng.random(d) < 0.8)
    F1 = S1['F'] * wts; F2 = S2['F'] * wts
    D = ((F1[:, None, :] - F2[None, :, :]) ** 2).sum(2)
    feat = zrow(-D)
    sim = feat + rng.normal(0, 0.3, feat.shape)
    frac = rng.uniform(0.25, 0.5)
    alpha = rng.uniform(0.3, 0.7)
    for _ in range(iters):
        r, c = linear_sum_assignment(-sim)
        marg = sim[r, c] - np.sort(sim, 1)[:, -2][r]
        k = max(10, int(frac * len(r)))
        anc = np.argsort(-marg)[:k]
        a1, a2 = r[anc], c[anc]
        X1 = np.log1p(S1['co'][:, a1]); X2 = np.log1p(S2['co'][:, a2])
        X1 /= np.linalg.norm(X1, axis=1, keepdims=True) + 1e-9
        X2 /= np.linalg.norm(X2, axis=1, keepdims=True) + 1e-9
        ctx = zrow(X1 @ X2.T)
        sim = alpha * feat + (1 - alpha) * ctx
    r, c = linear_sum_assignment(-sim)
    return dict(zip(r.tolist(), c.tolist()))


def consensus(S1, S2, seeds, base=0):
    votes = collections.defaultdict(collections.Counter)
    for s in range(seeds):
        for a, b in align_once(S1, S2, base + s).items():
            votes[a][b] += 1
    out = {}
    for a, v in votes.items():
        b, k = v.most_common(1)[0]
        out[S1['voc'][a]] = (S2['voc'][b], k / seeds)
    return out


def evaluate(name, s1, s2, truth_fn):
    t0 = time.time()
    S1, S2 = side_stats(s1), side_stats(s2)
    full = consensus(S1, S2, SEEDS, zlib.crc32(name.encode()) % 1000)
    # halves stability
    h1a, h1b = split_pages(s1, 21); h2a, h2b = split_pages(s2, 22)
    ca = consensus(side_stats(h1a), side_stats(h2a), SEEDS // 2, 1)
    cb = consensus(side_stats(h1b), side_stats(h2b), SEEDS // 2, 2)
    both = [w for w in ca if w in cb]
    agree = np.mean([ca[w][0] == cb[w][0] for w in both]) if both else 0.0
    # chance agreement: random partner within the same frequency band (+-5 ranks)
    res = {'name': name, 'n1': len(S1['voc']), 'n2': len(S2['voc']), 'halves_agree': float(agree), 'n_both': len(both)}
    # freq-rank-only baseline matcher: pair i-th most frequent with i-th
    fr = {S1['voc'][i]: S2['voc'][i] for i in range(min(len(S1['voc']), len(S2['voc'])))}
    if truth_fn:
        v2 = set(S2['voc'])
        el = [w for w in S1['voc'] if truth_fn(w) in v2]
        res['n_eval'] = len(el)
        res['p1'] = float(np.mean([full[w][0] == truth_fn(w) for w in el])) if el else 0
        res['p1_freqrank'] = float(np.mean([fr.get(w) == truth_fn(w) for w in el])) if el else 0
        conf = [w for w in el if full[w][1] >= 0.6]
        res['p1_conf'] = float(np.mean([full[w][0] == truth_fn(w) for w in conf])) if conf else 0
        res['n_conf'] = len(conf)
        res['chance'] = 1 / max(len(S2['voc']), 1)
    # spelling relation of pairs (meaningful only when both sides share a script)
    pairs = [(a, b, k) for a, (b, k) in full.items()]
    ed = np.mean([lev(a, b) / max(len(a), len(b)) for a, b, _ in pairs])
    rng = random.Random(0)
    bs = [b for _, b, _ in pairs]
    nulls = []
    for _ in range(200):
        rng.shuffle(bs)
        nulls.append(np.mean([lev(a, b) / max(len(a), len(b)) for (a, _, _), b in zip(pairs, bs)]))
    res['ed_pairs'] = float(ed); res['ed_null'] = float(np.mean(nulls)); res['ed_z'] = float((ed - np.mean(nulls)) / (np.std(nulls) + 1e-9))
    res['identical'] = int(sum(a == b for a, b, _ in pairs))
    res['identical_freqrank'] = int(sum(a == b for a, b in fr.items()))
    res['pairs'] = sorted([(a, b, k) for a, b, k in pairs], key=lambda x: -x[2])
    res['secs'] = time.time() - t0
    json.dump(res, open(os.path.join(CK, f'c2_{name}.json'), 'w'), ensure_ascii=False)
    print(name, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in res.items() if k != 'pairs'}, flush=True)
    return res


def jobs():
    A = voynich('ZL3b', 'A'); B = voynich('ZL3b', 'B')
    Ai = voynich('IT2a', 'A'); Bi = voynich('IT2a', 'B')
    I = isidore()
    odd = [p for p in I if p['chap'] % 2 == 1]; even = [p for p in I if p['chap'] % 2 == 0]
    i1 = take_tokens(odd, 11200, 1); i2 = take_tokens(even, 23000, 2)
    C = czech(); c1, c2 = split_pages(C, 5)
    c2o = [dict(p, lines=[[old_czech(w) for w in l] for l in p['lines']]) for p in c2]
    J = []
    enc_cz, kcz = opaque_verbose(c2o, 31)
    J.append(('C_Cz_old_opaque', c1, enc_cz, lambda w, k=kcz: ''.join(k.get(ch, '?') for ch in old_czech(w))))
    J.append(('C_Cz_old_plain', c1, c2o, old_czech))
    i2s = map_pages(i2, LAT_SCRIBAL)
    enc_is, kis = opaque_verbose(i2s, 32)
    J.append(('C_Isid_scribal_opaque', i1, enc_is, lambda w, k=kis: ''.join(k.get(ch, '?') for ch in apply_T(w, LAT_SCRIBAL))))
    enc_i, ki = opaque_verbose(i2, 33)
    J.append(('C_Isid_opaque', i1, enc_i, lambda w, k=ki: ''.join(k.get(ch, '?') for ch in w)))
    a1, a2 = split_pages(A, 7)
    J.append(('P_Vplant', a1, map_pages(a2, VOY_PLANT), lambda w: apply_T(w, VOY_PLANT)))
    J.append(('N_AA', a1, a2, lambda w: w))
    J.append(('U_Cz_Lat_opaque', c1, enc_i, None))
    J.append(('U_Ger_Cz', take_tokens(v30_corpus('G_Bav2'), 11200, 3), c2o, lambda w: w))
    J.append(('L_Ger_BavAlem', take_tokens(v30_corpus('G_Bav2'), 11200, 3), take_tokens(v30_corpus('G_Alem'), 23000, 4), lambda w: w))
    J.append(('V_AB', A, B, lambda w: w))
    J.append(('V_AB_IT', Ai, Bi, lambda w: w))
    J.append(('V_Aherb_Bherb', [p for p in A if p['sec'] == 'H'], [p for p in B if p['sec'] == 'H'], lambda w: w))
    J.append(('V_BA', B, A, lambda w: w))
    J.append(('N_AA_opaque', a1, opaque_verbose(a2, 34)[0], None))
    ka = opaque_verbose(a2, 34)[1]
    J[-1] = ('N_AA_opaque', a1, opaque_verbose(a2, 34)[0], lambda w, k=ka: ''.join(k.get(ch, '?') for ch in w))
    bsh = [dict(p, sec='X') for p in B]
    J.append(('N_A_Bsecshuf', A, word_shuffle(bsh, 4), lambda w: w))
    J.append(('N_A_Bmarkov', A, markov_resynth(B, 3), lambda w: w))
    J.append(('N_A_Bshuf', A, word_shuffle(B, 3), lambda w: w))
    J.append(('N_A_Bbigram', A, word_bigram_resynth(B, 3), lambda w: w))
    return J


J = []


def _run(i):
    return evaluate(*J[i])


if __name__ == '__main__':
    J = jobs()
    only = sys.argv[2].split(',') if len(sys.argv) > 2 else None
    if only:
        J[:] = [j for j in J if j[0] in only]
    with Pool(2) as pool:
        for _ in pool.imap_unordered(_run, range(len(J))):
            pass
