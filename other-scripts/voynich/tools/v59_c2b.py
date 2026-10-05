import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
"""v59 cycle 2b: do spelling-blind A<->B word pairs recur across independent page halves?

For each pair of sides: 8 replicates, each aligning a random half of side-1 pages with a random
half of side-2 pages (20 seeds per replicate). A pair (a, b) 'recurs' if it is the consensus
partner in >= 4 of 8 replicates. Nulls are re-drawn per replicate (fresh Markov / shuffle seeds).
"""
import sys, json, collections, random
import numpy as np
from multiprocessing import Pool
from v59_lib import *
from v59_c1 import take_tokens
from v59_c2 import side_stats, consensus, lev

R = 8
J = {}


def build():
    A = voynich('ZL3b', 'A'); B = voynich('ZL3b', 'B')
    Ai = voynich('IT2a', 'A'); Bi = voynich('IT2a', 'B')
    C = czech(); c1, c2 = split_pages(C, 5)
    c2o = [dict(p, lines=[[old_czech(w) for w in l] for l in p['lines']]) for p in c2]
    enc, k = opaque_verbose(c2o, 31)
    bav = take_tokens(v30_corpus('G_Bav2'), 22400, 3); alem = take_tokens(v30_corpus('G_Alem'), 46000, 4)
    a1, a2 = split_pages(A, 7)
    J['V_AB'] = (A, lambda r: B, lambda w: w)
    J['V_AB_IT'] = (Ai, lambda r: Bi, lambda w: w)
    J['N_A_Bmarkov'] = (A, lambda r: markov_resynth(B, 100 + r), lambda w: w)
    J['N_A_Bmarkov3'] = (A, lambda r: markov_resynth(B, 200 + r, order=3), lambda w: w)
    J['N_A_Bshuf'] = (A, lambda r: word_shuffle(B, 100 + r), lambda w: w)
    J['N_A_Bbigram'] = (A, lambda r: word_bigram_resynth(B, 100 + r), lambda w: w)
    J['N_AA'] = (A, lambda r: A, lambda w: w)       # halves of A against other halves of A
    J['C_Cz_old_opaque'] = (C, lambda r: opaque_verbose([dict(p, lines=[[old_czech(w) for w in l] for l in p['lines']]) for p in C], 31)[0],
                            lambda w, k=opaque_verbose([dict(p, lines=[[old_czech(w) for w in l] for l in p['lines']]) for p in C], 31)[1]: ''.join(k.get(ch, '?') for ch in old_czech(w)))
    J['L_Ger_BavAlem'] = (bav, lambda r: alem, lambda w: w)
    J['U_Ger_Cz'] = (bav, lambda r: c2o, lambda w: w)


def run(name):
    s1, mk2, truth = J[name]
    reps = []
    for r in range(R):
        s2 = mk2(r)
        x1, y1 = split_pages(s1, 1000 + r)
        x2, y2 = split_pages(s2, 2000 + r)
        if name in ('N_AA', 'C_Cz_old_opaque'):
            y2 = [p for p in s2 if p['id'] in {q['id'] for q in y1}]     # disjoint pages of the same text
        S1, S2 = side_stats(x1), side_stats(y2)
        al = consensus(S1, S2, 20, 50 * r)
        v2 = set(S2['voc'])
        ev = [w for w in S1['voc'] if truth(w) in v2]
        p1 = float(np.mean([al[w][0] == truth(w) for w in ev])) if ev else 0.0
        pairs = [(a, b) for a, (b, _) in al.items()]
        ed = float(np.mean([lev(a, b) / max(len(a), len(b)) for a, b in pairs]))
        rng = random.Random(r); bs = [b for _, b in pairs]; nl = []
        for _ in range(100):
            rng.shuffle(bs); nl.append(np.mean([lev(a, b) / max(len(a), len(b)) for (a, _), b in zip(pairs, bs)]))
        reps.append({'p1': p1, 'n_eval': len(ev), 'ed_z': float((ed - np.mean(nl)) / (np.std(nl) + 1e-9)), 'pairs': pairs})
    cnt = collections.Counter(p for rp in reps for p in rp['pairs'])
    rec = sorted([(a, b, c) for (a, b), c in cnt.items() if c >= 4], key=lambda x: -x[2])
    res = {'name': name, 'p1': [x['p1'] for x in reps], 'n_eval': [x['n_eval'] for x in reps],
           'ed_z': [x['ed_z'] for x in reps], 'n_recur4': len(rec), 'n_recur3': sum(c >= 3 for c in cnt.values()),
           'recur': rec, 'n_recur4_nonid': sum(a != b for a, b, _ in rec)}
    json.dump(res, open(os.path.join(CK, f'c2b_{name}.json'), 'w'), ensure_ascii=False)
    print(name, 'p1 %.3f+-%.3f' % (np.mean(res['p1']), np.std(res['p1'])), 'ed_z %.2f' % np.mean(res['ed_z']),
          'recur>=4/8', len(rec), 'nonid', res['n_recur4_nonid'], 'recur>=3', res['n_recur3'], rec[:12], flush=True)


if __name__ == '__main__':
    build()
    with Pool(2) as pool:
        list(pool.imap_unordered(run, list(J)))
