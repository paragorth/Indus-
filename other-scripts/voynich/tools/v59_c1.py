"""v59 cycle 1: thousands of random transducers, scored by held-out context retrieval.

For each corpus pair (side1 -> side2): split both sides into train/test pages; draw N random
rule sets T; Delta(T) = score(T(side1)) - score(side1) in side-2 space, on train and on test.
Then greedy stacking of the best rules on train, checked on test.
"""
import zlib, sys, json, time, random, collections
import numpy as np
from multiprocessing import Pool
from v59_lib import *

N = int(sys.argv[1]) if len(sys.argv) > 1 else 2000


def take_tokens(pages, n, seed):
    rng = random.Random(seed)
    P = pages[:]
    rng.shuffle(P)
    out, t = [], 0
    for p in P:
        if t >= n:
            break
        out.append(p); t += sum(map(len, p['lines']))
    return out


def pairs():
    A = voynich('ZL3b', 'A'); B = voynich('ZL3b', 'B')
    Ai = voynich('IT2a', 'A'); Bi = voynich('IT2a', 'B')
    out = {}
    out['V_AB'] = (A, B)
    out['V_AB_IT'] = (Ai, Bi)
    out['V_BA'] = (B, A)
    a1, a2 = split_pages(A, 7)
    out['P_Vplant'] = (a1, map_pages(a2, VOY_PLANT))
    out['N_AA'] = (a1, a2)
    out['N_Bmarkov'] = (A, markov_resynth(B, 3))
    out['N_Bshuf'] = (A, word_shuffle(B, 3))
    out['N_Bbigram'] = (A, word_bigram_resynth(B, 3))
    I = isidore()
    odd = [p for p in I if p['chap'] % 2 == 1]; even = [p for p in I if p['chap'] % 2 == 0]
    i1 = take_tokens(odd, 11200, 1); i2 = take_tokens(even, 23000, 2)
    out['C_Isid_scribal'] = (i1, map_pages(i2, LAT_SCRIBAL))
    C = czech()
    c1, c2 = split_pages(C, 5)
    out['C_Cz_old'] = (c1, [dict(p, lines=[[old_czech(w) for w in l] for l in p['lines']]) for p in c2])
    out['L_Ger_BavAlem'] = (take_tokens(v30_corpus('G_Bav2'), 11200, 3), take_tokens(v30_corpus('G_Alem'), 23000, 4))
    out['U_Cz_Lat'] = (c1, i2)
    out['U_Ger_Cz'] = (take_tokens(v30_corpus('G_Bav2'), 11200, 3), c2)
    return out


TRUTH = {'P_Vplant': VOY_PLANT, 'C_Isid_scribal': LAT_SCRIBAL,
         'C_Cz_old': [('sub', a, b) for a, b in OLD_CZ]}


def run(args):
    name, s1, s2 = args
    t0 = time.time()
    s1tr, s1te = split_pages(s1, 11)
    s2tr, s2te = split_pages(s2, 12)
    Str, Ste = Scorer(s2tr), Scorer(s2te)
    b_tr, b_te = Str.score(s1tr)[0], Ste.score(s1te)[0]
    RS = RuleSampler(s1tr, s2tr, seed=zlib.crc32(name.encode()))
    rows = []
    for i in range(N):
        T = RS.T()
        d_tr = Str.score(map_pages(s1tr, T))[0] - b_tr
        d_te = Ste.score(map_pages(s1te, T))[0] - b_te
        rows.append((d_tr, d_te, T))
    dtr = np.array([r[0] for r in rows]); dte = np.array([r[1] for r in rows])
    order = np.argsort(-dtr)
    top = order[:20]
    rng = np.random.default_rng(0)
    null = np.array([dte[rng.choice(len(dte), 20, replace=False)].mean() for _ in range(2000)])
    z_top = (dte[top].mean() - null.mean()) / (null.std() + 1e-12)
    rho = float(np.corrcoef(np.argsort(np.argsort(dtr)), np.argsort(np.argsort(dte)))[0, 1])
    # greedy stacking of single rules drawn from the top 200 sets, on train only
    cand = []
    for j in order[:200]:
        for r in rows[j][2]:
            if r not in cand:
                cand.append(r)
    T, best = [], 0.0
    for _ in range(8):
        gains = []
        for r in cand:
            if r in T:
                continue
            gains.append((Str.score(map_pages(s1tr, T + [r]))[0] - b_tr, r))
        if not gains:
            break
        g, r = max(gains, key=lambda x: x[0])
        if g <= best + 1e-4:
            break
        T.append(r); best = g
    g_te = Ste.score(map_pages(s1te, T))[0] - b_te if T else 0.0
    truth = TRUTH.get(name)
    tr_te = (Ste.score(map_pages(s1te, truth))[0] - b_te) if truth else None
    tr_tr = (Str.score(map_pages(s1tr, truth))[0] - b_tr) if truth else None
    res = {'name': name, 'base_tr': b_tr, 'base_te': b_te, 'rho_tr_te': rho,
           'top20_dte': float(dte[top].mean()), 'z_top20': float(z_top),
           'frac_pos_tr': float((dtr > 0).mean()),
           'frac_both_pos': float(((dtr > 0) & (dte > 0)).mean()),
           'greedy_T': T, 'greedy_dtr': best, 'greedy_dte': g_te,
           'truth_dtr': tr_tr, 'truth_dte': tr_te,
           'top_sets': [(float(dtr[j]), float(dte[j]), rows[j][2]) for j in order[:15]],
           'secs': time.time() - t0}
    json.dump(res, open(os.path.join(CK, f'c1_{name}.json'), 'w'), ensure_ascii=False)
    print(name, f"base {b_tr:.3f}/{b_te:.3f} rho {rho:+.2f} top20 dte {dte[top].mean():+.4f} z {z_top:+.1f} "
          f"greedy {best:+.4f}/{g_te:+.4f} truth {tr_tr} {tr_te} {time.time()-t0:.0f}s", flush=True)
    return res


if __name__ == '__main__':
    P = pairs()
    jobs = [(k, a, b) for k, (a, b) in P.items()]
    with Pool(2) as pool:
        for r in pool.imap_unordered(run, jobs):
            pass
