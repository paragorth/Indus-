#!/usr/bin/env python3
"""LA-41 cycle 1: find the near-copy families, calibrated.

For each corpus (Linear A; Linear B KN+PY) and each of 7 alignment schemes (5 local
Smith-Waterman variants, 2 order-free matchings), every candidate document pair is scored by the
information of the words it shares in alignment (rarer shared words weigh more). The family
threshold T is set per scheme so that a site-word-shuffle null (words permuted among all items of
the same site; list lengths, numbers, logograms kept) yields at most 10 % as many pairs as the real
corpus (FDR 0.1). A family is robust when >= 4 of 7 schemes call it.
Planted control: 15 copies of random LA lists with known edits, 10 replicates.
"""
import sys, os, json, pickle, random, time
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from la41_common import *

OUT = os.path.join(HERE, '..', 'loops', 'la41_cycle1.txt')
GRID = [x / 2 for x in range(0, 60)]


def run_corpus(docs):
    cands = candidate_pairs(docs); SD = site_df(docs)
    return {s: [(a, b, sc, n, cv) for a, b, sc, n, cv, p, wm in scored_pairs(docs, s, cands, SD)] for s in ALL_SCHEMES}


def null_job(args):
    tag, seed = args
    docs = la_docs() if tag == 'LA' else lb_docs()
    rng = random.Random(seed)
    return run_corpus(site_word_shuffle(docs, rng))


def plant_job(seed):
    rng = random.Random(1000 + seed)
    docs = la_docs()
    signs = sorted({s for d in docs for it in docs[d]['items'] if it['w'] for s in it['w']})
    common = [s for s, _ in Counter(s for d in docs for it in docs[d]['items'] if it['w'] for s in it['w']).most_common(40)]
    perm = common[:]; rng.shuffle(perm)
    sub_table = {common[i]: perm[i] for i in range(0, 20) if common[i] != perm[i]}
    new, truth = plant_copies(docs, 15, rng, sub_table)
    return truth, run_corpus(new)


def thresholds(real, nulls):
    T = {}
    for s in ALL_SCHEMES:
        T[s] = None
        for t in GRID:
            r = sum(1 for x in real[s] if x[2] >= t)
            n = sum(sum(1 for x in nu[s] if x[2] >= t) for nu in nulls) / len(nulls)
            if r >= 1 and n <= 0.1 * r:
                T[s] = (t, r, n); break
    return T


def families(res, T):
    calls = Counter(); info = {}
    for s in ALL_SCHEMES:
        if T[s] is None: continue
        for a, b, sc, n, cv in res[s]:
            if sc >= T[s][0]:
                calls[(a, b)] += 1
                info[(a, b)] = max(info.get((a, b), (0, 0, 0)), (sc, n, cv))
    return calls, info


def main():
    t0 = time.time()
    LA = la_docs(); LB = lb_docs()
    realLA = run_corpus(LA); realLB = run_corpus(LB)
    with Pool(2) as P:
        nLA = P.map(null_job, [('LA', s) for s in range(100)])
        nLB = P.map(null_job, [('LB', s) for s in range(12)])
        pl = P.map(plant_job, range(10))
    pickle.dump(dict(realLA=realLA, realLB=realLB, nLA=nLA, nLB=nLB, pl=pl), open(os.path.join(CK, 'c1.pkl'), 'wb'))
    TLA = thresholds(realLA, nLA); TLB = thresholds(realLB, nLB)
    rows = []
    for tag, real, nul, T in (('LA', realLA, nLA, TLA), ('LB', realLB, nLB, TLB)):
        calls, info = families(real, T)
        rob = sorted([k for k, c in calls.items() if c >= 4], key=lambda k: -info[k][0])
        # null robust family count
        nrob = []
        for nu in nul:
            c2, _ = families(nu, T); nrob.append(sum(1 for v in c2.values() if v >= 4))
        thr = '; '.join(f"{s} T={T[s][0]} real {T[s][1]} null {T[s][2]:.2f}" if T[s] else f"{s} none" for s in ALL_SCHEMES)
        fams = ', '.join(f"{a}~{b} ({calls[(a, b)]}/7, n{info[(a, b)][1]}, cov {info[(a, b)][2]:.2f})" for a, b in rob[:60])
        rows.append(f"| LA-41.1{'a' if tag == 'LA' else 'b'} | {tag}: family search, 7 schemes, threshold per scheme at FDR 0.1 against site-word shuffle ({len(nul)} reps); robust = >= 4/7 schemes. | Thresholds: {thr}. Robust families {len(rob)} (null robust mean {sum(nrob) / len(nrob):.2f}, max {max(nrob)}): {fams}{' ...' if len(rob) > 60 else ''} | {'see c' if True else ''} |")
        json.dump({'T': T, 'rob': [list(k) + list(info[k]) + [calls[k]] for k in rob]}, open(os.path.join(CK, f'c1_fam_{tag}.json'), 'w'))
    # planted recovery with LA thresholds
    rec = []; fp = []
    for truth, res in pl:
        calls, info = families(res, TLA)
        got = sum(1 for a, b in truth['pairs'] if calls.get((min(a, b), max(a, b)), 0) >= 4)
        rec.append(got / len(truth['pairs']))
        fp.append(sum(1 for (a, b), c in calls.items() if c >= 4 and ('PLANT' in a or 'PLANT' in b)
                      and (min(a, b), max(a, b)) not in {(min(x, y), max(x, y)) for x, y in truth['pairs']}))
    rows.append(f"| LA-41.1c | Planted control: 15 copies of random LA lists (>= 4 words) per replicate with drop 0.15, add 0.1, adjacent swap 0.1, spelling substitution 0.15, amount change 0.5, logogram change 0.05; LA thresholds; 10 reps. | Planted families recovered (>= 4/7 schemes) mean {sum(rec) / len(rec):.2f} (min {min(rec):.2f}, max {max(rec):.2f}); spurious robust families touching a plant {sum(fp) / len(fp):.2f}. | {'PASS' if sum(rec) / len(rec) >= 0.5 else 'WEAK'} |")
    for r in rows: wlog(OUT, r)
    print('\n'.join(rows)); print('time', time.time() - t0)


if __name__ == '__main__':
    main()
