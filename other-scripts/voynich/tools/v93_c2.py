"""v93 cycle 2: checkerboard (mutual-exclusion) word sets, random set search on train units, held-out replication."""
import os, sys, json, random, time, itertools
from collections import Counter, defaultdict
from multiprocessing import Pool
import numpy as np
import v93_ex as E
import v93_lib as L

CORPORA = ['HERB_IT', 'HERB_LA', 'HERB_KO', 'HERB_IT~gen', 'ZL3b', 'IT2a', 'SELFCIT', 'MK2', 'JUNC', 'WSHUF']
CONFIGS = [(lv, rp) for lv in ('page', 'para') for rp in ('whole', 'core', 'frame')]
NSETS = 20000; TOP = 50; R = 40


def zmat(U, rep, vocab=None, seed=0):
    X, voc = E.presence(U, rep, vocab=vocab)
    obs, mu, sd = E.cooc_z(X, [u[1] for u in U], R=R, seed=seed)
    z = (mu - obs) / sd; np.fill_diagonal(z, 0)
    return z, voc, X, mu


def run(job):
    name, (lv, rp) = job
    out = os.path.join(L.CK, 'c2_%s_%s_%s.json' % (name.replace('~', '_'), lv, rp))
    if os.path.exists(out): return out
    P = E.corpus(name); U = E.units(P, lv)
    tr = [u for u in U if u[2] == 0]; te = [u for u in U if u[2] != 0]
    if len(tr) < 25 or len(te) < 20:
        json.dump(dict(skip=True, n_tr=len(tr), n_te=len(te)), open(out, 'w')); return out
    ztr, voc, Xtr, mutr = zmat(tr, rp, seed=1)
    zte, _, Xte, mute = zmat(te, rp, vocab=voc, seed=2)
    # eligible words: expected to co-occur in held-out units (df >= 3 there)
    ok = np.nonzero((Xte.sum(0) >= 3) & (Xtr.sum(0) >= 3))[0]
    rng = random.Random(931)
    cand = []
    for _ in range(NSETS):                                     # massive random guessing: random k-sets
        k = rng.choice([2, 2, 3, 3, 4])
        S = rng.sample(list(ok), k)
        cand.append(S)
    # guided guesses: greedy growth from random seed pairs (still random starts)
    for _ in range(NSETS // 4):
        a = rng.choice(ok); b = ok[int(np.argmax(ztr[a, ok] + 1e-3 * np.array([rng.random() for _ in ok])))]
        S = [a, b]
        for _ in range(rng.choice([0, 1, 2])):
            sc = ztr[S][:, ok].mean(0); sc[[list(ok).index(x) for x in S if x in ok]] = -9
            S.append(ok[int(np.argmax(sc))])
        cand.append(S)
    def sc(z, S): return float(np.mean([z[a, b] for a, b in itertools.combinations(S, 2)]))
    scored = sorted(((sc(ztr, S), S) for S in cand), key=lambda t: -t[0])
    top, used = [], set()
    for s, S in scored:                                        # top sets with disjoint members
        if used & set(S): continue
        top.append((s, S)); used |= set(S)
        if len(top) >= TOP: break
    held = [sc(zte, S) for _, S in top]
    allz = zte[np.ix_(ok, ok)][np.triu_indices(len(ok), 1)]
    res = dict(name=name, lv=lv, rep=rp, n_tr=len(tr), n_te=len(te), V=len(voc), n_ok=len(ok),
               top_train=[s for s, _ in top], top_held=held, mean_held=float(np.mean(held)),
               rep_rate=float(np.mean([h > 1.0 for h in held])), pair_held_mean=float(allz.mean()), pair_held_sd=float(allz.std()),
               sets=[[voc[i] for i in S] for _, S in top])
    if 'orig' in P[0]['lines'][0]:
        dec = defaultdict(Counter)
        for p in P:
            for l in p['lines']:
                for w, o in zip(l['w'], l['orig']): dec[L.rep(w, rp)][o] += 1
        res['decoded'] = [[dec[voc[i]].most_common(1)[0][0] for i in S] for _, S in top]
    json.dump(res, open(out, 'w'))
    return out


if __name__ == '__main__':
    t = time.time()
    jobs = [(n, c) for n in CORPORA for c in CONFIGS]
    with Pool(2) as pool:
        for o in pool.imap_unordered(run, jobs):
            print(os.path.basename(o), round(time.time() - t), flush=True)
