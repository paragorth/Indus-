"""v75 cycle 3b: is the ABC held-out failure (position / head / bigram MI) the Voynich's line machinery? The first word of
every line is dropped from the Voynich chunks and from re-simulations of the accepted grammars (same seeds), and the
held-out error is recomputed. Usage: python3 v75_c3b.py"""
import os, sys, pickle, json, random
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import v75_lib as X, v72_lib as V
import v75_c3 as G

D = pickle.load(open(os.path.join(X.CK, 'c3_sims.pkl'), 'rb'))
TF = D['TYPEF']; FI = [TF.index(f) for f in D['FIT']]; HI = [TF.index(f) for f in D['HOLD']]
S = np.array([s for _, s in D['sims']]); ok = np.isfinite(S).all(1)
idx_all = np.arange(len(S))[ok]; S = S[ok]
med = np.median(S, 0); mad = np.median(np.abs(S - med), 0) * 1.4826 + 1e-6
NACC = 60


def drop_first(pages):
    return [dict(p, lines=[dict(l, w=l['w'][1:]) for l in p['lines'] if len(l['w']) > 1]) for p in pages]


def fp(chs, seed):
    return np.array([np.array(X.fingerprint(c, seed + i))[G.TI] for i, c in enumerate(chs)]).mean(0)


def resim(k, drop):
    i = int(idx_all[k]); seed = 750000 + i
    rng = random.Random(seed); P = G.prior(rng)
    chs = X.chunks(G.skeleton(), 6, seed); g = G.Grammar(P, None, seed)
    filled = [g.fill(ch) for ch in chs]
    if drop: filled = [drop_first(c) for c in filled]
    return fp(filled, seed)


out = {}
for name in ('ZL3b', 'IT2a'):
    pages = X.extract(X.voy_surface(name))
    chs = X.chunks(pages, 12, 4242)
    t_full = fp(chs, 7); t_drop = fp([drop_first(c) for c in chs], 7)
    d = np.sqrt((((S[:, FI] - t_full[FI]) / mad[FI]) ** 2).sum(1)); acc = np.argsort(d)[:NACC]
    R_full = np.array([resim(k, False) for k in acc]); R_drop = np.array([resim(k, True) for k in acc])
    e_full = np.abs(np.median(R_full[:, HI], 0) - t_full[HI]) / mad[HI]
    e_drop = np.abs(np.median(R_drop[:, HI], 0) - t_drop[HI]) / mad[HI]
    out[name] = dict(hold_full=float(e_full.mean()), hold_drop=float(e_drop.mean()),
                     by_full=dict(zip(D['HOLD'], np.round(e_full, 2).tolist())),
                     by_drop=dict(zip(D['HOLD'], np.round(e_drop, 2).tolist())),
                     target_full=dict(zip(TF, np.round(t_full, 3).tolist())), target_drop=dict(zip(TF, np.round(t_drop, 3).tolist())),
                     sims_drop_median=dict(zip(TF, np.round(np.median(R_drop, 0), 3).tolist())))
    print(name, json.dumps(out[name]), flush=True)
json.dump(out, open(os.path.join(X.CK, 'c3b.json'), 'w'), indent=1)
