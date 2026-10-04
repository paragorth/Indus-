#!/usr/bin/env python3
"""LA-37 cycle 2: the Linear A doublet search, and independent checks.
1. All pairs of the 64 signs with >= 8 tokens (2+ sign words), frozen statistic all3; within-word-shuffle null
   R=400 (per-pair z, BH q, max-z FWER) + frequency-matched percentile. Flag = q <= 0.1 and fm >= 0.95.
2. Stability: HT vs non-HT halves; 30 document bootstraps (share of resamples in the top 5 % of T).
3. Blind la21 consonant rows (independent of values): same-row probability of top / flagged pairs vs all pairs
   (permutation of pair sets); Spearman(T, P_row). Circularity control: T computed on a within-word-shuffled corpus.
   Linear B control of the same test (LB la21 rows vs LB T).
4. Outside check only: Linear B-derived values of the LA signs (same consonant / same vowel AUC).
"""
import os, sys, json, collections, time
os.environ.setdefault('OMP_NUM_THREADS', '1')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from scipy.stats import spearmanr
from multiprocessing import Pool
import la37_common as K
import la32_common as C

LOG = os.path.join(K.CK, 'c2.log')


def boot(seed):
    rng = np.random.default_rng(seed)
    U = K.la_units()
    by = collections.defaultdict(list)
    for r in U:
        by[r['doc']].append(r)
    docs = list(by)
    u = []
    for i, d in enumerate(rng.choice(len(docs), len(docs))):
        for r in by[docs[d]]:
            rr = dict(r); rr['doc'] = f'{docs[d]}#{i}'; u.append(rr)
    al, c = K.alphabet(U, 8)
    iu = np.triu_indices(len(al), 1)
    T = K.all3(K.stats(u, al, K.doc_halves(u, seed)), iu)
    return (T >= np.quantile(T, 0.95)).astype(int)


def subset(tag):
    U = K.la_units()
    u = [r for r in U if (r['site'] == 'Haghia Triada') == (tag == 'HT')]
    al, c = K.alphabet(K.la_units(), 8)
    iu = np.triu_indices(len(al), 1)
    return tag, K.all3(K.stats(u, al, K.doc_halves(u, 3)), iu)


def main_score(_):
    U = K.la_units(); al, c = K.alphabet(U, 8)
    return K.score2(U, al, c, R=400, seed=11)


def shuf_T(seed):
    U = K.la_units(); al, c = K.alphabet(U, 8)
    rng = np.random.default_rng(seed)
    u = K.shuffle_within(U, rng)
    return K.all3(K.stats(u, al, K.doc_halves(u, seed)), np.triu_indices(len(al), 1))


def lb_T(_):
    B = K.lb_units(); al, c = K.alphabet(B, 10)
    return al, K.all3(K.stats(B, al, K.doc_halves(B, 0)), np.triu_indices(len(al), 1))


def rowtest(al, iu, T, signs, P, sel, nperm=20000, seed=0):
    """mean same-row probability over selected pairs vs random pair sets of the same size (pairs inside la21 set)."""
    ix = {s: i for i, s in enumerate(signs)}
    ok = np.array([al[a] in ix and al[b] in ix for a, b in zip(*iu)])
    pr = np.array([P[ix[al[a]], ix[al[b]]] if o else np.nan for (a, b), o in zip(zip(*iu), ok)])
    s = sel & ok
    rho = spearmanr(T[ok], pr[ok]).correlation
    rng = np.random.default_rng(seed)
    pool = pr[ok]; k = int(s.sum())
    if k == 0:
        return dict(k=0, rho=rho)
    obs = pr[s].mean()
    null = np.array([rng.choice(pool, k, replace=False).mean() for _ in range(nperm)])
    return dict(k=k, obs=float(obs), base=float(pool.mean()), p=float((1 + (null >= obs).sum()) / (nperm + 1)),
                rho=float(rho), pr=pr)


if __name__ == '__main__':
    t0 = time.time()
    with Pool(2) as p:
        a_main = p.apply_async(main_score, (0,))
        a_boot = p.map_async(boot, range(30))
        a_sub = p.map_async(subset, ['HT', 'nonHT'])
        a_sh = p.map_async(shuf_T, range(20))
        a_lb = p.apply_async(lb_T, (0,))
        res = a_main.get(); boots = np.array(a_boot.get()); subs = dict(a_sub.get()); shT = a_sh.get()
        lbal, lbT = a_lb.get()
    al, iu, T = res['alph'], res['iu'], res['T']
    U = K.la_units(); _, cnt = K.alphabet(U, 8)
    stab = boots.mean(0)
    flag = K.flagged(res)
    top = np.zeros(len(T), bool); top[np.argsort(-T)[:30]] = True
    rows = C.la21_rows()
    sLA, PLA, nLA = rows['LA']; sLB, PLB, nLB = rows['LB']
    out = dict(alph=al, T=T.tolist(), z=res['z'].tolist(), q=res['q'].tolist(), fwer=res['fwer'].tolist(),
               fm=res['fm'].tolist(), stab=stab.tolist(), HT=subs['HT'].tolist(), nonHT=subs['nonHT'].tolist())
    lines = []
    lines.append(f'signs {len(al)}, pairs {len(T)}, flagged {int(flag.sum())}, FWER<=0.05 {int((res["fwer"] <= 0.05).sum())}, '
                 f'q<=0.1 {int((res["q"] <= 0.1).sum())}, fm>=0.95 {int((res["fm"] >= 0.95).sum())}')
    rho_ht = spearmanr(subs['HT'], subs['nonHT']).correlation
    lines.append(f'HT vs nonHT T correlation rho {rho_ht:.3f}')
    order = np.argsort(-T)
    ix = {s: i for i, s in enumerate(sLA)}
    def prow(a, b):
        return PLA[ix[a], ix[b]] if a in ix and b in ix else np.nan
    cand = []
    for j in order[:40]:
        a, b = al[iu[0][j]], al[iu[1][j]]
        cv = (C.lb_cv(a.lower()), C.lb_cv(b.lower()))
        cand.append(dict(a=a, b=b, na=cnt[a], nb=cnt[b], T=round(float(T[j]), 3), z=round(float(res['z'][j]), 2),
                         q=round(float(res['q'][j]), 3), fwer=round(float(res['fwer'][j]), 3), fm=round(float(res['fm'][j]), 3),
                         stab=round(float(stab[j]), 2), ht_pct=round(float((subs['HT'] < subs['HT'][j]).mean()), 2),
                         nonht_pct=round(float((subs['nonHT'] < subs['nonHT'][j]).mean()), 2),
                         row=round(float(prow(a, b)), 3), flag=bool(flag[j]), lbcv=str(cv)))
    for d in cand:
        lines.append(json.dumps(d))
    # flagged list (all)
    fl = [dict(a=al[iu[0][j]], b=al[iu[1][j]], T=round(float(T[j]), 3), z=round(float(res['z'][j]), 2),
               q=round(float(res['q'][j]), 3), fm=round(float(res['fm'][j]), 3), stab=round(float(stab[j]), 2),
               row=round(float(prow(al[iu[0][j]], al[iu[1][j]])), 3)) for j in np.where(flag)[0]]
    lines.append('FLAGGED ' + json.dumps(fl))
    # row tests
    for name, sel in (('top30', top), ('flagged', flag), ('stable>=0.5', stab >= 0.5)):
        r = rowtest(al, iu, T, sLA, PLA, sel)
        lines.append(f'LA rows {name}: k {r["k"]} same-row {r.get("obs", np.nan):.3f} vs base {r.get("base", np.nan):.3f} '
                     f'P {r.get("p", np.nan):.4f}; Spearman(T, Prow) {r["rho"]:.3f}')
    shr = [rowtest(al, iu, t, sLA, PLA, np.zeros(len(t), bool))['rho'] for t in shT]
    lines.append(f'circularity: Spearman(T_shuffled, Prow) mean {np.mean(shr):.3f} sd {np.std(shr):.3f} (20 within-word shuffles)')
    lbiu = np.triu_indices(len(lbal), 1)
    toplb = np.zeros(len(lbT), bool); toplb[np.argsort(-lbT)[:30]] = True
    r = rowtest(lbal, lbiu, lbT, sLB, PLB, toplb)
    lines.append(f'LB control rows top30: same-row {r["obs"]:.3f} vs base {r["base"]:.3f} P {r["p"]:.4f}; Spearman {r["rho"]:.3f}')
    # outside check: LB-derived values of LA signs
    labs = np.array([K.lb_label(al[a].lower(), al[b].lower()) for a, b in zip(*iu)], dtype=object)
    for g in ('sameC', 'sameV'):
        lines.append(f'outside check (LB values on LA signs) {g}: n {int((labs == g).sum())} AUC {K.auc(T[labs == g], T[labs == "other"]):.3f}')
    rel = np.isin(labs, ['sameC', 'sameV', 'doublet'])
    for name, sel in (('top30', top), ('flagged', flag)):
        s = sel & (labs != None)
        lines.append(f'outside check {name}: related {rel[s].mean() if s.sum() else np.nan:.3f} (n {int(s.sum())}) vs base {rel[labs != None].mean():.3f}')
    json.dump(out, open(os.path.join(K.CK, 'c2.json'), 'w'))
    for l in lines:
        K.log(LOG, l)
    K.log(LOG, f'done {time.time() - t0:.0f}s')
