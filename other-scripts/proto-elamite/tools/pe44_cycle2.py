"""pe44 cycle 2: survivor models applied to real PE.

Survivors from cycle 1 (Ur III held-out, system- and size-matched AUC + planted PE recovery).
Each survivor is refit on real PE; the 'planned' class is named blind (plan_score).
Ensemble plannedness pi_e = mean planned posterior over survivors.

(a) Held-out stability: survivors fit on PE tablet half A vs half B; Spearman of pi on all entries;
    the same for PE with values shuffled across entries within system (null).
(b) Sign / position scan: mean pi per key (final sign, first sign, header, any sign, surface,
    system) vs two nulls computed with the SAME fitted models:
      N1 values shuffled across entries within system (features recomputed), 200 reps
      N2 synthetic Poisson counts: each tablet x system total redistributed multinomially over its entries
    Planted sign: fake key on 40 random entries whose values were rounded to their leading denomination
    (must be flagged); BH over keys with >= 15 entries.
(c) Replication: flags (q < 0.1 vs N1) on tablet half A re-tested on half B.
"""
import json, math, os, random, sys
from collections import defaultdict
import numpy as np
from scipy.stats import spearmanr
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pe44_common import *  # noqa

C1 = os.path.join(CK, 'c1_models.jsonl')


def survivors(nmax=30):
    """No model passes the per-quantity Ur III test (system- and size-matched AUC >= 0.6: 0 models),
    so survivors are chosen on the planted PE recovery and the size-matched Ur III AUC."""
    rs = [json.loads(l) for l in open(C1)]
    ok = [r for r in rs if r['auc_plant'] >= 0.7 and abs(r['auc_plant_null'] - 0.5) < 0.05
          and r['auc_sm'] == r['auc_sm']]
    ok.sort(key=lambda r: -(r['auc_sm'] + r['auc_plant']))
    return ok[:nmax], len(rs)


def ur3_pooled(surv, rng, G=40, reps=300):
    """Group-level calibration: do survivors, refit on Ur III half A, give tablet-disjoint groups of
    G size-matched capacity quantities from PLANNED a higher mean pi than groups of measured GRAIN (half B)?
    Control: kinds shuffled before grouping."""
    from pe44_cycle1 import build
    UX, UK, UA, US = build()
    ia = np.where(UA)[0]
    sub = rng.choice(ia, 15000, replace=False)
    ms = fit_set(UX[sub], surv, 500)
    ib = np.where(~UA & (US == 'UR_CAP') & ((UK == 'PLANNED') | (UK == 'MEASURED')))[0]
    pi = pi_of(ms, UX[ib])
    kb = UK[ib]
    lv = np.floor(UX[ib][:, FEATS.index('LMAG')] / math.log(2))
    pos, neg = [], []
    for b in np.unique(lv):
        P = np.where((lv == b) & (kb == 'PLANNED'))[0]; N = np.where((lv == b) & (kb == 'MEASURED'))[0]
        n = min(len(P), len(N))
        if n:
            pos += list(rng.choice(P, n, replace=False)); neg += list(rng.choice(N, n, replace=False))
    pos, neg = np.array(pos), np.array(neg)
    def groups(a, b):
        ga = [pi[rng.choice(a, G)].mean() for _ in range(reps)]
        gb = [pi[rng.choice(b, G)].mean() for _ in range(reps)]
        return auc(ga, gb)
    real = groups(pos, neg)
    both = np.concatenate([pos, neg]); rng.shuffle(both)
    shuf = groups(both[:len(pos)], both[len(pos):])
    return {'n_pairs': int(len(pos)), 'item_auc': auc(pi[pos], pi[neg]), 'group_auc': real, 'group_auc_shuffled': shuf}


def fit_set(X, specs, seed):
    ms = []
    for k, r in enumerate(specs):
        rng = np.random.default_rng(seed + k)
        cols = [FEATS.index(c) for c in r['cols']]
        m = fit_mix(X, cols, r['K'], rng, iters=60)
        ms.append((m, int(np.argmax(plan_score(m)))))
    return ms


def pi_of(ms, X):
    return np.mean([posterior(m, X)[0][:, pc] for m, pc in ms], 0)


def shuffle_within_system(tabs, rng):
    pool = defaultdict(list)
    for t, recs in tabs.items():
        for r in recs:
            pool[r['sys']].append(r['val'])
    for k in pool:
        rng.shuffle(pool[k])
    it = {k: iter(v) for k, v in pool.items()}
    out = {}
    for t, recs in tabs.items():
        out[t] = [dict(r, val=next(it[r['sys']])) for r in recs]
    return out


def shuffle_within_size(tabs, rng):
    """N3: values shuffled across entries within (system, floor(log2 value)): keeps every entry's
    magnitude, destroys its exact digits, its tablet neighbours and its sign."""
    pool = defaultdict(list)
    for t, recs in tabs.items():
        for r in recs:
            pool[(r['sys'], int(math.log2(r['val'])))].append(r['val'])
    for k in pool:
        rng.shuffle(pool[k])
    it = {k: iter(v) for k, v in pool.items()}
    return {t: [dict(r, val=next(it[(r['sys'], int(math.log2(r['val'])))])) for r in recs] for t, recs in tabs.items()}


def poisson_same_total(tabs, rng):
    out = {}
    for t, recs in tabs.items():
        new = [dict(r) for r in recs]
        by = defaultdict(list)
        for i, r in enumerate(recs):
            by[r['sys']].append(i)
        for s, idx in by.items():
            T = sum(recs[i]['val'] for i in idx)
            n = len(idx)
            # every entry >= 1, remainder spread multinomially
            extra = rng.multinomial(T - n, [1 / n] * n) if T > n else np.zeros(n, int)
            for i, e in zip(idx, extra):
                new[i]['val'] = int(1 + e)
        out[t] = new
    return out


def keys_of(r, i, recs):
    ks = ['final:' + str(r['item']), 'hdr:' + str(r['hdr']), 'surf:' + r['surface'], 'sys:' + r['sys']]
    if r['signs']:
        ks.append('first:' + r['signs'][0])
    for s in set(r['signs']):
        ks.append('any:' + s)
    if not r['signs']:
        ks.append('nosign')
    ks.append('pos:' + ('first' if i == 0 else 'last' if i == len(recs) - 1 else 'mid'))
    return ks


def key_index(tabs, meta, tabset=None):
    idx = defaultdict(list)
    for j, (t, i) in enumerate(meta):
        if tabset is not None and t not in tabset:
            continue
        r = tabs[t][i]
        for k in keys_of(r, i, tabs[t]):
            idx[k].append(j)
    return idx


def scan(tabs, ms, rng, tabset=None, reps=200, plant=None):
    X, meta = corpus_matrix(tabs)
    pi = pi_of(ms, X)
    idx = key_index(tabs, meta, tabset)
    if plant is not None:
        idx['PLANTED'] = [j for j, mm in enumerate(meta) if mm in plant]
    idx = {k: v for k, v in idx.items() if len(v) >= 15}
    obs = {k: pi[v].mean() for k, v in idx.items()}
    nl = {'1': defaultdict(list), '2': defaultdict(list), '3': defaultdict(list)}
    for rep in range(reps):
        for tag, nullf in (('1', shuffle_within_system), ('3', shuffle_within_size), ('2', poisson_same_total)):
            if tag == '2' and rep >= reps // 4:
                continue
            tb = nullf(tabs, rng) if tag == '2' else nullf(tabs, random.Random(int(rng.integers(1e9))))
            Xn, _ = corpus_matrix(tb)
            pn = pi_of(ms, Xn)
            for k, v in idx.items():
                nl[tag][k].append(pn[v].mean())
    res = {}
    for k in idx:
        res[k] = {'n': len(idx[k]), 'pi': float(obs[k])}
        for tag in '123':
            a = np.array(nl[tag][k])
            p = (1 + np.sum(np.abs(a - a.mean()) >= abs(obs[k] - a.mean()))) / (1 + len(a))
            res[k].update({'n' + tag: float(a.mean()), 'z' + tag: float((obs[k] - a.mean()) / (a.std() + 1e-9)), 'p' + tag: float(p)})
    ks = sorted(res)
    for tag in '123':
        q = P23.bh([res[k]['p' + tag] for k in ks])
        for k, a in zip(ks, q):
            res[k]['q' + tag] = float(a)
    return res, pi, meta


if __name__ == '__main__':
    part = sys.argv[1] if len(sys.argv) > 1 else 'all'
    surv, ntot = survivors()
    print('survivors', len(surv), 'of', ntot, flush=True)
    tabs = pe_tabs()
    X, meta = corpus_matrix(tabs)
    rng = np.random.default_rng(11)
    out = {'n_surv': len(surv), 'n_models': ntot, 'surv': surv}
    if part in ('all', 'u'):
        out['ur3_pooled'] = ur3_pooled(surv, rng)
        print('UR3 pooled', out['ur3_pooled'], flush=True)
    if part in ('all', 'a'):
        ids = sorted(tabs); rr = random.Random(5)
        st = []
        for rep in range(5):
            A = {t for t in ids if rr.random() < 0.5}
            for name, tb in (('real', tabs), ('shuf', shuffle_within_system(tabs, rr))):
                Xr, mt = corpus_matrix(tb)
                inA = np.array([t in A for t, i in mt])
                mA = fit_set(Xr[inA], surv, 100 + rep); mB = fit_set(Xr[~inA], surv, 200 + rep)
                rho = spearmanr(pi_of(mA, Xr), pi_of(mB, Xr)).correlation
                st.append({'rep': rep, 'corpus': name, 'rho': float(rho)})
                print(st[-1], flush=True)
        out['stability'] = st
    if part in ('all', 'b'):
        ms = fit_set(X, surv, 1)
        # planted sign: 40 random entries rounded
        rr = random.Random(9)
        cand = [(t, i) for t, i in meta if tabs[t][i]['val'] >= DEN[tabs[t][i]['sys']][1]]
        pick = set(rr.sample(cand, 40))
        tp = {t: [dict(r) for r in recs] for t, recs in tabs.items()}
        for t, i in pick:
            r = tp[t][i]; den = DEN[r['sys']]
            hi = max(k for k, d in enumerate(den) if d <= r['val'])
            r['val'] = max(den[hi], int(round(r['val'] / den[hi])) * den[hi])
        resP, _, _ = scan(tp, ms, rng, reps=100, plant=pick)
        out['planted'] = resP['PLANTED']
        print('PLANTED', resP['PLANTED'], flush=True)
        res, pi, _ = scan(tabs, ms, rng, reps=200)
        out['scan'] = res
        np.save(os.path.join(CK, 'c2_pi.npy'), pi)
        json.dump(meta, open(os.path.join(CK, 'c2_meta.json'), 'w'))
        flags = {k: v for k, v in res.items() if v['q3'] < 0.1}
        print('flags', len(flags), flush=True)
        for k, v in sorted(res.items(), key=lambda kv: (kv[1]['p3'], -abs(kv[1]['z3'])))[:30]:
            print(k, {a: round(b, 4) for a, b in v.items()}, flush=True)
        # (c) replication
        ids = sorted(tabs); rr = random.Random(13)
        repl = []
        for rep in range(4):
            A = {t for t in ids if rr.random() < 0.5}
            B = set(ids) - A
            rA, _, _ = scan(tabs, ms, rng, tabset=A, reps=60)
            rB, _, _ = scan(tabs, ms, rng, tabset=B, reps=60)
            fl = [k for k, v in rA.items() if v['q3'] < 0.1 and k.split(':')[0] not in ('sys',)]
            ok = [k for k in fl if k in rB and rB[k]['p3'] < 0.05 and np.sign(rB[k]['z3']) == np.sign(rA[k]['z3'])]
            repl.append({'rep': rep, 'flagsA': fl, 'replicated': ok})
            print(repl[-1], flush=True)
        out['replication'] = repl
    json.dump(out, open(os.path.join(CK, 'c2_%s.json' % part), 'w'), indent=1)
