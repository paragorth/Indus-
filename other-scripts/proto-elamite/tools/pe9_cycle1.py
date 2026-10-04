"""pe9 cycle 1: held-out likelihood race between language-like and bag models.

Corpora: PE (fit set = 80% of tablets; 20% reserved, seed 9, for the cycle-3
prediction test), PE_SHUF (fit-set middles shuffled across tablets, counts per
tablet kept), UR3 (Ur III admin names per tablet), LINB (Linear B personnel names
per tablet), and PLANTED corpora written on the PE fit-set structure from
LM-bigram, POLYA and BAG_WOR (K 12, c 2, eps 0.15) generators.
Method: 5-fold CV over contiguous tablet blocks; each model's parameters chosen by
grid search on the training folds (base unigram/bigram, bigram smoothing,
theta / K / c / eps / session length S), then held-out bits per sign.
"""
import json, os, sys, time, itertools
import numpy as np
from multiprocessing import Pool
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from pe9_common import *  # noqa

KS = [2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 150, 300]
CS = [1, 2, 3, 5, 10]
EPS = [0.0, 0.05, 0.15, 0.3, 0.5, 0.7, 0.9]
THS = [0.1, 0.3, 1, 3, 10, 30, 100, 300]
SS = [1, 2, 4]
AS = [0.5, 2.0, 8.0]


def grid(mode):
    if mode == 0:
        return [dict(K=0, c=0, eps=0, th=1, S=1)]
    if mode == 1:
        return [dict(K=0, c=0, eps=0, th=t, S=S) for t in THS for S in SS]
    if mode in (2, 4):
        return [dict(K=k, c=0, eps=e, th=1, S=S) for k in KS for e in EPS for S in SS]
    return [dict(K=k, c=c, eps=e, th=1, S=S) for k in KS for c in CS for e in EPS for S in SS]


def reserve_split(T):
    rng = np.random.RandomState(9)
    idx = rng.permutation(T)
    hold = set(idx[:int(round(0.2 * T))].tolist())
    return sorted(set(range(T)) - hold), sorted(hold)


def pe_fit_tablets(C):
    tabs = C['PE']['tablets']
    fit, hold = reserve_split(len(tabs))
    return [tabs[i] for i in fit], [tabs[i] for i in hold], fit, hold


def shuffle_across(tabs, seed):
    rng = np.random.RandomState(seed)
    allm = [m for t in tabs for m in t]
    rng.shuffle(allm)
    out, k = [], 0
    for t in tabs:
        out.append(allm[k:k + len(t)])
        k += len(t)
    return out


def planted(tabs, kind, seed):
    E = Enc(tabs)
    allt = np.arange(E.T, dtype=np.int64)
    w, B = fit_base(E.tok, E.ss, E.ts, allt, E.V)
    sess = sess_flags(E.T, 1)
    if kind == 'LM':
        tk = simulate(E.ss, E.ts, sess, E.V, w, B, 1, 0, 0, 0, 0, 1, seed)
    elif kind == 'POLYA':
        tk = simulate(E.ss, E.ts, sess, E.V, w, B, 1, 1, 0, 0, 0, 3.0, seed)
    else:
        tk = simulate(E.ss, E.ts, sess, E.V, w, B, 1, 3, 12, 2, 0.15, 1, seed)
    out = []
    for i in range(E.T):
        out.append([[E.vocab[x] for x in tk[E.ss[j]:E.ss[j + 1]]] for j in range(E.ts[i], E.ts[i + 1])])
    return out


def cv(tabs, nfold=5):
    E = Enc(tabs)
    T = E.T
    folds = np.array_split(np.arange(T), nfold)
    res = {}
    for mode in range(5):
        tot_ll, best_pars = 0.0, []
        for f in folds:
            tr = np.array(sorted(set(range(T)) - set(f.tolist())), np.int64)
            te = np.array(f, np.int64)
            best = (-1e300, None)
            for a in AS:
                w, B = fit_base(E.tok, E.ss, E.ts, tr, E.V, a)
                for base in (0, 1):
                    if base == 0 and a != AS[0]:
                        continue
                    for p in grid(mode):
                        sess = sess_flags(T, p['S'])
                        ll = score(E.tok, E.ss, E.ts, tr, sess, E.V, w, B, base, mode,
                                   float(p['K']), float(p['c']), p['eps'], p['th']).sum()
                        if ll > best[0]:
                            best = (ll, dict(p, base=base, a=a))
            p = best[1]
            w, B = fit_base(E.tok, E.ss, E.ts, tr, E.V, p['a'])
            sess = sess_flags(T, p['S'])
            tot_ll += score(E.tok, E.ss, E.ts, te, sess, E.V, w, B, p['base'], mode,
                            float(p['K']), float(p['c']), p['eps'], p['th']).sum()
            best_pars.append(p)
        res[MODES[mode]] = {'bits_per_sign': -tot_ll / LN2 / len(E.tok), 'pars': best_pars}
    return res


def job(arg):
    name, tabs = arg
    ck = os.path.join(CKPT, 'c1_' + name + '.json')
    if os.path.exists(ck):
        return name, json.load(open(ck))
    t0 = time.time()
    r = cv(tabs)
    r['_sec'] = time.time() - t0
    json.dump(r, open(ck, 'w'))
    print(name, {k: round(v['bits_per_sign'], 3) for k, v in r.items() if k[0] != '_'}, flush=True)
    return name, r


if __name__ == '__main__':
    C = load_corpora()
    fit, hold, _, _ = pe_fit_tablets(C)
    jobs = [('PE', fit)]
    for s in range(3):
        jobs.append(('PE_SHUF%d' % s, shuffle_across(fit, s)))
    jobs += [('UR3', C['UR3']['tablets']), ('LINB', C['LINB']['tablets'])]
    for kind in ('LM', 'POLYA', 'BAG'):
        for s in range(2):
            jobs.append(('PLANT_%s%d' % (kind, s), planted(fit, kind, 100 + s)))
    with Pool(2) as pool:
        out = dict(pool.map(job, jobs, chunksize=1))
    json.dump(out, open(os.path.join(DATA, 'pe9_cycle1.json'), 'w'), indent=1)
    for k, r in out.items():
        bp = {m: r[m]['bits_per_sign'] for m in MODES}
        win = min(bp, key=bp.get)
        print('%-14s' % k, ' '.join('%s %.3f' % (m, bp[m]) for m in MODES), '| win', win,
              '| pars', r[win]['pars'][0])
