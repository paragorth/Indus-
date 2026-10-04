#!/usr/bin/env python3
"""LA-20 cycle 1 positive control: Linear B Pylos. Can the pecking-order rankers recover the
canonical order of the 16 district towns (9 Hither + 7 Further) from the lists alone?
Place spellings are mapped to slot ids by stems (control side only). Also: general held-out
pair accuracy on PY words/logograms/first signs, full and subsampled to Linear A size."""
import os, sys, json, time
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"): os.environ.setdefault(_v, "1")
from scipy.stats import kendalltau
from la20_common import *

STEMS = [('PI-*82', 1), ('ME-TA-PA', 2), ('PE-TO-NO', 3), ('PA-KI-JA', 4), ('A-PU2', 5), ('A-KE-RE', 6),
         ('RO-U-SO', 7), ('E-RA-TO', 7), ('E-RA-TE-I', 7), ('KA-RA-DO-RO', 8), ('RI-JO', 9),
         ('TI-MI-TO', 10), ('RA-U-RA', 11), ('RA-WA-RA', 11), ('SA-MA-RA', 12), ('A-SI-JA', 13),
         ('E-RA-TE-RE', 14), ('ZA-MA-E', 15), ('E-SA-RE', 16)]


def place(w):
    for s, k in sorted(STEMS, key=lambda x: -len(x[0])):
        if w == s or w.startswith(s + '-'): return 'P%02d' % k
    return w


def py_orders(docs, T='W'):
    out = []
    for d in docs:
        o = []
        for x, _ in d['items'][T]:
            x = place(x)
            if x not in o: o.append(x)
        if len(o) >= 2: out.append(o)
    return out


def tau(score):
    ps = sorted(k for k in score if k.startswith('P') and k[1:].isdigit())
    if len(ps) < 4: return float('nan'), len(ps)
    return kendalltau([int(p[1:]) for p in ps], [-score[p] for p in ps])[0], len(ps)


if __name__ == '__main__':
    NREP = int(sys.argv[1])
    lb = load_lb(('PY',))
    W = py_orders(lb)
    nplace = [sum(1 for x in o if x.startswith('P') and x[1:].isdigit()) for o in W]
    print('PY lists', len(W), 'lists with >=2 places', sum(n >= 2 for n in nplace), 'with >=6', sum(n >= 6 for n in nplace))
    res = {}
    W_full = W
    W = [[x for x in o if x.startswith('P') and x[1:].isdigit()] for o in W]
    W = [o for o in W if len(o) >= 2]
    print('place-only sublists', len(W)); nplace = [len(o) for o in W]
    for name, sub in [('all', W), ('no_big', [o for o, n in zip(W, nplace) if n < 6]),
                      ('only_small', [o for o, n in zip(W, nplace) if n < 4])]:
        for m in ['BT', 'ELO', 'POS', 'PL']:
            t, k = tau(FITTERS[m](sub))
            nt = []
            for r in range(NREP):
                nt.append(tau(FITTERS[m](shuffle_within(sub, random.Random(r))))[0])
            nt = np.array(nt)
            res[name + '_' + m] = (t, k, float(np.nanmean(nt)), float(np.nanstd(nt)), float(np.mean(nt >= t)))
            print(name, m, 'tau %.3f over %d places; null %.3f+-%.3f P %.4f' % res[name + '_' + m], flush=True)
        sc = fit_bt(sub)
        print(' BT order:', [p for p in sorted([p for p in sc if p.startswith('P') and p[1:].isdigit()], key=lambda p: -sc[p])])
    # general held-out pair accuracy, full and LA-sized subsamples
    for T in 'WLF':
        o = py_orders(lb, T) if T != 'W' else W_full
        real = cv_score(o, fit_bt); nl = [acc(cv_score(shuffle_within(o, random.Random(r)), fit_bt, seed=r))[0] for r in range(20)]
        res['gen_' + T] = (len(o), acc(real), real[[0, 2]].tolist(), float(np.mean(nl)), float(np.std(nl)))
        print('PY', T, res['gen_' + T], flush=True)
        # LA-sized: as many lists as LA has for this type
        n_la = {'W': 232, 'L': 82, 'F': 232}[T]
        sub_acc = []
        for r in range(10):
            rng = random.Random(500 + r); s = rng.sample(o, min(n_la, len(o)))
            a = acc(cv_score(s, fit_bt, seed=r)); b = acc(cv_score(shuffle_within(s, rng), fit_bt, seed=r))
            sub_acc.append((a[0], b[0]))
        sub_acc = np.array(sub_acc)
        res['la_sized_' + T] = sub_acc.mean(0).tolist() + [float(np.mean(sub_acc[:, 0] > sub_acc[:, 1]))]
        print('PY LA-sized', T, res['la_sized_' + T], flush=True)
    json.dump(res, open(os.path.join(CK, 'c1_pylos.json'), 'w'), default=str)
