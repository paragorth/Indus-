#!/usr/bin/env python3
"""la64 cycle 1: calibration of the physics ruler on Linear B (units and fraction values hidden),
Ur III (sila hidden) and planted Linear A-shaped archives, at full size and at Linear A size."""
import sys, json, math
import numpy as np
sys.path.insert(0, __import__('os').path.dirname(__file__))
from la64_lib import *

PART = sys.argv[1] if len(sys.argv) > 1 else 'all'
OUT = os.path.join(CK, 'c1_%s.json' % PART)
res = json.load(open(OUT)) if os.path.exists(OUT) else {}


def save():
    json.dump(res, open(OUT, 'w'), indent=1)


def interval(A, i, theta, ruler=None, drop=2.0):
    g, s = key_profile(A, i, theta, ruler)
    ok = g[s >= s.max() - drop]
    return float(np.exp(g[np.argmax(s)])), float(np.exp(ok.min())), float(np.exp(ok.max())), float(s.max())


def plabels(A, f):
    _, parts = score(A, f['logu'][None], f['theta'][None], return_parts=True)
    out = {}
    for pk in A['pkeys']:
        GF = parts['PF:' + pk]
        b = max(GF, key=lambda x: GF[x][0])
        out[pk] = b if GF[b][0] > 0 else 'none'
    return out


def run_ctrl(name, docs, clsf, truef, rng, H=5000):
    A = build(docs, clsf)
    f = fit_best(A, rng, H)
    rows = {}
    for i, c in enumerate(A['keys']):
        best, lo, hi, smax = interval(A, i, f['theta'])
        rows[c] = dict(u=best, lo=lo, hi=hi, gain=smax, n=int((A['k'] == i).sum()), true=truef(c),
                       ratio=best / truef(c), inside=bool(lo <= truef(c) <= hi))
    return dict(keys=rows, plab=plabels(A, f), theta=dict(zip(A['frkeys'], f['theta'].round(4).tolist())),
                S=f['S'], nE=len(A['n']), nP=len(A['pe']))


if __name__ == '__main__':
    part = sys.argv[1] if len(sys.argv) > 1 else 'all'
    LB = load_lb()
    UR = load_ur3()
    LA = load_la()
    nLA = sum(not e['tot'] for d in LA for e in d['E'])
    if part in ('lbfull', 'all') and 'lbfull' not in res:
        res['lbfull'] = run_ctrl('LB', LB, lb_class, lb_true_u, np.random.default_rng(1))
        save()
        print('lbfull', json.dumps(res['lbfull'])[:2000], flush=True)
    if part in ('lbthin', 'all'):
        for s in range(8):
            k = 'lbthin%d' % s
            if k in res:
                continue
            rng = np.random.default_rng(100 + s)
            res[k] = run_ctrl('LB', thin(LB, rng, nLA), lb_class, lb_true_u, rng, 3000)
            save()
            print(k, {c: round(v['ratio'], 2) for c, v in res[k]['keys'].items()}, res[k]['plab'], flush=True)
    if part in ('urthin', 'all'):
        for s in range(8):
            k = 'urthin%d' % s
            if k in res:
                continue
            rng = np.random.default_rng(200 + s)
            res[k] = run_ctrl('UR', thin(UR, rng, nLA), ur_class, lambda c: 1.0, rng, 3000)
            save()
            print(k, {c: round(v['ratio'], 2) for c, v in res[k]['keys'].items()}, res[k]['plab'], flush=True)
    if part in ('plant', 'all'):
        A0 = build(LA, la_class)
        for s in range(10):
            k = 'plant%d' % s
            if k in res:
                continue
            rng = np.random.default_rng(300 + s)
            docs = json.loads(json.dumps(LA))
            tu = {c: float(np.exp(rng.uniform(math.log(0.05), math.log(200)))) for c in A0['keys']}
            tth = {f: float(np.exp(rng.uniform(math.log(1 / 64), math.log(0.75)))) for f in A0['frkeys']}
            wrole = {}
            for d in docs:
                for e in d['E']:
                    if e['tot'] or e['c'] not in tu or rng.random() < 0.3:
                        continue
                    key = (e['w'], e['c'])
                    if key not in wrole:
                        wrole[key] = int(rng.integers(0, 6))
                    mu, sd = windows(la_class(e['c']))
                    r = wrole[key]
                    q = math.exp(rng.normal(mu[r], sd[r])) / tu[e['c']]
                    n = int(math.floor(q))
                    rem = q - n
                    opts = [([], 0.0)] + [([f], v) for f, v in tth.items()]
                    fr, v = min(opts, key=lambda o: abs(o[1] - rem))
                    if n == 0 and not fr:
                        fr = [min(tth, key=tth.get)]
                    e['n'], e['fr'] = n, fr
            # person counts: planted per-head rations for paired entries are not planted (counts kept)
            r = run_ctrl('PL', docs, la_class, lambda c: tu[c], rng, 3000)
            r['true_theta'] = tth
            res[k] = r
            save()
            print(k, {c: round(v['ratio'], 2) for c, v in r['keys'].items()}, flush=True)
