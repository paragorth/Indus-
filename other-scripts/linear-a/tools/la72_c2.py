#!/usr/bin/env python3
"""LA-72 cycle 2: held-out tests of the clean reading V2c on rd (primary) and read (check).

Per split (la60-main with the frozen V2c / la70 V2 files; la60-c2-split0..9 re-gated on their training
halves with the clean procedure and with the la70 procedure):
  V2c, 100 role shuffles, 50 random readings, EMPTY, la70 V2 (the old reading), la60 v1 PRIOR;
  ablations of the three small gains: -staple order (la60 order), -*308 rule, totals only (KU-RO alone),
  and -KI red, -D/B values.
Version 'rnd' (la71 control): as many tokens removed at random as 'read' removes.
Linear B calibration as la70 (KN+PY at Linear A size, a true matched partial reading and a one-third
corrupted copy, vs their shuffles), matched to V2c's size.  At most 2 workers."""
import json, os, sys, random, statistics as st
from multiprocessing import Pool
import la70_common as L7
from la70_common import (gate_v2, hash_v2, score_v2, shuffle_v2, random_v2, EMPTY, SITE_DEF, prior_reading,
                         lb_matched_reading, score_lb, load_lb, lb_draw)
from la60_common import shuffle_reading
from la72_common import *
from la72_c1 import build, set_ki

NSH = int(os.environ.get('NSH', 100))
LA70_FZ = os.path.join(D, 'la70_ckpt', 'frozen_V2.json')


def fz_reading(fn):
    fz = json.load(open(fn))
    R = dict(name=fz['name'], roles=fz['roles'], order=fz['order'], lib=set(fz['lib']), site_default=fz['site_default'],
             ent_slot_unknown=True, rules=fz['rules'])
    if fz.get('v1close'):
        R['v1close'] = True
    assert hash_v2(R) == fz['sha256'], fn
    return R


def summarize(real, L):
    out = {}
    for k in ('full', 'strict', 'close', 'agree', 'viol'):
        v = [x.get(k, 0) for x in L]
        out[k] = (st.mean(v), (sum(1 for x in v if x >= real.get(k, 0)) + 1) / (len(v) + 1))
    return out


def run_la(job):
    ver, s = job
    A = admin(ver)
    tr, te = split(A, 'la60-main' if s < 0 else 'la60-c2-split%d' % s)
    rng = random.Random(seed('la72-c2-%s-%d' % (ver, s)))
    if s < 0 and ver == 'rd':
        V = fz_reading(os.path.join(CK, 'frozen_V2c.json'))
    else:
        V, _ = build(tr, A, 'clean', random.Random(seed('la72-c2-gate-%s-%d' % (ver, s))))
    if s < 0:
        O = fz_reading(LA70_FZ)
    else:
        set_ki(True); O = gate_v2(tr, 'V2-la70')
    r = dict(ver=ver, split=s, ntest=len(te), hash=hash_v2(V))
    r['V2c'] = score_v2(tr, te, V)
    r['EMPTY'] = score_v2(tr, te, EMPTY)
    r['V2la70'] = score_v2(tr, te, O)
    P1 = prior_reading(); P1['rules'] = {}
    r['V1PRIOR'] = score_v2(tr, te, P1)
    sh = [score_v2(tr, te, shuffle_v2(V, rng)) for _ in range(NSH)]
    rd_ = [score_v2(tr, te, random_v2(tr, V, rng)) for _ in range(NSH // 2)]
    r['shuf_raw'] = [(x.get('strict', 0), x.get('close', 0), x.get('full', 0)) for x in sh]
    r['V2c_shuf'] = summarize(r['V2c'], sh); r['V2c_rand'] = summarize(r['V2c'], rd_)
    shO = [score_v2(tr, te, shuffle_v2(O, rng)) for _ in range(NSH // 2)]
    r['V2la70_shuf'] = summarize(r['V2la70'], shO)
    # ablations
    P = prior_reading()
    abl = {'-staple': dict(V, order=P['order']),
           '-frac308': dict(V, rules=dict(V['rules'], frac308=False)),
           'TOTonly': dict(EMPTY, name='TOT', roles={'KU-RO': 'TOT', 'PO-TO-KU-RO': 'TOT'}),
           '-DB': dict(V, v1close=True), '-order': dict(V, order={})}
    v = dict(V, roles={w: q for w, q in V['roles'].items() if q != 'RED'}, rules=dict(V['rules'], ki_red=False))
    if 'KI' in V['roles']:
        v['roles']['KI'] = 'COM'
    abl['-KIred'] = v
    for k, R in abl.items():
        r[k] = score_v2(tr, te, R)
    return r


def corrupt(R, tr, frac, rng):
    vocab = sorted({t[1] for d in tr for t in d['toks'] if t[0] == 'W'} - set(R['roles']))
    items = sorted(R['roles'].items()); k = int(round(len(items) * frac))
    idx = set(rng.sample(range(len(items)), k))
    roles = {}
    for i, (w, q) in enumerate(items):
        roles[rng.choice(vocab) if i in idx else w] = q
    return dict(R, name='LBCORRUPT', roles=roles)


def run_lb(s):
    A = admin('rd')
    LB = [d for d in load_lb() if any(t[0] == 'N' for t in d['toks'])]
    dr = lb_draw(LB, A, s)
    tr, te = split(dr, 'la60-c2-lbsplit%d' % s)
    rng = random.Random(seed('la72-c2-lb%d' % s))
    fz = json.load(open(os.path.join(CK, 'frozen_V2c.json')))
    M = lb_matched_reading(tr, dict(roles=fz['roles']))
    C = corrupt(M, tr, 1 / 3, rng)
    E = dict(name='EMPTY', roles={}, order={}, lib=set(), site_default={}, ent_slot_unknown=True)
    r = dict(split=s, ntest=len(te), nitems=len(M['roles']), EMPTY=score_lb(tr, te, E))
    for R in (M, C):
        real = score_lb(tr, te, R)
        sh = [score_lb(tr, te, shuffle_reading(R, rng)) for _ in range(NSH)]
        r[R['name']] = real; r[R['name'] + '_shuf'] = summarize(real, sh)
    return r


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else 'LA'
    with Pool(2) as p:
        if which == 'LA':
            res = p.map(run_la, [(v, s) for v in ('rd', 'read', 'rnd') for s in range(-1, 10)])
        else:
            res = p.map(run_lb, list(range(10)))
    json.dump(res, open(os.path.join(CK, 'c2_%s.json' % which), 'w'), indent=0)
    for r in res:
        print(json.dumps({k: v for k, v in r.items() if k != 'shuf_raw'}))


if __name__ == '__main__':
    main()
