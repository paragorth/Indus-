#!/usr/bin/env python3
"""LA-70 cycle 2: decode held-out documents with the frozen V2 (primary split) and with V2 re-gated on
each of 10 more splits' training halves; compare with V2 roles shuffled (100), random readings (50),
the EMPTY reading, la60 v1 PRIOR and INDUCED (la60 decoder, and PRIOR also through the v2 decoder).
Calibration: Linear B KN+PY at Linear A size (la60 draws), a true partial LB reading matched to V2 in
kind and size (LBMATCH), the same with a third of its items replaced by random words (LBCORRUPT, as V2
has a third C-grade items), vs their shuffles and EMPTY. At most 2 worker processes."""
import json, os, sys, random, statistics as st
from multiprocessing import Pool
from la70_common import *
from la60_common import shuffle_reading

NSH = int(os.environ.get('NSH', 100))


def summarize(real, L):
    out = {}
    for k in ('full', 'strict', 'close', 'agree', 'viol'):
        v = [x.get(k, 0) for x in L]
        out[k] = (st.mean(v), (sum(1 for x in v if x >= real.get(k, 0)) + 1) / (len(v) + 1))
    return out


def run_la(s):
    A = admin_docs(load_la())
    name = 'la60-main' if s < 0 else 'la60-c2-split%d' % s
    tr, te = split(A, name)
    rng = random.Random(seed('la70-c2-%d' % s))
    if s < 0:
        fz = json.load(open(os.path.join(CK, 'frozen_V2.json')))
        V2 = dict(name='V2', roles=fz['roles'], order=fz['order'], lib=set(fz['lib']), site_default=fz['site_default'],
                  ent_slot_unknown=True, rules=fz['rules'])
        assert hash_v2(V2) == fz['sha256'], 'frozen hash mismatch'
    else:
        V2 = gate_v2(tr, 'V2')
    V2G = gate_v2(tr, 'V2G', strict=True)
    P1 = prior_reading(); P1['rules'] = {}
    I1 = induce_reading(tr)
    r = dict(split=s, ntest=len(te))
    r['V2'] = score_v2(tr, te, V2)
    r['V2G'] = score_v2(tr, te, V2G)
    r['EMPTY'] = score_v2(tr, te, EMPTY)
    r['V1PRIOR_v2dec'] = score_v2(tr, te, P1)
    r['V1PRIOR'] = score_v1(tr, te, prior_reading())
    r['V1INDUCED'] = score_v1(tr, te, I1)
    sh = [score_v2(tr, te, shuffle_v2(V2, rng)) for _ in range(NSH)]
    rd = [score_v2(tr, te, random_v2(tr, V2, rng)) for _ in range(NSH // 2)]
    shp = [score_v2(tr, te, dict(shuffle_reading(P1, rng), rules={})) for _ in range(NSH // 2)]
    r['V2_shuf'] = summarize(r['V2'], sh); r['V2_rand'] = summarize(r['V2'], rd)
    r['V1PRIOR_shuf'] = summarize(r['V1PRIOR_v2dec'], shp)
    # per-document FULL sets: V2 vs V1 (same v2 decoder), for the gain decomposition
    acc2 = acceptor_v2(tr, V2); acc1 = acceptor_v2(tr, P1)
    r['full_ids_V2'] = sorted(d['id'] for d in te if decode_v2(d, V2, acc2)[0]['full'])
    r['full_ids_V1'] = sorted(d['id'] for d in te if decode_v2(d, P1, acc1)[0]['full'])
    return r


def corrupt(R, tr, frac, rng):
    vocab = sorted({t[1] for d in tr for t in d['toks'] if t[0] == 'W'} - set(R['roles']))
    items = sorted(R['roles'].items()); k = int(round(len(items) * frac))
    idx = set(rng.sample(range(len(items)), k))
    roles = {}
    for i, (w, r) in enumerate(items):
        roles[rng.choice(vocab) if i in idx else w] = r
    return dict(R, name='LBCORRUPT', roles=roles)


def run_lb(s):
    A = admin_docs(load_la())
    LB = [d for d in load_lb() if any(t[0] == 'N' for t in d['toks'])]
    dr = lb_draw(LB, A, s)
    tr, te = split(dr, 'la60-c2-lbsplit%d' % s)
    rng = random.Random(seed('la70-c2-lb%d' % s))
    fz = json.load(open(os.path.join(CK, 'frozen_V2.json')))
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
            res = p.map(run_la, list(range(-1, 10)))
        else:
            res = p.map(run_lb, list(range(10)))
    json.dump(res, open(os.path.join(CK, 'c2_%s.json' % which), 'w'), indent=0)
    for r in res:
        print(json.dumps({k: v for k, v in r.items() if not k.startswith('full_ids')}))


if __name__ == '__main__':
    main()
