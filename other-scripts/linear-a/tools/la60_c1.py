#!/usr/bin/env python3
"""LA-60 cycle 1: the reading as a generative grammar of whole documents, scored on held-out
documents against simpler models (B0 unigram, B1 trigram, G0 = same grammar with an EMPTY reading)
and against the same grammar with the reading's roles shuffled among its word types.
Linear A: INDUCED reading (train-only rules) and PRIOR reading (hand-assembled, contaminated).
Linear B (KN+PY, sign identities only) at Linear A size: INDUCED reading by the same rules."""
import json, os, sys, random, statistics as st
from la60_common import *
from la60_model import *

NSPLIT = int(os.environ.get('NSPLIT', 10)); NSH = int(os.environ.get('NSH', 20))
which = sys.argv[1] if len(sys.argv) > 1 else 'LA'


def bits(m, te):
    return sum(doc_bits(m, d) for d in te)


def run_split(tr, te, readings, lb, rng):
    res = {}
    res['B0'] = bits(Baseline(tr, 1), te)
    b1 = Baseline(tr, 3); res['B1'] = bits(b1, te)
    g0 = Grammar(tr, dict(name='EMPTY', roles={}, order={}, lib=set(), site_default={}), lb=lb)
    res['G0'] = bits(g0, te)
    for R in readings:
        n = R['name']
        g = Grammar(tr, R, lb=lb); res[n] = bits(g, te)
        res[n + '_theta'] = g.theta; res[n + '_beta'] = g.beta; res[n + '_arith'] = g.arith_hits
        res[n + '-noarith'] = bits(Grammar(tr, R, lb=lb, arith=False), te)
        res[n + '-noorder'] = bits(Grammar(tr, R, lb=lb, order=False), te)
        res[n + '-mixB1'] = sum(mix_bits(b1, g, d) for d in te)
        sh = []; shk = []
        for k in range(NSH):
            S = shuffle_reading(R, rng)
            sh.append(bits(Grammar(tr, S, lb=lb), te))
            S2 = shuffle_reading(R, rng, keep_tot=True)
            shk.append(bits(Grammar(tr, S2, lb=lb), te))
        res[n + '-shuf'] = sh; res[n + '-shufkeepTOT'] = shk
        res[n + '_size'] = len(R['roles'])
    return res


def main():
    out = []
    rng = random.Random(seed('la60-c1-' + which))
    if which == 'LA':
        A = admin_docs(load_la())
        for s in range(NSPLIT):
            tr, te = split(A, 'la60-c1-split%d' % s)
            RI = induce_reading(tr); RP = prior_reading()
            r = run_split(tr, te, [RI, RP], False, rng)
            r['split'] = s; r['ntest'] = len(te); r['ntok'] = sum(len(d['toks']) + 1 for d in te)
            out.append(r); print(json.dumps({k: (v if not isinstance(v, list) else round(st.mean(v), 1)) for k, v in r.items()}), flush=True)
    else:
        A = admin_docs(load_la())
        LB = [d for d in load_lb() if any(t[0] == 'N' for t in d['toks'])]
        RBP = lb_prior_reading(LB)
        for s in range(NSPLIT):
            dr = lb_draw(LB, A, s)
            tr, te = split(dr, 'la60-c1-lbsplit%d' % s)
            RI = induce_reading(tr, lb=True)
            r = run_split(tr, te, [RI, RBP], True, rng)
            r['split'] = s; r['ntest'] = len(te); r['ntok'] = sum(len(d['toks']) + 1 for d in te)
            out.append(r); print(json.dumps({k: (v if not isinstance(v, list) else round(st.mean(v), 1)) for k, v in r.items()}), flush=True)
    json.dump(out, open(os.path.join(CK, 'c1_%s.json' % which), 'w'))


if __name__ == '__main__':
    main()
