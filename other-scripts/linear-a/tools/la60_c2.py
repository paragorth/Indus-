#!/usr/bin/env python3
"""LA-60 cycle 2: freeze the reading on training documents (sha256), then decode the held-out half.
Counts: FULL (cover, roles, totals, order, substantive), STRICT (FULL and at least one closing total
or one order-agreeing commodity pair), closing totals, order agreements/violations.
Controls through the identical pipeline: the reading with roles shuffled among its types (and order
scores shuffled), a random reading of the same class sizes over training vocabulary, the EMPTY
reading. Calibration: Linear B KN+PY at Linear A size (matched mix of document lengths) with the
same rules (INDUCED) and with a true partial reading of the same kind (LBPRIOR)."""
import json, os, sys, random, statistics as st
from la60_common import *
from la60_decode import *

NSPLIT = int(os.environ.get('NSPLIT', 10)); NSH = int(os.environ.get('NSH', 100))
which = sys.argv[1] if len(sys.argv) > 1 else 'LA'


def score(tr, te, R, lb):
    acc = acceptor(tr, R)
    c = Counter()
    for d in te:
        ch, g = decode(d, R, acc, lb=lb)
        strict = ch['full'] and (ch['n_close'] > 0 or ch['n_order_agree'] > 0)
        c['full'] += ch['full']; c['strict'] += strict; c['close'] += ch['n_close']
        c['tot_tested'] += ch['n_tot']; c['agree'] += ch['n_order_agree']; c['viol'] += ch['n_order_viol']
        c['cover'] += ch['cover']; c['roles'] += ch['roles']
    return dict(c)


def random_reading(tr, R, rng):
    vocab = sorted({t[1] for d in tr for t in d['toks'] if t[0] == 'W'})
    labs = list(R['roles'].values())
    types = rng.sample(vocab, min(len(labs), len(vocab)))
    roles = dict(zip(types, labs))
    bases = sorted({base_of(t[1]) for d in tr for t in d['toks'] if t[0] == 'L'} | {base_of(w) for w, r in roles.items() if r == 'COM'})
    ov = list(R['order'].values()); rng.shuffle(ov)
    ob = rng.sample(bases, min(len(ov), len(bases)))
    return dict(name='RANDOM', roles=roles, order=dict(zip(ob, ov)), lib=set(), site_default={}, ent_slot_unknown=True)


def evaluate(tr, te, R, lb, rng):
    real = score(tr, te, R, lb)
    sh = [score(tr, te, shuffle_reading(R, rng), lb) for _ in range(NSH)]
    rnd = [score(tr, te, random_reading(tr, R, rng), lb) for _ in range(NSH // 2)]
    out = dict(real=real)
    for nm, L in [('shuf', sh), ('rand', rnd)]:
        for k in real:
            v = [x.get(k, 0) for x in L]
            out[nm + '_' + k] = (st.mean(v), sum(1 for x in v if x >= real[k]) / len(v))
    return out


def main():
    res = []
    rng = random.Random(seed('la60-c2-' + which))
    EMPTY = dict(name='EMPTY', roles={}, order={}, lib=set(), site_default={}, ent_slot_unknown=True)
    if which == 'LA':
        A = admin_docs(load_la())
        # --- primary split: freeze, then decode
        tr, te = split(A, 'la60-main')
        RI = induce_reading(tr); RP = prior_reading()
        frozen = {}
        for R in (RI, RP):
            fz = dict(roles=R['roles'], order=R['order'], lib=sorted(R['lib']), site_default=R['site_default'])
            json.dump(fz, open(os.path.join(CK, 'frozen_%s.json' % R['name']), 'w'), ensure_ascii=False, indent=0, sort_keys=True)
            frozen[R['name']] = reading_hash(R)
        json.dump(frozen, open(os.path.join(CK, 'frozen_hashes.json'), 'w'), indent=1)
        print('FROZEN', frozen, 'train docs', len(tr), 'test docs', len(te), flush=True)
        with open(os.path.join(CK, 'c2_glosses.txt'), 'w') as f:
            for R in (RP, RI):
                acc = acceptor(tr, R)
                for d in te:
                    ch, g = decode(d, R, acc)
                    f.write('%s\t%s\t%s\n' % (R['name'], 'FULL' if ch['full'] else '-', g))
        for s in range(-1, NSPLIT):
            if s >= 0:
                tr, te = split(A, 'la60-c2-split%d' % s)
                RI = induce_reading(tr)
            r = dict(split=s, ntest=len(te), EMPTY=score(tr, te, EMPTY, False))
            for R in (RI, RP):
                r[R['name']] = evaluate(tr, te, R, False, rng)
            res.append(r); print(json.dumps(r), flush=True)
    else:
        A = admin_docs(load_la())
        LB = [d for d in load_lb() if any(t[0] == 'N' for t in d['toks'])]
        RBP = lb_prior_reading(LB)
        for s in range(NSPLIT):
            dr = lb_draw(LB, A, s)
            tr, te = split(dr, 'la60-c2-lbsplit%d' % s)
            RI = induce_reading(tr, lb=True)
            r = dict(split=s, ntest=len(te), EMPTY=score(tr, te, EMPTY, True))
            for R in (RI, RBP):
                r[R['name']] = evaluate(tr, te, R, True, rng)
            res.append(r); print(json.dumps(r), flush=True)
            if s == 0:
                with open(os.path.join(CK, 'c2_glosses_LB.txt'), 'w') as f:
                    acc = acceptor(tr, RBP)
                    for d in te:
                        ch, g = decode(d, RBP, acc, lb=True)
                        f.write('%s\t%s\n' % ('FULL' if ch['full'] else '-', g))
    json.dump(res, open(os.path.join(CK, 'c2_%s.json' % which), 'w'))


if __name__ == '__main__':
    main()
