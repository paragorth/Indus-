#!/usr/bin/env python3
"""la65 cycle 3: do the dossier roles transfer to documents outside the dossiers?
(a) Reuse: words typed TPL (kept constant in an aligned dossier slot) vs VAR (only in swapped slots).
    Outside every dossier, a template word should come back on more tablets than a swapped word.
    Statistic = mean outside-tablet count of TPL words minus that of VAR words. Null: the same typing on
    matched random groups (same sizes and site mix), 60 draws. Linear B draws are the calibration.
(b) Entry shape: for TPL words, the dossier slot says whether the word stands with a logogram and with a
    number; predict the same on outside occurrences. Score = agreement minus the agreement obtained when
    each TPL word gets the shape of another TPL word (permutation, 2000).
(c) la60 frozen reading: crosstab TPL/VAR against la60 roles; a la60-typed word used as a swapped identifier
    is a contradiction. Null: matched random groups.
usage: la65_c3.py CORPUS"""
import os, sys, json, random
from collections import Counter, defaultdict
import numpy as np
import la65_common as K
from la65_c1 import corpus
from la65_c2 import analyse, slot_type


def typing(A):
    W = defaultdict(Counter)
    shape = defaultdict(Counter)
    for a in A:
        for r in a['recs']:
            t = slot_type(r)
            tp = 'TPL' if t in ('FIX', 'NUM', 'STEP', 'GEN') else 'VAR'
            for w, l, n in zip(r['ws'], r['ls'], r['ns']):
                for x in w.split():
                    W[x][tp] += 1
                    shape[x][(bool(l), n is not None)] += 1
    typ = {w: ('TPL' if c['TPL'] else 'VAR') for w, c in W.items()}
    return typ, shape


def outside_stats(T, members, typ):
    occ = defaultdict(set)
    oshape = defaultdict(Counter)
    for t in T:
        if t['id'] in members:
            continue
        for e in t['ents']:
            for x in e['w']:
                occ[x].add(t['id'])
                oshape[x][(bool(e['l']), e['n'] is not None)] += 1
    tp = [len(occ[w]) for w, k in typ.items() if k == 'TPL']
    va = [len(occ[w]) for w, k in typ.items() if k == 'VAR']
    return (np.mean(tp) if tp else 0) - (np.mean(va) if va else 0), np.mean(tp) if tp else 0, \
        np.mean(va) if va else 0, oshape


def shape_agree(typ, shape, oshape, rng, reps=2000):
    ws = [w for w, k in typ.items() if k == 'TPL' and sum(oshape[w].values()) > 0]
    if len(ws) < 3:
        return None
    pred = {w: shape[w].most_common(1)[0][0] for w in ws}

    def agree(P):
        return np.mean([oshape[w][P[w]] / sum(oshape[w].values()) for w in ws])
    obs = agree(pred)
    null = []
    vals = [pred[w] for w in ws]
    for _ in range(reps):
        rng.shuffle(vals)
        null.append(agree(dict(zip(ws, vals))))
    null = np.array(null)
    return {'n': len(ws), 'obs': float(obs), 'null': float(null.mean()), 'p': float((1 + (null >= obs).sum()) / (1 + reps))}


def main():
    name = sys.argv[1]
    R = json.load(open(os.path.join(K.CK, 'c1_dossiers.json')))
    D = R['LANF']['dossiers'] if name == 'LAN' else R[name]['dossiers'] + R[name + 'F']['dossiers']
    seen, DD = set(), []
    for d in D:
        k = tuple(sorted(d['ids']))
        if k not in seen and len(k) >= 2:
            seen.add(k); DD.append(d)
    T, meta = corpus(name)
    pos = {t['id']: t for t in T}
    bysite = defaultdict(list)
    for t in T:
        bysite[t['site']].append(t['id'])
    rng = random.Random(K.seed('la65-c3-' + name))
    A = [analyse(T, d['ids']) for d in DD]
    mem = {i for d in DD for i in d['ids']}
    typ, shape = typing(A)
    diff, mt, mv, osh = outside_stats(T, mem, typ)
    sa = shape_agree(typ, shape, osh, rng)
    if name == 'LAN':
        import la60_common as C
        roles = C.prior_reading()['roles']
    else:
        import la63_lib as L63
        roles = {w: r for r, ws in L63.LB_KEY_W.items() for w in ws.split() if r != 'PLA'}
    contra = [w for w, k in typ.items() if k == 'VAR' and w in roles]
    agree = [w for w, k in typ.items() if k == 'TPL' and w in roles]
    nd, nc, na, nsa = [], [], [], []
    for _ in range(60):
        AR, mm = [], set()
        for d in DD:
            g = set()
            for i in d['ids']:
                c = [x for x in bysite[pos[i]['site']] if x not in g]
                g.add(rng.choice(c))
            mm |= g
            AR.append(analyse(T, sorted(g)))
        ty, sh = typing(AR)
        dd, _, _, osr = outside_stats(T, mm, ty)
        nd.append(dd)
        nc.append(sum(1 for w, k in ty.items() if k == 'VAR' and w in roles) / max(1, sum(1 for w in ty if w in roles)))
        na.append(sum(1 for w, k in ty.items() if k == 'TPL' and w in roles))
        s2 = shape_agree(ty, sh, osr, random.Random(1), reps=200)
        nsa.append(s2['obs'] - s2['null'] if s2 else 0)
    nd = np.array(nd)
    out = {'name': name, 'n_doss': len(DD), 'reuse': {'obs': float(diff), 'tpl': float(mt), 'var': float(mv),
           'null_mean': float(nd.mean()), 'p': float((1 + (nd >= diff).sum()) / 61)},
           'shape': sa, 'shape_null_excess': float(np.mean(nsa)),
           'shape_p_vs_random_groups': float((1 + sum(x >= ((sa['obs'] - sa['null']) if sa else 0) for x in nsa)) / 61),
           'la60': {'tpl_typed': sorted(agree), 'var_typed': sorted(contra),
                    'contra_share': len(contra) / max(1, len(contra) + len(agree)),
                    'null_contra_share': float(np.mean(nc)),
                    'p_low': float((1 + sum(x <= len(contra) / max(1, len(contra) + len(agree)) for x in nc)) / 61)}}
    json.dump(out, open(os.path.join(K.CK, 'c3_%s.json' % name), 'w'))
    print(json.dumps(out))


if __name__ == '__main__':
    main()
