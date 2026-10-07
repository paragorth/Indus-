"""LA-79 cycle 1: re-grade earlier Linear A claims under every counterfactual excavation history.

Each claim is scored on the 'later finds' of 62 site-subset histories, the 3 real publication
cuts and 40 random-document splits (shuffled history), against its own decoy family.
Planted controls: a universal prefix and an HT-only prefix; a universal and an HT-only commodity order.
"""
import json, os, sys, copy
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from la79_common import load, splits, CK, GROUPS
import la79_claims as K

docs0 = load()
SPL = splits(docs0, n_random=40)
signs = sorted({s for d in docs0 for w in d['words'] for s in w['s']})
FEATS = [('first', None)] + K.make_feature_decoys(signs, n=300)
cvs = [s for s in signs if K.consonant(s) is not None]
real_map = {s: K.consonant(s) for s in cvs}
rng = np.random.default_rng(11)
vals = [real_map[s] for s in cvs]
MAPS = [real_map] + [dict(zip(cvs, rng.permutation(vals))) for _ in range(500)]
LEX = K.lb_lexicon()
PRE, SUF = ['I', 'SI', 'A', 'KI'], ['ME', 'JA', 'RE', 'TE']
FR_ITEMS = ['JE', 'L', 'E', 'J', 'Y', 'A', 'F', 'H', 'B', 'K', 'L6', 'L2', 'L4', 'D', 'W', 'X', 'L3', 'DD']


def pct_of(d, key):
    if key not in d:
        return None
    v = np.array([x for k, x in d.items() if k != key])
    return float((v < d[key]).mean())


def eval_split(docs, tr, te):
    TR = [docs[i] for i in tr]
    TE = [docs[i] for i in te]
    r = {}
    for side, keys in (('pre', PRE), ('suf', SUF)):
        ztr, zte = K.affix_z(TR, side), K.affix_z(TE, side)
        for k in keys:
            r['%s_%s' % (side, k)] = dict(z=zte.get(k), pct=pct_of(zte, k), ztr=ztr.get(k))
        za = K.affix_all_z(TE, side)
        if za is not None:
            r['%s_all' % side] = dict(z=za)
        # selection transfer: top-4 by train z, their mean test z vs all
        if ztr and zte:
            top = [k for k, _ in sorted(ztr.items(), key=lambda x: -x[1])[:4] if k in zte]
            if top:
                r['%s_sel' % side] = dict(z=float(np.mean([zte[k] for k in top]) - np.mean(list(zte.values()))), top=top)
    TEn = [d for d in TE if d['g'] != 'HT']
    zc = K.consec_z(TEn, FEATS)
    r['consec_first_nonHT'] = dict(z=float(zc[0]), pct=float((zc[1:] < zc[0]).mean()), n=len(TEn))
    zk = K.cons_avoid_z(TE, MAPS)
    if zk is not None:
        r['cons_avoid'] = dict(z=float(-zk[0]), pct=float((zk[1:] > zk[0]).mean()))
    z, obs = K.lb_z(TE, LEX)
    if z is not None:
        r['lb_shared'] = dict(z=float(z), obs=obs)
    co = K.order_eval([K.first_order(d['logos'], K.BIG) for d in TR], [K.first_order(d['logos'], K.BIG) for d in TE], K.BIG, n_dec=2000)
    if co:
        r['comm_order'] = dict(z=None, pct=co['pct'], agree=co['agree'], n=co['n'])
    fo = K.order_eval([K.first_order(f, FR_ITEMS) for d in TR for f, _ in d['fracs'] if len(set(f)) > 1],
                      [K.first_order(f, FR_ITEMS) for d in TE for f, _ in d['fracs'] if len(set(f)) > 1], FR_ITEMS, n_dec=2000)
    if fo:
        r['frac_order'] = dict(z=None, pct=fo['pct'], agree=fo['agree'], n=fo['n'])
    fc = K.frac_comm_eval(TR, TE, n_dec=100)
    if fc:
        r['frac_comm'] = dict(z=None, pct=fc['pct'], gain=fc['gain'], n=fc['n'])
    return r


def success(name, v):
    if v is None:
        return None
    if name.endswith('_sel'):
        return v['z'] > 1.0
    if name not in ('comm_order', 'frac_order', 'frac_comm'):
        if v.get('z') is None:
            return None
        ok = v['z'] >= 2.0
        if 'pct' in v and v['pct'] is not None:
            ok = ok and v['pct'] >= 0.9
        return ok
    return v['pct'] >= 0.95


def run(docs, spl, label):
    res = []
    for name, kind, tr, te in spl:
        res.append(dict(split=name, kind=kind, ntr=len(tr), nte=len(te), r=eval_split(docs, tr, te)))
        print(label, name, flush=True)
    return res


def plant_affix(docs, X, groups, frac=0.08, seed=21):
    rng = np.random.default_rng(seed)
    D = copy.deepcopy(docs)
    for g in groups:
        idx = [i for i, d in enumerate(D) if d['g'] == g and d['words']]
        T = sorted({w['s'] for i in idx for w in D[i]['words'] if len(w['s']) >= 2})
        k = max(1, int(frac * len(T)))
        for j in rng.choice(len(T), size=min(k, len(T)), replace=False):
            i = idx[rng.integers(len(idx))]
            D[i]['words'].append(dict(s=(X,) + T[j], read=True, nxt='num'))
    return D


def plant_order(docs, groups, seed=22):
    rng = np.random.default_rng(seed)
    true = list(rng.permutation(K.BIG))
    rk = {x: i for i, x in enumerate(true)}
    D = copy.deepcopy(docs)
    for d in D:
        big = [x for x in d['logos'] if x in K.BIG]
        if len(big) < 2:
            continue
        if d['g'] in groups and rng.random() < 0.9:
            nb = sorted(big, key=lambda x: rk[x])
        else:
            nb = list(rng.permutation(big))
        it = iter(nb)
        d['logos'] = [next(it) if x in K.BIG else x for x in d['logos']]
    return D


if __name__ == '__main__':
    mode = sys.argv[1]
    if mode == 'real':
        full = eval_split(docs0, np.arange(len(docs0)), np.arange(len(docs0)))
        res = run(docs0, SPL, 'real')
        json.dump(dict(full=full, res=res), open(os.path.join(CK, 'c1_real.json'), 'w'), default=float)
    else:
        # planted worlds: pick a prefix sign with ~0 real prefix z
        z = K.affix_z(docs0, 'pre')
        X = sorted([k for k, v in z.items() if abs(v) < 0.4], key=lambda k: k)[0]
        out = {}
        sub = [s for s in SPL if s[2] is not None]
        for tag, groups in (('uni', GROUPS), ('ht', ['HT'])):
            D = plant_order(plant_affix(docs0, X, groups), groups)
            rr = []
            for name, kind, tr, te in sub:
                TR = [D[i] for i in tr]; TE = [D[i] for i in te]
                zte = K.affix_z(TE, 'pre')
                co = K.order_eval([K.first_order(d['logos'], K.BIG) for d in TR],
                                  [K.first_order(d['logos'], K.BIG) for d in TE], K.BIG, n_dec=1000)
                rr.append(dict(split=name, kind=kind, z=zte.get(X), pct=pct_of(zte, X),
                               opct=co['pct'] if co else None, oagree=co['agree'] if co else None,
                               te_has_ht=any(d['g'] == 'HT' for d in TE), tr_has_ht=any(d['g'] == 'HT' for d in TR)))
            out[tag] = rr
            print('planted', tag, flush=True)
        json.dump(dict(X=X, out=out), open(os.path.join(CK, 'c1_planted.json'), 'w'), default=float)
