#!/usr/bin/env python3
"""la47 engine: random-grammar search on one corpus x target, held-out scoring.
For each of `splits` random halvings of the page units (A search / B held-out):
  * G random spatial grammars and G random order grammars (depth <= 3 trees over feature values only)
    are scored on A by 2-fold CV (units); the top 10 of each are refitted on A and scored on B.
  * saturated tables: all S features, all O features, O x S, each fitted on A, scored on B.
  * increment: best-on-A order grammar crossed with best-on-A spatial grammar, scored on B, minus order alone.
Returns bits/token (held-out gain over the unigram)."""
import random, collections, numpy as np
import la47_common as C


def cells(rows, key, feats):
    lab = {}; idx = []
    for r in rows:
        a = tuple(r[key][f] for f in feats); idx.append(lab.setdefault(a, len(lab)))
    inv = [dict(zip(feats, a)) for a in sorted(lab, key=lab.get)]
    return np.array(idx), inv


def grammar_roles(t, inv, idx):
    lab = {}; m = np.array([lab.setdefault(C.apply_grammar(t, f), len(lab)) for f in inv])
    return m[idx]


def bits_vec(roles, y, tr, te):
    return C.heldout_bits(list(roles[tr]), [y[i] for i in tr], list(roles[te]), [y[i] for i in te])


def cv_vec(roles, y, f1, f2):
    return 0.5 * (bits_vec(roles, y, f1, f2) + bits_vec(roles, y, f2, f1))


def bits_sel(roles, y, f1, f2, tr, te):
    """alpha chosen by 2-fold CV inside the search half, then fit on the whole search half, scored on held-out"""
    a = int(np.argmax(cv_vec(roles, y, f1, f2)))
    return float(bits_vec(roles, y, tr, te)[a]), float(cv_vec(roles, y, f1, f2)[a])


def run(rows, target, G=3000, splits=4, seed=0, sfeat=None, ofeat=None):
    sfeat = sfeat or C.SFEAT; ofeat = ofeat or C.OFEAT
    rng = random.Random(seed)
    rows = [r for r in rows if target in r]
    y = [r[target] for r in rows]
    units = sorted({r['unit'] for r in rows})
    si, sinv = cells(rows, 's', sfeat); oi, oinv = cells(rows, 'o', ofeat)
    svals = C.feat_values(rows, 's', sfeat); ovals = C.feat_values(rows, 'o', ofeat)
    # saturated
    so = np.array([hash((a, b)) for a, b in zip(si, oi)]); _, so = np.unique(so, return_inverse=True)
    res = collections.defaultdict(list)
    for sp in range(splits):
        u = units[:]; rng.shuffle(u); A = set(u[:len(u) // 2])
        tr = np.array([k for k, r in enumerate(rows) if r['unit'] in A]); te = np.array([k for k, r in enumerate(rows) if r['unit'] not in A])
        Au = sorted(A); rng.shuffle(Au); A1 = set(Au[:len(Au) // 2])
        f1 = np.array([k for k in tr if rows[k]['unit'] in A1]); f2 = np.array([k for k in tr if rows[k]['unit'] not in A1])
        res['sat_S'].append(bits_sel(si, y, f1, f2, tr, te)[0]); res['sat_O'].append(bits_sel(oi, y, f1, f2, tr, te)[0]); res['sat_OS'].append(bits_sel(so, y, f1, f2, tr, te)[0])
        best = {}
        for kind, inv, idx, vals in (('S', sinv, si, svals), ('O', oinv, oi, ovals)):
            sc = []
            for g in range(G):
                t = C.random_grammar(vals, rng)
                ro = grammar_roles(t, inv, idx)
                cv = float(cv_vec(ro, y, f1, f2).max())
                sc.append((cv, g, t, ro))
            sc.sort(key=lambda x: -x[0])
            top = sc[:10]
            tb = [bits_sel(ro, y, f1, f2, tr, te)[0] for _, _, _, ro in top]
            res['top10_' + kind].append(float(np.mean(tb))); res['best_' + kind].append(tb[0])
            res['cv_' + kind].append(top[0][0])
            best[kind] = top[0]
            if sp == 0: res['grammar_' + kind] = C.grammar_str(top[0][2])
        pr = np.array([hash((a, b)) for a, b in zip(best['O'][3], best['S'][3])]); _, pr = np.unique(pr, return_inverse=True)
        res['best_OxS'].append(bits_sel(pr, y, f1, f2, tr, te)[0])
        res['incr_S_over_O'].append(res['best_OxS'][-1] - res['best_O'][-1])
    out = {}
    for k, v in res.items():
        out[k] = v if isinstance(v, str) else round(float(np.mean(v)), 4)
    out['n'] = len(rows)
    return out


def run_fixed(rows, target, train_pred, G=2000, seed=0, sfeat=None, ofeat=None):
    """search on rows where train_pred(row) (2-fold CV inside), score top grammars on the rest"""
    sfeat = sfeat or C.SFEAT; ofeat = ofeat or C.OFEAT
    rng = random.Random(seed)
    rows = [r for r in rows if target in r]; y = [r[target] for r in rows]
    si, sinv = cells(rows, 's', sfeat); oi, oinv = cells(rows, 'o', ofeat)
    svals = C.feat_values(rows, 's', sfeat); ovals = C.feat_values(rows, 'o', ofeat)
    tr = np.array([k for k, r in enumerate(rows) if train_pred(r)]); te = np.array([k for k, r in enumerate(rows) if not train_pred(r)])
    Au = sorted({rows[k]['unit'] for k in tr}); rng.shuffle(Au); A1 = set(Au[:len(Au) // 2])
    f1 = np.array([k for k in tr if rows[k]['unit'] in A1]); f2 = np.array([k for k in tr if rows[k]['unit'] not in A1])
    out = {'n_train': len(tr), 'n_test': len(te)}
    best = {}
    for kind, inv, idx, vals in (('S', sinv, si, svals), ('O', oinv, oi, ovals)):
        sc = []
        for g in range(G):
            t = C.random_grammar(vals, rng); ro = grammar_roles(t, inv, idx)
            sc.append((float(cv_vec(ro, y, f1, f2).max()), t, ro))
        sc.sort(key=lambda x: -x[0])
        tb = [bits_sel(ro, y, f1, f2, tr, te)[0] for _, _, ro in sc[:10]]
        out['top10_' + kind] = round(float(np.mean(tb)), 4); out['best_' + kind] = round(tb[0], 4)
        out['grammar_' + kind] = C.grammar_str(sc[0][1]); best[kind] = sc[0]
    pr = np.array([hash((a, b)) for a, b in zip(best['O'][2], best['S'][2])]); _, pr = np.unique(pr, return_inverse=True)
    out['incr_S_over_O'] = round(bits_sel(pr, y, f1, f2, tr, te)[0] - out['best_O'], 4)
    return out
