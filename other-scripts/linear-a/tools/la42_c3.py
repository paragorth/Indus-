#!/usr/bin/env python3
"""LA-42 cycle 3: (a) 'new row' calibration: hide a whole consonant row (all its signs unvalued) and ask whether
each of its signs is put in a NEW row, on LB draws at LA size and on LA itself; (b) freeze predictions for the
unvalued LA signs used inside words (>= 5 tokens), independent and joint (one sign per cell, Hungarian) assignment,
with the LB-frozen model (ALL) and the LA nested model (ALL+r21) fitted on all LA leave-one-out cases."""
import os, sys, pickle, collections, json
import numpy as np
from scipy.optimize import linear_sum_assignment
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import la42_common as G

UNK = ['PA3', '*301', '*118', '*21F', '*306', '*310', '*86', '*188', '*305', '*34', '*47', '*28B', '*325', '*314']
R21 = G.la21('LA')


def name(cv):
    c, v = cv
    return (c.upper() + v.upper()) if c != G.NEW else f'NEW-{v}'


def row_hide(words, V, row, model, r21=None):
    mem = [s for s, x in V.items() if x[0] == row]
    Vh = {s: x for s, x in V.items() if x[0] != row}
    out = []
    for h in mem:
        Vx = dict(Vh); Vx[h] = V[h]
        c = G.make_case(words, Vx, h, r21=r21)
        if c is None:
            continue
        p = G.predict(model, c)
        rows = sorted({x[0] for x in c['cands']})
        pr = {r: p[[i for i, x in enumerate(c['cands']) if x[0] == r]].sum() for r in rows}
        pv = {v: p[[i for i, x in enumerate(c['cands']) if x[1] == v]].sum() for v in G.VOW}
        out.append(dict(sign=h, n=c['ntok'], pnew=pr[G.NEW], newtop=max(pr, key=pr.get) == G.NEW,
                        coltop=max(pv, key=pv.get) == V[h][1], toprow=max(pr, key=pr.get)))
    return out


def main():
    lines = []
    mALL = pickle.load(open(os.path.join(G.CK, 'c1_model_ALL.pkl'), 'rb'))
    # (a) LB
    U = G.lb_units()
    res = collections.defaultdict(list)
    for d in range(6):
        W = G.lb_draw(U, 3569, seed=500 + d)
        V = G.known_values(W)
        for row in ('j', 'z', 'w', 'q', 'm', 'n'):
            res[row] += row_hide(W, V, row, mALL)
    allr = [x for v in res.values() for x in v]
    lines.append(f"(a) LB draws (6, LA size), whole row hidden (j z w q m n): sign put in NEW row top-1 {np.mean([x['newtop'] for x in allr]):.2f} "
                 f"(n {len(allr)}), mean P(new) {np.mean([x['pnew'] for x in allr]):.2f}; column top-1 {np.mean([x['coltop'] for x in allr]):.2f}; "
                 + '; '.join(f"{r}: new {np.mean([x['newtop'] for x in v]):.2f}" for r, v in res.items()))
    # baseline: P(new) for signs whose row is present (c1 LB cases, full)
    c1 = pickle.load(open(os.path.join(G.CK, 'c1_lbcases.pkl'), 'rb'))
    pn = []
    for c in c1:
        if c['keep'] == 0 and c['cands'][c['truth']][0] != G.NEW:
            p = G.predict(mALL, c); pn.append(p[[i for i, x in enumerate(c['cands']) if x[0] == G.NEW]].sum())
    lines.append(f"    reference: signs whose row is present: mean P(new) {np.mean(pn):.2f}, share with P(new) > 0.5 {np.mean(np.array(pn) > .5):.2f}")
    print('\n'.join(lines), flush=True)
    W = [r['w'] for r in G.la_units()]
    V = G.known_values(W)
    la = []
    for row in ('j', 'z', 'w', 'q', 'm', 'n', 'p'):
        r = row_hide(W, V, row, mALL); la += r
        lines.append(f"    LA row {row} hidden: " + ', '.join(f"{x['sign']} -> {(x['toprow'] or 'V') if x['toprow'] != G.NEW else 'NEW'} P(new) {x['pnew']:.2f}" for x in r))
    lines.append(f"(a) LA whole row hidden: NEW top-1 {np.mean([x['newtop'] for x in la]):.2f} (n {len(la)}), mean P(new) {np.mean([x['pnew'] for x in la]):.2f}")
    print('\n'.join(lines[-8:]), flush=True)
    # (b) freeze
    c2 = pickle.load(open(os.path.join(G.CK, 'c2_cases.pkl'), 'rb'))
    lacases = [c for c in c2 if c['tag'] == 'LA' and c['keep'] == 0]
    mLA = G.fit(lacases, mALL['use'] + ['r21'])
    pickle.dump(mLA, open(os.path.join(G.CK, 'c3_model_LAr21.pkl'), 'wb'))
    lines.append('LA-fitted ALL+r21 weights: ' + ', '.join(f'{f} {w:+.3g}' for f, w in zip(mLA['use'], mLA['w'])))
    cnt = collections.Counter(s for w in W for s in w)
    frozen = {}
    P = {}
    for u in UNK:
        c = G.make_case(W, V, u, r21=R21)
        for nm, m in (('ALL', mALL), ('LAr21', mLA)):
            p = G.predict(m, c); o = np.argsort(-p)
            rows = sorted({x[0] for x in c['cands']})
            pr = {r: float(p[[i for i, x in enumerate(c['cands']) if x[0] == r]].sum()) for r in rows}
            pv = {v: float(p[[i for i, x in enumerate(c['cands']) if x[1] == v]].sum()) for v in G.VOW}
            frozen.setdefault(u, {})[nm] = dict(n=c['ntok'], top=[(name(c['cands'][i]), round(float(p[i]), 3)) for i in o[:5]],
                                                rows=sorted(pr.items(), key=lambda x: -x[1])[:3], cols=sorted(pv.items(), key=lambda x: -x[1]),
                                                in_la21=u in R21[0])
            P[(u, nm)] = (c['cands'], p)
        f = frozen[u]
        lines.append(f"(b) {u:5s} n={cnt[u]:3d} (in words {c['ntok']}) la21={'y' if u in R21[0] else 'n'} | ALL: {f['ALL']['top'][:3]} rows {[(r or 'V', round(q, 2)) for r, q in f['ALL']['rows']]} "
                     f"| LA+r21: {f['LAr21']['top'][:3]} rows {[(r or 'V', round(q, 2)) for r, q in f['LAr21']['rows']]} cols {[(v, round(q, 2)) for v, q in f['LAr21']['cols'][:2]]}")
        print(lines[-1], flush=True)
    # joint assignment (each existing empty cell used once; NEW cells may be shared -> duplicate NEW columns)
    for nm in ('ALL', 'LAr21'):
        cands = sorted({x for u in UNK for x in P[(u, nm)][0]})
        cols = []
        for x in cands:
            cols += [x] * (len(UNK) if x[0] == G.NEW else 1)
        M = np.full((len(UNK), len(cols)), 50.0)
        for i, u in enumerate(UNK):
            cc, p = P[(u, nm)]
            d = {x: -np.log(q + 1e-12) for x, q in zip(cc, p)}
            for j, x in enumerate(cols):
                if x in d:
                    M[i, j] = d[x]
        r, cidx = linear_sum_assignment(M)
        joint = {UNK[i]: name(cols[j]) for i, j in zip(r, cidx)}
        for u in UNK:
            frozen[u][nm]['joint'] = joint[u]
        lines.append(f"(b) joint one-per-cell assignment {nm}: " + ', '.join(f'{u}={joint[u]}' for u in UNK))
        print(lines[-1], flush=True)
    json.dump(frozen, open(os.path.join(G.CK, 'c3_frozen.json'), 'w'), indent=1)
    open(os.path.join(G.CK, 'c3_report.txt'), 'w').write('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
