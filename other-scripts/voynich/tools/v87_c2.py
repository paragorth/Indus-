"""v87 cycle 2: do Voynich sections differ in PLAINTEXT or only in ENCODING?
Calibration: planted section pairs (a) same source + same scheme, (b) same source + different scheme,
(c) different source + same scheme. Posterior differences (ptt, P(list), encoder unit) of the two sections
are compared between pair types; then the same differences are measured between Voynich sections."""
import os, json, random, math
from collections import Counter
from multiprocessing import Pool
import numpy as np
import v87_lib as L
import v87_common as C
import v87_bank as B

R, F, P = C.load_bank('base')
sid = np.array([r['sid'] for r in R]); unit = np.array([r['unit'] for r in R])
UNITS = ['letter', 'chunk', 'word', 'nomen']
scale = L.scale_of(F[:, C.TI])


def post(vec, excl):
    m = ~np.isin(sid, list(excl))
    po = L.abc(F[:, C.TI], P, vec[C.TI], scale, mask=m)
    u = Counter();
    for i, w in zip(po['idx'], po['w']): u[unit[i]] += w
    t = sum(u.values())
    po['unit'] = [u[x] / t for x in UNITS]
    return po


def make_pair(args):
    k, kind = args
    rng = random.Random(870000 + k)
    s1 = rng.choice(B.SIDS); sc1 = L.random_scheme(rng)
    s2 = s1 if kind in ('a', 'b') else rng.choice([s for s in B.SIDS if s != s1])
    sc2 = L.random_scheme(rng) if kind == 'b' else dict(sc1)
    _, l1, n1 = L.simulate(B.S[s1], sc1, rng)
    _, l2, n2 = L.simulate(B.S[s2], sc2, rng)
    if min(n1, n2) < 1800: return None
    return kind, s1, s2, sc1['unit'], sc2['unit'], C.fvec(l1).tolist(), C.fvec(l2).tolist()


def diff(p1, p2):
    return dict(dlist=abs(p1['is_list'] - p2['is_list']), dptt=abs(p1['ptt'] - p2['ptt']), dpwl=abs(p1['pwl'] - p2['pwl']),
                dunit=0.5 * sum(abs(a - b) for a, b in zip(p1['unit'], p2['unit'])))


def auc(pos, neg):
    pos = np.asarray(pos); neg = np.asarray(neg)
    return float(((pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()) / (len(pos) * len(neg)))


if __name__ == '__main__':
    out = {}
    jobs = [(k, kind) for kind in 'abc' for k in range({'a': 0, 'b': 1000, 'c': 2000}[kind], {'a': 0, 'b': 1000, 'c': 2000}[kind] + 220)]
    with Pool(2) as pool:
        pairs = [p for p in pool.map(make_pair, jobs, chunksize=10) if p]
    D = {'a': [], 'b': [], 'c': []}; unit_acc = []
    for kind, s1, s2, u1, u2, f1, f2 in pairs:
        p1 = post(np.array(f1), {s1, s2}); p2 = post(np.array(f2), {s1, s2})
        D[kind].append(diff(p1, p2))
        unit_acc.append(UNITS[int(np.argmax(p1['unit']))] == u1)
    out['n_pairs'] = {k: len(v) for k, v in D.items()}
    out['unit_recovery_acc'] = float(np.mean(unit_acc))
    out['unit_chance'] = float(max(Counter(unit).values()) / len(unit))
    cal = {}
    for m in ['dlist', 'dptt', 'dpwl', 'dunit']:
        A = [d[m] for d in D['a']]; Bv = [d[m] for d in D['b']]; Cv = [d[m] for d in D['c']]
        cal[m] = dict(a_med=float(np.median(A)), b_med=float(np.median(Bv)), c_med=float(np.median(Cv)),
                      auc_c_vs_b=auc(Cv, Bv), auc_b_vs_a=auc(Bv, A), auc_c_vs_a=auc(Cv, A))
    out['calibration'] = cal; print(json.dumps(cal, indent=1))

    # ---- Voynich sections
    for name in ['ZL3b', 'IT2a']:
        vl = L.voy_lines(name)
        secs = L.voy_chunks(vl, key=lambda l: l['sec'])
        rows = []
        for s, chs in secs.items():
            for c in chs:
                po = post(C.fvec(c), set())
                rows.append(dict(sec=s, is_list=po['is_list'], ptt=po['ptt'], pwl=po['pwl'], unit=po['unit'], dist=po['dist']))
        # section means
        S = {}
        for s in sorted(set(r['sec'] for r in rows)):
            rr = [r for r in rows if r['sec'] == s]
            S[s] = dict(n=len(rr), is_list=float(np.mean([r['is_list'] for r in rr])), ptt=float(np.mean([r['ptt'] for r in rr])),
                        types_per_1000=float(1000 * math.exp(np.mean([r['ptt'] for r in rr]))),
                        pwl=float(np.mean([r['pwl'] for r in rr])), unit=np.mean([r['unit'] for r in rr], 0).tolist(),
                        dist=float(np.mean([r['dist'] for r in rr])))
        # between-chunk differences: same section vs different section, and Currier A vs B
        def pd(r1, r2):
            return dict(dlist=abs(r1['is_list'] - r2['is_list']), dptt=abs(r1['ptt'] - r2['ptt']), dpwl=abs(r1['pwl'] - r2['pwl']),
                        dunit=0.5 * sum(abs(a - b) for a, b in zip(r1['unit'], r2['unit'])))
        same = []; diffsec = []; ab = []
        for i in range(len(rows)):
            for j in range(i + 1, len(rows)):
                d = pd(rows[i], rows[j])
                (same if rows[i]['sec'] == rows[j]['sec'] else diffsec).append(d)
                if rows[i]['sec'][1] != rows[j]['sec'][1] and '?' not in rows[i]['sec'] + rows[j]['sec']:
                    ab.append(d)
        V = {'sections': S, 'n_chunks': len(rows)}
        for m in ['dlist', 'dptt', 'dpwl', 'dunit']:
            V[m] = dict(same=float(np.median([d[m] for d in same])), diff=float(np.median([d[m] for d in diffsec])),
                        AvsB=float(np.median([d[m] for d in ab])),
                        # where the Voynich between-section difference falls among calibration pair types
                        pct_in_a=float(np.mean([d[m] for d in D['a']] <= np.median([d[m] for d in diffsec]))),
                        pct_in_b=float(np.mean([d[m] for d in D['b']] <= np.median([d[m] for d in diffsec]))),
                        pct_in_c=float(np.mean([d[m] for d in D['c']] <= np.median([d[m] for d in diffsec]))))
        # permutation test: is ptt between-section variance above chunk-label permutation?
        rs = np.random.default_rng(1)
        lab = [r['sec'] for r in rows]; vals = np.array([r['ptt'] for r in rows]); vl_ = np.array([r['is_list'] for r in rows])
        def bvar(lab, x):
            g = {}
            for l_, v in zip(lab, x): g.setdefault(l_, []).append(v)
            return sum(len(v) * (np.mean(v) - x.mean()) ** 2 for v in g.values())
        for nm, x in [('ptt', vals), ('is_list', vl_)]:
            obs = bvar(lab, x); null = [bvar(list(rs.permutation(lab)), x) for _ in range(2000)]
            V['perm_' + nm] = dict(obs=float(obs), p=float((np.sum(np.array(null) >= obs) + 1) / 2001))
        out[name] = V
        print(name, json.dumps(V, indent=1, default=float))
    json.dump(out, open(os.path.join(L.CK, 'c2.json'), 'w'), indent=1, default=float)
