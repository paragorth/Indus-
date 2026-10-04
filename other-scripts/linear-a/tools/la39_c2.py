#!/usr/bin/env python3
"""LA-39 cycle 2.
(a) Word context: do ligatures sharing an added part sit among the same WORDS across different
    bases (documents shared by both types removed), relative to other added parts on the same two
    bases?  Null: added parts shuffled among ligature types.  LB control: sex markers, SI.
(b) Massive random guessing: 5,000 random partitions of the added parts into K=2..4 latent
    'meaning classes'; each scored by cross-base transfer on 70 % of documents; the top 50 re-tested on
    the held-out 30 %.  Controls: the same search on type-shuffled labels (8 runs) and on a planted
    2-class system.
"""
import sys, os, json
import numpy as np
from collections import defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import la39_common as L

rng = np.random.default_rng(392)
LA_TOTALW = {'KU-RO', 'KI-RO', 'PO-TO-KU-RO', 'to-so', 'to-sa'}


def doc_words_la():
    C = json.load(open(os.path.join(L.D, 'corpus.json')))
    return {str(d['id']): set(w for w in d['words'] if w not in LA_TOTALW and '-' in w) for d in C}


def doc_words_lb():
    import re
    out = {}
    for line in open(os.path.join(L.D, 'damos_items.jsonl')):
        d = json.loads(line)
        ws = set()
        for t in (d.get('content') or '').split():
            t = L._strip(t).strip('[]')
            if re.match(r'^[a-z0-9]+(-[a-z0-9]+)+$', t) and t not in LA_TOTALW: ws.add(t)
        out[str(d['id'])] = ws
    return out


_SIMS = {}


def _sims(A, DW):
    key = id(A)
    if key in _SIMS: return _SIMS[key]
    types = defaultdict(set)
    for b, m, d in zip(A['base'], A['mod'], A['doc']):
        if m: types[(b, m)].add(d)
    tl = list(types); sims = {}
    for i in range(len(tl)):
        for j in range(i + 1, len(tl)):
            if tl[i][0] == tl[j][0]: continue
            d1, d2 = types[tl[i]] - types[tl[j]], types[tl[j]] - types[tl[i]]
            w1 = set().union(*[DW.get(d, set()) for d in d1]) if d1 else set()
            w2 = set().union(*[DW.get(d, set()) for d in d2]) if d2 else set()
            if not w1 or not w2: continue
            sims[(i, j)] = len(w1 & w2) / len(w1 | w2)
    _SIMS[key] = (tl, sims)
    return tl, sims


def context_stat(A, DW, mod, groups):
    tl, sims = _sims(A, DW)
    newlab = {}
    for b, m0, m in zip(A['base'], A['mod'], mod):
        if m0: newlab[(b, m0)] = m
    bp = defaultdict(lambda: [[], []])
    for (i, j), s in sims.items():
        (b1, _), (b2, _) = tl[i], tl[j]
        m1, m2 = newlab[tl[i]], newlab[tl[j]]
        key = tuple(sorted((b1, b2)))
        bp[key][int(m1 == m2)].append((s, m1 if m1 == m2 else None))
    out = defaultdict(list)
    for key, (diff, same) in bp.items():
        if not diff or not same: continue
        md = np.mean([s for s, _ in diff])
        for s, m in same:
            out['all'].append(s - md)
            for gn, gs in groups.items():
                if m in gs: out[gn].append(s - md)
    return {k: (float(np.mean(v)), len(v)) for k, v in out.items()}


def run_context(A, DW, groups, label, nperm=1000):
    obs = context_stat(A, DW, A['mod'], groups)
    null = defaultdict(list)
    for _ in range(nperm):
        r = context_stat(A, DW, L.shuffle_mods(A, rng), groups)
        for k, v in r.items(): null[k].append(v[0])
    res = {}
    for k, (v, n) in obs.items():
        nv = np.array(null[k])
        res[k] = (round(v, 4), n, round(float(nv.mean()), 4), round(float((1 + (nv >= v).sum()) / (1 + len(nv))), 4))
    print(label, 'context', res, flush=True)
    return res


def classes_label(A, part):
    return np.array([part.get(m, '') if m else '' for m in A['mod']])


def search(A, mod, sel, test, nh=5000, top=50, inner=4):
    """sel/test: boolean token masks (by document). Returns held-out scores of top hypotheses and best."""
    mods = sorted(set(m for m in mod if m))
    B = dict(A); B['mod'] = mod
    # inner splits within the selection documents
    Bs = {k: (v[sel] if isinstance(v, np.ndarray) else v) for k, v in B.items()}
    sp = L.make_splits(Bs, inner, rng)
    hyps = []
    for h in range(nh):
        K = rng.integers(2, 5)
        part = {m: 'c%d' % rng.integers(K) for m in mods}
        lab = np.array([part.get(m, '') if m else '' for m in Bs['mod']])
        hyps.append((L.transfer_score(Bs, sp, modlabel=lab)[0], part))
    hyps.sort(key=lambda x: -x[0])
    held = []
    for sc, part in hyps[:top]:
        lab = np.array([part.get(m, '') if m else '' for m in mod])
        held.append(L.transfer_score(B, [sel], modlabel=lab)[0])   # train = sel, test = held-out docs
    return float(np.mean(held)), hyps[0][0], [p for _, p in hyps[:top]]


def run_search(A, label, n_ctrl=8, nh=5000):
    ud = np.unique(A['doc']); seld = set(ud[rng.random(len(ud)) < 0.7])
    sel = np.array([d in seld for d in A['doc']])
    real, best, parts = search(A, A['mod'], sel, ~sel, nh=nh)
    ctrl = [search(A, L.shuffle_mods(A, rng), sel, ~sel, nh=nh)[0] for _ in range(n_ctrl)]
    # co-assignment of modifier pairs among top hypotheses (consensus classes)
    co = defaultdict(float)
    for p in parts:
        ks = sorted(p)
        for i in range(len(ks)):
            for j in range(i + 1, len(ks)):
                co[(ks[i], ks[j])] += (p[ks[i]] == p[ks[j]]) / len(parts)
    topco = sorted(co.items(), key=lambda x: -x[1])[:8]
    p = (1 + sum(c >= real for c in ctrl)) / (1 + len(ctrl))
    print(label, 'search held-out', round(real, 3), 'ctrl', [round(c, 3) for c in ctrl], 'P', round(p, 3),
          'top co-assigned', [(a, round(v, 2)) for a, v in topco], flush=True)
    return dict(real=real, best_sel=best, ctrl=ctrl, P=p, topco=[(list(a), v) for a, v in topco])


def planted(A, s=1.5, reps=4, nh=2000):
    """2 planted classes over the shared parts: class direction u_c * s added to their tokens."""
    sh = sorted(L.shared_mods(A)); out = []
    for r in range(reps):
        cls = {m: int(rng.integers(2)) for m in sh}
        U = rng.normal(size=(2, A['X'].shape[1])); U /= np.linalg.norm(U, axis=1, keepdims=True)
        X = A['X'].copy()
        for m, c in cls.items(): X[A['mod'] == m] += s * U[c]
        B = dict(A); B['X'] = X
        ud = np.unique(A['doc']); seld = set(ud[rng.random(len(ud)) < 0.7])
        sel = np.array([d in seld for d in A['doc']])
        real, _, parts = search(B, A['mod'], sel, ~sel, nh=nh)
        ctrl = [search(B, L.shuffle_mods(A, rng), sel, ~sel, nh=nh)[0] for _ in range(3)]
        # recovery: share of planted same-class pairs co-assigned in the top hypotheses vs diff-class
        agree = []
        for p in parts:
            same = [p.get(a) == p.get(b) for i, a in enumerate(sh) for b in sh[i + 1:] if cls[a] == cls[b] and a in p and b in p]
            diff = [p.get(a) == p.get(b) for i, a in enumerate(sh) for b in sh[i + 1:] if cls[a] != cls[b] and a in p and b in p]
            agree.append(np.mean(same) - np.mean(diff))
        out.append((round(real, 3), [round(c, 3) for c in ctrl], round(float(np.mean(agree)), 3)))
    print('planted s', s, out, flush=True)
    return out


if __name__ == '__main__':
    R = {}
    LAc = L.to_arrays(L.la_commodity_rows()); LB = L.to_arrays(L.lb_rows())
    G = {'sex': {'m', 'f', 'x'}, 'SI': {'SI'}, 'acro': {'TE', 'PA', 'KU', 'A', 'O', 'QE'}}
    R['ctx_LAc'] = run_context(LAc, doc_words_la(), {}, 'LA commodity')
    R['ctx_LB'] = run_context(LB, doc_words_lb(), G, 'LB')
    R['search_LAc'] = run_search(LAc, 'LA commodity')
    R['search_LB'] = run_search(LB, 'LB', n_ctrl=4, nh=1500)
    R['plant'] = planted(LAc)
    json.dump(R, open(os.path.join(L.CK, 'c2.json'), 'w'), indent=1, default=str)
