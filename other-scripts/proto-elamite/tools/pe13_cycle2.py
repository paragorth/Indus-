"""pe13 cycle 2: is the cycle-1 PE kernel memory, or layout?  And which signs carry it.

(A) Confound-stripped kernel fits on PE (same machinery as cycle 1, NK random kernels +
    expw / free MLE), each against 10 within-tablet line shuffles of the same variant:
      PE_ENT   : entries only (header and other non-numeral lines removed)
      PE_MID   : entries only, targets and history = middles (class sign removed)
      PE_CLS   : entries only, class signs only (lines without one dropped)
      PE_OBV   : entries only, obverse only (no surface factor to hide behind)
(B) Per-sign gap profile (entries only): for every sign on >= 8 tablets, observed pairs of
    sign-bearing lines at gap 1, gap 2-3 and gap >= 4 vs the exact expectation under random
    placement of the same lines (k-subset of L positions), variance from 400 shuffles.
    z_near (gap <= 3, excess = priming or runs), z_d1 (gap 1; deficit = refractory).
    False-positive control: 20 within-tablet-shuffled corpora analysed as if observed.
    Held-out: signs classified on tablet half A, re-tested on half B.
(C) Same per-sign analysis on the planted PRIME / TOPIC corpora (must separate).
usage: python3 pe13_cycle2.py [workers]
"""
import json, math, os, random, sys
import numpy as np
from collections import Counter, defaultdict
from multiprocessing import Pool
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pe13_common as C  # noqa
import pe13_cycle1 as C1  # noqa

C1.OUT = os.path.join(C.CK, 'c2')
os.makedirs(C1.OUT, exist_ok=True)


def variant(pe, kind):
    out = []
    for t in pe:
        L = [dict(l) for l in t['lines'] if l['ent']]
        if kind == 'MID':
            for l in L:
                l['toks'] = list(l['mid'])
        elif kind == 'CLS':
            L = [dict(l, toks=[l['cls']]) for l in L if l['cls']]
        elif kind == 'OBV':
            L = [l for l in L if l['surf'] == 0]
        L = [l for l in L if l['toks']]
        if len(L) >= 3:
            out.append({'id': t['id'], 'lines': L})
    return out


def gap_counts(tabs, signs):
    """per sign: observed pairs at gap 1, 2-3, >=4 and analytic expectations."""
    obs = {s: np.zeros(3) for s in signs}
    exp = {s: np.zeros(3) for s in signs}
    ntab = Counter()
    for t in tabs:
        L = len(t['lines'])
        pos = defaultdict(list)
        for i, l in enumerate(t['lines']):
            for s in set(l['toks']):
                if s in obs:
                    pos[s].append(i)
        for s, P in pos.items():
            k = len(P)
            ntab[s] += 1
            if k < 2:
                continue
            for a in range(k):
                for b in range(a + 1, k):
                    d = P[b] - P[a]
                    obs[s][0 if d == 1 else (1 if d <= 3 else 2)] += 1
            f = k * (k - 1) / (L * (L - 1))
            e1 = (L - 1) * f
            e23 = (max(L - 2, 0) + max(L - 3, 0)) * f
            tot = k * (k - 1) / 2
            exp[s] += [e1, e23, tot - e1 - e23]
    return obs, exp, ntab


def per_sign(tabs, signs, nshuf=400, seed=0):
    obs, exp, ntab = gap_counts(tabs, signs)
    rng = random.Random(seed)
    sims = {s: [] for s in signs}
    for r in range(nshuf):
        sh = C.shuffle_lines(tabs, rng)
        o, _, _ = gap_counts(sh, signs)
        for s in signs:
            sims[s].append(o[s])
    res = {}
    for s in signs:
        S = np.array(sims[s])
        sd = S.std(0) + 1e-9
        near_o = obs[s][0] + obs[s][1]
        near_s = S[:, 0] + S[:, 1]
        res[s] = {'ntab': ntab[s], 'obs': obs[s].tolist(), 'exp': exp[s].tolist(),
                  'z_d1': float((obs[s][0] - exp[s][0]) / sd[0]),
                  'z_near': float((near_o - exp[s][0] - exp[s][1]) / (near_s.std() + 1e-9)),
                  'r_d1': float((obs[s][0] + 0.5) / (exp[s][0] + 0.5)),
                  'r_far': float((obs[s][2] + 0.5) / (exp[s][2] + 0.5))}
    return res


def job(j):
    name, tabs = j
    if name.startswith('SIGN'):
        fn = os.path.join(C1.OUT, name + '.json')
        if os.path.exists(fn):
            return json.load(open(fn))
        signs, tb, ns, sd = tabs
        r = per_sign(tb, signs, ns, sd)
        json.dump(r, open(fn, 'w'))
        print(name, 'done', flush=True)
        return r
    return C1.analyse(name, tabs, 'ent', nk=800)


if __name__ == '__main__':
    nw = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    CO = C.corpora()
    pe = CO['PE']
    J = []
    for kind in ('ENT', 'MID', 'CLS', 'OBV'):
        v = variant(pe, kind)
        J.append(('PE_' + kind, v))
        for s in range(10):
            J.append(('PE_%s_shuf%d' % (kind, s), C.shuffle_lines(v, random.Random(200 + s))))
    ent = variant(pe, 'ENT')
    tcount = Counter(s for t in ent for s in set(x for l in t['lines'] for x in l['toks']))
    signs = sorted(s for s, n in tcount.items() if n >= 8)
    print('signs', len(signs), flush=True)
    J.append(('SIGN_PE', (signs, ent, 400, 1)))
    for s in range(20):
        J.append(('SIGN_SHUF%02d' % s, (signs, C.shuffle_lines(ent, random.Random(900 + s)), 200, 2)))
    ids = sorted(t['id'] for t in ent)
    rr = random.Random(77)
    A = set(rr.sample(ids, len(ids) // 2))
    J.append(('SIGN_HALFA', (signs, [t for t in ent if t['id'] in A], 400, 3)))
    J.append(('SIGN_HALFB', (signs, [t for t in ent if t['id'] not in A], 400, 4)))
    uni = Counter(x for t in ent for l in t['lines'] for x in l['toks'])
    for k in ('PRIME', 'TOPIC', 'MIX'):
        pl = C.gen_planted(ent, k, 31, uni)
        J.append(('SIGN_PL_' + k, (signs, pl, 200, 5)))
    with Pool(nw) as p:
        p.map(job, J, chunksize=1)
