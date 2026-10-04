"""pe12 cycle 3: THE AUDITOR'S TICK AND THE SLIP SPECTRUM (tablet-level check marks).

3a. If scribes verified accounts, a VERIFIED tablet (reverse total == sum of obverse
    entries) might carry a mark that an unverified one lacks.  For every PE tablet with
    >= 2 obverse entries and one reverse total, all in one system and fully legible:
    balanced flag B.  Candidate marks: every sign present on >= 5 such tablets (anywhere),
    the total-line first sign, 'total line carries signs', and 3,000 random sign-pairs (OR).
    Statistic: max |z| of the log-odds association with B.  Null: B permuted among tablets
    of the same stratum (system x entry-count band x damaged-line band), 2,000x  (FWER).
    Check-digit on the total itself: total-line sign vs total mod k, sum mod k,
    (total - sum) mod k, k = 2..12; lookup-table leave-one-out gain; null = total sign
    permuted within stratum.
    Planted control: a sign TICK added to 70% of balanced and 5% of unbalanced tablets.
3b. Slip spectrum.  If the mismatches are copying slips that a check mark would catch,
    |total - sum| should equal ONE numeral sign's value more often than magnitude-matched
    integers do.  Done under each counting value set (D3, DEC, SEX, SEXr) and capacity set
    (NOT, PCS): the right value set should give the most single-sign slips.
    Planted control: balanced tablets with one numeral sign deleted from a random entry.
"""
import os, sys, json, math
import numpy as np
from collections import Counter, defaultdict
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import load, base, is_sign  # noqa
from pe5_common import pe_system, pe_value, VSETS, CSETS  # noqa
from pe12_common import PEDATA  # noqa

LOG = open(os.path.join(PEDATA, 'pe12_cycle3.log'), 'a')
rng = np.random.default_rng(31)


def log(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True); LOG.write(s + '\n'); LOG.flush()


def tot_tablets(T, vs='D3', cs='NOT'):
    out = []
    for t in T:
        L = t['lines']
        num = [(i, l) for i, l in enumerate(L) if l['numerals']]
        obv = [l for i, l in num if l['surface'] == 'obverse']
        off = [l for i, l in num if l['surface'] == 'reverse']
        if not (len(off) == 1 and len(obv) >= 2):
            continue
        sy = pe_system(off[0]['numerals'])
        if not sy or any(pe_system(l['numerals']) != sy for l in obv):
            continue
        if any(l['lacuna'] for l in obv + off):
            continue
        tv = pe_value(off[0]['numerals'], sy, VSETS[vs], CSETS[cs])
        vals = [pe_value(l['numerals'], sy, VSETS[vs], CSETS[cs]) for l in obv]
        if tv is None or any(v is None for v in vals):
            continue
        allsg = set(base(x) for l in L for x in l['signs'] if is_sign(x))
        ts = [base(x) for x in off[0]['signs'] if is_sign(x)]
        dam = sum(1 for l in L if l['damaged'])
        out.append(dict(id=t['id'], sys=sy, tot=tv, sum=sum(vals), n=len(obv), dam=dam,
                        signs=allsg, tsign=ts[0] if ts else 'NONE', obv=obv, totnum=off[0]['numerals'],
                        hdr=([base(x) for x in L[0]['signs'] if is_sign(x)] or ['none'])[0]
                        if L and not L[0]['numerals'] else 'none'))
    return out


def strata(R):
    return [(r['sys'], min(r['n'] // 3, 4), min(r['dam'] // 2, 3)) for r in R]


def perm_within(x, groups, rng):
    x2 = x.copy()
    idx = defaultdict(list)
    for i, g in enumerate(groups):
        idx[g].append(i)
    for L in idx.values():
        L = np.array(L)
        x2[L] = x[rng.permutation(L)]
    return x2


def assoc_z(M, b):
    """M: H x n binary marks; b: n binary.  log-odds z per hypothesis (Haldane)."""
    a = (M * b).sum(1) + .5
    bb = (M * (1 - b)).sum(1) + .5
    c = ((1 - M) * b).sum(1) + .5
    d = ((1 - M) * (1 - b)).sum(1) + .5
    lo = np.log(a * d / (bb * c))
    return lo / np.sqrt(1 / a + 1 / bb + 1 / c + 1 / d)


def tick_test(R, label, nperm=2000, plant=False):
    b = np.array([int(abs(r['tot'] - r['sum']) < 1e-6) for r in R])
    sets = [set(r['signs']) for r in R]
    if plant:
        for s, bi in zip(sets, b):
            if rng.random() < (0.7 if bi else 0.05):
                s.add('TICK')
    cnt = Counter(x for s in sets for x in s)
    cand = [x for x, c in cnt.items() if c >= 5]
    names, rows = [], []
    for x in cand:
        names.append('has:' + x); rows.append([int(x in s) for s in sets])
    for x in set(r['tsign'] for r in R):
        names.append('tsign:' + x); rows.append([int(r['tsign'] == x) for r in R])
    for _ in range(3000):
        p, q = rng.choice(len(cand), 2, replace=False)
        names.append('or:%s|%s' % (cand[p], cand[q]))
        rows.append([int(cand[p] in s or cand[q] in s) for s in sets])
    M = np.array(rows)
    keep = (M.sum(1) >= 5) & (M.sum(1) <= len(R) - 5)
    M = M[keep]; names = [n for n, k in zip(names, keep) if k]
    z = assoc_z(M, b)
    G = strata(R)
    mx = []
    for _ in range(nperm):
        mx.append(np.abs(assoc_z(M, perm_within(b, G, rng))).max())
    mx = np.array(mx)
    o = np.argsort(-np.abs(z))[:6]
    top = [(names[i], round(float(z[i]), 2), float((1 + (mx >= abs(z[i])).sum()) / (1 + nperm))) for i in o]
    log('3a %s: %d tablets, %d balanced, %d hypotheses; top %s' % (label, len(R), b.sum(), len(names), top))
    return dict(n=len(R), balanced=int(b.sum()), H=len(names), top=top, null_max_mean=float(mx.mean()))


def total_checkdigit(R, nperm=500, label=''):
    y0 = [r['tsign'] for r in R]
    c = Counter(y0)
    y = np.array([sorted(c).index(v) for v in y0])
    feats = {}
    for k in range(2, 13):
        feats['tot%%%d' % k] = np.array([int(r['tot']) % k for r in R])
        feats['sum%%%d' % k] = np.array([int(r['sum']) % k for r in R])
        feats['diff%%%d' % k] = np.array([int(round(r['tot'] - r['sum'])) % k for r in R])
    feats['balanced'] = np.array([int(abs(r['tot'] - r['sum']) < 1e-6) for r in R])
    feats['totband'] = np.array([int(math.log10(max(r['tot'], 1))) for r in R])
    G = strata(R)
    gsys = [r['sys'] for r in R]

    def loo(x, yy):
        K = yy.max() + 1
        tab = np.zeros((x.max() + 1, K)); np.add.at(tab, (x, yy), 1)
        base_ = np.bincount(yy, minlength=K).astype(float)
        n = len(yy)
        pb = (base_[yy] - 1 + .5) / (n - 1 + .5 * K)
        nx = tab.sum(1)
        pm = (tab[x, yy] - 1 + 2 * pb) / (nx[x] - 1 + 2)
        return float(np.mean(np.log2(pm) - np.log2(pb)))

    real = {f: loo(x, y) for f, x in feats.items()}
    best = max(real, key=real.get)
    mx = []
    for _ in range(nperm):
        yp = perm_within(y, G, rng)
        mx.append(max(loo(x, yp) for x in feats.values()))
    mx = np.array(mx)
    p = float((1 + (mx >= real[best]).sum()) / (1 + nperm))
    top = sorted(real.items(), key=lambda kv: -kv[1])[:5]
    log('3a-check %s: total-line sign vs total/sum/diff mod k: best %s %+.4f bits (FWER p %.3f, null max mean %+.4f); top %s'
        % (label, best, real[best], p, mx.mean(), [(a, round(b, 4)) for a, b in top]))
    return dict(best=best, gain=real[best], p=p, top=top)


UNITS = {'D3': [1, 10, 100, 300, 3000], 'DEC': [1, 10, 100, 1000, 10000],
         'SEX': [1, 10, 60, 600, 3600], 'SEXr': [1, 10, 60, 600, 3600],
         'NOT': [1, 2, 4, 12, 24, 120, 720, 7200], 'PCS': [1, 3, 6, 30, 150, 900, 9000]}


def slip(R, vs, cs):
    """hits = |diff| equal to one numeral sign value; expected = magnitude-matched density."""
    hit, exp, n = 0, 0.0, 0
    for r in R:
        d = abs(r['tot'] - r['sum'])
        if d < 1e-6 or d != int(d):
            continue
        d = int(d)
        U = set(UNITS[vs] if r['sys'] == 'S' else UNITS[cs])
        n += 1
        hit += d in U
        lo, hi = max(1, int(d / 1.5)), int(d * 1.5) + 1
        exp += sum(1 for u in U if lo <= u <= hi) / (hi - lo + 1)
    return n, hit, exp


def slip_test(T, label, plant=False):
    drops = {}
    if plant:   # one numeral sign deleted from a random entry of tablets balanced under D3/NOT
        for r in tot_tablets(T, 'D3', 'NOT'):
            if abs(r['tot'] - r['sum']) > 1e-6:
                continue
            li = int(rng.integers(len(r['obv'])))
            ni = int(rng.integers(len(r['obv'][li]['numerals'])))
            drops[r['id']] = (li, ni)
    out = {}
    for vs, cs in [('D3', 'NOT'), ('DEC', 'NOT'), ('SEX', 'NOT'), ('SEXr', 'NOT'), ('D3', 'PCS')]:
        R = tot_tablets(T, vs, cs)
        if plant:
            R2 = []
            for r in R:
                if r['id'] not in drops:
                    continue
                li, ni = drops[r['id']]
                n, c = r['obv'][li]['numerals'][ni]
                tab = VSETS[vs] if r['sys'] == 'S' else CSETS[cs]
                r = dict(r); r['sum'] = r['sum'] - tab.get(c, 0)
                R2.append(r)
            R = R2
        n, h, e = slip(R, vs, cs)
        z = (h - e) / math.sqrt(max(e, 1e-9))
        out[vs + '/' + cs] = dict(n=n, hits=h, expected=round(e, 2), z=round(z, 2))
    log('3b %s slip spectrum: %s' % (label, out))
    return out


if __name__ == '__main__':
    T = load()
    res = {}
    R = tot_tablets(T)
    res['tick_plant'] = tick_test(R, 'PLANTED', plant=True)
    res['tick_real'] = tick_test(R, 'REAL')
    res['check_real'] = total_checkdigit(R, label='REAL')
    # planted check digit on the total: total sign = f(total mod 3) on 60% of tablets
    Rp = [dict(r, tsign=('CK%d' % (int(r['tot']) % 3)) if rng.random() < .6 else r['tsign']) for r in R]
    res['check_plant'] = total_checkdigit(Rp, label='PLANTED tot%3')
    res['slip_plant'] = slip_test(T, 'PLANTED one-sign deletions', plant=True)
    res['slip_real'] = slip_test(T, 'REAL')
    json.dump(res, open(os.path.join(PEDATA, 'pe12_cycle3.json'), 'w'), indent=1, default=str)
    log('done')
