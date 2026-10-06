"""pe71 cycle 2b: is the M153 compound a CLOSING SLOT (where Ur III writes the responsible party)?

Seen in c2 context dumps: |M153+X| and |M153+M342| stand alone, unnumbered, on the last signed line of short tablets.
(a) Closing slot: the tablet's last line that carries signs, with no numerals on it (numeral-only total lines after it are
    allowed).  Share of each compound family's occurrences that sit in the closing slot; M153 vs every other component
    family of >= 15 occurrences, and vs random unions of compound types (2,000).
(b) Does a closing slot go with sealing?  Tablets with a closing slot: sealed share vs label permutations within volume x
    M157 header x has-M288 (5,000), counted in sealing units.  Same with the slot filled by an M153 compound vs by
    anything else.
(c) What fills the closing slot (value = the full sign string): V/n and top share at n = 50 vs Ur III personal names and
    titles (c2 control: names 0.77 / 0.08, titles 0.22 / 0.34).
(d) CONTROL Ur III at PE size (Umma, 1,581 tablets, 10 draws): relative line position of the sealer's legend name in the
    text of sealed tablets vs all other words of those tablets; share in the last third.
usage: pe71_c2b.py -> data/pe71_ckpt/c2b.json
"""
import json, os, collections
import numpy as np
from pe71_lib import load, comps, unit_codes, fam_stats, perm_index, CK
from pe70_common import get
from pe71_c2 import TITLES


def closing(r):
    L = r['lines']
    k = max((i for i, l in enumerate(L) if l['signs']), default=None)
    if k is None or L[k]['nums']:
        return None, None
    return k, tuple(L[k]['signs'])


def main():
    rng = np.random.default_rng(7122)
    R = load()
    res = {}
    # (a)
    occ = collections.defaultdict(lambda: [0, 0])  # family -> [closing, total]
    tocc = collections.defaultdict(lambda: [0, 0])
    for r in R:
        k, s = closing(r)
        for i, l in enumerate(r['lines']):
            for t in set(l['signs']):
                c = int(i == k)
                tocc[t][0] += c; tocc[t][1] += 1
                if t.startswith('|'):
                    for x in set(comps(t)):
                        occ[x][0] += c; occ[x][1] += 1
    fam = {x: v for x, v in occ.items() if v[1] >= 15}
    rate = {x: v[0] / v[1] for x, v in fam.items()}
    m = rate['M153']
    ctypes = [t for t in tocc if t.startswith('|') and 'M153' not in comps(t)]
    n0 = fam['M153'][1]
    ru = []
    for _ in range(2000):
        a = b = 0
        for t in rng.permutation(ctypes):
            if b >= n0:
                break
            a += tocc[t][0]; b += tocc[t][1]
        ru.append(a / b)
    ru = np.array(ru)
    allc = sum(v[0] for t, v in tocc.items()); allt = sum(v[1] for v in tocc.values())
    res['a'] = dict(M153=fam['M153'], rate=m, rank=1 + sum(v > m for v in rate.values()), n_fams=len(rate),
                    top=sorted(((x, round(v, 3), fam[x]) for x, v in rate.items()), key=lambda z: -z[1])[:8],
                    random_unions=float(ru.mean()), p_unions=float((ru >= m).mean()), all_signs=allc / allt,
                    types_M153={t: tocc[t] for t in tocc if 'M153' in comps(t)})
    print('a', res['a'], flush=True)
    # (b)
    sealed = np.array([r['sealed'] for r in R], int); units = unit_codes(R)
    st = np.array(['%s|%d|%d' % (r['vol'], r['hdr'] == 'M157', r['m288']) for r in R])
    P = perm_index(st, rng, 5000)
    cl = [closing(r) for r in R]
    has = [i for i, c in enumerate(cl) if c[0] is not None]
    hasm = [i for i in has if any('M153' in comps(t) for t in cl[i][1] if t.startswith('|'))]
    hasn = [i for i in has if i not in set(hasm)]
    res['b'] = {nm: fam_stats(f, sealed, units, P) for nm, f in (('closing_any', has), ('closing_M153', hasm), ('closing_other', hasn))}
    print('b', res['b'], flush=True)
    # (c)
    vals = [(i, ' '.join(cl[i][1])) for i in has]
    prof = []
    for _ in range(50):
        sub = [vals[j] for j in rng.choice(len(vals), 50, replace=False)]
        c = collections.Counter(v for _, v in sub)
        prof.append((len(c) / 50, c.most_common(1)[0][1] / 50))
    cc = collections.Counter(v for _, v in vals)
    res['c'] = dict(n=len(vals), v_per_n50=float(np.mean([p[0] for p in prof])), top50=float(np.mean([p[1] for p in prof])),
                    commonest=cc.most_common(15))
    print('c', res['c'], flush=True)
    # (d) Ur III control
    U = [u for u in get('ur3') if u['vol'] == 'Umma']
    dd = []
    for d in range(10):
        S = [U[i] for i in rng.choice(len(U), 1581, replace=False)]
        pn, po = [], []
        for s in S:
            if not s['sealed'] or not s['legend'] or len(s['lines']) < 3:
                continue
            names = {t for t in s['legend'] if t not in TITLES}
            n = len(s['lines'])
            for i, l in enumerate(s['lines']):
                for t in l:
                    (pn if t in names else po).append(i / (n - 1))
        pn, po = np.array(pn), np.array(po)
        dd.append(dict(n_name=len(pn), mean_name=float(pn.mean()), mean_other=float(po.mean()),
                       last3_name=float((pn > 2 / 3).mean()), last3_other=float((po > 2 / 3).mean())))
    res['d'] = dict(draws=dd, mean_name=float(np.mean([x['mean_name'] for x in dd])), mean_other=float(np.mean([x['mean_other'] for x in dd])),
                    last3_name=float(np.mean([x['last3_name'] for x in dd])), last3_other=float(np.mean([x['last3_other'] for x in dd])))
    print('d UR3', {k: v for k, v in res['d'].items() if k != 'draws'}, flush=True)
    json.dump(res, open(os.path.join(CK, 'c2b.json'), 'w'), indent=1, default=str)


if __name__ == '__main__':
    main()
