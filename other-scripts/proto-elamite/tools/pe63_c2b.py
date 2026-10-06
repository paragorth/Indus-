#!/usr/bin/env python3
"""pe63 cycle 2b: COUPLED SLOTS. Inside each dossier, for every pair of numeral slots (i, j): does the
quantity in slot j move with slot i at one fixed rate across the copies (vj = r * vi)?
Values: PE lines read under both value maps (capacity CAP, pe59 ladder; count CNT, N01 1 / N14 10 / N34 60 /
N45 600, half-fractions); the statistic is the best modal-rate share over the 4 map pairs. Ur III values as
written. Null: vj permuted among the members (2,000x), same max over map pairs. Controls: Ur III Drehem /
Umma dossiers; planted rate (a numeral slot rewritten as 48 x the step slot in each planted dossier) and a
planted non-rate (same slot, random values)."""
import os, json, random, re
from fractions import Fraction as Fr
from collections import Counter
import numpy as np
import pe63_common as C
import pe63_c2 as C2

CAP = {'N39C': 1, 'N30D': 2, 'N30C': 4, 'N24': 12, 'N39B': 24, 'N01': 120, 'N14': 720, 'N45': 7200, 'N34': 21600}
CNT = {'N01': 1, 'N14': 10, 'N34': 60, 'N45': 600, 'N48': 3600, 'N08': Fr(1, 2), 'N8B': Fr(1, 2), 'N08A': Fr(1, 2)}


def parse(raw, m):
    t = Fr(0)
    for n, c in re.findall(r'(\d+)\((N[0-9A-Z]+)\)', raw or ''):
        if c not in m:
            return None
        t += int(n) * Fr(m[c])
    return t if t > 0 else None


def vals(x, corp):
    if corp == 'PE':
        return [parse(x.get('raw'), CAP), parse(x.get('raw'), CNT)]
    v = x.get('v')
    return [Fr(v).limit_denominator(1000) if v else None]


def modal(pairs):
    """pairs: list of (vi list, vj list) per member -> best (share count, n, rate, maps)"""
    best = (0, 0, None, None)
    nm_i = len(pairs[0][0]); nm_j = len(pairs[0][1])
    for a in range(nm_i):
        for b in range(nm_j):
            r = [p[1][b] / p[0][a] for p in pairs if p[0][a] and p[1][b]]
            if len(r) < 3:
                continue
            c = Counter(r).most_common(1)[0]
            if c[1] > best[0] or (c[1] == best[0] and len(r) < best[1]):
                best = (c[1], len(r), c[0], (a, b))
    return best


def scan(dossiers, corp, rng, R=2000):
    out = []
    for a in dossiers:
        num = [o for o in a['slots'] if o['numeral']]
        for i in num:
            for j in num:
                if i is j:
                    continue
                mem = [k for k in i['members'] if k in j['members']]
                P = [(vals(i['members'][k], corp), vals(j['members'][k], corp)) for k in mem]
                if len(P) < 3:
                    continue
                k, n, r, mp = modal(P)
                if k < 3 or r == 1:
                    continue
                # do the variants actually differ? (a rate needs vi to vary)
                vi = {tuple(p[0]) for p in P}
                if len(vi) < 2:
                    continue
                hit = 0
                for _ in range(R):
                    vj = [p[1] for p in P]; rng.shuffle(vj)
                    kk, nn, rr, _ = modal([(p[0], q) for p, q in zip(P, vj)])
                    if kk >= k and rr != 1:
                        hit += 1
                out.append({'ref': a['ref'], 'i': i['slot'], 'j': j['slot'], 'core_i': i['core'], 'core_j': j['core'],
                            'k': k, 'n': n, 'rate': str(r), 'maps': mp, 'p': (hit + 1) / (R + 1),
                            'mem': mem})
    return out


def main():
    rng = random.Random(6322)
    res = json.load(open(os.path.join(C.CK, 'c2_res.json')))
    out = {}
    for corp in ['PE', 'DR', 'UM']:
        sc = scan(res[corp]['real'], corp, rng)
        ntest = len(sc)
        sig = [x for x in sc if x['p'] * ntest <= 0.05]
        sig5 = [x for x in sc if x['p'] <= 0.01]
        out[corp] = {'tests': ntest, 'bonf': len(sig), 'p01': len(sig5), 'exp_p01': 0.01 * ntest, 'hits': sc}
        print(corp, 'tests', ntest, 'Bonferroni', len(sig), 'p<=0.01', len(sig5), 'expected', round(0.01 * ntest, 2))
        for x in sorted(sc, key=lambda x: x['p'])[:12]:
            print('   ', x['ref'], x['i'], '->', x['j'], ' '.join(x['core_i']), '->', ' '.join(x['core_j']),
                  'rate', x['rate'], 'maps', x['maps'], '%d/%d' % (x['k'], x['n']), 'p %.4f' % x['p'])
    # planted rate / non-rate on planted dossiers
    pl = []
    for s in (1, 2, 3, 4, 5):
        D, tr = C2.dossiers_for('PE', 'plant', s)
        ids = set(tr['ids'])
        d = max(D, key=lambda d: len(ids & {t['id'] for t in d}))
        d = [json.loads(json.dumps(t)) for t in d if t['id'] in ids]
        free = [i for i, l in enumerate(d[0]['lines']) if l['sys'] is not None and i not in (tr['A'], tr['B'], tr['C'])]
        if not free:
            continue
        for mode in ('rate', 'none'):
            dd = json.loads(json.dumps(d))
            for t in dd:
                vc = 2 ** int(t['id'][-2:])
                n = 48 * vc if mode == 'rate' else rng.choice([3, 7, 20, 50, 130, 300])
                t['lines'][free[0]]['raw'] = '%d(N01)' % n
                t['lines'][tr['C']]['raw'] = '%d(N01)' % vc
                t['lines'][free[0]]['sys'] = 'SDB'; t['lines'][tr['C']]['sys'] = 'SDB'
                t['lines'][free[0]]['v'] = float(n); t['lines'][tr['C']]['v'] = float(vc)
            a = C2.analyse(dd)
            sc = scan([a], 'PE', rng, R=500)
            hit = [x for x in sc if {x['i'], x['j']} == {tr['C'], free[0]}]
            pl.append({'seed': s, 'mode': mode, 'found': bool(hit and min(h['p'] for h in hit) <= 0.01),
                       'rate': hit[0]['rate'] if hit else None,
                       'false_other': sum(1 for x in sc if x['p'] <= 0.01 and {x['i'], x['j']} != {tr['C'], free[0]})})
    out['plant'] = pl
    print('plant', pl)
    json.dump(out, open(os.path.join(C.CK, 'c2b_res.json'), 'w'))


if __name__ == '__main__':
    main()
