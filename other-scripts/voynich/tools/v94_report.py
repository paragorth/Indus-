"""summarise v94 screen / stage2 jsonl. usage: v94_report.py screen_cal.jsonl [stage2_cal.jsonl]"""
import sys, json, os
import numpy as np
from collections import defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
PARTNERS = {'cal_celsus_recut_coded': ['v89_celsus_lat'], 'cal_celsus_abr_recut_coded': ['v89_celsus_lat'],
            'cal_pliny_en_vs_la': ['v89_pliny_nh_lat']}


def zdec(Bs, i):
    o = np.delete(Bs, i); return (Bs[i] - o.mean()) / (o.std() + 1e-9)


def screen(f):
    R = defaultdict(list)
    for l in open(f):
        d = json.loads(l)
        if not d.get('skip'): R[d['target']].append(d)
    allz = []
    out = {}
    for t, rows in sorted(R.items()):
        rows.sort(key=lambda d: -d['B'])
        Bs = np.array([d['B'] for d in rows])
        P = PARTNERS.get(t, [])
        print('== %s (%d texts)  zB over all: mean %.2f sd %.2f, >=4: %d' % (t, len(rows), np.mean([d['zB'] for d in rows]), np.std([d['zB'] for d in rows]), sum(d['zB'] >= 4 for d in rows)))
        for i, d in enumerate(rows[:6]):
            print('  %d %-45s B %.3f zB %5.1f zDecoy %5.1f  %s %s off %.0f span %.0f' % (i + 1, d['src'][:45], d['B'], d['zB'], zdec(Bs, i), d['a']['sch'], d['a']['map'], d['a']['off'], d['a']['span']))
        for p in P:
            ix = [d['src'] for d in rows].index(p) if p in [d['src'] for d in rows] else None
            if ix is not None:
                d = rows[ix]
                print('  partner %s rank %d/%d B %.3f zB %.1f zDecoy %.1f' % (p, ix + 1, len(rows), d['B'], d['zB'], zdec(Bs, ix)))
        dec = [(d, zdec(Bs, i)) for i, d in enumerate(rows) if d['src'] not in P]
        print('  decoys passing zB>=4 & zDecoy>=3: %d of %d' % (sum(1 for d, z in dec if d['zB'] >= 4 and z >= 3), len(dec)))
        out[t] = rows
    return out


def stage2(f):
    R = defaultdict(list)
    for l in open(f):
        d = json.loads(l); R[d['target']].append(d)
    for t, rows in sorted(R.items()):
        rows.sort(key=lambda d: -d['B_rc'])
        print('== stage2 %s' % t)
        for d in rows:
            print('  %-45s B_s2 %.3f zB_s2 %5.1f | recut B %.3f zB %5.1f' % (d['src'][:45], d['B_s2'], d['zB_s2'], d['B_rc'], d['zB_rc']))


if __name__ == '__main__':
    screen(sys.argv[1])
    if len(sys.argv) > 2: stage2(sys.argv[2])
