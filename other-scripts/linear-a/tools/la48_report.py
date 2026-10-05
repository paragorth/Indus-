#!/usr/bin/env python3
"""LA-48 summaries of cycle results (c1, c2, c3)."""
import sys, os, json, collections
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from la48_common import CK


def load(fn):
    p = os.path.join(CK, fn)
    return [json.loads(l) for l in open(p)] if os.path.exists(p) else []


def c1():
    R = load('c1_results.jsonl')
    by = collections.defaultdict(lambda: collections.defaultdict(list))
    for x in R:
        by[x['name']][x['kind']].append(x)
    print('dataset | REAL score K2 K3 PAIR AGG | QSHUF mean(sd) score; AGG mean; P(score), P(AGG) | POIS score mean | rand-label mean')
    for name in by:
        r = by[name]['REAL'][0] if by[name]['REAL'] else None
        q = by[name]['QSHUF']; p = by[name]['POIS']
        if not r:
            continue
        qs = np.array([x['score'] for x in q]); qa = np.array([x.get('AGG', 0) for x in q])
        Ps = (1 + (qs >= r['score']).sum()) / (1 + len(qs)) if len(qs) else float('nan')
        Pa = (1 + (qa >= r.get('AGG', 0)).sum()) / (1 + len(qa)) if len(qa) else float('nan')
        print('%s | %.0f %d %d %d %d | %.1f(%.1f) n%d; AGG %.2f; P %.3f, P_AGG %.3f | %.1f | %.2f' % (
            name, r['score'], r['K2'], r['K3'], r.get('PAIR', -1), r.get('AGG', -1), qs.mean() if len(qs) else -1,
            qs.std() if len(qs) else -1, len(qs), qa.mean() if len(qa) else -1, Ps, Pa,
            np.mean([x['score'] for x in p]) if p else -1, r.get('rand_mean', -1)))


def c2():
    R = load('c2_results.jsonl')
    for x in sorted(R, key=lambda x: (x['name'], x['obj'], x['kind'], x['rep'])):
        print(json.dumps({k: v for k, v in x.items() if k not in ('d', 'ids')}))


def c3():
    R = load('c3_results.jsonl')
    by = collections.defaultdict(lambda: collections.defaultdict(list))
    for x in R:
        by[x['name']][x['kind']].append(x)
    print('dataset | elig phys copy beyond randlab_phys randlab_beyond | QSHUF phys / copy / beyond means')
    for name in by:
        r = by[name]['REAL'][0] if by[name]['REAL'] else None
        if not r:
            continue
        q = by[name]['QSHUF']
        f = lambda k: np.mean([x[k] / max(1, x['elig']) for x in q]) * r['elig'] if q else -1
        print('%s | %d %d %d %d %.2f %.2f | %.2f / %.2f / %.2f (n%d)' % (
            name, r['elig'], r['phys'], r['copy'], r['beyond'], r['rand_phys'], r['rand_beyond'],
            f('phys'), f('copy'), f('beyond'), len(q)))


if __name__ == '__main__':
    for a in sys.argv[1:] or ['c1', 'c2', 'c3']:
        print('==', a); globals()[a]()
