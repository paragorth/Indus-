#!/usr/bin/env python3
"""LA-56 report: summarises c1 (rule search), c2 (minimal explanations + blame) and c3 (re-cut) checkpoints."""
import json, glob, os, sys
import numpy as np
import la56_common as C
import la56_c2 as C2
import la56_c1 as C1
import random

CK = C.CK


def c1():
    rows = {}
    for f in sorted(glob.glob(os.path.join(CK, 'c1_*_*_*.json'))):
        d = json.load(open(f))
        rows.setdefault((d['corpus'], d['job']), []).append(d)
    print('== c1: best full-corpus gain / mean held-out gain of best rule / # rules replicated in >= half the splits')
    for (c, j), L in sorted(rows.items()):
        b = [d['best_full'] for d in L]; h = [d['held_mean'] for d in L]; r = [d['n_rep'] for d in L]
        pr = [d.get('plant_rank') for d in L if d.get('plant_rank')]
        prs = [d.get('plant_rank_singles') for d in L if d.get('plant_rank_singles')]
        prep = [d.get('plant_rep') for d in L if 'plant_rep' in d]
        print('%-4s %-6s n=%2d best %5.2f [%5.2f-%5.2f]  held %5.2f [%5.2f-%5.2f]  nrep %5.1f%s' % (
            c, j, len(L), np.mean(b), min(b), max(b), np.mean(h), min(h), max(h), np.mean(r),
            ('  plant rank %s (singles %s) rep %s' % (pr, prs, prep)) if pr else ''))
    for c in ('LA', 'UR3', 'LB'):
        if (c, 'real') in rows:
            d = rows[(c, 'real')][0]
            print('-- %s real top rules (gain, splits replicated, sections touched)' % c)
            for t in d['top'][:12]:
                print('   %-48s %5.2f rep %2d %s' % (t[0], t[1], t[2], [d['secs'][k] for k in t[3]][:4]))


def c2():
    rows = {}
    cache = {}
    for f in sorted(glob.glob(os.path.join(CK, 'c2_*_*_*.json'))):
        d = json.load(open(f))
        key = (d['corpus'], d['job'])
        # rebuild the corpus to recompute enrichment with the corrected permutation groups
        secs = C1.load(d['corpus']) if d['corpus'] not in cache else cache[d['corpus']]
        cache[d['corpus']] = secs
        S = C1.make(secs, d['job'], random.Random(d['seed'] * 7919 + sum(map(ord, d['job']))))[0] if d['job'] != 'real' else secs
        e = C2.enrich(S, d['res'], nperm=1000)
        d['enrich2'] = e
        rows.setdefault(key, []).append(d)
    print('== c2: share of non-closing totals explained with <=1 / <=2 / <=3 changes; best family-wise blame')
    for (c, j), L in sorted(rows.items()):
        sh = []
        for d in L:
            cs = [r['cost'] for r in d['res']]; n = len(cs)
            sh.append((sum(x == 1 for x in cs) / n, sum(x is not None and x <= 2 for x in cs) / n,
                       sum(x is not None for x in cs) / n, n))
        sh = np.array(sh)
        tops = [d['enrich2']['top'][0] if d['enrich2']['top'] else None for d in L]
        minp = [t[6] for t in tops if t]
        plant = ''
        if L[0]['info']:
            hits = []
            for d in L:
                f = d['info']['planted']; op = 'drop' if d['info']['op'] == 'EXCL' else 'neg'
                rk = [i for i, t in enumerate(d['enrich2']['top']) if t[0] == f and t[1] == op]
                pf = [t[6] for t in d['enrich2']['top'] if t[0] == f and t[1] == op]
                hits.append((f, rk[0] + 1 if rk else None, pf[0] if pf else None))
            plant = ' planted: ' + str(hits)
        print('%-4s %-6s n=%d bad %.0f  <=1 %.2f <=2 %.2f <=3 %.2f  min fw-p %.3f%s' % (
            c, j, len(L), sh[:, 3].mean(), sh[:, 0].mean(), sh[:, 1].mean(), sh[:, 2].mean(),
            min(minp) if minp else 1, plant))
        if j == 'real':
            for t in L[0]['enrich2']['top'][:8]: print('     ', t)


def c3():
    rows = {}
    for f in sorted(glob.glob(os.path.join(CK, 'c3_LA_*_*.json'))):
        d = json.load(open(f)); rows.setdefault(d['job'], []).append(d)
    print('== c3: totals re-cut by a different contiguous run (one skip allowed)')
    for j, L in sorted(rows.items()):
        st = {k: np.mean([d['stats'][k] for d in L]) for k in L[0]['stats']}
        print('%-5s n=%d ' % (j, len(L)) + ' '.join('%s %.1f' % kv for kv in st.items()))
    if 'real' in rows:
        for r in rows['real'][0]['res']:
            if r['p_default'] < 0.5:
                print('   ', r['tab'], r['side'], r['kind'], r['total'], 'cost', r.get('cost'), 'best', r.get('best'),
                      'x-total' if r.get('crosses_total') else '', 'x-side' if r.get('crosses_side') else '',
                      'after' if r.get('after_total') else '', r.get('edge_words'), 'skip', r.get('skipped_word'))


if __name__ == '__main__':
    for a in sys.argv[1:]:
        {'c1': c1, 'c2': c2, 'c3': c3}[a]()
