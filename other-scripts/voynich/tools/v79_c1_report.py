"""v79 cycle 1 report: rows for loops/v79_cycle1.txt from data/v79_ckpt/c1_*.pkl."""
import sys, os, glob, json
import numpy as np
import v79_lib as L

ORDER = ['ZL3b', 'IT2a', 'C_DANTE', 'C_DOM', 'C_GOLD', 'C_ENTRY', 'N_ISI', 'G_SELFCIT', 'G_SC10', 'G_MK2', 'G_JUNC', 'G_STACK']
R = {n: L.pload('c1_%s.pkl' % n) for n in ORDER}
R = {n: r for n, r in R.items() if r is not None}
UNITS = ['line/para', 'line/page', 'line/book', 'entry/page', 'entry/book']

# true successor pairs of the planted channels (glyph -> glyph)
import random
g = L.OPAQUE[:12]; r2 = random.Random(11); r2.shuffle(g); DIG = g[:10]
TRUE = {'C_DANTE': {(DIG[v], DIG[(v + 1) % 10]) for v in range(10)},
        'C_DOM': {(DIG[v], DIG[(v + 1) % 7]) for v in range(7)},
        'C_GOLD': {(DIG[gn % 10], DIG[((gn + 7) % 19 + 1) % 10]) for gn in range(1, 20)},
        'C_ENTRY': {(DIG[v], DIG[(v + 1) % 10]) for v in range(10)}}


def rule_pairs(best, alpha):
    k, s, D = best
    return {(alpha[D[i]], alpha[D[(i + s) % k]]) for i in range(k)}


def summ(n, r, u):
    x = r['real'][u]
    out = dict(z=x['held_z'], top=x['top_held_z'], d=x['held_d'])
    for nk in ('shuf', 'mk1', 'rowperm'):
        v = np.array([y[u]['held_z'] for y in r['nulls'][nk]])
        out[nk] = (v.mean(), v.std())
    al = r['alpha_e'] if u.startswith('entry') else r['alpha']
    bp = rule_pairs(x['disc0']['best'], al)
    if n in TRUE:
        out['rec'] = len(bp & TRUE[n]) / len(bp)
    out['rule'] = ' '.join('%s>%s' % p for p in sorted(bp))
    return out


lines = []
for n, r in R.items():
    for u in UNITS:
        s = summ(n, r, u)
        lines.append((n, u, s))
        print('%-10s %-10s held z %5.1f (top20 %5.1f, excess %.3f) | shuf %5.1f+-%.1f mk1 %5.1f+-%.1f rowperm %5.1f+-%.1f | rec %s | %s' % (
            n, u, s['z'], s['top'], s['d'], *s['shuf'], *s['mk1'], *s['rowperm'], ('%.2f' % s['rec']) if 'rec' in s else '-', s['rule'][:80]))
    p = r['per']
    print('   periodic (bits/line beyond drift): ' + '; '.join('%s p%d %+.4f (drift+mod p%d %+.4f)' % (k, v['best_p'], v['best_mod_minus_drift'], v['best_p2'], v['drift_plus_mod']) for k, v in p.items()))
json.dump({'%s|%s' % (n, u): {k: (list(v) if isinstance(v, tuple) else v) for k, v in s.items()} for n, u, s in lines},
          open(os.path.join(L.CK, 'c1_summary.json'), 'w'), default=float, indent=0)
