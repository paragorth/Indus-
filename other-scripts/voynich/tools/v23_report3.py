"""v23 cycle-3 table: replicated random arrow probes by level and probe kind, real vs random-direction null."""
import os, sys, json, glob
from collections import defaultdict
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v23_lib as L

R = defaultdict(dict)
for f in glob.glob(os.path.join(L.CK, 'c3_*.json')):
    d = json.load(open(f)); R[d['name']][d['mode']] = d


def surv(probes, kind=None, lag=None):
    n = s = 0
    for kd, k, a, b in probes:
        if kind and kd != kind: continue
        if lag and k != lag: continue
        n += 1
        if abs(a) >= 3 and np.sign(a) == np.sign(b) and abs(b) >= 2: s += 1
    return s, n


names = [n for n in ['ZL', 'IT', 'LA', 'ITA', 'DE', 'CS', 'HE', 'PL_REV', 'MkG_1', 'MkW_1', 'RvG_0', 'RvW_0'] if n in R]
print('replicated probes / probes  (real | max over random-page-direction draws)')
for lev in ('glyph', 'word', 'line'):
    kinds = ['gly-partition'] if lev == 'glyph' else ['random', 'first', 'last', 'len', 'freq']
    print('\n' + lev + '  ' + '  '.join('%14s' % k for k in kinds) + '   corr(zA,zB)')
    for n in names:
        lv = R[n]['real']['levels'][lev]
        cells = []
        for k in kinds:
            s, m = surv(lv['probes'], k)
            nul = [surv(R[n][md]['levels'][lev]['probes'], k)[0] for md in R[n] if md != 'real']
            cells.append('%4d/%-4d|%3s' % (s, m, max(nul) if nul else '-'))
        nulc = [R[n][md]['levels'][lev]['corr_zA_zB'] for md in R[n] if md != 'real']
        print('%-7s' % n + '  '.join('%14s' % c for c in cells) + '   %.3f | %s' % (lv['corr_zA_zB'], ('%.3f' % max(nulc)) if nulc else '-'))
# word identity arrows ('random' partitions of word types) by lag
print('\nword-identity (random type partition) replicated / probes by lag')
for n in names:
    lv = R[n]['real']['levels']['word']
    print('%-7s' % n, ' '.join('lag%d %d/%d' % ((k,) + surv(lv['probes'], 'random', k)) for k in (1, 2, 3, 4)))
print('\nTop Voynich line-level and word-level replicated arrows (ZL)')
for lev in ('word', 'line'):
    for d in R['ZL']['real']['levels'][lev]['top'][:6]:
        print(lev, d['kind'], 'lag', d['lag'], 'A:', d['a'][:8], d['na'], 'B:', d['b'][:8], d['nb'], 'zA %.1f zB %.1f' % (d['zA'], d['zB']))
