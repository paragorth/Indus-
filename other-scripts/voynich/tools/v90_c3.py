"""v90 cycle 3: exhaustive wheel sweep at three unit levels. Every proper subset of the 7 fields
(126) plus the whole word is treated as a candidate concept wheel; for each unit level (page:
odd -> even lines; paragraph: odd -> even lines, baseline other paragraphs of the same
section x hand x language; line: even-position -> odd-position words, baseline other lines of
the same page) the conjunction statistic asks whether the wheel's rare combinations recur in the
unit beyond the unit's field marginals (within-unit field shuffles). A content wheel = a proper
subset whose conjunction z beats the whole word's and whose complement is flat.
All folios (both page halves) and anchor scheme 0; replicated in IT2a and GC2a."""
import os, sys, json, time, itertools
import numpy as np
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v90_lib as L
import v90_c2 as C2

R = int(os.environ.get('V90_R', '8'))
SUBSETS = [list(c) for k in range(1, L.NF + 1) for c in itertools.combinations(range(L.NF), k)]


def build(name):
    if name == 'VOY_GC':
        return L.voynich_tokens('GC2a')
    return C2.build(name)


def run(name):
    t0 = time.time()
    toks, order = build(name)
    enc = L.Enc(toks, order, 0)
    out = {'name': name, 'levels': {}}
    for lev in ('page', 'para', 'line'):
        unit, grp, A, B, idx = L.units(enc, toks, lev)
        rows = []
        for cols in SUBSETS:
            r = L.conj_unit(enc.F[:, cols], unit, grp, A, B, idx, R=R, seed=len(rows))
            r['cols'] = cols
            rows.append(r)
        out['levels'][lev] = rows
    out['secs'] = time.time() - t0
    json.dump(out, open(os.path.join(L.CK, 'c3_%s.json' % name), 'w'), default=float)
    return name, out['secs']


if __name__ == '__main__':
    names = sys.argv[1:] or ['VOY', 'VOY_IT', 'VOY_GC', 'SHUF0', 'MARK', 'C_HERB3', 'C_ASTR3', 'HABIT3', 'C_HERBH', 'SHUF1']
    with Pool(2) as P:
        for n, s in P.imap_unordered(run, names):
            print(n, '%.0fs' % s, flush=True)
