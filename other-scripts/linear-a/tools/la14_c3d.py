#!/usr/bin/env python3
"""LA-14 cycle 3, part D: univariate check of every scalar structure statistic on the cached cycle-1 forgeries.
For each seed: AUC of the statistic alone, real half B vs forgeries (0.5 = no difference; < 0.5 = higher in real).
Reported as 'real-side AUC' = 1 - AUC, so > 0.5 means the statistic is HIGHER in real tablets.
Calibration: the same forger against its own world (FW_X); copy forgers against COPY0.
Also binary probes suggested by the cycle-1 coefficient lists (found on the same data: grade C at best)."""
import json, os
from collections import defaultdict
import numpy as np
from sklearn.metrics import roc_auc_score
import la14_common as C
from la14_c3 import STATS, doc_stats

FD = os.path.join(C.OUT, 'forg')


def probes(d):
    c = C.ctoks(d); L = [t for t in c if t.startswith('L:')]
    base = [t.split('+')[0] for t in L]
    r = {}
    r['p:GRA<OLE'] = float('L:GRA' in base and 'L:OLE' in base and base.index('L:GRA') < base.index('L:OLE'))
    r['p:OLE<GRA'] = float('L:GRA' in base and 'L:OLE' in base and base.index('L:OLE') < base.index('L:GRA'))
    r['p:TE2'] = float(len(c) > 1 and c[1] == 'W:TE')
    r['p:KURO_last_line'] = float(any(t == 'W:KU-RO' for t in C.lines_of(d)[-1]))
    r['p:KURO_any'] = float('W:KU-RO' in c)
    W = [t for t in c if t.startswith('W:')]
    r['p:word_repeat'] = float(len(W) != len(set(W)))
    return r


def stat_rows(docs):
    out = []
    for d in docs:
        r = doc_stats(d); r.update(probes(d)); out.append(r)
    return out


KEYS = STATS + ['p:GRA<OLE', 'p:OLE<GRA', 'p:TE2', 'p:KURO_last_line', 'p:KURO_any', 'p:word_repeat']


def run(cname, fname, seeds):
    docs = C.corpus(cname); res = defaultdict(list)
    for s in seeds:
        p = os.path.join(FD, '%s_%s_%d_0.json' % (cname, fname, s))
        if not os.path.exists(p): continue
        fake, _ = json.load(open(p))
        A, B = C.split(docs, s); real = C.real_for(fname, B)
        R = stat_rows(real); F = stat_rows(fake)
        for k in KEYS:
            x = [r.get(k, np.nan) for r in R]; y = [r.get(k, np.nan) for r in F]
            v = np.array(x + y); lab = np.array([0] * len(x) + [1] * len(y)); m = ~np.isnan(v)
            if m.sum() < 20 or len(set(lab[m])) < 2 or len(set(v[m])) < 2: continue
            res[k].append(1 - roc_auc_score(lab[m], v[m]))
    return res


if __name__ == '__main__':
    seeds = range(6)
    pairs = [('LA', 'MK2', 'FW_MK2'), ('LA', 'FLAT', 'FW_FLAT'), ('LA', 'NEUR', 'FW_NEUR'),
             ('LB', 'MK2', 'FW_MK2'), ('LB', 'FLAT', 'FW_FLAT'), ('PLA', 'FLAT', 'FW_FLAT')]
    out = ['D. Univariate real-side AUC (>0.5: higher in real) minus the same in the forger\'s world; * = t>3 and |d|>0.04']
    cache = {}
    def get(c, f):
        if (c, f) not in cache: cache[(c, f)] = run(c, f, seeds)
        return cache[(c, f)]
    hdr = '%-18s' % 'stat' + ''.join('%15s' % ('%s/%s' % (c, f)) for c, f, _ in pairs)
    out.append(hdr)
    for k in KEYS:
        row = '%-18s' % k[:18]
        for c, f, w in pairs:
            a = np.array(get(c, f).get(k, [])); b = np.array(get(w, f).get(k, []))
            if len(a) < 3 or len(b) < 3: row += '%15s' % '-'; continue
            d = a.mean() - b.mean(); t = d / (np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b)) + 1e-9)
            row += '%9.3f%+5.2f%s' % (a.mean(), d, '*' if abs(t) > 3 and abs(d) > 0.04 else ' ')
        out.append(row)
    out.append('\nD2. Copy-edit forgers on LA: univariate real-side AUC minus COPY0 (* = t>3 and |d|>0.03)')
    cf = ['E_num', 'E_numnear', 'E_numshuf', 'E_swapline', 'E_word', 'E_first', 'E_last', 'E_wordshuf', 'E_splice', 'E_logo']
    out.append('%-18s' % 'stat' + ''.join('%11s' % f for f in cf))
    ref = get('LA', 'COPY0')
    for k in KEYS:
        row = '%-18s' % k[:18]; any_ = False
        for f in cf:
            a = np.array(get('LA', f).get(k, [])); b = np.array(ref.get(k, []))
            if len(a) < 3 or len(b) < 3: row += '%11s' % '-'; continue
            d = a.mean() - b.mean(); t = d / (np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b)) + 1e-9)
            sig = abs(t) > 3 and abs(d) > 0.03; any_ |= sig
            row += '%10.3f%s' % (d, '*' if sig else ' ')
        if any_: out.append(row)
    txt = '\n'.join(out); print(txt)
    open(os.path.join(C.OUT, 'c3d_report.txt'), 'w').write(txt + '\n')
