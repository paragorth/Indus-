#!/usr/bin/env python3
"""LA-14 cycle 3: inspect what still catches the forgers.

A. Feature stability. From every single-group LR run (c1, c2), count in how many seeds a feature is
   among the 12 most real-indicative (negative coefficient = 'real') features. Compare with the
   forger's own world (FW_X with forger X), where nothing should be stable.
B. Planted recovery. In PLA, do the head>tail pairs of the planted key reach the real-indicative lists?
C. Direct parametric bootstrap of scalar structure statistics: the statistic on all real documents
   vs 40 forged corpora of equal size from each forger trained on ALL documents (in-sample, so the
   forger has every advantage). Calibration: the same on the forger's own world, and on LB and PLA.
Usage: python3 la14_c3.py
"""
import os
for v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ[v] = '1'
import json, sys, warnings
from collections import Counter, defaultdict
from multiprocessing import Pool
import numpy as np
warnings.filterwarnings('ignore')
import la14_common as C

STATS = ['arith:tot_last', 'arith:near_last', 'arith:tot_run', 'arith:tot_all', 'arith:distinct', 'arith:desc',
         'arith:round10', 'arith:ones', 'arith:maxshare', 'arith:max_last', 'arith:lsd',
         'rep:tplsame', 'rep:tpldist', 'rep:wdist', 'rep:bgrep', 'rep:wmaxrep', 'long:same_fs', 'long:same_ls',
         'long:Lsame', 'long:nLdist', 'numw:wl_corr', 'len:llsd', 'len:nlines', 'len:llmean', 'len:fN', 'len:w1',
         'x:headnonum', 'x:lastword_num', 'x:numline_end', 'x:w_after_num', 'x:hapax_first']


def extra(d):
    """A few direct long-range probes (not used by the discriminators)."""
    c = C.ctoks(d); r = {}
    r['x:headnonum'] = float(c[0].startswith('W:') and not c[1].startswith('N:'))     # heading word without number
    W = [i for i, t in enumerate(c) if t.startswith('W:')]
    r['x:lastword_num'] = float(bool(W) and W[-1] + 1 < len(c) and c[W[-1] + 1].startswith('N:'))
    L = [l for l in C.lines_of(d) if l]
    r['x:numline_end'] = np.mean([l[-1].startswith('N:') or l[-1].startswith('F:') for l in L])
    r['x:w_after_num'] = np.mean([b.startswith('W:') for a, b in zip(c, c[1:]) if a.startswith('N:')] or [0])
    return r


def doc_stats(d):
    r = {}
    for g in ('arith', 'rep', 'long', 'numw', 'len'):
        for k, v in C.GROUPS[g](d).items(): r[g + ':' + k] = float(v)
    r.update(extra(d))
    return r


def corpus_stats(docs):
    rows = [doc_stats(d) for d in docs]
    out = {}
    for s in STATS:
        v = [r[s] for r in rows if s in r]
        out[s] = float(np.mean(v)) if v else float('nan')
    return out


def boot(args):
    cname, fname, rep = args
    docs = C.corpus(cname)
    F, _ = C.forge(fname, docs, len(docs), 500 + rep)
    return cname, fname, rep, corpus_stats(F)


def part_c():
    jobs = []
    for c, fs in [('LA', ['MK1', 'MK2', 'MK3', 'WMK2', 'FLAT', 'NEUR']), ('LB', ['MK2', 'FLAT', 'NEUR']),
                  ('PLA', ['MK2', 'FLAT']), ('FW_MK2', ['MK2']), ('FW_FLAT', ['FLAT']), ('FW_NEUR', ['NEUR'])]:
        for f in fs:
            for r in range(40 if f != 'NEUR' else 12): jobs.append((c, f, r))
    path = os.path.join(C.OUT, 'c3_boot.jsonl')
    done = set()
    if os.path.exists(path):
        for l in open(path):
            x = json.loads(l); done.add((x['c'], x['f'], x['rep']))
    jobs = [j for j in jobs if j not in done]
    for c in {j[0] for j in jobs}: C.corpus(c)
    with Pool(2) as P, open(path, 'a') as fo:
        for c, f, r, st in P.imap_unordered(boot, jobs):
            fo.write(json.dumps({'c': c, 'f': f, 'rep': r, 'st': st}) + '\n'); fo.flush()
    B = defaultdict(list)
    for l in open(path):
        x = json.loads(l); B[(x['c'], x['f'])].append(x['st'])
    lines = ['C. Parametric bootstrap: real statistic vs forged corpora (z = (real - mean)/sd; |z|>=3 marked)']
    real = {c: corpus_stats(C.corpus(c)) for c in {k[0] for k in B}}
    keys = sorted(B)
    lines.append('%-16s ' % 'stat' + ' '.join('%13s' % ('%s/%s' % k)[:13] for k in keys))
    Z = {}
    for s in STATS:
        row = []
        for k in keys:
            v = np.array([x[s] for x in B[k] if not np.isnan(x[s])])
            if len(v) < 3 or np.isnan(real[k[0]][s]): row.append('%13s' % '-'); continue
            z = (real[k[0]][s] - v.mean()) / (v.std(ddof=1) + 1e-9); Z[(s,) + k] = z
            row.append('%6.3f %+5.1f%s' % (real[k[0]][s], max(-99, min(99, z)), '!' if abs(z) >= 3 else ' '))
        lines.append('%-16s ' % s + ' '.join(row))
    return lines, Z


def part_ab():
    R = []
    for tag in ('c1', 'c2'):
        p = os.path.join(C.OUT, tag + '.jsonl')
        if os.path.exists(p): R += [json.loads(l) for l in open(p)]
    cnt = defaultdict(Counter); nseed = Counter(); cntp = defaultdict(Counter)
    for r in R:
        if 'neg' not in r or r['role'] != 'real': continue
        k = (r['c'], r['f'], r['g'][0]); nseed[k] += 1
        for nm, co in r['neg']:
            if co < 0: cnt[k][nm] += 1
        for nm, co in r['pos']:
            if co > 0: cntp[k][nm] += 1
    lines = ['A. Stable real-indicative features (in >= 2/3 of seeds); [world] = same forger on its own world']
    for f in ['MK2', 'MK3', 'FLAT', 'NEUR', 'WMK2', 'E_num', 'E_word', 'E_splice', 'E_numshuf', 'E_wordshuf',
              'E_swapline', 'E_first', 'E_last', 'E_logo', 'E_swapadj', 'E_numnear']:
        for g in C.ALLG:
            k = ('LA', f, g); n = nseed[k]
            if not n: continue
            st = [(nm, c) for nm, c in cnt[k].most_common() if c >= 2 * n / 3]
            if not st: continue
            wk = ('FW_' + f, f, g)
            wtxt = ''
            if nseed[wk]:
                wst = [nm for nm, c in cnt[wk].items() if c >= 2 * nseed[wk] / 3]
                wtxt = ' [world: %d stable]' % len(wst)
            lines.append('  %s/%s (n=%d)%s: %s' % (f, g, n, wtxt, ', '.join('%s %d' % x for x in st[:10])))
    lines.append('A2. Stable forged-indicative features (forger artefacts), LA, >= 2/3 of seeds')
    for f in ['MK2', 'FLAT', 'NEUR']:
        for g in C.ALLG:
            k = ('LA', f, g); n = nseed[k]
            st = [(nm, c) for nm, c in cntp[k].most_common() if c >= 2 * n / 3]
            if st: lines.append('  %s/%s (n=%d): %s' % (f, g, n, ', '.join('%s %d' % x for x in st[:8])))
    # B. planted
    _, key = C.plant_agreement(C.load_la()[0])
    planted = {'long:hl=%s>%s' % (h, t) for h, t in key.items()}
    lines.append('B. Planted head>tail pairs among real-indicative top-12 lists (long group)')
    for c in ('PLA', 'LA'):
        for f in ['MK2', 'MK3', 'FLAT', 'NEUR', 'E_last', 'E_splice', 'E_wordshuf', 'E_first', 'COPY0']:
            k = (c, f, 'long'); n = nseed[k]
            if not n: continue
            tot = sum(cnt[k].values()); hp = sum(v for nm, v in cnt[k].items() if nm in planted)
            hl = sum(v for nm, v in cnt[k].items() if nm.startswith('long:hl='))
            lines.append('  %s/%s n=%d: planted pairs %d of %d listed features (all hl= pairs %d)' % (c, f, n, hp, tot, hl))
    return lines


if __name__ == '__main__':
    out = part_ab()
    if '--noboot' not in sys.argv:
        l2, Z = part_c(); out += l2
    txt = '\n'.join(out); print(txt)
    open(os.path.join(C.OUT, 'c3_report.txt'), 'w').write(txt + '\n')
