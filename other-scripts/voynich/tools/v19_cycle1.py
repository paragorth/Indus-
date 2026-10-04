"""v19 cycle 1: per-stretch sorted-order search on all corpora, kinds and keys, with planted controls.

Jobs are (corpus, kind, key); each job is checkpointed to data/results/v19/c1/<job>.json.
Two worker processes.
"""
import json, os, random, sys, math, zlib
from multiprocessing import Pool
from v19_lib import *

OUT = os.path.join(RES, 'c1'); os.makedirs(OUT, exist_ok=True)
KINDS = ['pagewords', 'win5', 'lineinit', 'linefinal', 'parainit', 'labels']
KEYLIST = ['F', 'L2', 'L', 'R1', 'R']


def planted_lines(lines, seed=19):
    """8 ZL pages, 4 planting types, each with its own random glyph order. Returns lines + truth."""
    rng = random.Random(seed)
    order, by = pages_of(lines)
    cand = [p for p in order if len(by[p]) >= 12]
    pages = rng.sample(cand, 8)
    alltypes = Counter(tuple(w) for L in lines for w in L['words'])
    types = [list(w) for w, c in alltypes.items() if len(w) >= 2]
    syms = sorted({g for L in lines for w in L['words'] for g in w})
    out, truth = [], {}
    for k, p in enumerate(pages):
        ptype = ['full', 'first', 'head', 'noisy'][k % 4]
        perm = syms[:]; rng.shuffle(perm)
        rank = {g: i for i, g in enumerate(perm)}
        pid = 'PL-%s-%s' % (ptype, p)
        truth[pid] = (ptype, perm)
        src = [dict(lines[i]) for i in by[p]]
        flat = [w for L in src for w in L['words']]
        if ptype in ('full', 'noisy'):
            new = sorted(flat, key=sortkey_fn(rank))
            if ptype == 'noisy':  # 40% of entries moved to random places
                m = len(new)
                for _ in range(int(0.4 * m)):
                    i = rng.randrange(m); w = new.pop(i); new.insert(rng.randrange(m), w)
        elif ptype == 'first':
            new = sorted(flat, key=lambda w: (rank[w[0]], rng.random()))
        if ptype in ('full', 'noisy', 'first'):
            pos = 0
            for L in src:
                n = len(L['words']); L['words'] = new[pos:pos + n]; pos += n
        else:  # head: glossary headwords at every line start, sorted, rest untouched
            heads = sorted(rng.sample(types, len(src)), key=sortkey_fn(rank))
            for L, h in zip(src, heads):
                L['words'] = [h] + L['words'][1:]
        for L in src:
            L['page'] = pid
            L['para_start'] = False if ptype == 'head' else L['para_start']
            out.append(L)
    return out, truth


def corpora():
    zl = voynich_lines('ZL3b'); it = voynich_lines('IT2a')
    iso = json.load(open(os.path.join(DATA, 'derived', 'v19_isidore.json')))
    pl, truth = planted_lines(zl)
    return {'ZL': zl, 'IT': it, 'PLANT': pl, 'LatX': latin_lines(iso['X'], section='X'),
            'LatXVI': latin_lines(iso['XVI'], section='XVI'), 'LatXVII': latin_lines(iso['XVII'], section='XVII')}, truth


def stretches_for(C, cname, kind):
    if kind == 'labels':
        if cname != 'ZL':
            return []
        by = voynich_labels()
        return [(f, [(0, i, glyphs(w)) for i, w in enumerate(ws)]) for f, ws in by.items() if len(ws) >= 8]
    if cname == 'PLANT' and kind == 'parainit':
        return []
    return make_stretches(C[cname], kind)


def job(args):
    cname, kind, key = args
    fn = os.path.join(OUT, '%s_%s_%s.json' % (cname, kind, key))
    d = json.load(open(fn)) if os.path.exists(fn) else None
    if d == [] or (d and 'pooled_ws' in d):
        return fn
    C, _ = corpora()
    st = stretches_for(C, cname, kind)
    if not st:
        json.dump([], open(fn, 'w')); return fn
    sym = encode(st)
    big = cname in ('ZL', 'IT')
    if d is None:
        rows = run_engine(st, sym, key, R=60 if big else 100, restarts=5 if big else 6, ils=15 if big else 25, nrand=2000,
                          seed=zlib.crc32(fn.encode()) % 100000, tag=cname + kind + key)
        d = {'rows': rows}
    # pooled: ONE order for all stretches of the cell; null = within-stretch shuffles (nullmode 0).
    # (an earlier version used a global cross-stretch shuffle; kept as 'pooled' where it exists)
    d['pooled_ws'] = run_engine(st, sym, key, R=30, pooled=1, nullmode=0, restarts=6, ils=30, nrand=200000, seed=7,
                                tag='P' + cname + kind + key)[0]
    json.dump(d, open(fn + '.tmp', 'w'))
    os.replace(fn + '.tmp', fn)
    return fn


if __name__ == '__main__':
    C, truth = corpora()
    json.dump(truth, open(os.path.join(RES, 'c1_truth.json'), 'w'))
    jobs = [(c, k, key) for c in ['PLANT', 'LatX', 'LatXVI', 'LatXVII', 'ZL', 'IT'] for k in KINDS for key in KEYLIST
            if not (c in ('ZL', 'IT') and k == 'win5' and key in ('L2', 'R1'))]
    jobs.sort(key=lambda j: (j[0] in ('ZL', 'IT') and j[1] == 'win5', ['PLANT', 'LatX', 'LatXVI', 'LatXVII', 'ZL', 'IT'].index(j[0])))
    with Pool(2) as P:
        for fn in P.imap_unordered(job, jobs):
            print('done', os.path.basename(fn), flush=True)
