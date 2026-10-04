"""v19 cycle 3: THE GLOSSARY CAN BE ANYWHERE - a location scan that ignores page breaks.
Reading-order line sequence of the whole manuscript (P lines). Windows:
  LI12 / LI24 : line-initial words of 12 / 24 consecutive lines (step 6 / 12)
  AW4         : all words of 4 consecutive lines (step 2)
Each window: optimal order (ILS, engine) and z against 20 within-window shuffles (same optimiser).
Family-wise null: the SAME scan on 2 null corpora (words shuffled within page; line-initial words shuffled
within page) -> distribution of the scan maximum. Positive: a 15-line glossary (sorted headwords under a random
order) planted at a random place in a copy of ZL -> must top the scan and give its order.
GA check: for the top Voynich windows, a genetic search over orders (order crossover + swap mutation,
population 200, 300 generations) must not beat the ILS optimum (search adequacy).
"""
import json, os, random, sys, math, zlib
import numpy as np
from multiprocessing import Pool
from v19_lib import *

OUT = os.path.join(RES, 'c3'); os.makedirs(OUT, exist_ok=True)


def shuffled_within_page(lines, mode, seed):
    rng = random.Random(seed)
    out = [dict(L) for L in lines]
    order, by = pages_of(out)
    for p in order:
        idx = by[p]
        if mode == 'words':
            flat = [w for i in idx for w in out[i]['words']]; rng.shuffle(flat); pos = 0
            for i in idx:
                n = len(out[i]['words']); out[i]['words'] = flat[pos:pos + n]; pos += n
        else:
            idx = [i for i in idx if not out[i]['para_start']]
            heads = [out[i]['words'][0] for i in idx]; rng.shuffle(heads)
            for i, h in zip(idx, heads):
                out[i]['words'] = [h] + out[i]['words'][1:]
    return out


def planted_glossary(lines, seed):
    rng = random.Random(seed)
    out = [dict(L) for L in lines]
    syms = sorted({g for L in lines for w in L['words'] for g in w})
    perm = syms[:]; rng.shuffle(perm); rank = {g: i for i, g in enumerate(perm)}
    types = [list(w) for w in {tuple(w) for L in lines for w in L['words']} if len(w) >= 2]
    s = rng.randrange(0, len(out) - 15)
    heads = sorted(rng.sample(types, 15), key=sortkey_fn(rank))
    for k, h in enumerate(heads):
        out[s + k]['words'] = [h] + out[s + k]['words'][1:]
        out[s + k]['para_start'] = False
    return out, s, perm


def windows(lines, kind):
    st = []
    if kind in ('LI12', 'LI24'):
        W = 12 if kind == 'LI12' else 24
        nl = [L for L in lines if not L['para_start']]  # paragraph-initial lines excluded (gallows artefact)
        for s in range(0, len(nl) - W + 1, W // 2):
            st.append(('%d:%s' % (s, nl[s]['page']), [(0, k, nl[s + k]['words'][0]) for k in range(W)]))
    else:
        for s in range(0, len(lines) - 3, 2):
            ent = [(0, k, w) for k in range(4) for w in lines[s + k]['words']]
            st.append(('%d:%s' % (s, lines[s]['page']), ent))
    return st


def job(args):
    cname, kind, key = args
    fn = os.path.join(OUT, '%s_%s_%s.json' % (cname, kind, key))
    if os.path.exists(fn):
        return fn
    zl = voynich_lines('ZL3b')
    meta = {}
    if cname == 'ZL':
        lines = zl
    elif cname == 'IT':
        lines = voynich_lines('IT2a')
    elif cname.startswith('NULLW'):
        lines = shuffled_within_page(zl, 'words', int(cname[5:]))
    elif cname.startswith('NULLH'):
        lines = shuffled_within_page(zl, 'heads', int(cname[5:]))
    elif cname.startswith('PLANTG'):
        lines, s, perm = planted_glossary(zl, int(cname[6:]))
        meta = {'start': s, 'perm': perm}
    st = windows(lines, kind)
    sym = encode(st)
    rows = run_engine(st, sym, key, R=20, restarts=4, ils=10, nrand=0, seed=zlib.crc32(fn.encode()) % 99991, tag='c3' + cname + kind + key)
    rows = [dict(id=r['id'], n=r['n'], tau=r['tau'], z=r['z'], nmean=r['nmean'], order_s=r['order_s']) for r in rows]
    json.dump({'rows': rows, 'meta': meta}, open(fn + '.tmp', 'w')); os.replace(fn + '.tmp', fn)
    return fn


if __name__ == '__main__':
    jobs = []
    for kind in ['LI12', 'LI24', 'AW4']:
        for key in ['F', 'L']:
            for c in ['PLANTG1', 'PLANTG2', 'ZL', 'NULLW1', 'NULLH1', 'IT', 'NULLW2', 'NULLH2']:
                if kind == 'AW4' and c.startswith('NULLH'):
                    continue
                if kind != 'AW4' and c.startswith('NULLW'):
                    continue
                jobs.append((c, kind, key))
    with Pool(2) as P:
        for fn in P.imap_unordered(job, jobs):
            print('done', os.path.basename(fn), flush=True)
