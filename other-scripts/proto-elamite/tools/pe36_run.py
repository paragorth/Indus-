#!/usr/bin/env python3
"""pe36 job runner: pe36_run.py CORPUS VARIANT CHUNK NREST MOVES [TAGPREFIX]
VARIANT: real | shW<seed> (within-string shuffle) | h<0|1>[shW<seed>] (tablet half, for replication)
         | sub<seed>[shW<seed>] (another size-matched draw of a control corpus).
Every corpus except PEN is subsampled (by string type) to PEN's in-fragment top-65 bigram count, so all grids
see the same amount of evidence. Output: data/pe36_ckpt/<prefix><CORPUS>_<VARIANT>__<CHUNK>.npz"""
import sys, os, re, time, random, zlib
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pe36_common import *

_TARGET = None


def target():
    global _TARGET
    if _TARGET is None:
        _TARGET = int(counts([w for w, _ in types_of(corpus('PEN'))])[1].sum())
    return _TARGET


def matched(ty, seed, tgt):
    """random subset of types whose top-65 bigram count first reaches tgt"""
    ty = list(ty); random.Random(seed).shuffle(ty)
    lo, hi = 1, len(ty)
    if counts([w for w, _ in ty])[1].sum() <= tgt:
        return ty
    while lo < hi:
        m = (lo + hi) // 2
        if counts([w for w, _ in ty[:m]])[1].sum() >= tgt: hi = m
        else: lo = m + 1
    return ty[:lo]


def build(cname, variant):
    ty = types_of(corpus(cname))
    m = re.match(r'^(real|h[01]|sub\d+)(?:shW(\d+))?$', variant)
    assert m, variant
    head, sh = m.group(1), m.group(2)
    if head.startswith('h'):
        # tablet halves: each half gets ALL types of its tablets; control corpora first matched to 2x target
        if cname != 'PEN':
            ty = matched(ty, 0, 2 * target())
        tabs = sorted({t for _, t in ty}); random.Random(7).shuffle(tabs)
        half = {t: i % 2 for i, t in enumerate(tabs)}
        ty = [(w, t) for w, t in ty if half[t] == int(head[1])]
    elif cname != 'PEN':
        sd = 0 if head == 'real' else int(head[3:])
        ty = matched(ty, sd, target())
    ws = [w for w, _ in ty]
    if sh is not None:
        ws = shuffle_within(ws, int(sh))
    return counts(ws), len(ws)


if __name__ == '__main__':
    cname, variant, chunk, nrest, moves = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
    pre = sys.argv[6] if len(sys.argv) > 6 else ''
    tag = f'{pre}{cname}_{variant}'
    out = os.path.join(CK, f'{tag}__{chunk:03d}.npz')
    if os.path.exists(out):
        print('skip', out); sys.exit()
    d, ntypes = build(cname, variant)
    R1, K = grid_dims(len(d[0]))
    if pre.startswith('g'):                      # grid-size variants: g<R1>x<K>_
        R1, K = map(int, re.match(r'g(\d+)x(\d+)_', pre).groups())
    t = time.time()
    seed = zlib.crc32(f'{tag}#{chunk}'.encode()) % 1000003 + 1
    w = 0.0 if pre.startswith('ll_') else 1.0      # ll_: tier likelihood only, no OCP / vowel-initial universals
    rows, cols, sc = anneal(d, R1, K, nrest, moves, seed, wocp=w, wvin=w)
    np.savez_compressed(out, rows=rows, cols=cols, sc=sc, signs=np.array(d[0]), B=d[1], I=d[2], F=d[3], R1=R1, K=K,
                        ntypes=ntypes)
    print(tag, chunk, f'S={len(d[0])} types={ntypes} bigrams={int(d[1].sum())} grid {R1}x{K}', f'{time.time() - t:.0f}s',
          f'score {sc[:, 0].mean():.1f}', flush=True)
