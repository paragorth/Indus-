"""v76 cycle 1: build corpora (Voynich ZL3b/IT2a, planted controls CUL/BRU/API, generators fitted to every surface
incl. a fitted paragraph copy-and-vary generator PARCOPY and a planted item-slot PARCOPY_ID), then score T random
template definitions per corpus: paragraphs sharing a template signature (view region) must be more alike on the
disjoint evaluation region than same-section cross-page pairs.
Outputs data/v76_ckpt/c1_<corpus>.npz (scores per template on all / half0 / half1 pairs) and paragraphs json.gz."""
import os, sys, json, gzip, random, time, itertools
import numpy as np
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v76_lib as V
L = V.L
T = int(os.environ.get('V76_T', 20000))
SEEDS = (1, 2)


def fit_parcopy(surf, P_real):
    """grid-fit c, m so the mean top-1 cross-page similarity of the extracted stream matches the target."""
    def top1(P):
        S = V.sim_matrix(P); page = np.array([p['page'] for p in P]); sec = np.array([p['sec'] for p in P])
        S = np.where((page[:, None] != page[None, :]) & (sec[:, None] == sec[None, :]), S, -1)
        return float(np.nanmax(S, axis=1).clip(0).mean())
    tgt = top1(P_real); best = None
    v0 = top1(V.extracted_paragraphs(V.gen_parcopy(surf, seed=99, c=0.0)))
    for c in (0.15, 0.3, 0.5, 0.7):
        for m in (0.4, 0.6, 0.8, 0.9):
            g = V.gen_parcopy(surf, seed=99, c=c, m=m)
            v = top1(V.extracted_paragraphs(g))
            d = abs(v - tgt)
            if best is None or d < best[0]: best = (d, c, m, v)
    return dict(target=tgt, c=best[1], m=best[2], fit=best[3], nocopy=v0)


def corpora():
    out = {}
    for nm in ('ZL3b', 'IT2a'):
        out[nm] = L.voynich(nm)
    for nm in ('CUL', 'BRU', 'API'):
        out[nm] = V.control_surface(nm)[1]
    return out


# ------------------------------------------------------------------ templates
COMPS = ['OPEN', 'CLOSE', 'LINIT', 'LFIN', 'SKEL', 'PPOS']


def rand_template(rng):
    k = rng.choice([1, 1, 2, 2, 3])
    comps = rng.sample(COMPS, k)
    if all(c in ('SKEL', 'PPOS') for c in comps): comps.append(rng.choice(['OPEN', 'CLOSE', 'LINIT', 'LFIN']))
    t = dict(comps=comps)
    t['open'] = sorted(rng.sample([0, 1, 2], rng.randint(1, 2)))
    t['close'] = sorted(rng.sample([1, 2], rng.randint(1, 2)))
    t['linit'] = rng.randint(1, 3); t['lfin'] = rng.randint(1, 3)
    t['skel'] = rng.choice(['nl1', 'nl2', 'len10', 'len20'])
    t['ppos'] = rng.choice(['idx', 'first', 'last'])
    kcls = rng.choice([0, 0, 2, 3, 4, 6, 8])                # 0 = identity
    syms = list('oainylrsGHCd')
    cmap = {s: (rng.randrange(kcls) if kcls else s) for s in syms}
    t['cmap'] = cmap
    t['op'] = rng.choice(['full', 'full', 'first1', 'first2', 'first3', 'last1', 'last2', 'len'])
    return t


def reduce_tok(tok, t):
    if tok is None: return '#'
    x = ''.join(str(t['cmap'].get(c, c)) for c in tok)
    op = t['op']
    if op == 'full': return x
    if op == 'len': return str(len(tok))
    j = int(op[-1])
    return x[:j] if op.startswith('first') else x[-j:]


def para_view(p):
    ls = p['lines']; toks = p['toks']
    return dict(open=[ls[0][i] if i < len(ls[0]) else None for i in range(3)],
                close={1: toks[-1], 2: toks[-2] if len(toks) > 1 else None},
                linit=[ls[i][0] if i < len(ls) else None for i in range(1, 4)],
                lfin=[ls[i][-1] if i < len(ls) else None for i in range(0, 3)],
                nl=len(ls), n=len(toks), pidx=p['pidx'], npp=p['npp'])


def signatures(views, t, vocab_cache):
    cols = []
    def red(tok):
        key = tok
        if key not in vocab_cache: vocab_cache[key] = reduce_tok(tok, t)
        return vocab_cache[key]
    for c in t['comps']:
        if c == 'OPEN':
            for i in t['open']: cols.append([red(v['open'][i]) for v in views])
        elif c == 'CLOSE':
            for i in t['close']: cols.append([red(v['close'][i]) for v in views])
        elif c == 'LINIT':
            for i in range(t['linit']): cols.append([red(v['linit'][i]) for v in views])
        elif c == 'LFIN':
            for i in range(t['lfin']): cols.append([red(v['lfin'][i]) for v in views])
        elif c == 'SKEL':
            s = t['skel']
            cols.append([str(v['nl'] if s == 'nl1' else v['nl'] // 2 if s == 'nl2' else v['n'] // (10 if s == 'len10' else 20))
                         for v in views])
        elif c == 'PPOS':
            s = t['ppos']
            cols.append([str(min(v['pidx'], 2) if s == 'idx' else int(v['pidx'] == 0) if s == 'first' else int(v['pidx'] == v['npp'] - 1))
                         for v in views])
    keys = ['\x1f'.join(r) for r in zip(*cols)]
    lab = {}
    return np.array([lab.setdefault(k, len(lab)) for k in keys])


def score_corpus(args):
    name, pages = args
    fn = os.path.join(V.CK, 'c1_%s.npz' % name)
    if os.path.exists(fn): return name
    P = V.extracted_paragraphs(pages)
    with gzip.open(os.path.join(V.CK, 'par_%s.json.gz' % name), 'wt') as f: json.dump(P, f)
    S = V.sim_matrix(P); Z = V.baseline(P, S)
    ii, jj = np.where(np.triu(~np.isnan(Z), 1))
    zv = Z[ii, jj]
    half = np.array([p['half'] for p in P])
    h0 = (half[ii] == 0) & (half[jj] == 0); h1 = (half[ii] == 1) & (half[jj] == 1)
    views = [para_view(p) for p in P]
    rng = random.Random(7600)                      # same template list for every corpus
    sc = np.zeros((T, 3), np.float32); npair = np.zeros((T, 3), np.int32)
    for k in range(T):
        t = rand_template(rng)
        lab = signatures(views, t, {})
        sz = np.bincount(lab)
        big = sz[lab] > 30
        lab = np.where(big, -1 - np.arange(len(lab)), lab)
        E = lab[ii] == lab[jj]
        for c, msk in enumerate((E, E & h0, E & h1)):
            n = int(msk.sum()); npair[k, c] = n
            sc[k, c] = zv[msk].sum() / np.sqrt(n) if n >= 3 else 0.0
    np.savez_compressed(fn, sc=sc, npair=npair)
    return name


def main():
    t0 = time.time()
    C = corpora()
    jobs = []
    fits = {}
    fitfn = os.path.join(V.CK, 'c1_fits.json')
    if os.path.exists(fitfn): fits = json.load(open(fitfn))
    for nm, surf in C.items():
        jobs.append((nm, surf))
        if nm not in fits:
            fits[nm] = fit_parcopy(surf, V.extracted_paragraphs(surf)); json.dump(fits, open(fitfn, 'w'))
            print(nm, fits[nm], flush=True)
        f = fits[nm]
        for g in ('WSHUF', 'MK2', 'SELFCIT', 'JUNC', 'SC10'):
            for s in SEEDS: jobs.append(('%s__%s_%d' % (nm, g, s), V.GENS[g](surf, seed=s)))
        for s in SEEDS:
            jobs.append(('%s__PARCOPY_%d' % (nm, s), V.gen_parcopy(surf, seed=s, c=f['c'], m=f['m'])))
            jobs.append(('%s__PARCOPYID_%d' % (nm, s), V.gen_parcopy(surf, seed=s, c=f['c'], m=f['m'], idslot=True)))
    print('jobs', len(jobs), 'built', round(time.time() - t0), flush=True)
    with Pool(2) as pool:
        for nm in pool.imap_unordered(score_corpus, jobs):
            print('done', nm, round(time.time() - t0), flush=True)


if __name__ == '__main__':
    main()
