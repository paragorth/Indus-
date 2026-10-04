"""v22 cycle 3: WHERE IS THE PROGRAM ANCHORED? The left-to-right program starts in stage 0 and may stop anywhere.
Run forward it is anchored at the paragraph start (like a recipe: name first); fitted on reversed paragraphs it is
anchored at the paragraph end (like a closing formula). Held-out gain forward minus backward, for Voynich (all, A, B),
Latin and Italian herbals, and line-shuffled Voynich as a null. Second test: do Viterbi stage changes fall on line
starts more than chance (is the program clocked by lines)?"""
import os, sys, json, time, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from multiprocessing import Pool
import v22_lib as L

SL = [3, 6]
R = 5


def reverse(U):
    return [dict(u, lines=[list(reversed(l)) for l in reversed(u['lines'])]) for u in U]


def get_units(kind, lang, mode, seed):
    C = L.voynich_pages(kind) if kind in ('ZL3b', 'IT2a') else L.herbal_pages(kind)
    U = L.units(C, 'para')
    coder = L.Coder(U)
    if lang: U = [u for u in U if u['lang'] == lang]
    if 'lineshuf' in mode: U = L.shuffle_lines(U, random.Random(seed))
    if 'rev' in mode: U = reverse(U)
    return U, coder


def t_dir(spec):
    name, kind, lang, mode, seed = spec
    f = f'c3_{name}.json'
    got = L.load(f)
    if got: return got
    t0 = time.time()
    U, coder = get_units(kind, lang, mode, seed)
    codes = L.code_all(U, coder)
    res = L.evaluate(codes, coder.sizes, L.folds_by_unit(U, 5, 4242), SL, R, seed)
    # line-clock test on a full-data fit (S=6)
    seq = L.Seq(codes)
    m = L.fit_hmm(seq, coder.sizes, 6, R, np.random.default_rng(seed + 1))
    paths = L.viterbi_paths(m, seq, 6)
    ch_ls = ch_mid = n_ls = n_mid = 0
    for c, p in zip(codes, paths):
        chg = p[1:] != p[:-1]; ls = c['lp'][1:] == 0
        ch_ls += chg[ls].sum(); n_ls += ls.sum(); ch_mid += chg[~ls].sum(); n_mid += (~ls).sum()
    res['chg_at_linestart'] = ch_ls / max(1, n_ls); res['chg_elsewhere'] = ch_mid / max(1, n_mid)
    res.update(name=name, mode=mode, lang=lang, secs=time.time() - t0)
    L.save(f, res)
    print(name, {k: round(res[k], 1) for k in ('H3', 'H6', 'best_rigid', 'best_mix')},
          round(res['chg_at_linestart'], 3), round(res['chg_elsewhere'], 3), flush=True)
    return res


if __name__ == '__main__':
    jobs = []
    for kind, lang, tag in (('ZL3b', None, 'ZL'), ('ZL3b', 'A', 'ZLA'), ('ZL3b', 'B', 'ZLB'), ('LA', None, 'LA'), ('IT', None, 'IT')):
        for mode in ('fwd', 'rev'):
            jobs.append((f'{tag}_{mode}', kind, lang, mode, 3))
    for i in range(2):
        for mode in ('lineshuf_fwd', 'lineshuf_rev'):
            jobs.append((f'ZL_{mode}_{i}', 'ZL3b', None, mode, 30 + i))
    with Pool(2) as P:
        res = P.map(t_dir, jobs, chunksize=1)
    L.save('c3_all.json', res)
