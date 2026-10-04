"""v35 runner: the v28 runner (exact v25 statistics) on v35 corpora.  Results cached as data/v35_ckpt/res_<name>.pkl.
usage: python3 v35_run.py name [name ...]   (2 workers)"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import v35_lib as V, v28_lib as X, v25_lib as L


def sims_for(c):
    A = c['alph']; out = {}
    fonts = [f for f in c['fonts'] if all(X.covered(g, X.FONT[f]) and X.has_ink(g, X.FONT[f]) for g in A)]
    for k, f in enumerate(fonts):
        Im = X.image_sims(A, X.FONT[f], c['xref'])
        out[f'img:{f}'] = Im['combo']
        if k == 0:
            for d in ('zone', 'frame', 'topo'):
                out[f'{d}:{f}'] = Im[d]
    sh = c.get('hand')
    if sh and all(g in sh for g in A):
        F, _ = L.shape_matrix(sh, A)
        out['hand'] = L.jaccard_sim(F)
    return out


def run(name, nperm=5000):
    fn = f'res_{name}.pkl'
    r = X.load(fn)
    if r is not None:
        return r
    t = time.time()
    c = V.build(name)
    beh = X.load(f'beh_{name}.pkl')
    if beh is None:
        beh = L.behaviour(c['words'], c['alph'])
        beh = {k: v for k, v in beh.items() if k != '_W'}
        X.save(f'beh_{name}.pkl', beh)
    S = sims_for(c)
    X.save(f'sims_{name}.pkl', S)
    rows, summ, _ = X.analyse(name, c['words'], c['alph'], S, nperm=nperm, beh=beh)
    res = dict(rows=rows, summ=summ, n=len(c['alph']), tok=sum(map(len, c['words'])), secs=time.time() - t)
    X.save(fn, res)
    return res


def brief(name, res):
    cells = []
    for (s, m), (r, p, p2, pr) in sorted(res['summ'].items()):
        if not (s.startswith('img') or s == 'hand'):
            continue
        cells.append(f'{s}/{m} r {r:+.2f} p {p:.4f}')
    return f'{name} ({res["n"]} units, {res["tok"]} tokens, {res["secs"]:.0f}s): ' + '; '.join(cells)


if __name__ == '__main__':
    from multiprocessing import Pool
    names = sys.argv[1:]
    with Pool(2) as P:
        for n, res in zip(names, P.imap(run, names)):
            print(brief(n, res), flush=True)
