"""v28 runner: exact v25 statistics on one corpus.  Image combo similarity under each font (+ zone/frame/topo
of the first font, + featural/hand decomposition where one exists), Mantel r and permutation p against
ppmi / svd / potts behaviour, frequency-stratified p, partial r given the word-position profile.
Results cached as data/v28_ckpt/res_<name>.pkl."""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import v28_lib as X, v28_corpora as C, v25_lib as L


def sims_for(c):
    A = c['alph']; out = {}
    for k, f in enumerate(c['fonts']):
        Im = X.image_sims(A, X.FONT[f], c['xref'])
        out[f'img:{f}'] = Im['combo']
        if k == 0:
            for d in ('zone', 'frame', 'topo'):
                out[f'{d}:{f}'] = Im[d]
    if c.get('hand'):
        sh = c['hand']
        if all(g in sh for g in A):
            F, _ = L.shape_matrix(sh, A)
            out['hand'] = L.jaccard_sim(F)
    return out


def run(name, nperm=5000):
    fn = f'res_{name.replace("/", "-")}.pkl'
    r = X.load(fn)
    if r is not None:
        return r
    t = time.time()
    c = C.build(name)
    beh = X.load(f'beh_{name.replace("/", "-")}.pkl')
    if beh is None:
        beh = L.behaviour(c['words'], c['alph'])
        beh = {k: v for k, v in beh.items() if k != '_W'}
        X.save(f'beh_{name.replace("/", "-")}.pkl', beh)
    S = sims_for(c)
    rows, summ, _ = X.analyse(name, c['words'], c['alph'], S, nperm=nperm, beh=beh)
    res = dict(rows=rows, summ=summ, n=len(c['alph']), tok=sum(map(len, c['words'])), secs=time.time() - t)
    X.save(fn, res)
    return res


def brief(name, res, sims=None):
    """one-line summary: image combo r/p per font for ppmi/potts, hand if present."""
    cells = []
    for (s, m), (r, p, p2, pr) in sorted(res['summ'].items()):
        if sims and not any(s.startswith(x) for x in sims):
            continue
        if m == 'svd':
            continue
        cells.append(f'{s}/{m} r {r:+.2f} p {p:.4f}')
    return f'{name} ({res["n"]} units, {res["tok"]} tokens): ' + '; '.join(cells)


if __name__ == '__main__':
    from multiprocessing import Pool
    names = sys.argv[1:]
    with Pool(2) as P:
        for n, res in zip(names, P.imap(run, names)):
            print(brief(n, res, sims=('img', 'hand')), flush=True)
