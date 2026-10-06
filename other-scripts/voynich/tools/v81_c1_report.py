"""v81 cycle 1 report: each corpus searches its own best onset definition on the selection split; the winner is
scored on unseen leaves (leaf half 1). Voynich top definitions are re-scored on every corpus and on IT2a."""
import os, sys, glob, pickle, json
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np, v81_lib as L
import warnings; warnings.filterwarnings('ignore')

FN = 'v81_cycle1.txt'


def dstr(d):
    if d['part'] is None: return '%s/k%d/unmerged' % (d['v'], d['k'])
    cl = {}
    for c, x in zip(L.ALPHA, d['part']):
        if c != '$': cl.setdefault(x, []).append(c)
    return '%s/k%d/m%d' % (d['v'], d['k'], d['m'])


def main():
    R = []
    for f in sorted(glob.glob(os.path.join(L.CK, 'c1_sel_*.pkl'))): R += pickle.load(open(f, 'rb'))
    CC = L.pload('comp.pkl'); names = list(CC)
    print('defs', len(R))
    J = {k: np.array([r[k]['J'] for d, r in R]) for k in names}
    out = {'n': len(R)}
    hold = {}
    def H(d, k):
        c = CC[k]; o = L.onset(c, d); tr, te = L.masks(c, 'hold'); return L.stream_feats(c, o, tr, te)
    # own search
    own = {}
    for k in names:
        i = int(np.argmax(J[k])); d = R[i][0]
        h = H(d, k)
        if k == 'VOY_ZL': h_it = H(d, 'VOY_IT')
        own[k] = dict(def_=dstr(d), sel=float(J[k][i]), hold=h, med=float(np.median(J[k])), p95=float(np.percentile(J[k], 95)))
        print(k, own[k]['def_'], 'sel %.3f hold %.3f' % (own[k]['sel'], h['J']), 'median %.3f p95 %.3f' % (own[k]['med'], own[k]['p95']), flush=True)
    # Voynich top 10 definitions, scored on every corpus at holdout
    top = np.argsort(-J['VOY_ZL'])[:10]
    T = []
    for i in top:
        d = R[i][0]; row = {k: H(d, k) for k in names}
        T.append((dstr(d), float(J['VOY_ZL'][i]), row))
        print('VOYTOP', dstr(d), ' '.join('%s %.3f' % (k, row[k]['J']) for k in names), flush=True)
    # rank agreement: across all definitions, does the Voynich J profile correlate with init controls or generators?
    from scipy.stats import spearmanr
    corr = {k: float(spearmanr(J['VOY_ZL'], J[k])[0]) for k in names if k != 'VOY_ZL'}
    print('profile corr', corr)
    out.update(own={k: dict(v, hold={a: float(b) for a, b in v['hold'].items()}) for k, v in own.items()},
               top=[(a, b, {k: {x: float(y) for x, y in v.items()} for k, v in r.items()}) for a, b, r in T], corr=corr,
               voy_it_at_zl_best={a: float(b) for a, b in h_it.items()})
    json.dump(out, open(os.path.join(L.CK, 'c1_report.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
