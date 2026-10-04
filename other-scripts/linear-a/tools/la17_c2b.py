"""LA-17 cycle 2b: the epidemic's turnover signal read directly. If words are lost over time
(SIR), sites sampled at similar times share more words than richness and distance predict.
Observed/expected sharing (expectation = pair richness product / total types, plus a
document-label-permutation null); spectral seriation (Fiedler vector) of the residual; scored
against deposit phase and, separately, against geography. LB control: KN must sit at one end."""
import random, json
import numpy as np
from scipy.stats import spearmanr
from la17_common import *


def oe(inc):
    f = inc.astype(float); r = f.sum(0); S = f.T @ f
    E = np.outer(r, r) / len(inc)
    return S, E


def fiedler(A):
    A = A.copy(); np.fill_diagonal(A, 0); A = np.maximum(A, 0)
    Lp = np.diag(A.sum(1)) - A
    w, v = np.linalg.eigh(Lp)
    return v[:, 1]


def run(docs, sites, dates, nperm=300, tag=''):
    d = build(docs, sites); codes = d['codes']; K = len(codes)
    S, E = oe(d['inc'])
    # permutation null for each pair's shared count (doc labels permuted)
    null = []
    for i in range(nperm):
        dd = build(doc_shuffle(docs, sites, random.Random(i)), sites)
        null.append(oe(dd['inc'])[0])
    null = np.array(null)
    Z = (S - null.mean(0)) / (null.std(0) + 1e-9)
    np.fill_diagonal(Z, 0)
    # remove geography: regress Z on distance across pairs, keep residual
    iu = np.triu_indices(K, 1)
    x = np.log(d['D'][iu] + 5); y = Z[iu]
    b = np.polyfit(x, y, 1); res = np.zeros((K, K)); res[iu] = y - np.polyval(b, x); res = res + res.T
    out = {}
    for name, A in [('Z', Z), ('resid', res)]:
        fv = fiedler(A - A.min() * (A != 0))
        dv = np.array([dates.get(c, np.nan) for c in codes]); ok = ~np.isnan(dv)
        r = spearmanr(fv[ok], dv[ok])[0]
        rng = np.random.default_rng(0)
        nul = [abs(spearmanr(fv[ok], rng.permutation(dv[ok]))[0]) for _ in range(20000)]
        p = np.mean(np.array(nul) >= abs(r))
        out[name] = (r, p, ' < '.join(codes[i] for i in np.argsort(fv)))
        print(tag, name, 'seriation', out[name][2], '| |rho| with dates %.2f P %.3f' % (abs(r), p))
    # date-difference test: do pairs closer in date share more (z), controlling distance?
    dd_ = np.array([abs(dates.get(codes[i], np.nan) - dates.get(codes[j], np.nan)) for i, j in zip(*iu)])
    ok = ~np.isnan(dd_)
    from numpy.linalg import lstsq
    X = np.column_stack([np.ones(ok.sum()), x[ok], dd_[ok]])
    coef = lstsq(X, y[ok], rcond=None)[0]
    # permutation of dates over sites
    rng = np.random.default_rng(1); cs = []
    dl = [dates.get(c, np.nan) for c in codes]
    for _ in range(5000):
        pdv = rng.permutation(dl)
        q = np.array([abs(pdv[i] - pdv[j]) for i, j in zip(*iu)])
        ok2 = ~np.isnan(q)
        X2 = np.column_stack([np.ones(ok2.sum()), x[ok2], q[ok2]])
        cs.append(lstsq(X2, y[ok2], rcond=None)[0][2])
    pc = np.mean(np.array(cs) <= coef[2])
    print(tag, 'pair z ~ log dist + |date diff|: dist coef %.2f, date-diff coef %.2f (perm P one-sided %.3f)' % (coef[1], coef[2], pc))
    out['coef'] = (coef.tolist(), pc)
    out['Z'] = Z.tolist(); out['codes'] = codes
    return out, Z, codes, d


if __name__ == '__main__':
    dates = la_dates()[0]
    o, Z, codes, d = run(load_la(), LA_SITES, dates, tag='LA')
    i, j = codes.index('PH'), codes.index('HT')
    near = [(a, b) for a in range(len(codes)) for b in range(a + 1, len(codes)) if d['D'][a, b] < 60 and (a, b) != (min(i, j), max(i, j))]
    print('LA PH-HT (3 km, MM II vs LM IB) z %.2f; other pairs < 60 km: mean z %.2f (n %d)' % (Z[i, j], np.mean([Z[a, b] for a, b in near]), len(near)))
    for a in range(len(codes)):
        print('  ', codes[a], ' '.join(f'{Z[a, b]:5.1f}' for b in range(len(codes))))
    ob, Zb, cb, db = run(load_lb(), LB_SITES, LB_DATES, tag='LB')
    for a in range(len(cb)):
        print('  ', cb[a], ' '.join(f'{Zb[a, b]:5.1f}' for b in range(len(cb))))
    json.dump(dict(la=o, lb=ob), open(os.path.join(OUT, 'c2b.json'), 'w'), default=float)
