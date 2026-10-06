"""v78 cycle 1: at what LEVEL do repeated tokens arise? Section -> page -> line -> adjacency decomposition.

For each corpus (E1c tokens), the within-line rate of identical tokens at lag k (k = 1..5) is compared with three
nested nulls that keep the page/line skeleton: tokens shuffled within the section (SEC), within the page across its
lines (PAGE), and within each line (LINE). Ratios:
  page   = PAGE null / SEC null   (page-level clustering of vocabulary)
  line   = LINE null / PAGE null  (a line re-uses its own tokens beyond its page)
  adj    = observed lag 1 / LINE null (adjacency beyond line composition: true doubling)
  lag2   = observed lag 2 / LINE null; far = observed lags 3-5 / LINE null
  spike  = observed lag 1 / observed lags 2-3
Mechanism predictions: grammatical reduplication, dittography, tally and list-ditto all write the copy NEXT to its
source (adj >> 1, spike >> 1); generator self-copying from a window spreads copies over the window (page/line, spike
~1); chance gives 1 everywhere. Kill control: doubles planted into the Voynich at the observed rate must show up as adj.
Bootstrap: 200 page resamples. Discovery/held-out = leaf parity.
"""
import os, sys, random, pickle, json
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import v78_lib as L

FN = 'v78_cycle1.txt'


def page_counts(pages, nsh=10, seed=0, K=6):
    """per page arrays: obs[k], n[k], and null expectations (SEC, PAGE, LINE) for lags 1..K-1."""
    rng = random.Random(seed)
    T = [[[L.e1c(w) for w in l['w']] for l in p['lines']] for p in pages]
    by = {}
    for p, P in zip(pages, T):
        by.setdefault(p['sec'], []).extend(x for Lx in P for x in Lx)
    np_ = len(pages)
    obs = np.zeros((np_, K)); n = np.zeros((np_, K)); nul = {m: np.zeros((np_, K)) for m in ('SEC', 'PAGE', 'LINE')}
    def cnt(lines, arr, pi, w=1.0):
        for Lx in lines:
            for k in range(1, K):
                for i in range(len(Lx) - k):
                    if Lx[i] == Lx[i + k]: arr[pi, k] += w
    for pi, P in enumerate(T):
        cnt(P, obs, pi)
        for Lx in P:
            for k in range(1, K): n[pi, k] += max(0, len(Lx) - k)
    for s in range(nsh):
        pools = {k: v[:] for k, v in by.items()}
        for v in pools.values(): rng.shuffle(v)
        it = {k: iter(v) for k, v in pools.items()}
        for pi, (p, P) in enumerate(zip(pages, T)):
            cnt([[next(it[p['sec']]) for _ in Lx] for Lx in P], nul['SEC'], pi, 1 / nsh)
            flat = [x for Lx in P for x in Lx]; rng.shuffle(flat); j = 0; Q = []
            for Lx in P: Q.append(flat[j:j + len(Lx)]); j += len(Lx)
            cnt(Q, nul['PAGE'], pi, 1 / nsh)
            Q = [rng.sample(Lx, len(Lx)) for Lx in P]
            cnt(Q, nul['LINE'], pi, 1 / nsh)
    return obs, n, nul


def ratios(obs, n, nul, idx):
    o = obs[idx].sum(0); N = n[idx].sum(0); S = {m: v[idx].sum(0) for m, v in nul.items()}
    sec = S['SEC'][1:].sum(); pag = S['PAGE'][1:].sum(); lin = S['LINE'][1:].sum()
    r = dict(rate1=o[1] / N[1], page=pag / sec, line=lin / pag, adj=o[1] / S['LINE'][1], lag2=o[2] / S['LINE'][2],
             far=o[3:].sum() / S['LINE'][3:].sum(), spike=(o[1] / N[1]) / ((o[2] + o[3]) / (N[2] + N[3])))
    return r


def summarize(pages, B=200, seed=1):
    obs, n, nul = page_counts(pages)
    idx = np.arange(len(pages)); r0 = ratios(obs, n, nul, idx)
    rng = np.random.default_rng(seed); bs = {k: [] for k in r0}
    for b in range(B):
        ii = rng.integers(0, len(pages), len(pages))
        for k, v in ratios(obs, n, nul, ii).items(): bs[k].append(v)
    ci = {k: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) for k, v in bs.items()}
    return r0, ci


def fmt(r, ci, keys=('rate1', 'page', 'line', 'adj', 'lag2', 'far', 'spike')):
    return ', '.join('%s %.3f [%.2f-%.2f]' % (k, r[k], ci[k][0], ci[k][1]) if k != 'rate1' else 'rate %.4f' % r[k]
                     for k in keys)


def main():
    C = pickle.load(open(os.path.join(L.CK, 'corpora.pkl'), 'rb'))
    res = {}
    import sys
    only = sys.argv[1:] == ['plant']
    for name, c in ([] if only else C.items()):
        r, ci = summarize(c['pages'])
        res[name] = dict(kind=c['kind'], r=r, ci=ci)
        print(name, c['kind'], fmt(r, ci), flush=True)
    # Voynich halves and raw level
    for nm in ([] if only else ('VOY_ZL', 'VOY_IT')):
        for h in (0, 1):
            r, ci = summarize(L.half(C[nm]['pages'], h)); res['%s_h%d' % (nm, h)] = dict(kind='?', r=r, ci=ci)
            print(nm, 'half', h, fmt(r, ci), flush=True)
    # kill control: plant adjacency doubles into the Voynich (REDUP class, and DITTOG) at about the observed rate
    import v72_lib as V
    for mech, P in (('REDUP', dict(REDUP=dict(k=0.02, a=0.5, r=0.3, t=0.05))), ('DITTOG', dict(DITTOG=dict(r=0.02, x=0.2))),
                    ('GEN60', dict(GEN=dict(p=0.15, W=60, m=0.3)))):
        S, lab = L.apply(C['VOY_ZL']['pages'], P, 78)
        sh, nd = L.shares(S, lab)
        r, ci = summarize(S); res['PLANT_' + mech] = dict(kind='plant', r=r, ci=ci, share=sh)
        print('PLANT', mech, 'share of doubles planted %.2f' % sh[list(P)[0]], fmt(r, ci), flush=True)
    json.dump(res, open(os.path.join(L.CK, 'c1%s.json' % ('plant' if only else '')), 'w'), default=float, indent=1)


if __name__ == '__main__':
    main()
