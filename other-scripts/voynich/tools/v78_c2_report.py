"""v78 cycle 2 report: calibrated mechanism shares from the stored ABC runs (c2b*).

Groups: VISUAL copy = DITTOG + GEN (the glyphs already written are copied), MEANING-level = REDUP + LIST + TALLY
(the copy is re-written through the spelling layer), CHANCE (frequency-proportional independent recurrence).
Special mechanism = argmax over REDUP, LIST, DITTOG, TALLY.
Self-calibration: 150 simulated runs on the Voynich ZL base are used in turn as pseudo-observed data and the ABC is
re-run on the other runs (same acceptance), giving the correlation of estimated and true shares per mechanism.
Upper bounds: 90th percentile of each mechanism's share among accepted runs.
"""
import os, sys, pickle, json
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import v78_lib as L

GRP = dict(VISUAL=('DITTOG', 'GEN'), MEANING=('REDUP', 'LIST', 'TALLY'), CHANCE=('CHANCE', 'BASE'))
SPEC = ('REDUP', 'LIST', 'DITTOG', 'TALLY')


def load(tag, name):
    return pickle.load(open(os.path.join(L.CK, '%s_%s.pkl' % (tag, name)), 'rb'))


def post(D):
    R = D['R']; idx = D['idx']
    sh = {m: np.array([R[i]['sh'][m] for i in idx]) for m in L.MECHS + ['BASE']}
    g = {k: float(sum(sh[m].mean() for m in v)) for k, v in GRP.items()}
    spec = {m: float(sh[m].mean()) for m in SPEC}
    ub = {m: float(np.percentile(sh[m], 90)) for m in L.MECHS}
    return g, spec, max(spec, key=spec.get), ub


def selfcal(D, n=150, acc=0.03, seed=0):
    R = D['R']; F = np.array([r['f'] for r in R]); scale = np.array(D['scale'])
    S = {m: np.array([r['sh'][m] for r in R]) for m in L.MECHS}
    rng = np.random.default_rng(seed); test = rng.choice(len(R), n, replace=False)
    est = {m: [] for m in L.MECHS}; tru = {m: [] for m in L.MECHS}
    k = max(10, int(acc * len(R)))
    for t in test:
        d = np.sqrt((((F - F[t]) / scale) ** 2).sum(1)); d[t] = np.inf
        idx = np.argsort(d)[:k]
        for m in L.MECHS: est[m].append(S[m][idx].mean()); tru[m].append(S[m][t])
    return {m: float(np.corrcoef(est[m], tru[m])[0, 1]) for m in L.MECHS}


def main():
    out = {}
    vj = json.load(open(os.path.join(L.CK, 'c2bvoy.json'))); cj = json.load(open(os.path.join(L.CK, 'c2bctl.json')))
    for tag, J in (('c2bvoy', vj), ('c2bctl', cj)):
        for name, r in J.items():
            D = load(tag, name); g, spec, top, ub = post(D)
            out[name] = dict(kind=r['kind'], groups=g, special=spec, top_special=top, ub90=ub,
                             hold_pct=r['hold_pct'], dmin=r['dmin'], dprior=r['dprior'])
            print('%-12s %-6s groups %s | special %s -> %s | ub90 %s | hold pct %.3f' % (
                name, r['kind'], {k: round(v, 2) for k, v in g.items()}, {k: round(v, 2) for k, v in spec.items()},
                top, {k: round(v, 2) for k, v in ub.items()}, r['hold_pct']))
    cal = selfcal(load('c2bvoy', 'VOY_ZL')); out['_selfcal_ZL'] = cal
    print('self-calibration r (estimated vs true share, 150 pseudo-observed runs):', {k: round(v, 2) for k, v in cal.items()})
    json.dump(out, open(os.path.join(L.CK, 'c2b_report.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
