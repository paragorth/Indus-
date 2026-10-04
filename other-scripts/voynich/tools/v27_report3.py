"""v27 cycle 3 report: witness agreement over the random junction hypotheses.
Clean hypotheses only: witness A uses last word of line i -> first word of line j (disjoint from witness B,
which uses line interiors). Hypothesis 7 is excluded: its jitter seed equals witness B's (tie-breaks shared,
a caught artefact). Survivor rule: z on even pages > 3 AND above the 99th percentile of every null version's
clean-hypothesis z; re-test on odd pages (held out) and on IT2a."""
import sys, os, json, random, glob
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v27_lib as L
import v27_cycle3 as c

R = {tuple(json.load(open(f))['v']): json.load(open(f)) for f in glob.glob(os.path.join(L.CK, 'c3_*_*_*.json'))}
UNITS = {}


def hypinfo(v, hi):
    if v not in UNITS:
        C = c.build(*v)
        UNITS[v] = sorted({ch for p in C for pa in p['paras'] for l in pa for w in l for ch in w})
    return c.random_hyp(random.Random(hi), UNITS[v])


out = {}
clean = {}
for v, r in sorted(R.items()):
    nm = v[0]
    ok = [h for h in r['hyps'] if h['hi'] != 7 and (lambda H: H['a'] == 'last' and H['b'] == 'first')(hypinfo(v, h['hi']))]
    z = np.array([h['None'][2] for h in ok]); z0 = np.array([h['0'][2] for h in ok]); z1 = np.array([h['1'][2] for h in ok])
    clean[v] = {}
    for h in ok:
        H = hypinfo(v, h['hi'])
        key = (H['ea'], H['sb'], tuple(sorted(H['cmap'].items())))
        clean[v][key] = h
    out[str(v)] = dict(npara=r['npara'], n_hyp=len(ok), z_mean=round(float(z.mean()), 3), z_sd=round(float(z.std()), 3),
                       z_max=round(float(z.max()), 2), frac_gt2=round(float((z > 2).mean()), 4),
                       half_corr=round(float(np.corrcoef(z0, z1)[0, 1]), 3), Jword_z=round(r['Jword'][2], 2))


def survivors(real, nulls, held):
    if real not in clean: return None
    thr = max([3.0] + [float(np.quantile([h['None'][2] for h in clean[n].values()], 0.99)) for n in nulls if n in clean])
    S = [hi for hi, h in clean[real].items() if h['0'][2] > thr]  # hi = hypothesis key
    rep = [hi for hi in S if clean[real][hi]['1'][2] > 2 and all(clean[hv][hi]['None'][2] > 2 for hv in held if hv in clean and hi in clean[hv])]
    testable = [hi for hi in S if all(hv in clean and hi in clean[hv] for hv in held)]
    return dict(threshold=round(thr, 2), survivors_even=len(S), heldout_testable=len(testable), replicate_odd_and_heldout=len(rep),
                hyps=[(k[0], k[1], len(set(dict(k[2]).values())), clean[real][k]['0'][2], clean[real][k]['1'][2]) for k in rep[:20]])


out['surv_ZL'] = survivors(('ZL', 'real', 0), [('ZL', 'cross', 1), ('ZL', 'cross', 2), ('ZL', 'F5', 1)], [('IT2a', 'real', 0)])
out['surv_PLANT'] = survivors(('ZL', 'plant', 1), [('ZL', 'cross', 1), ('ZL', 'cross', 2), ('ZL', 'F5', 1)], [])
out['surv_IT'] = survivors(('IT', 'real', 0), [('IT', 'cross', 1)], [])
out['surv_LA'] = survivors(('LA', 'real', 0), [('LA', 'cross', 1)], [])
# paired comparison real vs null on the same hypotheses
for real, null in ((('ZL', 'real', 0), ('ZL', 'cross', 1)), (('ZL', 'real', 0), ('ZL', 'F5', 1)),
                   (('IT', 'real', 0), ('IT', 'cross', 1)), (('LA', 'real', 0), ('LA', 'cross', 1)),
                   (('ZL', 'plant', 1), ('ZL', 'cross', 1)), (('IT2a', 'real', 0), ('IT2a', 'cross', 1)),
                   (('ZL', 'cross', 2), ('ZL', 'cross', 1))):
    if real in clean and null in clean:
        ks = sorted(set(clean[real]) & set(clean[null]))
        d = np.array([clean[real][k]['None'][2] - clean[null][k]['None'][2] for k in ks])
        out['paired %s-%s' % (real, null)] = dict(mean_diff=round(float(d.mean()), 3), frac_pos=round(float((d > 0).mean()), 3), n=len(ks))
for k, v in out.items(): print(k, v)
json.dump(out, open(os.path.join(L.CK, 'c3_report.json'), 'w'))
