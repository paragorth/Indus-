"""Line-end pull index over cycle-1 checkpoints: where does the line-final S/B share fall on the
axis between the trigger context (C) and elsewhere (N), relative to the blind (overall) share?"""
import glob, json, math, os, sys
import numpy as np
import v61_lib as L
lg = lambda p: math.log(p / (1 - p))
def index(recs, top=100, nmin=20, kind='both'):
    out = []
    for r in recs[:400]:
        x = L.line_end_llr(r, kind)
        if x['n'] < nmin or r['z_test'] < 3:
            continue
        span = lg(x['pS_C']) - lg(x['pS_N'])
        if span < 0.5:
            continue
        t = (lg(x['pS_C']) - lg(x['pS_fin'])) / span; ta = (lg(x['pS_C']) - lg(x['pS_all'])) / span
        # binomial z of the final share against the blind share
        p = x['pS_all']; zf = (x['pS_fin'] - p) / math.sqrt(p * (1 - p) / x['n'])
        out.append((t - ta, zf))
        if len(out) >= top:
            break
    E = np.array([o[0] for o in out]); Z = np.array([o[1] for o in out])
    return {'n': len(out), 'med_absE': float(np.median(np.abs(E))) if len(E) else None,
            'frac_decisive': float(np.mean(np.abs(E) > 0.5)) if len(E) else None,
            'med_absZ': float(np.median(np.abs(Z))) if len(Z) else None,
            'frac_beyond': float(np.mean([(e > 0 and t) for e, t in []])) if False else None}
if __name__ == '__main__':
    for fn in sorted(glob.glob(os.path.join(L.CK, sys.argv[1] if len(sys.argv) > 1 else 'c1_*.json'))):
        d = json.load(open(fn))
        if 'recs' not in d: continue
        print(os.path.basename(fn), ' '.join('%s: n=%d |E|=%.2f dec=%.2f' % ((k,) + tuple(index(d['recs'], kind=k, nmin=15)[q] for q in ('n', 'med_absE', 'frac_decisive'))) for k in ('finL', 'finP')))
