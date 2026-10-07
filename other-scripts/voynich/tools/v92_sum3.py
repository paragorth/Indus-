"""ABC summary for v92 cycle 3: closest 5% by standardised distance; posterior entity share; self-validation."""
import json, os, sys, numpy as np
CK = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'v92_ckpt')
KEYS = ['rec', 'conf', 'front', 'sim', 'nbr', 'nrec', 'slope', 'hap']


def post(sims, obs, q=0.05, keys=KEYS):
    S = np.array([[s['s'][k] for k in keys] for s in sims]); o = np.array([obs[k] for k in keys])
    sd = S.std(0) + 1e-9
    d = np.sqrt((((S - o) / sd) ** 2).sum(1))
    k = max(10, int(q * len(sims)))
    idx = np.argsort(d)[:k]
    sh = np.array([sims[i]['share'] for i in idx])
    # typical nearest distance of sims to each other (is obs inside the cloud?)
    P = [sims[i]['P'] for i in idx]
    extra = dict(seed=float(np.mean([p['sd_p'] * (p['sd_n'] > 0) for p in P])), lx=float(np.mean([p['lx'] for p in P])),
                 nov=float(np.mean([p['nov'] for p in P])), ent=float(np.mean([p['r_e'] * (p['k_e'] > 0) for p in P])))
    return dict(extra=extra, mean=float(sh.mean()), lo=float(np.percentile(sh, 10)), hi=float(np.percentile(sh, 90)),
                p0=float((sh < 0.002).mean()), dmin=float(d[idx[0]]), dk=float(d[idx[-1]]), idx=idx)


def selfval(sims, n=60, seed=0):
    rng = np.random.default_rng(seed); tr = []; pm = []
    for i in rng.choice(len(sims), size=n, replace=False):
        rest = [s for j, s in enumerate(sims) if j != i]
        tr.append(sims[i]['share']); pm.append(post(rest, sims[i]['s'])['mean'])
    return float(np.corrcoef(tr, pm)[0, 1])


if __name__ == '__main__':
    names = sys.argv[1:] or sorted(f[3:-5] for f in os.listdir(CK) if f.startswith('c3_') and f.endswith('.json'))
    prior = None
    for n in names:
        r = json.load(open(os.path.join(CK, 'c3_%s.json' % n)))
        sims = r['sims']
        if prior is None: prior = np.mean([s['share'] for s in sims])
        p = post(sims, r['obs'])
        print('%-10s n=%d obs %s | share post mean %.4f [%.4f, %.4f] P(<0.002) %.2f | dmin %.2f dk %.2f | selfval r %.2f' % (
            n, len(sims), {k: round(v, 3) for k, v in r['obs'].items()}, p['mean'], p['lo'], p['hi'], p['p0'], p['dmin'], p['dk'],
            selfval(sims)), {k: round(v, 3) for k, v in p['extra'].items()})
    print('prior mean share %.4f' % prior)
