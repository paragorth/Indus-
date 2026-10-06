"""v78 cycle 2: thousands of random mechanism mixtures fitted to the doubles' fingerprint (ABC), held-out leaves.

For each corpus: base = discovery leaves (leaf parity 0) with every within-line double collapsed. N random mixtures
of the six operators (v78_lib.sample_params / apply: CHANCE, REDUP, LIST, DITTOG, TALLY, GEN; each on with p 0.5,
log-uniform rates) are applied to the base; the 18-feature fingerprint (v78_lib.FEATS) of the result is compared
with the corpus's OBSERVED discovery-leaf fingerprint (distance standardised by the simulations' MAD). The nearest
3% are accepted; the posterior share of the doubles written by each mechanism is the mean over accepted runs of the
labelled shares (BASE = doubles already in the base, none after collapsing, so in practice 0).
Held-out: the accepted mixtures are re-run on the held-out leaves' base and their distance to the held-out observed
fingerprint is compared with 200 prior mixtures on the same base (percentile of the accepted median distance);
a mixture family that predicts held-out doubles sits well below the prior.
Assignment = mechanism with the largest posterior share, where CHANCE and GEN are pooled as NO-RULE (no special
doubling act). Controls must be assigned: Malay/Tagalog/Hebrew -> REDUP, Plaoul -> DITTOG, planted list -> LIST,
planted tally -> TALLY, generators -> NO-RULE.
"""
import os, sys, random, pickle, json, math, time
import numpy as np
from multiprocessing import Pool
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import v78_lib as L

C = None


def _init():
    global C
    C = pickle.load(open(os.path.join(L.CK, 'corpora.pkl'), 'rb'))


_BASE = {}


def base(name, h):
    k = (name, h)
    if k not in _BASE:
        _BASE[k] = L.collapse(L.half(C[name]['pages'], h))
    return _BASE[k]


def sim(job):
    name, h, seed, P = job
    rng = random.Random(seed)
    if P is None: P = L.sample_params(rng)
    S, lab = L.apply(base(name, h), P, seed)
    sh, nd = L.shares(S, lab)
    return dict(name=name, h=h, seed=seed, P=P, f=L.fvec(L.feats(S)).tolist(), sh=sh, nd=nd)


def dist(F, obs, scale):
    return np.sqrt((((F - obs) / scale) ** 2).sum(1))


GROUP = {'CHANCE': 'NO-RULE', 'GEN': 'NO-RULE', 'BASE': 'NO-RULE', 'REDUP': 'REDUP', 'LIST': 'LIST',
         'DITTOG': 'DITTOG', 'TALLY': 'TALLY'}


def main(names, nsim, nhold=200, acc=0.03, tag='c2'):
    _init()
    out = {}
    pool = Pool(2, initializer=_init)
    for name in names:
        t0 = time.time()
        obs = {h: L.fvec(L.feats(L.half(C[name]['pages'], h))) for h in (0, 1)}
        jobs = [(name, 0, 780000 + i, None) for i in range(nsim)]
        R = pool.map(sim, jobs, chunksize=8)
        F = np.array([r['f'] for r in R])
        scale = np.median(np.abs(F - np.median(F, 0)), 0) * 1.4826 + 1e-3
        d = dist(F, obs[0], scale); k = max(10, int(acc * nsim)); idx = np.argsort(d)[:k]
        post = {m: float(np.mean([R[i]['sh'][m] for i in idx])) for m in L.MECHS + ['BASE']}
        grp = {}
        for m, v in post.items(): grp[GROUP[m]] = grp.get(GROUP[m], 0) + v
        on = {m: float(np.mean([m in R[i]['P'] for i in idx])) for m in L.MECHS}
        # per-feature fit of the accepted runs (z of observed within accepted)
        zf = {L.FEATS[j]: float((obs[0][j] - F[idx, j].mean()) / (F[idx, j].std() + 1e-9)) for j in range(len(L.FEATS))}
        # held-out
        hj = [(name, 1, 790000 + int(i), R[int(i)]['P']) for i in idx] + [(name, 1, 791000 + i, None) for i in range(nhold)]
        H = pool.map(sim, hj, chunksize=4)
        FH = np.array([r['f'] for r in H]); dh = dist(FH, obs[1], scale)
        da, dp = dh[:len(idx)], dh[len(idx):]
        pct = float((dp < np.median(da)).mean())
        out[name] = dict(kind=C[name]['kind'], post=post, group=grp, assigned=max(grp, key=grp.get), on=on,
                         dmin=float(d[idx].mean()), dprior=float(np.median(d)), zfeat=zf,
                         hold_med=float(np.median(da)), hold_prior_med=float(np.median(dp)), hold_pct=pct,
                         best=[R[i]['P'] for i in idx[:5]], nd_obs=None)
        pickle.dump(dict(R=R, H=H, obs={h: obs[h].tolist() for h in obs}, idx=idx.tolist(), scale=scale.tolist()),
                    open(os.path.join(L.CK, '%s_%s.pkl' % (tag, name)), 'wb'))
        print(name, C[name]['kind'], 'assigned', out[name]['assigned'], {k: round(v, 3) for k, v in grp.items()},
              'on', {k: round(v, 2) for k, v in on.items()}, 'd %.2f (prior %.2f) hold %.2f vs prior %.2f pct %.3f' %
              (out[name]['dmin'], out[name]['dprior'], out[name]['hold_med'], out[name]['hold_prior_med'], pct),
              '%.0fs' % (time.time() - t0), flush=True)
        json.dump(out, open(os.path.join(L.CK, '%s.json' % tag), 'w'), default=float, indent=1)
    pool.close()


if __name__ == '__main__':
    which = sys.argv[1]
    if which == 'voy':
        main(['VOY_ZL', 'VOY_IT'], 2500, tag='c2voy')
    else:
        main(['MS_1001', 'TL_MED', 'TL_NOLI', 'HE', 'PL_DITT', 'LIST_SYON', 'LIST_SIN', 'TALLY_ING',
              'GEN_SELFCIT', 'GEN_STACK', 'GEN_JUNC', 'GEN_MK2', 'GEN_SC10', 'LA_ISID', 'IT_BRUM'], 800, tag='c2ctl')
