#!/usr/bin/env python3
"""R-3 ABC: which simulated worlds grow corpora like each real corpus?
usage: r3_abc.py OUTTAG SIMTAG[,SIMTAG...] [--panel FULL|NONUM] [--frac 0.01]
Writes data/r3_ckpt/abc_OUTTAG.json."""
import sys, os, json, random, math
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from r3_lib import *

TARGETS = ['PE', 'LA', 'VMS', 'VMS_A', 'VMS_B', 'UR3', 'LB', 'LAT', 'ITA']
SHUF = ['PE', 'LA', 'VMS', 'UR3', 'LB', 'LAT']


def real_panels():
    f = os.path.join(CK, 'real_panels.json')
    if os.path.exists(f):
        return json.load(open(f))
    out = {}
    truth = {}
    for nm in TARGETS:
        C = load_real(nm)
        m, rows = panel_mean(C, 8, seed=11)
        out[nm] = m
        coms = set(); des = set()
        for d in C:
            for ws, q in d:
                for w in ws:
                    (coms if len(w) == 1 else des).add(w)
        truth[nm] = {'docs': len(C)}
        if nm in SHUF:
            for s in range(3):
                m2, _ = panel_mean(shuffle_corpus(C, seed=s), 4, seed=21 + s)
                out['%s_shuf%d' % (nm, s)] = m2
    json.dump(out, open(f, 'w'), indent=1)
    return out


def load_sims(tags):
    TH, S, H = [], [], []
    for t in tags:
        for l in open(os.path.join(CK, 'sims_%s.jsonl' % t)):
            r = json.loads(l)
            s = r['s']
            if any(s[k] is None or not np.isfinite(s[k]) for k in FIT_NAMES + HELD_NAMES):
                continue
            TH.append([r['th'][k] for k in PNAMES]); S.append([s[k] for k in FIT_NAMES]); H.append([s[k] for k in HELD_NAMES])
    return np.array(TH, float), np.array(S, float), np.array(H, float)


def logit_t(TH):
    lo = np.array([PRIOR[k][0] for k in PNAMES], float); hi = np.array([PRIOR[k][1] for k in PNAMES], float)
    z = (TH - lo) / (hi - lo)
    z = np.clip(z, 1e-3, 1 - 1e-3)
    return np.log(z / (1 - z)), lo, hi


def inv_t(Z, lo, hi):
    return lo + (hi - lo) / (1 + np.exp(-Z))


class ABC:
    def __init__(self, TH, S, H, cols, frac):
        self.TH, self.S, self.H = TH, S, H
        self.cols = cols
        X = S[:, cols]
        self.med = np.median(X, 0)
        self.mad = np.median(np.abs(X - self.med), 0) * 1.4826 + 1e-6
        self.X = (X - self.med) / self.mad
        self.k = max(50, int(frac * len(TH)))
        self.Z, self.lo, self.hi = logit_t(TH)

    def fit(self, s, exclude=None):
        x = (np.asarray(s)[self.cols] - self.med) / self.mad
        d = np.sqrt(((self.X - x) ** 2).sum(1))
        if exclude is not None:
            d[exclude] = np.inf
        idx = np.argsort(d)[:self.k]
        h = d[idx[-1]] + 1e-9
        w = 1 - (d[idx] / h) ** 2
        # local-linear regression adjustment (Beaumont 2002) in logit space
        Xa = self.X[idx] - x
        A = np.hstack([np.ones((len(idx), 1)), Xa])
        W = np.sqrt(w)[:, None]
        beta, *_ = np.linalg.lstsq(A * W, self.Z[idx] * W, rcond=None)
        Zadj = self.Z[idx] - Xa @ beta[1:]
        post = inv_t(Zadj, self.lo, self.hi)
        return {'idx': idx, 'w': w, 'post': post, 'dmin': float(d[idx[0]]), 'dmed': float(np.median(d[idx])),
                'raw': self.TH[idx]}


def wq(x, w, qs):
    o = np.argsort(x); x = x[o]; w = w[o]
    c = np.cumsum(w); c = c / c[-1]
    return [float(np.interp(q, c, x)) for q in qs]


def summarize(res):
    out = {}
    for j, k in enumerate(PNAMES):
        lo, hi = PRIOR[k][0], PRIOR[k][1]
        q = wq(res['post'][:, j], res['w'], [0.05, 0.25, 0.5, 0.75, 0.95])
        out[k] = {'q05': q[0], 'q25': q[1], 'med': q[2], 'q75': q[3], 'q95': q[4],
                  'contr': 1 - (q[3] - q[1]) / (0.5 * (hi - lo))}
    # derived: people-to-goods ratio
    d = res['post'][:, PNAMES.index('lg_people')] - res['post'][:, PNAMES.index('lg_goods')]
    q = wq(d, res['w'], [0.05, 0.5, 0.95])
    out['people_over_goods'] = {'q05': q[0], 'med': q[1], 'q95': q[2]}
    return out


def main():
    outtag = sys.argv[1]; tags = sys.argv[2].split(',')
    pn = 'FULL'; frac = 0.01
    a = sys.argv[3:]
    if '--panel' in a:
        pn = a[a.index('--panel') + 1]
    if '--frac' in a:
        frac = float(a[a.index('--frac') + 1])
    TH, S, H = load_sims(tags)
    cols = [i for i, k in enumerate(FIT_NAMES) if pn == 'FULL' or k not in NUM_STATS]
    abc = ABC(TH, S, H, cols, frac)
    R = real_panels()
    rng = np.random.default_rng(5)
    # pseudo-data control: held-out simulations (parameter recovery, calibration of the distance and of predictions)
    ps = rng.choice(len(TH), 300, replace=False)
    rec = {k: [] for k in PNAMES}; dmins = []; cover = {h: [] for h in HELD_NAMES}
    for i in ps:
        r = abc.fit(S[i], exclude=[i])
        dmins.append(r['dmin'])
        sm = summarize(r)
        for j, k in enumerate(PNAMES):
            rec[k].append((TH[i, j], sm[k]['med']))
        for hj, hn in enumerate(HELD_NAMES):
            lo_, hi_ = wq(H[r['idx'], hj], r['w'], [0.05, 0.95])
            cover[hn].append(lo_ <= H[i, hj] <= hi_)
    recov = {k: float(np.corrcoef(np.array(v).T)[0, 1]) for k, v in rec.items()}
    out = {'panel': pn, 'n_sims': len(TH), 'k': abc.k, 'recovery_r': recov,
           'pseudo_dmin_q50_q95': [float(np.quantile(dmins, 0.5)), float(np.quantile(dmins, 0.95))],
           'pseudo_cover90': {h: float(np.mean(v)) for h, v in cover.items()}, 'targets': {}}
    for nm, s in R.items():
        sv = [s[k] for k in FIT_NAMES]
        r = abc.fit(sv)
        sm = summarize(r)
        pred = {}
        for hj, hn in enumerate(HELD_NAMES):
            q = wq(H[r['idx'], hj], r['w'], [0.05, 0.5, 0.95])
            pctl = float(np.sum(r['w'] * (H[r['idx'], hj] < s[hn])) / np.sum(r['w']))
            pred[hn] = {'q05': q[0], 'med': q[1], 'q95': q[2], 'real': s[hn], 'pctl': pctl}
        dpct = float(np.mean(np.array(dmins) < r['dmin']))
        ppc = {}
        for j, kn in enumerate(FIT_NAMES):
            q = wq(S[r['idx'], j], r['w'], [0.05, 0.5, 0.95])
            ppc[kn] = (s[kn] - q[1]) / ((q[2] - q[0]) / 3.29 + 1e-9)
        out['targets'][nm] = {'dmin': r['dmin'], 'dmed': r['dmed'], 'dmin_pctl_vs_pseudo': dpct,
                              'post': sm, 'pred': pred, 'ppc': ppc,
                              'acc': [dict(zip(PNAMES, map(float, row))) for row in r['raw'][:100]]}
    json.dump(out, open(os.path.join(CK, 'abc_%s.json' % outtag), 'w'), indent=1)
    # compact print
    print('panel', pn, 'sims', len(TH), 'k', abc.k)
    print('recovery r:', ' '.join('%s %.2f' % (k, v) for k, v in recov.items()))
    print('pseudo dmin q50/q95', out['pseudo_dmin_q50_q95'], 'cover90', out['pseudo_cover90'])
    keyp = ['p_ledger', 'lg_people', 'lg_goods', 'p_num', 'topical', 'text_len', 'p_func', 'lg_syl', 'wlen',
            'lg_logo', 'n_gen', 'bottleneck', 'p_var', 'n_scribes', 'shared', 'secrecy', 'p_div']
    for nm, t in out['targets'].items():
        p = t['post']
        print('%-10s dmin %.2f (pctl %.2f) ' % (nm, t['dmin'], t['dmin_pctl_vs_pseudo']) +
              ' '.join('%s %.2f[%.2f,%.2f]' % (k, p[k]['med'], p[k]['q05'], p[k]['q95']) for k in keyp) +
              ' P/G %.2f' % p['people_over_goods']['med'] +
              ' | ' + ' '.join('%s pred %.3f[%.3f,%.3f] real %.3f pctl %.2f' % (h, v['med'], v['q05'], v['q95'], v['real'], v['pctl'])
                               for h, v in t['pred'].items()))


if __name__ == '__main__':
    main()
