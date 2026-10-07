"""v96 cycle 2 report: ABC regression of the topic dial (tau) and key presence on the twin-relative summaries.
usage: v96_c2rep.py cv       -> leave-one-text-out CV + fresh / generator targets, writes frozen spec (sha256)
       v96_c2rep.py voynich  -> applies the frozen spec to the Voynich targets"""
import os, sys, json, glob, hashlib
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v96_lib as L

FEATS = ['dC', 'dR', 'dW', 'dF', 'conf', 'loc']
FZ = os.path.join(L.DATA, 'v96_frozen_c2.json')


def sims():
    S = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(L.CK, 'c2_sim_*.json')))]
    X = np.array([[s['sm'][f] for f in FEATS] for s in S]); y = np.array([s['tau'] for s in S])
    k = np.array([s['key'] for s in S], float); g = np.array([L.TEXTS.index(s['X']) for s in S])
    return X, y, k, g, S


def fit_predict(Xtr, ytr, Xte, kn=25):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    A = (Xtr - mu) / sd; B = (Xte - mu) / sd
    D = ((B[:, None, :] - A[None, :, :]) ** 2).sum(-1)
    nn = np.argsort(D, 1)[:, :kn]
    knn = ytr[nn].mean(1)
    P = lambda Z: np.hstack([np.ones((len(Z), 1)), Z, Z ** 2])
    Pa = P(A); w = np.linalg.solve(Pa.T @ Pa + 1.0 * np.eye(Pa.shape[1]), Pa.T @ ytr)
    rid = np.clip(P(B) @ w, -0.2, 1.2)
    md = np.sqrt(np.sort(D, 1)[:, :kn].mean(1))
    return knn, rid, md


def target(n):
    p = os.path.join(L.CK, 'c2_tgt_%s.json' % n)
    return json.load(open(p))['sm'] if os.path.exists(p) else None


def cv():
    X, y, k, g, S = sims()
    yt = np.zeros(len(y)); yr = np.zeros(len(y)); kt = np.zeros(len(y))
    for t in range(len(L.TEXTS)):
        tr, te = g != t, g == t
        yt[te], yr[te], _ = fit_predict(X[tr], y[tr], X[te])
        kt[te], _, _ = fit_predict(X[tr], k[tr], X[te])
    r_knn = np.corrcoef(yt, y)[0, 1]; r_rid = np.corrcoef(yr, y)[0, 1]
    pos, neg = kt[k == 1], kt[k == 0]
    auc = float(np.mean([[a > b for b in neg] for a in pos]))
    # per-feature correlation with tau, among key-on and key-off books
    fc = {f: [float(np.corrcoef(X[k == kk, i], y[k == kk])[0, 1]) for kk in (0, 1)] for i, f in enumerate(FEATS)}
    print('n sims', len(y), 'LOTO r(tau) knn %.3f ridge %.3f; key AUC %.3f' % (r_knn, r_rid, auc))
    print('feature r with tau [key off, key on]:', {f: [round(a, 2) for a in v] for f, v in fc.items()})
    # calibration of tau_hat in bins
    for lo in np.arange(0, 1, 0.2):
        m = (y >= lo) & (y < lo + 0.2); print('  tau %.1f-%.1f: knn %.2f ridge %.2f (n %d)' % (lo, lo + 0.2, yt[m].mean(), yr[m].mean(), m.sum()))
    tg = {}
    for n in L.FRESH_W1 + [f + '_DEALT' for f in L.FRESH_W1] + L.GENS:
        s = target(n)
        if s is None: continue
        x = np.array([[s[f] for f in FEATS]])
        a, b, md = fit_predict(X, y, x); kk, _, _ = fit_predict(X, k, x)
        tg[n] = dict(tau_knn=float(a[0]), tau_ridge=float(b[0]), key=float(kk[0]), nn_dist=float(md[0]), sm=s)
        print('%-16s tau knn %.2f ridge %.2f key %.2f nn-dist %.2f  %s' % (n, a[0], b[0], kk[0], md[0], {f: round(s[f], 3) for f in FEATS}))
    _, _, mds = fit_predict(X, y, X[:200])
    fz = dict(feats=FEATS, kn=25, ridge=1.0, n_sims=len(y), loto_r_knn=r_knn, loto_r_ridge=r_rid, key_auc=auc,
              nn_dist_p95=float(np.percentile(mds, 95)), targets=tg,
              predictions=['R1 (W2 only, no topic): Voynich tau (knn and ridge) < 0.35 in ZL3b, IT2a and GC2a, with key >= 0.7',
                           'R2 (topic under the key): Voynich tau > 0.65 in all three',
                           'R3 (method): fresh W1 targets (real order) tau > 0.6 and their dealt twins < 0.4 (checked before Voynich)',
                           'Out-of-cloud: a Voynich nn-dist above the sims 95th percentile makes any tau reading extrapolation'])
    s = json.dumps(fz, sort_keys=True, default=str); open(FZ, 'w').write(s)
    hx = hashlib.sha256(s.encode()).hexdigest(); open(FZ.replace('.json', '.sha256'), 'w').write(hx + '  v96_frozen_c2.json\n')
    print('frozen sha256', hx)


def voynich():
    fz = json.load(open(FZ)); X, y, k, g, S = sims(); out = {}
    for n in L.VOY + [v + '_DEALT' for v in L.VOY]:
        s = target(n)
        if s is None: continue
        x = np.array([[s[f] for f in FEATS]])
        a, b, md = fit_predict(X, y, x); kk, _, _ = fit_predict(X, k, x)
        # where the Voynich sits on each feature among key-on sims
        pct = {f: float((X[k == 1, i] < s[f]).mean()) for i, f in enumerate(FEATS)}
        out[n] = dict(tau_knn=float(a[0]), tau_ridge=float(b[0]), key=float(kk[0]), nn_dist=float(md[0]), pct=pct, sm=s)
        print('%-12s tau knn %.2f ridge %.2f key %.2f nn-dist %.2f (sims p95 %.2f) %s pct %s' % (
            n, a[0], b[0], kk[0], md[0], fz['nn_dist_p95'], {f: round(s[f], 3) for f in FEATS}, {f: round(v, 2) for f, v in pct.items()}))
    json.dump(out, open(os.path.join(L.CK, 'c2_voynich.json'), 'w'))


if __name__ == '__main__':
    voynich() if sys.argv[1:] == ['voynich'] else cv()
