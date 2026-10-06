"""v71 cycle 3 report: arrow-of-order profiles, genre classification from them, generators, Voynich."""
import os, sys, json, glob, random
os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np
from collections import defaultdict, Counter
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import v71_lib as L
import v71_report as R1
SC = os.environ.get("V71_SC", "W,L,Lm,P").split(",")

def main():
    D = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(L.CK, 'c3', '*.json')))]
    def vec(d):
        v = []
        for s in SC:
            a, b = d['arrow'].get(s), d['null'].get(s)
            v.append(np.mean(a['acc']) - np.mean(b['acc']) if a and b else 0.0)
        return np.array(v)
    def raw(d, s):
        a = d['arrow'].get(s); return np.mean(a['acc']) if a else np.nan
    out = []; P = lambda s: (print(s), out.append(s))
    groups = defaultdict(list)
    for d in D:
        m = d['meta']
        lab = (m['coarse'] if m['kind'] == 'ref' else 'gen_' + m['gen'] + ('_V' if m['src'].startswith('V') else '_ref') if m['kind'] == 'gen'
               else ('V_%s_%s' % (m['sec'], m['tr'][:2]) if m['kind'] == 'voy' else m['name']))
        groups[lab].append(d)
    P('ARROW EXCESS = held-out direction accuracy (mean of 10 configurations) minus the same on the unit-shuffled chunk; raw accuracy in brackets')
    for lab in sorted(groups):
        G = groups[lab]; V = np.array([vec(d) for d in G])
        P('  %-20s n %3d  ' % (lab, len(G)) + '  '.join('%s %+.3f+-%.3f [%.3f]' % (s, V[:, i].mean(), V[:, i].std() / np.sqrt(len(G)), np.nanmean([raw(d, s) for d in G])) for i, s in enumerate(SC)))
    # per-text table for references
    P('PER TEXT (excess W L Lm P):')
    bt = defaultdict(list)
    for d in D:
        if d['meta']['kind'] == 'ref': bt[(d['meta']['coarse'], d['meta']['text'])].append(vec(d))
    for (c, t), V in sorted(bt.items()):
        V = np.array(V); P('  %-9s %-16s ' % (c, t) + ' '.join('%+.3f' % x for x in V.mean(0)))
    # classification
    ref = [d for d in D if d['meta']['kind'] == 'ref']
    X = np.array([vec(d) for d in ref]); y = np.array([d['meta']['coarse'] for d in ref])
    texts = np.array([d['meta']['text'] for d in ref]); grp = np.array([R1.GROUP.get(t, t) for t in texts])
    pred = np.empty(len(ref), dtype=object); dself = np.zeros(len(ref))
    for g in sorted(set(grp)):
        te = grp == g; tr = ~te
        mu, sd, clf, cent = R1.fit(X[tr], y[tr], C=1.0)
        pred[te] = clf.predict((X[te] - mu) / sd)
        for j in np.where(te)[0]:
            c, dd, ds = R1.dist_nearest(X[j], mu, sd, cent); dself[j] = ds.get(y[j], dd)
    bacc = np.mean([(pred[y == c] == c).mean() for c in R1.CLASSES])
    rng = random.Random(3); ut = sorted(set(texts)); lab0 = {t: y[texts == t][0] for t in ut}; nulls = []
    for _ in range(200):
        pm = list(lab0.values()); rng.shuffle(pm); pl = dict(zip(ut, pm)); yp = np.array([pl[t] for t in texts]); pr = np.empty(len(ref), dtype=object)
        for g in sorted(set(grp)):
            te = grp == g; tr = ~te
            mu, sd, clf, _ = R1.fit(X[tr], yp[tr], C=1.0); pr[te] = clf.predict((X[te] - mu) / sd)
        nulls.append(np.mean([(pr[yp == c] == c).mean() for c in R1.CLASSES]))
    P('GENRE FROM ARROW PROFILE (leave-one-source-out): balanced accuracy %.3f (null mean %.3f, 95th %.3f, p %.3f); recall %s' % (
        bacc, np.mean(nulls), np.percentile(nulls, 95), (1 + sum(n >= bacc for n in nulls)) / 201, {c: round(float((pred[y == c] == c).mean()), 2) for c in R1.CLASSES}))
    thr = np.percentile(dself, 95)
    mu, sd, clf, cent = R1.fit(X, y, C=1.0)
    P('PLACEMENT (threshold %.3f):' % thr)
    for lab in sorted(groups):
        if lab in R1.CLASSES: continue
        G = groups[lab]; land = 0; posts = []
        for d in G:
            x = vec(d); pr = clf.predict_proba(((x - mu) / sd)[None])[0]; cl = list(clf.classes_)
            pp = {c: pr[cl.index(c)] for c in R1.CLASSES}; c, dd, _ = R1.dist_nearest(x, mu, sd, cent)
            land += dd <= thr and max(pp.values()) >= 0.6; posts.append([pp[c] for c in R1.CLASSES])
        mp = np.mean(posts, 0)
        P('  %-20s n %2d lands %2d  mean posterior %s' % (lab, len(G), land, ' '.join('%s %.2f' % (c[:4], v) for c, v in zip(R1.CLASSES, mp))))
    json.dump({'lines': out}, open(os.path.join(L.CK, 'report_c3.json'), 'w'))

if __name__ == '__main__':
    main()
