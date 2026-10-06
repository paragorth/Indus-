"""v71 report: genre classification from order-dependence fingerprints.
usage: python3 v71_report.py c1 [feature-set]
feature sets: all (default), noS0 (drop glyph-scale damage), coarse (scale|ALL only)."""
import os, sys, json, glob, random
import numpy as np
from collections import defaultdict, Counter
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import v71_lib as L
from sklearn.linear_model import LogisticRegression

CYC = sys.argv[1] if len(sys.argv) > 1 else 'c1'
FSET = sys.argv[2] if len(sys.argv) > 2 else 'all'
GROUP = {'apicius_lat': 'apicius', 'apicius_eng': 'apicius', 'apicius_index': 'apicius', 'celsus_lat': 'celsus', 'celsus_eng': 'celsus',
         'pliny_lat': 'pliny', 'pliny_index': 'pliny', 'culpeper': 'culpeper', 'culpeper_index': 'culpeper',
         'konrad_plants': 'konrad', 'konrad_other': 'konrad', 'forme_of_cury': 'cury', 'cury_glossary': 'cury'}
CLASSES = ['procedure', 'entry', 'list', 'narrative']

def load(cyc=CYC):
    D = []
    for f in sorted(glob.glob(os.path.join(L.CK, cyc, '*.json'))):
        d = json.load(open(f)); D.append(d)
    return D

def features(D, fset):
    keys = sorted(set(k for d in D for k in d['fp']))
    if fset == 'noS0': keys = [k for k in keys if not k.startswith('S0|')]
    if fset == 'coarse': keys = [k for k in keys if k.endswith('|ALL') or k.endswith('|ALLrel')]
    X = np.array([[d['fp'].get(k, 0.0) for k in keys] for d in D])
    return keys, X

def fit(Xtr, ytr, C=0.3):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    clf = LogisticRegression(C=C, max_iter=3000, class_weight='balanced')
    clf.fit((Xtr - mu) / sd, ytr)
    cent = {c: ((Xtr[ytr == c] - mu) / sd).mean(0) for c in set(ytr)}
    return mu, sd, clf, cent

def dist_nearest(x, mu, sd, cent):
    z = (x - mu) / sd
    ds = {c: float(np.sqrt(((z - v) ** 2).mean())) for c, v in cent.items()}
    c = min(ds, key=ds.get); return c, ds[c], ds

def main():
    D = load(); keys, X = features(D, FSET)
    meta = [d['meta'] for d in D]
    ref = [i for i, m in enumerate(meta) if m['kind'] == 'ref' and m['coarse'] in CLASSES]
    y = np.array([meta[i]['coarse'] for i in ref]); Xr = X[ref]
    grp = np.array([GROUP.get(meta[i]['text'], meta[i]['text']) for i in ref])
    texts = np.array([meta[i]['text'] for i in ref])
    out = []
    P = lambda s: (print(s), out.append(s))
    P('cycle %s, feature set %s: %d features, %d reference chunks from %d texts' % (CYC, FSET, len(keys), len(ref), len(set(texts))))
    P('chunks per class: %s' % dict(Counter(y)))
    # leave-one-group-out
    pred = np.empty(len(ref), dtype=object); post = np.zeros((len(ref), len(CLASSES))); dself = np.zeros(len(ref))
    for g in sorted(set(grp)):
        te = grp == g; tr = ~te
        mu, sd, clf, cent = fit(Xr[tr], y[tr])
        pr = clf.predict_proba((Xr[te] - mu) / sd)
        cl = list(clf.classes_)
        post[te] = np.array([[pr[j][cl.index(c)] if c in cl else 0 for c in CLASSES] for j in range(pr.shape[0])])
        pred[te] = [CLASSES[k] for k in post[te].argmax(1)]
        for j in np.where(te)[0]:
            c, dd, ds = dist_nearest(Xr[j], mu, sd, cent); dself[j] = ds.get(y[j], dd)
    acc = (pred == y).mean()
    # per text majority (mean posterior)
    tacc = []
    for t in sorted(set(texts)):
        m = texts == t; pc = CLASSES[post[m].mean(0).argmax()]
        tacc.append((t, y[m][0], pc, round(float(post[m].mean(0).max()), 2)))
    tac = np.mean([a[1] == a[2] for a in tacc])
    # balanced accuracy
    bacc = np.mean([(pred[y == c] == c).mean() for c in CLASSES if (y == c).any()])
    # permutation null: shuffle class labels across texts (keeping each text's chunks together)
    rng = random.Random(5); nulls = []
    ut = sorted(set(texts)); lab = {t: y[texts == t][0] for t in ut}
    for it in range(100):
        perm = list(lab.values()); rng.shuffle(perm); pl = dict(zip(ut, perm))
        yp = np.array([pl[t] for t in texts]); pr_ = np.empty(len(ref), dtype=object)
        for g in sorted(set(grp)):
            te = grp == g; tr = ~te
            if len(set(yp[tr])) < 2: pr_[te] = yp[tr][0]; continue
            mu, sd, clf, _ = fit(Xr[tr], yp[tr]); pr_[te] = clf.predict((Xr[te] - mu) / sd)
        nulls.append(np.mean([(pr_[yp == c] == c).mean() for c in CLASSES if (yp == c).any()]))
    pv = (1 + sum(n >= bacc for n in nulls)) / (1 + len(nulls))
    P('HELD-OUT (leave-one-source-out) chunk accuracy %.3f, balanced %.3f (label-permutation null mean %.3f, 95th %.3f, p %.3f); per-text majority %.3f' %
      (acc, bacc, np.mean(nulls), np.percentile(nulls, 95), pv, tac))
    for c in CLASSES:
        m = y == c; P('  %-9s n=%3d recall %.2f  predicted as %s' % (c, m.sum(), (pred[m] == c).mean(), dict(Counter(pred[m]))))
    P('  per text: ' + '; '.join('%s(%s)->%s %.2f' % a for a in tacc))
    thr = float(np.percentile(dself, 95))
    P('held-out real chunks: distance to own-genre centroid median %.3f, 95th pct %.3f (= "lands on a genre" threshold)' % (np.median(dself), thr))
    # final model on all references
    mu, sd, clf, cent = fit(Xr, y)
    def place(i):
        pr = clf.predict_proba(((X[i] - mu) / sd)[None])[0]; cl = list(clf.classes_)
        pp = {c: float(pr[cl.index(c)]) for c in CLASSES}
        c, dd, ds = dist_nearest(X[i], mu, sd, cent)
        return pp, c, dd
    P('GENERATOR NULLS (must not land on a genre):')
    gl = 0; gn = 0
    for i, m in enumerate(meta):
        if m['kind'] != 'gen': continue
        pp, c, dd = place(i); gn += 1
        lands = dd <= thr and max(pp.values()) >= 0.6; gl += lands
        P('  %-34s post %s nearest %s dist %.3f %s' % (m['name'], ' '.join('%s %.2f' % (k[:4], v) for k, v in pp.items()), c, dd, 'LANDS' if lands else 'off'))
    P('  generators landing: %d of %d' % (gl, gn))
    P('VOYNICH:')
    by = defaultdict(list)
    for i, m in enumerate(meta):
        if m['kind'] != 'voy': continue
        pp, c, dd = place(i); by[(m['tr'], m['sec'])].append((pp, c, dd))
        P('  %-30s post %s nearest %s dist %.3f %s' % (m['name'], ' '.join('%s %.2f' % (k[:4], v) for k, v in pp.items()), c, dd, 'in-genre' if dd <= thr else 'off-genre'))
    P('VOYNICH by section (mean posterior; distance; ZL vs IT2a):')
    for (tr, sec), L_ in sorted(by.items()):
        mp = {c: np.mean([x[0][c] for x in L_]) for c in CLASSES}
        P('  %-5s %-8s %s | top %s | dist %.3f (thr %.3f)' % (tr, sec, ' '.join('%s %.2f' % (k[:4], v) for k, v in mp.items()), max(mp, key=mp.get), np.mean([x[2] for x in L_]), thr))
    # scale profile table (mean ALL damage, bits/token) by class
    P('SCALE PROFILE (mean damage, bits/token over all model types; ALLrel = damage/gain):')
    rows = defaultdict(list)
    for i, m in enumerate(meta):
        lab = m.get('coarse') if m['kind'] == 'ref' else ('gen_' + m['gen'] if m['kind'] == 'gen' else 'V_' + m['sec'] + ('' if m['tr'] == 'ZL3b' else '_IT'))
        rows[lab].append(D[i]['fp'])
    for lab in sorted(rows):
        fs = rows[lab]
        P('  %-14s n=%2d ' % (lab, len(fs)) + ' '.join('%s %.3f/%.2f' % (s, np.mean([f['%s|ALL' % s] for f in fs]), np.mean([f['%s|ALLrel' % s] for f in fs])) for s in L.SCALES))
    # most discriminating features
    F = []
    for j, k in enumerate(keys):
        g = [Xr[y == c, j] for c in CLASSES]
        bw = np.var([v.mean() for v in g]); ww = np.mean([v.var() for v in g]) + 1e-12
        F.append((bw / ww, k))
    F.sort(reverse=True)
    P('top features (between/within variance): ' + ', '.join('%s %.2f' % (k, f) for f, k in F[:12]))
    json.dump({'lines': out}, open(os.path.join(L.CK, 'report_%s_%s.json' % (CYC, FSET)), 'w'))

if __name__ == '__main__':
    main()
