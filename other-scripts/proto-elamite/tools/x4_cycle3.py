"""X-4 cycle 3: the Voynich v31 generator-vs-language classifier, retrained with list-genre languages.

Features: the 32 v31 alphabet-free features per 100-word sample (x4_feats.py for X-4 corpora; the v31
reference samples from voynich/data/v31_ckpt/feats_N100.json).
Reference sets (training classes LANG, GEN, GIBB, MAGIC, INVENT):
  REF0  v31 as published (prose languages only in LANG)
  REF1  REF0 + list-genre languages in LANG (Linear B W/E, Ur III W/E, proto-cuneiform E) + syllabic prose
        (Iliad in a CV syllabary) + the X-4 prose re-encodings
  REF2  REF1 + generators fitted to the known list corpora (TRI, SLOT, CPV, WBG of LB, LB_E, UR3, UR3_E, PC_E)
        in GEN
  REF2-<fam>  REF2 with one list-language family left out, which is then scored (honest score for LB, UR3, PC)
Massive random guessing as in v31: NH random classifiers (3-12 random features x logistic / kNN / LDA / small RF);
selection by grouped 4-fold CV inside half A of the corpora (balanced corpus-level accuracy, top 10%),
re-test on half B (>= 0.5); survivors vote on test objects (argmax of the mean posterior over a corpus' samples).
Null: corpus labels permuted, same threshold (survivor rate). Planted: generators fitted to LA and PE must vote GEN.
"""
import os, sys, json, random, time
import numpy as np
from collections import Counter, defaultdict
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import x4_lib as X

CLASSES = ['LANG', 'MAGIC', 'INVENT', 'GIBB', 'GEN']
NH = int(os.environ.get('NH', 1000))
LISTL = ['LB', 'LB_E', 'UR3', 'UR3_E', 'PC_E']
FAM = {'LB_E': 'LB', 'UR3_E': 'UR3', 'LA_E': 'LA'}


def model(kind, seed):
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.ensemble import RandomForestClassifier
    if kind == 'lr': return LogisticRegression(C=0.5, max_iter=2000, class_weight='balanced')
    if kind == 'knn': return KNeighborsClassifier(7)
    if kind == 'lda': return LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
    return RandomForestClassifier(60, max_depth=6, class_weight='balanced', random_state=seed, n_jobs=1)


def fitpred(kind, seed, Xa, ya, Xb):
    mu = Xa.mean(0); sd = Xa.std(0); sd[sd == 0] = 1
    m = model(kind, seed).fit((Xa - mu) / sd, ya)
    return m.predict_proba((Xb - mu) / sd), list(m.classes_)


def corp_level(p, cl, corp):
    return {c: cl[int(np.argmax(p[corp == c].mean(0)))] for c in set(corp)}


def bal(pred, truth):
    per = defaultdict(list)
    for c, p in pred.items(): per[truth[c]].append(p == truth[c])
    return float(np.mean([np.mean(v) for v in per.values()]))


def load_all():
    keys = None; recs = []
    for r in json.load(open(os.path.join(X.OS, 'voynich', 'data', 'v31_ckpt', 'feats_N100.json'))):
        recs.append(('v31:' + r['corpus'], r['cls'], r['F']))
    for r in X.load('feats.json'):
        recs.append((f"x4:{r['corpus']}:{r['cond']}", None, r['F']))
    keys = sorted(recs[0][2])
    M = np.array([[r[2][k] for k in keys] for r in recs], float)
    M = np.nan_to_num(M, nan=0.0, posinf=0.0, neginf=0.0)
    return recs, keys, M


def build(recs, M, ref, leave=None):
    """returns training mask, labels, corpus ids, group ids; test corpora dict name -> mask"""
    n = len(recs); y = np.array([''] * n, object); grp = np.array([''] * n, object)
    corp = np.array([r[0] for r in recs], object)
    for i, (c, cls, _) in enumerate(recs):
        if c.startswith('v31:') and cls in CLASSES:
            y[i] = cls; grp[i] = c
        if not c.startswith('x4:'): continue
        _, name, cond = c.split(':')
        fam = FAM.get(name, name)
        if ref >= 1 and cond == 'real' and (name in LISTL or name in X.PROSE):
            y[i] = 'LANG'; grp[i] = 'x4:' + fam
        if ref >= 2 and name in LISTL and cond.endswith('_a'):
            y[i] = 'GEN'; grp[i] = f'x4:{fam}:{cond}'
        if leave and fam == leave: y[i] = ''
    tests = {}
    for c in sorted(set(corp)):
        if c.startswith('x4:') and (c.endswith(':real') or c.split(':')[1] in ('LA', 'LA_E', 'PE_E', 'VOY')) \
                and c.endswith(('real', '_a')):
            tests[c] = corp == c
    tests['v31:V_ZL'] = corp == 'v31:V_ZL'; tests['v31:V_IT'] = corp == 'v31:V_IT'
    return y, grp, tests


def hypothesis(a):
    h, cols, kind, seed, D = a
    Xm, y, grp, half, truth, tests = D['X'], D['y'], D['grp'], D['half'], D['truth'], D['tests']
    Xc = Xm[:, cols]
    T = y != ''
    A = T & (half == 0); B = T & (half == 1)
    ga = sorted(set(grp[A])); rng = np.random.RandomState(seed % (2 ** 31))
    fold = dict(zip(ga, rng.permutation(len(ga)) % 4))
    pred = {}
    for f in range(4):
        te = A & np.array([fold.get(g, -1) == f for g in grp]); tr = A & ~te
        if len(set(y[tr])) < 5 or te.sum() == 0: return None
        p, cl = fitpred(kind, seed, Xc[tr], y[tr], Xc[te]); pred.update(corp_level(p, cl, grp[te]))
    sA = bal(pred, truth)
    p, cl = fitpred(kind, seed, Xc[A], y[A], Xc[B]); sB = bal(corp_level(p, cl, grp[B]), truth)
    out = {'h': h, 'cols': cols, 'kind': kind, 'sA': sA, 'sB': sB}
    if sB < 0.5: return out
    names = list(tests); rows = np.vstack([Xc[tests[k]] for k in names])
    p, cl = fitpred(kind, seed, Xc[T], y[T], rows)
    votes, post, i = {}, {}, 0
    for k in names:
        n = int(tests[k].sum()); m = p[i:i + n].mean(0); i += n
        votes[k] = cl[int(np.argmax(m))]; post[k] = dict(zip(cl, m.round(3).tolist()))
    out['votes'] = votes; out['post'] = post
    return out


def run_ref(recs, keys, M, ref, leave=None, permute=False, nh=NH, seed0=1):
    y, grp, tests = build(recs, M, ref, leave)
    T = y != ''
    groups = sorted(set(grp[T])); truth = {g: y[grp == g][0] for g in groups}
    if permute:
        r = random.Random(seed0); labs = [truth[g] for g in groups]; r.shuffle(labs)
        truth = dict(zip(groups, labs))
        y = np.array([truth.get(g, '') if t else '' for g, t in zip(grp, T)], object)
    rng = np.random.RandomState(7); half_of = {}
    for k in CLASSES:
        gs = [g for g in groups if truth[g] == k]; rng.shuffle(gs)
        for i, g in enumerate(gs): half_of[g] = i % 2
    half = np.array([half_of.get(g, -1) for g in grp])
    D = {'X': M, 'y': y, 'grp': grp, 'half': half, 'truth': truth, 'tests': tests}
    r = random.Random(seed0); jobs = []
    for h in range(nh):
        k = r.randint(3, 12)
        jobs.append((h, sorted(r.sample(range(len(keys)), k)), r.choice(['lr', 'lr', 'knn', 'lda', 'rf']),
                     seed0 * 100000 + h, D))
    with Pool(int(os.environ.get('W', '2'))) as p:
        R = [x for x in p.imap_unordered(hypothesis, jobs, chunksize=10) if x]
    sA = np.array([x['sA'] for x in R]); thr = float(np.percentile(sA, 90))
    surv = [x for x in R if x['sA'] >= thr and x['sB'] >= 0.5]
    summ = {}
    for k in tests:
        vs = Counter(x['votes'][k] for x in surv if 'votes' in x)
        tot = sum(vs.values()) or 1
        summ[k] = {c: round(vs.get(c, 0) / tot, 3) for c in CLASSES}
        summ[k]['n'] = tot
    meanpost = {k: {c: float(np.mean([x['post'][k].get(c, 0) for x in surv if 'post' in x] or [0])) for c in CLASSES}
                for k in tests}
    return {'ref': ref, 'leave': leave, 'permute': permute, 'n': len(R), 'thr': thr, 'nsurv': len(surv),
            'ntrain_groups': len(groups), 'votes': summ, 'meanpost': meanpost,
            'sB_mean': float(np.mean([x['sB'] for x in surv])) if surv else None}


def main():
    recs, keys, M = load_all()
    out = X.load('c3.json') or {}
    plan = [('REF0', 0, None, False), ('REF1', 1, None, False), ('REF2', 2, None, False),
            ('REF2-LB', 2, 'LB', False), ('REF2-UR3', 2, 'UR3', False), ('REF2-PC_E', 2, 'PC_E', False),
            ('NULL0', 0, None, True), ('NULL2', 2, None, True)]
    for tag, ref, leave, perm in plan:
        if tag in out: continue
        t0 = time.time()
        res = run_ref(recs, keys, M, ref, leave, perm, nh=NH if not perm else NH // 2)
        out[tag] = res; X.save('c3.json', out)
        v = res['votes']
        print(tag, 'surv', res['nsurv'], '/', res['n'], 'thr', round(res['thr'], 2), round(time.time() - t0), 's', flush=True)
        for k in sorted(v):
            if v[k]['n']: print('   ', k, {c: v[k][c] for c in CLASSES if v[k][c]}, flush=True)


if __name__ == '__main__':
    main()
