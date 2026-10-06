#!/usr/bin/env python3
"""LA-57 cycle 3: A PARLIAMENT OF BUREAUCRACIES.
Instead of pooling, every known system trains its OWN experts (NSPEC random specs: feature subset x family, fitted on
that system alone).  Each system then votes on every other system and on Linear A.
(1) Kinship matrix: AUC of system A's expert ensemble on system B, per role, against a label-permuted null
    (A's labels permuted at type level, same specs).
(2) Agreement as a confidence gauge: on each held-out known system, the rank correlation between the votes of
    the other systems' ensembles; is high agreement associated with high accuracy?  Then the same agreement on
    Linear A, Linear A S1 and S2 shuffles.
(3) Parliament verdict on Linear A types: role score = mean over voting systems; a type is 'assigned' when a
    majority of systems put it in their top decile.
Usage: la57_c3.py [NSPEC] [NNULL] [tag]"""
import os, sys, json, time, pickle, warnings, collections
import numpy as np
from scipy.stats import rankdata, spearmanr
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import la57_common as C
warnings.filterwarnings('ignore')
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

NSPEC = int(sys.argv[1]) if len(sys.argv) > 1 else 60
NNULL = int(sys.argv[2]) if len(sys.argv) > 2 else 5
TAG = sys.argv[3] if len(sys.argv) > 3 else 'c3'
LAK = sys.argv[4] if len(sys.argv) > 4 else 'LA_ADM'
OUT = os.path.join(C.CK, TAG)
os.makedirs(OUT, exist_ok=True)
LOG = os.path.join(OUT, 'log.txt')
G = {}


def log(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True)
    C.wlog(LOG, s)


def spec(rng):
    k = int(rng.integers(2, 16))
    f = np.sort(rng.choice(C.NF, k, replace=False))
    fam = ['LR', 'TREE', 'NB'][int(rng.integers(0, 3))]
    par = dict(C=float(10 ** rng.uniform(-3, 1))) if fam == 'LR' else (
        dict(depth=int(rng.integers(1, 5)), leaf=int(rng.integers(20, 200))) if fam == 'TREE' else {})
    return (f, fam, par)


def make(sp):
    f, fam, par = sp
    if fam == 'LR':
        return LogisticRegression(C=par['C'], max_iter=300, class_weight='balanced')
    if fam == 'TREE':
        return DecisionTreeClassifier(max_depth=par['depth'], min_samples_leaf=par['leaf'], class_weight='balanced')
    return GaussianNB()


def targets():
    """every corpus the parliament votes on: known systems (labelled rows), LA, LA shuffles, PLANT."""
    T = {}
    for k in C.KNOWN:
        T[k] = G['X'][k]
    T['LA'] = G['F'][(LAK, 0)]['X']
    for j in range(10):
        T['LA_S1_%d' % j] = G['F'][(LAK + '_S1', j)]['X']
        T['LA_S2_%d' % j] = G['F'][(LAK + '_S2', j)]['X']
    for j in range(3):
        T['PLANT_%d' % j] = G['F'][('PLANT', j)]['X']
    return T


def job(args):
    role, src, rep = args
    y = G['Y'][role][src][rep]
    X = G['X'][src]
    T = targets()
    rng = np.random.default_rng(C.seed('la57c3-%s-%s' % (role, src)))   # same specs for real and null
    acc = {t: np.zeros(len(T[t])) for t in T}
    n = 0
    for _ in range(NSPEC):
        sp = spec(rng)
        m = make(sp)
        m.fit(X[:, sp[0]], y)
        for t, Xt in T.items():
            acc[t] += rankdata(m.predict_proba(Xt[:, sp[0]])[:, 1]) / len(Xt)
        n += 1
    return role, src, rep, {t: (v / n).astype(np.float32) for t, v in acc.items()}


def init(g):
    G.update(g)


def main():
    t0 = time.time()
    F, sysd = C.load()
    sysd['KH']['kh'] = True
    rng = np.random.default_rng(C.seed('la57-' + TAG))
    g = dict(X={}, Y={}, F=F, rows={})
    for k in C.KNOWN:
        rows = np.array([i for i, l in enumerate(sysd[k]['labs']) if l is not None])
        g['rows'][k] = rows
        g['X'][k] = sysd[k]['X'][rows]
    elig = {}
    for role in C.ROLES:
        g['Y'][role] = {}
        elig[role] = []
        for k in C.KNOWN:
            labs = sysd[k]['labs']
            if sum(1 for l in labs if l == role) >= 30 and sum(1 for l in labs if l is not None and l != role) >= 30:
                elig[role].append(k)
                g['Y'][role][k] = C.label_sets(sysd[k], role, NNULL, rng)[1]
    tasks = [(role, s, rep) for role in C.ROLES if len(elig[role]) >= 3 for s in elig[role] for rep in range(NNULL + 1)]
    log('tasks', len(tasks))
    V = {}
    with Pool(2, initializer=init, initargs=(g,)) as pool:
        for role, src, rep, acc in pool.imap_unordered(job, tasks):
            V[(role, src, rep)] = acc
    G.update(g)
    pickle.dump(V, open(os.path.join(OUT, 'votes.pkl'), 'wb'))
    la = F[(LAK, 0)]
    summary = {}
    for role in C.ROLES:
        el = elig[role]
        if len(el) < 3:
            continue
        log('=== %s  (systems %s)' % (role, el))
        # (1) kinship
        K = {}
        for a in el:
            line = []
            for b in el:
                if a == b:
                    line.append('   -  '); continue
                yb = g['Y'][role][b][0]
                r = C.auc(V[(role, a, 0)][b], yb)
                nl = [C.auc(V[(role, a, rep)][b], yb) for rep in range(1, NNULL + 1)]
                K[(a, b)] = (r, float(np.median(nl)), float(np.max(nl)))
                line.append('%.2f%s' % (r, '*' if r > max(nl) else ' '))
            log('  %-4s votes on %s: %s' % (a, ' '.join('%-5s' % b for b in el), ' '.join(line)))
        # parliament on each held-out known system: mean of the others' votes
        acc_h, agr_h = {}, {}
        for h in el:
            others = [a for a in el if a != h]
            M = np.array([V[(role, a, 0)][h] for a in others])
            ens = M.mean(0)
            acc_h[h] = C.auc(ens, g['Y'][role][h][0])
            nul = [C.auc(np.array([V[(role, a, rep)][h] for a in others]).mean(0), g['Y'][role][h][0]) for rep in range(1, NNULL + 1)]
            agr_h[h] = float(np.mean([spearmanr(M[i], M[j])[0] for i in range(len(others)) for j in range(i + 1, len(others))]))
            log('  parliament on %-4s AUC %.3f (null med %.3f max %.3f) | voter agreement rho %.3f' % (h, acc_h[h], np.median(nul), max(nul), agr_h[h]))
        # LA agreement vs shuffles
        def agr(t):
            M = np.array([V[(role, a, 0)][t] for a in el])
            return float(np.mean([spearmanr(M[i], M[j])[0] for i in range(len(el)) for j in range(i + 1, len(el))]))
        a_la = agr('LA')
        a_s1 = [agr('LA_S1_%d' % j) for j in range(10)]
        a_s2 = [agr('LA_S2_%d' % j) for j in range(10)]
        pl = []
        for j in range(3):
            f = F[('PLANT', j)]
            rows = [i for i, l in enumerate(f['labs']) if l is not None]
            if any(f['labs'][i] == role for i in rows):
                ens = np.array([V[(role, a, 0)]['PLANT_%d' % j] for a in el]).mean(0)
                pl.append(C.auc(ens[rows], np.array([f['labs'][i] == role for i in rows])))
        log('  voter agreement on LA %.3f | S1 med %.3f max %.3f | S2 med %.3f max %.3f | PLANT parliament AUC %s' % (
            a_la, np.median(a_s1), max(a_s1), np.median(a_s2), max(a_s2), ' '.join('%.2f' % x for x in pl)))
        # (3) verdict on LA types
        M = np.array([V[(role, a, 0)]['LA'] for a in el])
        T = collections.defaultdict(list)
        for i, t in enumerate(la['types']):
            T[t].append(i)
        keep = [t for t, ii in T.items() if len(ii) >= 3]
        per = np.array([[M[a][T[t]].mean() for a in range(len(el))] for t in keep])
        top = per >= np.quantile(per, 0.9, axis=0)[None, :]
        votes = top.sum(1)
        o = np.lexsort((-per.mean(1), -votes))
        ver = [(keep[i], int(votes[i]), round(float(per[i].mean()), 3), len(T[keep[i]])) for i in o[:10]]
        log('  LA verdict (votes of %d systems in top decile): %s' % (len(el), '; '.join('%s %d/%d m%.3f n%d' % (w, v, len(el), m, n) for w, v, m, n in ver)))
        # shuffle control for majority: how many types get a majority in S1/S2
        def nmaj(t):
            Ms = np.array([V[(role, a, 0)][t] for a in el])
            f = F[(LAK + '_' + t.split('_')[1], int(t.split('_')[2]))]
            TT = collections.defaultdict(list)
            for i, x in enumerate(f['types']):
                TT[x].append(i)
            kk = [x for x, ii in TT.items() if len(ii) >= 3]
            pp = np.array([[Ms[a][TT[x]].mean() for a in range(len(el))] for x in kk])
            return int(((pp >= np.quantile(pp, 0.9, axis=0)[None, :]).sum(1) > len(el) / 2).sum())
        maj_la = int((votes > len(el) / 2).sum())
        maj_s1 = [nmaj('LA_S1_%d' % j) for j in range(10)]
        maj_s2 = [nmaj('LA_S2_%d' % j) for j in range(10)]
        log('  LA types with a majority vote: %d of %d | S1 med %.0f max %d | S2 med %.0f max %d' % (
            maj_la, len(keep), np.median(maj_s1), max(maj_s1), np.median(maj_s2), max(maj_s2)))
        summary[role] = dict(elig=el, kin={'%s>%s' % k: v for k, v in K.items()}, parl=acc_h, agree_known=agr_h,
                             agree_la=a_la, agree_s1=a_s1, agree_s2=a_s2, plant=pl, verdict=ver,
                             maj_la=maj_la, maj_s1=maj_s1, maj_s2=maj_s2)
    json.dump(summary, open(os.path.join(OUT, 'summary.json'), 'w'), default=str)
    log('done %.0fs' % (time.time() - t0))


if __name__ == '__main__':
    main()
