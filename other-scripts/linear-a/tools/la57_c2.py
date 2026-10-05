#!/usr/bin/env python3
"""LA-57 cycle 2: thousands of random TRAINED role taggers, leave-one-system-out and leave-one-civilisation-out.
Spec = random feature subset (2-15 of 42 abstract features) x random family (logistic regression with random C,
decision tree with random depth / leaf size, Gaussian naive Bayes).  For every held-out unit the spec is fitted on
the other systems (each system weighted 1, classes balanced) and scored (occurrence AUC) on the held-out system.
A spec SURVIVES if its held-out AUC >= THR on every held-out known system that carries the role.
Null: role labels permuted within every training corpus (type level; khipu occurrence level), same specs.
Survivors are refitted on all known systems and read out on PLANT, Linear A and shuffled Linear A.
Usage: la57_c2.py [NSPEC] [NNULL] [THR] [tag] [roles]"""
import os, sys, pickle, json, time, warnings
import numpy as np
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import la57_common as C
import la57_c1 as P1
from la57_read import readout
warnings.filterwarnings('ignore')
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

NSPEC = int(sys.argv[1]) if len(sys.argv) > 1 else 300
NNULL = int(sys.argv[2]) if len(sys.argv) > 2 else 4
THR = float(sys.argv[3]) if len(sys.argv) > 3 else 0.6
TAG = sys.argv[4] if len(sys.argv) > 4 else 'c2'
ROLESEL = sys.argv[5].split(',') if len(sys.argv) > 5 else C.ROLES
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
        return LogisticRegression(C=par['C'], max_iter=300)
    if fam == 'TREE':
        return DecisionTreeClassifier(max_depth=par['depth'], min_samples_leaf=par['leaf'])
    return GaussianNB()


def fit(sp, systems, ys):
    X = np.vstack([G['X'][k] for k in systems])
    y = np.concatenate(ys)
    w = []
    for k, yy in zip(systems, ys):
        n1 = max(1, yy.sum()); n0 = max(1, len(yy) - yy.sum())
        w.append(np.where(yy, 0.5 / n1, 0.5 / n0))
    w = np.concatenate(w); w = w / w.mean()
    if y.all() or not y.any():
        return None
    m = make(sp)
    m.fit(X[:, sp[0]], y, sample_weight=w)
    return m


def scorefn(m, f):
    return lambda X: m.predict_proba(X[:, f])[:, 1] if hasattr(m, 'predict_proba') else m.decision_function(X[:, f])


def job(args):
    role, si, sp, units, rep = args
    res = {}
    for name, hold in units:
        tr = [k for k in G['elig'][role] if k not in hold]
        m = fit(sp, tr, [G['Y'][role][k][rep] for k in tr])
        for h in hold:
            if m is None:
                res[(name, h)] = 0.5; continue
            s = m.predict_proba(G['X'][h][:, sp[0]])[:, 1]
            res[(name, h)] = C.auc(s + 1e-9 * np.arange(len(s)) * 0, G['Y'][role][h][0])
    return role, si, rep, res


def init(g):
    G.update(g)


def main():
    t0 = time.time()
    F, sysd = P1.load()
    sysd['KH']['kh'] = True
    rng = np.random.default_rng(C.seed('la57-' + TAG))
    g = dict(X={}, Y={}, elig={})
    rows_of = {}
    for role in C.ROLES:
        g['Y'][role] = {}
        g['elig'][role] = []
    for k in C.KNOWN:
        rows = np.array([i for i, l in enumerate(sysd[k]['labs']) if l is not None])
        rows_of[k] = rows
        g['X'][k] = sysd[k]['X'][rows]
    for role in ROLESEL:
        for k in C.KNOWN:
            labs = sysd[k]['labs']
            npos = sum(1 for l in labs if l == role); nneg = sum(1 for l in labs if l is not None and l != role)
            if npos >= 30 and nneg >= 30:
                g['elig'][role].append(k)
                _, Ys = P1.label_sets(sysd[k], role, NNULL, rng)
                g['Y'][role][k] = Ys
    tasks, specs = [], {}
    for role in ROLESEL:
        el = g['elig'][role]
        if len(el) < 3:
            log(role, 'skipped, eligible', el); continue
        units = [(h, [h]) for h in el]
        civs = sorted({C.CIV2[k] for k in el})
        if len(civs) >= 2:
            units += [('CIV:' + c, [k for k in el if C.CIV2[k] == c]) for c in civs]
        specs[role] = [spec(rng) for _ in range(NSPEC)]
        for si, sp in enumerate(specs[role]):
            for rep in range(NNULL + 1):
                tasks.append((role, si, sp, units, rep))
    log('tasks', len(tasks))
    R = {}
    with Pool(2, initializer=init, initargs=(g,)) as pool:
        for n, (role, si, rep, res) in enumerate(pool.imap_unordered(job, tasks, chunksize=8)):
            R[(role, si, rep)] = res
            if n % 2000 == 0:
                log('  %d/%d %.0fs' % (n, len(tasks), time.time() - t0))
    G.update(g)
    summary, models = {}, {}
    for role in specs:
        el = g['elig'][role]
        loso = [(h, h) for h in el]
        civk = sorted(k for k in R[(role, 0, 0)] if k[0].startswith('CIV:'))
        def surv(rep, keys, thr=THR):
            return [si for si in range(NSPEC) if all(R[(role, si, rep)][k] >= thr for k in keys)]
        real = surv(0, loso)
        nulls = [len(surv(rep, loso)) for rep in range(1, NNULL + 1)]
        realc = surv(0, civk) if civk else []
        nullc = [len(surv(rep, civk)) for rep in range(1, NNULL + 1)] if civk else []
        # per held-out system: median AUC over specs, real vs null
        per = {}
        for key in loso + civk:
            a = np.array([R[(role, si, 0)][key] for si in range(NSPEC)])
            an = np.array([R[(role, si, rep)][key] for si in range(NSPEC) for rep in range(1, NNULL + 1)])
            per['%s>%s' % key if key[0] != key[1] else key[0]] = (float(np.median(a)), float(np.quantile(a, 0.9)),
                                                               float(np.median(an)), float(np.quantile(an, 0.9)),
                                                               float(np.mean(a >= THR)), float(np.mean(an >= THR)))
        summary[role] = dict(elig=el, n_surv=len(real), null_surv=nulls, n_surv_civ=len(realc), null_surv_civ=nullc, per=per,
                             surv_specs=[(specs[role][si][0].tolist(), specs[role][si][1], specs[role][si][2]) for si in real])
        log('%s: systems %s | survivors (AUC>=%.2f on every held-out system) %d / %d (null %s) | civ-out survivors %d (null %s)' % (
            role, el, THR, len(real), NSPEC, nulls, len(realc), nullc))
        for k, v in per.items():
            log('     held-out %-12s spec AUC median %.3f q90 %.3f | null median %.3f q90 %.3f | share>=thr %.2f vs null %.2f' % ((k,) + v))
        # refit survivors on all systems (real labels) for the readout
        ms = []
        for si in real[:300]:
            sp = specs[role][si]
            m = fit(sp, el, [g['Y'][role][k][0] for k in el])
            if m is not None:
                ms.append(scorefn(m, sp[0]))
        models[role] = ms
    json.dump(summary, open(os.path.join(OUT, 'summary.json'), 'w'))
    ro = readout(models, F, list(models), log)
    json.dump(ro, open(os.path.join(OUT, 'readout.json'), 'w'), default=str)
    log('done %.0fs' % (time.time() - t0))


if __name__ == '__main__':
    main()
