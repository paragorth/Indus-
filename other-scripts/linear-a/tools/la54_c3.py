#!/usr/bin/env python3
"""LA-54 cycle 3: is the unwritten commodity anything more than the site?
Jobs (written to data/la54_ckpt/c3_<job>.json):
  WS_k     labels permuted WITHIN site among labelled docs (keeps every site's commodity mix), full pipeline
  HT       Hagia Triada labelled docs only (site cannot help), full pipeline;  HTSH_k  its label shuffles
  LOSO     train on Hagia Triada, predict labelled docs from all other sites (no site features);
  LOSOSH_k same with HT labels shuffled
  LBWS_k / LBHT: the same within-site test on Linear B (site = KN/PY/...), to see what a known script gives.
Usage: la54_c3.py WORKER NWORKERS"""
import sys, os, json, random, time, collections
import numpy as np
import la54_common as C, la54_engine as E
from la54_c1 import restrict

NCFG, NOUT = 300, 3
NOSITE = ['NUM', 'FRAC', 'ROUND', 'WORD', 'SUPP', 'LAYOUT']


def jobs():
    J = ['HT', 'LOSO'] + ['WS_%d' % k for k in range(8)] + ['HTSH_%d' % k for k in range(6)] + ['LOSOSH_%d' % k for k in range(6)]
    J += ['LBWS_%d' % k for k in range(3)] + ['LBREAL_%d' % k for k in range(3)]
    return J


def perm_within(docs, key, rng):
    g = collections.defaultdict(list)
    for i, d in enumerate(docs): g[d[key]].append(i)
    ys = [d['label'] for d in docs]
    out = list(ys)
    for ii in g.values():
        v = [ys[i] for i in ii]; rng.shuffle(v)
        for i, x in zip(ii, v): out[i] = x
    return [dict(d, label=y) for d, y in zip(docs, out)]


def run(job):
    rng = random.Random(C.seed('la54c3' + job))
    base = job.split('_')[0]
    la = C.la_docs()
    res = {'job': job}
    if base in ('LBWS', 'LBREAL'):
        pool = C.lb_docs()
        lab = C.sample_like([dict(d) for d in pool], 227, rng)
        F = C.Featurizer(lab)
        if base == 'LBWS': lab = perm_within(lab, 'site', rng)
        cfgs = restrict(E.random_configs(NCFG, F, rng), F, NOSITE, rng)
        agg, out = E.pipeline(lab, F, cfgs, rng, n_outer=NOUT)
        res['agg'] = agg
    elif base in ('WS',):
        F = C.Featurizer(la)
        lab = perm_within([d for d in la if d['label']], 'site', rng)
        cfgs = E.random_configs(NCFG, F, rng)
        res['agg'], _ = E.pipeline(lab, F, cfgs, rng, n_outer=NOUT)
    elif base in ('HT', 'HTSH'):
        ht = [d for d in la if d['site'] == 'Haghia Triada']
        F = C.Featurizer(ht)
        lab = [d for d in ht if d['label']]
        c = collections.Counter(d['label'] for d in lab)
        lab = [dict(d, label=d['label'] if c[d['label']] >= 5 else 'OTH') for d in lab]
        if base == 'HTSH':
            ys = [d['label'] for d in lab]; rng.shuffle(ys); lab = [dict(d, label=y) for d, y in zip(lab, ys)]
        cfgs = restrict(E.random_configs(NCFG, F, rng), F, NOSITE, rng)
        res['agg'], _ = E.pipeline(lab, F, cfgs, rng, n_outer=NOUT)
        res['n'] = len(lab); res['classes'] = dict(collections.Counter(d['label'] for d in lab))
    elif base in ('LOSO', 'LOSOSH'):
        F = C.Featurizer(la)
        lab = [d for d in la if d['label']]
        tr = [d for d in lab if d['site'] == 'Haghia Triada']; te = [d for d in lab if d['site'] != 'Haghia Triada']
        if base == 'LOSOSH':
            ys = [d['label'] for d in tr]; rng.shuffle(ys); tr = [dict(d, label=y) for d, y in zip(tr, ys)]
        cfgs = restrict(E.random_configs(NCFG, F, rng), F, NOSITE, rng)
        Xtr, Xte = F.X(tr), F.X(te)
        ytr = np.array([d['label'] for d in tr]); yte = np.array([d['label'] for d in te])
        classes = sorted(set(ytr) | set(yte))
        sc = E.inner_scores(cfgs, Xtr, ytr, classes, rng)
        top = sorted(range(len(cfgs)), key=lambda k: (-sc[k][0], sc[k][1]))[:25]
        P = np.mean([E.proba(cfgs[k], Xtr, ytr, Xte, classes) for k in top], 0)
        cnt = collections.Counter(ytr)
        prior = np.array([(cnt[c] + 0.5) / (len(ytr) + 0.5 * len(classes)) for c in classes])
        m = E.metrics(P, yte, classes, prior)
        m['train_majority_on_test'] = float((yte == cnt.most_common(1)[0][0]).mean())
        res['agg'] = m
        res['pred'] = {d['id']: [d['label'], classes[int(p.argmax())]] for d, p in zip(te, P)}
    json.dump(res, open(os.path.join(C.CK, 'c3_%s.json' % job), 'w'))
    print(job, json.dumps(res['agg']), flush=True)


if __name__ == '__main__':
    w, n = int(sys.argv[1]), int(sys.argv[2])
    for i, j in enumerate(jobs()):
        if i % n == w and not os.path.exists(os.path.join(C.CK, 'c3_%s.json' % j)):
            run(j)
