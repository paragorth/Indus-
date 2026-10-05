#!/usr/bin/env python3
"""LA-54 cycle 2: fill in the unwritten commodity across the archive.
Usage: la54_c2.py MODEL SEED   MODEL in FULL (all feature groups) | BLIND (no word features: numbers, fractions,
rounding, site, support, layout). Writes data/la54_ckpt/c2_<MODEL>_<SEED>.json with
  oof: out-of-fold predictions for every labelled document (nested: selection inside each outer fold) -> calibration
  imp: predictions for every unlabelled document (selection by CV on all labelled docs)."""
import sys, os, json, random, time, collections
import numpy as np
import la54_common as C, la54_engine as E
from la54_c1 import restrict

NCFG = 1500


def select(cfgs, X, y, classes, rng, topk=25):
    sc = E.inner_scores(cfgs, X, y, classes, rng, nfold=4)
    return sorted(range(len(cfgs)), key=lambda k: (-sc[k][0], sc[k][1]))[:topk], sc


def main(model, sd):
    rng = random.Random(C.seed('la54c2' + model + str(sd)))
    la = C.la_docs()
    F = C.Featurizer(la)
    lab = [d for d in la if d['label']]
    unl = [d for d in la if not d['label']]
    X, Xu = F.X(lab), F.X(unl)
    y = np.array([d['label'] for d in lab]); classes = sorted(set(y))
    cfgs = E.random_configs(NCFG, F, rng)
    if model == 'BLIND':
        cfgs = restrict(cfgs, F, ['NUM', 'FRAC', 'ROUND', 'SITE', 'SUPP', 'LAYOUT'], rng)
    t = time.time()
    # nested out-of-fold predictions for calibration
    from sklearn.model_selection import StratifiedKFold
    oof = np.zeros((len(lab), len(classes)))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=sd).split(X, y):
        top, _ = select(cfgs, X[tr], y[tr], classes, rng)
        oof[te] = np.mean([E.proba(cfgs[k], X[tr], y[tr], X[te], classes) for k in top], 0)
    top, sc = select(cfgs, X, y, classes, rng)
    Pu = np.mean([E.proba(cfgs[k], X, y, Xu, classes) for k in top], 0)
    res = {'model': model, 'seed': sd, 'classes': classes, 'lab_ids': [d['id'] for d in lab], 'y': y.tolist(),
           'oof': oof.tolist(), 'unl_ids': [d['id'] for d in unl], 'imp': Pu.tolist(),
           'top': [{'fam': cfgs[k]['fam'], 'groups': cfgs[k]['groups'], 'acc': sc[k][0]} for k in top],
           'groups': E.group_importance(cfgs, sc), 'secs': time.time() - t}
    json.dump(res, open(os.path.join(C.CK, 'c2_%s_%d.json' % (model, sd)), 'w'))
    print(model, sd, 'oof acc', float((oof.argmax(1) == np.array([classes.index(c) for c in y])).mean()), round(time.time() - t))


if __name__ == '__main__':
    for a in sys.argv[1:]:
        m, s = a.split(':')
        if not os.path.exists(os.path.join(C.CK, 'c2_%s_%s.json' % (m, s))):
            main(m, int(s))
