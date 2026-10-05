#!/usr/bin/env python3
"""LA-54 cycle 1: calibration. Can masked commodities be read back from what remains?
Usage: la54_c1.py WORKER NWORKERS   (jobs split round-robin; each job writes data/la54_ckpt/c1_<job>.json)
Jobs: LA (real), LA_big (3000 configs), LSH_k (labels shuffled), NSH_k (numbers shuffled),
      LB_k / UR_k (controls at LA size), PL_k (planted numbers-only commodity), PLN_k (plant labels, no signature),
      PLW_k (planted with a weak word signature), LA_NOWORD (no word group), LA_NUMONLY (NUM+FRAC+ROUND only)."""
import sys, os, json, random, time
import numpy as np
import la54_common as C, la54_engine as E

NCFG, NOUT = 300, 3


def jobs():
    J = ['LA', 'LA_NOWORD', 'LA_NUMONLY', 'LA_SITEONLY']
    J += ['LB_%d' % k for k in range(3)] + ['UR_%d' % k for k in range(3)]
    J += ['PL_%d' % k for k in range(3)] + ['PLN_%d' % k for k in range(3)]
    J += ['LSH_%d' % k for k in range(10)] + ['NSH_%d' % k for k in range(6)]
    J += ['PLS_%d' % k for k in range(3)] + ['PLW_%d' % k for k in range(3)]
    return J


def restrict(cfgs, F, groups, rng):
    out = []
    for c in cfgs:
        g = [x for x in c['groups'] if x in groups] or [rng.choice(groups)]
        cols = [j for j in c['cols'] if F.group[j] in g] or [j for j, gg in enumerate(F.group) if gg in g]
        out.append(dict(c, groups=g, cols=cols))
    return out


def run(job):
    rng = random.Random(C.seed('la54c1' + job))
    la = C.la_docs()
    base = job.split('_')[0]
    if base in ('LB', 'UR'):
        pool = C.lb_docs() if base == 'LB' else C.ur_docs()
        lab = C.sample_like([dict(d) for d in pool], 227, rng)
        ids = {d['id'] for d in lab}
        unl = [dict(d, label=None) for d in rng.sample([d for d in pool if d['id'] not in ids], 166)]
        F = C.Featurizer(lab + unl)
    else:
        F = C.Featurizer(la)
        lab = [d for d in la if d['label']]
    if base == 'LSH':
        ys = [d['label'] for d in lab]; rng.shuffle(ys)
        lab = [dict(d, label=y) for d, y in zip(lab, ys)]
    if base == 'NSH':
        lab = [d for d in C.shuffle_numbers(la, rng) if d['label']]
    if base == 'PL':
        lab = [d for d in C.plant(la, rng) if d['label']]
    if base == 'PLS':
        lab = [d for d in C.plant(la, rng, factor=(4.0, 6.0)) if d['label']]
    if base == 'PLW':
        pl = C.plant(la, rng, wordsig=True)
        F = C.Featurizer(pl)
        lab = [d for d in pl if d['label']]
    if base == 'PLN':
        lab = [d for d in C.plant(la, rng, factor=(1.0, 1.0)) if d['label']]
    ncfg = 3000 if job == 'LA_big' else NCFG
    cfgs = E.random_configs(ncfg, F, rng)
    if job == 'LA_NOWORD': cfgs = restrict(cfgs, F, ['NUM', 'FRAC', 'ROUND', 'SITE', 'SUPP', 'LAYOUT'], rng)
    if job == 'LA_NUMONLY': cfgs = restrict(cfgs, F, ['NUM', 'FRAC', 'ROUND'], rng)
    if job == 'LA_SITEONLY': cfgs = restrict(cfgs, F, ['SITE', 'SUPP'], rng)
    t = time.time()
    agg, out, det = E.pipeline(lab, F, cfgs, rng, n_outer=NOUT, return_detail=True)
    res = {'job': job, 'agg': agg, 'outer': out, 'secs': time.time() - t, 'n_lab': len(lab),
           'classes': det[0]['classes']}
    # group importance (inner CV, first outer split)
    res['groups'] = E.group_importance(cfgs, det[0]['sc'])
    res['fam'] = {f: float(np.mean([det[0]['sc'][k][0] for k, c in enumerate(cfgs) if c['fam'] == f] or [0])) for f in set(E.FAMS)}
    if base in ('PL', 'PLN', 'PLS', 'PLW'):
        rec = []
        y = np.array([d['label'] for d in lab])
        for dd in det:
            cl = dd['classes']; P = np.array(dd['P']); te = dd['te']
            yt = y[te]; pred = np.array(cl)[P.argmax(1)]
            m = yt == 'PLANT'
            rec.append({'recall': float((pred[m] == 'PLANT').mean()) if m.any() else None,
                        'precision': float((yt[pred == 'PLANT'] == 'PLANT').mean()) if (pred == 'PLANT').any() else 0.0,
                        'pplant_true': float(P[m, cl.index('PLANT')].mean()), 'pplant_other': float(P[~m, cl.index('PLANT')].mean())})
        res['plant'] = rec
    json.dump(res, open(os.path.join(C.CK, 'c1_%s.json' % job), 'w'))
    print(job, json.dumps(agg), round(res['secs']), flush=True)


if __name__ == '__main__':
    w, n = int(sys.argv[1]), int(sys.argv[2])
    for i, j in enumerate(jobs()):
        if i % n == w and not os.path.exists(os.path.join(C.CK, 'c1_%s.json' % j)):
            run(j)
