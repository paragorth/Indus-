#!/usr/bin/env python3
"""LA-37 cycle 1: calibration of the doublet detector.
A Linear B full (KN+PY), blind; scored post hoc with LB labels (doublet / same consonant / same vowel / other).
B Linear B cut to Linear A size (10 document draws).
C Kill control: the same LB corpora with signs shuffled inside words (must lose the related-pair signal).
D Planted doublets in the real Linear A corpus: 6 hosts x 4 modes (token free variation, lexical by word type,
  by site group, word-initial only) at rate 0.4; recovery = planted pair's percentile and flag.
"""
import os, sys, json, collections, time
os.environ.setdefault('OMP_NUM_THREADS', '1')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from multiprocessing import Pool
import la37_common as K

OUT = os.path.join(K.LOOPS, 'la37_cycle1.txt')
LOG = os.path.join(K.CK, 'c1.log')


def lb_metrics(res):
    al, iu = res['alph'], res['iu']
    labs = np.array([K.lb_label(al[a], al[b]) for a, b in zip(*iu)], dtype=object)
    rel = np.isin(labs, ['doublet', 'sameC', 'sameV'])
    ok = labs != None
    m = {}
    for key in ('T', 'z', 'fm'):
        v = res[key]
        m[key] = {g: K.auc(v[labs == g], v[labs == 'other']) for g in ('doublet', 'sameC', 'sameV')}
        top = np.argsort(-np.where(ok, v, -9))[:20]
        m[key]['top20rel'] = float(rel[top].mean())
    f = K.flagged(res) & ok
    m['nflag'] = int(f.sum()); m['flag_rel'] = float(rel[f].mean()) if f.sum() else np.nan
    m['base_rel'] = float(rel[ok].mean())
    m['ndoub'] = int((labs == 'doublet').sum())
    m['doub_pct'] = [(al[a], al[b], round(float(res['fm'][j]), 2), round(float(res['z'][j]), 2))
                     for j, (a, b) in enumerate(zip(*iu)) if labs[j] == 'doublet']
    return m


def job(args):
    kind, seed = args
    rng = np.random.default_rng(seed)
    if kind.startswith('LB'):
        B = K.lb_units()
        if kind in ('LBfull', 'LBfull_sh'):
            u, fmin, R = B, 10, 100
        else:
            u, fmin, R = K.lb_draw(B, 3918, rng), 8, 150
        if kind.endswith('_sh'):
            u = K.shuffle_within(u, rng)
        al, c = K.alphabet(u, fmin)
        res = K.score2(u, al, c, R=R, seed=seed)
        return kind, seed, lb_metrics(res)
    # planted LA
    _, host, mode = kind.split('|')
    U = K.la_units()
    u = K.plant(U, host, 0.4, mode, rng)
    al, c = K.alphabet(u, 8)
    if 'X*' not in al:
        return kind, seed, None
    res = K.score2(u, al, c, R=100, seed=seed)
    iu = res['iu']
    j = [k for k, (a, b) in enumerate(zip(*iu)) if {al[a], al[b]} == {host, 'X*'}][0]
    xs = [k for k, (a, b) in enumerate(zip(*iu)) if 'X*' in (al[a], al[b])]
    rank_x = int((res['T'][xs] > res['T'][j]).sum()) + 1
    return kind, seed, dict(nX=c['X*'], T=float(res['T'][j]), fm=float(res['fm'][j]), z=float(res['z'][j]),
                            q=float(res['q'][j]), fwer=float(res['fwer'][j]), flag=bool(K.flagged(res)[j]),
                            rank_among_Xpairs=rank_x, of=len(xs),
                            pct_all=float((res['T'] < res['T'][j]).mean()))


if __name__ == '__main__':
    t0 = time.time()
    jobs = [('LBfull', 0), ('LBfull_sh', 0)] + [('LBdraw', s) for s in range(10)] + [('LBdraw_sh', s) for s in range(5)]
    hosts = ['KA', 'SI', 'TA', 'NA', 'DA', 'RE']
    jobs += [(f'PL|{h}|{m}', 100 + i) for i, h in enumerate(hosts) for m in ('token', 'type', 'group', 'initial')]
    with Pool(2) as p:
        res = p.map(job, jobs, chunksize=1)
    json.dump([(k, s, r) for k, s, r in res], open(os.path.join(K.CK, 'c1.json'), 'w'), default=str)
    K.log(LOG, f'done in {time.time() - t0:.0f}s')
