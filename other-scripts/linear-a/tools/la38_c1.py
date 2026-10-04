#!/usr/bin/env python3
"""LA-38 cycle 1: calibrate on the positive controls before Linear A is tested in full.
Linear B (DAMOS KN+PY) full, Linear B drawn at Linear A size (10 draws), Cypriot (Idalion tablet ICS 217),
and wrong-value controls (a random relabeling of LB-at-LA-size treated as the truth).
Composites declared before the LA run: ALL7 = ocpC ocpP vinit son freqC freqV comp (a-priori directions);
SEL[tier] = measures with z >= 1.64 under that tier on full LB (chosen here, applied unchanged to LA):
R1 freqC; R2a comp; R2b and R3 ocpC ocpP vinit."""
import sys, os, json, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import la38_common as L
from multiprocessing import Pool

ALL7 = ['ocpC', 'ocpP', 'vinit', 'son', 'freqC', 'freqV', 'comp']
TIERS = ['R1', 'R2a', 'R2b', 'R3']


def corpus(tag):
    if tag == 'LBfull':
        c = L.Corpus(L.lb_words(), name=tag); s, P = L.la21_for('LB'); c.set_la21(s, P); return c
    if tag.startswith('LBd'):
        ntok = L.Corpus(L.la_words()).ntok
        c = L.Corpus(L.lb_draw(ntok, int(tag[3:])), name=tag); s, P = L.la21_for('LB'); c.set_la21(s, P); return c
    if tag.startswith('LBs'):   # LB drawn at LA size, signs shuffled inside words
        ntok = L.Corpus(L.la_words()).ntok
        c = L.Corpus(L.shuffle_words(L.lb_draw(ntok, int(tag[3:])), 900 + int(tag[3:])), name=tag)
        s, P = L.la21_for('LB'); c.set_la21(s, P); return c
    if tag == 'CYP':
        return L.Corpus(L.cyp_words(), name=tag)
    if tag == 'LA':
        c = L.Corpus(L.la_words(), name=tag); s, P = L.la21_for('LA'); c.set_la21(s, P); return c
    if tag.startswith('LAsh'):
        c = L.Corpus(L.shuffle_words(L.la_words(), int(tag[4:])), name=tag); s, P = L.la21_for('LA'); c.set_la21(s, P); return c
    raise ValueError(tag)


def job(args):
    tag, tier, N, seed, fake = args
    c = corpus(tag)
    base = None
    if fake is not None:   # wrong values: a random full relabeling treated as the truth
        rng = np.random.default_rng(10_000 + fake)
        Cf, Vf = L.relabel(c, 'R3', 1, rng); base = (Cf[0], Vf[0])
    C0, V0 = (c.C0, c.V0) if base is None else base
    obs = L.measures(c, C0[None], V0[None])
    null = L.null_measures(c, tier, N, seed, base=base)
    sel = json.load(open(os.path.join(L.CK, 'sel.json')))[tier] if os.path.exists(os.path.join(L.CK, 'sel.json')) else ALL7
    r_all = L.compare(obs, null, ALL7)
    r_sel = L.compare(obs, null, sel)
    out = {k: v for k, v in r_all.items() if k != 'COMP'}
    out['ALL7'] = r_all.get('COMP'); out['SEL'] = r_sel.get('COMP')
    return dict(tag=tag, tier=tier, N=N, fake=fake, res=out)


def run(jobs, outname, procs=2):
    t = time.time()
    with Pool(procs) as p:
        res = p.map(job, jobs, chunksize=1)
    json.dump(res, open(os.path.join(L.CK, outname), 'w'), default=float)
    print('done', outname, round(time.time() - t), 's')
    return res


if __name__ == '__main__':
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    stage = sys.argv[1]
    if stage == 'lbfull':
        res = run([('LBfull', t, 100_000, 1, None) for t in TIERS], 'c1_lbfull.json')
        sel = {r['tier']: [k for k in ALL7 if k in r['res'] and not r['res'][k]['inv'] and r['res'][k]['z'] >= 1.64]
               for r in res}
        json.dump(sel, open(os.path.join(L.CK, 'sel.json'), 'w')); print('SEL', sel)
    elif stage == 'ctrl':
        jobs = [(f'LBd{d}', t, 10_000, 2 + d, None) for d in range(10) for t in TIERS]
        jobs += [('CYP', t, 100_000, 3, None) for t in TIERS]
        run(jobs, 'c1_ctrl.json')
    elif stage == 'fake':
        jobs = [(f'LBd{f % 10}', t, 1000, 50 + f, f) for f in range(100) for t in TIERS]
        run(jobs, 'c1_fake.json')
