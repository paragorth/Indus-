#!/usr/bin/env python3
"""LA-42 cycle 1: calibrate grid completion on Linear B (its own values hidden one sign at a time).
LB drawn by document at LA's valued-token count (10 draws). Each valued sign is hidden in turn (full count, thinned
to 10 and to 5 tokens) and all candidates are scored. Weights fitted on draws 0-4, tested on draws 5-9, and vice
versa; feature sets: PRIOR (grid structure only), PHON, SIM, ALL. Full LB with the blind la21 LB rows: r21 check."""
import os, sys, pickle, collections
import numpy as np
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import la42_common as G

NTOK = 3569
SETS = dict(PRIOR=['empty', 'lrow', 'lcol', 'new'],
            PHON=['empty', 'lrow', 'lcol', 'new', 'bits', 'ocpC', 'ocpP', 'vI', 'vM', 'sameV'],
            SIM=['empty', 'lrow', 'lcol', 'new', 'simC', 'simV'],
            ALL=['empty', 'lrow', 'lcol', 'new', 'bits', 'ocpC', 'ocpP', 'vI', 'vM', 'sameV', 'simC', 'simV'],
            ALLnoE=['lrow', 'lcol', 'new', 'bits', 'ocpC', 'ocpP', 'vI', 'vM', 'sameV', 'simC', 'simV'])


def job(args):
    d, words, h, keep = args
    V = G.known_values(words)
    c = G.make_case(words, V, h, keep=keep, seed=d * 1000 + sum(map(ord, h)))
    if c is not None:
        c['draw'] = d; c['keep'] = keep or 0
    return c


def main():
    U = G.lb_units()
    jobs = []
    for d in range(10):
        W = G.lb_draw(U, NTOK, seed=100 + d)
        cnt = collections.Counter(s for w in W for s in w)
        V = G.known_values(W)
        for h in sorted(V):
            if cnt[h] < 3:
                continue
            for keep in (None, 10, 5):
                if keep is None or cnt[h] > keep:
                    jobs.append((d, W, h, keep))
    print(len(jobs), 'cases', flush=True)
    with Pool(2) as p:
        cases = [c for c in p.imap(job, jobs, chunksize=4) if c is not None]
    pickle.dump(cases, open(os.path.join(G.CK, 'c1_lbcases.pkl'), 'wb'))
    lines = []
    for nm, use in SETS.items():
        for keep in (0, 10, 5):
            e1 = G.evaluate(G.fit([c for c in cases if c['draw'] < 5], use), [c for c in cases if c['draw'] >= 5 and c['keep'] == keep])
            e2 = G.evaluate(G.fit([c for c in cases if c['draw'] >= 5], use), [c for c in cases if c['draw'] < 5 and c['keep'] == keep])
            e = {k: (e1[k] * e1['n'] + e2[k] * e2['n']) / (e1['n'] + e2['n']) for k in e1 if k != 'n'} | dict(n=e1['n'] + e2['n'])
            lines.append(f"LB {nm} keep={keep or 'full'}: {G.fmt_eval(e)}")
            print(lines[-1], flush=True)
    full = G.fit(cases, SETS['ALL'])
    lines.append('weights ALL (all LB draws): ' + ', '.join(f'{f} {w:+.3g}' for f, w in zip(full['use'], full['w'])))
    pickle.dump(full, open(os.path.join(G.CK, 'c1_model_ALL.pkl'), 'wb'))
    for nm in SETS:
        pickle.dump(G.fit(cases, SETS[nm]), open(os.path.join(G.CK, f'c1_model_{nm}.pkl'), 'wb'))
    # full LB with blind la21 rows (la21 ran on full LB)
    UW = [r['w'] for r in U]
    V = G.known_values(UW)
    r21 = G.la21('LB')
    sim = None
    fcases = []
    for h in sorted(V):
        c = G.make_case(UW, V, h, r21=r21)
        if c is not None:
            fcases.append(c)
    pickle.dump(fcases, open(os.path.join(G.CK, 'c1_lbfull.pkl'), 'wb'))
    has = [c for c in fcases if np.any(c['F'][:, G.FEATS.index('r21')] != 0)]
    for nm, use in (('ALL', SETS['ALL']), ('ALL+r21', SETS['ALL'] + ['r21']), ('PRIOR+r21', SETS['PRIOR'] + ['r21'])):
        # leave-one-sign-out fit on full LB
        hits = []
        for i in range(len(has)):
            m = G.fit(has[:i] + has[i + 1:], use)
            hits.append(G.evaluate(m, [has[i]]))
        e = {k: np.mean([x[k] for x in hits]) for k in hits[0] if k != 'n'} | dict(n=len(hits))
        lines.append(f"LB full (27k tokens), signs in la21 run, nested LOO fit, {nm}: {G.fmt_eval(e)}")
        print(lines[-1], flush=True)
    open(os.path.join(G.CK, 'c1_report.txt'), 'w').write('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
