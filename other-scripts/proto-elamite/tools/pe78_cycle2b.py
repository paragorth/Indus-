"""pe78 cycle 2b: kill tests for the HERD -> LIVING lead of cycle 2.
(a) within-tablet shuffle: values permuted across keys inside each tablet (50x); HERD mean P(LIVING) recomputed.
    If herd tablets as a whole carry the signal, the shuffle keeps it.
(b) tablet-type matched null: random key sets drawn only from keys whose groups sit on the same tablets as HERD.
(c) split-half rho real vs shuffled per split (paired)."""
import sys, os, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pe78_common as C
from pe78_cycle1 import Reader
from pe78_cycle2 import read, HERD
d = np.load(os.path.join(C.CK, 'sims_c1.npz'), allow_pickle=True)
R = Reader(d['X'], d['y'])
E = C.pe_entries()
P, S = read(R, E, np.random.default_rng(3))
real = np.mean([P[k][0] for k in P if k in HERD])
sh = []
for r in range(50):
    Ps, _ = read(R, C.shuffle_within_tablet(E, np.random.default_rng(700 + r)), np.random.default_rng(3))
    hk = [k for k in Ps if k in HERD]
    if hk:
        sh.append(np.mean([Ps[k][0] for k in hk]))
sh = np.array(sh)
pa = float((sh >= real).mean())
print('HERD P(L) real %.3f shuffled %.3f +- %.3f p %.3f (n %d)' % (real, sh.mean(), sh.std(), pa, len(sh)))
G = C.groups(E)
htabs = set(t for k in HERD if k in G for t, _ in G[k])
share = {k: np.mean([t in htabs for t, _ in G[k]]) for k in P}
print('herd-tablet share per key', {k: round(v, 2) for k, v in sorted(share.items(), key=lambda x: -x[1])[:12]})
cand = [k for k in P if k not in HERD and share[k] >= 0.3]
print('non-HERD keys on herd tablets', cand, [round(P[k][0], 2) for k in cand])
c2 = json.load(open(os.path.join(C.CK, 'c2_res.json')))
dr = [a[0] - b[0] for a, b in zip(c2['real'], c2['shuf'])]
print('split-half paired diffs', np.round(dr, 2), 'real>shuf in %d/10' % sum(x > 0 for x in dr))
rows = ['| PE-78.2.5 | Kill test for the HERD lead: values permuted across keys within each tablet (50x), HERD read again; keys sharing the herd tablets listed | HERD P(LIVING) real %.2f vs shuffled %.2f +- %.2f (p %.2f); non-HERD keys with >= 30%% of groups on herd tablets: %s | %s |' % (
    real, sh.mean(), sh.std(), pa, ', '.join('%s %.2f' % (k, P[k][0]) for k in cand) or 'none',
    'KILLED: the herd tablets value sets carry it, not the herd keys' if pa > 0.05 else 'survives the within-tablet shuffle'),
    '| PE-78.2.6 | Split-half stability, paired by split (real vs within-tablet shuffle, same halves) | rho differences %s; real > shuffled in %d of 10 splits | %s |' % (
    ' '.join('%.2f' % x for x in dr), sum(x > 0 for x in dr), 'weak key-level stability' if sum(x > 0 for x in dr) >= 8 else 'no key-level stability beyond tablet value sets')]
open(os.path.join(C.CK, 'c2b_rows.txt'), 'w').write('\n'.join(rows) + '\n')
print('\n'.join(rows))
