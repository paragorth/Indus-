"""pe78 cycle 4: the cycle-2 C- kill line, run as written.  HERD P(LIVING) vs 50 within-tablet shuffles with a
new seed (a) on the full corpus and (b) without the column lists MDP 17,085 (P008283) and 17,097 (P008295),
(c) without the four herd-office tablets that carry the copies (+ P008294, P008389)."""
import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pe78_common as C
from pe78_cycle1 import Reader
from pe78_cycle2 import read, HERD
d = np.load(os.path.join(C.CK, 'sims_c1.npz'), allow_pickle=True)
R = Reader(d['X'], d['y'])
E0 = C.pe_entries()
rows = []
for tag, drop in (('full', set()), ('no 17,085/17,097', {'P008283', 'P008295'}),
                  ('no 4 herd-office copies', {'P008283', 'P008295', 'P008294', 'P008389'})):
    E = [e for e in E0 if e[0] not in drop]
    P, S = read(R, E, np.random.default_rng(3))
    hk = [k for k in P if k in HERD]
    real = np.mean([P[k][0] for k in hk])
    sh = []
    for r in range(50):
        Ps, _ = read(R, C.shuffle_within_tablet(E, np.random.default_rng(5000 + r)), np.random.default_rng(3))
        h2 = [k for k in Ps if k in HERD]
        if h2:
            sh.append(np.mean([Ps[k][0] for k in h2]))
    sh = np.array(sh)
    p = float((sh >= real).mean())
    print(tag, hk, 'real %.3f shuf %.3f +- %.3f p %.3f' % (real, sh.mean(), sh.std(), p), flush=True)
    rows.append((tag, len(hk), real, sh.mean(), p))
s = '; '.join('%s: %d keys, %.2f vs %.2f (p %.2f)' % r for r in rows)
ok = all(r[4] <= 0.05 for r in rows)
line = '| PE-78.4.1 | Kill line of the cycle-2 C- (HERD fluctuates like simulated herds): 50 new within-tablet shuffles (new seed), full corpus and with the column lists / herd-office copies removed | %s | %s |' % (
    s, 'survives (stays C-, uninterpreted: calibration failed)' if ok else 'KILLED by its own kill line')
open(os.path.join(C.CK, 'c4_rows.txt'), 'w').write(line + '\n')
print(line)
