#!/usr/bin/env python3
"""LA-38 cycle 4: held-out replication. The whole-map tests (R2a, R2b, R3; 10^4 relabelings) and the row / sign
decomposition are repeated on Hagia Triada alone and on all other sites alone. A row (or sign) whose verdict is
real should hold in both halves; LB has no such split, so the control is the LB-at-LA-size draws of cycle 3
scaled down (LB draws at half size)."""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import la38_common as L
from la38_c1 import run
import la38_c3 as T
from multiprocessing import Pool

if __name__ == '__main__':
    stage = sys.argv[1]
    if stage == 'whole':
        run([(h, t, 10_000, 11, None) for h in ('LAHT', 'LAnHT') + tuple(f'LBh{d}' for d in range(6)) for t in ('R2a', 'R2b', 'R3')],
            'c4_whole.json')
    elif stage == 'rows':
        t0 = time.time()
        with Pool(2) as p:
            res = p.map(T.analyse, [('LAHT', False, 201), ('LAnHT', False, 202)] + [(f'LBh{d}', False, 210 + d) for d in range(4)],
                        chunksize=1)
        json.dump(res, open(os.path.join(L.CK, 'c4_rows.json'), 'w'), default=float)
        print('done', round(time.time() - t0))
