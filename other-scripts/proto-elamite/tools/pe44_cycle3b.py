"""pe44 cycle 3b: are PE tablets 'pure' (all round or all ragged) beyond magnitude?
Same tablet fingerprints as cycle 3, nulls: N1 values shuffled across tablets within system,
N3 within (system, log2 size).  Reference: Ur III (random 1,585 tablets with >= 3 quantities, 5 draws),
where tablets of known type exist.  Per-tablet round share also compared with tablet kind in Ur III
(balanced accounts vs others)."""
import json, os, random, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pe44_common import *  # noqa
from pe44_cycle2 import shuffle_within_system, shuffle_within_size
from pe44_cycle3 import tab_fp, FPS


def excess(tabs, rng, reps):
    real = {t: tab_fp(r) for t, r in tabs.items()}
    out = {}
    for tag, nf in (('N1', shuffle_within_system), ('N3', shuffle_within_size)):
        nl = {f: [] for f in FPS}
        for _ in range(reps):
            tb = nf(tabs, random.Random(int(rng.integers(1e9))))
            fp = [tab_fp(r) for r in tb.values()]
            for f in FPS:
                nl[f].append(sum(v[f] for v in fp))
        for f in FPS:
            a = np.array(nl[f]); x = sum(v[f] for v in real.values())
            out[tag + ':' + f] = {'real': float(x), 'null': float(a.mean()), 'z': float((x - a.mean()) / (a.std() + 1e-9))}
    return out


if __name__ == '__main__':
    rng = np.random.default_rng(31)
    res = {'PE': excess(pe_tabs(), rng, 100)}
    print('PE', json.dumps(res['PE']), flush=True)
    u = ur3_tabs()
    ok = [t for t, d in u.items() if len(d['recs']) >= 3]
    for k in range(3):
        rr = random.Random(k)
        pick = rr.sample(ok, 1585)
        tabs = {t: [dict(r) for r in u[t]['recs'] if r['sys'] == 'UR_CNT' or True][:60] for t in pick}
        res['UR3_%d' % k] = excess(tabs, rng, 60)
        print('UR3', k, json.dumps(res['UR3_%d' % k]), flush=True)
    json.dump(res, open(os.path.join(CK, 'c3b.json'), 'w'), indent=1)
