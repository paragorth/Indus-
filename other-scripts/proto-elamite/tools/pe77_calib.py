"""pe77 calibration on proto-cuneiform: are the unforgeable PC signs the commodity / counted-goods signs?"""
import sys, os, json, collections
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pe77_common as pc
from pe77_analyze import load, unforg, rank_auc


def sign_scores(U, prefixes=('BIND_', 'MAG_')):
    acc = collections.defaultdict(list)
    for f, v in U.items():
        for p in prefixes:
            if f.startswith(p) and f not in ('BIND_all', 'MAG_all'):
                acc[f[len(p):]].append(v[0])
    return {s: float(np.mean(v)) for s, v in acc.items()}


def report(tag, out):
    R = load(tag)
    U, As, Aw = unforg(R)
    res = {'n': len(R), 'A_strong': As, 'A_weak': Aw}
    for nm, pref in [('bindmag', ('BIND_', 'MAG_')), ('pres', ('P_',)), ('slot', ('I_', 'F_')), ('hdr', ('H_',))]:
        S = sign_scores(U, pref)
        res[nm] = rank_auc(S, pc.PC_WORLD, pc.PC_CONV)
        if nm == 'bindmag':
            res['bindmag_top'] = sorted(S.items(), key=lambda x: -x[1])[:15]
            res['bindmag_bottom'] = sorted(S.items(), key=lambda x: x[1])[:8]
    glob = {f: v[0] for f, v in U.items() if pc.family(f) in ('STRUCT', 'NUMFORM', 'ARITH', 'TABLET')
            or f in ('BIND_all', 'MAG_all', 'MAGSTR')}
    res['global_rank'] = sorted(glob.items(), key=lambda x: -x[1])
    out[tag] = res
    print(tag, json.dumps({k: v for k, v in res.items()}, default=str)[:3000])


if __name__ == '__main__':
    out = {}
    for tag in sys.argv[1:]:
        report(tag, out)
    json.dump(out, open(os.path.join(pc.CKPT, 'calib_%s.json' % '_'.join(sys.argv[1:])), 'w'), indent=1)
