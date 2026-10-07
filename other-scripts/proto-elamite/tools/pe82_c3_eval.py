"""pe82 cycle 3: score the frozen protection definitions on PE with the pe80 frozen ruler; ruler-permutation null,
line-order-shuffle kill, planted 'heavy things get checksums' control."""
import json, os, sys, random, math, collections
import numpy as np
from scipy.stats import spearmanr
import pe82_common as pc
import pe82_c3 as c3

FR = json.load(open(os.path.join(pc.DATA, 'pe82_c3_frozen_defs.json')))['defs']


def stat(recs, ruler, keys, groups, perm=None):
    rk = list(ruler)
    if perm is not None:
        vals = [ruler[k] for k in rk]; perm.shuffle(vals); ruler = dict(zip(rk, vals))
    out = []
    for d in FR:
        pi = c3.PROT_GRID.index(tuple(d['prot']))
        ex, _ = c3.score_tab(recs, pi, *d['score'], keys, groups)
        out.append(c3.rho(ex, ruler)[0])
    return float(np.nanmedian(out)), out


def plant(T, ruler, rng, q=0.3):
    """Append a true checksum line after the entries of tablets whose entries include a top-third (heaviest) sign,
    with probability q; tablets with only bottom-third signs get none."""
    ks = sorted(ruler, key=lambda k: -ruler[k]); top = set(ks[:len(ks) // 3])
    for t in T:
        L = t['lines']
        if len(L) >= 2 and all(l['v'] is not None for l in L) and any(l['tok'] and l['tok'][-1] in top for l in L) \
                and rng.random() < q:
            L.append(dict(tok=[], first=None, last=None, v=sum(l['v'] for l in L), surf='reverse', nums=[]))
    return T


if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv) > 1 else 'pe'
    rng = random.Random(82)
    T = c3.load_pe_tabs()
    if mode == 'plant':
        T = plant(T, c3.RULER, rng, q=float(sys.argv[2]) if len(sys.argv) > 2 else 0.3)
    keys = set(c3.RULER)
    out = {}
    recs = c3.line_table(T)
    for name, G in (('A', {'A'}), ('B', {'B'}), ('all', None)):
        med, per = stat(recs, c3.RULER, keys, G)
        null = [stat(recs, c3.RULER, keys, G, random.Random(1000 + i))[0] for i in range(300)]
        p = (1 + sum(x >= med for x in null)) / (1 + len(null))
        out[name] = dict(median=med, per=per, null95=float(np.nanpercentile(null, 95)), p=p)
        print(mode, name, 'median rho', round(med, 3), 'null95', round(out[name]['null95'], 3), 'p', round(p, 3),
              'n defs with rho', sum(1 for x in per if x == x))
    kills = []
    for s in range(5):
        rk = c3.line_table(T, random.Random(500 + s))
        kills.append(stat(rk, c3.RULER, keys, None)[0])
    out['kill_all'] = kills
    print(mode, 'order-shuffle kill medians', [round(k, 3) for k in kills])
    json.dump(out, open(os.path.join(pc.CK, "c3_eval_%s%s.json" % (mode, sys.argv[2] if len(sys.argv) > 2 else "")), "w"))
