"""pe82 cycle 1 analysis: survivors per run, planted recovery, singleton ranking, PC calibration."""
import json, os, sys, glob, collections, random
import numpy as np
import pe82_common as pc


def summ(fn):
    d = json.load(open(fn))
    res = d['res']
    surv = [r for r in res if r[3] >= 3.5 and r[4] >= 2]
    by = collections.defaultdict(dict)
    for m, si, h, zd, zc, nd, nc in res:
        if si == 0:
            by[(tuple(m), tuple(h))] = (zd, zc)
    both = [k for k, v in by.items() if min(v) >= 2]
    bothneg = [k for k, v in by.items() if max(v) <= -2]
    # singleton mean z over models (half A + half B)
    sing = collections.defaultdict(list)
    for (m, h), v in by.items():
        if len(h) == 1:
            sing[h[0]].append(v)
    S = {s: (float(np.mean([a for a, b in v])), float(np.mean([b for a, b in v]))) for s, v in sing.items()}
    L = np.array([[x[2], x[3]] for x in d['length'] if x[1] == 0])
    return dict(n=len(by), surv=len(surv), both=len(both), bothneg=len(bothneg),
                plant=[k for k in both if 'PLANT' in k[1]], S=S,
                len_both=int(((L[:, 0] >= 2) & (L[:, 1] >= 2)).sum()), len_mean=L.mean(0).round(2).tolist(),
                top=sorted(both, key=lambda k: -min(by[k]))[:12], byv={str(k): by[k] for k in both})


if __name__ == '__main__':
    out = {}
    for fn in sorted(glob.glob(os.path.join(pc.CK, 'c1_*_*_*.json'))):
        tag = os.path.basename(fn)[3:-5]
        s = summ(fn)
        out[tag] = s
        print(tag, 'hyps', s['n'], 'surv', s['surv'], 'both>=2', s['both'], 'both<=-2', s['bothneg'],
              'plant', len(s['plant']), 'len both', s['len_both'], s['len_mean'])
    # PC calibration: singletons, PC_WORLD vs others
    for tag in [t for t in out if t.startswith('PC_real')]:
        S = out[tag]['S']
        sc = {k: min(v) for k, v in S.items()}
        w = [sc[k] for k in sc if k in pc.PC_WORLD]
        c = [sc[k] for k in sc if k in pc.PC_CONV]
        o = [sc[k] for k in sc if k not in pc.PC_WORLD]
        allv = list(sc.values())
        rng = random.Random(0)
        obs = np.mean(w)
        null = [np.mean(rng.sample(allv, len(w))) for _ in range(20000)]
        p = (1 + sum(x >= obs for x in null)) / 20001
        print(tag, 'PC_WORLD n', len(w), 'mean min-half z', round(obs, 3), 'others', round(np.mean(o), 3),
              'CONV n', len(c), round(np.mean(c), 3) if c else None, 'p(world >= random)', round(p, 4))
        print('  top PC singletons', sorted(sc.items(), key=lambda kv: -kv[1])[:15])
    for tag in [t for t in out if t.startswith('PE_real')]:
        S = out[tag]['S']
        sc = sorted(((min(v), k, v) for k, v in S.items()), reverse=True)
        print(tag, 'top PE singletons', [(k, round(a, 2)) for a, k, v in sc[:15]])
        print(tag, 'bottom PE singletons', [(k, round(a, 2)) for a, k, v in sc[-8:]])
    json.dump({k: {kk: vv for kk, vv in v.items() if kk != 'byv'} for k, v in out.items()},
              open(os.path.join(pc.CK, 'c1_summary.json'), 'w'), indent=0, default=str)
