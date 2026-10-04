"""pe14 cycle 1: WEAVE SCAN. Every feature x every lag 2-12, observed lag agreement
vs (a) within-tablet shuffle and (b) the per-feature first-order adjacency null
(tuned to reproduce lag 1, i.e. the pe13 dip or a control's runs). Tablets split
into halves A / B; search on A, FWER from the null surrogates' own search maxima,
top hits re-tested on B.

usage: python3 pe14_cycle1.py <config> <half>   (half A | B | ALL)
configs: PE_ENT, PE_LIN, UR3_LIN, ARCH_LIN, PL_P2LAST, PL_P3SIZE, PL_P4FIRST, NEG_TOPIC
"""
import json, os, random, sys
from collections import Counter
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from pe14_common import _prep, units, features, lag_agree, null_stats, tune_rho, plant_weave, LAGS, CK  # noqa

NADJ, NSHUF = 120, 60


def corpus(cfg):
    if cfg == 'PE_ENT':
        return units('PE', 'entries')
    if cfg == 'PE_LIN':
        return units('PE', 'lines')
    if cfg in ('UR3_LIN', 'ARCH_LIN'):
        T = units('UR3' if cfg == 'UR3_LIN' else 'ARCH_ADM', 'lines')
        return random.Random(7).sample(T, min(1000, len(T)))
    if cfg.startswith('PL_'):
        T, pl = plant_weave(units('PE', 'entries'), cfg[3:], 0.25, 11)
        json.dump(sorted(pl), open(os.path.join(CK, 'c1', cfg + '_planted.json'), 'w'))
        return T
    if cfg == 'NEG_TOPIC':
        # pe13 flat-topic model: tokens redrawn from a per-tablet cache (lam 0.4) or the
        # corpus unigram, numeral fields shuffled within tablet: no order anywhere.
        T = units('PE', 'entries')
        rng = random.Random(5)
        uni = Counter(s for t in T for u in t['u'] for s in u['toks'])
        keys = list(uni)
        w = np.array([uni[k] for k in keys], float)
        cum = np.cumsum(w / w.sum())
        out = []
        for t in T:
            pool, NU = [], []
            num = [(u['sys'], u['lead'], u['size']) for u in t['u']]
            rng.shuffle(num)
            for u, (sy, le, sz) in zip(t['u'], num):
                tk = []
                for _ in u['toks']:
                    if pool and rng.random() < 0.4:
                        tk.append(pool[rng.randrange(len(pool))])
                    else:
                        tk.append(keys[int(np.searchsorted(cum, rng.random()))])
                pool += tk
                NU.append(dict(u, toks=tk, sys=sy, lead=le, size=sz))
            out.append({'id': t['id'], 'u': NU})
        return out


def run(cfg, half):
    T = corpus(cfg)
    if half != 'ALL':
        r = random.Random(1234)
        ids = sorted(t['id'] for t in T)
        r.shuffle(ids)
        A = set(ids[: len(ids) // 2])
        T = [t for t in T if (t['id'] in A) == (half == 'A')]
    F = features(T, with_num=cfg.endswith('_LIN'))
    rng = random.Random(sum(map(ord, cfg + half)) * 7919)
    res = {}
    for k, seqs in F.items():
        obs = lag_agree(seqs)
        pp = _prep(seqs)
        S = null_stats(seqs, 1.0, NSHUF, rng, pp)
        rho = tune_rho(seqs, obs[0], rng, nsur=6, prep=pp)
        R = null_stats(seqs, rho, NADJ, rng, pp)
        res[k] = {'obs': obs.tolist(), 'shuf_m': S.mean(0).tolist(), 'shuf_s': S.std(0).tolist(),
                  'rho': rho, 'adj_m': R.mean(0).tolist(), 'adj_s': R.std(0).tolist(),
                  'adj_sur': R.tolist()}
        print(cfg, half, k, 'rho %.2f' % rho, 'obs', obs[:4].astype(int).tolist(),
              'adj', np.round(R.mean(0)[:4], 1).tolist(), flush=True)
    out = {'cfg': cfg, 'half': half, 'ntab': len(T), 'res': res}
    json.dump(out, open(os.path.join(CK, 'c1', '%s_%s.json' % (cfg, half)), 'w'))


if __name__ == '__main__':
    os.makedirs(os.path.join(CK, 'c1'), exist_ok=True)
    run(sys.argv[1], sys.argv[2])
