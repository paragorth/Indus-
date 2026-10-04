"""pe14 cycle 2: HIDDEN WEAVE MODELS. Cyclic HMMs (period K = 2-12) vs a free HMM
(any transitions, more capacity) vs a flat mixture, fitted by EM with random
restarts, scored by 5-fold held-out log-likelihood on whole tablets.
Observation per unit: last sign, first sign (top 12 + other), numeral size bin,
entry length, numeral system (+ numbered/unnumbered for line corpora).
Nulls: SHUF (within-tablet shuffle) and DIP (joint adjacency null: units drawn one
by one with weight rho^(signs shared with the previous unit), rho tuned to the
observed lag-1 sharing -- the pe13 adjacent-avoidance effect alone).
usage: python3 pe14_cycle2.py <config> [seed]
configs: PE_ENT, PE_SHUF, PE_DIP, PL_P2LAST, PL_P3SIZE, PL_P4FIRST, UR3_LIN, ARCH_LIN, NEG_TOPIC
"""
import json, math, os, random, sys
from collections import Counter
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from pe14_common import CK, units  # noqa
from pe14_cycle1 import corpus  # noqa
import pe14_hmm as H  # noqa

RESTARTS, FOLDS = 8, 5
MODELS = [('MIX', k) for k in range(1, 7)] + [('FREE', k) for k in range(2, 7)] + [('CYC', k) for k in range(2, 13)]


def shared(a, b):
    return len(set(a['toks']) & set(b['toks']))


def lag1_shared(T):
    return sum(shared(t['u'][i], t['u'][i + 1]) for t in T for i in range(len(t['u']) - 1))


def dip_sample(T, rho, rng):
    out = []
    for t in T:
        rem = list(t['u'])
        seq = [rem.pop(rng.randrange(len(rem)))]
        while rem:
            w = [rho ** shared(seq[-1], u) for u in rem]
            r = rng.random() * sum(w)
            acc = 0
            for j, wj in enumerate(w):
                acc += wj
                if acc >= r:
                    break
            seq.append(rem.pop(j))
        out.append({'id': t['id'], 'u': seq})
    return out


def dip_null(T, rng):
    obs = lag1_shared(T)
    lo, hi = 0.05, 5.0
    for _ in range(12):
        mid = math.sqrt(lo * hi)
        m = np.mean([lag1_shared(dip_sample(T, mid, rng)) for _ in range(3)])
        if m < obs:
            lo = mid
        else:
            hi = mid
    rho = math.sqrt(lo * hi)
    return dip_sample(T, rho, rng), rho, obs


def encode(T, ref, with_num):
    def top(fn, k=12):
        c = Counter(fn(u) for t in ref for u in t['u'])
        return {v: i for i, (v, _) in enumerate(c.most_common(k))}
    mp = [(lambda u: u['toks'][-1], top(lambda u: u['toks'][-1])),
          (lambda u: u['toks'][0], top(lambda u: u['toks'][0])),
          (lambda u: u['size'], top(lambda u: u['size'], 6)),
          (lambda u: min(len(u['toks']), 5), top(lambda u: min(len(u['toks']), 5), 5)),
          (lambda u: u['sys'], top(lambda u: u['sys'], 5))]
    if with_num:
        mp.append((lambda u: u['num'], {0: 0, 1: 1}))
    V = [len(m) + 1 for _, m in mp]
    seqs = [[tuple(m.get(fn(u), len(m)) for fn, m in mp) for u in t['u']] for t in T]
    return seqs, V


def main(cfg, seed):
    rng = random.Random(seed)
    nr = np.random.RandomState(seed)
    info = {}
    if cfg in ('PE_SHUF', 'PE_DIP'):
        T0 = corpus('PE_ENT')
        if cfg == 'PE_SHUF':
            T = [{'id': t['id'], 'u': rng.sample(t['u'], len(t['u']))} for t in T0]
        else:
            T, rho, obs = dip_null(T0, rng)
            info = {'rho': rho, 'lag1_obs': obs, 'lag1_null': lag1_shared(T)}
    else:
        T = corpus(cfg)
    T = [t for t in T if len(t['u']) >= 3]
    seqs, V = encode(T, T, cfg.endswith('_LIN'))
    idx = np.random.RandomState(99).permutation(len(seqs))
    res = {f'{k}{K}': 0.0 for k, K in MODELS}
    extra = {}
    nunits = sum(len(s) for s in seqs)
    for f in range(FOLDS):
        te = set(idx[f::FOLDS].tolist())
        tr = [seqs[i] for i in range(len(seqs)) if i not in te]
        ts = [seqs[i] for i in sorted(te)]
        Xtr, Mtr = H.pack(tr, len(V))
        Xte, Mte = H.pack(ts, len(V))
        for kind, K in MODELS:
            best = None
            for r in range(RESTARTS if K > 1 else 1):
                m = H.fit(Xtr, Mtr, V, kind, K, nr)
                if best is None or m['train_ll'] > best['train_ll']:
                    best = m
            res[f'{kind}{K}'] += H.score(Xte, Mte, best)
            if kind == 'CYC':
                extra.setdefault(f'eps{K}', []).append(best['eps'])
        print(cfg, seed, 'fold', f, flush=True)
    bits = {k: -v / nunits / math.log(2) for k, v in res.items()}
    out = {'cfg': cfg, 'seed': seed, 'ntab': len(T), 'nunits': nunits, 'bits': bits, 'eps': extra, 'info': info}
    json.dump(out, open(os.path.join(CK, 'c2', f'{cfg}_{seed}.json'), 'w'), indent=1)
    b = bits
    bm = min(b[f'MIX{k}'] for k in range(1, 7))
    bf = min(b[f'FREE{k}'] for k in range(2, 7))
    print(cfg, 'bestMIX %.4f bestFREE %.4f' % (bm, bf),
          ' '.join('CYC%d %.4f' % (k, b[f'CYC{k}']) for k in range(2, 13)))


if __name__ == '__main__':
    os.makedirs(os.path.join(CK, 'c2'), exist_ok=True)
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 0)
