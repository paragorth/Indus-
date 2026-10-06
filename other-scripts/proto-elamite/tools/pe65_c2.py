#!/usr/bin/env python3
"""PE-65 cycle 2: rejection ABC with held-out units.
For each held-out unit h (3 Susa batches, 4 outposts), accept the 500 bank worlds nearest to the corpus on every
statistic except h's own 13 and its pair overlaps; predict h's profile (median of accepted); error = mean |z|.
Done for all worlds (posterior), random prior worlds, and within each world type (type-restricted posterior):
the type whose worlds best predict held-out units is the held-out model choice.
Corpora: PE publication batches, PE random batches, 5 unit-label shuffles, Ur III and Linear B (3 replicates each),
planted Susa-state and planted null worlds (3 each)."""
import sys, os, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pe65_common import *
from pe65_run import load_bank
from pe65_fit import clean
from pe65_c1 import plant

NACC = 500
Un = NB + K - 1


def unit_idx(h):
    own = list(range(h * 13, h * 13 + 13))
    p = Un * 13; pr = []
    for a in range(Un):
        for b in range(a + 1, Un):
            if a == h or b == h:
                pr.append(p)
            p += 1
    return own, pr


def main():
    S, T, meta = load_bank('PE')
    S = clean(S)
    typ = T[:, 0].astype(int)
    med = np.median(S, 0); mad = np.median(np.abs(S - med), 0) * 1.4826
    mad[mad < 1e-3] = np.maximum(S.std(0)[mad < 1e-3], 1e-3)
    Z = ((S - med) / mad).astype(np.float32)
    rng = np.random.RandomState(1)
    corp = {'PE_pub': stats(pe_docs_all('pub')), 'PE_rand': stats(pe_docs_all('rand'))}
    base_docs = pe_docs_all('pub')
    for i in range(5):
        corp['PE_shuf%d' % i] = stats(shuffle_units(base_docs, 'c2-%d' % i))
    for nm in ('UR', 'LB'):
        for r in range(3):
            corp['%s%d' % (nm, r)] = stats(control_docs(nm, r))
    for kind in ('susa_state', 'null', 'peers'):
        for r in range(3):
            corp['PL_%s%d' % (kind, r)] = plant(kind, 100 + r)[0]
    prior = rng.choice(len(S), NACC, replace=False)
    tix = {t: np.where(typ == t)[0] for t in range(7)}
    res = {}
    for name, s in corp.items():
        z = ((clean(s[None, :])[0] - med) / mad).astype(np.float32)
        d_all = np.abs(Z - z)
        dfull = d_all.mean(1)
        acc = np.argsort(dfull)[:NACC]
        post_type = {TYPES[t]: round(float((typ[acc] == t).mean()), 3) for t in range(7)}
        role0 = T[acc, 2].astype(int)
        r = dict(type=post_type, susa_centre=round(float(((role0 == 1) & np.isin(typ[acc], [0, 1, 3])).mean()), 3),
                 centre=[round(float((T[acc, 2 + k] == 1).mean()), 3) for k in range(K)], held={})
        for h in range(Un):
            own, pr = unit_idx(h)
            keep = np.ones(S.shape[1], bool); keep[own] = False; keep[pr] = False
            dk = d_all[:, keep].mean(1)
            tgt = own + pr
            def err(ix):
                pred = np.median(Z[ix][:, tgt], 0)
                return float(np.abs(pred - z[tgt]).mean())
            a = np.argsort(dk)[:NACC]
            e = dict(post=err(a), prior=err(prior))
            for t in range(7):
                ii = tix[t]
                e[TYPES[t]] = err(ii[np.argsort(dk[ii])[:NACC]])
            r['held'][UNITS[h]] = {k: round(v, 3) for k, v in e.items()}
        # held-out model choice: mean error per type over units, and rank
        mt = {TYPES[t]: round(float(np.mean([r['held'][u][TYPES[t]] for u in UNITS])), 3) for t in range(7)}
        r['type_heldout_err'] = mt
        r['best_type'] = min(mt, key=mt.get)
        r['gain'] = round(float(np.mean([r['held'][u]['prior'] - r['held'][u]['post'] for u in UNITS])), 3)
        res[name] = r
        print(name, r['type'], 'susaC', r['susa_centre'], 'gain', r['gain'], 'best', r['best_type'], mt, flush=True)
    jdump(res, 'c2_results.json')


if __name__ == '__main__':
    main()
