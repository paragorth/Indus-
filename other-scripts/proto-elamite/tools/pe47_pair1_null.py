"""pe47 cycle 3b: more nulls for the PE 'shared hapax-pair value' link (pair1): 200 N1 worlds and 200
N1s worlds (values shuffled within last token, system AND tablet entry-count class 1-2/3-5/6-10/11+)."""
import collections, json, random, sys
import numpy as np
from pe47_rows import get, pair1


def shuf(corpus, seed, strat):
    r = random.Random(seed)
    cls = lambda n: 0 if n <= 2 else 1 if n <= 5 else 2 if n <= 10 else 3
    groups = collections.defaultdict(list)
    for ti, t in enumerate(corpus):
        for ei, e in enumerate(t['ents']):
            k = (e[0][-1], e[2]) + ((cls(len(t['ents'])),) if strat else ())
            groups[k].append((ti, ei))
    new = [dict(t, ents=[list(e) for e in t['ents']]) for t in corpus]
    for L in groups.values():
        vals = [corpus[ti]['ents'][ei][1] for ti, ei in L]
        r.shuffle(vals)
        for (ti, ei), v in zip(L, vals):
            new[ti]['ents'][ei][1] = v
    return new


out = {}
for nm in sys.argv[1:]:
    c = get(nm, 0)
    real = pair1(c)
    res = {'real': real}
    for strat in (0, 1):
        v = [pair1(shuf(c, s, strat)) for s in range(200)]
        res['N1s' if strat else 'N1'] = dict(jac_mean=float(np.mean([x['jac'] for x in v])), jac_sd=float(np.std([x['jac'] for x in v])),
                                              p=(1 + sum(x['jac'] >= real['jac'] for x in v)) / 201.0,
                                              n_pairs=float(np.mean([x['n_pairs'] for x in v])))
        if 'sameyear' in real:
            res['N1s' if strat else 'N1']['sameyear_mean'] = float(np.mean([x['sameyear'] for x in v]))
            res['N1s' if strat else 'N1']['p_sameyear'] = (1 + sum(x['sameyear'] >= real['sameyear'] for x in v)) / 201.0
    out[nm] = res
    print(nm, json.dumps(res), flush=True)
json.dump(out, open('../data/pe47_ckpt/c3b.json', 'w'))
