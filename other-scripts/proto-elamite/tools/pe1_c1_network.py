#!/usr/bin/env python3
"""PE-1.1: designation x tablet network. Exact and one-sign-variant recurrence
against (N1) global token shuffle keeping slot lengths/tablets and sign
frequencies, and (N2) a Markov-2 middle generator trained on the same middles
(same slots). Person-like designations should recur across tablets MORE than
N2 and repeat WITHIN a tablet LESS than N2."""
import json, random, sys
from pe1_lib import *

R = int(sys.argv[1]) if len(sys.argv) > 1 else 200
T = load()
res = {}
for label, kw in [('len>=2', dict(minlen=2)), ('len>=3', dict(minlen=3)),
                  ('len>=2 prefix-stripped', dict(minlen=2, strip_prefix=True)),
                  ('len>=2 clean-only', dict(minlen=2, clean_only=True))]:
    DE = designation_entries(T, **kw)
    D = [(e['tablet'], e['des']) for e in DE]
    obs = recurrence_stats(D)
    obs['variant_pairs'] = variant_pairs(D)
    rng = random.Random(11)
    mk = Markov2([d for _, d in D])
    n1, n2 = [], []
    for r in range(R):
        a = shuffle_tokens(D, rng); s = recurrence_stats(a); s['variant_pairs'] = variant_pairs(a) if r < 50 else None; n1.append(s)
        b = markov_null(D, mk, rng); s = recurrence_stats(b); s['variant_pairs'] = variant_pairs(b) if r < 50 else None; n2.append(s)
    out = {'obs': obs, 'n_tablets': len({t for t, _ in D})}
    for k in ['types_on_2plus_tablets', 'tokens_in_recurring', 'within_tablet_dups', 'cross_tablet_type_pairs', 'variant_pairs', 'types']:
        for nm, nl in [('shuffle', n1), ('markov2', n2)]:
            vals = [x[k] for x in nl if x[k] is not None]
            out[f'{k}|{nm}'] = zp(obs[k], vals)
    # within/ cross ratio: share of repeated tokens that repeat on the SAME tablet
    def wshare(s):
        rep = s['within_tablet_dups'] + s['tokens_in_recurring']
        return s['within_tablet_dups'] / rep if rep else 0
    out['within_share|markov2'] = zp(wshare(obs), [wshare(x) for x in n2], greater=False)
    res[label] = out
    print('==', label, 'tokens', obs['tokens'], 'types', obs['types'], 'tablets', out['n_tablets'])
    for k, v in out.items():
        if '|' in k:
            print('  %-40s obs %-7s null %8.2f sd %6.2f z %6.2f p %.4f' % (k, round(v['obs'], 3), v['null_mean'], v['null_sd'], v['z'], v['p']))

# top recurring designations (len>=2), for the record
DE = designation_entries(T, minlen=2)
by = collections.defaultdict(set)
for e in DE:
    by[e['des']].add(e['tablet'])
top = sorted(by.items(), key=lambda kv: -len(kv[1]))[:30]
res['top_recurring'] = [[' '.join(d), len(ts), sorted(ts)[:8]] for d, ts in top]
print('top recurring:', [(a, b) for a, b, _ in res['top_recurring'][:20]])
json.dump(res, open(os.path.join(DATA, 'pe1_c1_network.json'), 'w'), indent=1)
