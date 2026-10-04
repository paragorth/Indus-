#!/usr/bin/env python3
"""LA-42 cycle 4: internal consistency of the frozen placements. For every unvalued sign u and every occupied grid
cell X (a known LA sign), read u as X in each of u's words and count the readings identical to a different word type
attested elsewhere in LA (u's own words excluded). Empty-cell and NEW-row placements cannot create identities, so the
check only bears on 'u is a doublet of X'. Null: the same count for every occupied cell (percentile of the model's
best occupied cell) and, as a reference, the same statistic for known signs read as every other known sign."""
import os, sys, json, collections
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import la42_common as G
import la42_c3 as C3

W = [r['w'] for r in G.la_units()]
V = G.known_values(W)
types = collections.Counter(W)
inv = {}
for s, cv in V.items():
    inv[cv] = s


def matches(u, X):
    hits = []
    for w in set(x for x in W if u in x):
        r = tuple(X if s == u else s for s in w)
        if r != w and r in types:
            hits.append('-'.join(r))
    return hits


def main():
    frozen = json.load(open(os.path.join(G.CK, 'c3_frozen.json')))
    lines = []
    # reference: known signs re-read as every other known sign
    ref = []
    for h in sorted(V):
        n = [len(matches(h, X)) for X in sorted(V) if X != h]
        ref.append(np.mean(n))
    lines.append(f"reference: a known sign re-read as another known sign creates {np.mean(ref):.2f} identities per substitution on average (max over signs {max(ref):.2f})")
    for u in C3.UNK:
        cnt = {X: len(matches(u, X)) for X in sorted(V)}
        best = [x for x in sorted(cnt, key=lambda k: -cnt[k]) if cnt[x] > 0][:4]
        f = frozen[u]['ALL']
        occ_pred = [t for t, p in f['top'] if not t.startswith('NEW') and t in V]
        mp = occ_pred[0] if occ_pred else None
        pct = (np.mean([cnt[X] < cnt[mp] for X in cnt]) + 0.5 * np.mean([cnt[X] == cnt[mp] for X in cnt])) if mp else float('nan')
        lines.append(f"{u:5s} top-1 {f['top'][0][0]} | best occupied cell in model top-5: {mp} ({cnt.get(mp, 0)} identities, percentile {pct:.2f}) | "
                     f"most identities: " + ', '.join(f"{X} {cnt[X]} ({'; '.join(matches(u, X)[:3])})" for X in best))
        print(lines[-1], flush=True)
    open(os.path.join(G.CK, 'c4_report.txt'), 'w').write('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
