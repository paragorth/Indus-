#!/usr/bin/env python3
"""la47 cycle 2b: is a commodity sign glued to the word before it?  Continuous gap (left edge minus
previous item's right edge, in median sign widths) between consecutive items on one physical line,
both from SigLA boxes.  Compare pairs word->commodity (la45 COMMODITY), word->other word, word->header.
Null: gaps permuted among the word-preceded pairs within each page (2,000 reps).  Size control: same test
with the gap divided by the mean width of the two items instead of the page median width.
Output data/la47_ckpt/c2b.json"""
import json, os, collections
import numpy as np
import la47_common as C, la47_c2 as M


def pairs(P):
    out = []
    for pi, p in enumerate(P):
        it = [i for i in p['items'] if i['k'] != 'd']
        for a, b in zip(it, it[1:]):
            if a['k'] == 'w' and b['k'] == 'w' and a.get('box') and b.get('box') and a['pl2'] == b['pl']:
                ga = (b['box'][0][0] - (a['box'][-1][0] + a['box'][-1][2]))
                wa = np.mean([x[2] for x in a['box']] + [x[2] for x in b['box']])
                out.append((pi, C.la45_class(b['id']), ga / p['mw'], ga / wa, b['logo'], C.la45_class(a['id'])))
    return out


def test(D, sel, col, rng, reps=2000):
    pid = np.array([d[0] for d in D]); g = np.array([d[col] for d in D]); m = np.array([sel(d) for d in D])
    obs = np.median(g[m]) - np.median(g[~m]); null = []
    groups = [np.where(pid == p)[0] for p in np.unique(pid)]
    for _ in range(reps):
        gp = g.copy()
        for k in groups: gp[k] = gp[rng.permutation(k)]
        null.append(np.median(gp[m]) - np.median(gp[~m]))
    null = np.array(null)
    return {'n': int(m.sum()), 'n_rest': int((~m).sum()), 'median_sel': round(float(np.median(g[m])), 3), 'median_rest': round(float(np.median(g[~m])), 3),
            'diff': round(float(obs), 3), 'P_two_sided': round(float((np.abs(null - null.mean()) >= abs(obs - null.mean())).mean()), 4)}


if __name__ == '__main__':
    rng = np.random.default_rng(11)
    P = M.geo_pages(); D = pairs(P)
    out = {'pairs': len(D), 'by_class': dict(collections.Counter(d[1] for d in D))}
    out['commodity_vs_rest_pagewidth'] = test(D, lambda d: d[1] == 'commodity', 2, rng)
    out['commodity_vs_rest_pairwidth'] = test(D, lambda d: d[1] == 'commodity', 3, rng)
    out['logogram_vs_rest'] = test(D, lambda d: d[4], 2, rng)
    out['logo_commodity_vs_other_logo'] = test([d for d in D if d[4]], lambda d: d[1] == 'commodity', 2, rng)
    out['syllabic_commodity_vs_other_syllabic'] = test([d for d in D if not d[4]], lambda d: d[1] == 'commodity', 2, rng)
    out['header_vs_rest'] = test(D, lambda d: d[1] == 'header', 2, rng)
    out['after_header_vs_rest'] = test(D, lambda d: d[5] == 'header', 2, rng)
    json.dump(out, open(os.path.join(C.CK, 'c2b.json'), 'w'), indent=1)
    for k, v in out.items(): print(k, v)
