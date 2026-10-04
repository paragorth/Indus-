"""v19 cycle 3 check of the page drift: (1) do lines get shorter down the page? (2) does the q/a drift survive when only
line-interior words (not first, not last) are counted and the null permutes whole lines only among lines of the SAME
word count on the page (line length and line-edge vocabulary held fixed)?"""
import numpy as np, json, os
from v19_lib import *
from v19_cycle1 import corpora

def run(lines, R=1000, seed=2):
    rng = np.random.default_rng(seed)
    order, by = pages_of(lines)
    pos_len = []
    recs = []  # (page, line rel pos, nwords, interior first glyphs)
    for p in order:
        idx = [i for i in by[p]]
        n = len(idx)
        if n < 6: continue
        for k, i in enumerate(idx):
            L = lines[i]
            rp = k / (n - 1)
            pos_len.append((rp, len(L['words'])))
            if L['para_start']: continue
            recs.append((p, rp, len(L['words']), [w[0] for w in L['words'][1:-1]]))
    a = np.array(pos_len); r_len = np.corrcoef(a[:, 0], a[:, 1])[0, 1]
    # stratified null: permute rel. positions among lines of same page and same length
    groups = {}
    for j, (p, rp, nw, gs) in enumerate(recs):
        groups.setdefault((p, nw), []).append(j)
    RP = np.array([r[1] for r in recs])
    syms = ['q', 'd', 'T', 'K', 'C', 'S', 'o', 's', 'y', 'k', 'l', 'a', 't', 'r', 'e', 'p', 'f']
    def means(rp):
        out = {}
        for g in syms:
            num = den = 0.0
            for j, r in enumerate(recs):
                c = r[3].count(g)
                num += c * rp[j]; den += c
            out[g] = num / den if den else np.nan
        return out
    real = means(RP)
    sims = {g: [] for g in syms}
    gl = [np.array(v) for v in groups.values() if len(v) > 1]
    for t in range(R):
        rp = RP.copy()
        for v in gl:
            rp[v] = RP[rng.permutation(v)]
        m = means(rp)
        for g in syms: sims[g].append(m[g])
    z = {g: (real[g] - np.nanmean(sims[g])) / np.nanstd(sims[g]) for g in syms if np.nanstd(sims[g]) > 0}
    return r_len, z

if __name__ == '__main__':
    C, _ = corpora()
    out = {}
    for c in ['ZL', 'IT', 'LatXVI']:
        r_len, z = run(C[c], R=300)
        out[c] = {'r_len_pos': r_len, 'z': z}
        print(c, 'corr(line position, words per line) = %.3f' % r_len, ' '.join('%s:%+.1f' % (g, v) for g, v in sorted(z.items(), key=lambda x: x[1])))
    json.dump(out, open(os.path.join(RES, 'c3_drift_stratified.json'), 'w'))
