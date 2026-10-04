"""v19 cycle 3, kill attempts on the q-up / a-down page gradient: (A) paragraph-last lines removed as well;
(B) position measured inside the paragraph (paragraphs of >= 5 lines), null permutes lines of the same length inside
the same paragraph. Line-interior words only, as in v19_drift2."""
import json, os
import numpy as np
from v19_lib import *

def load(name):
    recs = json.load(open(os.path.join(DATA, 'derived', name + '_lines.json')))
    out = []
    for r in recs:
        if r['ltype'] != 'P': continue
        ws = [w for w in r['words'] if w and '?' not in w]
        if not ws: continue
        out.append(dict(page=r['folio'], ps=r['para_start'], pe=r['para_end'], words=[glyphs(w) for w in ws]))
    return out

def test(lines, mode, R=300, seed=3):
    rng = np.random.default_rng(seed)
    recs = []
    if mode == 'A':
        order, by = pages_of(lines)
        for p in order:
            idx = by[p]; n = len(idx)
            if n < 6: continue
            for k, i in enumerate(idx):
                L = lines[i]
                if L['ps'] or L['pe']: continue
                recs.append(((p, len(L['words'])), k / (n - 1), [w[0] for w in L['words'][1:-1]]))
    else:
        para, pid = [], 0
        cur = []
        for i, L in enumerate(lines):
            if L['ps'] and cur:
                para.append(cur); cur = []
            cur.append(i)
        para.append(cur)
        for q, idx in enumerate(para):
            n = len(idx)
            if n < 5: continue
            for k, i in enumerate(idx):
                L = lines[i]
                if L['ps']: continue
                recs.append(((q, len(L['words'])), k / (n - 1), [w[0] for w in L['words'][1:-1]]))
    groups = {}
    for j, r in enumerate(recs): groups.setdefault(r[0], []).append(j)
    gl = [np.array(v) for v in groups.values() if len(v) > 1]
    RP = np.array([r[1] for r in recs])
    syms = ['q', 'a', 'l', 'k', 'y', 'd', 'T', 'o', 's', 'C', 'S']
    cnt = {g: np.array([r[2].count(g) for r in recs], float) for g in syms}
    def m(rp): return {g: (cnt[g] * rp).sum() / cnt[g].sum() for g in syms}
    real = m(RP); sims = {g: [] for g in syms}
    for t in range(R):
        rp = RP.copy()
        for v in gl: rp[v] = RP[rng.permutation(v)]
        mm = m(rp)
        for g in syms: sims[g].append(mm[g])
    return {g: float((real[g] - np.mean(sims[g])) / np.std(sims[g])) for g in syms}, len(recs)

if __name__ == '__main__':
    out = {}
    for name in ['ZL3b', 'IT2a']:
        L = load(name)
        for mode in ['A', 'B']:
            z, n = test(L, mode)
            out['%s_%s' % (name, mode)] = z
            print(name, mode, n, ' '.join('%s:%+.1f' % (g, v) for g, v in sorted(z.items(), key=lambda x: x[1])))
    json.dump(out, open(os.path.join(RES, 'c3_drift_kill.json'), 'w'))
