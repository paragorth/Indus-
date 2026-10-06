"""v72 cycle 3b: where does the residual identity link (g_id, previous token's identity beyond its last glyph)
sit? Split the held-out per-token gain by the length of the previous payload token and list the pairs that
earn most of it. Short previous tokens (<= 2 symbols: o, or, ar, s, r, ol, al ...) are the candidates for
word pieces separated by a written space (v16, v52)."""
import os, sys, json, math
from collections import defaultdict
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v72_lib as L
import v72_c2 as C2
import v72_c3 as C3


def run(name):
    fz = C2.freeze()
    S, plain, half = C2.corpus(name)
    X = L.extract(S, L.RULES[fz['rule']])
    tr = [p for p, h in zip(X, half) if h == 0]; te = [p for p, h in zip(X, half) if h == 1]
    r = L.ladder_generic(tr, te, C3.COMPS, C3.MODELS, W=20, seed=0)
    d = r['GEN+JBIG'] - r['GEN+JBIG+BIGJ']
    prev, pair = [], []
    for p in te:
        for l in p['lines']:
            for i, w in enumerate(l['w']):
                prev.append(len(l['w'][i - 1]) if i else 0); pair.append((l['w'][i - 1] if i else '^', w))
    prev = np.array(prev)
    out = dict(name=name, n=len(d), g_id=float(d.mean()))
    for lab, m in [('line_start', prev == 0), ('prev_len<=2', (prev > 0) & (prev <= 2)), ('prev_len>=3', prev >= 3)]:
        out[lab] = dict(share_tokens=float(m.mean()), gain_sum_share=float(d[m].sum() / d.sum()), mean=float(d[m].mean()))
    agg = defaultdict(float); cnt = defaultdict(int)
    for (a, b), x in zip(pair, d): agg[(a, b)] += x; cnt[(a, b)] += 1
    top = sorted(agg.items(), key=lambda kv: -kv[1])[:25]
    out['top_pairs'] = [(a, b, round(v, 1), cnt[(a, b)]) for (a, b), v in top]
    out['top25_share'] = float(sum(v for _, v in top) / d.sum())
    return out


if __name__ == '__main__':
    res = {n: run(n) for n in ['ZL', 'IT', 'ISI-merge', 'DEU-merge']}
    json.dump(res, open(os.path.join(L.CK, 'c3b.json'), 'w'), default=float)
    for n, o in res.items():
        print(n, round(o['g_id'], 4), {k: o[k] for k in ('line_start', 'prev_len<=2', 'prev_len>=3')},
              round(o['top25_share'], 2), o['top_pairs'][:12])
