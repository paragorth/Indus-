"""v43 cycle 1 report: island runs above the uniform-generator ceiling; gap ratio inside vs outside the island."""
import numpy as np, random
from collections import Counter
import v43_lib as L
R = L.load('cycle1.json'); P = L.voynich('ZL3b'); S = L.stream_of_pages(P)
cs = L.lscore(L.probs(L.load('F_ZL.json'))); cp = [S[i*100+50][1] for i in range(len(cs))]
out = {}
for W in (6, 12, 24):
    ceil = max(R['uniform'][u][str(W)]['max'] for u in R['uniform'])
    ws = L.window_scores(cs, W); runs = L.runs_above(ws, ceil)
    rr = []
    for a, b in runs:
        pg = sorted({cp[j] for j in range(a, b + W)})
        rr.append(dict(chunks=(a, b + W - 1), ntok=(b + W - a) * 100, pages=[P[pg[0]]['id'], P[pg[-1]]['id']],
                       secs=dict(Counter(P[cp[j]]['sec'] for j in range(a, b + W))), hands=dict(Counter(P[cp[j]]['hand'] for j in range(a, b + W))),
                       quires=dict(Counter(P[cp[j]]['quire'] for j in range(a, b + W)))))
    out[W] = dict(ceil=ceil, runs=rr)
    print(W, 'ceiling', round(ceil, 3), rr)
# gap ratio: island chunks vs other BB, vs rest, vs languages, at 1200 tokens
bat = L.load('bat_c1.json')
isl = set()
for r in out[12]['runs']: isl |= set(range(r['chunks'][0], r['chunks'][1] + 1))
g_in, g_out = [], []
for w in bat['ZL']:
    c0 = w['start'] // 100
    (g_in if len(set(range(c0, c0 + 12)) & isl) >= 6 else g_out).append(w['gap'])
gl = [w['gap'] for n in L.LANGS for w in bat[n]]
print('gap island windows', np.round(g_in, 3), 'rest median', round(np.median(g_out), 3), 'languages p5-p95', np.round(np.percentile(gl, [5, 50, 95]), 3))
# gap ratio on the island itself (contiguous tokens) vs 20 random same-size contiguous Voynich stretches and languages
r12 = max(out[12]['runs'], key=lambda r: r['ntok']) if out[12]['runs'] else None
if r12:
    a, b = r12['chunks']; sw = S[a*100:(b+1)*100]
    gi = np.mean([L.gap_ratio_tokens(L.window_lines(sw), s) for s in range(3)])
    rng = random.Random(0); n = len(sw); gr = []
    for _ in range(20):
        s0 = rng.randrange(0, len(S) - n); gr.append(L.gap_ratio_tokens(L.window_lines(S[s0:s0+n]), 0))
    glang = []
    for nm in L.LANGS:
        SL = L.stream_of_lines(L.corpus_lines(nm, maxtok=n + 10))[:n]
        glang.append(L.gap_ratio_tokens(L.window_lines(SL), 0))
    print('island', r12['pages'], n, 'gap', round(gi, 3), 'random Voynich stretches', np.round(np.percentile(gr, [5, 50, 95]), 3), 'languages', np.round(sorted(glang), 3))
    out['island_gap'] = dict(gap=gi, rand=gr, lang=glang, ntok=n)
L.save('report1.json', out)
