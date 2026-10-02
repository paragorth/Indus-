"""N4: line modes. N3 showed two line types inside paragraphs: 'q-type' lines (q, gallows, e, d, y: qokedy-like)
and 'a-type' lines (a, i, r, n: aiin/ar-like). Score each line: share of its words that start with q minus share
containing 'ai' or 'ar'. Then: (1) is the score bimodal? (2) do consecutive lines persist (drift), alternate
(fixed key schedule) or vary freely? Statistic: lag-1 and lag-2 correlation of the score between lines of the same
paragraph, vs paragraph-internal line-order shuffles (1000x). By Currier language."""
import sys, os, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vlib

def score(ws):
    n = len(ws)
    return (sum(w.startswith('q') for w in ws) - sum(('ai' in w) or ('ar' in w) for w in ws)) / n

def paras(lines):
    P, cur = [], []
    for L in lines:
        if L.get('para_start') and cur: P.append(cur); cur = []
        cur.append(L)
    if cur: P.append(cur)
    return [p for p in P if len(p) >= 4]

def lagcorr(P, lag):
    xs, ys = [], []
    for p in P:
        for i in range(len(p) - lag): xs.append(p[i]); ys.append(p[i+lag])
    n = len(xs); mx = sum(xs)/n; my = sum(ys)/n
    vx = sum((x-mx)**2 for x in xs); vy = sum((y-my)**2 for y in ys)
    return sum((x-mx)*(y-my) for x, y in zip(xs, ys)) / (vx*vy)**.5

vl = vlib.load_voynich('ZL3b', drop_uncertain=True)
for lang in ('A', 'B', None):
    sel = [L for L in vl if lang is None or L.get('lang') == lang]
    P = [[score(L['words']) - sum(score(x['words']) for x in p)/len(p) for L in p] for p in paras(sel)]  # demeaned per paragraph
    rng = random.Random(1); out = []
    for lag in (1, 2):
        o = lagcorr(P, lag); s = []
        for _ in range(1000): s.append(lagcorr([rng.sample(p, len(p)) for p in P], lag))
        m = sum(s)/len(s); sd = (sum((x-m)**2 for x in s)/len(s))**.5
        out.append((lag, round(o, 3), round(m, 3), round((o-m)/sd, 1)))
    allv = [x for p in P for x in p]
    print(f'Currier {lang or "all"}: paragraphs {len(P)}, lines {len(allv)}; lag (obs, null, z):', out)
