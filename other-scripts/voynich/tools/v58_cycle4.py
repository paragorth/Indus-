"""v58 cycle 4: CHANGE THE MEASURING STICK. If Voynich word breaks are not source word breaks
(spaces inserted by a generator-like surface), a word count is the wrong ruler. Measure each Voynich
unit four ways and each source entry two ways, and take the best of the combinations:
  Voynich: words, glyph units (vlib.glyphs), lines, 'heavy' glyphs (gallows + benches only: a
           would-be nomenclator/numeral layer)
  source: words, letters
Score = max over the 8 combinations of the local alignment. Null: the SAME max-over-8 under a joint
block shuffle / plain shuffle of the Voynich units (all rulers permuted together), so the search over
rulers is paid for. Planted control: encoded Celsus (glyph ruler = letters*1.5, word ruler = re-cut
tokens) found through the glyph ruler.
"""
import sys, os, json, random, time
from collections import defaultdict
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v58_lib as L
import vlib

PAR = dict(lam=1.0, s=0.30, pm=0.30, g=0.5)
OUT = os.path.join(L.CK, 'cycle4.json')
NNULL = int(os.environ.get('NNULL', 30))
HEAVY = set('ktpfKTPFCS')

def units4(name='ZL3b'):
    recs = json.load(open(os.path.join(L.VD, 'data', 'derived', name + '_lines.json')))
    paras, cur, fol = [], None, None
    for r in recs:
        if r['ltype'] != 'P': continue
        if r['para_start'] or r['folio'] != fol or cur is None:
            if cur: paras.append(cur)
            cur = {'folio': r['folio'], 'illus': r['illus'], 'lang': r['lang'], 'm': np.zeros(4)}
        fol = r['folio']
        gs = [vlib.glyphs(w) for w in r['words']]
        cur['m'] += [len(r['words']), sum(len(g) for g in gs), 1, sum(1 for g in gs for u in g if u in HEAVY)]
        if r.get('para_end'): paras.append(cur); cur = None
    if cur: paras.append(cur)
    paras = [p for p in paras if p['m'][0] > 0]
    paras.sort(key=lambda p: L.fnum(p['folio']))
    seqs = {}
    for code, nm in [('S', 'stars_paras'), ('B', 'bio_paras'), ('P', 'pharma_paras'), ('H', 'herbal_paras')]:
        seqs[nm] = np.array([p['m'] for p in paras if p['illus'] == code])
    pg = defaultdict(lambda: np.zeros(4)); order = []
    for p in paras:
        if p['illus'] != 'H': continue
        if p['folio'] not in pg: order.append(p['folio'])
        pg[p['folio']] += p['m']
    seqs['herbal_pages'] = np.array([pg[f] for f in order])
    return seqs

def best8(X, ys):
    best = -1
    for j in range(X.shape[1]):
        x = np.maximum(X[:, j], 1)
        for y in ys:
            v = L.align(x, y, PAR)[0]
            if v > best: best = v
    return best

def perm_rows(X, idx):
    return X[idx]

def block_idx(n, rng, b=8):
    return np.array(L.block_shuffle(list(range(n)), rng, b))

def run(key, X, ys, rng, res):
    if key in res: return
    t0 = time.time()
    obs = best8(X, ys)
    nx = [best8(X[rng.sample(range(len(X)), len(X))], ys) for _ in range(NNULL)]
    nb = [best8(X[block_idx(len(X), rng)], ys) for _ in range(NNULL)]
    zx, px = L.zp(obs, nx); zb, pb = L.zp(obs, nb)
    res[key] = dict(obs=round(obs, 2), zx=round(float(zx), 2), zb=round(float(zb), 2), px=round(float(px), 3), pb=round(float(pb), 3))
    print(key, res[key], '%.0fs' % (time.time() - t0), flush=True)
    json.dump(res, open(OUT, 'w'))

def main():
    T = json.load(open(os.path.join(L.CK, 'texts.json')))
    S = units4()
    res = json.load(open(OUT)) if os.path.exists(OUT) else {}
    rng = random.Random(584)
    # planted: encoded Celsus poured into stars-like units; Voynich-side rulers = (re-cut tokens, glyphs, lines, heavy)
    for rep in range(2):
        us = T['celsus_lat']['units']
        st = rng.randrange(len(us) - 120); win = us[st:st + 120]
        rows = []
        for u in win:
            g = u['c'] + np.random.default_rng(rng.randrange(1 << 30)).binomial(u['c'], 0.5)
            tok = max(1, int(g / rng.uniform(3, 9)))         # word breaks unrelated to source words (noisy re-cut)
            rows.append([tok, g * np.exp(rng.gauss(0, 0.15)), max(1, round(g / 40)), g * 0.2])
        X = np.array(rows, float)
        ys = [[u['w'] for u in T['celsus_eng']['units']], [u['c'] for u in T['celsus_eng']['units']]]
        run('PLANT|celsus_lat->eng|%d' % rep, X, ys, rng, res)
        ys = [[u['w'] for u in T['apicius_eng']['units']], [u['c'] for u in T['apicius_eng']['units']]]
        run('PLANTNEG|celsus_lat->apicius_eng|%d' % rep, X, ys, rng, res)
    cands = [k for k in T if T[k]['genre'] in ('herbal', 'recipe', 'astro', 'baths', 'regimen', 'botany', 'medical') and len(T[k]['units']) >= 20]
    cands.sort(key=lambda k: len(T[k]['units']))
    for sk in ['pharma_paras', 'bio_paras', 'herbal_pages', 'stars_paras', 'herbal_paras']:
        for ck in cands:
            ys = [[u['w'] for u in T[ck]['units']], [max(1, u['c']) for u in T[ck]['units']]]
            run(sk + '|' + ck, S[sk], ys, rng, res)

if __name__ == '__main__':
    main()
