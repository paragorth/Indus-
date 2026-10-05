"""v58 cycle 2: the Voynich against every candidate source, by section.

For each Voynich length sequence (herbal pages / paragraphs, stars / bio / pharma / text paragraphs)
and each candidate work: best local length alignment (scale free), against
  zx: Voynich units shuffled within the section,
  zb: Voynich units block-shuffled (blocks of 8; keeps short-range autocorrelation),
  zy: candidate entries shuffled.
Results cached in data/v58_ckpt/cycle2.json (resume-safe).
"""
import sys, os, json, random, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v58_lib as L

NNULL = int(os.environ.get('NNULL', 40))
OUT = os.path.join(L.CK, os.environ.get('V58OUT', 'cycle2.json'))
PAR = dict(lam=1.0, s=0.30, pm=0.30, g=0.5)
SEQS = ['stars_paras', 'herbal_pages', 'herbalA_pages', 'herbalB_pages', 'herbal_paras', 'bio_paras', 'pharma_paras', 'text_paras']

def main():
    T = json.load(open(os.path.join(L.CK, 'texts.json')))
    S = L.voynich_sequences(os.environ.get('V58TR', 'ZL3b'))
    res = json.load(open(OUT)) if os.path.exists(OUT) else {}
    rng = random.Random(582)
    cands = sorted(T, key=lambda k: len(T[k]['units']))
    for sk in SEQS:
        x = S[sk][0]
        for ck in cands:
            key = sk + '|' + ck
            if key in res: continue
            y = [u['w'] for u in T[ck]['units']]
            if len(y) < 20: continue
            t0 = time.time()
            obs, c = L.align(x, y, PAR)
            nx = [L.align(rng.sample(x, len(x)), y, PAR)[0] for _ in range(NNULL)]
            nb = [L.align(L.block_shuffle(x, rng), y, PAR)[0] for _ in range(NNULL)]
            ny = [L.align(x, rng.sample(y, len(y)), PAR)[0] for _ in range(NNULL // 2)]
            zx, px = L.zp(obs, nx); zb, pb = L.zp(obs, nb); zy, py = L.zp(obs, ny)
            res[key] = dict(obs=round(obs, 2), c=c, zx=round(float(zx), 2), zb=round(float(zb), 2), zy=round(float(zy), 2),
                            px=round(float(px), 4), pb=round(float(pb), 4), py=round(float(py), 4), n=len(x), m=len(y))
            print(key, res[key], '%.0fs' % (time.time() - t0), flush=True)
            json.dump(res, open(OUT, 'w'))

if __name__ == '__main__':
    main()
