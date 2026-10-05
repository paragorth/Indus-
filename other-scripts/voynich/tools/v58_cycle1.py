"""v58 cycle 1: calibrate the shape-alignment machine on known pairs before touching the Voynich.

Positive controls: the same work in two languages, one side passed through a simulated opaque verbose
cipher with re-segmented tokens, entry drops / merges / splits and layout jitter, cut to a window of the
size of a Voynich section. Must find its partner (z vs shuffled nulls).
Negative controls: the same encoded window against a different work. Must not.
"""
import sys, os, json, random, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v58_lib as L

T = json.load(open(os.path.join(L.CK, 'texts.json')))
OUT = os.path.join(L.CK, 'cycle1.json')
NNULL = int(os.environ.get('NNULL', 60))

PAIRS = [('celsus_lat', 'celsus_eng'), ('apicius_lat', 'apicius_eng'), ('psalms_he', 'psalms_en'),
         ]  # pliny run separately (slow)
NEG = {'celsus_lat': 'apicius_eng', 'apicius_lat': 'celsus_eng', 'psalms_he': 'forme_of_cury', 'pliny_nh_lat': 'culpeper'}
SIZES = [44, 87, 128, 292]
SETTINGS = {'s0.30': dict(lam=1.0, s=0.30, pm=0.30, g=0.5), 's0.20': dict(lam=1.0, s=0.20, pm=0.30, g=0.5),
            's0.45': dict(lam=1.0, s=0.45, pm=0.30, g=0.5)}

def test(x, y, par, rng):
    obs, c = L.align(x, y, par)
    nx = [L.align(rng.sample(x, len(x)), y, par)[0] for _ in range(NNULL)]
    ny = [L.align(x, rng.sample(y, len(y)), par)[0] for _ in range(NNULL // 2)]
    zx, px = L.zp(obs, nx); zy, py = L.zp(obs, ny)
    return dict(obs=round(obs, 2), c=c, zx=round(zx, 2), px=round(px, 4), zy=round(zy, 2), py=round(py, 4))

def main():
    rng = random.Random(58)
    res = json.load(open(OUT)) if os.path.exists(OUT) else {}
    for a, b in PAIRS:
        ua = T[a]['units']; ub = [u['w'] for u in T[b]['units']]
        for n in SIZES:
            for sk, par in SETTINGS.items():
                for rep in range(2):
                    key = '%s|%s|%d|%s|%d' % (a, b, n, sk, rep)
                    if key in res: continue
                    enc = L.encode_lengths(ua, rng)
                    if len(enc) > n:
                        st = rng.randrange(len(enc) - n); enc = enc[st:st + n]
                    t0 = time.time()
                    pos = test(enc, ub, par, rng)
                    neg = test(enc, [u['w'] for u in T[NEG[a]]['units']], par, rng)
                    res[key] = dict(pos=pos, neg=neg, n=len(enc))
                    print(key, 'POS', pos, 'NEG', neg, '%.0fs' % (time.time() - t0), flush=True)
                    json.dump(res, open(OUT, 'w'))

if __name__ == '__main__':
    main()
