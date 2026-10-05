"""v58: re-test the best cycle-2 hits with 200 nulls, in both transcriptions (ZL3b, IT2a),
and record the alignment path (which Voynich unit <-> which source entry) for the record."""
import sys, os, json, random, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v58_lib as L

PAR = dict(lam=1.0, s=0.30, pm=0.30, g=0.5)
OUT = os.path.join(L.CK, 'replicate.json')
N = int(os.environ.get('NNULL', 200))

def main():
    T = json.load(open(os.path.join(L.CK, 'texts.json')))
    jobs = json.loads(sys.argv[1])
    res = json.load(open(OUT)) if os.path.exists(OUT) else {}
    rng = random.Random(5858)
    for tr in ['ZL3b', 'IT2a']:
        S = L.voynich_sequences(tr)
        for sk, ck in jobs:
            key = '%s|%s|%s' % (tr, sk, ck)
            if key in res: continue
            x, fol = S[sk]; y = [u['w'] for u in T[ck]['units']]
            t0 = time.time()
            obs, c = L.align(x, y, PAR)
            nx = [L.align(rng.sample(x, len(x)), y, PAR)[0] for _ in range(N)]
            nb = [L.align(L.block_shuffle(x, rng), y, PAR)[0] for _ in range(N)]
            zx, px = L.zp(obs, nx); zb, pb = L.zp(obs, nb)
            sc, steps = L.path(x, y, c, PAR)
            pairs = [(fol[i - 1], T[ck]['units'][j - 1]['t'], k, l) for i, j, k, l in steps if k and l]
            res[key] = dict(obs=round(obs, 2), c=c, zx=round(float(zx), 2), px=round(float(px), 4), zb=round(float(zb), 2), pb=round(float(pb), 4),
                            n_pairs=len(pairs), span_v=[pairs[0][0], pairs[-1][0]] if pairs else None,
                            span_src=[pairs[0][1], pairs[-1][1]] if pairs else None, pairs=pairs[:400])
            print(key, {k: v for k, v in res[key].items() if k != 'pairs'}, '%.0fs' % (time.time() - t0), flush=True)
            json.dump(res, open(OUT, 'w'))

if __name__ == '__main__':
    main()
