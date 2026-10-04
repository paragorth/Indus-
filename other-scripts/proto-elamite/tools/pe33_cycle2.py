"""pe33 cycle 2: pooled test. Do tablets sealed by DIFFERENT seals that show the same kind of picture
write more similar content than tablets of different seals with different pictures?
Statistic per motif feature m: mean content similarity (cosine on tf-idf sign + system + header vectors)
over cross-seal pairs both-with-m minus cross-seal pairs with exactly one having m. Same-seal pairs are
excluded. Null: motif vectors permuted among seals within volume (seal-block). Controls: planted
3-sign bundle on BOVID tablets; Ur III seal legends at PE size (same statistic)."""
import json, sys, os, collections
import numpy as np
from pe33_common import load, CK, perm_seal, MOTIFS
sys.path.insert(0, os.path.dirname(__file__))



def vecs(contents):
    df = collections.Counter(f for c in contents for f in c)
    F = sorted(f for f, c in df.items() if c >= 2 and not f.startswith('Z:'))
    n = len(contents)
    idf = np.array([np.log(n / df[f]) for f in F])
    V = np.array([[f in c for f in F] for c in contents], dtype=float) * idf
    V /= np.linalg.norm(V, axis=1, keepdims=True) + 1e-9
    return V @ V.T


def stat(S, Y, seal, mot):
    n = len(seal)
    cross = seal[:, None] != seal[None, :]
    iu = np.triu_indices(n, 1)
    out = []
    for k in range(Y.shape[1]):
        y = Y[:, k].astype(bool)
        both = (y[:, None] & y[None, :]) & cross
        one = (y[:, None] ^ y[None, :]) & cross
        b = S[iu][both[iu]]; o = S[iu][one[iu]]
        out.append(b.mean() - o.mean() if len(b) and len(o) else np.nan)
    return np.array(out)


def run(rows, S, mot, NP, rng, permfn):
    seal = np.array([r['seal'] for r in rows])
    Y = np.array([[m in r['motif'] for m in mot] for r in rows], dtype=np.int8)
    real = stat(S, Y, seal, mot)
    null = np.array([stat(S, permfn(Y), seal, mot) for _ in range(NP)])
    p = (np.nan_to_num(null, nan=-9) >= real).mean(0)
    pooled_real = np.nansum(real); pooled_null = np.nansum(null, 1)
    return real, p, float((pooled_null >= pooled_real).mean()), null


if __name__ == '__main__':
    NP = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    rows, _ = load(True)
    rng = np.random.default_rng(2)
    mot = [m for m in MOTIFS if 4 <= sum(m in r['motif'] for r in rows) <= len(rows) - 4]
    # number of distinct seals per motif: need >= 2 seals with m to have cross-seal 'both' pairs
    nseal = {m: len({r['seal'] for r in rows if m in r['motif']}) for m in mot}
    S = vecs([r['content'] for r in rows])
    real, p, pp, _ = run(rows, S, mot, NP, rng, lambda Y: perm_seal(Y, rows, rng))
    real2, p2, pp2, _ = run(rows, S, mot, NP, rng, lambda Y: perm_seal(Y, rows, rng, by_vol=False))
    out = dict(motifs=mot, nseal=nseal, real=dict(zip(mot, map(float, real))), p_seal_vol=dict(zip(mot, map(float, p))),
               p_seal_free=dict(zip(mot, map(float, p2))), pooled_p_vol=pp, pooled_p_free=pp2)
    print(json.dumps(out, indent=0), flush=True)
    # sign-only and system-only variants
    for pref in ('S:', 'Y:', 'H:', 'O:'):
        Sx = vecs([{f for f in r['content'] if f.startswith(pref)} or {'none'} for r in rows])
        rr, pq, ppq, _ = run(rows, Sx, mot, 1000, rng, lambda Y: perm_seal(Y, rows, rng))
        out['family_' + pref] = dict(pooled_p=ppq, p={m: float(v) for m, v in zip(mot, pq)})
        print(pref, ppq, {m: round(float(v), 3) for m, v in zip(mot, pq)}, flush=True)
    # planted: 3 random mid-frequency signs added to 40% of BOVID tablets
    det = []
    for k in range(20):
        r2 = np.random.default_rng(500 + k)
        signs = collections.Counter(f for r in rows for f in r['content'] if f.startswith('S:'))
        cand = [s for s, c in signs.items() if 3 <= c <= 10]
        pl = set(r2.choice(cand, 3, replace=False))
        rows2 = []
        for r in rows:
            c = set(r['content'])
            if 'BOVID' in r['motif'] and r2.random() < 0.4:
                c |= pl
            rows2.append(dict(r, content=c))
        S2 = vecs([r['content'] for r in rows2])
        rr, pq, ppq, _ = run(rows2, S2, mot, 300, r2, lambda Y: perm_seal(Y, rows2, r2))
        det.append(float(pq[mot.index('BOVID')]))
    out['planted_bovid_p'] = det
    out['planted_power'] = float(np.mean(np.array(det) < 0.05))
    print('planted power', out['planted_power'], det, flush=True)
    json.dump(out, open(f'{CK}/cycle2.json', 'w'), indent=1)
