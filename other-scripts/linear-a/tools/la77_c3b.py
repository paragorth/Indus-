"""LA-77 cycle 3b: the no-HT distance-decay hint (c2: P 0.015) against within-site sign shuffles,
and the near-variant locality (la13 B) against the same shuffles, on equal footing."""
import json, numpy as np
from la77_geo import *
rules = np.random.default_rng(7702).normal(0, 1, (3000, 4))
surv = np.load(os.path.join(CKPT, 'c2_rules_surv.npy'))
docs = load_docs(); codes = sorted({d['site'] for d in docs if d['words']}); docs = [d for d in docs if d['site'] in codes]
S = len(codes); km = np.log(km_matrix(codes) + 1); hti = codes.index('HT')
m2 = 1 - np.eye(S); m2[hti, :] = 0; m2[:, hti] = 0
rng = np.random.default_rng(7704)


def stat(M):
    o = (M + M.T) * m2
    return float((o * km).sum() / max(o.sum(), 1e-12)), float(np.trace(M) / max(M.sum(), 1e-12))


out = []
for s in range(-1, 10):
    d = docs if s < 0 else shuffle_within_site(docs, np.random.default_rng(900 + s))
    eng = Engine(type_table(d)[0], 150, rng, codes, ntop_pool=5000)
    for h in range(eng.H):
        eng.root_coef[h] = rules[surv[h % len(surv)]]
    o, loc = stat(eng.transmissions(eng.X(d)))
    nn = np.array([stat(eng.transmissions(eng.X(permute_sites(d, rng)))) for _ in range(150)])
    r = dict(which='real' if s < 0 else f'shuffle {s}', noHT_km=o, null=float(nn[:, 0].mean()),
             z=float((o - nn[:, 0].mean()) / nn[:, 0].std()), p_lower=float((np.sum(nn[:, 0] <= o) + 1) / 151),
             local=loc, local_null=float(nn[:, 1].mean()), local_z=float((loc - nn[:, 1].mean()) / nn[:, 1].std()))
    out.append(r); print(json.dumps(r), flush=True)
json.dump(out, open(os.path.join(CKPT, 'c3b.json'), 'w'), indent=1)
