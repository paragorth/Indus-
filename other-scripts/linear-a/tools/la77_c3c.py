"""LA-77 cycle 3c: phase-gap coefficient (pair z on log km + |phase gap|) for real data and for
10 within-site sign shuffles, same engine size (150 trees, 150 permutations)."""
import json, numpy as np
from la77_geo import *
rules = np.random.default_rng(7702).normal(0, 1, (3000, 4))
surv = np.load(os.path.join(CKPT, 'c2_rules_surv.npy'))
PH = {'MMIA': 0, 'MMII': 1, 'MMIII': 2, 'MMIIIA': 2, 'MMIIIB': 2.25, 'LMIA': 3, 'LMI': 3.5, 'LMIB': 4}
docs = load_docs(); codes = sorted({d['site'] for d in docs if d['words']}); docs = [d for d in docs if d['site'] in codes]
ph = collections.defaultdict(list)
for d in docs:
    if d['context'] in PH:
        ph[d['site']].append(PH[d['context']])
phase = np.array([np.median(ph[c]) if c in ph else np.nan for c in codes])
S = len(codes); lkm = np.log(km_matrix(codes) + 1); iu = np.triu_indices(S, 1)
gap = np.abs(phase[:, None] - phase[None, :])
rng = np.random.default_rng(7705)
out = []
for s in range(-1, 10):
    d = docs if s < 0 else shuffle_within_site(docs, np.random.default_rng(900 + s))
    eng = Engine(type_table(d)[0], 150, rng, codes, ntop_pool=5000)
    for h in range(eng.H):
        eng.root_coef[h] = rules[surv[h % len(surv)]]
    M = eng.transmissions(eng.X(d)); M = M + M.T
    NS = np.array([(lambda m: m + m.T)(eng.transmissions(eng.X(permute_sites(d, rng)))) for _ in range(150)])
    Z = (M - NS.mean(0)) / np.maximum(NS.std(0), 1e-9)
    ok = ~np.isnan(gap[iu]) & (NS.std(0)[iu] > 0)
    A = np.column_stack([np.ones(ok.sum()), lkm[iu][ok], gap[iu][ok]])
    b = np.linalg.lstsq(A, Z[iu][ok], rcond=None)[0]
    r = dict(which='real' if s < 0 else f'shuffle {s}', b_logkm=float(b[1]), b_gap=float(b[2]), n=int(ok.sum()))
    out.append(r); print(json.dumps(r), flush=True)
json.dump(out, open(os.path.join(CKPT, 'c3c.json'), 'w'), indent=1)
