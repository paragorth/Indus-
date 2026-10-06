"""v80 cycle 3: THE TABLE WRITTEN COLUMN BY COLUMN. If each line were a column of a table (its value set = the line
mode), line types would recur down the page with the period of the table (lines per row-block). Random hypotheses:
token representation, idf weighting, which word positions count, group (page/paragraph), lag 2..14. Statistic: the
periodic excess PE_g = sim_g - (sim_{g-1} + sim_{g+1})/2 of the mean line-line cosine at lag g, as a z against
within-group line-order permutations. Chosen on one leaf half, scored on the other.
usage: python3 v80_c3.py NHYP corpus [corpus ...]"""
import sys, json, os, time, random
import numpy as np
import v80_lib as L
from scipy import sparse

NH = int(sys.argv[1])
C = L.corpora()
TR = L.pload('transposed.pkl')
if TR is None:
    TR = {}
    for i, (k, f) in enumerate((('X_KAL', L.kalendar_pages), ('X_ALF', L.alfonsine_pages), ('X_HOR', L.hours_pages),
                                ('X_CON', L.concordance_pages), ('X_DOS', L.dosage_pages))):
        TR[k] = ('TRANSPOSED', L.through_surface(L.transposed_pages(f(), prefix=k.lower()), 8400 + 10 * i))
    L.psave('transposed.pkl', TR)
C = dict(C, **TR)
rng = random.Random(803)
HY = []
for _ in range(NH):
    pos = None
    if rng.random() < 0.5:
        pos = sorted(rng.sample(range(10), rng.randint(2, 7)))
    HY.append(dict(rep=rng.choice(L.REPRS), idf=rng.random() < 0.6, pos=pos, grp=rng.choice(['page', 'para'])))
NPERM = 60


def build(T, h):
    t = T['r_' + h['rep']]
    m = np.ones(len(t), bool) if h['pos'] is None else np.isin(T['i'], h['pos'])
    ln = T['line']; nL = ln.max() + 1; V = t.max() + 1
    X = sparse.csr_matrix((np.ones(m.sum()), (ln[m], t[m])), shape=(nL, V))
    if h['idf']:
        df = np.asarray((X > 0).sum(0)).ravel(); X = X @ sparse.diags(np.log((nL + 1) / (df + 1)))
    nrm = np.sqrt(np.asarray(X.multiply(X).sum(1)).ravel()); nrm[nrm == 0] = 1
    X = (sparse.diags(1 / nrm) @ X).tocsr()
    key = T['page'] if h['grp'] == 'page' else T['para']
    gl = np.zeros(nL, int); gl[ln] = key
    hl = np.zeros(nL, int); hl[ln] = T['half']
    groups = {}
    for l in range(nL): groups.setdefault(gl[l], []).append(l)
    return X, groups, hl


def stats(X, groups, hl, half, rng):
    G = [np.array(g) for g in groups.values() if len(g) >= 6 and hl[g[0]] == half]
    sims = [(X[g] @ X[g].T).toarray() for g in G]
    def prof(perms):
        s = np.zeros(16); c = np.zeros(16)
        for S, p in zip(sims, perms):
            Sp = S[p][:, p] if p is not None else S
            for k in range(1, min(15, len(S) - 1) + 1):
                dg = np.diagonal(Sp, k); s[k] += dg.sum(); c[k] += len(dg)
        return np.where(c > 0, s / np.maximum(c, 1), np.nan)
    real = prof([None] * len(sims))
    nulls = np.array([prof([rng.permutation(len(S)) for S in sims]) for _ in range(NPERM)])
    def pe(p): return np.array([p[g] - (p[g - 1] + p[g + 1]) / 2 if g + 1 < 16 else np.nan for g in range(16)])
    PE = pe(real); PEn = np.array([pe(n) for n in nulls])
    z = (PE - PEn.mean(0)) / (PEn.std(0) + 1e-9)
    zr = (real - nulls.mean(0)) / (nulls.std(0) + 1e-9)
    return z, zr, real


for nm in sys.argv[2:]:
    out = os.path.join(L.CK, 'c3', nm.replace('~', '_') + '.json')
    if os.path.exists(out): continue
    P = C[nm.split('~')[0]][1]
    if nm.endswith('~ls'):   # null: line order shuffled inside each page
        r2 = random.Random(5); P = [dict(p, lines=r2.sample(p['lines'], len(p['lines']))) for p in P]
    T = L.flatten(P)
    t0 = time.time(); res = []
    for hi, h in enumerate(HY):
        X, groups, hl = build(T, h)
        r = random.Random(hi); npr = np.random.default_rng(hi)
        z0, zr0, p0 = stats(X, groups, hl, 0, npr)
        z1, zr1, p1 = stats(X, groups, hl, 1, npr)
        res.append(dict(h, z0=[float(x) for x in np.nan_to_num(z0)], z1=[float(x) for x in np.nan_to_num(z1)],
                        zr0=[float(x) for x in np.nan_to_num(zr0)], zr1=[float(x) for x in np.nan_to_num(zr1)],
                        p0=[float(x) for x in np.nan_to_num(p0)]))
    json.dump(res, open(out, 'w'))
    print(nm, '%.0fs' % (time.time() - t0), flush=True)
