"""pe67 cycle 3: is it the material or just the place?
(a) PHANTOM MATERIALS: every possible ordinal column over the training sites (3^5 = 243, constants dropped)
    is scored like a real material; real columns are ranked among them (CAP and CNT signs separately).
    Same in Ur III PE-shaped corpora (control: real product columns should rank high).
(b) THE TABLETS DIG THE SITE: leave one outpost out, learn links elsewhere, predict the held-out outpost's
    material row from its signs; Spearman with the true row vs the rows of other sites (null). Ur III control.
(c) CODING ROBUSTNESS: 200 perturbed PE material tables (entries +-1 with p 0.2; Yahya chlorite 1; SiS lapis 1).
(d) ONE-TABLET OUT-OF-SAMPLE: Shahr-i Sokhta and Ozbaki tablets vs links learned on the 5 training sites."""
import itertools, time
from multiprocessing import Pool
from scipy.stats import spearmanr
from pe67_lib import *
from pe67_c1 import UMATS, UKINDS, TRUTH

MATS = list(MAT['materials']); KINDS = [MAT['materials'][m]['kind'] for m in MATS]


def col_scores(Z_col, mask, k=3):
    v = np.sort(Z_col[mask])[::-1]
    return float(v[0]) if len(v) else 0.0, float(v[:k].sum()) if len(v) else 0.0


def phantom(D, M, kinds, mats):
    J = [j for j in range(len(D['sites'])) if D['n'][j] >= 3]
    scl = sign_class(D['SY'][:, J].sum(1))
    cols = [c for c in itertools.product([0, 1, 2], repeat=len(J)) if len(set(c)) > 1]
    P = np.zeros((len(D['sites']), len(cols)))
    for ci, c in enumerate(cols):
        P[J, ci] = c
    Zp = zmat(D['K'], D['n'], P, J)
    Zr = zmat(D['K'], D['n'], M, J)
    out = {}
    for mi, (m, kd) in enumerate(zip(mats, kinds)):
        mask = (scl == 'CAP') if kd == 'GRAIN' else ((scl == 'CNT') if kd in ('ANIMAL', 'OBJ') else (scl != 'NONE'))
        r1, r3 = col_scores(Zr[:, mi], mask)
        ph = np.array([col_scores(Zp[:, ci], mask) for ci in range(len(cols))])
        out[m] = dict(best=r1, top3=r3, pct_best=float((ph[:, 0] < r1).mean()), pct_top3=float((ph[:, 1] < r3).mean()),
                      col=[int(M[j, mi]) for j in J])
    return out, [D['sites'][j] for j in J]


def dig(D, M, kinds, outposts, rng):
    res = {}
    for oname in outposts:
        o = D['sites'].index(oname)
        J = [j for j in range(len(D['sites'])) if j != o and D['n'][j] >= 3]
        Z = zmat(D['K'], D['n'], M, J); C = compat(sign_class(D['SY'][:, J].sum(1)), kinds)
        rs = random_search(Z, C, rng, nh=8000); links = select_links(rs['support'], Z, 20)
        p = D['K'][:, J].sum(1) / D['n'][J].sum()
        pred = np.zeros(M.shape[1])
        for s, m in links:
            E = D['n'][o] * p[s]
            pred[m] += (D['K'][s, o] - E) / math.sqrt(max(E * (1 - p[s]), 1e-6))
        def rho(row):
            if len(set(row)) < 2 or len(set(pred)) < 2:
                return 0.0
            return float(spearmanr(pred, row).correlation)
        true = rho(M[o])
        others = [rho(M[j]) for j in range(len(D['sites'])) if j != o and len(set(M[j])) > 1]
        res[oname] = dict(rho=true, null=others, rank=float(np.mean([true > x for x in others])) if others else None)
    return res


def job(a):
    kind, i = a
    rng = random.Random(seed('pe67c3-%s-%d' % (kind, i)))
    if kind == 'pe':
        D = build(pe_docs()); M = mat_matrix(MAT['PE'], D['sites'], MATS)
        ph, J = phantom(D, M, KINDS, MATS)
        dg = dig(D, M, KINDS, PE_OUT, rng)
        return dict(kind=kind, phantom=ph, J=J, dig=dg)
    if kind == 'ur3':
        hub = ['Umma', 'Girsu', 'Puzriš-Dagan'][i % 3]
        r2 = random.Random(seed('pe67-ur3-%d' % i))
        docs = ur3_docs_all(); others = [s for s in MAT['UR3'] if s != hub]; r2.shuffle(others)
        sub = ur3_shape(docs, hub, others, rng=r2)
        D = build(sub); M = mat_matrix(MAT['UR3'], D['sites'], UMATS)
        ph, J = phantom(D, M, UKINDS, UMATS)
        dg = dig(D, M, UKINDS, others[:4], rng)
        return dict(kind=kind, i=i, hub=hub, phantom=ph, J=J, dig=dg)
    if kind == 'perturb':
        D = build(pe_docs())
        tab = json.loads(json.dumps(MAT['PE']))
        for s in tab:
            for m in tab[s]:
                if rng.random() < 0.2:
                    tab[s][m] = min(2, max(0, tab[s][m] + rng.choice([-1, 1])))
        if i % 2:
            tab['Yahya']['chlorite'] = 1; tab['Shahr-i Sokhta']['lapis'] = 1
        M = mat_matrix(tab, D['sites'], MATS)
        J = [j for j in range(len(D['sites'])) if D['n'][j] >= 3]
        Z = zmat(D['K'], D['n'], M, J); C = compat(sign_class(D['SY'][:, J].sum(1)), KINDS)
        rs = random_search(Z, C, rng, nh=8000); links = select_links(rs['support'], Z, 10)
        return dict(kind=kind, i=i, links=[(D['signs'][s], MATS[m]) for s, m in links])
    if kind == 'single':
        docs = pe_docs()
        D = build(docs, min_tab=3); M = mat_matrix(MAT['PE'], D['sites'], MATS)
        J = [j for j in range(len(D['sites'])) if D['n'][j] >= 3]
        Z = zmat(D['K'], D['n'], M, J); scl = sign_class(D['SY'][:, J].sum(1)); C = compat(scl, KINDS)
        out = {}
        for sname in PE_SING:
            row = np.array([MAT['PE'][sname][m] for m in MATS], float)
            mu = M[J].mean(0)
            # predicted log-lift of each sign at the new site: sum over compatible materials of z * (row - mean)
            score = ((np.where(C, Z, 0)) * (row - mu)[None, :]).sum(1)
            tab = [x for x in docs if x['site'] == sname][0]
            on = [D['signs'].index(s) for s in tab['signs'] if s in D['signs']]
            if not on:
                out[sname] = None; continue
            obs = float(np.mean(score[on]))
            # null: random sets of the same size drawn with Susa frequency weights
            p = D['K'][:, J].sum(1); p = p / p.sum()
            nul = [float(np.mean(score[np.random.default_rng(seed(sname) + k).choice(len(p), len(on), replace=False, p=p)])) for k in range(2000)]
            top = [D['signs'][s] for s in np.argsort(-score)[:8]]
            out[sname] = dict(n_signs=len(on), signs=[D['signs'][s] for s in on], obs=obs, p=float(np.mean([x >= obs for x in nul])), top_predicted=top)
        return dict(kind=kind, out=out)


if __name__ == '__main__':
    t0 = time.time()
    jobs = [('pe', 0), ('single', 0)] + [('ur3', k) for k in range(9)] + [('perturb', k) for k in range(200)]
    with Pool(2) as P:
        res = P.map(job, jobs, chunksize=2)
    json.dump(res, open(os.path.join(CK, 'c3.json'), 'w'), default=float)
    print('time', time.time() - t0)
