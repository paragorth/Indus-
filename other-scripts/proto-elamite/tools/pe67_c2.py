"""pe67 cycle 2: the real PE run. Full-data discovery (5 seeds), discovery nulls (all 120 material-row
permutations over the 5 training sites; 100 site-label shuffles), leave-one-outpost-out with nulls
(200 material-row permutations over all 7 sites; 100 site-label shuffles). Also: Ur III discovery precision
under material-row shuffles (the fair baseline for PE-67.1a)."""
import itertools, time
from multiprocessing import Pool
from pe67_lib import *
from pe67_c1 import UMATS, UKINDS, TRUTH

MATS = list(MAT['materials']); KINDS = [MAT['materials'][m]['kind'] for m in MATS]
DOCS = pe_docs()


def disc(D, M, rng, nh=20000):
    J = [j for j in range(len(D['sites'])) if D['n'][j] >= 3]
    Z = zmat(D['K'], D['n'], M, J)
    C = compat(sign_class(D['SY'][:, J].sum(1)), KINDS)
    rs = random_search(Z, C, rng, nh=nh)
    links = select_links(rs['support'], Z, 10)
    return Z, rs, links, J


def stat(Z, rs, links):
    return rs['best'], float(np.mean([Z[s, m] for s, m in links])) if links else 0.0


def job(a):
    kind, i = a
    rng = random.Random(seed('pe67c2-%s-%d' % (kind, i)))
    if kind == 'real':
        D = build(DOCS); M = mat_matrix(MAT['PE'], D['sites'], MATS)
        Z, rs, links, J = disc(D, M, rng)
        loo = run_loo(D, M, KINDS, PE_OUT, rng, nh=8000)
        return dict(kind=kind, i=i, stat=stat(Z, rs, links),
                    links=[(D['signs'][s], MATS[m], round(float(Z[s, m]), 2), rs['support'][(s, m)]) for s, m in links],
                    loo={k: dict(T=v['T'], used=v['used'], links=[(w, MATS[m]) for w, m in v['links']]) for k, v in loo.items()})
    if kind == 'mperm':     # discovery null: permutation i of the 5 training-site rows
        D = build(DOCS); M = mat_matrix(MAT['PE'], D['sites'], MATS)
        J = [j for j in range(len(D['sites'])) if D['n'][j] >= 3]
        perm = list(itertools.permutations(J))[i]
        M2 = M.copy(); M2[list(J)] = M[list(perm)]
        Z, rs, links, _ = disc(D, M2, rng, nh=8000)
        return dict(kind=kind, i=i, ident=list(perm) == list(J), stat=stat(Z, rs, links))
    if kind == 'sshuf':
        D = build(shuffle_sites(DOCS, rng)); M = mat_matrix(MAT['PE'], D['sites'], MATS)
        Z, rs, links, _ = disc(D, M, rng, nh=8000)
        loo = run_loo(D, M, KINDS, PE_OUT, rng, nh=8000)
        return dict(kind=kind, i=i, stat=stat(Z, rs, links), T=sum(v['T'] for v in loo.values()),
                    perT={k: v['T'] for k, v in loo.items()})
    if kind == 'mperm7':
        D = build(DOCS); M = mat_matrix(MAT['PE'], D['sites'], MATS)
        perm = list(range(len(D['sites']))); rng.shuffle(perm)
        loo = run_loo(D, M[perm], KINDS, PE_OUT, rng, nh=8000)
        return dict(kind=kind, i=i, T=sum(v['T'] for v in loo.values()), perT={k: v['T'] for k, v in loo.items()})
    if kind == 'ur3prec':   # Ur III discovery precision, true vs shuffled material rows
        cfg, hub = i // 10, ['Umma', 'Girsu', 'Puzriš-Dagan'][(i // 10) % 3]
        r2 = random.Random(seed('pe67-ur3-%d' % cfg))
        docs = ur3_docs_all(); others = [s for s in MAT['UR3'] if s != hub]; r2.shuffle(others)
        sub = ur3_shape(docs, hub, others, rng=r2)
        D = build(sub); M = mat_matrix(MAT['UR3'], D['sites'], UMATS)
        if i % 10:
            perm = list(range(len(D['sites']))); rng.shuffle(perm); M = M[perm]
        J = [j for j in range(len(D['sites'])) if D['n'][j] >= 3]
        Z = zmat(D['K'], D['n'], M, J); C = compat(sign_class(D['SY'][:, J].sum(1)), UKINDS)
        rs = random_search(Z, C, rng, nh=8000); links = select_links(rs['support'], Z, 10)
        return dict(kind=kind, i=i, true=(i % 10 == 0), prec=float(np.mean([(D['signs'][s], UMATS[m]) in TRUTH for s, m in links])))


if __name__ == '__main__':
    t0 = time.time()
    jobs = [('real', k) for k in range(5)] + [('mperm', k) for k in range(120)] + [('sshuf', k) for k in range(100)] \
        + [('mperm7', k) for k in range(200)] + [('ur3prec', k) for k in range(90)]
    with Pool(2) as P:
        res = P.map(job, jobs, chunksize=4)
    json.dump(res, open(os.path.join(CK, 'c2.json'), 'w'), default=float)
    print('time', time.time() - t0)
