"""pe67 cycle 1: calibration. (a) Ur III provinces at PE shape with known site products (words opaque);
(b) planted site-product signs in the real PE corpus. Random search + leave-one-outpost-out, with
material-row-shuffle and site-label-shuffle nulls."""
import sys, json, random, time
import numpy as np
from multiprocessing import Pool
from pe67_lib import *

UMATS = list(MAT['UR3_materials']); UKINDS = [MAT['UR3_materials'][m]['kind'] for m in UMATS]
TRUTH = {(w, m) for m, ws in MAT['UR3_truth'].items() for w in ws}


def full_links(D, M, kinds, rng, nh=20000, nsel=10):
    J = [j for j in range(len(D['sites'])) if D['n'][j] >= 3]
    Z = zmat(D['K'], D['n'], M, J)
    C = compat(sign_class(D['SY'][:, J].sum(1)), kinds)
    rs = random_search(Z, C, rng, nh=nh)
    return select_links(rs['support'], Z, nsel), Z, C, rs


def ur3_job(args):
    ci, hub = args
    rng = random.Random(seed('pe67-ur3-%d' % ci))
    docs = ur3_docs_all()
    others = [s for s in MAT['UR3'] if s != hub]
    rng.shuffle(others)
    sub = ur3_shape(docs, hub, others, rng=rng)
    D = build(sub)
    M = mat_matrix(MAT['UR3'], D['sites'], UMATS)
    links, Z, C, rs = full_links(D, M, UKINDS, rng)
    lk = [(D['signs'][s], UMATS[m]) for s, m in links]
    prec = np.mean([x in TRUTH for x in lk]) if lk else 0
    pc = np.argwhere(C)
    base = np.mean([(D['signs'][s], UMATS[m]) in TRUTH for s, m in pc]) if len(pc) else 0
    outs = others[:4]
    real = run_loo(D, M, UKINDS, outs, rng, nh=8000)
    Treal = sum(v['T'] for v in real.values())
    looprec = np.mean([ (w, UMATS[m]) in TRUTH for v in real.values() for w, m in v['links']])
    nullM, nullS = [], []
    for k in range(30):
        perm = list(range(len(D['sites']))); rng.shuffle(perm)
        r = run_loo(D, M[perm], UKINDS, outs, rng, nh=8000)
        nullM.append(sum(v['T'] for v in r.values()))
        Ds = build(shuffle_sites(sub, rng))
        Ms = mat_matrix(MAT['UR3'], Ds['sites'], UMATS)
        r = run_loo(Ds, Ms, UKINDS, outs, rng, nh=8000)
        nullS.append(sum(v['T'] for v in r.values()))
    pM = (1 + sum(x >= Treal for x in nullM)) / 31; pS = (1 + sum(x >= Treal for x in nullS)) / 31
    return dict(cfg=ci, hub=hub, outposts=others, links=lk, prec=float(prec), base=float(base),
                loo_prec=float(looprec), T=Treal, perT={k: v['T'] for k, v in real.items()},
                nullM=nullM, nullS=nullS, pM=pM, pS=pS)


PLANT = [('silver_lead', 1.5), ('copper', 1.0), ('shell', 1.2), ('chlorite', 1.5), ('caprine', 1.0)]


def plant_job(args):
    ri, beta, b0 = args
    rng = random.Random(seed('pe67-plant-%d-%s-%s' % (ri, beta, b0)))
    docs = [dict(x, signs=set(x['signs']), sys=dict(x['sys'])) for x in pe_docs()]
    mats = list(MAT['materials']); kinds = [MAT['materials'][m]['kind'] for m in mats]
    for m, _ in PLANT:
        name = 'PLANT_' + m
        for x in docs:
            v = MAT['PE'].get(x['site'], {}).get(m, 0)
            if rng.random() < min(0.9, b0 * math.exp(beta * v)):
                x['signs'].add(name)
                x['sys'][name] = [0, 1]
    D = build(docs)
    M = mat_matrix(MAT['PE'], D['sites'], mats)
    links, Z, C, rs = full_links(D, M, kinds, rng)
    lk = [(D['signs'][s], mats[m]) for s, m in links]
    hit = {m: (('PLANT_' + m, m) in lk) for m, _ in PLANT}
    real = run_loo(D, M, kinds, PE_OUT, rng, nh=8000)
    loohit = {o: [l for l in v['links'] if l[0].startswith('PLANT')] for o, v in real.items()}
    plantT = {}
    J_all = None
    for o, v in real.items():
        oi = D['sites'].index(o)
        J = [j for j in range(len(D['sites'])) if j != oi and D['n'][j] >= 3]
        pl = [(D['signs'].index(w), m) for w, m in v['links'] if w.startswith('PLANT')]
        plantT[o] = loo_stat(D['K'], D['n'], M, J, oi, pl)[0] if pl else None
    return dict(rep=ri, beta=beta, b0=b0, links=lk, hit=hit,
                loo_planted=[(o, [(w, mats[m]) for w, m in l]) for o, l in loohit.items()],
                plantT=plantT, T=sum(v['T'] for v in real.values()))


if __name__ == '__main__':
    t0 = time.time()
    jobs = [(i, h) for i, h in enumerate(['Umma', 'Girsu', 'Puzriš-Dagan'] * 3)]
    pj = [(r, beta, b0) for r in range(4) for beta, b0 in [(1.0, 0.02), (1.5, 0.02), (2.0, 0.01)]]
    with Pool(2) as P:
        ur = P.map(ur3_job, jobs)
        json.dump(ur, open(os.path.join(CK, 'c1_ur3.json'), 'w'), default=str)
        pl = P.map(plant_job, pj)
    json.dump(pl, open(os.path.join(CK, 'c1_plant.json'), 'w'), default=str)
    print('time', time.time() - t0)
    for r in ur:
        print(r['hub'], r['outposts'][:4], 'prec %.2f base %.3f looprec %.2f T %.1f pM %.2f pS %.2f' % (r['prec'], r['base'], r['loo_prec'], r['T'], r['pM'], r['pS']))
        print('   ', r['links'][:10])
    for r in pl:
        print(r['beta'], r['b0'], r['hit'], r['plantT'])
