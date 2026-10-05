"""pe44 cycle 1: calibrate random per-quantity mixtures on Ur III (known kinds) and on planted PE.

Ur III kinds (from wording, never seen by the models):
  COUNTED  = LIVESTOCK (Drehem), PEOPLE        MEASURED = GRAIN (non-field grain gur)
  PLANNED  = RATION (per-head norm), DEBIT (obligation block of a balanced account), ESTIMATE (harvest projection)
Each random model: random feature subset (3-7 of FEATS), K in 2..4, EM on 15,000 quantities of
tablet half A.  The 'planned' class is chosen BLIND from the class profile (plan_score), then:
  NMI(cluster, kind) on half B  vs the same with kinds shuffled (label null)
  AUC(planned posterior: PLANNED vs COUNTED+MEASURED) on half B, raw and size-matched
PE planted control: 25% of PE tablets, each entry w.p. 0.5 turned into a plan value
(rounded to its leading denomination; w.p. 0.3 it copies the previous plan value on the tablet);
AUC(planned posterior: planted vs untouched entries on the same tablets). Null: same tablets,
entries chosen but values untouched (AUC must be ~0.5).
"""
import json, math, os, random, sys
from multiprocessing import Pool
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pe44_common import *  # noqa

KIND = {'LIVESTOCK': 'COUNTED', 'PEOPLE': 'COUNTED', 'GRAIN': 'MEASURED',
        'RATION': 'PLANNED', 'DEBIT': 'PLANNED', 'ESTIMATE': 'PLANNED', 'DEFICIT': 'DEFICIT'}


def build():
    u = ur3_tabs()
    tabs = {t: d['recs'] for t, d in u.items()}
    X, meta = corpus_matrix(tabs)
    kind = np.array([KIND.get(tabs[t][i]['lab'], '') for t, i in meta])
    syst = np.array([tabs[t][i]['sys'] for t, i in meta])
    ids = sorted(tabs)
    rng = random.Random(1)
    half = {t: rng.random() < 0.5 for t in ids}
    inA = np.array([half[t] for t, i in meta])
    return X, kind, inA, syst


def plant_pe(rng, frac=0.25, p=0.5, alter=True):
    tabs = pe_tabs()
    lab = {}
    for t, recs in tabs.items():
        if rng.random() >= frac:
            continue
        last = None
        for i, r in enumerate(recs):
            den = DEN[r['sys']]
            if rng.random() >= p:
                lab[(t, i)] = 0
                continue
            v = r['val']
            hi = max(k for k, d in enumerate(den) if d <= v)
            if hi == 0:
                lab[(t, i)] = 0
                continue
            if alter:
                if last is not None and rng.random() < 0.3 and last[0] == r['sys']:
                    nv = last[1]
                else:
                    nv = max(den[hi], int(round(v / den[hi])) * den[hi])
                r['val'] = nv
                last = (r['sys'], nv)
            lab[(t, i)] = 1
    return tabs, lab


def worker(args):
    seed, = args
    rng = np.random.default_rng(seed)
    pr = random.Random(seed)
    nf = len(FEATS)
    k = pr.randint(3, 7)
    cols = sorted(pr.sample(range(nf), k))
    K = pr.randint(2, 4)
    # Ur III
    ia = np.where(UA)[0]
    sub = rng.choice(ia, min(15000, len(ia)), replace=False)
    m = fit_mix(UX[sub], cols, K, rng, iters=40)
    ps = plan_score(m)
    pc = int(np.argmax(ps))
    ib = np.where(~UA & (UK != ''))[0]
    R, _ = posterior(m, UX[ib])
    hard = R.argmax(1)
    kb = UK[ib]
    nmi = ami(hard, kb)
    nmi0 = ami(hard, rng.permutation(kb))
    pp = R[:, pc]
    a_raw = auc(pp[kb == 'PLANNED'], pp[(kb == 'COUNTED') | (kb == 'MEASURED')])
    # size matched: within each log2 bin, equal numbers of PLANNED and OTHER
    lv = np.floor(UX[ib][:, FEATS.index('LMAG')] / math.log(2))
    sb = US[ib]
    pos, neg = [], []
    for b in np.unique(lv):
        P = np.where((lv == b) & (kb == 'PLANNED'))[0]
        N = np.where((lv == b) & ((kb == 'COUNTED') | (kb == 'MEASURED')))[0]
        n = min(len(P), len(N))
        if n:
            pos += list(rng.choice(P, n, replace=False)); neg += list(rng.choice(N, n, replace=False))
    a_sm = auc(pp[pos], pp[neg])
    # system- and size-matched: capacity lines only, PLANNED vs measured GRAIN
    pos, neg = [], []
    for b in np.unique(lv):
        P = np.where((lv == b) & (kb == 'PLANNED') & (sb == 'UR_CAP'))[0]
        N = np.where((lv == b) & (kb == 'MEASURED'))[0]
        n = min(len(P), len(N))
        if n:
            pos += list(rng.choice(P, n, replace=False)); neg += list(rng.choice(N, n, replace=False))
    a_cap = auc(pp[pos], pp[neg])
    a_def = auc(pp[kb == 'DEFICIT'], pp[(kb == 'COUNTED') | (kb == 'MEASURED')])
    # PE planted
    Xp, lab = PE_PLANT
    mp = fit_mix(Xp, cols, K, rng, iters=40)
    pcp = int(np.argmax(plan_score(mp)))
    Rp, _ = posterior(mp, Xp)
    a_pl = auc(Rp[PL_POS, pcp], Rp[PL_NEG, pcp])
    Xn = PE_NULL
    mn = fit_mix(Xn, cols, K, rng, iters=40)
    pcn = int(np.argmax(plan_score(mn)))
    Rn, _ = posterior(mn, Xn)
    a_null = auc(Rn[PN_POS, pcn], Rn[PN_NEG, pcn])
    return {'seed': seed, 'cols': [FEATS[c] for c in cols], 'K': K, 'nmi': nmi, 'nmi0': nmi0,
            'auc_raw': a_raw, 'auc_sm': a_sm, 'auc_cap': a_cap, 'auc_def': a_def, 'auc_plant': a_pl, 'auc_plant_null': a_null}


def init():
    global UX, UK, UA, US, PE_PLANT, PL_POS, PL_NEG, PE_NULL, PN_POS, PN_NEG
    UX, UK, UA, US = build()
    tabs, lab = plant_pe(random.Random(7))
    X, meta = corpus_matrix(tabs)
    PE_PLANT = (X, lab)
    PL_POS = [j for j, mm in enumerate(meta) if lab.get(mm) == 1]
    PL_NEG = [j for j, mm in enumerate(meta) if lab.get(mm) == 0]
    tabs0, lab0 = plant_pe(random.Random(7), alter=False)
    X0, meta0 = corpus_matrix(tabs0)
    PE_NULL = X0
    PN_POS = [j for j, mm in enumerate(meta0) if lab0.get(mm) == 1]
    PN_NEG = [j for j, mm in enumerate(meta0) if lab0.get(mm) == 0]


if __name__ == '__main__':
    M = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    out = os.path.join(CK, 'c1_models.jsonl')
    with Pool(2, initializer=init) as pool, open(out, 'w') as f:
        for i, r in enumerate(pool.imap_unordered(worker, [(s,) for s in range(M)], chunksize=10)):
            f.write(json.dumps(r) + '\n')
            if i % 200 == 0:
                print(i, r, flush=True)
