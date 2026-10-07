"""la76 cycle 1: massive random search of hidden value vectors; train/test by documents.
Arms: REAL, PLANTED (economy on real skeleton, half of documents), NOECON (independent quantities)."""
import sys, os, json, time
os.environ.setdefault('OMP_NUM_THREADS', '2'); os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
sys.path.insert(0, os.path.dirname(__file__))
from la76_common import *
from scipy.stats import spearmanr

NV = int(sys.argv[1]) if len(sys.argv) > 1 else 200000
NSPLIT = int(sys.argv[2]) if len(sys.argv) > 2 else 8
TOP = 50; NPERM = 200; CH = 20000
os.makedirs(CK, exist_ok=True)
rows = entries(read_only=os.environ.get('RD') is None, frac_mode=os.environ.get('FRACM', 'int'))
rows = [r for r in rows if r['q'] >= float(os.environ.get('MINQ', '0'))]
rows = [r for r in rows if r['com'] not in os.environ.get('EXCL', '').split(',')]
goods = goods_list(rows); mk = Market(rows, goods); K = len(goods)


def score(LV, lq, pm1, pm2):
    out = np.zeros(len(LV))
    for i in range(0, len(LV), CH):
        out[i:i + CH] = mk.s1(LV[i:i + CH], lq, pm1) + mk.s2(LV[i:i + CH], lq, pm2)
    return out


def run_arm(lq, seed, lv_true=None):
    rng = np.random.default_rng(seed)
    res = []
    for sp in range(NSPLIT):
        perm = rng.permutation(len(mk.docs)); tr = set(mk.docs[i] for i in perm[:len(perm) // 2])
        te = set(mk.docs) - tr
        p1tr, p2tr = mk.doc_mask(tr); p1te, p2te = mk.doc_mask(te)
        LV, s_tr = climb(lq, p1tr, p2tr, rng)
        top = np.argsort(-s_tr)[:TOP]
        LVt = LV[top]
        te_real = score(LVt, lq, p1te, p2te)
        uni = score(np.zeros((1, K)), lq, p1te, p2te)[0]
        rnd = score(random_values(rng, TOP, K), lq, p1te, p2te)          # unselected random vectors
        nl = [perm_planted(lq, rng) for _ in range(NPERM)]
        nulls = np.array([score(LVt, x, p1te, p2te).mean() for x in nl])
        unulls = np.array([score(np.zeros((1, K)), x, p1te, p2te)[0] for x in nl])
        best = np.median(LVt, 0)
        rec = spearmanr(best[1:], lv_true[1:])[0] if lv_true is not None else None
        res.append(dict(split=sp, tr_best=float(s_tr[top[0]]), tr_top_mean=float(s_tr[top].mean()),
                        tr_rand_mean=float(s_tr.mean()), te_top=float(te_real.mean()), te_uniform=float(uni),
                        te_random=float(rnd.mean()), te_null_mean=float(nulls.mean()),
                        te_uni_null=float(unulls.mean()), p_uni=float((1 + (unulls >= uni).sum()) / (NPERM + 1)),
                        excess_top=float(te_real.mean() - nulls.mean()), excess_uni=float(uni - unulls.mean()),
                        p_perm=float((1 + (nulls >= te_real.mean()).sum()) / (NPERM + 1)),
                        rec=rec, best=best.tolist()))
        print(json.dumps({k: v for k, v in res[-1].items() if k != 'best'}), flush=True)
    return res


NR = 2000; NSTEP = NV // NR


def climb(lq, p1, p2, rng):
    """NR random restarts x NSTEP random proposals each (NV evaluations in all)."""
    LV = random_values(rng, NR, K); S = score(LV, lq, p1, p2)
    for st in range(NSTEP):
        P = LV.copy(); j = rng.integers(1, K, NR)
        jump = rng.random(NR) < 0.3
        P[np.arange(NR), j] = np.where(jump, rng.uniform(-np.log(60), np.log(60), NR), P[np.arange(NR), j] + rng.normal(0, 0.15, NR))
        SP = score(P, lq, p1, p2); acc = SP >= S
        LV[acc] = P[acc]; S[acc] = SP[acc]
    return LV, S


def perm_planted(lq, rng):
    lq2 = lq.copy()
    for ix in mk.strata: lq2[ix] = lq2[rng.permutation(ix)]
    return lq2


if __name__ == '__main__':
    t0 = time.time()
    print('goods', goods, 'rows', len(mk.rows), 'docs', len(mk.docs), 'pairs', len(mk.A), 'blockpairs', len(mk.BA))
    out = {}
    out['REAL'] = run_arm(mk.lq, 1) if not os.environ.get('ONLYP') else []
    rng = np.random.default_rng(99)
    lv_true = random_values(rng, 1, K, span=np.log(20))[0]
    out['PLANTED'] = run_arm(planted_quantities(mk, rng, lv_true, frac=float(os.environ.get('PFRAC', '0.5'))), 2, lv_true); out['lv_true'] = lv_true.tolist()
    out['NOECON'] = run_arm(mk.permute(np.random.default_rng(5)), 3) if not os.environ.get('ONLYP') else []
    out['goods'] = goods
    json.dump(out, open(os.path.join(CK, os.environ.get('OUT', 'c1.json')), 'w'))
    for arm in ('REAL', 'PLANTED', 'NOECON'):
        R = out[arm]
        if not R: continue
        f = lambda k: np.mean([r[k] for r in R])
        print(arm, 'te_top %.2f uniform %.2f random %.2f null %.2f | P<=0.05 in %d/%d | rec %s' % (
            f('te_top'), f('te_uniform'), f('te_random'), f('te_null_mean'),
            sum(r['p_perm'] <= 0.05 for r in R), len(R),
            None if R[0]['rec'] is None else '%.2f' % f('rec')))
        print('   excess top %.2f uniform %.2f | P_uni<=0.05 %d' % (f('excess_top'), f('excess_uni'), sum(r['p_uni'] <= 0.05 for r in R)))
    print('time', time.time() - t0)
