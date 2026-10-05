#!/usr/bin/env python3
"""LA-53 cycle 3: LET THE CONTROLS TEACH WHAT STATUS LOOKS LIKE IN CLAY, THEN CROSS THE SEA.
6,000 random attention specs (feature subset + Dirichlet weights + covariates + within/whole-document) are
scored on Linear B (AUC for the deity / wanax / official list). The top 5 % survive and are re-tested on a
DIFFERENT administration, Ur III (20k occurrences), and vice versa. Kill control: the same selection with
status labels permuted among frequency-matched types (300 times): if survivors of the real selection do not
transfer better than survivors of permuted selections, there is no portable 'status layout'.
If (and only if) transfer passes, survivors chosen on BOTH controls rank Linear A.
Output: data/la53_ckpt/c3.json, c3.log"""
import os
os.environ['OMP_NUM_THREADS'] = '1'; os.environ['OPENBLAS_NUM_THREADS'] = '1'
import json, sys, math, random, collections
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import la53_common as C
from scipy.stats import rankdata

NSPEC = int(os.environ.get('NSPEC', 6000)); NPERM = int(os.environ.get('NPERM', 300))
LOG = None


def P(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); LOG.write(s + '\n'); LOG.flush()


def per_spec(B, specs):
    """Matrix (types x specs) of standardized surpluses."""
    nt = len(B['types']); cnt = np.bincount(B['ti'], minlength=nt).astype(float); ok = cnt >= 2
    out = np.zeros((nt, len(specs)))
    groups = collections.defaultdict(list)
    for j, s in enumerate(specs):
        groups[(tuple(sorted(s['cov'].items())), s['docfe'])].append(j)
    Ad = C.demean(B['A'], B['doc'])
    for (key, fe), js in groups.items():
        X = C.design(B, dict(key))
        A = B['A']
        if fe: X = C.demean(X, B['doc']); A = Ad
        U, sv, _ = np.linalg.svd(X, full_matrices=False); Q = U[:, sv > 1e-8 * sv[0]]
        RA = A - Q @ (Q.T @ A)
        T = np.zeros((nt, A.shape[1])); np.add.at(T, B['ti'], RA)
        W = np.column_stack([specs[j]['w'] for j in js]); K = np.array([specs[j]['k'] for j in js])
        S = (T @ W) / (cnt[:, None] + K[None, :])
        S /= S[ok].std(0)[None, :] + 1e-9
        out[:, js] = S
    return out, ok


def auc_cols(M, lab):
    R = np.apply_along_axis(rankdata, 0, M)
    n1 = lab.sum(); n0 = len(lab) - n1
    return (R[lab].sum(0) - n1 * (n1 + 1) / 2) / (n1 * n0)


def freq_perm(lab, cnt, rng):
    bins = collections.defaultdict(list)
    for i, c in enumerate(cnt): bins[min(6, int(math.log2(c)))].append(i)
    out = lab.copy()
    for b, mem in bins.items():
        mem = np.array(mem); out[mem] = lab[rng.permutation(mem)]
    return out


def sample_docs(docs, mode, n_occ, rs):
    idx = list(range(len(docs))); rs.shuffle(idx)
    out = []; c = 0
    for i in idx:
        if c >= n_occ: break
        r = C.occurrences([docs[i]], mode)
        if r: out.append(docs[i]); c += len(r)
    return out


def main():
    global LOG
    LOG = open(os.path.join(C.CK, 'c3.log'), 'w')
    rng = np.random.default_rng(C.seed('la53-c3'))
    specs = C.random_specs(NSPEC, rng)
    for s in specs: s['docfe'] = bool(rng.random() < 0.5)
    la = C.la_docs(); lb = C.lb_docs(); ur = C.ur3_docs()
    sets = {}
    for tag, rows, tf in (('LB', C.occurrences(lb, 'w'), C.lb_status),
                          ('UR', C.occurrences(sample_docs(ur, 'qty', 20000, random.Random(C.seed('urbig'))), 'qty'), C.ur_status),
                          ('LA', C.occurrences(la, 'w'), lambda t: False)):
        B = C.build(rows); M, ok = per_spec(B, specs)
        lab = np.array([tf(t) for t in B['types']])
        cnt = np.bincount(B['ti'], minlength=len(B['types']))
        sets[tag] = {'B': B, 'M': M[ok], 'lab': lab[ok], 'cnt': cnt[ok], 'types': [t for t, o in zip(B['types'], ok) if o]}
        P(tag, 'types>=2', int(ok.sum()), 'status', int(lab[ok].sum()))
    a = {t: auc_cols(sets[t]['M'], sets[t]['lab']) for t in ('LB', 'UR')}
    P('spec AUC LB mean %.3f max %.3f | UR mean %.3f max %.3f | corr(LB,UR) over specs %.3f' % (
        a['LB'].mean(), a['LB'].max(), a['UR'].mean(), a['UR'].max(), np.corrcoef(a['LB'], a['UR'])[0, 1]))
    top = max(1, NSPEC // 20)
    res = {}
    for src, dst in (('LB', 'UR'), ('UR', 'LB')):
        surv = np.argsort(-a[src])[:top]
        obs = float(a[dst][surv].mean())
        # consensus of survivors on the destination
        cons = sets[dst]['M'][:, surv].mean(1); cons_auc = float(C.auc(cons, sets[dst]['lab']))
        pr = np.random.default_rng(C.seed('perm' + src))
        nulls = []; ncons = []
        for _ in range(NPERM):
            lp = freq_perm(sets[src]['lab'], sets[src]['cnt'], pr)
            ap = auc_cols(sets[src]['M'], lp)
            sp = np.argsort(-ap)[:top]
            nulls.append(a[dst][sp].mean()); ncons.append(C.auc(sets[dst]['M'][:, sp].mean(1), sets[dst]['lab']))
        nulls = np.array(nulls); ncons = np.array(ncons)
        res[src + '->' + dst] = {'surv_mean_auc_dst': round(obs, 3), 'all_mean_auc_dst': round(float(a[dst].mean()), 3),
                                 'null_mean': round(float(nulls.mean()), 3), 'p': round(float((np.sum(nulls >= obs) + 1) / (NPERM + 1)), 4),
                                 'cons_auc_dst': round(cons_auc, 3), 'null_cons_mean': round(float(ncons.mean()), 3),
                                 'p_cons': round(float((np.sum(ncons >= cons_auc) + 1) / (NPERM + 1)), 4),
                                 'surv_src_auc': round(float(a[src][surv].mean()), 3)}
        # what survivors look like
        W = np.array([specs[j]['w'] for j in surv]).mean(0)
        res[src + '->' + dst]['surv_weights'] = {f: round(float(w), 3) for f, w in zip(C.FEATS, W)}
        res[src + '->' + dst]['surv_docfe_share'] = round(float(np.mean([specs[j]['docfe'] for j in surv])), 2)
        P(src + '->' + dst, json.dumps(res[src + '->' + dst]))
    # specs good on both controls (rank-sum), applied to Linear A
    both = np.argsort(-(rankdata(a['LB']) + rankdata(a['UR'])))[:top]
    W = np.array([specs[j]['w'] for j in both]).mean(0)
    P('both-survivors: LB auc %.3f UR auc %.3f weights %s docfe %.2f' % (
        a['LB'][both].mean(), a['UR'][both].mean(), {f: round(float(w), 3) for f, w in zip(C.FEATS, W)},
        np.mean([specs[j]['docfe'] for j in both])))
    L = sets['LA']; sc = L['M'][:, both].mean(1)
    # LA null: rows permuted within strata, same survivor specs
    B = L['B']; okm = np.bincount(B['ti'], minlength=len(B['types'])) >= 2
    pr = np.random.default_rng(C.seed('la-c3-null'))
    NUL = []
    sp_both = [specs[j] for j in both]
    for _ in range(NPERM):
        pi = C.perm_within(B['strata'], pr)
        Bp = dict(B, A=B['A'][pi])
        Mp, _ = per_spec(Bp, sp_both)
        NUL.append(Mp[okm].mean(1))
    NUL = np.array(NUL)
    p = (np.sum(NUL >= sc[None, :], 0) + 1) / (NPERM + 1)
    o = np.argsort(-sc)
    rows = C.occurrences(la, 'w')
    rank = []
    for i in o:
        t = L['types'][i]
        rr = [r for r in rows if r['type'] == t]
        rank.append({'type': t, 'n': int(L['cnt'][i]), 'score': round(float(sc[i]), 3), 'p': round(float(p[i]), 4),
                     'prof': {f: round(float(np.mean([r[f] for r in rr])), 2) for f in C.FEATS},
                     'qty': [r['qty'] for r in rr][:8], 'logo': sorted(set(r['logo'] for r in rr))[:4],
                     'docs': sorted(set(rows_doc_id(la, r['doc']) for r in rr))[:6]})
    ps = np.array([r['p'] for r in rank]); m = len(ps); oo = np.argsort(ps); q = np.empty(m); prev = 1.0
    for j in range(m - 1, -1, -1):
        prev = min(prev, ps[oo[j]] * m / (j + 1)); q[oo[j]] = prev
    for r, qq in zip(rank, q): r['q'] = round(float(qq), 4)
    for r in rank[:20]: P('LA', json.dumps(r))
    P('LA p<0.01:', int((ps < 0.01).sum()), 'of', m, '(chance %.1f)' % (0.01 * m), 'q<0.1:', int((q < 0.1).sum()))
    json.dump({'transfer': res, 'LA_rank': rank}, open(os.path.join(C.CK, 'c3.json'), 'w'))


_LA = None
def rows_doc_id(la, di):
    return la[di]['id']


if __name__ == '__main__':
    main()
