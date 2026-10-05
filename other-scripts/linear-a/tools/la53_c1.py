#!/usr/bin/env python3
"""LA-53 cycle 1: calibrate the surplus-attention ranker on Linear B and Ur III (opaque ids; status lists used
only to score), on planted hierarchies inside Linear A, then rank Linear A types.
Null N1: attention rows permuted within (commodity x document-size) strata. Null N2: quantities shuffled.
Output: data/la53_ckpt/c1.json, c1.log"""
import json, sys, os, math, random, collections, copy
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import la53_common as C

NSPEC = int(os.environ.get('NSPEC', 2000)); NNULL = int(os.environ.get('NNULL', 200))
rng = np.random.default_rng(C.seed('la53-c1'))
SPECS = C.random_specs(NSPEC, rng)
log = open(os.path.join(C.CK, 'c1.log'), 'w')


def P(*a):
    s = ' '.join(str(x) for x in a); print(s); log.write(s + '\n'); log.flush()


def evaluate(rows, truthf, nnull, tag, keep=False):
    B = C.build(rows)
    cnt = np.bincount(B['ti'], minlength=len(B['types']))
    ok = cnt >= 2
    obs = C.consensus_fast(B, SPECS)
    lab = np.array([truthf(t) for t in B['types']])
    a_obs = C.auc(obs[ok], lab[ok])
    # per-feature (single feature, full covariates) AUCs
    feat = {}
    for j, f in enumerate(C.FEATS):
        w = np.zeros(len(C.FEATS)); w[j] = 1
        sp = [{'w': w, 'cov': {'q': True, 'q2': False, 'C': True, 'S': False, 'ne': True, 'nsig': f != 'nsig'}, 'k': 2.0}]
        feat[f] = round(C.auc(C.consensus_fast(B, sp)[ok], lab[ok]), 3)
    nr = np.random.default_rng(C.seed('null' + tag))
    a_null = []; a_n2 = []
    sub = SPECS[:400]
    for _ in range(nnull):
        pi = C.perm_within(B['strata'], nr)
        a_null.append(C.auc(C.consensus_fast(B, sub, A=B['A'][pi])[ok], lab[ok]))
    for _ in range(max(20, nnull // 5)):
        pq = nr.permutation(len(B['q']))
        a_n2.append(C.auc(C.consensus_fast(B, sub, q=(B['q'][pq], B['qm'][pq]))[ok], lab[ok]))
    a_null = np.array(a_null)
    res = {'tag': tag, 'n_occ': len(rows), 'n_types2': int(ok.sum()), 'n_status2': int(lab[ok].sum()),
           'auc': round(a_obs, 3), 'auc_null_mean': round(float(np.nanmean(a_null)), 3),
           'auc_null_sd': round(float(np.nanstd(a_null)), 3),
           'p_N1': float((np.sum(a_null >= a_obs) + 1) / (len(a_null) + 1)),
           'auc_N2_qshuf_mean': round(float(np.nanmean(a_n2)), 3), 'feat_auc': feat}
    order = np.argsort(-obs)
    top = [B['types'][i] for i in order if ok[i]][:20]
    res['top20'] = top; res['top20_status'] = int(sum(truthf(t) for t in top))
    res['base_rate_top20'] = round(20 * lab[ok].mean(), 2)
    P(tag, json.dumps({k: v for k, v in res.items() if k != 'top20'}))
    if keep: res['B'] = B; res['obs'] = obs
    return res


def sample_docs(docs, mode, n_occ, rs):
    idx = list(range(len(docs))); rs.shuffle(idx)
    out = []; c = 0
    for i in idx:
        if c >= n_occ: break
        r = C.occurrences([docs[i]], mode)
        if r:
            out.append(docs[i]); c += len(r)
    return C.occurrences(out, mode)


def main():
    la = C.la_docs(); lb = C.lb_docs(); ur = C.ur3_docs()
    la_rows = C.occurrences(la, 'w')
    nla = len(la_rows)
    P('LA occurrences', nla, 'specs', NSPEC, 'nulls', NNULL)
    out = {'controls': []}
    out['controls'].append(evaluate(C.occurrences(lb, 'w'), C.lb_status, NNULL // 2, 'LB_full'))
    for k in range(5):
        rs = random.Random(C.seed('lbs%d' % k))
        out['controls'].append(evaluate(sample_docs(lb, 'w', nla, rs), C.lb_status, NNULL // 2, 'LB_LAsize_%d' % k))
    for k in range(5):
        rs = random.Random(C.seed('urs%d' % k))
        out['controls'].append(evaluate(sample_docs(ur, 'qty', nla, rs), C.ur_status, NNULL // 2, 'UR_LAsize_%d' % k))
    rs = random.Random(C.seed('urbig'))
    out['controls'].append(evaluate(sample_docs(ur, 'qty', 20000, rs), C.ur_status, NNULL // 4, 'UR_20k'))
    # planted hierarchies in Linear A rows
    cnt = collections.Counter(r['type'] for r in la_rows)
    pool = sorted(t for t, c in cnt.items() if c >= 2)
    for k, (p, nplant) in enumerate([(0.7, 15), (0.5, 15), (0.3, 15), (0.5, 8), (0.3, 30)]):
        rs = random.Random(C.seed('plant%d' % k))
        S = set(rs.sample(pool, nplant))
        rows = copy.deepcopy(la_rows)
        for r in rows:
            if r['type'] in S and rs.random() < p:
                r['alone'] = 1.0
                if rs.random() < 0.5: r['head'] = 1.0; r['early'] = 1.0
                if rs.random() < 0.5: r['qual'] += 1.0
                r['lines'] += 1.0 if rs.random() < 0.5 else 0.0
        res = evaluate(rows, lambda t, S=S: t in S, NNULL // 2, 'PLANT_p%.1f_n%d' % (p, nplant))
        out['controls'].append(res)
    # Linear A itself
    res = evaluate(la_rows, lambda t: False, 0, 'LA')
    B = C.build(la_rows); obs = C.consensus_fast(B, SPECS)
    cntv = np.bincount(B['ti'], minlength=len(B['types']))
    ok = cntv >= 2
    nr = np.random.default_rng(C.seed('la-null'))
    NUL = []
    for _ in range(NNULL * 2):
        pi = C.perm_within(B['strata'], nr)
        NUL.append(C.consensus_fast(B, SPECS[:400], A=B['A'][pi]))
    NUL = np.array(NUL)
    p = (np.sum(NUL >= obs[None, :], axis=0) + 1) / (len(NUL) + 1)
    # spread of consensus across specs (how many specs put the type in its top decile)
    order = np.argsort(-obs)
    rank = []
    for i in order:
        if not ok[i]: continue
        rank.append({'type': B['types'][i], 'n': int(cntv[i]), 'score': round(float(obs[i]), 3),
                     'p_N1': round(float(p[i]), 4),
                     'prof': {f: round(float(np.mean([r[f] for r in la_rows if r['type'] == B['types'][i]])), 2)
                              for f in C.FEATS + ['qty']
                              if True} if True else None})
    # BH
    ps = np.array([r['p_N1'] for r in rank]); o = np.argsort(ps); m = len(ps)
    q = np.empty(m); prev = 1.0
    for j in range(m - 1, -1, -1):
        prev = min(prev, ps[o[j]] * m / (j + 1)); q[o[j]] = prev
    for r, qq in zip(rank, q): r['q'] = round(float(qq), 4)
    out['LA_rank'] = rank
    P('LA top 25:')
    for r in rank[:25]: P(' ', json.dumps(r))
    P('LA bottom 5:', [r['type'] for r in rank[-5:]])
    P('LA types with p<0.01:', sum(1 for r in rank if r['p_N1'] < 0.01), 'q<0.1:', sum(1 for r in rank if r['q'] < 0.1),
      'expected p<0.01 by chance:', round(0.01 * len(rank), 1))
    json.dump(out, open(os.path.join(C.CK, 'c1.json'), 'w'), default=lambda o: None)


if __name__ == '__main__':
    main()
