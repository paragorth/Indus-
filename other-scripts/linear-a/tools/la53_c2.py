#!/usr/bin/env python3
"""LA-53 cycle 2: (a) split the attention surplus into a WITHIN-document part (document fixed effects: how much
more clay a word gets than its neighbours on the same tablet) and a BETWEEN-document part (how lavish the
documents it appears in are); calibrate both on LB / Ur III / planted. (b) Split-half replication of the
Linear A type ranking vs the N1 null. (c) Independent behaviour of high-surplus words: site spread,
non-administrative objects, round quantities, small quantities, each frequency-matched by permutation;
the same tests run on Linear B with the status list as the 'answer'.
Output: data/la53_ckpt/c2.json, c2.log"""
import os
os.environ['OMP_NUM_THREADS'] = '1'; os.environ['OPENBLAS_NUM_THREADS'] = '1'; os.environ['MKL_NUM_THREADS'] = '1'
import json, sys, os, math, random, collections, copy
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import la53_common as C
from scipy.stats import spearmanr, rankdata

NSPEC = int(os.environ.get('NSPEC', 2000)); NNULL = int(os.environ.get('NNULL', 60))
rng = np.random.default_rng(C.seed('la53-c2'))
SPECS = C.random_specs(NSPEC, rng)
SUB = SPECS[:300]
LOG = None


def P(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); LOG.write(s + '\n'); LOG.flush()


def score(B, specs, variant, A=None):
    A = B['A'] if A is None else A
    if variant == 'base': return C.consensus_fast(B, specs, A=A)
    if variant == 'within': return C.consensus_fast(B, specs, A=A, docfe=True)
    if variant == 'between': return C.consensus_fast(B, specs, A=A - C.demean(A, B['doc']))


def evaluate(rows, truthf, tag, variants=('base', 'within', 'between'), nnull=NNULL):
    B = C.build(rows)
    cnt = np.bincount(B['ti'], minlength=len(B['types'])); ok = cnt >= 2
    lab = np.array([truthf(t) for t in B['types']])
    out = {'tag': tag, 'n_status2': int(lab[ok].sum()), 'n_types2': int(ok.sum())}
    nr = np.random.default_rng(C.seed('n2' + tag))
    perms = [C.perm_within(B['strata'], nr) for _ in range(nnull)]
    for v in variants:
        a = C.auc(score(B, SPECS, v)[ok], lab[ok])
        an = np.array([C.auc(score(B, SUB, v, A=B['A'][pi])[ok], lab[ok]) for pi in perms])
        out[v] = {'auc': round(a, 3), 'null': round(float(np.nanmean(an)), 3), 'sd': round(float(np.nanstd(an)), 3),
                  'p': round(float((np.sum(an >= a) + 1) / (len(an) + 1)), 3)}
    P(tag, json.dumps(out))
    return out


def sample_docs(docs, mode, n_occ, rs):
    idx = list(range(len(docs))); rs.shuffle(idx)
    out = []; c = 0
    for i in idx:
        if c >= n_occ: break
        r = C.occurrences([docs[i]], mode)
        if r: out.append(docs[i]); c += len(r)
    return out


def split_half(docs, mode, tag, nsplit=int(os.environ.get('NSPLIT', 40))):
    """Rank replication across disjoint document halves; null = rows permuted within strata first."""
    rs = random.Random(C.seed('sh' + tag))
    res = {'base': [], 'within': [], 'null_base': [], 'null_within': []}
    for k in range(nsplit):
        idx = list(range(len(docs))); rs.shuffle(idx)
        h = [set(idx[:len(idx) // 2]), set(idx[len(idx) // 2:])]
        rows = C.occurrences(docs, mode)
        B = C.build(rows)
        nr = np.random.default_rng(k)
        pi = C.perm_within(B['strata'], nr)
        for v in ('base', 'within'):
            for null in (False, True):
                A = B['A'][pi] if null else B['A']
                sc = []
                for hh in h:
                    m = np.array([r['doc'] in hh for r in rows])
                    sub = [r for r, mm in zip(rows, m) if mm]
                    Bh = C.build(sub)
                    # standardization of A must use the global one: rebuild A from global rows
                    Bh['A'] = A[m]
                    s = score(Bh, SUB, v)
                    cnt = np.bincount(Bh['ti'], minlength=len(Bh['types']))
                    sc.append({t: s[i] for i, t in enumerate(Bh['types']) if cnt[i] >= 2})
                common = sorted(set(sc[0]) & set(sc[1]))
                if len(common) >= 8:
                    rho = spearmanr([sc[0][t] for t in common], [sc[1][t] for t in common])[0]
                    res[('null_' if null else '') + v].append(rho)
    out = {k: round(float(np.mean(v)), 3) for k, v in res.items()}
    out['n_common_example'] = len(common)
    out['p_base'] = round(float(np.mean(np.array(res['null_base']) >= np.mean(res['base']))), 3)
    out['p_within'] = round(float(np.mean(np.array(res['null_within']) >= np.mean(res['within']))), 3)
    P('SPLIT', tag, json.dumps(out))
    return out


def type_props(docs, mode, rows):
    """Behaviour outside the attention measure, per type: number of sites, share of quantities that are
    'round' (1, or a multiple of 5), quantity relative to its commodity median, non-administrative objects."""
    sites = collections.defaultdict(set); nonadm = collections.Counter(); tot = collections.Counter()
    for d in docs:
        for line in d['lines']:
            for t in line:
                if t[0] == 'W':
                    sites[t[1]].add(d['site']); tot[t[1]] += 1
                    if d.get('support') not in C.ADMIN and d.get('support') not in ('Nodule', 'Roundel', 'Sealing') \
                            and mode == 'w' and d['site'] != 'x' and d.get('support') not in (None,) \
                            and not d['id'][:2] in ('KN', 'PY', 'TH', 'MY', 'TI'):
                        nonadm[t[1]] += 1
    med = collections.defaultdict(list)
    for r in rows:
        if r['qty']: med[r['logo']].append(r['qty'])
    med = {k: np.median(v) for k, v in med.items()}
    rd = collections.defaultdict(list); sm = collections.defaultdict(list)
    for r in rows:
        q = r['qty']
        if q is None or q <= 0: continue
        rd[r['type']].append(float(q == 1 or (abs(q - round(q)) < 1e-9 and round(q) % 5 == 0)))
        sm[r['type']].append(math.log(q / med[r['logo']]))
    return {'nsites': {t: len(s) for t, s in sites.items()}, 'nonadmin': {t: float(nonadm[t] > 0) for t in tot},
            'round': {t: float(np.mean(v)) for t, v in rd.items()}, 'logq_rel': {t: float(np.mean(v)) for t, v in sm.items()},
            'tot': dict(tot)}


def behaviour(score_by_type, n_by_type, props, tag, nperm=int(os.environ.get('NPERM', 2000))):
    """Spearman between surplus and each property, with a permutation null that shuffles scores among
    types in the same frequency bin (so frequency cannot drive it)."""
    types = [t for t in score_by_type if n_by_type[t] >= 2]
    bins = collections.defaultdict(list)
    for i, t in enumerate(types): bins[min(6, int(math.log2(props['tot'].get(t, 1))))].append(i)
    s = np.array([score_by_type[t] for t in types])
    out = {}
    r = np.random.default_rng(C.seed('beh' + tag))
    for p in ('nsites', 'nonadmin', 'round', 'logq_rel'):
        ii = [i for i, t in enumerate(types) if t in props[p]]
        if len(ii) < 10: continue
        x = np.array([props[p][types[i]] for i in ii]); ss = s[ii]
        if np.std(x) == 0: continue
        obs = spearmanr(ss, x)[0]
        pos = {i: k for k, i in enumerate(ii)}
        null = []
        for _ in range(nperm):
            sp = s.copy()
            for b, mem in bins.items():
                sp[mem] = sp[r.permutation(mem)]
            null.append(spearmanr(sp[ii], x)[0])
        null = np.array(null)
        out[p] = {'rho': round(float(obs), 3), 'null': round(float(np.mean(null)), 3),
                  'p_hi': round(float((np.sum(null >= obs) + 1) / (nperm + 1)), 4),
                  'p_lo': round(float((np.sum(null <= obs) + 1) / (nperm + 1)), 4), 'n': len(ii)}
    P('BEHAV', tag, json.dumps(out))
    return out


def main():
    global LOG
    LOG = open(os.path.join(C.CK, 'c2.log'), 'w')
    la = C.la_docs(); lb = C.lb_docs(); ur = C.ur3_docs()
    la_rows = C.occurrences(la, 'w'); nla = len(la_rows)
    out = {'cal': [], 'split': {}, 'behav': {}}
    out['cal'].append(evaluate(C.occurrences(lb, 'w'), C.lb_status, 'LB_full'))
    lbs = []
    for k in range(4):
        d = sample_docs(lb, 'w', nla, random.Random(C.seed('lbs%d' % k))); lbs.append(d)
        out['cal'].append(evaluate(C.occurrences(d, 'w'), C.lb_status, 'LB_LAsize_%d' % k))
    for k in range(4):
        d = sample_docs(ur, 'qty', nla, random.Random(C.seed('urs%d' % k)))
        out['cal'].append(evaluate(C.occurrences(d, 'qty'), C.ur_status, 'UR_LAsize_%d' % k))
    d = sample_docs(ur, 'qty', 20000, random.Random(C.seed('urbig')))
    out['cal'].append(evaluate(C.occurrences(d, 'qty'), C.ur_status, 'UR_20k', nnull=30))
    # planted WITHIN-document hierarchy: planted words promoted relative to their tablet neighbours
    cnt = collections.Counter(r['type'] for r in la_rows)
    pool = sorted(t for t, c in cnt.items() if c >= 2)
    for k, p in enumerate((0.5, 0.3)):
        rs = random.Random(C.seed('plant2%d' % k)); S = set(rs.sample(pool, 15))
        rows = copy.deepcopy(la_rows)
        for r in rows:
            if r['type'] in S and rs.random() < p:
                r['alone'] = 1.0; r['qual'] += 1.0
        out['cal'].append(evaluate(rows, lambda t, S=S: t in S, 'PLANT_alone+qual_p%.1f' % p))
    # split-half replication
    out['split']['LA'] = split_half(la, 'w', 'LA')
    out['split']['LB_LAsize'] = split_half(lbs[0], 'w', 'LB_LAsize')
    # behaviour of high-surplus words
    for tag, docs, mode in (('LA', la, 'w'), ('LB_full', lb, 'w')):
        rows = C.occurrences(docs, mode); B = C.build(rows)
        cntv = np.bincount(B['ti'], minlength=len(B['types']))
        props = type_props(docs, mode, rows)
        for v in ('base', 'within', 'between'):
            sc = score(B, SPECS, v)
            out['behav'][tag + '_' + v] = behaviour({t: sc[i] for i, t in enumerate(B['types'])},
                                                     {t: cntv[i] for i, t in enumerate(B['types'])}, props, tag + '_' + v)
        if tag == 'LB_full':
            # do the KNOWN status words behave as expected (sites, round, small)? calibration of the behaviour tests
            lab = {t: float(C.lb_status(t)) for t in B['types']}
            out['behav']['LB_truth'] = behaviour(lab, {t: cntv[i] for i, t in enumerate(B['types'])}, props, 'LB_truth')
        if tag == 'LA':
            ranks = {}
            for v in ('base', 'within', 'between'):
                sc = score(B, SPECS, v)
                o = [B['types'][i] for i in np.argsort(-sc) if cntv[i] >= 2]
                ranks[v] = o
                P('LA top15', v, o[:15])
            out['LA_rank'] = ranks
            out['LA_props'] = {k: {t: props[k].get(t) for t in ranks['base'][:40]} for k in ('nsites', 'nonadmin', 'round', 'logq_rel', 'tot')}
    json.dump(out, open(os.path.join(C.CK, 'c2.json'), 'w'))


if __name__ == '__main__':
    main()
