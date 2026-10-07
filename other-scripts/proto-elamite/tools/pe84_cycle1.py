"""pe84 cycle 1: clay clock (primary) + 5,000 random physical x text hypotheses, held out by publication half.
usage: python3 pe84_cycle1.py feats.json [planted_feats.json planted_ids.txt] -> data/pe84_ckpt/c1.json"""
import sys, os, json
import numpy as np
from scipy.stats import norm
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pe84_common as C

rng = np.random.default_rng(84)


def zp(r, n):
    if not np.isfinite(r) or n < 25:
        return 1.0
    z = np.arctanh(np.clip(r, -0.999, 0.999)) * np.sqrt(n - 3 - 8)
    return float(2 * norm.sf(abs(z)))


def arr(R, key, src):
    return np.array([(r[src].get(key) if r[src].get(key) is not None else np.nan) for r in R], float)


def density(R):
    return np.array([np.log1p(r['tx']['n_signs'] + r['tx']['n_num_marks']) - r['la'] for r in R])


def primary(R, B=int(os.environ.get('PE84_B', 2000))):
    out = {}
    have = [r for r in R if r['tx']['rv_any'] and 'd_depth' in r['ph']]
    for kind in ('rv_total_only',):
        for key in ('d_depth', 'd_E08', 'd_E04', 'd_wid'):
            S = [r for r in have if r['tx']['rv_total_only'] or r['tx']['rv_cont']]
            x = arr(S, key, 'ph'); y = arr(S, kind, 'tx')
            ex = [arr(S, 'ob_ink', 'ph'), arr(S, 'rv_ink', 'ph'), np.array([r['rvb'] for r in S]), density(S)]
            Z = C.covmat(S, ex)
            r0, n = C.pr(x, y, Z)
            s = C.strata(S); cnt = 0
            for _ in range(B):
                idx = C.permute_within(s, rng)
                rp, _ = C.pr(x, y[idx], Z)
                cnt += abs(rp) >= abs(r0)
            halves = {}
            for h in 'AB':
                Sh = [i for i, r in enumerate(S) if r['half'] == h]
                halves[h] = C.pr(x[Sh], y[Sh], Z[Sh])
            out[key] = dict(r=round(r0, 3), n=n, p=round((cnt + 1) / (B + 1), 4),
                            n_total_only=int(np.nansum(y)), A=[round(halves['A'][0], 3), halves['A'][1]],
                            B=[round(halves['B'][0], 3), halves['B'][1]])
            # raw means (no covariates) for the record
            out[key]['mean_total_only'] = round(float(np.nanmean(x[y == 1])), 4)
            out[key]['mean_cont'] = round(float(np.nanmean(x[y == 0])), 4)
    # secondary: obverse bottom-minus-top vs obverse ending in a numeric-only line
    for key in ('ob_t2m0_depth', 'ob_t2m0_E08'):
        x = arr(R, key, 'ph'); y = arr(R, 'ob_last_numonly', 'tx')
        Z = C.covmat(R, [arr(R, 'ob_t2m0_ink', 'ph'), density(R)])
        r0, n = C.pr(x, y, Z)
        s = C.strata(R); cnt = 0
        for _ in range(B):
            rp, _ = C.pr(x, y[C.permute_within(s, rng)], Z)
            cnt += abs(rp) >= abs(r0)
        out['sec_' + key] = dict(r=round(r0, 3), n=n, p=round((cnt + 1) / (B + 1), 4))
    return out


def engine(R, H, textperm=None):
    """H: list of (phys key, text key, extra covariate keys). Returns replicated set for A->B and B->A."""
    TX = R if textperm is None else [R[i] for i in textperm]
    half = np.array([r['half'] for r in R])
    Zb = C.covmat(R, [density(TX)])
    cache_x, cache_y = {}, {}
    rep = []
    scoresA = []
    for (pk, tk, ex) in H:
        x = cache_x.setdefault(pk, arr(R, pk, 'ph'))
        y = cache_y.setdefault(tk, arr(TX, tk, 'tx'))
        Z = Zb if not ex else np.column_stack([Zb] + [np.nan_to_num(arr(R, e, 'ph')) for e in ex])
        res = []
        for h in 'AB':
            m = half == h
            r, n = C.pr(x[m], y[m], Z[m])
            res.append((r, n, zp(r, n)))
        scoresA.append(res)
    # selection: top 2% by p within the selection half and p < 0.01, re-test: same sign and p < 0.05
    for sel, tst in ((0, 1), (1, 0)):
        ps = np.array([s[sel][2] for s in scoresA])
        cut = min(0.01, np.quantile(ps, 0.02))
        for i, s in enumerate(scoresA):
            if s[sel][2] <= cut and np.isfinite(s[tst][0]) and np.sign(s[tst][0]) == np.sign(s[sel][0]) and s[tst][2] < 0.05:
                rep.append((i, sel))
    return rep, scoresA


def hyps(R, N, rng, extra_text=()):
    pk = sorted({k for r in R for k in r['ph']})
    tk = sorted({k for r in R for k in r['tx']}) + list(extra_text)
    covpool = ['ob_ink', 'rv_ink', 'ob_E16', 'ob_coh', 'ob_wid']
    H = []
    for _ in range(N):
        ex = tuple(rng.choice(covpool, size=rng.integers(0, 3), replace=False))
        H.append((str(rng.choice(pk)), str(rng.choice(tk)), ex))
    return H


AMOUNT = ('n_ob', 'n_rv', 'rv_any', 'rv_cont', 'n_signs', 'n_distinct', 'n_num_marks', 'rv_signs_share', 'rv_nums_share',
          'num_lines', 'rv_total_only', 'rv_nums_per_line', 'ob_nums_per_line', 'edge_any')


def main():
    feats = sys.argv[1]
    R = C.rows(feats)
    print('rows', len(R), 'with reverse photo', sum('d_depth' in r['ph'] for r in R),
          'half A', sum(r['half'] == 'A' for r in R), flush=True)
    out = {'n': len(R)}
    if os.environ.get('PE84_ONLYPLANT'):
        N = int(os.environ.get('PE84_N', 5000))
        PL = set(x.strip() for x in open(sys.argv[3]))
        Rp = C.rows(sys.argv[2])
        for r in Rp:
            r['tx']['PLANT'] = float(r['id'] in PL)
        Hp = hyps(Rp, N, np.random.default_rng(841), extra_text=('PLANT',))
        repp, scp = engine(Rp, Hp)
        sm = summarise(repp, Hp)
        out['planted'] = dict(n_rep=sm['n_rep'], plant_hyps=sum(1 for h in Hp if h[1] == 'PLANT'),
                              plant_rep=[x for x in sm['top'] if x[1] == 'PLANT'])
        # primary on planted features with the planted flag as 'total-only'
        x = arr(Rp, 'd_depth', 'ph'); y = arr(Rp, 'PLANT', 'tx')
        out['planted']['d_depth_r'] = C.pr(x, y, C.covmat(Rp))
        out['planted']['d_E04_r'] = C.pr(arr(Rp, 'd_E04', 'ph'), y, C.covmat(Rp))
        print('planted', out['planted'], flush=True)
        json.dump(out, open(os.path.join(C.CK, 'c1_planted.json'), 'w'), indent=1)
        return
    out['primary'] = primary(R)
    print(json.dumps(out['primary'], indent=0), flush=True)
    N = int(os.environ.get('PE84_N', 5000)); NN = int(os.environ.get('PE84_NULLS', 20))
    H = hyps(R, N, rng)
    rep, sc = engine(R, H)
    out['real'] = summarise(rep, H)
    print('real', out['real']['n_rep'], out['real']['n_rep_nonamount'], flush=True)
    s = C.strata(R)
    nulls = []
    for k in range(NN):
        perm = C.permute_within(s, rng)
        repn, _ = engine(R, H, perm)
        sm = summarise(repn, H)
        nulls.append([sm['n_rep'], sm['n_rep_nonamount']])
        print('null', k, nulls[-1], flush=True)
    out['null'] = nulls
    if len(sys.argv) > 3:
        PL = set(x.strip() for x in open(sys.argv[3]))
        Rp = C.rows(sys.argv[2])
        for r in Rp:
            r['tx']['PLANT'] = float(r['id'] in PL)
        Hp = hyps(Rp, N, np.random.default_rng(841), extra_text=('PLANT',))
        repp, _ = engine(Rp, Hp)
        sm = summarise(repp, Hp)
        out['planted'] = dict(n_rep=sm['n_rep'], plant_hyps=sum(1 for h in Hp if h[1] == 'PLANT'),
                              plant_rep=[x for x in sm['top'] if x[1] == 'PLANT'])
        print('planted', out['planted'], flush=True)
    json.dump(out, open(os.path.join(C.CK, 'c1.json'), 'w'), indent=1)


def summarise(rep, H):
    import collections
    cnt = collections.Counter((H[i][0], H[i][1]) for i, sel in rep)
    nonam = [(k, v) for k, v in cnt.items() if k[1] not in AMOUNT]
    return dict(n_rep=len(rep), n_rep_nonamount=sum(v for k, v in nonam), n_pairs=len(cnt),
                top=[[a, b, v] for (a, b), v in cnt.most_common(60)])


if __name__ == '__main__':
    main()
