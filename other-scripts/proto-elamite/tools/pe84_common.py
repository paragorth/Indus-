"""pe84 shared: tablet rows (text features + pe81 covariates) merged with photo features of the writing."""
import sys, os, json, re, collections
import numpy as np
from scipy.stats import rankdata
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common
import pe83_common as P

SCR = P.SCR
CK = os.path.join(common.DATA, 'pe84_ckpt')
os.makedirs(CK, exist_ok=True)
FA = {'MDP 06', 'MDP 17', 'MDP 31'}  # publication half A; B = MDP 26, 26S and the rest
PHYS_CORE = ('E04', 'E08', 'E16', 'G1', 'G2', 'G3', 'G4', 'ink', 'depth', 'wid', 'ncomp', 'coh', 'diag')


def text_feats(t, topsigns):
    L = t['lines']
    ob = [l for l in L if l['surface'] == 'obverse']
    rv = [l for l in L if l['surface'] == 'reverse']
    f = {}
    f['n_ob'] = len(ob); f['n_rv'] = len(rv)
    f['rv_any'] = float(bool(rv))
    f['rv_total_only'] = float(0 < len(rv) <= 2 and all(l['numerals'] for l in rv))
    f['rv_cont'] = float(len(rv) >= 3)
    f['edge_any'] = float(any(l['surface'] not in ('obverse', 'reverse') for l in L))
    f['header'] = float(common.header(t) is not None)
    h = common.header(t) or []
    f['hdr_M157'] = float('M157' in h)
    toks = [g for l in L for g in l['signs']]
    sg = [common.base(g) for g in toks if common.is_sign(g)]
    f['n_signs'] = len(sg); f['n_distinct'] = len(set(sg))
    f['x_share'] = sum(g == 'x' for g in toks) / max(1, len(toks))
    f['dam_share'] = sum(l['damaged'] or l['lacuna'] for l in L) / max(1, len(L))
    nums = [n or 0 for l in L for n, c in l['numerals']]
    f['n_num_marks'] = sum(nums)
    f['num_lines'] = sum(1 for l in L if l['numerals']) / max(1, len(L))
    f['numonly_share'] = sum(1 for l in L if l['numerals'] and not [g for g in l['signs'] if common.is_sign(g)]) / max(1, len(L))
    f['ob_last_numonly'] = float(bool(ob) and bool(ob[-1]['numerals']) and not [g for g in ob[-1]['signs'] if common.is_sign(g)])
    ents = [l for l in L if l['numerals'] and l['signs']]
    f['signs_per_entry'] = np.mean([len(l['signs']) for l in ents]) if ents else 0.0
    sysc = collections.Counter(common.system_of(l['numerals']) for l in L if l['numerals'])
    tot = max(1, sum(sysc.values()))
    for s in ('C', 'B', 'N23', 'S-frac', 'SDB', 'C*'):
        f['sys_' + s] = sysc.get(s, 0) / tot
    codes = collections.Counter(c for l in L for n, c in l['numerals'])
    for c in ('N01', 'N14', 'N34', 'N39B', 'N24', 'N30C', 'N45', 'N48', 'N50', 'N51', 'N57'):
        f['has_' + c] = float(codes.get(c, 0) > 0)
    f['big_value'] = float(any(c in ('N45', 'N48', 'N50', 'N34') for c in codes))
    ss = set(sg)
    for s in topsigns:
        f['sg_' + s] = float(s in ss)
    f['rv_signs_share'] = sum(len(l['signs']) for l in rv) / max(1, sum(len(l['signs']) for l in L))
    f['rv_nums_share'] = sum(n or 0 for l in rv for n, c in l['numerals']) / max(1, sum(nums))
    f['ob_nums_per_line'] = sum(n or 0 for l in ob for n, c in l['numerals']) / max(1, len(ob))
    f['rv_nums_per_line'] = sum(n or 0 for l in rv for n, c in l['numerals']) / max(1, len(rv)) if rv else 0.0
    return f


def phys_feats(p):
    o, r = p.get('ob'), p.get('rv')
    if not o:
        return None
    f = {}
    for k in PHYS_CORE:
        f['ob_' + k] = o.get(k)
    for i in range(3):
        for k in ('E08', 'depth', 'ink'):
            f['ob_t%d_%s' % (i, k)] = o.get('t%d_%s' % (i, k))
    for k in ('E08', 'depth', 'ink'):
        a, b = o.get('t0_' + k), o.get('t2_' + k)
        f['ob_t2m0_' + k] = (b - a) if (a is not None and b is not None) else None
    f['ob_absang'] = abs(o['ang']) if o.get('ang') is not None else None
    if r:
        for k in PHYS_CORE:
            f['rv_' + k] = r.get(k)
            a, b = o.get(k), r.get(k)
            f['d_' + k] = (b - a) if (a is not None and b is not None) else None
    return f


def rows(featfile, mode=None):
    PH = json.load(open(featfile))
    R0 = P.rows()
    sc = collections.Counter(common.base(g) for r in R0 for l in r['t']['lines'] for g in set(l['signs']) if common.is_sign(g))
    top = [s for s, n in sc.most_common(40)]
    out = []
    for r in R0:
        p = PH.get(r['id'])
        if not p or 'err' in p:
            continue
        pf = phys_feats(p)
        if not pf:
            continue
        o = p['ob']
        r = dict(r)
        r['tx'] = text_feats(r['t'], top)
        r['ph'] = pf
        r['bright'] = o['bright']; r['s_orig'] = o['s_orig']; r['gx'] = o['gx']; r['gy'] = o['gy']
        r['rvb'] = p['rv']['bright'] if p.get('rv') else np.nan
        r['vol'] = r['t'].get('volume') or 'other'
        r['half'] = 'A' if r['vol'] in FA else 'B'
        out.append(r)
    return out


BASE_COV = ('la', 'asp', 'tw', 'nl', 'bright', 's_orig', 'gx', 'gy')


def covmat(R, extra=()):
    cols = [np.ones(len(R))] + [np.array([r[k] for r in R], float) for k in BASE_COV]
    for e in extra:
        cols.append(np.asarray(e, float))
    Z = np.column_stack(cols)
    Z[~np.isfinite(Z)] = 0.0
    return Z


def pr(x, y, Z):
    """partial Spearman-type r: rank(x), rank(y) residualised on Z; rows with nan dropped."""
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 25:
        return np.nan, int(ok.sum())
    Zk = Z[ok]
    res = lambda v: v - Zk @ np.linalg.lstsq(Zk, v, rcond=None)[0]
    a = res(rankdata(x[ok])); b = res(rankdata(y[ok]))
    sa, sb = a.std(), b.std()
    if sa < 1e-9 or sb < 1e-9:
        return np.nan, int(ok.sum())
    return float(np.corrcoef(a, b)[0, 1]), int(ok.sum())


def strata(R):
    la = np.array([r['la'] for r in R]); nl = np.array([r['nl'] for r in R])
    q1 = np.digitize(la, np.quantile(la, [.33, .67])); q2 = np.digitize(nl, np.quantile(nl, [.33, .67]))
    v = np.array([hash(r['vol']) % 997 for r in R])
    return v * 10 + q1 * 3 + q2


def permute_within(s, rng):
    idx = np.arange(len(s))
    for k in np.unique(s):
        m = np.where(s == k)[0]
        idx[m] = rng.permutation(m)
    return idx
