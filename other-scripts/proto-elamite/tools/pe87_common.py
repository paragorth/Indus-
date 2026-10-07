"""pe87 shared code: community-assembly (island/species) null models on tablets x sign forms."""
import os, sys, json, re
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common

CK = os.path.join(common.DATA, 'pe87_ckpt')
os.makedirs(CK, exist_ok=True)
M327F = {'M327', '|M327+M342|', '|M327+X|'}


def pe_base(f):
    return common.base(f)


def pc_base(f):
    if f.startswith('|'):
        return '|' + re.sub(r'~[A-Za-z0-9]+', '', f.strip('|')) + '|'
    return re.sub(r'~[A-Za-z0-9]+', '', f)


def is_form(s):
    return s != 'x' and not s.startswith('X') and (s.startswith('M') or s.startswith('|') or s[:1].isalpha()) \
        and s not in ('X',)


def tablet_lines(t):
    """Lines as (forms, system, is_header_line)."""
    out = []
    for i, l in enumerate(t['lines']):
        fs = [s for s in l['signs'] if s != 'x' and not s.lower().startswith('x') and s != '...']
        sysm = common.system_of(l['numerals']) or 'none'
        out.append((fs, sysm, i == 0 and not l['numerals']))
    return out


def pe_strata(t):
    h = common.header(t)
    if h is None:
        hf = 'none'
    elif h[0] == 'M157':
        hf = 'M157'
    elif h[0] in M327F or h[0].startswith('|M327'):
        hf = 'M327'
    else:
        hf = 'oth'
    cap = any(common.system_of(l['numerals']) in ('C', 'C*') for l in t['lines'])
    v = t.get('volume', '')
    vg = 'M26' if v in ('MDP 26', 'MDP 26S') else ('M17' if v == 'MDP 17' else ('M06' if v == 'MDP 06' else 'oth'))
    return f'{hf}|{int(cap)}|{vg}'


def pc_strata(t):
    cap = any(common.system_of([[n, common.norm_code(c)] for n, c in l['numerals']]) in ('C', 'C*') for l in t['lines'])
    return f"{t.get('period', '?')}|{int(cap)}"


def incidence(tabs, min_tab=6, rename=None):
    """tabs: list of (tid, stratum, lines). rename: {(tid, form): newform}. Returns forms, rows (list of sets), strata."""
    rows, strata = [], []
    cnt = {}
    for tid, st, lines in tabs:
        s = set()
        for fs, _, _ in lines:
            for f in fs:
                f2 = rename.get((tid, f), f) if rename else f
                s.add(f2)
        rows.append(s)
        strata.append(st)
        for f in s:
            cnt[f] = cnt.get(f, 0) + 1
    forms = sorted(f for f, c in cnt.items() if c >= min_tab)
    idx = {f: i for i, f in enumerate(forms)}
    R = [set(idx[f] for f in s if f in idx) for s in rows]
    return forms, R, strata


def to_mat(R, F):
    M = np.zeros((len(R), F), np.float32)
    for i, s in enumerate(R):
        if s:
            M[i, list(s)] = 1
    return M


def curveball(R, strata, rng, n_trades):
    """In-place curveball trades between rows of the same stratum."""
    groups = {}
    for i, s in enumerate(strata):
        groups.setdefault(s, []).append(i)
    gl = [g for g in groups.values() if len(g) >= 2]
    w = np.array([len(g) for g in gl], float); w /= w.sum()
    gi = rng.choice(len(gl), size=n_trades, p=w)
    for k in gi:
        g = gl[k]
        a, b = rng.choice(len(g), 2, replace=False)
        a, b = g[a], g[b]
        A, B = R[a], R[b]
        da = A - B; db = B - A
        if not da or not db:
            continue
        pool = list(da | db)
        rng.shuffle(pool)
        na = len(da)
        common_ = A & B
        R[a] = common_ | set(pool[:na])
        R[b] = common_ | set(pool[na:])
    return R


def seg_z(R, strata, F, rng, n_null=300, thin=None):
    M = to_mat(R, F)
    obs = M.T @ M
    Rn = [set(s) for s in R]
    thin = thin or 2 * len(R)
    curveball(Rn, strata, rng, 5 * len(R))
    s1 = np.zeros((F, F)); s2 = np.zeros((F, F))
    for _ in range(n_null):
        curveball(Rn, strata, rng, thin)
        C = to_mat(Rn, F)
        C = C.T @ C
        s1 += C; s2 += C.astype(np.float64) ** 2
    mu = s1 / n_null
    sd = np.sqrt(np.maximum(s2 / n_null - mu ** 2, 0)) + 0.5
    z = (mu - obs) / sd
    return z, obs, mu


def contexts(tabs, forms, rename=None):
    """Context count vectors per form: left, right, position class, numeral system."""
    idx = {f: i for i, f in enumerate(forms)}
    ctx = [dict() for _ in forms]
    for tid, st, lines in tabs:
        for fs, sysm, hl in lines:
            fs2 = [rename.get((tid, f), f) if rename else f for f in fs]
            n = len(fs2)
            for j, f in enumerate(fs2):
                if f not in idx:
                    continue
                d = ctx[idx[f]]
                # neighbours are mapped to base forms to avoid the planted rename leaking through neighbours
                L = fs2[j - 1] if j > 0 else '^'
                Rr = fs2[j + 1] if j < n - 1 else '$'
                pos = 'H' if hl else ('S' if n == 1 else ('I' if j == 0 else ('F' if j == n - 1 else 'M')))
                for key in ('L:' + L, 'R:' + Rr, 'P:' + pos, 'Y:' + sysm):
                    d[key] = d.get(key, 0) + 1
    return ctx


def ctx_matrix(ctx):
    keys = {}
    for d in ctx:
        for k in d:
            keys.setdefault(k, len(keys))
    X = np.zeros((len(ctx), len(keys)))
    for i, d in enumerate(ctx):
        for k, v in d.items():
            X[i, keys[k]] = v
    # normalise each block separately then concatenate
    blocks = {}
    for k, j in keys.items():
        blocks.setdefault(k[:2], []).append(j)
    for b, js in blocks.items():
        sub = X[:, js]
        nrm = np.linalg.norm(sub, axis=1, keepdims=True) + 1e-9
        X[:, js] = sub / nrm
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    return X @ X.T


def auc(pos, neg):
    pos = np.asarray(pos); neg = np.asarray(neg)
    if len(pos) == 0 or len(neg) == 0:
        return float('nan')
    allv = np.concatenate([pos, neg])
    r = allv.argsort().argsort() + 1.0
    # ties: average ranks
    from scipy.stats import rankdata
    r = rankdata(allv)
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def load_pe():
    T = common.load()
    return [(t['id'], pe_strata(t), tablet_lines(t)) for t in T], T


def load_pc():
    P = json.load(open(os.path.join(common.DATA, 'pe2_pc_corpus.json')))
    out = []
    for t in P:
        for l in t['lines']:
            l['numerals'] = [[n, common.norm_code(c)] for n, c in l['numerals']]
        out.append((t['id'], pc_strata(t), tablet_lines(t)))
    return out
