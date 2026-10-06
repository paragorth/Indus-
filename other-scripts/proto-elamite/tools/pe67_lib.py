"""pe67 EACH OUTPOST WRITES ABOUT WHAT IT DIGS UP: shared library.

Arrow in the dark: every PE outpost's excavated material assemblage is data outside the script. Random
sign-to-material assignments are scored by how well outpost-specific signs line up with the outposts'
distinctive materials, with a number-system constraint (GRAIN-like goods take capacity numerals; animals and
objects take counting numerals), tested leave-one-outpost-out.

Corpus format: docs = list of dict(site, signs=set(ids), sys={id: [n_cap, n_cnt]}).
Material table: dict site -> dict material -> ordinal (0/1/2); kinds GRAIN / ANIMAL / OBJ / EITHER.

Score (per sign s, material m, over a site set J):
   r_sj = (k_sj - n_j p_s) / sqrt(n_j p_s (1 - p_s))   (tablet-presence residual vs the pooled rate)
   z(s,m) = sum_j (M_jm - mean_J M_m) r_sj / sqrt(sum_j (M_jm - mean_J M_m)^2)
A hypothesis is K compatible (sign, material) pairs with distinct signs; its score is mean z.
"""
import os, sys, json, math, random, hashlib
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
import numpy as np
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import load, base, is_sign, system_of  # noqa: E402

PEROOT = os.path.abspath(os.path.join(HERE, '..'))
DATA = os.path.join(PEROOT, 'data')
LOOPS = os.path.join(PEROOT, 'loops')
CK = os.path.join(DATA, 'pe67_ckpt'); os.makedirs(CK, exist_ok=True)
MAT = json.load(open(os.path.join(DATA, 'pe67_materials.json')))

PE_SITE = {'Susa (mod. Shush)': 'Susa', 'Susa (mod. Shush) ?': 'Susa',
           'uncertain (mod. Tepe Yahya)': 'Yahya', 'Anšan (mod. Tell Malyan)': 'Malyan',
           'uncertain (mod. Tepe Sialk)': 'Sialk', 'uncertain (mod. Tepe Sofalin)': 'Sofalin',
           'uncertain (mod. Shahr-i Sokhta)': 'Shahr-i Sokhta', 'uncertain (mod. Ozbaki)': 'Ozbaki'}
PE_OUT = ['Yahya', 'Malyan', 'Sialk', 'Sofalin']          # outposts with >= 11 tablets (LOO targets)
PE_SING = ['Ozbaki', 'Shahr-i Sokhta']                     # one tablet each: prediction only


def seed(name):
    return int(hashlib.sha256(name.encode()).hexdigest()[:8], 16)


def pe_docs():
    out = []
    for t in load():
        site = PE_SITE.get(t['provenience'])
        if not site:
            continue
        signs, sysd = set(), defaultdict(lambda: [0, 0])
        for l in t['lines']:
            sg = [base(s) for s in l['signs'] if is_sign(s)]
            if not sg:
                continue
            signs.update(sg)
            sy = system_of(l['numerals']) if l['numerals'] else None
            for s in set(sg):
                if sy in ('C', 'C*'):
                    sysd[s][0] += 1
                elif sy in ('SDB', 'B', 'N23'):
                    sysd[s][1] += 1
        if signs:
            out.append(dict(id=t['id'], site=site, signs=signs, sys=dict(sysd)))
    return out


def ur3_docs_all():
    fn = os.path.join(CK, 'ur3_bags.json')
    if os.path.exists(fn):
        d = json.load(open(fn))
        for x in d:
            x['signs'] = set(x['signs'])
        return d
    d = json.load(open(os.path.join(DATA, 'pe38_ckpt', 'ur3_docs.json')))
    keep = set(MAT['UR3'])
    out = []
    for x in d:
        site = 'Irisagrig' if x['site'] == 'Irisaĝrig' else x['site']
        if site not in keep:
            continue
        signs, sysd = set(), defaultdict(lambda: [0, 0])
        for l in x['lines']:
            ws = [w for w in l['toks'] if w and w not in ('x', '...')]
            if not ws:
                continue
            signs.update(ws)
            for w in set(ws):
                if l['sys'] == 2:
                    sysd[w][0] += 1
                elif l['sys'] in (1, 3):
                    sysd[w][1] += 1
        if signs:
            out.append(dict(id=x['id'], site=site, signs=sorted(signs), sys=dict(sysd)))
    json.dump(out, open(fn, 'w'))
    for x in out:
        x['signs'] = set(x['signs'])
    return out


def ur3_shape(docs, hub, outposts, sizes=(1500, 27, 22, 12, 11, 1, 1), rng=None):
    """Draw a PE-shaped Ur III corpus: hub of 1500 docs, outposts of 27/22/12/11/1/1."""
    by = defaultdict(list)
    for x in docs:
        by[x['site']].append(x)
    out = rng.sample(by[hub], sizes[0])
    for s, n in zip(outposts, sizes[1:]):
        out += rng.sample(by[s], n)
    return out


# ------------------------------------------------------------------ profiles
def build(docs, min_tab=5):
    """Return signs list, sites list, K (signs x sites presence counts), n (tablets per site), sysprof."""
    sites = sorted({x['site'] for x in docs})
    sidx = {s: i for i, s in enumerate(sites)}
    cnt = Counter()
    for x in docs:
        cnt.update(x['signs'])
    signs = sorted(s for s, c in cnt.items() if c >= min_tab)
    gidx = {s: i for i, s in enumerate(signs)}
    K = np.zeros((len(signs), len(sites)))
    n = np.zeros(len(sites))
    SY = np.zeros((len(signs), len(sites), 2))
    for x in docs:
        j = sidx[x['site']]
        n[j] += 1
        for s in x['signs']:
            i = gidx.get(s)
            if i is not None:
                K[i, j] += 1
        for s, v in x['sys'].items():
            i = gidx.get(s)
            if i is not None:
                SY[i, j] += v
    return dict(signs=signs, sites=sites, K=K, n=n, SY=SY)


def sign_class(SYsum, thr=0.5, min_lines=2):
    """CAP / CNT / NONE from capacity vs counting lines (summed over the given sites)."""
    cap, cn = SYsum[:, 0], SYsum[:, 1]
    tot = cap + cn
    out = np.array(['NONE'] * len(cap), dtype=object)
    out[(tot >= min_lines) & (cap / np.maximum(tot, 1) >= thr)] = 'CAP'
    out[(tot >= min_lines) & (cn / np.maximum(tot, 1) > 1 - thr)] = 'CNT'
    return out


def compat(sclass, kinds, allow_none=False):
    """Boolean matrix signs x materials."""
    C = np.zeros((len(sclass), len(kinds)), bool)
    for m, k in enumerate(kinds):
        if k == 'GRAIN':
            C[:, m] = sclass == 'CAP'
        elif k in ('ANIMAL', 'OBJ'):
            C[:, m] = sclass == 'CNT'
        else:
            C[:, m] = sclass != 'NONE'
        if allow_none:
            C[:, m] |= sclass == 'NONE'
    return C


def residuals(K, n, J):
    """r (signs x len(J)) and pooled p over sites J."""
    KJ, nJ = K[:, J], n[J]
    p = KJ.sum(1) / nJ.sum()
    E = np.outer(p, nJ)
    V = np.maximum(E * (1 - p)[:, None], 1e-6)
    return (KJ - E) / np.sqrt(V), p


def zmat(K, n, M, J):
    """z (signs x materials) over sites J; M is sites x materials (all sites)."""
    r, p = residuals(K, n, J)
    MJ = M[J]
    D = MJ - MJ.mean(0, keepdims=True)
    nd = np.sqrt((D ** 2).sum(0))
    nd[nd == 0] = np.inf
    return (r @ D) / nd[None, :]


def random_search(Z, C, rng, nh=20000, kmin=3, kmax=8, top=0.01):
    """Thousands of random compatible sign-material hypotheses; return pair support among survivors,
    best score and survivor threshold."""
    pairs = np.argwhere(C & np.isfinite(Z))
    if len(pairs) == 0:
        return dict(support=Counter(), best=0.0, thr=0.0, n=0)
    zp = Z[pairs[:, 0], pairs[:, 1]]
    scores, hyps = np.empty(nh), []
    for h in range(nh):
        k = rng.randint(kmin, kmax)
        idx = rng.sample(range(len(pairs)), min(k * 2, len(pairs)))
        seen, pick = set(), []
        for i in idx:
            s = pairs[i, 0]
            if s in seen:
                continue
            seen.add(s); pick.append(i)
            if len(pick) == k:
                break
        scores[h] = zp[pick].mean()
        hyps.append(pick)
    thr = np.quantile(scores, 1 - top)
    sup = Counter()
    for h in np.where(scores >= thr)[0]:
        for i in hyps[h]:
            sup[(int(pairs[i, 0]), int(pairs[i, 1]))] += 1
    return dict(support=sup, best=float(scores.max()), thr=float(thr), n=len(pairs))


def select_links(sup, Z, nsel=10):
    """Top pairs by survivor support (ties by z); one material per sign."""
    items = sorted(sup.items(), key=lambda kv: (-kv[1], -Z[kv[0]]))
    out, used = [], set()
    for (s, m), c in items:
        if s in used or Z[s, m] <= 0:
            continue
        used.add(s); out.append((s, m))
        if len(out) == nsel:
            break
    return out


def loo_stat(K, n, M, Jtrain, o, links):
    """Held-out outpost o: sum over links of (M_om - mean_train M_m) * r_so (r from the training pooled rate)."""
    KJ, nJ = K[:, Jtrain], n[Jtrain]
    p = KJ.sum(1) / nJ.sum()
    T, used = 0.0, 0
    mu = M[Jtrain].mean(0)
    for s, m in links:
        d = M[o, m] - mu[m]
        if d == 0:
            continue
        E = n[o] * p[s]
        r = (K[s, o] - E) / math.sqrt(max(E * (1 - p[s]), 1e-6))
        T += d * r; used += 1
    return T, used


def run_loo(D, M, kinds, outposts, rng, nh=20000, nsel=10, allow_none=False, hub=None):
    """Leave-one-outpost-out over the given outpost site names. D from build(); M is sites x materials
    aligned to D['sites']. Returns per-outpost T and links."""
    sites = D['sites']
    res = {}
    for oname in outposts:
        o = sites.index(oname)
        J = [j for j in range(len(sites)) if j != o and D['n'][j] >= 3]
        Z = zmat(D['K'], D['n'], M, J)
        scl = sign_class(D['SY'][:, J].sum(1))
        C = compat(scl, kinds, allow_none)
        rs = random_search(Z, C, rng, nh=nh)
        links = select_links(rs['support'], Z, nsel)
        T, used = loo_stat(D['K'], D['n'], M, J, o, links)
        res[oname] = dict(T=T, used=used, links=[(D['signs'][s], int(m)) for s, m in links], best=rs['best'])
    return res


def mat_matrix(table, sites, mats):
    M = np.zeros((len(sites), len(mats)))
    for i, s in enumerate(sites):
        row = table.get(s, {})
        for j, m in enumerate(mats):
            M[i, j] = row.get(m, 0)
    return M


def shuffle_sites(docs, rng):
    labs = [x['site'] for x in docs]
    rng.shuffle(labs)
    return [dict(x, site=l) for x, l in zip(docs, labs)]
