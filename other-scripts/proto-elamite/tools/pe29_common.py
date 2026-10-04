"""pe29 shared code: REBUILD THE WRITING SESSIONS.

Evidence matrices (tablet x tablet similarity, NaN = not measurable):
  pool  : shared rare signs (IDF-weighted cosine over signs found on 2..40 tablets)       [PE base signs | Ur III words]
  hand  : free variant choice: for every base form written on both tablets, agreement of the
          variant chosen minus the chance agreement from corpus frequencies              [PE ~a/~b | Ur III sign index]
  clay  : batch-centred colour distance, same photo batch only (pe26 features)           [PE only]
  mus   : museum-number adjacency exp(-gap/25), same prefix only
  size  : -|log h, log w, log t| distance
Clustering: random evidence weights, kNN graph, Louvain at random resolution, many restarts -> co-clustering matrix.
"""
import json, os, re, csv, collections, math
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, '..', 'data')
CK = os.path.join(DATA, 'pe29_ckpt')
os.makedirs(CK, exist_ok=True)
SCRATCH = '/tmp/claude-0/-home-user-Indus-/874df4c7-80d6-5f08-b42c-eea96a214079/scratchpad'
EVS = ['pool', 'hand', 'clay', 'mus', 'size']

import sys
sys.path.insert(0, HERE)
from common import load, base, is_sign, system_of  # noqa


def _num(s):
    try:
        v = float(s)
        return v if v > 0 else None
    except Exception:
        return None


def _musparse(m):
    m = (m or '').strip()
    mm = re.match(r'^(.*?)(\d+)\s*$', m.replace('—', '').replace('?', '').strip())
    if not mm or '—' in m:
        return None, None
    pre = re.sub(r'\s+', ' ', mm.group(1)).strip()
    return pre, int(mm.group(2))


# ------------------------------------------------------------------ PE
def pe_tablets():
    T = load()
    cat = json.load(open(os.path.join(DATA, 'pe17_ckpt', 'pe_cat.json')))
    out = []
    for t in T:
        if 'Susa' not in (t['provenience'] or ''):
            continue
        signs = [s for l in t['lines'] for s in l['signs'] if is_sign(s)]
        if len(signs) < 2:
            continue
        c = cat[t['id']]
        pre, no = _musparse(c['museum_no'])
        sysc = collections.Counter(system_of(l['numerals']) for l in t['lines'] if l['numerals'])
        out.append(dict(id=t['id'], signs=signs, bases=[base(s) for s in signs], lines=t['lines'],
                        pre=pre, no=no, h=_num(c['height']), w=_num(c['width']), t=_num(c['thickness']),
                        nl=len(t['lines']), sys=(sysc.most_common(1)[0][0] if sysc else 'none'),
                        pub=c['designation'].split(',')[0]))
    return out


def pe_clay(tabs):
    """within-batch-centred colour vectors and batch keys (None when no photo)."""
    from pe26_common import load as l26, X_of, within_batch_centre, COLOUR
    rows = [r for r in l26() if r['group'] == 'PE']
    X = within_batch_centre(X_of(rows, COLOUR), [r['batch'] for r in rows])
    X = (X - X.mean(0)) / (X.std(0) + 1e-9)
    d = {r['id']: (X[i], r['batch']) for i, r in enumerate(rows)}
    V = np.full((len(tabs), len(COLOUR)), np.nan); B = []
    for i, t in enumerate(tabs):
        if t['id'] in d:
            V[i] = d[t['id']][0]; B.append(d[t['id']][1])
        else:
            B.append(None)
    return V, B


# ------------------------------------------------------------------ Ur III control
def ur3_tablets(king='Amar-Suen', year='05', cache=None):
    cache = cache or os.path.join(CK, f'ur3_{king[:3]}{year}.json')
    if os.path.exists(cache):
        return json.load(open(cache))
    csv.field_size_limit(10 ** 9)
    keep = {}
    for row in csv.DictReader(open(os.path.join(SCRATCH, 'cdli_cat.csv'), encoding='utf-8')):
        if row['period'].startswith('Ur III') and 'Puzri' in row['provenience']:
            m = re.match(r'^([^.]+)\.(\d\d)\.(\d\d)\.(\d\d)', row['dates_referenced'])
            if m and m.group(1) == king and m.group(2) == year and m.group(3) != '00' and m.group(4) != '00':
                keep['P%06d' % int(row['id_text'])] = dict(date=m.group(0), mus=row['museum_no'], h=row['height'],
                                                           w=row['width'], t=row['thickness'])
    texts = collections.defaultdict(list); cur = None; seal = False
    for raw in open(os.path.join(SCRATCH, 'cdli.atf'), encoding='utf-8', errors='replace'):
        if raw.startswith('&P'):
            pid = raw[1:8]; cur = pid if pid in keep else None; seal = False; continue
        if not cur:
            continue
        if raw.startswith('@seal'):
            seal = True; continue
        if raw.startswith('@obverse') or raw.startswith('@reverse') or raw.startswith('@edge'):
            seal = False; continue
        if seal or not raw[:1].isdigit():
            continue
        m = re.match(r"^\d+'?\.\s+(.*)$", raw.rstrip())
        if not m:
            continue
        body = m.group(1)
        tk = [re.sub(r'[#!?\[\]<>]', '', x) for x in body.split()]
        tk = [x for x in tk if x and x not in ('...', 'x')]
        if not tk:
            continue
        # date formulas would leak the answer: drop month, year and day lines entirely
        if tk[0] in ('iti', 'mu', 'u4', 'u4-1(u)-kam', 'iti-ta') or tk[0].startswith('u4-') or tk[0].startswith('mu-us2'):
            continue
        texts[cur].append(tk)
    out = []
    for pid, L in texts.items():
        k = keep[pid]
        words = [w for l in L for w in l if not re.match(r'^\d|^n\(', w)]
        if len(words) < 2:
            continue
        signs = [s for w in words for s in re.split(r'[-.{}]', w) if s]
        pre, no = _musparse(k['mus'])
        out.append(dict(id=pid, date=k['date'], bases=words, signs=signs, pre=pre, no=no,
                        h=_num(k['h']), w=_num(k['w']), t=_num(k['t']), nl=len(L)))
    json.dump(out, open(cache, 'w'))
    return out


# ------------------------------------------------------------------ evidence matrices
def _idf_cos(lists, lo=2, hi=40):
    n = len(lists)
    df = collections.Counter(x for L in lists for x in set(L))
    voc = {x: i for i, x in enumerate(sorted(x for x, c in df.items() if lo <= c <= hi))}
    M = np.zeros((n, len(voc)), np.float32)
    for i, L in enumerate(lists):
        for x in set(L):
            if x in voc:
                M[i, voc[x]] = math.log(n / df[x])
    nr = np.linalg.norm(M, axis=1); ok = nr > 0
    M[ok] /= nr[ok, None]
    S = M @ M.T
    S[~ok, :] = np.nan; S[:, ~ok] = np.nan
    return S


def sim_pool(tabs):
    return _idf_cos([t['bases'] for t in tabs])


def _variant_key(s, ur):
    if ur:
        m = re.match(r'^([a-z%\'šŋḫṣṭ]+?)(\d+|x)?$', s)
        if not m:
            return None, None
        return m.group(1), m.group(2) or '1'
    b = base(s)
    return b, s


def sim_hand(tabs, ur=False):
    n = len(tabs)
    prof = []
    tot = collections.defaultdict(collections.Counter)
    for t in tabs:
        d = collections.defaultdict(collections.Counter)
        for s in t['signs']:
            b, v = _variant_key(s, ur)
            if b:
                d[b][v] += 1
        prof.append(d)
        for b, c in d.items():
            for v in c:
                tot[b][v] += 1
    vb = {b for b, c in tot.items() if len(c) >= 2 and sum(c.values()) >= 10}
    pexp = {}
    for b in vb:
        c = tot[b]; s = sum(c.values()); pexp[b] = sum((x / s) ** 2 for x in c.values())
    bases = sorted(vb); bi = {b: i for i, b in enumerate(bases)}
    # main variant per base per tablet
    V = np.full((n, len(bases)), -1, np.int32); vid = {}
    for i, d in enumerate(prof):
        for b, c in d.items():
            if b in bi:
                v = c.most_common(1)[0][0]
                V[i, bi[b]] = vid.setdefault((b, v), len(vid))
    pe = np.array([pexp[b] for b in bases], np.float32)
    has = (V >= 0).astype(np.float32)
    S = np.zeros((n, n), np.float32); N = has @ has.T
    for j in range(len(bases)):
        col = V[:, j]; idx = np.where(col >= 0)[0]
        if len(idx) < 2:
            continue
        same = (col[idx, None] == col[None, idx]).astype(np.float32) - pe[j]
        S[np.ix_(idx, idx)] += same
    with np.errstate(invalid='ignore', divide='ignore'):
        S = S / np.sqrt(N)
    S[N < 1] = np.nan
    return S


def sim_mus(tabs, scale=25.0):
    n = len(tabs)
    pre = np.array([t['pre'] or '' for t in tabs]); no = np.array([t['no'] if t['no'] is not None else -10 ** 9 for t in tabs], float)
    S = np.exp(-np.abs(no[:, None] - no[None, :]) / scale)
    ok = (pre[:, None] == pre[None, :]) & (pre[:, None] != '') & (no[:, None] > -1e8) & (no[None, :] > -1e8)
    S[~ok] = np.nan
    return S.astype(np.float32)


def sim_size(tabs):
    X = np.array([[math.log(t[k]) if t[k] else np.nan for k in ('h', 'w', 't')] for t in tabs])
    ok = ~np.isnan(X).any(1)
    S = np.full((len(tabs), len(tabs)), np.nan, np.float32)
    Xo = X[ok]
    D = np.sqrt(((Xo[:, None, :] - Xo[None, :, :]) ** 2).sum(-1))
    S[np.ix_(ok, ok)] = -D
    return S


def sim_clay(V, B):
    n = len(B)
    S = np.full((n, n), np.nan, np.float32)
    bb = collections.defaultdict(list)
    for i, b in enumerate(B):
        if b is not None:
            bb[b].append(i)
    for b, idx in bb.items():
        if len(idx) < 2:
            continue
        X = V[idx]
        D = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(-1))
        S[np.ix_(idx, idx)] = -D
    return S


def zmat(S):
    S = S.astype(np.float32).copy()
    np.fill_diagonal(S, np.nan)
    m = ~np.isnan(S)
    if m.sum() == 0:
        return np.zeros_like(S)
    mu, sd = S[m].mean(), S[m].std() + 1e-9
    Z = (S - mu) / sd
    Z[~m] = 0.0
    return Z


def permute(S, groups, rng):
    """relabel tablets of one evidence matrix inside groups (shuffled-evidence null)."""
    n = len(S); p = np.arange(n)
    g = collections.defaultdict(list)
    for i, x in enumerate(groups):
        g[x].append(i)
    for idx in g.values():
        p[idx] = rng.permutation(idx)
    return S[np.ix_(p, p)]


# ------------------------------------------------------------------ clustering
def cocluster(Z, evs, nrest, rng, k=5, res=(0.6, 2.5), wdir=1.0):
    import networkx as nx
    n = next(iter(Z.values())).shape[0]
    C = np.zeros((n, n), np.float32)
    parts = []
    for r in range(nrest):
        w = rng.dirichlet([wdir] * len(evs))
        A = sum(w[j] * Z[e] for j, e in enumerate(evs))
        A = A + rng.normal(0, 0.05, A.shape).astype(np.float32)
        np.fill_diagonal(A, -1e9)
        nb = np.argpartition(-A, k, axis=1)[:, :k]
        G = nx.Graph(); G.add_nodes_from(range(n))
        for i in range(n):
            for j in nb[i]:
                if A[i, j] > 0:
                    G.add_edge(i, int(j), weight=float(A[i, j]))
        comms = nx.community.louvain_communities(G, weight='weight', resolution=float(rng.uniform(*res)), seed=int(rng.integers(1 << 30)))
        lab = np.empty(n, int)
        for c, mem in enumerate(comms):
            lab[list(mem)] = c
        parts.append(lab)
        C += (lab[:, None] == lab[None, :])
    C /= nrest
    np.fill_diagonal(C, 0)
    return C, parts


def lift(C, S, thr=0.5):
    """mean z of held-out evidence over pairs co-clustered >= thr (only measurable pairs), and Spearman-like corr."""
    iu = np.triu_indices(len(C), 1)
    c = C[iu]; s = S[iu]; ok = ~np.isnan(s)
    c, s = c[ok], s[ok]
    s = (s - s.mean()) / (s.std() + 1e-9)
    sel = c >= thr
    return float(s[sel].mean()) if sel.sum() else float('nan'), int(sel.sum()), float(np.corrcoef(c, s)[0, 1])


def pair_auc(score, truth_lab):
    from sklearn.metrics import roc_auc_score
    lab = np.array(truth_lab)
    iu = np.triu_indices(len(lab), 1)
    y = (lab[:, None] == lab[None, :])[iu]
    return float(roc_auc_score(y, score[iu]))


def consensus_partition(C, thr=0.5):
    """connected components of the 'co-clustered in >= thr of restarts' graph."""
    import networkx as nx
    n = len(C)
    G = nx.Graph(); G.add_nodes_from(range(n))
    ii, jj = np.where(np.triu(C >= thr, 1))
    G.add_edges_from(zip(ii.tolist(), jj.tolist()))
    lab = np.empty(n, int)
    for c, mem in enumerate(nx.connected_components(G)):
        lab[list(mem)] = c
    return lab


def colink(Z, evs, nrest, rng, k=2, wdir=1.0, drop=0.0):
    """mutual-kNN session links under random evidence weights; returns link frequency matrix.
    drop: probability that each evidence is switched off in a restart (random evidence subsets)."""
    n = next(iter(Z.values())).shape[0]
    C = np.zeros((n, n), np.float32)
    for r in range(nrest):
        w = rng.dirichlet([wdir] * len(evs))
        if drop:
            m = rng.random(len(evs)) > drop
            if not m.any():
                m[rng.integers(len(evs))] = True
            w = w * m
        A = np.zeros((n, n), np.float32)
        for j, e in enumerate(evs):
            if w[j] > 0:
                A += np.float32(w[j]) * Z[e]
        np.fill_diagonal(A, -1e9)
        nb = np.argpartition(-A, k, axis=1)[:, :k]
        rows = np.repeat(np.arange(n), k); cols = nb.ravel()
        S = set(zip(rows.tolist(), cols.tolist()))
        mu = [(a, b) for a, b in S if (b, a) in S]
        if mu:
            a, b = np.array(mu).T
            C[a, b] += 1
    C /= nrest
    return C
