"""pe49: find the clerks by their tics, then read their job descriptions.

Common representation (both corpora reduced to it):
  tablet = {'id', 'group' (true scribe or None), 'lines': [ {surf, kind, toks, words, nums, numpos, divider, mark} ]}
  toks  : opaque sign codes (PE: sign with variant; Ur III: sign value -> 'S<n>')
  words : tuples of toks (PE: one entry string; Ur III: hyphen-joined words)
  nums  : [(count, unit)] in written order
Habit features (content-blind tics) and content features (bag of base signs +
number-unit set) are computed by the same code for every corpus.
"""
import json, os, re, sys, collections, random
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, '..', 'data')
CK = os.path.join(DATA, 'pe49_ckpt')
os.makedirs(CK, exist_ok=True)
sys.path.insert(0, HERE)


# ---------------------------------------------------------------- PE ----
def load_pe(min_numeric=4, susa_only=True):
    import common
    T = common.load()
    out = []
    for t in T:
        if susa_only and not t['provenience'].startswith('Susa'):
            continue
        L = []
        for l in t['lines']:
            raw = l.get('raw', '')
            toks = list(l['signs'])
            L.append({'surf': l['surface'], 'col': l.get('column', 1), 'kind': 'text', 'toks': toks,
                      'words': [tuple(toks)] if toks else [], 'base': [common.base(s) for s in toks],
                      'nums': [(n, c) for n, c in l['numerals']],
                      'numpos': ('post' if toks and l['numerals'] else ('only' if l['numerals'] else 'none')),
                      'between': 0, 'divider': bool(l['has_comma']),
                      'mark': raw.count('!'), 'label': l.get('label', '')})
        if sum(1 for x in L if x['nums']) < min_numeric:
            continue
        out.append({'id': t['id'], 'group': None, 'desig': t.get('designation', ''), 'lines': L,
                    'dollars': [], 'pubser': re.sub(r'[ ,].*$', '', t.get('designation', '')) + ' ' +
                    (t.get('designation', '').split(',')[0].split()[-1] if t.get('designation') else '')})
    _attach_pe_dollars(out)
    return out


def _attach_pe_dollars(out):
    """$ lines and edge/column structure from the raw ATF."""
    idx = {t['id']: t for t in out}
    cur = None
    surf = None
    for raw in open(os.path.join(DATA, 'pe_raw.atf'), encoding='utf-8'):
        if raw.startswith('&P'):
            cur = idx.get(raw[1:8])
            surf = None
            if cur is not None:
                cur['surfs'] = set()
                cur['ncol'] = 1
            continue
        if cur is None:
            continue
        if raw.startswith('@'):
            tag = raw[1:].split()[0] if raw[1:].split() else ''
            if tag in ('obverse', 'reverse', 'top', 'bottom', 'left', 'right', 'edge'):
                surf = tag
                cur['surfs'].add(tag)
            if tag == 'column':
                try:
                    cur['ncol'] = max(cur['ncol'], int(raw.split()[1]))
                except Exception:
                    pass
        elif raw.startswith('$'):
            cur['dollars'].append((surf, raw[1:].strip()))
        elif raw.startswith('#') and 'sic' in raw:
            cur.setdefault('sic', 0)
            cur['sic'] = cur.get('sic', 0) + 1


# ------------------------------------------------------------- Ur III ----
NUMW = re.compile(r'^(\d+(?:/\d+)?)\(([^)]+)\)$')


def _ur_sign_split(word):
    w = re.sub(r'[#?*\[\]<>]', '', word)
    parts = re.split(r'(\{[^}]*\})|[-.]', w)
    return [p for p in parts if p]


def load_ur3(n_scribes=12, per=60, prov='Umma', seed=0, min_numeric=2, min_tabs=100):
    R = json.load(open(os.path.join(CK, 'ur3_scribes.json')))
    cnt = collections.Counter(r['scribe'] for r in R if r['prov'] == prov)
    rng = random.Random(seed)
    cands = [s for s, c in cnt.items() if c >= min_tabs]
    rng.shuffle(cands)
    code = {}

    def oc(s):
        if s not in code:
            code[s] = 'S%d' % len(code)
        return code[s]
    out = []
    chosen = []
    for s in cands:
        rs = [r for r in R if r['prov'] == prov and r['scribe'] == s]
        tabs = []
        for r in rs:
            t = _ur_tab(r, oc)
            if t and sum(1 for x in t['lines'] if x['kind'] == 'text' and x['nums']) >= min_numeric:
                tabs.append(t)
        if len(tabs) < per:
            continue
        rng.shuffle(tabs)
        out += tabs[:per]
        chosen.append(s)
        if len(chosen) == n_scribes:
            break
    return out, chosen


def _ur_tab(r, oc):
    L = []
    dollars = []
    surfs = set()
    ncol = 1
    for l in r['lines']:
        if l['kind'] == 'column':
            ncol += 1
            continue
        if l['kind'] == 'dollar':
            dollars.append((l['surf'], l['txt']))
            continue
        if l['kind'] == 'comment':
            continue
        surfs.add(l['surf'])
        txt = l['txt']
        mark = txt.count('!')
        ws = txt.split()
        toks, words, nums, base = [], [], [], []
        order = []
        for w in ws:
            wc = re.sub(r'[#?!*\[\]<>]', '', w)
            m = NUMW.match(wc)
            if m:
                nums.append((float(eval(m.group(1))) if '/' in m.group(1) else int(m.group(1)), 'U_' + m.group(2)))
                order.append('n')
                continue
            raw_s = _ur_sign_split(w)
            ss = [oc(s) for s in raw_s]
            base += [oc('STEM:' + re.sub(r'(\d+|x)$', '', re.sub(r'\(.*\)', '', s))) for s in raw_s]
            if ss:
                toks += ss
                words.append(tuple(ss))
                order.append('s')
        os_ = ''.join(order)
        numpos = 'none'
        if nums and toks:
            numpos = 'pre' if os_[0] == 'n' else 'post'
        elif nums:
            numpos = 'only'
        between = 1 if re.search(r'ns+n', os_) else 0
        L.append({'surf': l['surf'], 'col': 1, 'kind': 'text', 'toks': toks, 'words': words, 'base': base,
                  'nums': nums, 'numpos': numpos, 'between': between, 'divider': None, 'mark': mark})
    if not L:
        return None
    return {'id': r['pid'], 'group': r['scribe'], 'desig': r['desig'], 'lines': L, 'dollars': dollars,
            'surfs': surfs, 'ncol': ncol, 'sic': 0}


# ------------------------------------------------------------ features ---
def variant_families(tabs, kind, min_tok=8, min_tab=5):
    """Interchangeable forms. kind 'pe': base sign -> its ~variants (given by the
    transliteration). kind 'word': word forms one insertion/deletion apart (generic,
    used for both corpora)."""
    fam = collections.defaultdict(collections.Counter)
    ftabs = collections.defaultdict(set)
    if kind == 'pe':
        for t in tabs:
            for l in t['lines']:
                for s, b in zip(l['toks'], l['base']):
                    if not s.startswith('|'):
                        fam[b][s] += 1
                        ftabs[(b, s)].add(t['id'])
        F = {}
        for b, c in fam.items():
            ok = [s for s, n in c.items() if n >= min_tok and len(ftabs[(b, s)]) >= min_tab]
            if len(ok) >= 2:
                F['V:' + b] = set(ok)
        return F
    wc = collections.Counter()
    wt = collections.defaultdict(set)
    for t in tabs:
        for l in t['lines']:
            for w in l['words']:
                wc[w] += 1
                wt[w].add(t['id'])
    good = {w for w, n in wc.items() if n >= min_tok and len(wt[w]) >= min_tab and len(w) >= 2}
    F = {}
    for w in good:
        for i in range(len(w)):
            s = w[:i] + w[i + 1:]
            if s in good and len(s) >= 2:
                F['W:' + '.'.join(s) + '|' + str(i)] = {s, w}
    return F


def habit_matrix(tabs, Fpe, Fw):
    """Tablet x habit-feature matrix with NaN = not observable on this tablet."""
    names = []
    rows = []
    # unit precedence (corpus majority) for order violations
    prec = collections.Counter()
    for t in tabs:
        for l in t['lines']:
            u = [c for _, c in l['nums']]
            for i in range(len(u)):
                for j in range(i + 1, len(u)):
                    if u[i] != u[j]:
                        prec[(u[i], u[j])] += 1
    q99 = collections.defaultdict(list)
    for t in tabs:
        for l in t['lines']:
            for n, c in l['nums']:
                if isinstance(n, (int, float)):
                    q99[c].append(n)
    q99 = {c: np.percentile(v, 99) for c, v in q99.items() if len(v) >= 30}
    w2fam = {}
    for f, forms in Fw.items():
        for w in forms:
            w2fam.setdefault(w, []).append(f)
    s2fam = {}
    for f, forms in Fpe.items():
        for s in forms:
            s2fam[s] = f
    famkeys = sorted(Fpe) + sorted(Fw)
    dom = {}
    for f in Fpe:
        c = collections.Counter()
        for t in tabs:
            for l in t['lines']:
                for s in l['toks']:
                    if s in Fpe[f]:
                        c[s] += 1
        dom[f] = c.most_common(1)[0][0]
    for f in Fw:
        c = collections.Counter()
        for t in tabs:
            for l in t['lines']:
                for w in l['words']:
                    if w in Fw[f]:
                        c[w] += 1
        dom[f] = c.most_common(1)[0][0]
    base_names = ['div_omit', 'ord_viol', 'unexch', 'between', 'numfirst', 'multi_grp', 'blank', 'ruling', 'edge',
                  'multicol', 'rev_share', 'nonnum_mid', 'first_numeric', 'last_numeric', 'mark', 'toks_per_line',
                  'units_per_num', 'rev_used', 'sic']
    names = base_names + famkeys
    for t in tabs:
        T = [l for l in t['lines'] if l['kind'] == 'text']
        num = [l for l in T if l['nums']]
        v = {}
        dv = [l['divider'] for l in num if l['divider'] is not None and l['toks']]
        v['div_omit'] = (1 - np.mean(dv)) if len(dv) >= 3 else np.nan
        multi = [l for l in num if len(l['nums']) >= 2]
        if len(multi) >= 2:
            bad = 0
            for l in multi:
                u = [c for _, c in l['nums']]
                bad += any(prec[(u[j], u[i])] > prec[(u[i], u[j])] for i in range(len(u)) for j in range(i + 1, len(u)) if u[i] != u[j])
            v['ord_viol'] = bad / len(multi)
        else:
            v['ord_viol'] = np.nan
        allnum = [(n, c) for l in num for n, c in l['nums'] if c in q99 and isinstance(n, (int, float))]
        v['unexch'] = np.mean([n > q99[c] for n, c in allnum]) if len(allnum) >= 3 else np.nan
        v['between'] = np.mean([l['between'] for l in num]) if num else np.nan
        mixed = [l for l in num if l['toks']]
        v['numfirst'] = np.mean([l['numpos'] == 'pre' for l in mixed]) if len(mixed) >= 3 else np.nan
        v['multi_grp'] = np.mean([len(set(c for _, c in l['nums'])) < len(l['nums']) for l in num]) if num else np.nan
        dl = [d for _, d in t['dollars']]
        v['blank'] = sum('blank' in d for d in dl) / max(1, len(T))
        v['ruling'] = sum('ruling' in d for d in dl) / max(1, len(T))
        su = t.get('surfs', set())
        v['edge'] = float(bool(su & {'top', 'bottom', 'left', 'right', 'edge'}))
        v['multicol'] = float(t.get('ncol', 1) > 1)
        v['rev_share'] = np.mean([l['surf'] == 'reverse' for l in T]) if T else np.nan
        v['rev_used'] = float(any(l['surf'] == 'reverse' for l in T))
        # non-numeric line sandwiched between numeric lines: wrap / continuation habit
        flags = [bool(l['nums']) for l in T]
        mids = [i for i in range(1, len(flags) - 1) if not flags[i] and flags[i - 1] and flags[i + 1]]
        v['nonnum_mid'] = len(mids) / max(1, len(T))
        v['first_numeric'] = float(flags[0]) if flags else np.nan
        v['last_numeric'] = float(flags[-1]) if flags else np.nan
        v['mark'] = sum(l['mark'] for l in T) / max(1, sum(len(l['toks']) for l in T))
        v['toks_per_line'] = np.mean([len(l['toks']) for l in T]) if T else np.nan
        v['units_per_num'] = np.mean([len(l['nums']) for l in num]) if num else np.nan
        v['sic'] = t.get('sic', 0) / max(1, len(T))
        # variant preferences
        cnt = collections.defaultdict(lambda: [0, 0])
        for l in T:
            for s in l['toks']:
                f = s2fam.get(s)
                if f:
                    cnt[f][0] += 1
                    cnt[f][1] += (s == dom[f])
            for w in l['words']:
                for f in w2fam.get(w, ()):
                    cnt[f][0] += 1
                    cnt[f][1] += (w == dom[f])
        for f in famkeys:
            n, d = cnt.get(f, (0, 0))
            v[f] = d / n if n >= 1 else np.nan
        rows.append([v[k] for k in names])
    return np.array(rows, dtype=float), names


def content_matrix(tabs, min_tab=10):
    """Bag of base signs (presence) + numeral unit set; used only for specialties."""
    df = collections.Counter()
    sets = []
    for t in tabs:
        s = set()
        for l in t['lines']:
            s.update('B:' + b for b in l['base'])
            s.update('U:' + c for _, c in l['nums'])
        sets.append(s)
        df.update(s)
    vocab = sorted(k for k, n in df.items() if n >= min_tab and n <= 0.9 * len(tabs))
    vi = {k: i for i, k in enumerate(vocab)}
    X = np.zeros((len(tabs), len(vocab)), dtype=np.float32)
    for i, s in enumerate(sets):
        for k in s:
            if k in vi:
                X[i, vi[k]] = 1
    return X, vocab


def coarse_covariates(tabs):
    """Length and dominant unit -- residualised out of habits; strata for content-matched nulls."""
    ln = np.array([np.log1p(sum(1 for l in t['lines'] if l['kind'] == 'text')) for t in tabs])
    dom = []
    for t in tabs:
        c = collections.Counter(u for l in t['lines'] for _, u in l['nums'])
        dom.append(c.most_common(1)[0][0] if c else 'none')
    du = collections.Counter(dom)
    dom = [d if du[d] >= 15 else 'rare' for d in dom]
    levels = sorted(set(dom))
    D = np.array([[d == lv for lv in levels] for d in dom], dtype=float)
    lb = np.digitize(ln, np.quantile(ln, [0.25, 0.5, 0.75]))
    strata = np.array([f'{d}|{b}' for d, b in zip(dom, lb)])
    return np.column_stack([ln, ln ** 2, D]), strata


def residualise(H, C):
    """Per feature: OLS on observed rows, standardised residuals, NaN kept."""
    R = np.full_like(H, np.nan)
    keep = []
    for j in range(H.shape[1]):
        m = ~np.isnan(H[:, j])
        if m.sum() < 20 or np.nanstd(H[m, j]) < 1e-9:
            continue
        X = np.column_stack([np.ones(m.sum()), C[m]])
        b, *_ = np.linalg.lstsq(X, H[m, j], rcond=None)
        r = H[m, j] - X @ b
        sd = r.std()
        if sd < 1e-9:
            continue
        R[m, j] = r / sd
        keep.append(j)
    return R[:, keep], keep


# ---------------------------------------------------------- clustering ---
def masked_dist(X, cols, min_co=3):
    Z = X[:, cols]
    M = (~np.isnan(Z)).astype(np.float64)
    Z0 = np.nan_to_num(Z)
    Z2 = Z0 ** 2
    C = M @ M.T
    D = Z2 @ M.T + M @ Z2.T - 2 * Z0 @ Z0.T
    with np.errstate(invalid='ignore', divide='ignore'):
        D = D / C
    D[C < min_co] = np.nan
    med = np.nanmedian(D)
    D = np.where(np.isnan(D), med, D)
    np.fill_diagonal(D, 0)
    return np.sqrt(np.maximum(D, 0))


def embed(D, dim=8):
    n = D.shape[0]
    J = np.eye(n) - 1.0 / n
    B = -0.5 * J @ (D ** 2) @ J
    w, V = np.linalg.eigh(B)
    idx = np.argsort(w)[::-1][:dim]
    w = np.maximum(w[idx], 0)
    return V[:, idx] * np.sqrt(w)


def consensus(R, n_runs, rng, k_range=(6, 16), frac_feat=0.5, frac_tab=0.8, dim=8):
    from sklearn.cluster import KMeans
    n, p = R.shape
    A = np.zeros((n, n), dtype=np.float32)
    N = np.zeros((n, n), dtype=np.float32)
    for r in range(n_runs):
        cols = rng.choice(p, max(3, int(frac_feat * p)), replace=False)
        rows = np.sort(rng.choice(n, int(frac_tab * n), replace=False))
        D = masked_dist(R[rows], cols)
        E = embed(D, dim)
        k = int(rng.integers(k_range[0], k_range[1] + 1))
        lab = KMeans(k, n_init=2, random_state=int(rng.integers(1 << 30))).fit_predict(E)
        same = (lab[:, None] == lab[None, :]).astype(np.float32)
        A[np.ix_(rows, rows)] += same
        N[np.ix_(rows, rows)] += 1
    return A / np.maximum(N, 1)


def cut(A, k):
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import squareform
    D = 1 - A
    np.fill_diagonal(D, 0)
    Z = linkage(squareform(D, checks=False), 'average')
    return fcluster(Z, k, 'maxclust') - 1


def spectral_cut(A, k, seed=0):
    from sklearn.cluster import SpectralClustering
    S = A.copy()
    np.fill_diagonal(S, 1)
    return SpectralClustering(k, affinity='precomputed', random_state=seed).fit_predict(S)


def stability(A, lab):
    same = lab[:, None] == lab[None, :]
    off = ~np.eye(len(lab), dtype=bool)
    return float(A[same & off].mean() - A[~same].mean())


def nmi(a, b):
    from sklearn.metrics import normalized_mutual_info_score
    return float(normalized_mutual_info_score(a, b))


def specialty(Xc, lab):
    """Content separation: mean over content features of MI(feature; hand), in bits."""
    lab = np.asarray(lab)
    ks = np.unique(lab)
    n = len(lab)
    tot = 0.0
    p1 = Xc.mean(0)
    Hf = -(p1 * np.log2(np.clip(p1, 1e-12, 1)) + (1 - p1) * np.log2(np.clip(1 - p1, 1e-12, 1)))
    Hc = np.zeros(Xc.shape[1])
    for k in ks:
        m = lab == k
        w = m.sum() / n
        q = Xc[m].mean(0)
        Hc += w * -(q * np.log2(np.clip(q, 1e-12, 1)) + (1 - q) * np.log2(np.clip(1 - q, 1e-12, 1)))
    return float((Hf - Hc).mean()), Hf - Hc


def strata_perm(lab, strata, rng):
    out = np.array(lab).copy()
    for s in np.unique(strata):
        idx = np.where(strata == s)[0]
        out[idx] = out[rng.permutation(idx)]
    return out


def shuffle_habits(R, rng):
    """Each habit column permuted across tablets (keeps marginals and missingness per column)."""
    S = R.copy()
    for j in range(S.shape[1]):
        S[:, j] = S[rng.permutation(S.shape[0]), j]
    return S
