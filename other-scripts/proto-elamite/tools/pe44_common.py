"""pe44 WHICH NUMBERS ARE GUESSES?  Hidden-plan deconvolution of every quantity.

pe23 already typed numbers by digit fingerprints of whole GROUPS (roundness, lowest
denomination, mod 5, leading digit, same-tablet repeats).  pe44 changes the frame:
every single quantity is read as   value = hidden anchor (plan) + deviation (reality),
and is described by RELATIONAL fingerprints that pe23 did not use:
  SHORT / OVER   just below / just above a round anchor of its own system (a quota missed / exceeded)
  XREP           the same (item, value) recurs on OTHER tablets (a norm or standard allotment)
  RATIO          an exact simple ratio (2:1, 3:2, 4:3 ...) to an adjacent entry (scaled allocation)
  TREP, ONE, LOW, MAG  (same-tablet repeat, one denomination, lowest denomination, size vs tablet median)
Random latent-class mixtures (thousands) sort quantities into kinds; Ur III lines of
known type and planted plan/actual splits are the controls.

Data:   PE entries via pe23_common.pe_entries (grade-B value set).
        Ur III: every numeric line of every Ur III tablet in CDLI bulk ATF (scratchpad),
        in tablet order, with known-type labels where the wording fixes them:
          LIVESTOCK, PEOPLE (actual counts), GRAIN (measured), RATION, TARGET (norms),
          ESTIMATE (computed harvest projection), DEFICIT ('la2-ia3' line of a
          balanced account), DEBIT (line carrying 'sag-nig2-gur11').
        Tablet flag BAL = balanced account (has both sag-nig2-gur11 and la2-ia3).
"""
import gzip, json, math, os, re, sys, csv
from collections import Counter, defaultdict
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, '..', 'data')
CK = os.path.join(DATA, 'pe44_ckpt')
LOOPS = os.path.join(HERE, '..', 'loops')
os.makedirs(CK, exist_ok=True)
sys.path.insert(0, HERE)
import pe23_common as P23  # noqa: E402

SCRATCH = P23.SCRATCH
DEN = {'PE_CNT': [1, 10, 100, 300], 'PE_CAP': [1, 2, 4, 12, 24, 120, 720],
       'UR_CNT': P23.UR_CNT_DEN, 'UR_CAP': P23.UR_CAP_DEN}
FEATS = ['ONE', 'NLEV', 'LOW', 'SHORT', 'OVER', 'TREP', 'XREP', 'RATIO', 'MAG', 'LMAG']
BINF = {'ONE', 'SHORT', 'OVER', 'TREP', 'RATIO'}


# ------------------------------------------------------------------ data
def pe_tabs():
    """PE tablets -> ordered list of quantity records (CNT and CAP only)."""
    tabs = defaultdict(list)
    for e in P23.pe_entries('B'):
        if e['sys'] not in ('CNT', 'CAP'):
            continue
        tabs[e['tab']].append({'sys': 'PE_' + e['sys'], 'val': int(e['val']), 'item': e['final'] or '-',
                               'signs': e['signs'], 'hdr': e['hdr'], 'surface': e['surface'],
                               'nsign': e['nsign'], 'line': e['line'], 'lab': None, 'raw': e['raw']})
    return dict(tabs)


SKIPW = {'gur', 'sila3', 'la2', 'sze', 'gin2', 'ma-na'}


def _item(toks, j):
    while j < len(toks) and (P23.NUMTOK.match(toks[j]) or toks[j] in SKIPW):
        j += 1
    return toks[j] if j < len(toks) else '-'


def ur3_tabs(rebuild=False):
    fn = os.path.join(CK, 'ur3_tabs.json.gz')
    if os.path.exists(fn) and not rebuild:
        return json.load(gzip.open(fn, 'rt'))
    csv.field_size_limit(10 ** 9)
    keep = {}
    for row in csv.DictReader(open(os.path.join(SCRATCH, 'cdli_cat.csv'), encoding='utf-8')):
        if row['period'].startswith('Ur III'):
            keep['P%06d' % int(row['id_text'])] = row['provenience'].split(' (')[0]
    texts = defaultdict(list)
    cur = None
    for raw in open(os.path.join(SCRATCH, 'cdli.atf'), encoding='utf-8', errors='replace'):
        if raw.startswith('&P'):
            pid = raw[1:8]
            cur = pid if pid in keep else None
            continue
        if cur and raw[:1].isdigit():
            m = re.match(r"^\d+'?\.\s+(.*)$", raw.rstrip())
            if m:
                texts[cur].append(m.group(1))
    out = {}
    for pid, lines in texts.items():
        prov = keep[pid]
        has_gan = any('GAN2' in l for l in lines)
        bal = any('sag-nig2-gur11' in l for l in lines) and any('la2-ia3' in l for l in lines)
        recs = []
        # debit block: numeric lines before the first 'sag-nig2-gur11-ra-kam' line of a balanced account
        dend = next((k for k, l in enumerate(lines) if 'sag-nig2-gur11' in l), -1) if bal else -1
        for li, body in enumerate(lines):
            if '...' in body or ' x ' in f' {body} ':
                continue
            toks = P23._clean(body)
            if not toks or any(t == 'n' or t.startswith('n(') for t in toks):
                continue
            lab = None
            pre = None
            if toks[0] in ('szu-nigin2', 'szunigin', 'szu-nigin', 'szunigin2'):
                pre = 'TOTAL'; toks = toks[1:]
            elif toks[0] == 'la2-ia3':
                pre = 'DEFICIT'; toks = toks[1:]
            elif toks[0] == 'sze-bi' and has_gan:
                pre = 'ESTIMATE'; toks = toks[1:]
            if not toks or not P23.NUMTOK.match(toks[0]):
                continue
            val, sysn = None, None
            capish = any(t == 'gur' or t.startswith('gur-') or t.startswith('sila3') for t in toks) or \
                re.search(r'\((barig|ban2)\)', body)
            if capish:
                val = P23.ur_cap(toks); sysn = 'UR_CAP'
            else:
                v, j = P23.ur_int(toks, 0)
                val = v; sysn = 'UR_CNT'
            if not val or val <= 0 or abs(val - round(val)) > 1e-9:
                continue
            val = int(round(val))
            # find item token
            j = 0
            while j < len(toks) and (P23.NUMTOK.match(toks[j]) or toks[j] in SKIPW):
                j += 1
            item = toks[j] if j < len(toks) else '-'
            if pre == 'DEFICIT':
                lab = 'DEFICIT'
            elif pre == 'TOTAL':
                lab = 'TOTAL'
            elif li <= dend:
                lab = 'DEBIT'
            elif pre == 'ESTIMATE':
                lab = 'ESTIMATE'
            elif sysn == 'UR_CAP' and 'GAN2' in toks and re.search(r'gur-ta\b', body):
                lab = 'TARGET'
            elif sysn == 'UR_CAP' and re.search(r'-ta\b', body) and not has_gan:
                lab = 'RATION'
            elif sysn == 'UR_CNT' and '-ta' not in body and not has_gan:
                if item in P23.ANIMALS and prov.startswith('Puzri'):
                    lab = 'LIVESTOCK'
                elif item in P23.PEOPLE:
                    lab = 'PEOPLE'
            elif sysn == 'UR_CAP' and 'sze' in toks and 'gur' in toks and not has_gan and '-ta' not in body:
                lab = 'GRAIN'
            recs.append({'sys': sysn, 'val': val, 'item': item, 'lab': lab})
        if len(recs) >= 2:
            out[pid] = {'prov': prov, 'bal': bal, 'recs': recs}
    json.dump(out, gzip.open(fn, 'wt'))
    return out


# ------------------------------------------------------------------ features
def canon(v, den):
    rem = v
    cnt = [0] * len(den)
    for k in range(len(den) - 1, -1, -1):
        cnt[k] = rem // den[k]
        rem %= den[k]
    return cnt


def value_feats(v, den):
    """Features of one value in its own notation."""
    cnt = canon(v, den)
    nz = [k for k, c in enumerate(cnt) if c]
    hi, lo = nz[-1], nz[0]
    one = 1 if (len(nz) == 1 and v >= den[1]) else 0
    # anchors: multiples of the leading denomination and of the next one up
    if hi == 0:
        return one, len(nz), lo, 0, 0
    tol = max(1, 0.2 * den[hi])
    grids = [den[hi]] + ([den[hi + 1]] if hi + 1 < len(den) else [])
    short = 1 if any(0 < (-v) % g <= tol for g in grids) and v % den[hi] != 0 else 0
    over = 1 if (0 < v % den[hi] <= tol and (-v) % den[hi] > tol) else 0
    return one, len(nz), lo, short, over


SIMPLE = sorted({p / q for p in range(1, 7) for q in range(1, 7) if p != q})


def tab_feats(recs, xcount):
    """recs: list of dicts with sys, val, item. xcount[(sys,item,val)] = #tablets carrying it."""
    out = []
    vals = [r['val'] for r in recs]
    med = defaultdict(list)
    for r in recs:
        med[r['sys']].append(math.log(r['val']))
    medv = {k: float(np.median(v)) for k, v in med.items()}
    tc = Counter((r['sys'], r['val']) for r in recs)
    for i, r in enumerate(recs):
        den = DEN[r['sys']]
        one, nlev, lo, short, over = value_feats(r['val'], den)
        ratio = 0
        for j in (i - 1, i + 1):
            if 0 <= j < len(recs) and recs[j]['sys'] == r['sys']:
                q = r['val'] / recs[j]['val']
                if any(abs(q - s) < 1e-9 for s in SIMPLE) and r['val'] >= 2 and recs[j]['val'] >= 2:
                    ratio = 1
        xr = xcount.get((r['sys'], r['item'], r['val']), 1) - 1
        out.append([one, nlev, lo, short, over, 1 if tc[(r['sys'], r['val'])] > 1 else 0,
                    math.log1p(max(0, xr)), ratio, math.log(r['val']) - medv[r['sys']], math.log(r['val'])])
    return out


def corpus_matrix(tabs, recs_key=None):
    """tabs: dict tab -> list of recs. Returns X (n x F), meta list (tab, idx)."""
    xc = Counter()
    for t, recs in tabs.items():
        for k in {(r['sys'], r['item'], r['val']) for r in recs}:
            xc[k] += 1
    X, meta = [], []
    for t, recs in tabs.items():
        for i, f in enumerate(tab_feats(recs, xc)):
            X.append(f); meta.append((t, i))
    return np.array(X, float), meta


# ------------------------------------------------------------------ latent class mixture
def fit_mix(X, cols, K, rng, iters=60, binf=None):
    """Mixed Bernoulli (binary cols) / Gaussian (others) latent class model by EM."""
    binf = binf if binf is not None else [FEATS[c] in BINF for c in cols]
    Z = X[:, cols]
    n = len(Z)
    R = rng.dirichlet(np.ones(K), n)
    for _ in range(iters):
        w = R.sum(0) + 1e-9
        pi = w / n
        mu = (R.T @ Z) / w[:, None]
        var = (R.T @ (Z ** 2)) / w[:, None] - mu ** 2
        var = np.maximum(var, 0.02)
        mu_b = np.clip(mu, 0.01, 0.99)
        L = logp(Z, pi, mu, var, mu_b, binf)
        m = L.max(1, keepdims=True)
        R = np.exp(L - m); R /= R.sum(1, keepdims=True)
    return {'cols': cols, 'pi': pi, 'mu': mu, 'var': var, 'mu_b': mu_b, 'binf': binf, 'K': K}


def logp(Z, pi, mu, var, mu_b, binf):
    K = len(pi)
    L = np.tile(np.log(pi + 1e-12), (len(Z), 1))
    for c, isb in enumerate(binf):
        z = Z[:, c][:, None]
        if isb:
            L += z * np.log(mu_b[:, c]) + (1 - z) * np.log(1 - mu_b[:, c])
        else:
            L += -0.5 * ((z - mu[:, c]) ** 2 / var[:, c] + np.log(2 * np.pi * var[:, c]))
    return L


def posterior(model, X):
    Z = X[:, model['cols']]
    L = logp(Z, model['pi'], model['mu'], model['var'], model['mu_b'], model['binf'])
    m = L.max(1, keepdims=True)
    ll = (m[:, 0] + np.log(np.exp(L - m).sum(1)))
    R = np.exp(L - m); R /= R.sum(1, keepdims=True)
    return R, ll


def plan_score(model):
    """'Plannedness' of each class from its profile: ONE + TREP + XREP + RATIO - SHORT - OVER - NLEV."""
    cols = model['cols']
    s = np.zeros(model['K'])
    sign = {'ONE': 1, 'TREP': 1, 'XREP': 1, 'RATIO': 1, 'SHORT': -1, 'OVER': -1, 'NLEV': -1, 'LOW': 1}
    for j, c in enumerate(cols):
        f = FEATS[c]
        if f in sign:
            s += sign[f] * (model['mu'][:, j] - model['mu'][:, j].mean()) / (model['mu'][:, j].std() + 1e-9)
    return s


def ami(a, b):
    """Normalised mutual information (arithmetic) between two label arrays."""
    a = np.asarray(a); b = np.asarray(b)
    ca, cb = Counter(a.tolist()), Counter(b.tolist())
    cab = Counter(zip(a.tolist(), b.tolist()))
    n = len(a)
    mi = sum(c / n * math.log(c * n / (ca[x] * cb[y])) for (x, y), c in cab.items())
    ha = -sum(c / n * math.log(c / n) for c in ca.values())
    hb = -sum(c / n * math.log(c / n) for c in cb.values())
    return mi / max(1e-9, (ha + hb) / 2)


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


def write_rows(fn, rows):
    with open(os.path.join(LOOPS, fn), 'w') as f:
        for r in rows:
            f.write('| ' + ' | '.join(r) + ' |\n')
