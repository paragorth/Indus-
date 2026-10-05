#!/usr/bin/env python3
"""LA-53 shared code: CLAY IS ATTENTION, AND ATTENTION IS STATUS.

A clerk spends clay unevenly. For every word occurrence inside an entry we measure how much layout
attention it receives, then ask how much of that is NOT predicted by the entry's quantity, commodity,
document size and the word's own length. Word types are ranked by their surplus.

Attention features (per word occurrence; opaque word ids only, no sound values):
  alone   the word is the only word on its physical line
  lines   physical lines spanned by its entry
  head    its entry is the first entry of the document
  qual    number of other words in the same entry (extra qualifying words)
  early   1 - entry index / (entries - 1)
  bulk    tokens of its entry / median tokens per entry in the document
  nsig    signs in the word (determinatives stripped in Ur III)   [attention in some specs, covariate in others]
Covariates: log(1+quantity), quantity missing, commodity one-hot, log entries in document, site one-hot,
  nsig (when not an attention feature).
Random baseline models: each spec draws a random subset of attention features with Dirichlet weights, a
random covariate set and a shrinkage constant; surplus = residual of the weighted attention on covariates,
averaged per type with shrinkage, standardized. Consensus = mean over specs.
Corpora: Linear A (lineara.xyz), Linear B (DAMOS, line layout kept), Ur III (CDLI ATF lines, from the
proto-elamite pe38 checkpoint). Status truth lists for LB and Ur III are used for calibration only.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1'); os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import json, re, sys, math, random, hashlib, collections, unicodedata
from fractions import Fraction as Fr
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, '..', 'data')
CK = os.path.join(D, 'la53_ckpt')
os.makedirs(CK, exist_ok=True)
UR3 = os.path.join(HERE, '..', '..', 'proto-elamite', 'data', 'pe38_ckpt', 'ur3_docs.json')
FEATS = ['alone', 'lines', 'head', 'qual', 'early', 'bulk', 'nsig']


def seed(name):
    return int(hashlib.sha256(name.encode()).hexdigest()[:8], 16)

# doc = {'id','site','support','lines': [[tok,...],...]}; tok = ('W', type, nsig) | ('L', logo) | ('N', value)

ADMIN = {'Tablet', 'Lames (short thin tablet)', '3-sided bar', '4-sided bar', 'Label'}


def la_docs():
    C = json.load(open(os.path.join(D, 'corpus.json')))
    out = []
    for ins in C:
        lines = [[]]
        for t in ins['tokens']:
            if t['t'] == 'nl':
                lines.append([])
            elif t['t'] == 'word':
                lines[-1].append(('W', '-'.join(t['s']), len(t['s'])))
            elif t['t'] == 'logo':
                lines[-1].append(('L', t['v']))
            elif t['t'] == 'num':
                lines[-1].append(('N', float(t['v']) + (0.5 if t['frac'] else 0.0)))
        lines = [l for l in lines if l]
        if lines:
            out.append({'id': ins['id'], 'site': ins['site'] or '?', 'support': ins['support'], 'lines': lines})
    return out


# ------------------------------------------------------------------ Linear B (DAMOS, lines kept)
WORD = re.compile(r'^[a-z0-9*]+(-[a-z0-9*]+)*$')
MEAS = set('TVZSMNPQ')
LIQ = {'OLE', 'VIN', 'ME+RI', 'ME±RI'}
DRYU = {'T': 0.1, 'V': 1 / 60, 'Z': 1 / 240}
LIQU = {'S': 1 / 3, 'V': 1 / 18, 'Z': 1 / 72}


def _strip(t):
    return ''.join(ch for ch in unicodedata.normalize('NFD', t) if unicodedata.category(ch) != 'Mn')


def lb_docs():
    fn = os.path.join(CK, 'lb_docs.json')
    if os.path.exists(fn):
        return _tup(json.load(open(fn)))
    out = []
    for raw in open(os.path.join(D, 'damos_items.jsonl')):
        d = json.loads(raw)
        h = d.get('heading') or ''
        m = re.match(r'^([A-Z]{2,3})\s+([A-Z][a-z]?\d?)', h)
        if not m: continue
        site = m.group(1)
        lines = []
        for ln in (d.get('content') or '').split('\n'):
            toks = _strip(ln).split()
            L = []; meas = None; logo = None
            for t in toks:
                s = re.sub(r'[\[\]⟦⟧?!,⌞⌟\'"]', '', t)
                if not s or s.startswith('.') or s in ('/', 'vac', 'vac.', 'v.', 'r.', 'lat.', 'inf.', 'sup.'): continue
                if s.startswith('v.') or s.startswith('lat') or s in ('↓', '→'): continue
                if s in MEAS and L:
                    meas = s; continue
                if re.fullmatch(r'\d+', s):
                    v = int(s)
                    if meas:
                        tab = LIQU if logo in LIQ else DRYU
                        v = v * tab.get(meas, 1.0)
                    if L and L[-1][0] == 'N' and meas:
                        L[-1] = ('N', L[-1][1] + v)
                    else:
                        L.append(('N', float(v)))
                    meas = None; continue
                if WORD.match(s) and any(c.isalpha() for c in s) and s == s.lower():
                    L.append(('W', s, len(s.split('-')))); meas = None; continue
                b = s.split('+')[0]
                if (b.isupper() and len(b) >= 2) or re.fullmatch(r'\*\d+[A-Z]*', b):
                    logo = b; L.append(('L', b)); meas = None; continue
            if L:
                lines.append(L)
        if lines:
            out.append({'id': h, 'site': site, 'support': m.group(2), 'lines': lines})
    json.dump(out, open(fn, 'w'))
    return out


def _tup(docs):
    for d in docs:
        d['lines'] = [[tuple(t) for t in l] for l in d['lines']]
    return docs


# ------------------------------------------------------------------ Ur III (CDLI lines)
def ur3_docs(n=6000):
    fn = os.path.join(CK, 'ur3_docs_%d.json' % n)
    if os.path.exists(fn):
        return _tup(json.load(open(fn)))
    src = json.load(open(UR3))
    rng = random.Random(seed('la53-ur3'))
    rng.shuffle(src)
    out = []
    for d in src:
        if len(out) >= n: break
        lines = []
        for l in d['lines']:
            L = []
            if l['val'] is not None and l['sys'] in (1, 2):
                try:
                    L.append(('N', float(Fr(l['val']))))
                except Exception:
                    pass
            for t in l['toks']:
                if not t: continue
                core = re.sub(r'\{[^}]*\}', '', t)
                ns = len([x for x in re.split(r'[-.]', core) if x])
                L.append(('W', t, max(1, ns)))
            if L: lines.append(L)
        # keep documents with at least 3 quantity lines (lists)
        if sum(1 for l in lines if l and l[0][0] == 'N') >= 2:
            out.append({'id': d['id'], 'site': d.get('site') or '?', 'support': 'tablet', 'lines': lines})
    json.dump(out, open(fn, 'w'))
    return out


# ------------------------------------------------------------------ status truth (controls only)
LB_STATUS = set('''po-ti-ni-ja a-ta-na-po-ti-ni-ja da-pu2-ri-to-jo da-pu2-ri-to-jo-po-ti-ni-ja u-po-jo-po-ti-ni-ja
si-to-po-ti-ni-ja po-ti-ni-ja-we-jo po-ti-ni-ja-we-ja di-wo di-we di-wi-jo di-wi-ja di-wi-je-we di-u-ja di-u-jo e-ra
po-si-da-o po-se-da-o-ne po-si-da-e-ja po-si-da-i-jo pa-si-te-o-i wa-na-ka wa-na-ka-te wa-na-ka-te-ro wa-na-ka-te-ra
wa-na-se-wi-ja wa-na-se-wi-jo ra-wa-ke-ta ra-wa-ke-si-jo ra-wa-ke-ja e-ri-nu e-ri-nu-we a-ne-mo i-je-re-ja i-je-re-u
i-je-ro i-je-ra e-nu-wa-ri-jo pa-ja-wo-ne ma-na-sa ti-ri-se-ro-e do-po-ta pe-re-*82 i-qe-ja e-ma-a2 a-re a-re-ja
ma-ri-ne-u ma-ri-ne-we qe-ra-si-ja pa-de pi-pi-tu-na ko-ma-we-te-ja di-ri-mi-jo te-o te-o-jo te-o-i ka-ra-wi-po-ro
ka-ra-wi-po-ro-jo da-da-re-jo-de da-da-re-jo pa-ki-ja-na pa-ki-ja-ni-ja pa-ki-ja-ne e-ra-wo po-ti-ni-ja-wi-jo
a-ti-mi-te a-ti-mi-to me-tu-wo-ne-wo ma-te-re-te-i ma-ka di-pi-si-jo di-pi-si-jo-i i-pe-me-de-ja dı-ka-ta-jo
di-ka-ta-jo a-ka-wi-ja-de e-re-u-ti-ja e-re-u-ti-ja wa-na-ka-te-ro ko-re-te ko-re-te-re po-ro-ko-re-te
po-ro-ko-re-te-re qa-si-re-u qa-si-re-wi-ja e-qe-ta e-qe-ta-e mo-ri-wo-do e-re-ta-o'''.split())
# note: ko-re-te / qa-si-re-u / e-qe-ta are officials; kept as high status.

UR_GODS = set('''en-lil2 nin-lil2 nanna suen utu inanna nin-urta nusku en-ki nin-hur-sag szara2 nin-gir2-su ba-ba6 ba-u2
dumu-zi gu-la nansze iszkur nergal nin-sun2 lugal-banda3 bil3-ga-mes nin-gal nin-tin-ug5-ga an al-la-tum
be-la-at-suh-ner be-la-at-dar-ra-ba-an an-nu-ni-tum ul-masz-i-tum nin-a-zu nin-sikil-la szul-gi amar-{d}suen
szu-{d}suen i-bi2-{d}suen nin-mar-ki nin-sun nin-e2-gal ga2-tum3-dug3 nin-isin2-na da-gan nin-gublaga
nin-shubur nin-szubur dumu-zi-abzu nin-dar-a hendur-sag-ga mes-lam-ta-e3-a a-ba-ba6 dam-gal-nun-na nin-kasz
asar-lu2-hi nin-hursag nin-ti ezinu2 asznan lisi4 nin-ezen nin-mug za-ba4-ba4 nun-gal ma-mi-tum szu-zi-an-na
nin-tu nin-mu2 ha-ia3 nisaba nin-hur-sag-ga2'''.split())
UR_ELITE = set('''lugal nin ensi2 sukkal-mah szabra sanga dumu-lugal nin-dingir e2-gal e2-lugal lu2-mah en'''.split())
_SUF = ('-ra', '-sze3', '-ke4', '-ka', '-ta', '-a', '-la2', '-e', '-kam', '-ak', '-ka-sze3')


def _ur_core(w):
    c = w
    for _ in range(2):
        for s in sorted(_SUF, key=len, reverse=True):
            if c.endswith(s) and len(c) > len(s) + 1:
                c = c[:-len(s)]; break
    return c


def ur_status(w):
    for c in (w, _ur_core(w)):
        if c in UR_ELITE: return True
        if c.startswith('{d}') and c[3:] in UR_GODS: return True
        if c in ('amar-{d}suen', 'szu-{d}suen', 'i-bi2-{d}suen'): return True
        if c.startswith('e2-{d}') and _ur_core(c[6:]) in UR_GODS: return True
    return False


def lb_status(w):
    return w in LB_STATUS


# ------------------------------------------------------------------ entries and occurrences
def entries_wordfirst(doc):
    """LA / LB: an entry is a run of words (+logograms) closed by number(s); a word after a number starts a
    new entry. Returns list of entries: {'words': [(type,nsig,lineidx)], 'qty', 'logo', 'lines': set}."""
    ents = []; cur = None; lastlogo = None
    for li, line in enumerate(doc['lines']):
        for t in line:
            if t[0] == 'W':
                if cur is None or cur['closed']:
                    cur = {'words': [], 'qty': None, 'logo': None, 'lines': set(), 'closed': False, 'ntok': 0}
                    ents.append(cur)
                cur['words'].append((t[1], t[2], li)); cur['lines'].add(li); cur['ntok'] += 1
            elif t[0] == 'L':
                lastlogo = t[1]
                if cur is None or cur['closed']:
                    cur = {'words': [], 'qty': None, 'logo': None, 'lines': set(), 'closed': False, 'ntok': 0}
                    ents.append(cur)
                cur['logo'] = t[1]; cur['lines'].add(li); cur['ntok'] += 1
            elif t[0] == 'N':
                if cur is None:
                    cur = {'words': [], 'qty': None, 'logo': None, 'lines': set(), 'closed': False, 'ntok': 0}
                    ents.append(cur)
                cur['qty'] = (cur['qty'] or 0.0) + t[1]; cur['closed'] = True; cur['lines'].add(li); cur['ntok'] += 1
                if cur['logo'] is None: cur['logo'] = lastlogo
    return [e for e in ents if e['words']]


def entries_qtyfirst(doc):
    """Ur III: an entry is a quantity line plus the following word-only lines."""
    ents = []; cur = None
    for li, line in enumerate(doc['lines']):
        if line[0][0] == 'N':
            ws = [t for t in line[1:] if t[0] == 'W']
            com = ws[0][1] if ws else None
            cur = {'words': [(t[1], t[2], li) for t in ws[1:]], 'qty': line[0][1], 'logo': com,
                   'lines': {li}, 'ntok': len(line)}
            ents.append(cur)
        elif cur is not None:
            if line[0][0] == 'W' and line[0][1] in ('iti', 'mu', 'u4', 'szu-nigin2', 'szunigin', 'szu-nigin'):
                cur = None; continue
            cur['words'] += [(t[1], t[2], li) for t in line if t[0] == 'W']
            cur['lines'].add(li); cur['ntok'] += len(line)
    return [e for e in ents if e['words']]


def occurrences(docs, mode, min_entries=2):
    """One row per word occurrence in an entry of a document with >= min_entries entries."""
    rows = []
    for di, d in enumerate(docs):
        ents = entries_qtyfirst(d) if mode == 'qty' else entries_wordfirst(d)
        if len(ents) < min_entries: continue
        n = len(ents)
        med = float(np.median([e['ntok'] for e in ents])) or 1.0
        wcount = collections.Counter()
        for li, line in enumerate(d['lines']):
            wcount[li] = sum(1 for t in line if t[0] == 'W')
        # entries starting per line
        for k, e in enumerate(ents):
            for (w, ns, li) in e['words']:
                rows.append({'doc': di, 'type': w, 'nsig': ns, 'alone': float(wcount[li] == 1),
                             'lines': float(len(e['lines'])), 'head': float(k == 0),
                             'qual': float(len(e['words']) - 1), 'early': 1.0 - k / max(1, n - 1),
                             'bulk': e['ntok'] / med, 'qty': e['qty'], 'logo': e['logo'] or 'NONE',
                             'nent': n, 'site': d['site'], 'ent': k})
    return rows


# ------------------------------------------------------------------ matrices
def build(rows, topc=12, tops=6):
    A = np.array([[r[f] for f in FEATS] for r in rows], float)
    A = (A - A.mean(0)) / (A.std(0) + 1e-9)
    q = np.array([math.log1p(r['qty']) if r['qty'] is not None else 0.0 for r in rows])
    qm = np.array([float(r['qty'] is None) for r in rows])
    cc = collections.Counter(r['logo'] for r in rows)
    coms = [c for c, _ in cc.most_common(topc)]
    C = np.array([[float(r['logo'] == c) for c in coms] for r in rows])
    sc = collections.Counter(r['site'] for r in rows)
    sites = [s for s, _ in sc.most_common(tops)]
    S = np.array([[float(r['site'] == s) for s in sites] for r in rows])
    ne = np.array([math.log(r['nent']) for r in rows])
    types = sorted(set(r['type'] for r in rows))
    tix = {t: i for i, t in enumerate(types)}
    ti = np.array([tix[r['type']] for r in rows])
    strata = np.array([hash((r['logo'], min(4, int(math.log2(r['nent']))))) for r in rows])
    return {'A': A, 'q': q, 'qm': qm, 'C': C, 'S': S, 'ne': ne, 'ti': ti, 'types': types, 'strata': strata,
            'nsig': A[:, FEATS.index('nsig')], 'rows': rows,
            'doc': np.unique(np.array([r['doc'] for r in rows]), return_inverse=True)[1]}


def random_specs(n, rng):
    specs = []
    for _ in range(n):
        k = rng.integers(2, len(FEATS) + 1)
        fs = rng.choice(len(FEATS), size=k, replace=False)
        w = np.zeros(len(FEATS)); w[fs] = rng.dirichlet(np.ones(k))
        cov = {'q': rng.random() < 0.9, 'q2': rng.random() < 0.5, 'C': rng.random() < 0.8, 'S': rng.random() < 0.5,
               'ne': rng.random() < 0.7, 'nsig': (w[FEATS.index('nsig')] == 0) and rng.random() < 0.7}
        specs.append({'w': w, 'cov': cov, 'k': float(rng.choice([1.0, 2.0, 4.0]))})
    return specs


def design(B, cov):
    cols = [np.ones(len(B['q']))]
    if cov['q']: cols += [B['q'], B['qm']]
    if cov['q2']: cols.append(B['q'] ** 2)
    if cov['C'] and B['C'].shape[1]: cols += list(B['C'].T)
    if cov['S'] and B['S'].shape[1]: cols += list(B['S'].T)
    if cov['ne']: cols.append(B['ne'])
    if cov['nsig']: cols.append(B['nsig'])
    return np.column_stack(cols)


def demean(M, doc):
    M = np.asarray(M, float)
    if M.ndim == 1: M = M[:, None]
    n = np.bincount(doc).astype(float)
    S = np.zeros((len(n), M.shape[1])); np.add.at(S, doc, M)
    return M - (S / np.maximum(n, 1)[:, None])[doc]


def consensus_fast(B, specs, A=None, q=None, docfe=False):
    A = B['A'] if A is None else A
    if docfe: A = demean(A, B['doc'])
    Bq = B if q is None else dict(B, q=q[0], qm=q[1])
    nt = len(B['types']); cnt = np.bincount(B['ti'], minlength=nt).astype(float)
    ok = cnt >= 2
    groups = collections.defaultdict(list)
    for s in specs:
        groups[tuple(sorted(s['cov'].items()))].append(s)
    tot = np.zeros(nt)
    # type-summing matrix applied once per group: sums of residual features per type
    for key, ss in groups.items():
        X = design(Bq, dict(key))
        if docfe: X = demean(X, B['doc'])
        U, sv, _ = np.linalg.svd(X, full_matrices=False)
        Q = U[:, sv > 1e-8 * sv[0]]
        RA = A - Q @ (Q.T @ A)                       # residual of every attention feature
        T = np.zeros((nt, A.shape[1]))
        np.add.at(T, B['ti'], RA)                    # per-type residual sums, per feature
        W = np.column_stack([s['w'] for s in ss])
        K = np.array([s['k'] for s in ss])
        SUR = (T @ W) / (cnt[:, None] + K[None, :])
        SUR /= SUR[ok].std(0)[None, :] + 1e-9
        tot += SUR.sum(1)
    return tot / len(specs)


def perm_within(strata, rng):
    idx = np.arange(len(strata))
    out = idx.copy()
    order = np.argsort(strata, kind='stable')
    s = strata[order]
    cuts = np.flatnonzero(np.diff(s)) + 1
    for g in np.split(order, cuts):
        out[g] = rng.permutation(g)
    return out


def auc(scores, labels):
    scores = np.asarray(scores); labels = np.asarray(labels, bool)
    if labels.sum() == 0 or (~labels).sum() == 0: return float('nan')
    from scipy.stats import rankdata
    r = rankdata(scores)
    n1 = labels.sum(); n0 = (~labels).sum()
    return float((r[labels].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
