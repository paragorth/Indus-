"""pe17 ('predict the clay before looking'): shared code.

Per-tablet fingerprint features in five families:
  SIGN  base-sign presence (variant suffix stripped)
  VAR   full variant-form presence (only forms carrying ~x) plus variant-use share
  NUM   numeral-sign type presence (number-system use)
  FMT   entry format (lines, entries with numbers, comma use, compounds, surfaces, header ...)
  BIG   within-line base-sign bigrams
Only content is used: damage marks are excluded (they record preservation, not the scribe).
"""
import json, os, re, math, hashlib
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, '..', 'data')
CKPT = os.path.join(DATA, 'pe17_ckpt')
os.makedirs(CKPT, exist_ok=True)

PLATEAU = {'Anšan (mod. Tell Malyan)': 'Malyan', 'uncertain (mod. Tepe Yahya)': 'Yahya',
           'uncertain (mod. Tepe Sialk)': 'Sialk', 'uncertain (mod. Tepe Sofalin)': 'Sofalin',
           'uncertain (mod. Ozbaki)': 'Ozbaki', 'uncertain (mod. Shahr-i Sokhta)': 'ShahriSokhta'}
DIST_KM = {'Susa': 0, 'Sialk': 353, 'Sofalin': 444, 'Malyan': 463, 'Ozbaki': 469, 'Yahya': 931,
           'ShahriSokhta': 1252}


def site_of(t):
    p = t['provenience']
    if p.startswith('Susa'):
        return 'Susa'
    return PLATEAU.get(p)


def base(s):
    return s.split('~')[0]


def atoms(sign):
    """split a compound |A+B| into atoms; keep the compound flag separately"""
    s = sign.strip('|')
    return [a for a in re.split(r'[+.x&]', s) if a]


def load():
    d = json.load(open(os.path.join(DATA, 'pe_corpus.json')))
    out = []
    for t in d:
        if t['object_type'] != 'tablet' or not t['lines']:
            continue
        st = site_of(t)
        if st is None:
            continue
        out.append(dict(id=t['id'], des=t['designation'], site=st, lines=t['lines']))
    return out


def tablet_tokens(t):
    toks = {'SIGN': set(), 'VAR': set(), 'NUM': set(), 'BIG': set()}
    nsig = nvar = ncomp = 0
    for l in t['lines']:
        bl = []
        for s in l['signs']:
            nsig += 1
            if s.startswith('|'):
                ncomp += 1
            for a in atoms(s):
                toks['SIGN'].add(base(a))
                if '~' in a:
                    toks['VAR'].add(a)
                    nvar += 1
            bl.append(base(s.strip('|')) if not s.startswith('|') else 'C:' + '+'.join(base(a) for a in atoms(s)))
        for a, b in zip(bl, bl[1:]):
            toks['BIG'].add(a + '>' + b)
        for n in l['numerals']:
            toks['NUM'].add(n[1])
    L = t['lines']
    nl = len(L)
    withnum = sum(1 for l in L if l['numerals'])
    fmt = [math.log1p(nl), withnum / nl, sum(l['has_comma'] for l in L) / nl,
           np.mean([len(l['signs']) for l in L]), ncomp / max(nsig, 1), nvar / max(nsig, 1),
           float(any(l['surface'] == 'reverse' for l in L)), float(any(l['surface'] == 'top' for l in L)),
           float(any(l['column'] > 1 for l in L)), float(any(l.get('header_comment') for l in L)),
           np.mean([len(l['numerals']) for l in L]), math.log1p(len(toks['NUM'])),
           math.log1p(sum((n[0] or 0) for l in L for n in l['numerals']))]
    return toks, np.array(fmt, float)


FMT_NAMES = ['log_lines', 'share_num_lines', 'share_comma', 'signs_per_line', 'compound_share',
             'variant_share', 'has_reverse', 'has_top', 'multi_column', 'has_header', 'numtypes_per_line',
             'log_num_types', 'log_numeral_mass']


def build(tabs, min_df=5):
    TT = [tablet_tokens(t) for t in tabs]
    names, fam, cols = [], [], []
    for F in ['SIGN', 'VAR', 'NUM', 'BIG']:
        df = {}
        for tk, _ in TT:
            for x in tk[F]:
                df[x] = df.get(x, 0) + 1
        keep = sorted(x for x, c in df.items() if c >= min_df)
        for x in keep:
            names.append(F + ':' + x)
            fam.append(F)
            cols.append(np.array([1.0 if x in tk[F] else 0.0 for tk, _ in TT]))
    X = np.array(cols).T
    fm = np.array([f for _, f in TT])
    fm = (fm - fm.mean(0)) / (fm.std(0) + 1e-9)
    X = np.hstack([X, fm])
    names += ['FMT:' + n for n in FMT_NAMES]
    fam += ['FMT'] * len(FMT_NAMES)
    return X, names, np.array(fam)


def length_bin(t):
    n = len(t['lines'])
    return 0 if n <= 1 else 1 if n == 2 else 2 if n == 3 else 3 if n <= 5 else 4 if n <= 9 else 5


def auc(pos, neg):
    pos = np.asarray(pos, float); neg = np.asarray(neg, float)
    if len(pos) == 0 or len(neg) == 0:
        return float('nan')
    allv = np.concatenate([pos, neg])
    from scipy.stats import rankdata
    r = rankdata(allv)
    return (r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def sha(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()
