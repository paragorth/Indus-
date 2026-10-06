#!/usr/bin/env python3
"""la68 'the sealing signs are the tablets' words in shorthand'. Shared loaders and the rule engine.

Unit = a site (LA: Haghia Triada, Khania, Knossos, Phaistos, Zakros; LB: KN, PY, TH, MY, TI).
  abbr[s]  : counts of single-sign 'abbreviations' at the site
             LA: single-sign words on roundels, nodules and sealings (receipts).
             LB (calibration only, sign identities only): free-standing one-syllable words
             (pa, ne, pe, ki, ku ...) and syllabic ligature adjuncts (OLE+PA, TELA+PU, SUS+SI ...).
  words[s] : tablet word tokens of 2+ signs, each with features (signs, followed by a number,
             amount, line-initial).
A rule maps each tablet word to one of its signs (position option chosen per word length),
filters and weights the tokens, and predicts the abbreviation counts. Score = mean over units of
bits per abbreviation token gained over the unit's own any-position sign frequency.
Only sign identities are used; no sound values enter the Linear A side.
"""
import json, os, re, unicodedata, collections, math
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, '..', 'data')
CK = os.path.join(DATA, 'la68_ckpt')
os.makedirs(CK, exist_ok=True)

LA_R = {'Roundel', 'Nodule', 'Sealing'}
LA_T = {'Tablet', 'Lames (short thin tablet)'}
LA_SITES = ['Haghia Triada', 'Khania', 'Knossos', 'Phaistos', 'Zakros']

# position options per word: 0..5 = index from start, 6 last, 7 penultimate, 8 rarest sign (corpus-wide),
# 9 commonest sign, 10 = 'any' (weight split over all signs)
OPT_NAMES = ['p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'last', 'penult', 'rarest', 'commonest', 'any']
NOPT = len(OPT_NAMES)
ANY = 10
LMAX = 6           # lengths 2..6+ get their own option (6 = 6 or more)
FILTERS = ['all', 'num', 'nonum', 'lineinit', 'len3']
WEIGHTS = ['token', 'type', 'amount']


def la_findspots():
    meta = json.load(open(os.path.join(DATA, 'la28_ckpt', 'meta.json')))
    return {k: v.get('findspot', '') for k, v in meta.items()}


def load_la():
    """Return dict unit -> dict(abbr=Counter, abbr_docs=[list of sign lists per receipt], words=[...])."""
    d = json.load(open(os.path.join(DATA, 'corpus.json')))
    fs = la_findspots()
    units = {s: dict(abbr=collections.Counter(), abbr_docs=[], words=[], deps=collections.Counter())
             for s in LA_SITES}
    for r in d:
        s = r['site']
        if s not in units:
            continue
        if r['support'] in LA_R:
            signs = [t['s'][0] for t in r['tokens'] if t['t'] == 'word' and len(t['s']) == 1]
            if signs:
                units[s]['abbr_docs'].append(signs)
                for x in signs:
                    units[s]['abbr'][x] += 1
        elif r['support'] in LA_T:
            toks = r['tokens']
            first_on_line = True
            for i, t in enumerate(toks):
                if t['t'] == 'nl':
                    first_on_line = True
                    continue
                if t['t'] != 'word':
                    continue
                amt = None
                for u in toks[i + 1:i + 4]:
                    if u['t'] == 'num':
                        amt = u['v']
                        break
                    if u['t'] in ('word', 'nl'):
                        break
                if len(t['s']) >= 2:
                    units[s]['words'].append(dict(signs=tuple(t['s']), amt=amt, li=first_on_line,
                                                  doc=r['id'], dep=fs.get(r['id'], '')))
                    units[s]['deps'][fs.get(r['id'], '')] += 1
                first_on_line = False
    return units


def _clean(s):
    s = unicodedata.normalize('NFD', s)
    return ''.join(ch for ch in s if not unicodedata.combining(ch))


LB_SITES = ['KN', 'PY', 'TH', 'MY', 'TI']


def load_lb():
    units = {s: dict(abbr=collections.Counter(), abbr_docs=[], words=[], deps=collections.Counter())
             for s in LB_SITES}
    sylls = set()
    raw = []
    for l in open(os.path.join(DATA, 'damos_items.jsonl')):
        r = json.loads(l)
        h = r.get('heading') or ''
        m = re.match(r'([A-Z]+)\s', h)
        if not m or m.group(1) not in units:
            continue
        txt = _clean(r.get('content') or '')
        txt = re.sub(r'supra\s+sigillum(=[A-Z0-9=]+)?', ' ', txt)
        raw.append((m.group(1), h, txt))
        for w in re.findall(r"(?<![\w\[\]\-*])([a-z0-9*]+(?:-[a-z0-9*]+)+)(?![\w\[\]\-])", txt):
            for x in w.split('-'):
                sylls.add(x)
    sylls = {x for x in sylls if re.fullmatch(r'[a-z]{1,2}[0-9]?|\*\d+', x)}
    for site, h, txt in raw:
        u = units[site]
        abbr_doc = []
        for line in txt.split('\n'):
            toks = re.split(r'\s+', line.strip())
            first = True
            for i, tok in enumerate(toks):
                if re.fullmatch(r'\.[0-9a-zA-Z]+', tok):
                    continue
                # free-standing one-syllable word, 2+ letters (single letters collide with side labels)
                if re.fullmatch(r'[a-z]{2}[0-9]?', tok) and tok in sylls:
                    abbr_doc.append(tok)
                    first = False
                    continue
                # ligature adjuncts
                if re.fullmatch(r'[A-Z*][A-Z0-9*]*(\+[A-Z0-9*]+)+', tok):
                    for p in tok.split('+')[1:]:
                        pl = p.lower()
                        if pl in sylls:
                            abbr_doc.append(pl)
                    continue
                if re.fullmatch(r'[a-z0-9*]+(-[a-z0-9*]+)+', tok):
                    sg = tok.split('-')
                    if all(x in sylls for x in sg):
                        amt = None
                        for v in toks[i + 1:i + 4]:
                            if v.isdigit():
                                amt = int(v)
                                break
                            if re.fullmatch(r'[a-z0-9*]+(-[a-z0-9*]+)+', v):
                                break
                        u['words'].append(dict(signs=tuple(sg), amt=amt, li=first, doc=h, dep=''))
                    first = False
        if abbr_doc:
            u['abbr_docs'].append(abbr_doc)
            for x in abbr_doc:
                u['abbr'][x] += 1
    return units


# ------------------------------------------------------------------ engine
class Engine:
    """Precomputes, for every tablet word token, its sign under each position option."""

    def __init__(self, units, unit_names):
        self.names = unit_names
        signs = set()
        for s in unit_names:
            signs |= set(units[s]['abbr'])
            for w in units[s]['words']:
                signs |= set(w['signs'])
        self.signs = sorted(signs)
        self.idx = {x: i for i, x in enumerate(self.signs)}
        V = self.V = len(self.signs)
        glob = collections.Counter()
        for s in unit_names:
            for w in units[s]['words']:
                glob.update(w['signs'])
        self.U = {}
        for s in unit_names:
            ws = units[s]['words']
            n = len(ws)
            P = np.zeros((n, NOPT - 1), dtype=np.int32)
            L = np.zeros(n, dtype=np.int32)
            feat = {f: np.ones(n, dtype=bool) for f in FILTERS}
            amtw = np.ones(n)
            first_type = np.zeros(n, dtype=bool)
            seen = set()
            anymat = np.zeros((n, V))
            bg = np.zeros(V)
            for k, w in enumerate(ws):
                sg = w['signs']
                l = len(sg)
                L[k] = min(l, LMAX)
                ids = [self.idx[x] for x in sg]
                for o in range(6):
                    P[k, o] = ids[min(o, l - 1)]
                P[k, 6] = ids[-1]
                P[k, 7] = ids[-2]
                P[k, 8] = ids[min(range(l), key=lambda j: (glob[sg[j]], j))]
                P[k, 9] = ids[max(range(l), key=lambda j: (glob[sg[j]], -j))]
                for j in ids:
                    anymat[k, j] += 1.0 / l
                    bg[j] += 1
                feat['num'][k] = w['amt'] is not None
                feat['nonum'][k] = w['amt'] is None
                feat['lineinit'][k] = bool(w['li'])
                feat['len3'][k] = l >= 3
                amtw[k] = 1 + math.log2(1 + (w['amt'] or 0))
                if sg not in seen:
                    first_type[k] = True
                    seen.add(sg)
            a = np.zeros(V)
            for x, c in units[s]['abbr'].items():
                a[self.idx[x]] += c
            bgp = (bg + 0.5) / (bg.sum() + 0.5 * V)
            self.U[s] = dict(P=P, L=L, feat=feat, amtw=amtw, ftype=first_type, any=anymat, a=a,
                             bgp=bgp, logbg=np.log2(bgp), n=n)

    def predicted(self, s, rule, words_perm=None):
        """rule = (posmap: array len LMAX+1 of option ids indexed by length, filter, weight)."""
        u = self.U[s]
        posmap, filt, wt = rule
        mask = u['feat'][filt]
        if wt == 'token':
            w = mask.astype(float)
        elif wt == 'type':
            w = (mask & u['ftype']).astype(float)
        else:
            w = mask * u['amtw']
        opt = posmap[u['L']]
        t = np.zeros(self.V)
        isany = opt == ANY
        if isany.any():
            t += (u['any'][isany] * w[isany, None]).sum(0)
        nz = ~isany
        if nz.any():
            sel = u['P'][np.where(nz)[0], opt[nz]]
            t += np.bincount(sel, weights=w[nz], minlength=self.V)
        return t

    def score_unit(self, s, t, a=None, lam=0.5):
        u = self.U[s]
        if a is None:
            a = u['a']
        N = a.sum()
        if N == 0 or t.sum() == 0:
            return 0.0
        p = lam * t / t.sum() + (1 - lam) * u['bgp']
        return float((a * (np.log2(p) - u['logbg'])).sum() / N)

    def score(self, rule, units=None, abbrs=None):
        units = units or self.names
        vals = []
        for s in units:
            t = self.predicted(s, rule)
            vals.append(self.score_unit(s, t, None if abbrs is None else abbrs[s]))
        return float(np.mean(vals))


def named_rules():
    out = {}
    for o, nm in enumerate(OPT_NAMES):
        pm = np.full(LMAX + 1, o, dtype=np.int32)
        for f in FILTERS:
            for w in WEIGHTS:
                out[f'{nm}|{f}|{w}'] = (pm, f, w)
    return out


def random_rule(rng):
    pm = np.zeros(LMAX + 1, dtype=np.int32)
    for l in range(2, LMAX + 1):
        opts = [o for o in range(NOPT) if o >= 6 or o < l]
        pm[l] = rng.choice(opts)
    return (pm, FILTERS[rng.integers(len(FILTERS))], WEIGHTS[rng.integers(len(WEIGHTS))])


def rule_name(r):
    pm, f, w = r
    return '/'.join(OPT_NAMES[pm[l]] for l in range(2, LMAX + 1)) + f'|{f}|{w}'


def rule_family(r):
    """dominant option over lengths 2..4 (the bulk of words)."""
    pm = r[0]
    c = collections.Counter(OPT_NAMES[pm[l]] for l in (2, 3, 4))
    # 'p1' and initial are one family; for length 2, 'last' == p2, 'penult' == p1
    return c.most_common(1)[0][0]
