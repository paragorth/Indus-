#!/usr/bin/env python3
"""la47 shared code: documents as 2-D pages (items with spatial and order features), Linear B pages from
DAMOS, null transforms (within-page identity permutation; re-flow onto a donor page's line breaks),
planted spatial rule, and a random-grammar engine scored by held-out bits.
No sign values are used anywhere; identities only appear as targets."""
import json, os, re, collections, random, unicodedata
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, '..', 'data'); CK = os.path.join(D, 'la47_ckpt')

COMMOD = {'VIN', 'OLE', 'OLIV', 'VIR', 'GRA', 'NI', 'KI', 'DI', 'RE', 'TI', 'MA'}   # la45 COMMODITY (VIR+[?] -> VIR*)
HEADER = {'*301', 'KA', 'KU', 'SI', 'RO', 'ZE', 'A-DU', 'A-SA-SA-RA-ME'}            # la45 HEADER
TOTAL = {'KU-RO', 'PO-TO-KU-RO'}


def la45_class(w):
    if w in TOTAL: return 'total'
    if w in COMMOD or w.startswith('VIR'): return 'commodity'
    if w in HEADER: return 'header'
    return 'other'


# ---------------------------------------------------------------- pages
def la_pages(min_items=3, supports=('Tablet',), min_q=0.9):
    L = json.load(open(os.path.join(CK, 'layout.json')))
    pages = []
    for did, d in L.items():
        if supports and d['support'] not in supports: continue
        if d['align_q'] < min_q: continue
        items = []
        for t in d['toks']:
            if t['t'] in ('word', 'logo'):
                items.append({'k': 'w', 'id': t['w'], 'pl': t['pl'], 'pl2': t['pl2'], 'n': len(t['w'].split('-')) if t['t'] == 'word' else 1,
                              'logo': t['t'] == 'logo', 'box': t.get('box')})
            elif t['t'] == 'num':
                k = sum(1 for c in str(int(t['v'])) if c != '0') if t.get('v') else 0
                items.append({'k': 'n', 'v': t['v'] or 0, 'frac': bool(t.get('frac')), 'pl': t['pl'], 'pl2': t['pl2'], 'n': max(1, k + len(t.get('frac', [])))})
            elif t['t'] == 'div':
                items.append({'k': 'd', 'pl': t['pl'], 'pl2': t['pl2'], 'n': 1})
        if sum(i['k'] != 'd' for i in items) < min_items: continue
        m = re.match(r'(.*\d)([ab])$', did)
        side = m.group(2) if m and not did.startswith('HT154') else 'x'
        unit = m.group(1) if side != 'x' else did
        pages.append({'id': did, 'unit': unit, 'site': d['site'], 'side': side, 'nlines': d['nlines'], 'items': items,
                      'vb': d.get('vb')})
    return pages


MEAS = set('SVZTMNPQL')


def _clean(s):
    s = unicodedata.normalize('NFD', s)
    s = ''.join(c for c in s if unicodedata.category(c) != 'Mn')
    return re.sub(r'[\[\]⟦⟧<>{}?•]', '', s)


def lb_pages(min_items=3, sites=None):
    pages = []
    for l in open(os.path.join(D, 'damos_items.jsonl')):
        d = json.loads(l)
        if not d.get('heading') or not d.get('content'): continue
        site = d['heading'].split()[0]
        if sites and site not in sites: continue
        side = 'a'; items = []; pl = -1; nlines = 0
        for line in d['content'].split('\n'):
            s = line.strip()
            if not s: continue
            if s.startswith('v.'):
                side = 'b'; s = s[2:].lstrip('↓→←↑ ')
                if not s: continue
            m = re.match(r'^(\.\S+|lat\.\s*\S+)\s*(.*)$', s)
            if m: body = m.group(2)
            elif re.match(r'^(sup\.|inf\.|vac|reliqu|sigill|fragm|deest|vest)', s): continue
            else: body = s
            if re.search(r'\b(vac\.?|vacat|mut\.|deest|vest\.)', body) and not re.search(r'[a-z]-[a-z]|\d', _clean(body)): continue
            pl += 1; nlines += 1
            pend_meas = False
            for tok in _clean(body).split():
                if tok in (',', '/', '|', '+'): 
                    if tok == ',': items.append({'k': 'd', 'pl': pl, 'pl2': pl, 'n': 1, 'side': side})
                    continue
                tok = tok.strip(',./+')
                if not tok or tok in ('vac', 'vac.', 'vacat', 'mut', 'inf', 'sup', 'deest', 'vest', 'lat', 'o', 'qs', 'nihil'): 
                    if tok == 'o': pass
                    continue
                if re.fullmatch(r'\d+', tok):
                    if items and items[-1]['k'] == 'n' and items[-1]['pl'] == pl and pend_meas:
                        items[-1]['v'] += 0  # measure sub-unit: same numeral group
                    else:
                        items.append({'k': 'n', 'v': int(tok), 'frac': pend_meas, 'pl': pl, 'pl2': pl, 'n': len(tok), 'side': side})
                    pend_meas = False
                    continue
                if tok in MEAS: pend_meas = True; continue
                pend_meas = False
                if re.search(r'[a-z]', tok) and re.fullmatch(r"[a-z0-9*\-']+", tok):
                    w = tok.strip('-')
                    if w: items.append({'k': 'w', 'id': w, 'logo': False, 'pl': pl, 'pl2': pl, 'n': len(w.split('-')), 'side': side})
                elif re.fullmatch(r'[A-Z*0-9+:a-z]+', tok) and re.search(r'[A-Z*]', tok):
                    items.append({'k': 'w', 'id': tok, 'logo': True, 'pl': pl, 'pl2': pl, 'n': 1, 'side': side})
        if sum(i['k'] != 'd' for i in items) < min_items or nlines == 0: continue
        sides = {i['side'] for i in items}
        pages.append({'id': d['heading'].strip(), 'unit': d['heading'].strip(), 'site': site, 'side': 'x', 'nlines': nlines,
                      'items': items, 'lbsides': len(sides) > 1})
    return pages


# ---------------------------------------------------------------- features
SFEAT = ['vline', 'linit', 'lfin', 'wrap', 'side', 'numline', 'shape', 'vq', 'perline']
OFEAT = ['rank', 'nextnum', 'prevnum', 'prevdiv', 'rq', 'len']


def featurize(p):
    """returns list of (item, sdict, odict) for word/num items"""
    it = p['items']; nl = max(i['pl2'] for i in it) + 1
    byline = collections.defaultdict(list)
    for k, i in enumerate(it):
        if i['k'] != 'd': byline[i['pl']].append(k)
    real = [k for k, i in enumerate(it) if i['k'] != 'd']
    N = len(real); out = []
    for r, k in enumerate(real):
        i = it[k]; L = byline[i['pl']]
        vline = 'only' if nl == 1 else ('first' if i['pl'] == 0 else ('last' if i['pl'] == nl - 1 else 'mid'))
        side = i.get('side', p['side'])
        if p.get('lbsides') is not None: side = i.get('side', 'a') if p.get('lbsides') else 'x'
        later_same_line = [it[j] for j in L if j > k]
        earlier_same_line = [it[j] for j in L if j < k]
        if i['k'] == 'w': numline = int(any(x['k'] == 'n' for x in later_same_line))
        else: numline = int(any(x['k'] == 'w' for x in earlier_same_line))
        s = {'vline': vline, 'linit': int(L[0] == k), 'lfin': int(L[-1] == k), 'wrap': int(i['pl2'] > i['pl']), 'side': side,
             'numline': numline, 'shape': min(nl, 7) if nl < 4 else (4 if nl < 7 else 7),
             'vq': 0 if nl == 1 else min(3, int(4 * i['pl'] / nl)), 'perline': min(len(L), 4)}
        nxt = it[real[r + 1]] if r + 1 < N else None; prv = it[real[r - 1]] if r > 0 else None
        o = {'rank': 'first' if r == 0 else ('last' if r == N - 1 else ('second' if r == 1 else 'mid')),
             'nextnum': int(bool(nxt and nxt['k'] == 'n')), 'prevnum': int(bool(prv and prv['k'] == 'n')),
             'prevdiv': int(k > 0 and it[k - 1]['k'] == 'd'), 'rq': min(3, int(4 * r / N)),
             'len': 0 if N < 4 else (1 if N < 8 else (2 if N < 16 else 3))}
        out.append((i, s, o))
    return out


# ---------------------------------------------------------------- nulls
def null_permute(pages, rng):
    """identities shuffled among same-kind slots within each page (positions, numbers of signs kept per slot kind)"""
    out = []
    for p in pages:
        q = dict(p); its = [dict(i) for i in p['items']]
        for kind in ('w', 'n'):
            idx = [k for k, i in enumerate(its) if i['k'] == kind]
            vals = [{x: its[k][x] for x in its[k] if x not in ('pl', 'pl2', 'side')} for k in idx]
            rng.shuffle(vals)
            for k, v in zip(idx, vals):
                v.update({x: its[k][x] for x in ('pl', 'pl2', 'side') if x in its[k]}); its[k] = v
        q['items'] = its; out.append(q)
    return out


def _breaks(p):
    """fractions (0..1) of cumulative character length at which physical lines break"""
    it = p['items']; tot = sum(i['n'] for i in it); c = 0; br = []
    for a, b in zip(it, it[1:]):
        c += a['n']
        if b['pl'] > a['pl2']: br.append(c / tot)
    return br


def null_reflow(pages, rng, snap=True):
    """reading order kept, physical lines replaced by a DONOR page's line-break profile (same line count),
    i.e. page shapes held fixed while contents are swapped among them"""
    byn = collections.defaultdict(list)
    for p in pages: byn[p['nlines']].append(p)
    out = []
    for p in pages:
        donors = [x for x in byn[p['nlines']] if x is not p] or [p]
        br = _breaks(rng.choice(donors))
        if p['nlines'] > 1 and not br: br = sorted(rng.random() for _ in range(p['nlines'] - 1))
        it = [dict(i) for i in p['items']]; tot = sum(i['n'] for i in it); c = 0
        if snap:  # move each break to the nearest item boundary (no item is split), keep line count where possible
            bnd = np.cumsum([i['n'] for i in it])[:-1] / tot
            if len(bnd):
                br = sorted({float(bnd[np.argmin(np.abs(bnd - x))]) for x in br})
        for i in it:
            a = c / tot; b = (c + i['n'] - 1e-9) / tot; c += i['n']
            i['pl'] = sum(x <= a + 1e-12 for x in br); i['pl2'] = sum(x <= b for x in br)
            if i['k'] == 'd': i['pl2'] = i['pl']
        q = dict(p); q['items'] = it; out.append(q)
    return out


def plant_rule(pages, rng, frac_types=0.2, strength=1.0):
    """synthetic corpus: real identities shuffled within page (destroys real layout signal), then a PLANTED
    spatial rule: tokens of a random 20 % of word types are moved so that they start a physical line;
    pages are re-flowed so that each such token begins a new line (line count may change)."""
    types = sorted({i['id'] for p in pages for i in p['items'] if i['k'] == 'w'})
    X = set(rng.sample(types, int(frac_types * len(types))))
    out = []
    for p in null_permute(pages, rng):
        it = [dict(i) for i in p['items']]
        # re-flow: keep original line capacity (mean chars per line), but force a break before X tokens
        tot = sum(i['n'] for i in it); cap = max(2, round(tot / max(1, p['nlines'])))
        line = 0; c = 0
        for k, i in enumerate(it):
            force = i['k'] == 'w' and i['id'] in X and rng.random() < strength and k > 0
            if (force and c > 0) or (c + i['n'] > cap and c > 0 and not force):
                line += 1; c = 0
            i['pl'] = line; i['pl2'] = line; c += i['n']
        q = dict(p); q['items'] = it; q['nlines'] = line + 1; q['planted'] = X; out.append(q)
    return out


# ---------------------------------------------------------------- targets and table
def targets(item, la=True):
    if item['k'] == 'n':
        v = item['v']; return {'mag': ('frac' if v == 0 else ('1' if v == 1 else ('2-9' if v < 10 else ('10-99' if v < 100 else '100+'))))}
    w = item['id']
    return {'type': w, 'cls': la45_class(w) if la else lb_class(w)}


def lb_class(w):
    if w in ('to-so', 'to-sa', 'to-so-de', 'to-sa-de'): return 'total'
    if re.fullmatch(r'[A-Z*0-9+:a-z]+', w) and re.search(r'[A-Z*]', w): return 'commodity'
    return 'other'


def table(pages, la=True):
    """rows: dict with page index, kind, s-feature tuple, o-feature tuple, targets"""
    rows = []
    for pi, p in enumerate(pages):
        for i, s, o in featurize(p):
            r = {'p': pi, 'unit': p['unit'], 'site': p['site'], 'kind': i['k'], 's': s, 'o': o}
            r.update(targets(i, la)); rows.append(r)
    return rows


# ---------------------------------------------------------------- held-out bits
ALPHAS = np.array([1, 3, 10, 30, 100, 300, 1000.])


def heldout_bits(roles_tr, y_tr, roles_te, y_te, alpha=None, vocab_min=2):
    """alpha None -> vector over ALPHAS"""
    """bits/token gained by P(y|role) over P(y) on test tokens.  Types with < vocab_min train tokens -> RARE."""
    cnt = collections.Counter(y_tr)
    m = lambda y: y if cnt[y] >= vocab_min else '<R>'
    ytr = [m(y) for y in y_tr]; yte = [m(y) for y in y_te]
    V = sorted(set(ytr) | {'<R>'}); vi = {v: k for k, v in enumerate(V)}
    a = np.array([vi[y] for y in ytr]); b = np.array([vi.get(y, vi['<R>']) for y in yte])
    R = max(max(roles_tr), max(roles_te)) + 1
    ra = np.asarray(roles_tr); rb = np.asarray(roles_te)
    P0 = (np.bincount(a, minlength=len(V)) + 0.5) / (len(a) + 0.5 * len(V))
    J = np.zeros((R, len(V))); np.add.at(J, (ra, a), 1)
    n = J.sum(1, keepdims=True)
    al = ALPHAS if alpha is None else np.array([alpha], float)
    Jb = J[rb, b]; nb = n[rb, 0]; p0 = P0[b]
    g = np.log2((Jb[None, :] + al[:, None] * p0[None, :]) / (nb[None, :] + al[:, None])) - np.log2(p0)[None, :]
    g = g.mean(1)
    return g if alpha is None else float(g[0])


# ---------------------------------------------------------------- random grammars
def feat_values(rows, key, feats):
    return {f: sorted({r[key][f] for r in rows}, key=str) for f in feats}


def random_grammar(vals, rng, max_depth=3):
    """random decision tree over features (values only, no identities). returns nested tuple"""
    feats = list(vals)
    def node(d):
        if d == 0 or rng.random() < 0.15 * (max_depth - d): return None
        f = rng.choice(feats); vs = vals[f]
        if len(vs) < 2: return None
        left = set(rng.sample(vs, rng.randint(1, len(vs) - 1)))
        return (f, frozenset(left), node(d - 1), node(d - 1))
    t = node(max_depth)
    while t is None: t = node(max_depth)
    return t


def apply_grammar(t, feat):
    path = 0; depth = 0
    while t is not None:
        f, left, l, r = t
        go = feat[f] in left
        path = path * 2 + (0 if go else 1); depth += 1
        t = l if go else r
    return (depth, path)


def grammar_str(t):
    if t is None: return '*'
    f, left, l, r = t
    return '[%s in %s ? %s : %s]' % (f, sorted(left, key=str), grammar_str(l), grammar_str(r))


def role_ids(t, rows, key, extra=None):
    """map rows to integer role ids; extra: optional second grammar (product partition)"""
    lab = {}
    out = []
    for r in rows:
        a = apply_grammar(t, r[key])
        if extra is not None: a = (a, apply_grammar(extra[0], r[extra[1]]))
        out.append(lab.setdefault(a, len(lab)))
    return out, lab


def full_roles(rows, keys_feats, ref=None):
    """saturated partition: every combination of the listed features is a role"""
    lab = {} if ref is None else ref; out = []
    for r in rows:
        a = tuple(r[k][f] for k, fs in keys_feats for f in fs)
        out.append(lab.setdefault(a, len(lab)))
    return out, lab
