#!/usr/bin/env python3
"""LA-60 THE CRACK ATTEMPT: one explicit, machine-checkable working reading of the Linear A
administrative documents, turned into a generative document grammar (cycle 1), frozen and used to
decode held-out documents (cycle 2), and turned into frozen outside-corpus predictions (cycle 3).

Two readings are used:
  PRIOR  - assembled by hand from earlier loops' B/C results (FINDINGS.md: la8, la20, la40, la45,
           la46, la47, la49, la53, la54, la56). It was derived on the whole corpus, so on held-out
           Linear A documents it is contaminated; it is reported, but the clean test is INDUCED.
  INDUCED - the same kinds of statement produced by fixed, corpus-agnostic rules from the training
           documents only (total word = the word whose number closes the running sum; commodity
           class = logograms + single signs that take a number; heading signs = single signs that
           usually do not; heading/transaction words = words that open documents; commodity order
           = Bradley-Terry on training lists; site default = modal commodity per site).
           The same rules run on Linear B (KN+PY, sign identities only) as calibration.
No Linear B sound value enters any Linear A reading. Fraction letters are opaque.
"""
import json, os, re, sys, math, random, hashlib
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, '..', 'data')
CK = os.path.join(D, 'la60_ckpt')
os.makedirs(CK, exist_ok=True)
sys.path.insert(0, HERE)

LIB_SUPPORTS = {'Stone vessel', 'Metal object', 'Stone object', 'Architecture', 'ivory object',
                'Inked inscription', 'Graffito', 'Triton'}


def seed(name):
    return int(hashlib.sha256(name.encode()).hexdigest()[:8], 16)


def tab_of(i):
    return re.sub(r'[ab]$', '', i)


# ------------------------------------------------------------------ corpora
def _pub_source():
    s = open(os.path.join(D, 'LinearAInscriptions.js')).read()
    src = {}
    for m in re.finditer(r'\["([^"]+)",\{(.*?)\n\}\]', s, re.S):
        u = re.search(r'"imageRightsURL": "([^"]*)"', m.group(2))
        u = re.sub(r'#.*', '', u.group(1)) if u else ''
        g = re.search(r'GORILA-Vol(\d)', u)
        src[m.group(1)] = ('G' + g.group(1)) if g else ('blank' if u == '' else 'post')
    return src


SITE_CODE = {'Haghia Triada': 'HT', 'Khania': 'KH', 'Phaistos': 'PH', 'Knossos': 'KN', 'Zakros': 'ZA',
             'Arkhalkhori': 'ARKH', 'Malia': 'MA', 'Palaikastro': 'PK', 'Tylissos': 'TY', 'Petras': 'PE',
             'Thera': 'THE', 'Gournia': 'GO'}


def load_la():
    """All Linear A documents (joins whose parts are also listed are dropped)."""
    fn = os.path.join(CK, 'la_docs.json')
    if os.path.exists(fn):
        return json.load(open(fn))
    C = json.load(open(os.path.join(D, 'corpus.json')))
    src = _pub_source()
    ids = {d['id'] for d in C}
    out = []
    for d in C:
        if '+' in d['id']:
            parts = re.split(r'\+', re.sub(r'[ab]$', '', d['id']))
            pre = re.match(r'[A-Z]+[A-Za-z]*?(?=\d)', parts[0])
            pre = pre.group(0) if pre else ''
            comp = [parts[0]] + [p if not p[0].isdigit() else pre + p for p in parts[1:]]
            if all(any(i.startswith(c) for i in ids if i != d['id']) for c in comp):
                continue
        toks = []
        for t in d['tokens']:
            if t['t'] == 'word':
                toks.append(['W', '-'.join(t['s']), None])
            elif t['t'] == 'logo':
                toks.append(['L', t['v'], None])
            elif t['t'] == 'num':
                toks.append(['N', int(t['v']), ''.join(sorted(t['frac'])) if t['frac'] else ''])
            elif t['t'] == 'nl':
                if toks and toks[-1][0] != 'NL':
                    toks.append(['NL', None, None])
        while toks and toks[-1][0] == 'NL':
            toks.pop()
        while toks and toks[0][0] == 'NL':
            toks.pop(0)
        sup = d['support']
        out.append(dict(id=d['id'], tab=tab_of(d['id']), site=SITE_CODE.get(d['site'], d['site'] or '?'),
                        support=sup, pub=src.get(d['id'], 'blank'), scribe=d.get('scribe') or '',
                        lib=sup in LIB_SUPPORTS, toks=toks))
    json.dump(out, open(fn, 'w'))
    return out


def admin_docs(docs):
    """Administrative documents to decode: not a libation/object support, >= 1 number."""
    return [d for d in docs if not d['lib'] and any(t[0] == 'N' for t in d['toks'])]


def load_lb():
    """Linear B KN+PY reduced to sign identities: words, logograms (opaque), numbers (value)."""
    fn = os.path.join(CK, 'lb_docs.json')
    if os.path.exists(fn):
        return json.load(open(fn))
    from la41_common import lb_docs
    out = []
    for k, d in lb_docs().items():
        toks = []
        for it in d['items']:
            if toks:
                toks.append(['NL', None, None])
            if it['w']:
                toks.append(['W', '-'.join(it['w']).upper(), None])
            for c in it.get('ctx', []):
                toks.append(['W', '-'.join(c).upper(), None])
            if it['logo']:
                toks.append(['L', it['logo'], None])
            if it['num'] is not None:
                v = float(it['val'])
                iv = int(math.floor(v + 1e-9))
                fr = round(v - iv, 3)
                toks.append(['N', iv, ('%g' % fr) if fr > 0 else ''])
        if sum(t[0] != 'NL' for t in toks) >= 2:
            out.append(dict(id=k, tab=k, site=d['site'], support='Tablet', pub='G', scribe=d['support'],
                            lib=False, toks=toks))
    json.dump(out, open(fn, 'w'))
    return out


def split(docs, name, frac=0.5):
    """Split by tablet (both sides together), stratified by site."""
    rng = random.Random(seed(name))
    by = defaultdict(list)
    for d in docs:
        by[d['site']].append(d['tab'])
    test_tabs = set()
    for s, tabs in sorted(by.items()):
        u = sorted(set(tabs))
        rng.shuffle(u)
        test_tabs |= set(u[:int(round(len(u) * frac))])
    return [d for d in docs if d['tab'] not in test_tabs], [d for d in docs if d['tab'] in test_tabs]


# ------------------------------------------------------------------ readings
ROLES = ['HDR', 'TRX', 'TOT', 'RES', 'COM', 'CENT', 'ENT', 'SGL']


def base_of(x):
    """Commodity base of a logogram or commodity sign: part before '+', brackets stripped."""
    b = re.sub(r"[\[\]'?]", '', x.split('+')[0]).strip()
    return b or x


def prior_reading():
    """The PRIOR reading, assembled from FINDINGS.md (B and strongest C results)."""
    roles = {}
    for w in ['NI', 'KI', 'DI', 'RE', 'TI', 'MA', '*304', '*308', '*306', 'E', 'SU', '*307',
              'VIN', 'OLE', 'OLIV', 'VIR', 'GRA', 'CYP']:
        roles[w] = 'COM'                                      # la45 B, la8 B, la49 C, la20 C(*307)
    for w in ['*301', 'KA', 'KU', 'SI', 'RO', 'ZE', 'TE', 'A', 'I', 'TA', 'O', 'JA', 'DA', '*164',
              '*411-VS', 'U']:
        roles[w] = 'HDR'                                      # la45 B, la49 B, la28 C, la40 C
    for w in ['SA-RA₂', 'PA-DE', 'A-DU', 'KA-PA', 'KU-PA', 'A-PA-RA-NE', 'MA-KA-RI-TE', '*516',
              'KU-NI-SU', 'KU-RE', 'KA-NA']:
        roles[w] = 'TRX'                                      # la8 B, la53 B/C, la40 C
    roles['KU-RO'] = 'TOT'; roles['PO-TO-KU-RO'] = 'TOT'    # FINDINGS 2a, A
    roles['KI-RO'] = 'RES'                                    # FINDINGS 2a, residual/heading
    for w in ['OLE+U', 'OLE+MI', 'OLE+DI']:
        roles[w] = 'CENT'                                     # la45 C
    # commodity order: lower = written earlier (la20 A, la46 B SA-RA2 basket, la50 C)
    order = {'CYP': 0.0, 'GRA': 0.0, '*307': 0.0, 'NI': 1.0, 'VIR': 1.5, 'OLE': 1.5,
             'OLIV': 2.5, 'VIN': 2.5, 'QA2': 2.5}
    lib = ['A-TA-I-*301-WA-JA', 'JA-SA-SA-RA-ME', 'SI-RU-TE', 'I-PI-NA-MA', 'A-SA-SA-RA-ME',
           'I-DA-MA-TE', 'JA-DI-KI-TU', 'U-NA-KA-NA-SI', 'TA-NA-RA-TE-U-TI-NU', 'I-PI-NA-MI-NA']
    return dict(name='PRIOR', roles=roles, order=order, lib=set(lib),
                site_default={'KH': 'CYP', 'ZA': 'GRA', 'ARKH': 'GRA'}, ent_slot_unknown=True)


def _num_after(toks, i, skip_L=True):
    """Index of the number belonging to token i (next token, or after one logogram)."""
    j = i + 1
    if j < len(toks) and toks[j][0] == 'L' and skip_L:
        j += 1
    if j < len(toks) and toks[j][0] == 'N':
        return j
    return None


def close_test(total, s_int, s_fr, nfr, tol_lb=False):
    """Does a written total match the running sum? LA: integer part equals the integer sum, or exceeds
    it by at most nfr//2 + 1 when fractions are present (unknown fraction values). LB: values."""
    if tol_lb:
        return abs(total - (s_int + s_fr)) < 0.051 + 1e-9 * total
    if nfr == 0:
        return total == s_int
    return s_int <= total <= s_int + nfr // 2 + 1


def sections(doc, roles, lb=False):
    """Yield (tot_index, total_value, entries[(idx, v, frac)], closes) for each TOT in a doc."""
    toks = doc['toks']
    out = []
    ent = []
    skip = set()
    for i, t in enumerate(toks):
        if t[0] == 'W' and roles.get(t[1]) in ('TOT', 'RES'):
            j = _num_after(toks, i)
            if j is not None:
                skip.add(j)
                if roles.get(t[1]) == 'TOT':
                    tv = toks[j][1] + (float(toks[j][2]) if lb and toks[j][2] else 0.0)
                    s_int = sum(e[1] for e in ent)
                    s_fr = sum(float(e[2]) for e in ent if lb and e[2])
                    nfr = sum(1 for e in ent if e[2])
                    ok = close_test(tv, s_int, s_fr, nfr, lb) if len(ent) >= 2 else None
                    out.append((i, tv, list(ent), ok))
            ent = []
            continue
        if t[0] == 'N' and i not in skip:
            ent.append((i, t[1], t[2]))
    return out


def induce_reading(train, lb=False, name='INDUCED'):
    """Corpus-agnostic induction of the same kinds of statement, from training documents only."""
    occ = Counter(); fol_num = Counter(); first = Counter(); ndoc = Counter(); with_logo = Counter()
    for d in train:
        toks = d['toks']
        has_L = any(t[0] == 'L' for t in toks)
        content = [i for i, t in enumerate(toks) if t[0] != 'NL']
        seen = set()
        for i, t in enumerate(toks):
            if t[0] != 'W':
                continue
            w = t[1]
            occ[w] += 1
            if i + 1 < len(toks) and toks[i + 1][0] == 'N':
                fol_num[w] += 1
            if content and i == content[0]:
                first[w] += 1
            if w not in seen:
                ndoc[w] += 1; seen.add(w)
                if has_L:
                    with_logo[w] += 1
    # totals: words whose number equals the running sum of the entries above (>= 2 terms)
    close = Counter(); trials = Counter()
    for d in train:
        toks = d['toks']
        ent = []
        for i, t in enumerate(toks):
            if t[0] == 'W':
                j = _num_after(toks, i)
                if j is not None and len(ent) >= 2:
                    tv = toks[j][1] + (float(toks[j][2]) if lb and toks[j][2] else 0.0)
                    s_int = sum(e[1] for e in ent); s_fr = sum(float(e[2]) for e in ent if lb and e[2])
                    nfr = sum(1 for e in ent if e[2])
                    trials[t[1]] += 1
                    if close_test(tv, s_int, s_fr, nfr, lb) and tv > max(e[1] for e in ent):
                        close[t[1]] += 1
            if t[0] == 'N':
                ent.append((i, t[1], t[2]))
            if t[0] == 'W' and close[t[1]] >= 1 and trials[t[1]] and close[t[1]] / trials[t[1]] >= 0.25:
                ent = []                       # a known closing word resets the section
    roles = {}
    for w, c in close.items():
        if c >= 2 and c / trials[w] >= 0.2:
            roles[w] = 'TOT'
    nsign = lambda w: len(w.split('-'))
    for w, n in occ.items():
        if w in roles or n < 3:
            continue
        r = fol_num[w] / n
        if nsign(w) == 1:
            if r >= 0.6 and with_logo[w] / ndoc[w] >= 0.5:
                roles[w] = 'COM'
            elif r <= 0.35:
                roles[w] = 'HDR'
    # heading / transaction words: multi-sign words that open LISTS (documents with >= 3 numbers)
    lfirst = Counter(); lndoc = Counter()
    for d in train:
        toks = d['toks']
        if sum(t[0] == 'N' for t in toks) < 3:
            continue
        content = [i for i, t in enumerate(toks) if t[0] != 'NL']
        ws = set(t[1] for t in toks if t[0] == 'W')
        for w in ws:
            lndoc[w] += 1
        if content and toks[content[0]][0] == 'W':
            lfirst[toks[content[0]][1]] += 1
    for w, n in lndoc.items():
        if w in roles or nsign(w) < 2 or n < 2:
            continue
        if lfirst[w] / n >= 0.5:
            roles[w] = 'TRX'
    # logograms: compound with a non-fraction second element = compound entry (OLE+X kind)
    order = bt_order(train, roles)
    site_c = defaultdict(Counter)
    for d in train:
        for t in d['toks']:
            if t[0] == 'L' or (t[0] == 'W' and roles.get(t[1]) == 'COM'):
                site_c[d['site']][base_of(t[1])] += 1
    site_default = {s: c.most_common(1)[0][0] for s, c in site_c.items() if sum(c.values()) >= 5}
    return dict(name=name, roles=roles, order=order, lib=set(), site_default=site_default,
                ent_slot_unknown=True)


def com_seq(doc, roles):
    """Ordered distinct commodity bases in a document (first occurrence)."""
    out = []
    for t in doc['toks']:
        if t[0] == 'L' or (t[0] == 'W' and roles.get(t[1]) == 'COM'):
            b = base_of(t[1])
            if b not in out:
                out.append(b)
    return out


def bt_order(train, roles, iters=200):
    """Bradley-Terry scores from 'written before' pairs; returned as a rank position (lower=earlier)."""
    wins = Counter(); items = Counter()
    for d in train:
        s = com_seq(d, roles)
        for i in range(len(s)):
            items[s[i]] += 1
            for j in range(i + 1, len(s)):
                wins[(s[i], s[j])] += 1
    its = [x for x, c in items.items() if c >= 2 and any((x, y) in wins or (y, x) in wins for y in items)]
    if len(its) < 2:
        return {}
    p = {x: 1.0 for x in its}
    for _ in range(iters):
        newp = {}
        for x in its:
            W = sum(wins[(x, y)] for y in its) + 0.1
            den = sum((wins[(x, y)] + wins[(y, x)]) / (p[x] + p[y]) for y in its if y != x) + 0.2 / (p[x] + 1)
            newp[x] = W / den
        g = math.exp(sum(math.log(v) for v in newp.values()) / len(newp))
        p = {x: v / g for x, v in newp.items()}
    # rank: earlier = higher strength; convert to position scale
    return {x: -math.log(p[x]) for x in its}


def shuffle_reading(R, rng, keep_tot=False):
    """Role labels permuted among the lexicon's word types (class sizes kept); order scores permuted."""
    items = sorted(R['roles'].items())
    types = [w for w, r in items if not (keep_tot and r == 'TOT')]
    labs = [r for w, r in items if not (keep_tot and r == 'TOT')]
    rng.shuffle(labs)
    roles = dict(zip(types, labs))
    if keep_tot:
        roles.update({w: r for w, r in items if r == 'TOT'})
    ok = sorted(R['order']); ov = [R['order'][k] for k in ok]
    rng.shuffle(ov)
    sd = dict(R['site_default'])
    return dict(name=R['name'] + '-shuf', roles=roles, order=dict(zip(ok, ov)), lib=R['lib'],
                site_default=sd, ent_slot_unknown=True)


def reading_hash(R):
    s = json.dumps(dict(roles=sorted(R['roles'].items()), order=sorted((k, round(v, 6)) for k, v in R['order'].items()),
                        lib=sorted(R['lib']), site_default=sorted(R['site_default'].items())), ensure_ascii=False)
    return hashlib.sha256(s.encode()).hexdigest()


def role_of(t, R):
    if t[0] == 'W':
        r = R['roles'].get(t[1])
        if r:
            return r
        return 'SGL' if len(t[1].split('-')) == 1 else 'ENT'
    if t[0] == 'L':
        r = R['roles'].get(t[1])
        if r:
            return r
        return 'COM'
    return t[0]


def lb_prior_reading(LBall):
    """CALIBRATION ONLY: a partial Linear B reading of the same kind and rough size as the Linear A
    PRIOR, from Linear B knowledge (to-so 'total', o-pe-ro 'deficit', transaction/heading words),
    with the commodity order fitted on the whole LB corpus (contaminated, like the LA PRIOR)."""
    roles = {}
    for w in ['TO-SO', 'TO-SA', 'TO-SO-DE', 'TO-SA-DE']:
        roles[w] = 'TOT'
    roles['O-PE-RO'] = 'RES'
    for w in ['A-PU-DO-SI', 'O-U-DI-DO-SI', 'PA-RO', 'E-KE', 'E-KE-QE', 'O-NA-TO', 'KE-KE-ME-NA',
              'KI-TI-ME-NA', 'DA-MO', 'ME-TA-QE', 'PE-MO', 'PE-MA', 'O-DA-A2', 'KO-TO-NA', 'DO-SO-MO',
              'O-PI', 'DE-DE-ME-NO', 'DO-SE', 'TE-RE-TA', 'WO-ZE-E', 'E-RO2']:
        roles[w] = 'TRX'
    return dict(name='LBPRIOR', roles=roles, order=bt_order(LBall, roles), lib=set(), site_default={},
                ent_slot_unknown=True)


def _nbin_doc(d):
    n = sum(t[0] == 'N' for t in d['toks'])
    return 0 if n <= 1 else 1 if n == 2 else 2 if n <= 5 else 3


def lb_draw(LB, A, s):
    """Linear B documents at Linear A size, matching LA's mix of documents by number count."""
    rng = random.Random(seed('la60-lbdraw%d' % s))
    target = Counter()
    for d in A:
        target[_nbin_doc(d)] += sum(t[0] != 'NL' for t in d['toks'])
    pools = defaultdict(list)
    for d in LB:
        pools[_nbin_doc(d)].append(d)
    out = []
    for b, tgt in target.items():
        p = list(pools[b]); rng.shuffle(p); c = 0
        for d in p:
            if c >= tgt:
                break
            out.append(d); c += sum(t[0] != 'NL' for t in d['toks'])
    return out
