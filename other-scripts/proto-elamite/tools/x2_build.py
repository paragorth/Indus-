#!/usr/bin/env python3
"""X-2 build: four commodity-account corpora in one entry format.

Each document = {'id', 'site', 'entries': [entry...]}; entry = {
  'com': commodity token or None, 'des': [designation tokens], 'q': integer part or None,
  'frac': sub-unit present (bool), 'tot': total line (bool), 'kind': 'E' (entry with number),
  'H' (header line, no number, before the first entry) or 'T' (trailer, after the last entry),
  'multi': the same line carries more than one commodity }.

LA  : lineara.xyz corpus.json, tablets only; commodity = logogram base (la6_common rules);
      designation = the word heading the entry; totals = KU-RO / PO-TO-KU-RO.
PE  : CDLI pe_corpus.json; commodity = last sign of an entry when it is a final-class sign
      (data-driven: >= 20 final tokens and final share >= 0.6 in multi-sign entries);
      designation = the other signs (variants stripped); totals = numeric lines on the reverse
      when the reverse has <= 2 numeric lines; quantity: N01 1, N14 10, N34 60, N45 600,
      N48 3600; any other numeral code = sub-unit/other system (frac flag).
LB  : DAMOS items; commodity = logogram base (split at : ; +); words lower-case;
      totals = lines starting to-so / to-sa / to-so-de / to-sa-de; sub-units T V Z S M N P Q.
UR3 : CDLI ATF, Ur III administrative tablets (catalogue period 'Ur III'), seal lines dropped;
      entry = line starting with a numeral; commodity = first cleaned word after the numerals
      and units (determinatives removed; merged to its first hyphen component when that
      component stands alone as a commodity >= 20 times); designation = rest of the line and
      up to 2 following lines without a numeral; totals = szu-nigin2 lines; gur / ma-na main
      units, barig ban2 sila3 gin2 and n/m fractions = sub-units.
Output: proto-elamite/data/x2/corpus_{LA,PE,LB}.json; Ur III (105 MB) to $X2_BIG (scratch), not the repo
"""
import csv, json, os, re, sys, random
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
OS = os.path.join(HERE, '..', '..')
OUT = os.path.join(HERE, '..', 'data', 'x2')
os.makedirs(OUT, exist_ok=True)
SCR = os.environ.get('X2_SCRATCH', '/tmp/x2_scratch')   # holds cdli/cdli.atf and cdli_cat.csv
ATF = os.path.join(SCR, 'cdli', 'cdli.atf')
CAT = os.path.join(SCR, 'cdli_cat.csv')

sys.path.insert(0, os.path.join(OS, 'linear-a', 'tools'))


def E(com=None, des=None, q=None, frac=False, tot=False, kind='E', multi=False):
    return {'com': com, 'des': des or [], 'q': q, 'frac': bool(frac), 'tot': bool(tot), 'kind': kind,
            'multi': bool(multi)}


def mark_ht(entries):
    """Lines without numbers: header before first numeric entry, trailer after the last."""
    idx = [i for i, e in enumerate(entries) if e['kind'] == 'E']
    if not idx:
        return []
    f, l = idx[0], idx[-1]
    out = []
    for i, e in enumerate(entries):
        if e['kind'] != 'E':
            if i < f:
                e['kind'] = 'H'
            elif i > l:
                e['kind'] = 'T'
            else:
                e['kind'] = 'M'   # unnumbered line inside the list: kept as designation-only
        out.append(e)
    return out


# ------------------------------------------------------------------ Linear A
def build_la():
    import la6_common as L6
    C = json.load(open(os.path.join(OS, 'linear-a', 'data', 'corpus.json')))
    keep = {c['id'] for c in C if c['support'] == 'Tablet'}
    site = {c['id']: c['site'] for c in C}
    ents = L6.la_entries(use_frac=False)
    docs = defaultdict(list)
    for e in ents:
        if e['doc'] not in keep:
            continue
        lab = e['label']
        des = [lab] if (lab and lab not in L6.TOTALS) else []
        tot = e['role'] in ('total', 'grand')
        if lab in L6.TOTALS:
            des = [lab]          # the total word itself is a word item (KU-RO, KI-RO)
        items = []
        for base, v, fr in e['raw']:
            items.append(E(base, list(des), int(v), bool(fr), tot, 'E'))
        for v in e['bare']:
            items.append(E(None, list(des), int(v), v != int(v), tot, 'E'))
        if len(e['raw']) > 1:
            for it in items:
                if it['com']:
                    it['multi'] = True
        if not items:
            items = [E(None, list(des), None, False, False, 'X')]
        docs[e['doc']].extend(items)
    out = []
    for d, es in docs.items():
        es = mark_ht(es)
        if sum(1 for e in es if e['kind'] == 'E') >= 1:
            out.append({'id': d, 'site': site[d], 'entries': es})
    return out


# ------------------------------------------------------------------ Proto-Elamite
PE_VAL = {'N01': 1, 'N1': 1, 'N14': 10, 'N34': 60, 'N45': 600, 'N48': 3600}


def pe_norm(s):
    s = re.sub(r'~[a-z0-9]+', '', s)
    s = re.sub(r'@[a-z]', '', s)
    return s.strip('#?!')


def build_pe():
    P = json.load(open(os.path.join(HERE, '..', 'data', 'pe_corpus.json')))
    # final-class signs, data-driven
    fin = Counter(); multi_fin = Counter(); multi_any = Counter()
    for t in P:
        for l in t['lines']:
            if l['numerals'] and l['signs']:
                sg = [pe_norm(s) for s in l['signs'] if s != 'x']
                if not sg:
                    continue
                fin[sg[-1]] += 1
                if len(sg) >= 2:
                    multi_fin[sg[-1]] += 1
                    for s in sg:
                        multi_any[s] += 1
    cls = {s for s in fin if fin[s] >= 20 and multi_any[s] >= 5 and multi_fin[s] / multi_any[s] >= 0.6 and s not in ("n", "x", "X")}
    out = []
    for t in P:
        if t.get('object_type', 'tablet') != 'tablet':
            continue
        lines = [l for l in t['lines'] if l['surface'] in ('obverse', 'reverse', 'top')]
        rev_num = [l for l in lines if l['surface'] == 'reverse' and l['numerals']]
        obv_num = [l for l in lines if l['surface'] != 'reverse' and l['numerals']]
        es = []
        for l in lines:
            sg = [pe_norm(s) for s in l['signs'] if s != 'x']
            if l['numerals']:
                q = 0; frac = False
                for n, code in l['numerals']:
                    c0 = re.sub(r'@[a-z]', '', code)
                    if n is None:
                        continue
                    if c0 in PE_VAL:
                        q += n * PE_VAL[c0]
                    elif c0 != 'n':
                        frac = True
                tot = l['surface'] == 'reverse' and len(rev_num) <= 2 and len(obv_num) >= 2
                com = sg[-1] if sg and sg[-1] in cls else None
                des = sg[:-1] if com else sg
                es.append(E(com, des, q, frac, tot, 'E'))
            elif sg:
                es.append(E(None, sg, None, False, False, 'X'))
        es = mark_ht(es)
        if any(e['kind'] == 'E' for e in es):
            out.append({'id': t['id'], 'site': t['provenience'].split(' ')[0], 'entries': es})
    return out, sorted(cls)


# ------------------------------------------------------------------ Linear B
LB_SUB = set('T V Z S M N P Q'.split())
LB_TOT = {'to-so', 'to-sa', 'to-so-de', 'to-sa-de', 'to-to'}
LOGO = re.compile(r'^(\*\d+[A-Z]*|[A-Z][A-Z±]+)')


def lb_tok(t):
    return t.strip('[]⟦⟧,."\'').replace('̣', '')


def build_lb():
    out = []
    for line in open(os.path.join(OS, 'linear-a', 'data', 'damos_items.jsonl')):
        d = json.loads(line)
        h = d.get('heading') or ''
        es = []
        for ln in (d.get('content') or '').split('\n'):
            toks = [lb_tok(t) for t in ln.split()]
            toks = [t for t in toks if t and t not in ('/', ',', 'vac', 'vacat', 'v.', 'v.↓', 'lat.', 'inf.', 'sup.')
                    and not re.match(r'^\.\w*$', t)]
            words = []; items = []; cur = None; tot = False; sub_open = False
            for i, t in enumerate(toks):
                if re.match(r'^[a-z0-9*][a-z0-9*\-]*[a-z0-9]$', t) and '-' in t or re.match(r'^[a-z]{2,}$', t):
                    if t in LB_TOT and not items:
                        tot = True
                    words.append(t)
                    continue
                m = LOGO.match(t)
                if m and not (len(t) == 1):
                    base = re.split(r'[:;+]', t)[0]
                    if len(base) >= 2:
                        cur = {'com': base, 'q': 0, 'frac': False, 'has': False}
                        items.append(cur)
                    continue
                if t in LB_SUB and cur is not None:
                    sub_open = True
                    continue
                if re.match(r'^\d+$', t) and cur is not None:
                    if sub_open:
                        cur['frac'] = True
                    else:
                        cur['q'] += int(t)
                    cur['has'] = True
                    sub_open = False
                    continue
            items = [it for it in items if it['has']]
            des = list(words)   # total words (to-so) kept as word items
            if items:
                for it in items:
                    es.append(E(it['com'], list(des), it['q'], it['frac'], tot, 'E', len(items) > 1))
            elif des:
                es.append(E(None, des, None, False, False, 'X'))
        es = mark_ht(es)
        if any(e['kind'] == 'E' for e in es):
            m = re.match(r'^([A-Z]{2,3})', h)
            out.append({'id': h, 'site': m.group(1) if m else '', 'entries': es})
    return out


# ------------------------------------------------------------------ Ur III
UR_NUM = {'disz': 1, 'asz': 1, 'u': 10, 'gesz2': 60, "gesz'u": 600, 'szar2': 3600, "szar'u": 36000,
          'barig': 1, 'ban2': 1, 'diš': 1}
UR_MAIN = {'gur', 'ma-na', 'gu2'}
UR_SUB = {'sila3', 'gin2'}
UR_BAD = {'sar', 'iku', 'GAN2', 'bur3', 'esze3', 'kusz3', 'szu-si'}
UR_TOT = {'szunigin', 'szunigin2', 'szu-nigin2', 'szu-nigin'}
NUMTOK = re.compile(r"^(\d+(?:/\d+)?)\(([a-z0-9']+)(?:@[a-z])?\)$")


def ur_clean(t):
    t = re.sub(r'[\[\]#!?*<>]', '', t)
    t = re.sub(r'\{[^}]*\}', '', t)
    return t.strip('-')


def ur_parse(toks):
    """Return (q, frac, rest_tokens) or None if the line does not start with a numeral."""
    if not toks or not NUMTOK.match(toks[0].strip('#!?')):
        return None
    q = 0; frac = False; i = 0; seen_main = False
    while i < len(toks):
        t = toks[i].strip('#!?')
        m = NUMTOK.match(t)
        if m:
            num, unit = m.group(1), m.group(2)
            if '/' in num:
                frac = True
            elif unit in ('barig', 'ban2'):
                frac = True
            elif unit in UR_NUM:
                q += int(num) * UR_NUM[unit]
            else:
                return 'bad'
            i += 1; continue
        if t in UR_MAIN:
            seen_main = True; i += 1; continue
        if t in UR_SUB:
            frac = True
            if not seen_main:
                q = 0      # sila3 / gin2 alone: the count is of a sub-unit
            i += 1; continue
        if t in UR_BAD:
            return 'bad'
        if t == 'la2':
            i += 1; continue
        break
    rest = [ur_clean(x) for x in toks[i:]]
    rest = [x for x in rest if x and x not in UR_MAIN and x not in UR_SUB and not NUMTOK.match(x)]
    return q, frac, rest


def build_ur3(max_docs=None, seed=7):
    csv.field_size_limit(10 ** 9)
    keep = {}
    for row in csv.DictReader(open(CAT, encoding='utf-8', errors='replace')):
        if row.get('period', '').startswith('Ur III') and row.get('genre', '') == 'Administrative':
            pid = 'P%06d' % int(row['id_text']) if row.get('id_text', '').isdigit() else None
            if pid:
                keep[pid] = (row.get('provenience') or '').split(' (')[0]
    print('Ur III admin catalogue', len(keep), file=sys.stderr)
    raw = {}
    pid = None; lines = []; in_seal = False
    for line in open(ATF, encoding='utf-8', errors='replace'):
        if line.startswith('&P'):
            if pid in keep and lines:
                raw[pid] = lines
            pid = line.split()[0][1:]; lines = []; in_seal = False
            continue
        if pid not in keep:
            continue
        s = line.strip()
        if s.startswith('@'):
            if s.startswith('@seal') or s.startswith('@envelope'):
                in_seal = True
            elif s.startswith('@tablet'):
                in_seal = False
            continue
        if in_seal:
            continue
        m = re.match(r"^\d+'?\.\s*(.*)$", s)
        if m:
            lines.append(m.group(1))
    if pid in keep and lines:
        raw[pid] = lines
    print('Ur III admin with text', len(raw), file=sys.stderr)
    # pass 1: commodity first words
    first = Counter()
    parsed = {}
    for p, ls in raw.items():
        pl = []
        for body in ls:
            toks = body.split()
            tot = bool(toks) and ur_clean(toks[0]) in UR_TOT
            if tot:
                toks = toks[1:]
            r = ur_parse(toks)
            pl.append((tot, r, [ur_clean(x) for x in toks]))
            if r and r != 'bad' and r[2]:
                first[r[2][0]] += 1
        parsed[p] = pl
    def com_of(w):
        h = w.split('-')[0]
        return h if (h != w and first.get(h, 0) >= 20) else w
    out = []
    for p, pl in parsed.items():
        es = []; bad = False
        for j, (tot, r, ctoks) in enumerate(pl):
            if r == 'bad':
                bad = True; continue
            if r is None:
                ws = [x for x in ctoks if x]
                if ws:
                    es.append(E(None, ws, None, False, False, 'X'))
                continue
            q, frac, rest = r
            if not rest:
                continue
            com = com_of(rest[0])
            if com in ("...", "x", "n", "X") or "..." in com:
                com = None
            es.append(E(com, (['szu-nigin2'] if tot else []) + rest[1:], q, frac, tot, 'E'))
        if bad and len(es) < 2:
            continue
        es = mark_ht(es)
        if any(e['kind'] == 'E' for e in es):
            out.append({'id': p, 'site': keep[p], 'entries': es})
    return out


if __name__ == '__main__':
    which = sys.argv[1:] or ['LA', 'PE', 'LB', 'UR3']
    for w in which:
        if w == 'LA':
            D = build_la()
        elif w == 'PE':
            D, cls = build_pe()
            print('PE class signs', cls)
        elif w == 'LB':
            D = build_lb()
        elif w == 'UR3':
            D = build_ur3()
        od = os.environ.get('X2_BIG', os.path.join(SCR, 'x2')) if w == 'UR3' else OUT   # Ur III corpus too big for the repo
        os.makedirs(od, exist_ok=True)
        json.dump(D, open(os.path.join(od, 'corpus_%s.json' % w), 'w'))
        ne = sum(len(d['entries']) for d in D)
        cc = Counter(e['com'] for d in D for e in d['entries'] if e['com'])
        print(w, 'docs', len(D), 'entries', ne, 'commodity tokens', sum(cc.values()), 'types', len(cc))
        print('  top', cc.most_common(25))
        print('  totals', sum(1 for d in D for e in d['entries'] if e['tot']),
              'headers', sum(1 for d in D for e in d['entries'] if e['kind'] == 'H'))
