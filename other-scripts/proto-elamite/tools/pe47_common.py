"""pe47 shared code: THE TABLETS ARE PRINTOUTS OF ONE HIDDEN SPREADSHEET.

Every corpus is reduced to the same shape:
  tablet = {id, ctx: [context tokens (header / non-numeric lines)],
            ents: [(tokens, value, system)], meta: {...}}
PE: entry tokens = base sign forms of a line with a fully readable numeral;
    context = signs of the header lines (no numeral) before the first numeral line.
    Values: count system sexagesimal (N14 10, N34 60, N45 600, N48 3600, fractions 1/2),
    capacity in the a-priori bundling chain of pe24 (N39C = 1).
Ur III (Drehem = Puzris-Dagan, Umma): entry tokens = words after the numeral,
    context = words of non-numeric lines (iti lines tagged 'iti:'), values parsed by
    pe24_ur3.parse_ur_line, meta = catalogue date (king, year, month).
PLANT: a master table X[unit, commodity, period] sliced into tablets (see plant()).
"""
import collections, csv, json, os, random, re, sys
from fractions import Fraction as Fr

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, '..', 'data')
CK = os.path.join(DATA, 'pe47_ckpt')
os.makedirs(CK, exist_ok=True)
SCR = '/tmp/claude-0/-home-user-Indus-/874df4c7-80d6-5f08-b42c-eea96a214079/scratchpad'
sys.path.insert(0, HERE)
from common import load, base, is_sign  # noqa: E402

CAPSET = {'N39B', 'N30C', 'N24', 'N30D', 'N39C', 'N39A', 'N28', 'N29B', 'N39N'}
CAPV = {"N39C": 1, "N30D": 2, "N30C": 4, "N24": 12, "N39B": 24, "N01": 120, "N14": 720,
        "N45": 7200, "N34": 21600, "N48": 216000, "N51": 120, "N08": Fr(1, 2), "N08A": Fr(1, 2),
        "N8B": Fr(1, 2), "N02": Fr(1, 2)}
CNTV = {'N01': 1, 'N14': 10, 'N34': 60, 'N45': 600, 'N48': 3600, 'N51': 120,
        'N08': Fr(1, 2), 'N08A': Fr(1, 2), 'N8B': Fr(1, 2), 'N02': Fr(1, 2)}


def pe_value(nums):
    codes = {c for _, c in nums}
    if any(c is None or n is None or c == 'n' or '@' in str(c) for n, c in nums):
        return None
    if codes & CAPSET:
        tab, sysn = CAPV, 'cap'
    else:
        tab, sysn = CNTV, 'cnt'
    v = Fr(0)
    for n, c in nums:
        if c not in tab:
            return None
        v += Fr(n) * tab[c]
    return (v, sysn) if v > 0 else None


def build_pe():
    out = os.path.join(CK, 'pe.json')
    if os.path.exists(out):
        return json.load(open(out))
    T = load()
    res = []
    for t in T:
        lines = t['lines']
        ctx, ents, seen_num = [], [], False
        for l in lines:
            raw = l['raw']
            sg = [base(s) for s in l['signs'] if is_sign(s)]
            if not l['numerals']:
                if not seen_num:
                    ctx += sg
                continue
            seen_num = True
            tail = raw.split(',')[-1]
            if '...' in tail or '[' in tail or '?' in tail:
                continue
            pv = pe_value(l['numerals'])
            if pv is None or not sg or 'x' in l['signs']:
                continue
            ents.append([sg, float(pv[0]), pv[1], l['surface'] != 'obverse'])
        if ents:
            res.append(dict(id=t['id'], ctx=sorted(set(ctx)), ents=ents,
                            meta=dict(site=t.get('provenience', ''))))
    json.dump(res, open(out, 'w'))
    return res


LINE = re.compile(r"^(\d+'?)\.\s+(.*)$")
NUMSTART = re.compile(r"^(\d+(?:/\d+)?)\(")


def build_ur3(site):
    """site 'drehem' or 'umma'. Returns all administrative tablets with >= 1 parsed entry."""
    out = os.path.join(CK, 'ur3_%s.json' % site)
    if os.path.exists(out):
        return json.load(open(out))
    sys.path.insert(0, HERE)
    from pe24_ur3 import parse_ur_line
    key = 'Puzri' if site == 'drehem' else 'Umma'
    csv.field_size_limit(10 ** 9)
    meta = {}
    for row in csv.DictReader(open(os.path.join(SCR, 'cdli_cat.csv'), encoding='utf-8')):
        if row['period'].startswith('Ur III') and key in row['provenience'] and row['genre'].startswith('Admin'):
            m = re.match(r'^([^.]+)\.(\d\d|[a-z0-9]+)\.(\d\d)', row['dates_referenced'])
            d = dict(king='', year='', month=0)
            if m:
                d = dict(king=m.group(1), year=m.group(2), month=int(m.group(3)) if m.group(3).isdigit() else 0)
            meta['P%06d' % int(row['id_text'])] = d
    txt = open(os.path.join(SCR, 'cdli.atf'), encoding='utf-8', errors='replace').read()
    res = []
    for d in re.split(r'\n(?=&P)', txt):
        pid = d[1:8]
        if pid not in meta:
            continue
        ctx, ents = [], []
        surf = 'obverse'
        for raw in d.split('\n'):
            if raw.startswith('@'):
                if raw.startswith('@obverse') or raw.startswith('@reverse') or raw.startswith('@seal') or raw.startswith('@envelope'):
                    surf = raw[1:].split()[0]
                continue
            if surf in ('seal', 'envelope'):
                continue
            m = LINE.match(raw.strip())
            if not m:
                continue
            t = m.group(2)
            if '[' in t or 'x' in t.split():
                continue
            t = re.sub(r'[#?!<>]', '', t).strip()
            if NUMSTART.match(t):
                p = parse_ur_line(t)
                if p is None:
                    continue
                val, tot, rest, dim = p
                toks = [w for w in rest.split() if not NUMSTART.match(w)]
                if not toks or tot:
                    continue
                ents.append([toks[:4], float(val), dim, surf != 'obverse'])
            else:
                toks = t.split()
                if not toks:
                    continue
                if toks[0] == 'iti':
                    ctx += ['iti:' + w for w in toks[1:]]
                elif toks[0] == 'mu' or toks[0].startswith('u4'):
                    continue
                else:
                    ctx += [w for w in toks if not re.match(r'^\d', w)]
        if ents:
            res.append(dict(id=pid, ctx=sorted(set(ctx)), ents=ents, meta=meta[pid]))
    json.dump(res, open(out, 'w'))
    return res


def sample_like(corpus, n, seed):
    r = random.Random(seed)
    c = list(corpus)
    r.shuffle(c)
    return c[:n]


def plant(n_target, ent_len_dist, seed, U=60, C=40, P=12, survive=0.06, noise_tok=60):
    """Master table X[u,c,p] (sparse, low rank) printed as slices, a random 'survive' share kept.
    Tablet kinds: 'UP' (header unit+period, entries = commodity signs), 'CP' (header commodity+period,
    entries = unit signs), 'U*' summary (header unit, entries = commodities, values summed over periods),
    'C*' (header commodity, entries = units summed over periods). Entries carry 0-2 noise tokens.
    Truth: sign roles U, C, P."""
    r = random.Random(seed)
    import math
    act = {u: set(r.sample(range(C), r.randint(3, 10))) for u in range(U)}
    su = [math.exp(r.gauss(0, 0.8)) for _ in range(U)]
    bc = [math.exp(r.gauss(1.5, 1.0)) for _ in range(C)]
    sp = [math.exp(r.gauss(0, 0.4)) for _ in range(P)]
    X = {}
    for u in range(U):
        for c in act[u]:
            for p in range(P):
                v = max(1, int(round(su[u] * bc[c] * sp[p] * math.exp(r.gauss(0, 0.15)))))
                X[(u, c, p)] = v
    tabs = []
    kinds = ['UP'] * 5 + ['CP'] * 3 + ['U*', 'C*']
    n_print = int(n_target / survive)
    for i in range(n_print):
        k = r.choice(kinds)
        if k in ('UP', 'U*'):
            u = r.randrange(U)
            p = r.randrange(P)
            items = sorted(act[u])
            ctx = ['U%d' % u] + (['P%d' % p] if k == 'UP' else [])
            ents = []
            for c in items:
                v = X[(u, c, p)] if k == 'UP' else sum(X[(u, c, q)] for q in range(P))
                ents.append((['C%d' % c], v))
        else:
            c = r.randrange(C)
            p = r.randrange(P)
            us = [u for u in range(U) if c in act[u]]
            if not us:
                continue
            ctx = ['C%d' % c] + (['P%d' % p] if k == 'CP' else [])
            ents = []
            for u in us:
                v = X[(u, c, p)] if k == 'CP' else sum(X[(u, c, q)] for q in range(P))
                ents.append((['U%d' % u], v))
        tabs.append((k, ctx, ents))
    r.shuffle(tabs)
    tabs = tabs[:n_target]
    out = []
    for i, (k, ctx, ents) in enumerate(tabs):
        L = r.choice(ent_len_dist)
        r.shuffle(ents)
        ents = ents[:max(1, L)]
        E = []
        for toks, v in ents:
            toks = ['Z%d' % r.randrange(noise_tok) for _ in range(r.choice([0, 0, 1, 2]))] + toks
            E.append([toks, float(v), 'cnt', False])
        ctx = ctx + ['Z%d' % r.randrange(noise_tok) for _ in range(r.choice([0, 1]))]
        out.append(dict(id='PL%05d' % i, ctx=sorted(set(ctx)), ents=E, meta=dict(kind=k)))
    return out


def ent_lengths(corpus):
    return [len(t['ents']) for t in corpus]
