"""pe21 'THE SCRIBE KNEW THE LENGTH': shared data.

A scribe who knows how many entries a list will have sizes the clay to fit; one who writes
as events happen does not.  Per tablet we record physical size (CDLI catalogue, mm), text
amount (lines, glyphs = sign tokens + numeral impressions), how the faces were used
(lines per face, edge writing, '$ blank' notes per face) and labels for splits (number
system, seal, site, pe15 office, total present).

Ur III control tablets (CDLI bulk ATF + catalogue in the scratchpad, never committed) get
the same features plus a kind label:
  DAILY   = a single-day dated transaction (u4 N-kam) with a receipt / disbursement verb,
            no grand total (szu-nigin2), <= 20 lines;
  SUMMARY = a grand total (szu-nigin2) plus evidence of compilation (several day dates,
            several months, a month span -ta ... -sze3, or nig2-ka9-ak).
No sign readings from anyone are used.
"""
import csv, json, os, re, sys, collections
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from pe10_common import load as pe10_load, SCRATCH  # noqa
from pe18_common import line_system, seal_map, site, region  # noqa

DATA = os.path.join(HERE, '..', 'data')
CK = os.path.join(DATA, 'pe21_ckpt')
LOOPS = os.path.join(HERE, '..', 'loops')
os.makedirs(CK, exist_ok=True)

OFFICES = {
    'GRAIN': {'M010', 'M106', 'M002', 'M243', 'M075', 'M081', 'M265', 'M266', 'M296', 'M112'},
    'CLASS': {'M387', 'M388', 'M218', 'M124', 'M009', 'M066', 'M057'},
    'BARE': {'M054', 'M367', 'M001', 'M370', 'M032', '|M036+1(N30D)|', 'M206', 'M269', 'M059', 'M102'},
}
EDGES = ('top', 'left', 'bottom', 'right', 'edge')
LOST = re.compile(r'broken|missing|lines? (?:broken|missing)|start missing', re.I)


def _base(s):
    if s.startswith('|'):
        return s
    return re.sub(r'~[A-Za-z0-9]+', '', s)


def pe_tablets():
    T = pe10_load()
    S = seal_map()
    out = []
    for t in T:
        U = t['units']
        if not U:
            continue
        lost = any(LOST.search(m[3]) for m in t['markers']) or any(u['prime'] for u in U)
        faces = collections.Counter(u['face'] for u in U)
        blank = collections.Counter(m[0] for m in t['markers'] if 'blank' in m[3].lower())
        sysl = [line_system([(n, c) for n, c in u['nums']]) for u in U if u['nums']]
        sysc = collections.Counter(s for s in sysl if s)
        signs = [_base(s) for u in U for s in u['signs'] if s != 'x']
        off = collections.Counter()
        for s in signs:
            for k, v in OFFICES.items():
                if s in v:
                    off[k] += 1
        ent = [u for u in U if u['entry']]
        tot = [u for u in U if u['total']]
        nums = [(n, c) for u in U for n, c in u['nums']]
        out.append({
            'id': t['id'], 'h': t['h'], 'w': t['w'], 'th': t['th'], 'prov': t['prov'],
            'site': site(t['prov'] or ''), 'region': region(t['prov'] or ''),
            'intact': (not lost) and bool(t['h']) and bool(t['w']),
            'clean': not t['any_broken'],
            'n_lines': len(U), 'n_ent': len(ent), 'n_tot': len(tot),
            'glyphs': sum(len(u['signs']) + sum(n for n, _ in u['nums']) for u in U),
            'signs': sum(len(u['signs']) for u in U),
            'imps': sum(n for u in U for n, _ in u['nums']),
            'obv': faces.get('obverse', 0), 'rev': faces.get('reverse', 0),
            'edge': sum(faces.get(e, 0) for e in EDGES),
            'cols_obv': t['n_cols_obv'],
            'blank_obv': blank.get('obverse', 0), 'blank_rev': blank.get('reverse', 0),
            'spill': t['spill'],
            'sys': dict(sysc), 'sealed': S.get(t['id'], [False])[0],
            'office': off.most_common(1)[0][0] if off else 'NONE',
            'header': bool(U and U[0]['header']),
            'nums': nums,
            'ent_nums': [u['nums'] for u in ent],
            'tot_face': [u['face'] for u in tot],
        })
    return out


def dom_sys(t):
    s = t['sys']
    if not s:
        return 'NONUM'
    k = max(s, key=s.get)
    return {'C@': 'C', 'S@': 'SEX', 'AMB': 'AMB'}.get(k, k)


# ---------------------------------------------------------------- Ur III control
UR3_CACHE = os.path.join(CK, 'ur3.json')


def _graphemes(s):
    s = re.sub(r'[\[\]#?!<>_]', '', s)
    return [g for g in re.split(r'[\s\-.]+', s) if g and g != '...']


def ur3_tablets():
    if os.path.exists(UR3_CACHE):
        return json.load(open(UR3_CACHE))
    csv.field_size_limit(10 ** 9)
    cat = {}
    for row in csv.DictReader(open(os.path.join(SCRATCH, 'cdli_cat.csv'), encoding='utf-8')):
        if not row['period'].startswith('Ur III'):
            continue
        try:
            h = float(row['height']); w = float(row['width'])
        except Exception:
            continue
        if h <= 0 or w <= 0 or row['object_type'] != 'tablet':
            continue
        try:
            th = float(row['thickness'])
        except Exception:
            th = None
        cat['P%06d' % int(row['id_text'])] = (h, w, th, row['provenience'][:20])
    out = []
    cur = None
    face = None

    def finish(c):
        if not c or not c['L']:
            return
        txt = ' '.join(s for _, s in c['L'])
        faces = collections.Counter(f for f, _ in c['L'])
        days = set(re.findall(r'u4 (\d+)\(disz\)-kam', txt))
        itis = set(re.findall(r'iti ([^\s_]+)', txt))
        total = bool(re.search(r'szu-nigin|(?:^| )nigin2? \d', txt))
        verb = bool(re.search(r'mu-kux\(DU\)|mu-DU|ba-zi|i3-dab5|szu ba-ti', txt))
        comp = (len(days) >= 2 or len(itis) >= 2 or 'nig2-ka9' in txt
                or re.search(r'-ta\b.*iti .*-sze3', txt) is not None)
        kind = None
        if total and comp:
            kind = 'SUMMARY'
        elif len(days) == 1 and verb and not total and len(c['L']) <= 20 and len(itis) <= 1:
            kind = 'DAILY'
        h, w, th, prov = cat[c['id']]
        out.append({'id': c['id'], 'h': h, 'w': w, 'th': th, 'prov': prov, 'kind': kind,
                    'ndays': len(days), 'nitis': len(itis), 'verb': verb,
                    'nigka': 'nig2-ka9' in txt, 'span': re.search(r'-ta\b.*iti .*-sze3', txt) is not None,
                    'intact': not c['lost'], 'n_lines': len(c['L']),
                    'glyphs': sum(len(_graphemes(s)) for _, s in c['L']),
                    'obv': faces.get('obverse', 0), 'rev': faces.get('reverse', 0),
                    'edge': sum(faces.get(e, 0) for e in EDGES),
                    'blank_obv': c['blank'].get('obverse', 0), 'blank_rev': c['blank'].get('reverse', 0),
                    'total': total})

    for raw in open(os.path.join(SCRATCH, 'cdli.atf'), encoding='utf-8', errors='replace'):
        line = raw.rstrip()
        if line.startswith('&P'):
            finish(cur)
            pid = line[1:8]
            cur = {'id': pid, 'L': [], 'lost': False, 'blank': collections.Counter()} if pid in cat else None
            face = None
            continue
        if cur is None:
            continue
        if line.startswith('@'):
            tag = line[1:].split()[0] if line[1:].split() else ''
            if tag in ('envelope', 'seal'):
                face = None
            elif tag in ('obverse', 'reverse') or tag in EDGES:
                face = tag
            continue
        if line.startswith('$'):
            if LOST.search(line):
                cur['lost'] = True
            if 'blank' in line.lower() and face:
                cur['blank'][face] += 1
            continue
        m = re.match(r"^(\d+'?)\.\s+(.*)$", line)
        if m and face:
            if "'" in m.group(1) or '...' in m.group(2):
                cur['lost'] = True
            cur['L'].append((face, m.group(2)))
    finish(cur)
    json.dump(out, open(UR3_CACHE, 'w'))
    return out


# ---------------------------------------------------------------- fit metrics
def fit_stats(T, content='glyphs'):
    """Fill density D = content / area; planned clay -> tight D.
    Returns sd(log D), slope and R^2 of log content on log area, n."""
    A = np.log(np.array([t['h'] * t['w'] for t in T], float))
    C = np.log(np.array([max(t[content], 1) for t in T], float))
    D = C - A
    if len(T) < 5:
        return None
    sl, ic = np.polyfit(A, C, 1)
    r = np.corrcoef(A, C)[0, 1]
    return {'n': len(T), 'sdD': float(np.std(D)), 'slope': float(sl), 'r2': float(r * r),
            'r': float(r)}


def reassign_null(T, groups, stat, n=500, rng=None):
    """Keep each tablet's size; re-deal texts among tablets of the same group."""
    rng = rng or np.random.default_rng(0)
    idx = collections.defaultdict(list)
    for i, g in enumerate(groups):
        idx[g].append(i)
    vals = []
    for _ in range(n):
        perm = np.arange(len(T))
        for g, ii in idx.items():
            ii = np.array(ii)
            perm[ii] = rng.permutation(ii)
        T2 = [dict(T[i], h=T[j]['h'], w=T[j]['w']) for i, j in zip(range(len(T)), perm)]
        vals.append(stat(T2))
    return np.array(vals)


def write_rows(path, rows, header=''):
    with open(path, 'w') as f:
        if header:
            f.write(header.rstrip() + '\n\n')
        f.write('| id | method and control | result | verdict |\n|---|---|---|---|\n')
        for r in rows:
            f.write('| ' + ' | '.join(r) + ' |\n')


def ur3_kind(t):
    if t['ndays'] >= 2 or t['nigka'] or t['span'] or t['total'] or (t['nitis'] >= 2 and t['n_lines'] >= 8):
        return 'COMPILED'
    if t['ndays'] == 1 and t['verb'] and t['n_lines'] <= 20 and t['nitis'] <= 1:
        return 'DAILY'
    return None
