"""pe10 'the scribe ran out of clay': layout-aware parse of the PE corpus.

Each tablet -> ordered list of 'units' (text lines) with:
  face (obverse / reverse / top / left / bottom / right / other), column, line index on
  face+column, signs (M-signs, compounds, x), numerals [(count, code)], glyphs
  (signs + numeral impressions), flags (broken, header, total-like).
Face-level markers: blank space after text, broken.
Catalogue dimensions (mm) from CDLI cdli_cat.csv (cached to data/pe10_dims.json).
"""
import json, os, re, csv, random, math
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, '..', 'data')
CKPT = os.path.join(DATA, 'pe10_ckpt')
os.makedirs(CKPT, exist_ok=True)
ATF = os.path.join(DATA, 'pe_raw.atf')
DIMS = os.path.join(DATA, 'pe10_dims.json')
SCRATCH = '/tmp/claude-0/-home-user-Indus-/874df4c7-80d6-5f08-b42c-eea96a214079/scratchpad'

NUM_RE = re.compile(r'^(\d+|n)\(([^)]+)\)$')
FACES = {'obverse', 'reverse', 'top', 'left', 'bottom', 'right', 'edge'}


def clean_tok(t):
    t = t.replace('[', '').replace(']', '').replace('#', '').replace('?', '').replace('!', '')
    t = t.replace('<', '').replace('>', '')
    return t


def parse_body(body):
    signs, nums = [], []
    broken = '[' in body or '...' in body
    lhs, _, rhs = body.partition(',')
    for part in (lhs, rhs):
        for tok in part.split():
            if '...' in tok:
                continue
            c = clean_tok(tok)
            if not c:
                continue
            m = NUM_RE.match(c)
            if m:
                n = m.group(1)
                nums.append((int(n) if n != 'n' else 1, m.group(2)))
                if n == 'n':
                    broken = True
            elif c.startswith('M') or c.startswith('|') or c == 'x':
                signs.append(c)
    return signs, nums, broken, (',' in body)


def parse_atf(path=ATF):
    tabs = []
    cur = None
    face = None
    col = 1
    for raw in open(path, encoding='utf-8'):
        line = raw.rstrip('\n').rstrip()
        if line.startswith('&'):
            pid = line[1:].split('=')[0].strip()
            cur = {'id': pid, 'units': [], 'markers': [], 'faces_seen': []}
            tabs.append(cur)
            face, col = None, 1
            continue
        if cur is None:
            continue
        s = line.strip()
        if s.startswith('@'):
            tag = s[1:].split()[0] if s[1:].split() else ''
            if tag in FACES:
                face, col = tag, 1
                cur['faces_seen'].append(face)
            elif tag == 'column':
                try:
                    col = int(s.split()[1])
                except Exception:
                    col = 1
            continue
        if s.startswith('$'):
            cur['markers'].append((face, col, len(cur['units']), s[1:].strip()))
            continue
        m = re.match(r"^([0-9]+'?(?:\.[a-z0-9]+)*'?)\.\s+(.*)$", s)
        if m and face is not None:
            label, body = m.group(1), m.group(2)
            signs, nums, broken, comma = parse_body(body)
            cur['units'].append({'face': face, 'col': col, 'label': label,
                                 'sub': '.' in label, 'prime': "'" in label,
                                 'signs': signs, 'nums': nums, 'broken': broken,
                                 'raw': body})
    return tabs


def load_dims():
    if os.path.exists(DIMS):
        return json.load(open(DIMS))
    csv.field_size_limit(10 ** 9)
    out = {}
    for row in csv.DictReader(open(os.path.join(SCRATCH, 'cdli_cat.csv'), encoding='utf-8')):
        if not row['period'].startswith('Proto-Elamite'):
            continue
        pid = 'P%06d' % int(row['id_text'])

        def f(x):
            try:
                v = float(x)
                return v if v > 0 else None
            except Exception:
                return None
        out[pid] = {'h': f(row['height']), 'w': f(row['width']), 't': f(row['thickness']),
                    'pres': row['object_preservation'], 'surf': row['surface_preservation'],
                    'prov': row['provenience']}
    json.dump(out, open(DIMS, 'w'), indent=0)
    return out


def glyphs(u):
    return len(u['signs']) + sum(n for n, _ in u['nums'])


def annotate(tabs):
    """Add per-tablet layout facts and per-unit roles."""
    for t in tabs:
        U = t['units']
        mk = [m[3].lower() for m in t['markers']]
        t['any_broken'] = any(('broken' in m or 'missing' in m) for m in mk) or any(u['broken'] or u['prime'] for u in U)
        t['rev_blank'] = any(m[0] == 'reverse' and 'blank' in m[3].lower() for m in t['markers'])
        t['obv_blank'] = any(m[0] == 'obverse' and 'blank' in m[3].lower() for m in t['markers'])
        obv = [u for u in U if u['face'] == 'obverse']
        rev = [u for u in U if u['face'] == 'reverse']
        # header: first obverse line with signs and no numerals
        for i, u in enumerate(U):
            u['idx'] = i
            u['header'] = (i == 0 and u['face'] == 'obverse' and not u['nums'] and bool(u['signs']))
            u['entry'] = bool(u['signs']) and bool(u['nums']) and not u['header']
        # total-like: numeric lines off the obverse when reverse has exactly one numeric line,
        # and any numeric line on top/left edges
        rev_num = [u for u in rev if u['nums']]
        for u in U:
            u['total'] = False
            if u['face'] in ('top', 'left', 'bottom', 'right', 'edge') and u['nums']:
                u['total'] = True
            if u['face'] == 'reverse' and len(rev_num) == 1 and u is rev_num[0]:
                u['total'] = True
        rev_entries = [u for u in rev if u['entry'] and not u['total']]
        t['spill'] = len(rev_entries) >= 1      # entries continue on reverse
        t['n_obv'] = len(obv)
        t['n_obv_entries'] = sum(1 for u in obv if u['entry'])
        t['n_rev_entries'] = len(rev_entries)
        t['n_cols_obv'] = len({u['col'] for u in obv}) if obv else 0
        t['n_cols_rev'] = len({u['col'] for u in rev}) if rev else 0
    return tabs


def load():
    tabs = annotate(parse_atf())
    dims = load_dims()
    for t in tabs:
        d = dims.get(t['id'], {})
        t['h'], t['w'], t['th'] = d.get('h'), d.get('w'), d.get('t')
        t['pres'] = d.get('pres', '')
        t['prov'] = d.get('prov', '')
    return tabs


def entry_string(u):
    return tuple(u['signs'])


def base(s):
    if s.startswith('|'):
        parts = re.split(r'([+.x&])', s.strip('|'))
        return '|' + ''.join(re.sub(r'~[A-Za-z0-9]+', '', p) for p in parts) + '|'
    return re.sub(r'~[A-Za-z0-9]+', '', s)


def write_rows(path, rows, header=None):
    with open(path, 'w') as f:
        if header:
            f.write(header.rstrip() + '\n\n')
        f.write('| id | method and control | result | verdict |\n|---|---|---|---|\n')
        for r in rows:
            f.write('| ' + ' | '.join(r) + ' |\n')
