"""pe13 ('the scribe's memory has a shape'): build ordered per-tablet line corpora.

Each tablet = list of lines in reading order; each line = {toks, surf (0 obverse,
1 reverse/other), col, li (line index over all written lines), ent (1 if an entry
with a numeral), cls (final sign if a class sign), mid (tokens minus class sign)}.

PE        : Proto-Elamite, every line with >= 1 readable sign (x dropped), lone
            reverse totals dropped (same rule as common.entries). Base signs.
ARCH_LEX  : proto-cuneiform lexical lists (Uruk III/IV) - COPIED from a fixed
            canonical sequence (CDLI composite line refs >>Q... give the truth).
ARCH_ADM  : proto-cuneiform administrative tablets (Uruk III/IV), PE's elder sibling.
UR3       : Ur III Drehem administrative tablets (cut before the date / totals block).
LINB      : Linear B (DAMOS), Knossos and Pylos, lines with words or ideograms.

usage: python3 pe13_build.py <cdli.atf> <cdli_cat.csv>  -> data/pe13_ckpt/corpora.json
"""
import csv, json, os, random, re, sys
from collections import defaultdict
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import load, base, is_sign  # noqa
from pe4_common import FINAL  # noqa
ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
CK = os.path.join(HERE, '..', 'data', 'pe13_ckpt')
os.makedirs(CK, exist_ok=True)


def pe():
    out = []
    for t in load():
        lines = t['lines']
        off = [l for l in lines if l['surface'] != 'obverse' and l['numerals']]
        L = []
        for i, l in enumerate(lines):
            if l['surface'] != 'obverse' and len(off) == 1 and l is off[0]:
                continue
            sg = [base(s) for s in l['signs'] if is_sign(s)]
            if not sg:
                continue
            cls = sg[-1] if (sg[-1] in FINAL and len(sg) >= 2) else ''
            L.append({'toks': sg, 'surf': 0 if l['surface'] == 'obverse' else 1,
                      'col': l.get('column') or 1, 'li': i, 'ent': int(bool(l['numerals'])),
                      'cls': cls, 'mid': sg[:-1] if cls else sg})
        if len(L) >= 3:
            out.append({'id': t['id'], 'lines': L})
    return out


NUM = re.compile(r"^[\d/]+\(")


def atf_tablets(atf, keep):
    cur, surf, col, buf = None, 0, 1, []
    with open(atf, errors='replace') as f:
        for line in f:
            if line.startswith('&P'):
                if cur and buf:
                    yield cur, buf
                cur = line[1:8] if line[1:8] in keep else None
                surf, col, buf = 0, 1, []
                continue
            if cur is None:
                continue
            s = line.rstrip('\n')
            if s.startswith('@'):
                w = s[1:].split()
                if w and w[0] == 'obverse':
                    surf = 0
                elif w and w[0] in ('reverse', 'left', 'right', 'top', 'bottom', 'edge', 'seal', 'envelope'):
                    surf = 1
                if w and w[0] == 'column':
                    try:
                        col = int(re.sub(r'\D', '', w[1]) or 1)
                    except Exception:
                        col = 1
                continue
            if s.startswith('>>Q') and buf:
                buf[-1]['q'] = s[2:].strip()
                continue
            m = re.match(r"^([0-9]+[a-z0-9.']*)\.\s+(.*)$", s)
            if not m:
                continue
            buf.append({'surf': surf, 'col': col, 'raw': m.group(2)})
    if cur and buf:
        yield cur, buf


def cl(w):
    w = re.sub(r'[#!?*<>]', '', w)
    return w


def arch_toks(raw):
    if '[' in raw:
        raw = re.sub(r'\[[^\]]*\]', ' ', raw)
    parts = raw.split(',', 1)
    body = parts[1] if len(parts) == 2 else parts[0]
    out = []
    for w in body.split():
        w = cl(w)
        if not w or NUM.match(w) or w in ('x', '...', 'X') or w.startswith('$'):
            continue
        w = re.sub(r'~[A-Za-z0-9]+', '', w)
        out.append(w)
    return out


def sum_toks(raw):
    if '[' in raw:
        raw = re.sub(r'\[[^\]]*\]', ' ', raw)
    out = []
    for w in raw.split():
        w = cl(w)
        if not w or NUM.match(w) or w in ('x', '...'):
            continue
        w = re.sub(r'\{[^}]*\}', lambda m: m.group(0).strip('{}') + '-', w)
        for s in re.split(r'[-.]', w):
            if s and s not in ('x', '...') and not re.match(r'^\d', s):
                out.append(s)
    return out


def cdli(atf, catf):
    sets = defaultdict(set)
    with open(catf, newline='') as f:
        for r in csv.DictReader(f):
            if not r['id_text'].isdigit():
                continue
            pid = 'P%06d' % int(r['id_text'])
            p, g = r['period'], r['genre']
            if p.startswith('Uruk III') or p.startswith('Uruk IV'):
                if g.startswith('Lexical'):
                    sets['ARCH_LEX'].add(pid)
                elif g.startswith('Admin'):
                    sets['ARCH_ADM'].add(pid)
            elif p.startswith('Ur III') and 'Puzri' in r['provenience'] and g.startswith('Admin'):
                sets['UR3'].add(pid)
    who = {}
    for k, v in sets.items():
        for p in v:
            who[p] = k
    out = defaultdict(list)
    for pid, buf in atf_tablets(atf, set(who)):
        k = who[pid]
        L = []
        for i, b in enumerate(buf):
            if k == 'UR3':
                w0 = b['raw'].split()[:1]
                if w0 and cl(w0[0]) in ('iti', 'mu', 'u4', 'szu-nigin2', 'szunigin', 'mu-DU', 'ba-zi', 'ba-zi-ga'):
                    break
                tk = sum_toks(b['raw'])
            else:
                tk = arch_toks(b['raw'])
            if not tk:
                continue
            L.append({'toks': tk, 'surf': b['surf'], 'col': b['col'], 'li': i, 'ent': 1,
                      'cls': '', 'mid': tk, 'q': b.get('q', '')})
        if len(L) >= 3:
            out[k].append({'id': pid, 'lines': L})
    rng = random.Random(13)
    for k in out:
        if len(out[k]) > 2500:
            out[k] = rng.sample(out[k], 2500)
    return out


def linb():
    out = []
    for l in open(os.path.join(ROOT, 'other-scripts', 'linear-a', 'data', 'damos_items.jsonl')):
        it = json.loads(l)
        h = it.get('heading', '')
        if not re.match(r'(KN|PY)\s', h):
            continue
        surf, L = 0, []
        for i, ln in enumerate((it.get('content') or '').split('\n')):
            if ln.strip().startswith('v.') or ln.strip().startswith('rev'):
                surf = 1
            m = re.match(r'^\s*\.(\d+[ab]?)\s+(.*)$', ln)
            if not m:
                continue
            tk = []
            for w in m.group(2).split():
                w = re.sub(r"[\[\]̣,/'?]", '', w)
                if not w or re.fullmatch(r'[0-9]+', w) or w in ('vac.', 'vest.', 'deest', 'qs', 'inf.', 'mut.', 'sup.', 'lat.', 'inf', 'sup', 'mut'):
                    continue
                if re.fullmatch(r'[A-Z]', w):  # metric letters S V Z T ...
                    continue
                if '-' in w:
                    tk += [s for s in w.split('-') if s]
                elif re.match(r'^[A-Z*]', w) or re.fullmatch(r'[a-z0-9]+', w):
                    tk.append(w)
            if tk:
                L.append({'toks': tk, 'surf': surf, 'col': 1, 'li': i, 'ent': 1, 'cls': '', 'mid': tk})
        if len(L) >= 3:
            out.append({'id': h, 'lines': L})
    return out


if __name__ == '__main__':
    C = {'PE': pe(), 'LINB': linb()}
    C.update(cdli(sys.argv[1], sys.argv[2]))
    json.dump(C, open(os.path.join(CK, 'corpora.json'), 'w'))
    for k, v in C.items():
        n = sum(len(t['lines']) for t in v)
        tok = sum(len(l['toks']) for t in v for l in t['lines'])
        rev = sum(l['surf'] for t in v for l in t['lines'])
        print(k, 'tablets', len(v), 'lines', n, 'tokens', tok, 'rev lines', rev)
