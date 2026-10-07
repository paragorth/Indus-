"""pe80 shared loaders: PE and Ur III tablets as ordered numeric lines.

Line record: dict(tok=[tokens], first=tok0|None, last=tokN|None, v=int|None, surf=str)
v is defined only when the numeral group uses unit and ten signs alone (PE N01/N14,
Ur III disz/u), so its value does not depend on the number system.
"""
import os, re, json, csv, random, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common

DATA = os.path.join(HERE, '..', 'data')
CK = os.path.join(DATA, 'pe80_ckpt')
os.makedirs(CK, exist_ok=True)
SCRATCH = os.environ.get('PE80_SCRATCH',
    '/tmp/claude-0/-home-user-Indus-/874df4c7-80d6-5f08-b42c-eea96a214079/scratchpad')


def pe_value(nums):
    if not nums:
        return None
    codes = {c for _, c in nums}
    if not codes <= {'N01', 'N14'} or any(n is None for n, _ in nums):
        return None
    return sum(n * (10 if c == 'N14' else 1) for n, c in nums)


def load_pe(mode=None):
    T = common.load(mode)
    out = []
    for t in T:
        L = []
        for l in t['lines']:
            if not l['numerals'] or l.get('lacuna'):
                continue
            toks = [common.base(s) for s in l['signs'] if s != 'x']
            L.append(dict(tok=toks, first=toks[0] if toks else None, last=toks[-1] if toks else None,
                          v=pe_value(l['numerals']), surf=l['surface'], nums=l['numerals']))
        out.append(dict(id=t['id'], lines=L, vol=t.get('volume'), site=t.get('site')))
    return out


NUMRE = re.compile(r"(\d+(?:/\d+)?)\(([^)]+)\)")


def ur3_ids(proven=('Puzrish-Dagan', 'Umma', 'Girsu')):
    csv.field_size_limit(10 ** 9)
    ids = {}
    with open(os.path.join(SCRATCH, 'cdli_cat.csv'), newline='', encoding='utf-8', errors='replace') as f:
        for row in csv.DictReader(f):
            if not row.get('period', '').startswith('Ur III'):
                continue
            p = row.get('provenience', '')
            for k in proven:
                if p.startswith(k):
                    ids['P%06d' % int(row['id_text'])] = k
                    break
    return ids


def build_ur3():
    fn = os.path.join(CK, 'ur3.json')
    if os.path.exists(fn):
        return json.load(open(fn))
    ids = ur3_ids()
    out = []
    cur = None
    surf = 'obverse'
    with open(os.path.join(SCRATCH, 'ur.atf'), encoding='utf-8', errors='replace') as f:
        for raw in f:
            raw = raw.rstrip('\n')
            if raw.startswith('&P'):
                pid = raw[1:8]
                cur = dict(id=pid, prov=ids.get(pid), lines=[]) if pid in ids else None
                if cur is not None:
                    out.append(cur)
                surf = 'obverse'
                continue
            if cur is None:
                continue
            if raw.startswith('@'):
                w = raw[1:].split()[0] if raw[1:].split() else ''
                if w in ('obverse', 'reverse', 'envelope', 'seal', 'left', 'edge', 'bottom', 'top'):
                    surf = w
                continue
            m = re.match(r"^\d+'?\.\s+(.*)$", raw)
            if not m or surf in ('seal', 'envelope'):
                continue
            txt = m.group(1)
            if '[' in txt or 'x' in re.findall(r"\b[x]\b", txt):
                # keep damaged lines out of the value series (as PE lacuna)
                if NUMRE.search(txt):
                    continue
            nums = NUMRE.findall(txt)
            if not nums:
                continue
            v = 0
            ok = True
            for n, u in nums:
                if '/' in n or u not in ('disz', 'u'):
                    ok = False
                    break
                v += int(n) * (10 if u == 'u' else 1)
            rest = NUMRE.sub(' ', txt)
            rest = re.sub(r"[_#?!*\[\]<>]", ' ', rest)
            toks = [x for x in re.split(r"[\s\-\.]+", rest) if x and not x.startswith('$')]
            cur['lines'].append(dict(tok=toks, first=toks[0] if toks else None, last=toks[-1] if toks else None,
                                     v=v if ok else None, surf=surf))
    out = [t for t in out if len(t['lines']) >= 2]
    json.dump(out, open(fn, 'w'))
    return out


def opaque(tabs, seed=0):
    """Replace every token with an opaque id (fixed map)."""
    rng = random.Random(seed)
    vocab = sorted({x for t in tabs for l in t['lines'] for x in l['tok']})
    ids = list(range(len(vocab)))
    rng.shuffle(ids)
    mp = {w: 'W%05d' % i for w, i in zip(vocab, ids)}
    for t in tabs:
        for l in t['lines']:
            l['tok'] = [mp[x] for x in l['tok']]
            l['first'] = l['tok'][0] if l['tok'] else None
            l['last'] = l['tok'][-1] if l['tok'] else None
    return tabs, mp
