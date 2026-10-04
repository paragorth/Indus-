"""pe24 control corpus: Ur III balanced accounts (CDLI dump in the scratchpad).

A text qualifies if it has a 'sag-nig2-gur11' (debit) section, a 'sza3-bi-ta'
(credit follows) line and a final 'la2-ia3 N' (deficit) line after it, and every
line that starts with a numeral parses as a count (disz/u/gesz2 ...). Target =
the final la2-ia3 value. Members = every other numeric line. Known truth:
debit lines (before sza3-bi-ta) +1, credit lines (after) -1, so debit - credit =
deficit. Features of a member: its section ('SEC_D' / 'SEC_C'), 'SUM' for a
szu-nigin2 line or a line carrying sag-nig2-gur11 / zi-ga summary words, 'LA2'
for an earlier (carried-over) la2-ia3 line, and decoys: the first word after the
numeral ('W_<word>'). The search sees only feature names, not their meaning.
"""
import os, re, json
from fractions import Fraction as Fr
from pe24_common import SCRATCH, CK

CNTU = {"disz": 1, "asz": 1, "gesz": 60, "u": 10, "gesz2": 60, "gesz'u": 600, "szar2": 3600, "szar'u": 36000}
NT = re.compile(r"^(\d+(?:/\d+)?)\(([a-z'0-9]+)\)$")


def parse_ur_line(txt):
    """Leading numeral of an Ur III line -> (value, is_total, rest, dimension) or None.
    dimension: 'n' count, 'c' capacity in sila3 (gur 300, barig 60, ban2 10),
    'w' weight in gin2 (ma-na 60, sze 1/180)."""
    toks = txt.split()
    tot = False
    if toks and toks[0].startswith('szu-nigin'):
        tot = True
        toks = toks[1:]
    nums = []
    k = 0
    while k < len(toks):
        m = NT.match(toks[k])
        if not m:
            break
        nums.append((Fr(m.group(1)), m.group(2)))
        k += 1
    if not nums:
        return None
    sub = None
    if k < len(toks) and toks[k] == 'la2':
        m = NT.match(toks[k + 1]) if k + 1 < len(toks) else None
        if not m or m.group(2) not in CNTU:
            return None
        sub = Fr(m.group(1)) * CNTU[m.group(2)]
        k += 2
    rest = toks[k:]
    rs = ' ' + ' '.join(rest) + ' '
    units = [u for _, u in nums]
    if any(u not in CNTU and u not in ('barig', 'ban2') for u in units):
        return None
    if 'barig' in units or 'ban2' in units or ' gur ' in rs or rs.endswith(' gur ') or ' gur' in rs:
        # capacity: count digits before barig/ban2 are gur unless no 'gur' word
        gur = ' gur' in rs
        v = Fr(0); phase = 0
        for n, u in nums:
            if u in ('barig', 'ban2'):
                phase = 1
                v += n * (60 if u == 'barig' else 10)
            elif phase == 0 and gur:
                v += n * CNTU[u] * 300
            else:
                v += n * CNTU[u]
        if sub:
            return None
        dim = 'c'
    elif rest and rest[0] == 'sila3':
        v = sum(n * CNTU[u] for n, u in nums) - (sub or 0); dim = 'c'
    elif rest and rest[0] in ('ma-na', 'gin2', 'sze'):
        f = {'ma-na': 60, 'gin2': 1, 'sze': Fr(1, 180)}[rest[0]]
        v = (sum(n * CNTU[u] for n, u in nums) - (sub or 0)) * f; dim = 'w'
        # trailing 'N gin2' after ma-na
        if rest[0] == 'ma-na' and len(rest) > 2 and NT.match(rest[1]) and rest[2] == 'gin2':
            m = NT.match(rest[1]); v += Fr(m.group(1)) * CNTU.get(m.group(2), 0)
    else:
        v = sum(n * CNTU[u] for n, u in nums) - (sub or 0); dim = 'n'
    if v <= 0:
        return None
    return v, tot, ' '.join(rest), dim

LINE = re.compile(r"^(\d+'?)\.\s+(.*)$")
NUMSTART = re.compile(r"^(\d+)\(")


def build_ur3(cache=os.path.join(CK, 'ur3_balanced.json')):
    if os.path.exists(cache):
        C = json.load(open(cache))
        for c in C:
            for h in c['hyps']:
                for p in h:
                    p['T'] = [Fr(x) for x in p['T']]
                    p['E'] = [[Fr(x) for x in e] for e in p['E']]
        return C
    txt = open(os.path.join(SCRATCH, 'cdli.atf'), encoding='utf-8', errors='replace').read()
    docs = re.split(r'\n(?=&P)', txt)
    out = []
    for d in docs:
        if 'sag-nig2-gur11' not in d or 'sza3-bi-ta' not in d or 'la2-ia3' not in d:
            continue
        pid = d[1:8]
        if '[' in d and re.search(r'\[\.\.\.\]|\$ .*broken', d):
            continue
        lines = []
        for raw in d.split('\n'):
            m = LINE.match(raw.strip())
            if m:
                t = m.group(2).replace('#', '').replace('?', '').replace('!', '').replace('[', '').replace(']', '')
                blank = '($ blank space $)' in t
                t = t.replace('($ blank space $)', '').strip()
                t = re.sub(r'^szunigin\b', 'szu-nigin2', t)
                lines.append(('szu-nigin2 ' + t) if blank and not t.startswith('szu-nigin') else t)
        sec = 'SEC_D'
        mem = []
        la2 = []
        bad = False
        dims = set()
        for i, t in enumerate(lines):
            if 'sza3-bi-ta' in t and not NUMSTART.match(t):
                sec = 'SEC_C'
                continue
            body = t
            is_la2 = False
            if body.startswith('la2-ia3 '):
                body = body[len('la2-ia3 '):]
                is_la2 = True
            if not NUMSTART.match(body.replace('szu-nigin2 ', '')):
                continue
            p = parse_ur_line(body)
            if p is None:
                bad = True
                break
            val, tot, rest, dim = p
            dims.add(dim)
            if is_la2:
                la2.append((len(mem), val, sec))
            w = rest.split()[0] if rest.split() else 'NONE'
            feats = [sec, 'W_' + w]
            if tot or re.search(r'sag-nig2-gur11|zi-ga-am3|zi-ga-a-am3', t):
                feats.append('SUM')
            if is_la2:
                feats.append('LA2')
            mem.append((val, feats, is_la2, sec))
        if bad or not la2 or len(dims) != 1:
            continue
        ti, tv, tsec = la2[-1]
        if tsec != 'SEC_C':
            continue
        members = [m for j, m in enumerate(mem) if j != ti and j < ti]
        if len(members) < 2 or not any(m[3] == 'SEC_D' for m in members) or not any(m[3] == 'SEC_C' for m in members):
            continue
        h = [{'cls': 'ur3', 'T': [Fr(tv)], 'E': [[Fr(m[0])] for m in members],
              'mk': [sorted(set(m[1])) for m in members], 'sg': [sorted(set(m[1])) for m in members]}]
        out.append({'id': pid, 'hyps': [h], 'cls': 'ur3'})
    json.dump(out, open(cache, 'w'), default=str)
    return out


if __name__ == '__main__':
    C = build_ur3()
    print(len(C))
    for c in C[:5]:
        print(c['id'], c['hyps'][0][0]['T'], c['hyps'][0][0]['E'], c['hyps'][0][0]['mk'])
