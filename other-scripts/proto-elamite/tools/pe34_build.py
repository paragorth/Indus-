"""pe34 ('compound signs are sums'): build per-token behaviour records for
Proto-Elamite (PE), proto-cuneiform administrative tablets (ARCH, control) and
Linear B ideograms (LINB, control).

Each token = {s: sign as written (variants kept), t: tablet id, f: {feature: value}}.
Features (same families in all corpora, values corpus-specific):
  sys:<k>   numeral system / code signature of the line (one-hot, 'none' if no numeral)
  logq      log1p(total numeral marks) (LINB: log1p(number after the ideogram))
  num       line carries a numeral
  first / last / alone   position of the sign in the line's sign string
  logn      log number of signs in the line
  line0     sign is on the tablet's first written line
  rev       not on the obverse
  hdr:<k>   tablet header / series (one-hot over the 8 commonest)
  L:<k> R:<k>  left / right neighbour (one-hot over the 6 commonest neighbours)

usage: python3 pe34_build.py <cdli.atf> <cdli_cat.csv>  -> data/pe34_ckpt/tokens.json
"""
import csv, json, math, os, re, sys
from collections import Counter, defaultdict
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import load, is_sign, system_of  # noqa
from pe13_build import atf_tablets  # noqa
ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
CK = os.path.join(HERE, '..', 'data', 'pe34_ckpt')
os.makedirs(CK, exist_ok=True)
csv.field_size_limit(10 ** 9)


def finish(recs):
    """recs: list of (sign, tab, base feats, hdr, left, right). Adds one-hots."""
    H = Counter(r[3] for r in recs if r[3])
    hk = [h for h, _ in H.most_common(8)]
    LC = Counter(r[4] for r in recs if r[4])
    RC = Counter(r[5] for r in recs if r[5])
    lk = [x for x, _ in LC.most_common(6)]
    rk = [x for x, _ in RC.most_common(6)]
    out = []
    for s, t, f, h, l, r in recs:
        f = dict(f)
        for k in hk:
            f['hdr:' + k] = float(h == k)
        for k in lk:
            f['L:' + k] = float(l == k)
        for k in rk:
            f['R:' + k] = float(r == k)
        out.append({'s': s, 't': t, 'f': f})
    return out


def line_feats(sg, i, sysk, logq, li, rev, systems):
    f = {'sys:' + k: float(sysk == k) for k in systems}
    f.update({'logq': logq, 'num': float(sysk != 'none'), 'first': float(i == 0),
              'last': float(i == len(sg) - 1), 'alone': float(len(sg) == 1),
              'logn': math.log(len(sg)), 'line0': float(li == 0), 'rev': float(rev)})
    return f


def pe():
    recs, systems = [], ['none', 'SDB', 'C', 'B', 'N23', 'S-frac', 'C*', 'mod*']
    for t in load():
        hdr = None
        L0 = t['lines'][0] if t['lines'] else None
        if L0 and not L0['numerals']:
            hs = [s for s in L0['signs'] if is_sign(s)]
            hdr = hs[0] if hs else None
        for li, l in enumerate(t['lines']):
            sg = [s for s in l['signs'] if is_sign(s) or s == 'x']
            if not sg:
                continue
            sysk = system_of(l['numerals']) or 'none'
            logq = math.log1p(sum((n or 0) for n, _ in l['numerals']))
            rev = l['surface'] != 'obverse'
            for i, s in enumerate(sg):
                if s == 'x':
                    continue
                f = line_feats(sg, i, sysk, logq, li, rev, systems)
                recs.append((s, t['id'], f, hdr, sg[i - 1] if i else None,
                             sg[i + 1] if i + 1 < len(sg) else None))
    return finish(recs)


NUM = re.compile(r"^([\d/]+)\(([^)]+)\)$")


def arch(atf, catf):
    keep = set()
    with open(catf, newline='', errors='replace') as f:
        for r in csv.DictReader(f):
            if not r['id_text'].isdigit():
                continue
            p, g = r['period'], r['genre']
            if (p.startswith('Uruk III') or p.startswith('Uruk IV')) and g.startswith('Admin'):
                keep.add('P%06d' % int(r['id_text']))
    raw = []
    for pid, buf in atf_tablets(atf, keep):
        rows = []
        for li, b in enumerate(buf):
            s = re.sub(r'\[[^\]]*\]', ' ', b['raw'])
            s = re.sub(r'[#!?*<>]', '', s)
            nums, sg = [], []
            for w in s.replace(',', ' ').split():
                m = NUM.match(w)
                if m:
                    try:
                        n = float(eval(m.group(1))) if '/' in m.group(1) else int(m.group(1))
                    except Exception:
                        n = 1
                    nums.append((n, m.group(2)))
                elif w in ('x', 'X', '...') or w.startswith('$') or not re.match(r'^[|A-Z]', w):
                    continue
                else:
                    sg.append(w)
            if sg:
                rows.append((li, b['surf'], sg, nums))
        if rows:
            raw.append((pid, rows))
    sigC = Counter()
    for _, rows in raw:
        for _, _, _, nums in rows:
            if nums:
                sigC['+'.join(sorted({c.split('~')[0] for _, c in nums}))] += 1
    systems = ['none'] + [k for k, _ in sigC.most_common(8)] + ['other']
    recs = []
    for pid, rows in raw:
        hdr = rows[0][2][0] if rows and not rows[0][3] else None
        for li, surf, sg, nums in rows:
            if nums:
                k = '+'.join(sorted({c.split('~')[0] for _, c in nums}))
                k = k if k in systems else 'other'
            else:
                k = 'none'
            logq = math.log1p(sum(n for n, _ in nums))
            for i, s in enumerate(sg):
                f = line_feats(sg, i, k, logq, li, surf, systems)
                recs.append((s, pid, f, hdr, sg[i - 1] if i else None,
                             sg[i + 1] if i + 1 < len(sg) else None))
    return finish(recs)


def linb():
    recs = []
    for line in open(os.path.join(ROOT, 'other-scripts', 'linear-a', 'data', 'damos_items.jsonl')):
        it = json.loads(line)
        h = it.get('heading', '')
        m = re.match(r'(KN|PY)\s+([A-Z][a-z]?)', h)
        if not m:
            continue
        series = m.group(1) + m.group(2)
        surf, li = 0, -1
        for ln in (it.get('content') or '').split('\n'):
            if ln.strip().startswith('v.'):
                surf = 1
            mm = re.match(r'^\s*\.(\w+)\s+(.*)$', ln)
            if not mm:
                continue
            li += 1
            ws = [re.sub(r"[\[\]̣,/'?⟦⟧]", '', w) for w in mm.group(2).split()]
            ws = [w for w in ws if w]
            ideo = []  # (index, sign, number)
            hassyl = 0
            for j, w in enumerate(ws):
                if re.match(r'^\*?[A-Z][A-Z]', w):
                    num = None
                    if j + 1 < len(ws) and re.fullmatch(r'\d+', ws[j + 1]):
                        num = int(ws[j + 1])
                    ideo.append((j, w, num))
                elif '-' in w or re.fullmatch(r'[a-z]+\d?', w):
                    hassyl = 1
            sg = [w for _, w, _ in ideo]
            for i, (j, w, num) in enumerate(ideo):
                f = line_feats(sg, i, 'none', math.log1p(num or 0), li, surf, [])
                f['num'] = float(num is not None)
                f['syl'] = float(hassyl)
                recs.append((w, h, f, series, sg[i - 1] if i else None,
                             sg[i + 1] if i + 1 < len(sg) else None))
    return finish(recs)


if __name__ == '__main__':
    out = {'PE': pe(), 'LINB': linb(), 'ARCH': arch(sys.argv[1], sys.argv[2])}
    json.dump(out, open(os.path.join(CK, 'tokens.json'), 'w'))
    for k, v in out.items():
        print(k, 'tokens', len(v), 'types', len({r['s'] for r in v}), 'features', len(v[0]['f']))
