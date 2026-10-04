"""pe14 ('the tablet is woven'): ordered per-tablet units with numeral features.

Each tablet = list of units in reading order. A unit is one written line:
  toks  : signs / words (numerals removed)
  num   : 1 if the line carries a numeral
  sys   : numeral system (PE: common.system_of; others: lead numeral unit)
  lead  : code of the first numeral (largest unit written first)
  size  : bin of the summed numeral glyph count (0 none, 1, 2, 3-4, 5-9, 10+)
  surf  : 0 obverse, 1 other
  hdr   : 1 for a PE header line (first line, no numeral)
PE      : Proto-Elamite (same line rules as pe13: lone reverse total dropped).
UR3     : Ur III Drehem admin (cut at the date / summary block), lines WITH their
          unnumbered name lines kept (the known 'animals, then from/to PN' rhythm).
ARCH_ADM: proto-cuneiform admin (Uruk III/IV).
usage: python3 pe14_build.py <cdli.atf> <cdli_cat.csv> -> data/pe14_ckpt/units.json
"""
import csv, json, os, random, re, sys
from collections import defaultdict
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import load, base, is_sign, system_of  # noqa
from pe13_build import atf_tablets, cl, arch_toks, sum_toks  # noqa
CK = os.path.join(HERE, '..', 'data', 'pe14_ckpt')
os.makedirs(CK, exist_ok=True)


def sbin(n):
    if n <= 0:
        return 0
    return 1 if n == 1 else 2 if n == 2 else 3 if n <= 4 else 4 if n <= 9 else 5


def pe():
    out = []
    for t in load():
        lines = t['lines']
        off = [l for l in lines if l['surface'] != 'obverse' and l['numerals']]
        U = []
        for i, l in enumerate(lines):
            if l['surface'] != 'obverse' and len(off) == 1 and l is off[0]:
                continue
            sg = [base(s) for s in l['signs'] if is_sign(s)]
            if not sg:
                continue
            nm = l['numerals']
            U.append({'toks': sg, 'num': int(bool(nm)), 'sys': system_of(nm) or '-',
                      'lead': nm[0][1].split('@')[0] if nm else '-',
                      'size': sbin(sum(n for n, _ in nm if isinstance(n, int))),
                      'surf': 0 if l['surface'] == 'obverse' else 1,
                      'hdr': int(i == 0 and not nm)})
        if len(U) >= 3:
            out.append({'id': t['id'], 'u': U})
    return out


NUMP = re.compile(r"^([\d/]+)\(([^)]*)\)")


def nums(raw):
    out = []
    for w in raw.split():
        w = cl(w)
        m = NUMP.match(w)
        if m:
            try:
                n = int(m.group(1)) if '/' not in m.group(1) else 1
            except ValueError:
                n = 1
            out.append((n, m.group(2).split('@')[0]))
    return out


def cdli(atf, catf):
    sets = defaultdict(set)
    with open(catf, newline='') as f:
        for r in csv.DictReader(f):
            if not r['id_text'].isdigit():
                continue
            pid = 'P%06d' % int(r['id_text'])
            p, g = r['period'], r['genre']
            if (p.startswith('Uruk III') or p.startswith('Uruk IV')) and g.startswith('Admin'):
                sets['ARCH_ADM'].add(pid)
            elif p.startswith('Ur III') and 'Puzri' in r['provenience'] and g.startswith('Admin'):
                sets['UR3'].add(pid)
    who = {p: k for k, v in sets.items() for p in v}
    out = defaultdict(list)
    for pid, buf in atf_tablets(atf, set(who)):
        k = who[pid]
        U = []
        for b in buf:
            if k == 'UR3':
                w0 = b['raw'].split()[:1]
                if w0 and cl(w0[0]) in ('iti', 'mu', 'u4', 'szu-nigin2', 'szunigin', 'mu-DU', 'ba-zi', 'ba-zi-ga'):
                    break
                tk = sum_toks(b['raw'])
            else:
                tk = arch_toks(b['raw'])
            nm = nums(b['raw'])
            if not tk:
                continue
            U.append({'toks': tk, 'num': int(bool(nm)), 'sys': nm[0][1] if nm else '-',
                      'lead': nm[0][1] if nm else '-', 'size': sbin(sum(n for n, _ in nm)),
                      'surf': b['surf'], 'hdr': 0})
        if len(U) >= 3:
            out[k].append({'id': pid, 'u': U})
    rng = random.Random(14)
    for k in out:
        if len(out[k]) > 2500:
            out[k] = rng.sample(out[k], 2500)
    return out


if __name__ == '__main__':
    C = {'PE': pe()}
    C.update(cdli(sys.argv[1], sys.argv[2]))
    json.dump(C, open(os.path.join(CK, 'units.json'), 'w'))
    for k, v in C.items():
        n = sum(len(t['u']) for t in v)
        print(k, 'tablets', len(v), 'units', n, 'numbered', sum(u['num'] for t in v for u in t['u']))
