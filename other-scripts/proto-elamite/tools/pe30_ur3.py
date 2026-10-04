"""pe30 control corpus: Ur III / Sumerian count accounts with szu-nigin2 totals.

From the CDLI dump (scratchpad, not committed). A text qualifies if it has
2-40 count entry lines (parse_ur_line from pe24, dimension 'n'), then one or
more 'szu-nigin2 N <word>' lines, every one parsing as a count, no '[...]' and
no '$ ... broken' state line. Each szu-nigin2 line is one case, members = all entry lines before the first
szu-nigin2 line.
Known accounting rule (from Ur III practice, not from the search): a
szu-nigin2 total sums the entries of ITS OWN commodity word (category totals,
e.g. 'szu-nigin2 5 udu' / 'szu-nigin2 3 masz2'), i.e. KEEP(T_share) on texts
with several category totals. The search sees only the feature names.
Features (same families as PE): P_ positions, S_<word> every word, F_/L_ first
and last word, N0/N1/N2/N3p word count, Z_ sizes, D_ duplicates, T_share
(entry's first word = the total's first word), T_none, T_last.
"""
import os, re, json, collections
from fractions import Fraction as Fr
from pe24_ur3 import parse_ur_line
from pe30_common import SCRATCH, CK

LINE = re.compile(r"^(\d+'?)\.\s+(.*)$")
NUMSTART = re.compile(r"^(\d+(?:/\d+)?)\(")


def feats_ur(members, tot_words):
    n = len(members)
    vals = [m[0] for m in members]
    tot = sum(vals)
    vc = collections.Counter(vals); sc = collections.Counter(tuple(m[1]) for m in members)
    out = []
    for i, (v, w) in enumerate(members):
        f = []
        if i == 0: f.append('P_first')
        if i == n - 1: f.append('P_last')
        if i == 1: f.append('P_second')
        if i == n - 2: f.append('P_penult')
        f += ['S_' + x for x in set(w)]
        if w:
            f += ['F_' + w[0], 'L_' + w[-1]]
        f.append('N0' if not w else ('N1' if len(w) == 1 else ('N2' if len(w) == 2 else 'N3p')))
        if v == max(vals): f.append('Z_max')
        if v == min(vals): f.append('Z_min')
        if v == 1: f.append('Z_one')
        if 2 * v > tot: f.append('Z_big')
        if vc[v] > 1: f.append('D_val')
        if sc[tuple(w)] > 1 and w: f.append('D_sig')
        if w and tot_words and w[0] == tot_words[0]: f.append('T_share')
        else: f.append('T_none')
        if w and tot_words and w[-1] == tot_words[-1]: f.append('T_last')
        out.append(sorted(set(f)))
    return out


def build_ur(cache=os.path.join(CK, 'ur3_szunigin.json')):
    if os.path.exists(cache):
        return json.load(open(cache))
    txt = open(os.path.join(SCRATCH, 'cdli.atf'), encoding='utf-8', errors='replace').read()
    docs = re.split(r'\n(?=&P)', txt)
    out = []
    for d in docs:
        if 'szu-nigin' not in d or 'lang sux' not in d[:200]:
            continue
        if re.search(r'\n\$ .*(broken|missing)', d):
            continue
        lines = []
        for raw in d.split('\n'):
            m = LINE.match(raw.strip())
            if m:
                if '...' in m.group(2) and NUMSTART.match(m.group(2).replace('szu-nigin2 ', '').lstrip('[')):
                    lines.append('BROKEN'); continue
                t = re.sub(r'[#?!\[\]]', '', m.group(2)).strip()
                t = re.sub(r'^szunigin\b', 'szu-nigin2', t)
                lines.append(t)
        mem, tots, bad, phase = [], [], False, 0
        if 'BROKEN' in lines:
            continue
        for t in lines:
            if t.startswith('szu-nigin'):
                phase = 1
                p = parse_ur_line(t)
                if p is None or p[3] != 'n':
                    bad = True; break
                tots.append((p[0], p[2].split()))
                continue
            if phase == 1:
                break                      # stop after the total block
            if not NUMSTART.match(t):
                continue
            p = parse_ur_line(t)
            if p is None or p[3] != 'n':
                bad = True; break
            mem.append((p[0], [w for w in p[2].split() if not NUMSTART.match(w)]))
        if bad or not tots or not (2 <= len(mem) <= 40):
            continue
        for j, (tv, tw) in enumerate(tots):
            F = feats_ur(mem, tw)
            h = [{'cls': 'ur', 'T': [str(tv)], 'E': [[str(m[0])] for m in mem], 'F': F}]
            out.append({'id': '%s.%d' % (d[1:8], j), 'tier': 0, 'ntot': len(tots),
                        'tword': tw[0] if tw else '', 'hyps': [h]})
    json.dump(out, open(cache, 'w'))
    return out


if __name__ == '__main__':
    C = build_ur()
    print(len(C), collections.Counter(min(c['ntot'], 3) for c in C))


def build_balanced():
    """Second Ur III control, with a known exclusion: in the 21 balanced accounts
    of pe24 (CDLI), the debit summary line (SEC_D + SUM, 'sag-nig2-gur11') is the
    total of the debit entries only. Target = that line; members = every other
    numeric line of the account (debit entries, credit entries, credit summaries,
    the final la2-ia3). Known rule: the credit section is left out, i.e.
    EX(S_SECC) (equivalently KEEP(S_SECD); credit summaries are inside SEC_C).
    Texts with != 1 debit summary line are skipped. Features: section and summary
    tags as S_ features, first word W_ as S_ (decoys), positions, sizes, dups."""
    C = json.load(open(os.path.join(CK, '..', 'pe24_ckpt', 'ur3_balanced.json')))
    out = []
    for c in C:
        p = c['hyps'][0][0]
        mem = [(Fr(e[0]), m) for e, m in zip(p['E'], p['mk'])]
        mem.append((Fr(p['T'][0]), ['SEC_C', 'LA2', 'W_la2']))
        ds = [i for i, (v, m) in enumerate(mem) if 'SEC_D' in m and 'SUM' in m]
        if len(ds) != 1:
            continue
        tv = mem[ds[0]][0]
        rest = [(v, [x.replace('SEC_', 'SEC').replace('W_', '') for x in m]) for j, (v, m) in enumerate(mem) if j != ds[0]]
        F = feats_ur(rest, [])
        F = [sorted([f for f in fl if f != 'T_none']) for fl in F]
        out.append({'id': c['id'], 'tier': 0, 'hyps': [[{'cls': 'ur', 'T': [str(tv)],
                    'E': [[str(v)] for v, _ in rest], 'F': F}]]})
    return out
