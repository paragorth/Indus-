#!/usr/bin/env python3
"""pe60 cycle 2: score a photo transliteration (made blind) against the CDLI ATF of the same tablets.

Usage: pe60_score_read.py MY.atf TRUTH.atf [--json out.json]
Metrics per tablet and pooled:
  E   numeric entries (lines with numerals): mine vs true count
  NUM numeral tokens expanded by code (e.g. 3(N01) -> 3 tokens N01): precision / recall / F1 over the multiset
  SYS tablet system class (CAP if any capacity code, else count/other)
  TOT reverse numeric line present (yes / no)
  SEQ line-level: i-th numeric entry of mine has exactly the numerals of the i-th true entry
  SGN non-numeric signs named (not x): precision / recall over base-sign multisets
Controls: (a) a 'prior reader' that writes, for every tablet, the corpus-median tablet (2 entries of 1(N01)... see
prior()); (b) my readings shuffled across tablets (same readings, wrong tablets), 1000 permutations.
"""
import sys, os, re, json, random
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from build_corpus import parse_side  # noqa: E402

CAP = {'N39B', 'N30C', 'N24', 'N30D', 'N39C', 'N39A', 'N28', 'N29B', 'N39N', 'N24A', 'N28C', 'N29A', 'N30A', 'N26', 'N27'}


def blocks(path):
    t = open(path, encoding='utf8').read()
    return {b[1:8]: b for b in re.split(r'\n(?=&P)', t) if b.startswith('&P')}


def norm(c):
    c = c.replace('N1@', 'N01@').split('@')[0]
    c = {'N1': 'N01', 'N8A': 'N08A', 'N8B': 'N08B', 'N08': 'N08A'}.get(c, c)
    return c


def parse(b):
    surf = 'obverse'
    L = []
    for raw in b.splitlines()[1:]:
        s = raw.strip()
        if s.startswith('@'):
            kw = s[1:].split()
            if kw and kw[0] in ('obverse', 'reverse', 'top', 'bottom', 'left', 'right', 'edge'):
                surf = kw[0]
            continue
        m = re.match(r"^(\S+?)\.\s+(.*)$", s)
        if not m or s.startswith('#') or s.startswith('$'):
            continue
        body = m.group(2)
        left, right = (body.split(',', 1) + [''])[:2] if ',' in body else (body, '')
        s1, n1, _ = parse_side(left)
        s2, n2, _ = parse_side(right)
        nums = [(n, norm(c)) for n, c in n1 + n2 if isinstance(n, int)]
        signs = [re.sub(r'~[A-Za-z0-9]+', '', x) for x in s1 + s2 if x.startswith('M') or x.startswith('|')]
        L.append({'surf': surf, 'signs': signs, 'nums': nums})
    return L


def feats(L):
    ent = [l for l in L if l['nums']]
    tok = Counter()
    for l in ent:
        for n, c in l['nums']:
            tok[c] += n
    sg = Counter(s for l in L for s in l['signs'])
    return {'E': len(ent), 'tok': tok, 'sys': 'CAP' if any(c in CAP for c in tok) else 'CNT',
            'tot': any(l['surf'] == 'reverse' for l in ent),
            'seq': [tuple(sorted(sum((Counter({c: n}) for n, c in l['nums']), Counter()).items())) for l in ent],
            'sg': sg}


def ms_overlap(a, b):
    return sum((a & b).values()), sum(a.values()), sum(b.values())


def score(mine, truth, ids):
    agg = Counter()
    per = {}
    for p in ids:
        a, b = feats(mine[p]), feats(truth[p])
        o, na, nb = ms_overlap(a['tok'], b['tok'])
        so, sa, sb = ms_overlap(a['sg'], b['sg'])
        seq = sum(1 for x, y in zip(a['seq'], b['seq']) if x == y)
        agg.update({'num_hit': o, 'num_mine': na, 'num_true': nb, 'E_abs': abs(a['E'] - b['E']), 'E_true': b['E'],
                    'sys_ok': a['sys'] == b['sys'], 'tot_ok': a['tot'] == b['tot'], 'seq_hit': seq,
                    'sg_hit': so, 'sg_mine': sa, 'sg_true': sb, 'n': 1})
        per[p] = {'E': [a['E'], b['E']], 'num': [o, na, nb], 'sys': [a['sys'], b['sys']], 'tot': [a['tot'], b['tot']],
                  'seq': [seq, b['E']], 'signs': [so, sa, sb], 'true_tok': dict(b['tok']), 'my_tok': dict(a['tok'])}
    P = agg['num_hit'] / max(1, agg['num_mine']); R = agg['num_hit'] / max(1, agg['num_true'])
    out = {'tablets': agg['n'], 'numeral_precision': round(P, 3), 'numeral_recall': round(R, 3),
           'numeral_F1': round(2 * P * R / max(1e-9, P + R), 3),
           'entries_mean_abs_err': round(agg['E_abs'] / agg['n'], 2), 'entries_true_total': agg['E_true'],
           'system_correct': f"{agg['sys_ok']}/{agg['n']}", 'total_line_correct': f"{agg['tot_ok']}/{agg['n']}",
           'line_exact_numerals': f"{agg['seq_hit']}/{agg['E_true']}",
           'sign_precision': round(agg['sg_hit'] / max(1, agg['sg_mine']), 3),
           'sign_recall': round(agg['sg_hit'] / max(1, agg['sg_true']), 3), 'per': per}
    return out


def main():
    mine_b, truth_b = blocks(sys.argv[1]), blocks(sys.argv[2])
    ids = [p for p in mine_b if p in truth_b]
    mine = {p: parse(mine_b[p]) for p in ids}
    truth = {p: parse(truth_b[p]) for p in ids}
    res = score(mine, truth, ids)
    # control (b): readings permuted across tablets
    rng = random.Random(6060)
    f1s, seqs = [], []
    for _ in range(1000):
        perm = ids[:]
        rng.shuffle(perm)
        sh = {p: mine[q] for p, q in zip(ids, perm)}
        r = score(sh, truth, ids)
        f1s.append(r['numeral_F1']); seqs.append(int(r['line_exact_numerals'].split('/')[0]))
    res['control_shuffled_F1_mean'] = round(sum(f1s) / len(f1s), 3)
    res['control_shuffled_F1_p'] = round(sum(f >= res['numeral_F1'] for f in f1s) / len(f1s), 3)
    res['control_shuffled_lineexact_mean'] = round(sum(seqs) / len(seqs), 2)
    # control (a): constant prior reader (every tablet: 3 entries of 1(N01))
    pri = {p: [{'surf': 'obverse', 'signs': [], 'nums': [(1, 'N01')]}] * 3 for p in ids}
    rp = score(pri, truth, ids)
    res['control_prior_F1'] = rp['numeral_F1']; res['control_prior_lineexact'] = rp['line_exact_numerals']
    if '--json' in sys.argv:
        json.dump(res, open(sys.argv[sys.argv.index('--json') + 1], 'w'), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != 'per'}, indent=1))
    for p, v in res['per'].items():
        print(p, v)


if __name__ == '__main__':
    main()
