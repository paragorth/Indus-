"""pe27 cycle 5: the herd 'column-ration' lists MDP 17,085 (P008283) and 17,097 (P008295).
Each block = herd line |M362+X| (count) followed by M269/M260/M367~a/M362~a lines, then M106
(plain, N01 count) or M106~a (capacity), then M009, M206~g, M102~e, M309~a.
Hypotheses found by the cycle 1-4 scans (P008283 had a 5-pair exact rate list at M106 = herd):
  H1  plain M106 = herd count (1 per head; with a ~a sub-herd line, herd + sub-herd)
  H2  M106~a capacity = r x herd count, one r for all blocks.
Test: exact match counts per block; r scanned over every ratio the data allow (search-corrected);
null = M106 values re-dealt among the blocks of both tablets (100,000 permutations).  Also: which
capacity value sets make H2 hold, and the arithmetic of the written M106 total on 17,097."""
import json, os, sys, re, itertools
import numpy as np
from fractions import Fraction as Fr
from collections import Counter
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pe27_common import CK, CAPV, load  # noqa

T = {t['id']: t for t in load()}
CAPU = {'N39C': 'c', 'N30D': 'd', 'N30C': 'C', 'N24': 'G', 'N39B': 'B', 'N01': 'U', 'N14': 'T'}


def clean_nums(l):
    tail = l['raw'].split(',')[-1]
    if '[' in tail or '...' in tail or not l['numerals']:
        return None
    if any(not isinstance(n, int) or c.startswith('n') for n, c in l['numerals']):
        return None
    return l['numerals']


def blocks(pid):
    out = []
    cur = None
    for l in T[pid]['lines']:
        if l['surface'] != 'obverse':
            continue
        sg = [s for s in l['signs'] if s.startswith('M') or s.startswith('|')]
        head = sg[0] if sg else ''
        if 'M362' in head and '~a' not in head or ('M362' in ''.join(sg) and len(sg) >= 1 and 'M362~a' not in sg):
            cur = {'herd': None, 'sub': [], 'm106': None, 'm106a': None, 'raw': [l['raw']]}
            out.append(cur)
            n = clean_nums(l)
            if n and all(c in ('N01', 'N14') for _, c in n):
                cur['herd'] = sum(k * (10 if c == 'N14' else 1) for k, c in n)
            continue
        if cur is None:
            continue
        cur['raw'].append(l['raw'])
        n = clean_nums(l)
        if not sg:
            continue
        s0 = re.sub(r'#|\?', '', sg[0])
        if s0 in ('M367~a', 'M362~a'):
            cur['sub'].append(sum(k * (10 if c == 'N14' else 1) for k, c in n) if n else None)
        elif s0 == 'M106':
            cur['m106'] = sum(k * (10 if c == 'N14' else 1) for k, c in n) if n and all(c in ('N01', 'N14') for _, c in n) else 'x'
        elif s0 == 'M106~a':
            cur['m106a'] = n if n else 'x'
    return out


def capval(n, vs):
    return sum(k * vs[c] for k, c in n) if all(c in vs for _, c in n) else None


if __name__ == '__main__':
    res = {}
    B = blocks('P008283') + blocks('P008295')
    for b in B:
        print(b['herd'], b['sub'], b['m106'], b['m106a'] if b['m106a'] in (None, 'x') else ' '.join(f'{k}({c})' for k, c in b['m106a']))
    # H1
    h1 = [(b['herd'], b['sub'], b['m106']) for b in B if isinstance(b['m106'], int) and b['herd']]
    ok1 = sum(1 for h, s, m in h1 if m == h + sum(x for x in s if x)) if h1 else 0
    ok1b = sum(1 for h, s, m in h1 if m == h)
    # null for H1: permute m106 values among the eligible blocks
    rng = np.random.default_rng(0)
    vals = [m for _, _, m in h1]
    tgt = [h + sum(x for x in s if x) for h, s, _ in h1]
    nul = []
    for _ in range(100000):
        p = rng.permutation(len(vals))
        nul.append(sum(1 for i in range(len(vals)) if vals[p[i]] == tgt[i]))
    nul = np.array(nul)
    res['H1'] = {'n_blocks': len(h1), 'match_herd_plus_sub': ok1, 'match_herd_only': ok1b,
                 'null_mean': float(nul.mean()), 'p': float(((nul >= ok1).sum() + 1) / (len(nul) + 1)), 'blocks': h1}
    print('H1', res['H1'])
    # H2 with every value set; r scanned
    h2 = [(b['herd'], b['m106a']) for b in B if b['herd'] and isinstance(b['m106a'], list)]
    res['H2'] = {}
    for vsn, vs in CAPV.items():
        caps = [capval(n, vs) for _, n in h2]
        herds = [h for h, _ in h2]
        ratios = Counter(Fr(c, h) for c, h in zip(caps, herds) if c)
        rbest, kbest = ratios.most_common(1)[0]
        # null: permute capacity values among blocks; statistic = max over r of matches
        nul = []
        for _ in range(20000):
            p = rng.permutation(len(caps))
            rr = Counter(Fr(caps[p[i]], herds[i]) for i in range(len(caps)) if caps[p[i]])
            nul.append(rr.most_common(1)[0][1])
        nul = np.array(nul)
        res['H2'][vsn] = {'n_blocks': len(h2), 'best_r_N39C': str(rbest), 'matches': kbest,
                          'null_mean_max': float(nul.mean()), 'p_corr': float(((nul >= kbest).sum() + 1) / (len(nul) + 1)),
                          'pairs': [(h, c) for h, c in zip(herds, caps)]}
        print('H2', vsn, res['H2'][vsn])
    # unit-free version: which integer unit ratios N30C:N24:N39B make cap = herd * N30C?
    sols = []
    for g in range(1, 9):
        for bb in range(1, 13):
            vs = {'N30C': 1, 'N24': g, 'N39B': bb, 'N30D': Fr(1, 2), 'N39C': Fr(1, 4), 'N01': 5 * bb, 'N14': 30 * bb}
            caps = [capval(n, vs) for _, n in h2]
            m = sum(1 for (h, _), c in zip(h2, caps) if c == h)
            sols.append((m, g, bb))
    sols.sort(reverse=True)
    res['unit_solutions_top'] = sols[:6]
    print('units (matches, N24 in N30C, N39B in N30C):', sols[:6])
    # 17,097 reverse: M106 total 4(N14) 2(N01) = 42 vs plain-M106 blocks
    b97 = blocks('P008295')
    seen = sum(b['m106'] for b in b97 if isinstance(b['m106'], int))
    res['total_97'] = {'written_total': 42, 'visible_plain_sum': seen,
                       'blocks_without_any_M106_reading': [(b['herd'], b['raw'][0]) for b in b97 if b['m106'] is None and b['m106a'] is None]}
    print('17,097 M106 total', res['total_97'])
    # 17,085 reverse: M362 total 6(N14) 5(N01) = 65; M106~a total 1(N01) 1(N39B) 1(N24) = 39 N30C (a-priori units)
    b85 = blocks('P008283')
    herds = [b['herd'] for b in b85]
    # H1 fills a broken herd from its plain M106 (block with herd None, no sub-herd, M106 int)
    filled = [b['m106'] if b['herd'] is None and isinstance(b['m106'], int) and not b['sub'] else b['herd'] for b in b85]
    known = sum(h for h in filled if h)
    unknown = [i for i, h in enumerate(filled) if not h]
    a_known = [(b['herd'], b['m106a']) for b in b85 if isinstance(b['m106a'], list)]
    a_units = sum(capval(n, {'N30C': 1, 'N24': 3, 'N39B': 6, 'N01': 30, 'N30D': Fr(1, 2), 'N39C': Fr(1, 4)}) for _, n in a_known)
    a_marked = [b['herd'] for b in b85 if b['m106a'] == 'x']
    res['total_85'] = {'herd_total': 65, 'known_herds_after_H1_fill': known, 'unknown_blocks': len(unknown),
                       'unknown_herd_sum': 65 - known, 'M106a_written_units_N30C': str(a_units),
                       'M106a_blocks_with_broken_value_herds': a_marked, 'M106a_total_N30C': 39,
                       'note': 'M312 block carries 1(N24) 2(N30C) on the M269 line; counted as its M106~a amount'}
    print('17,085', res['total_85'])
    json.dump(res, open(os.path.join(CK, 'cycle5.json'), 'w'), indent=1, default=str)
