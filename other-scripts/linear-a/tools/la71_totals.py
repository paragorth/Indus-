#!/usr/bin/env python3
"""la71 cycle 2: KU-RO totals (FINDINGS 2a), KI-RO smaller-than-entry, HT 127b running total and the
la70 'KI reduced amount' line, in three corpus versions (all / rd = read+damaged / read only).

Same section rule as tools/totals_test.py (entries = quantities since the previous KU-RO / KI-RO /
PO-TO-KU-RO or the top; total = first quantity within 3 tokens after KU-RO).  In versions rd/read a
section is TESTABLE only if no '#R' placeholder (a removed token) sits inside its span or is its total;
an untestable section is dropped, never counted as a miss.
Control: totals shuffled across the testable sections (20,000), as in 2a.
usage: la71_totals.py  -> prints rows, writes data/la71_ckpt/c2_totals.json"""
import os, sys, json, random
from fractions import Fraction as Fr
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import la71_parse as P

MARK = {'KU-RO', 'KI-RO', 'PO-TO-KU-RO'}


def wstr(t): return '-'.join(t['s']) if t['t'] == 'word' else None


def removed(x, arith=True):
    """a placeholder that matters for arithmetic: a removed number, or a removed section word."""
    if not (x['t'] == 'unk' and x.get('v') == '#R'): return False
    if not arith: return True
    o = x.get('o', {})
    return x.get('k') == 'num' or (o.get('t') == 'word' and '-'.join(o.get('s', [])) in MARK)


def sections(C, term):
    out = []
    for ins in C:
        T = ins['tokens']; last = 0
        for i, t in enumerate(T):
            w = wstr(t)
            if w in MARK:
                if w == term:
                    tot = None; ti = None; bad = False
                    for j in range(i + 1, min(i + 4, len(T))):
                        if removed(T[j]): bad = True; break
                        if T[j]['t'] == 'num': tot = T[j]; ti = j; break
                        if T[j]['t'] == 'word': break
                    span = T[last:i]
                    bad = bad or any(removed(x) for x in span)
                    entries = [x for x in span if x['t'] == 'num']
                    out.append({'id': ins['id'], 'tot': tot, 'entries': entries, 'bad': bad})
                last = i + 1
    return out


def run(ver, C):
    rng = random.Random(71)
    ku_all = [s for s in sections(C, 'KU-RO') if s['entries'] and s['tot']]
    ku = [s for s in ku_all if not s['bad']]
    sums = [sum(e['v'] for e in s['entries']) for s in ku]; tots = [s['tot']['v'] for s in ku]
    obs = sum(a == b for a, b in zip(sums, tots))
    null = []
    for _ in range(20000):
        p = tots[:]; rng.shuffle(p); null.append(sum(a == b for a, b in zip(sums, p)))
    P_ = (sum(n >= obs for n in null) + 1) / 20001
    hits = sorted(s['id'] for s, a, b in zip(ku, sums, tots) if a == b)
    near = sum(1 for a, b in zip(sums, tots) if abs(a - b) <= 1)
    # KI-RO number smaller than entry before it
    ki = [s for s in sections(C, 'KI-RO') if s['tot'] and s['entries'] and not s['bad']]
    sm = sum(1 for s in ki if s['tot']['v'] + 0.0 < s['entries'][-1]['v'])
    return {'ver': ver, 'sections_total': len(ku_all), 'testable': len(ku), 'exact': obs, 'hits': hits,
            'within1': near, 'null_mean': sum(null) / len(null), 'null_max': max(null), 'P': P_,
            'kiro_smaller': [sm, len(ki)]}


def ht127b(C):
    d = next(x for x in C if x['id'] == 'HT127b')
    return ' '.join(('#R' if t['t'] == 'unk' and t.get('v') == '#R' else
                     ('-'.join(t['s']) if t['t'] == 'word' else str(t['v']))) for t in d['tokens'] if t['t'] not in ('nl', 'div'))


def ki_reduced(C):
    """la70/la66 line: KI before a number marks a reduced amount (KI amount < the entry above)."""
    n = k = 0; bad = 0
    for d in C:
        T = [t for t in d['tokens'] if t['t'] not in ('nl', 'div')]
        for i, t in enumerate(T):
            if t['t'] == 'word' and t['s'] == ['KI'] and i + 1 < len(T) and T[i + 1]['t'] == 'num':
                prev = next((T[j] for j in range(i - 1, -1, -1) if T[j]['t'] in ('num', 'unk')), None)
                if prev is None: continue
                if prev['t'] == 'unk': bad += 1; continue  # removed or illegible amount above
                n += 1; k += T[i + 1]['v'] < prev['v']
    return k, n, bad


def main():
    res = {}
    for ver in ('all', 'rd', 'read'):
        C = P.load(ver)
        r = run(ver, C); r['ht127b'] = ht127b(C); r['ki'] = ki_reduced(C)
        res[ver] = r
        print(ver, json.dumps({k: v for k, v in r.items()}, ensure_ascii=False))
    json.dump(res, open(os.path.join(P.CK, 'c2_totals.json'), 'w'), ensure_ascii=False, indent=1)


if __name__ == '__main__':
    main()
