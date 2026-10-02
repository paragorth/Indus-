#!/usr/bin/env python3
"""Tests 2a (KU-RO / KI-RO totals) and 2b (fraction values).

Section rule: entries for a KU-RO = all quantities after the previous
KU-RO / KI-RO / PO-TO-KU-RO (or the start of the inscription) and before the KU-RO.
Total = the first quantity within 3 tokens after KU-RO (logograms/dividers skipped).
"""
import json, os, random, itertools
from fractions import Fraction as Fr
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
C = json.load(open(os.path.join(HERE, '..', 'data', 'corpus.json')))
random.seed(1)

MARK = {'KU-RO', 'KI-RO', 'PO-TO-KU-RO'}

def wstr(t): return '-'.join(t['s']) if t['t'] == 'word' else None

def sections(term):
    out = []
    for ins in C:
        T = ins['tokens']
        last = 0
        for i, t in enumerate(T):
            w = wstr(t)
            if w in MARK:
                if w == term:
                    tot = None
                    for j in range(i + 1, min(i + 4, len(T))):
                        if T[j]['t'] == 'num': tot = T[j]; break
                        if T[j]['t'] == 'word': break
                    entries = [x for x in T[last:i] if x['t'] == 'num']
                    after = []
                    for x in T[i + 1:]:
                        if x['t'] == 'word' and wstr(x) in MARK: break
                        if x['t'] == 'num': after.append(x)
                    if tot is not None and after and after[0] is tot: after = after[1:]
                    out.append({'id': ins['id'], 'tot': tot, 'entries': entries, 'after': after})
                last = i + 1
    return out

# ---- fraction systems -------------------------------------------------------
SITE = {'J': Fr(1, 2), 'E': Fr(1, 4), 'F': Fr(1, 8), 'K': Fr(1, 16), 'D': Fr(1, 5), 'B': Fr(1, 3),
        'A': Fr(1, 6), 'H': Fr(1, 6), 'JE': Fr(3, 4), 'L2': Fr(3, 20), 'X': Fr(3, 10), 'Y': Fr(1, 4),
        'L': Fr(1, 10), 'L4': Fr(1, 40), 'L3': Fr(1, 30), 'L6': Fr(1, 60), 'W': Fr(1, 24), 'DD': Fr(1, 10)}
# NB: values for L, L3, L4, L6, W, DD are placeholders (the site gives none); they are rare.

def val(q, sysm, ints_only=False):
    v = Fr(q['v'])
    if ints_only: return v
    for f in q['frac']:
        v += sysm.get(f, Fr(0))
    return v

def balance(secs, sysm, ints_only=False):
    ok = 0; res = []
    for s in secs:
        S = sum((val(e, sysm, ints_only) for e in s['entries']), Fr(0))
        T = val(s['tot'], sysm, ints_only)
        res.append((s['id'], S, T))
        ok += (S == T)
    return ok, res

def main():
    ku = [s for s in sections('KU-RO') if s['entries'] and s['tot']]
    print('KU-RO sections with a total and >=1 entry:', len(ku))
    # integer-only and with SITE fractions
    for name, kw in [('integers only', dict(ints_only=True)), ('site fraction values', {})]:
        ok, res = balance(ku, SITE, **kw)
        near = sum(1 for _, S, T in res if abs(S - T) <= 1)
        pct = sum(1 for _, S, T in res if T and abs(S - T) / T <= 0.05)
        print(f'  [{name}] exact {ok}/{len(ku)}  |diff|<=1: {near}  within 5%: {pct}')
    ok, res = balance(ku, SITE)
    print('  detail (id, sum of entries, KU-RO total):')
    for r in res: print('   ', r[0], float(r[1]), float(r[2]), 'OK' if r[1] == r[2] else '')

    # control 1: shuffle totals across sections (integers only)
    sums = [sum(e['v'] for e in s['entries']) for s in ku]
    tots = [s['tot']['v'] for s in ku]
    obs = sum(a == b for a, b in zip(sums, tots))
    null = []
    for _ in range(20000):
        p = tots[:]; random.shuffle(p)
        null.append(sum(a == b for a, b in zip(sums, p)))
    print(f'  control (totals shuffled across tablets, ints): mean {sum(null)/len(null):.2f}, '
          f'max {max(null)}, P(null>=obs={obs}) = {sum(n >= obs for n in null)/len(null):.5f}')
    # control 2: random subset of entries equals the total?
    hits = 0; trials = 0
    for s in ku:
        n = len(s['entries'])
        if n < 2: continue
        for _ in range(200):
            k = random.randint(1, n - 1)
            sub = random.sample(s['entries'], k)
            hits += (sum(e['v'] for e in sub) == s['tot']['v']); trials += 1
    print(f'  control (random proper subset of entries == total): {hits/trials:.3f}')
    # where does it fail: entries > total or < total?
    over = sum(1 for a, b in zip(sums, tots) if a > b); under = sum(1 for a, b in zip(sums, tots) if a < b)
    print(f'  integer sums: over total {over}, under total {under}, equal {obs}')

    # ---- KI-RO -----------------------------------------------------------
    ki = sections('KI-RO')
    print('\nKI-RO occurrences:', len(ki), ' with a number right after:', sum(1 for s in ki if s['tot']))
    for s in ki:
        T = float(val(s['tot'], SITE)) if s['tot'] else None
        S = float(sum((val(e, SITE) for e in s['entries']), Fr(0)))
        A = float(sum((val(e, SITE) for e in s['after']), Fr(0)))
        prev = float(val(s['entries'][-1], SITE)) if s['entries'] else None
        print(f"   {s['id']:14s} KI-RO value={T}  sum before={S}  last before={prev}  sum after={A} (n after={len(s['after'])})")
    kt = [s for s in ki if s['tot'] and s['entries']]
    smaller = sum(1 for s in kt if val(s['tot'], SITE) < val(s['entries'][-1], SITE))
    print(f'  KI-RO value < the quantity just before it: {smaller}/{len(kt)}')
    kt2 = [s for s in ku if s['entries']]
    smaller2 = sum(1 for s in kt2 if val(s['tot'], SITE) < val(s['entries'][-1], SITE))
    print(f'  (compare KU-RO value < quantity just before it: {smaller2}/{len(kt2)})')
    nonum = [s['id'] for s in ki if not s['tot']]
    print('  KI-RO with no number (header use):', nonum)
    ones = [s['id'] for s in ki if s['after'] and all(e['v'] == 1 and not e['frac'] for e in s['after'][:5])]
    print('  KI-RO followed by a list of "1"s (person lists):', ones)

    # ---- 2b fractions ------------------------------------------------------
    print('\n2b. Fraction values')
    fr_secs = [s for s in ku if s['tot']['frac'] or any(e['frac'] for e in s['entries'])]
    print('  KU-RO sections that contain any fraction:', len(fr_secs), [s['id'] for s in fr_secs])
    int_ok = [s for s in fr_secs if sum(e['v'] for e in s['entries']) <= s['tot']['v'] <= sum(e['v'] for e in s['entries']) + sum(len(e['frac']) for e in s['entries'])]
    letters = sorted({f for s in fr_secs for q in s['entries'] + [s['tot']] for f in q['frac']})
    print('  letters involved:', letters)
    ok_site, _ = balance(fr_secs, SITE)
    print(f'  site values balance {ok_site}/{len(fr_secs)}')
    # Control: permute the values among the letters
    vals = [SITE[l] for l in letters]
    null = []
    for _ in range(20000):
        p = vals[:]; random.shuffle(p)
        null.append(balance(fr_secs, dict(zip(letters, p)))[0])
    print(f'  control (site values permuted among letters): mean {sum(null)/len(null):.2f}, '
          f'P(>= {ok_site}) = {sum(n >= ok_site for n in null)/len(null):.4f}')
    # Search: exhaustive over a grid for the 4 commonest letters, others fixed at site value
    grid = [Fr(1, n) for n in (2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 30)] + [Fr(3, 4), Fr(2, 3), Fr(3, 8)]
    top = [l for l, _ in Counter(f for s in fr_secs for q in s['entries'] + [s['tot']] for f in q['frac']).most_common(4)]
    best = []
    for combo in itertools.product(grid, repeat=len(top)):
        sysm = dict(SITE); sysm.update(zip(top, combo))
        best.append((balance(fr_secs, sysm)[0], combo))
    best.sort(key=lambda x: -x[0])
    mx = best[0][0]
    winners = [b for b in best if b[0] == mx]
    print(f'  grid search over {top}: best balances {mx}/{len(fr_secs)}; {len(winners)} of {len(best)} value-sets tie for best')
    for b in winners[:8]: print('    ', dict(zip(top, [str(x) for x in b[1]])))
    site_rank = sum(1 for b in best if b[0] > ok_site)
    print(f'  value-sets strictly better than site values: {site_rank}')
    # same-letter identity tests: J+J vs 1, JE vs J+E
    print('  JE (ligature) vs J and E separately: JE used', sum(1 for i in C for t in i['tokens'] if t['t']=='num' for f in t['frac'] if f=='JE'))

    print('  the balancing fraction section(s):', [r[0] for r in balance(fr_secs, SITE)[1] if r[1] == r[2]])

    # ---- 2a': contiguous-run rule ----------------------------------------
    # KU-RO total equals the sum of SOME contiguous run of the section's entries
    # (allows a header number or a second commodity). Control: same rule, totals shuffled.
    def run_hit(entries, T):
        v = [e['v'] for e in entries]
        for a in range(len(v)):
            s = 0
            for b in range(a, len(v)):
                s += v[b]
                if s == T and b > a: return True
        return False
    obs = sum(run_hit(s['entries'], s['tot']['v']) for s in ku)
    null = []
    for _ in range(5000):
        p = tots[:]; random.shuffle(p)
        null.append(sum(run_hit(s['entries'], t) for s, t in zip(ku, p)))
    print(f"\n2a'. KU-RO = sum of some contiguous run (>=2 entries, integers): {obs}/{len(ku)}; "
          f"control mean {sum(null)/len(null):.2f}, P(>=obs) = {sum(n >= obs for n in null)/len(null):.4f}")

    # ---- HT123+124a by commodity (hand-split: OLIV column and *308 column) ----
    oliv = [Fr(31), Fr(31) + SITE['J'], Fr(16), Fr(15)]
    s308 = [8 + SITE['E'], 8 + SITE['JE'], 4 + SITE['A'], 4 + SITE['E']]
    print(f"  HT123+124a OLIV column sum {float(sum(oliv))} vs KU-RO OLIV 93+J = 93.5 ;"
          f" *308 column sum {float(sum(s308))} vs KU-RO 25+H = {float(25 + SITE['H'])}")
    print('   *308 / OLIV ratios per line:', [round(float(b / a), 3) for a, b in zip(oliv, s308)])

    # ---- 2b': order of letters inside compound fractions -----------------
    comp = [t['frac'] for i in C for t in i['tokens'] if t['t'] == 'num' and len(t['frac']) >= 2]
    pairs = Counter()
    for f in comp:
        for a in range(len(f)):
            for b in range(a + 1, len(f)):
                if f[a] != f[b]: pairs[(f[a], f[b])] += 1
    print(f"\n2b'. compound fractions: {len(comp)}; ordered letter pairs:")
    seen = set()
    cons = inc = 0
    for (a, b), n in pairs.most_common():
        if (a, b) in seen: continue
        m = pairs.get((b, a), 0); seen.add((b, a))
        print(f'    {a} before {b}: {n}   {b} before {a}: {m}')
        cons += max(n, m); inc += min(n, m)
    print(f'  pair orders that follow the majority direction: {cons}/{cons+inc}')
    desc = sum(1 for (a, b), n in pairs.items() for _ in range(n) if SITE.get(a, 0) > SITE.get(b, 0))
    tot_pairs = sum(n for (a, b), n in pairs.items() if SITE.get(a, 0) != SITE.get(b, 0))
    print(f'  pairs written larger-first under site values: {desc}/{tot_pairs} (random order: 50%)')
    def consistency(comps):
        pc = Counter()
        for f in comps:
            for a in range(len(f)):
                for b in range(a + 1, len(f)):
                    if f[a] != f[b]: pc[(f[a], f[b])] += 1
        c = i_ = 0; done = set()
        for (a, b), n in pc.items():
            if (a, b) in done: continue
            m = pc.get((b, a), 0); done.add((b, a)); done.add((a, b))
            c += max(n, m); i_ += min(n, m)
        return c, i_
    null = []
    for _ in range(5000):
        sh = [random.sample(f, len(f)) for f in comp]
        c, i_ = consistency(sh)
        null.append(i_)
    print(f'  control (letters shuffled within each compound): mean minority-direction pairs '
          f'{sum(null)/len(null):.2f}; P(0 minority) = {sum(n == 0 for n in null)/len(null):.4f}')
    # topological order implied
    import collections
    g = collections.defaultdict(set); nodes = set()
    for (a, b) in pairs: g[a].add(b); nodes |= {a, b}
    indeg = {n: 0 for n in nodes}
    for a in g:
        for b in g[a]: indeg[b] += 1
    order = []; q = sorted(n for n in nodes if indeg[n] == 0)
    while q:
        n = q.pop(0); order.append(n)
        for b in sorted(g[n]):
            indeg[b] -= 1
            if indeg[b] == 0: q.append(b)
    print('  implied writing order (first-written to last-written), acyclic=%s:' % (len(order) == len(nodes)), order)
    print('  site values in that order:', [str(SITE.get(x)) for x in order])
    rep = Counter('+'.join(f) for f in comp if len(set(f)) < len(f))
    print('  repeated same letter in one quantity:', dict(rep))

if __name__ == '__main__':
    main()
