#!/usr/bin/env python3
"""LA-6 cycle 2: rounding and unit preferences per commodity (numbers only).

For every quantity written after a commodity (entries and heads; totals excluded):
  frac%  share carrying a fraction sign (sub-unit used)
  ones%  share with integer part 0 or 1 (the unit is large relative to one entry)
  r5     share of integers >= 5 that are multiples of 5
  r10    share of integers >= 10 that are multiples of 10
  med    median integer part
Null: commodity labels permuted across all quantity tokens (the number pool is kept),
5,000 runs; two-sided P per commodity and metric.
Positive control: Linear B. Counted goods (VIR, MUL, OVIS, CAP, SUS, BOS) never carry a
sub-unit; measured goods do. Linear A internal control: VIR (persons) should behave as counted.
"""
import sys, os, re, json, random
from collections import defaultdict, Counter
sys.path.insert(0, os.path.dirname(__file__))
from la6_common import la_entries, D, LB_COM, NUM

random.seed(62)
R = 5000
LA_COMS = ['GRA', 'OLE', 'OLIV', 'VIN', 'NI', 'CYP', '*304', 'VIR', '*308', '*86', '*305', 'AROM', 'HIDE']


def la_tokens():
    E = la_entries()
    rows = []
    for e in E:
        if e['role'] in ('total', 'grand', 'deficit'): continue
        for c, iv, fr in e['raw']:
            rows.append((c if c in LA_COMS else 'other', iv, bool(fr)))
        for v in e['bare']:
            rows.append(('bare', int(v), v != int(v)))
    return rows


def lb_tokens():
    rows = []
    for line in open(os.path.join(D, 'damos_items.jsonl')):
        d = json.loads(line)
        for ln in (d.get('content') or '').split('\n'):
            toks = ln.split()
            i = 0
            while i < len(toks):
                b = toks[i].strip('[]⟦⟧').split('+')[0]
                if b in LB_COM:
                    iv, sub, j = 0, False, i + 1
                    if j < len(toks) and NUM.match(toks[j]):
                        iv = int(NUM.match(toks[j]).group(1)); j += 1
                    while j + 1 < len(toks) and toks[j].strip('[]') in ('T', 'V', 'Z', 'S', 'M', 'N', 'P', 'Q') and NUM.match(toks[j + 1]):
                        sub = True; j += 2
                    if iv or sub:
                        rows.append((b, iv, sub))
                    i = j
                else:
                    i += 1
    return rows


def metrics(rows):
    n = len(rows)
    if n == 0: return None
    fr = sum(f for _, _, f in rows) / n
    ones = sum(iv <= 1 for _, iv, _ in rows) / n
    g5 = [iv for _, iv, _ in rows if iv >= 5]
    g10 = [iv for _, iv, _ in rows if iv >= 10]
    r5 = sum(x % 5 == 0 for x in g5) / len(g5) if g5 else float('nan')
    r10 = sum(x % 10 == 0 for x in g10) / len(g10) if g10 else float('nan')
    ivs = sorted(iv for _, iv, _ in rows)
    return {'n': n, 'frac': fr, 'ones': ones, 'r5': r5, 'r10': r10, 'med': ivs[n // 2], 'n5': len(g5), 'n10': len(g10)}


def table(rows, coms, label, out):
    by = defaultdict(list)
    for r in rows: by[r[0]].append(r)
    obs = {c: metrics(by[c]) for c in coms if len(by[c]) >= 8}
    keys = ['frac', 'ones', 'r5', 'r10']
    ge = {c: {k: 0 for k in keys} for c in obs}
    le = {c: {k: 0 for k in keys} for c in obs}
    labs = [r[0] for r in rows]
    for _ in range(R):
        random.shuffle(labs)
        b2 = defaultdict(list)
        for l, r in zip(labs, rows): b2[l].append(r)
        for c in obs:
            m = metrics(b2[c])
            for k in keys:
                if m[k] != m[k] or obs[c][k] != obs[c][k]: continue
                ge[c][k] += m[k] >= obs[c][k]; le[c][k] += m[k] <= obs[c][k]
    out.append(f'## {label}: {len(rows)} quantities')
    out.append('commodity | n | median | frac% (P) | int<=1 % (P) | x5 among >=5 (n, P) | x10 among >=10 (n, P)')
    allm = metrics(rows)
    out.append(f"ALL | {allm['n']} | {allm['med']} | {allm['frac']*100:.0f} | {allm['ones']*100:.0f} | {allm['r5']*100:.0f} ({allm['n5']}) | {allm['r10']*100:.0f} ({allm['n10']})")
    for c in sorted(obs, key=lambda c: -obs[c]['n']):
        m = obs[c]
        p = {k: min(1, 2 * min(ge[c][k] + 1, le[c][k] + 1) / (R + 1)) for k in keys}
        out.append(f"{c} | {m['n']} | {m['med']} | {m['frac']*100:.0f} ({p['frac']:.3f}) | {m['ones']*100:.0f} ({p['ones']:.3f}) | "
                   f"{m['r5']*100:.0f} ({m['n5']}, {p['r5']:.3f}) | {m['r10']*100:.0f} ({m['n10']}, {p['r10']:.3f})")
    return obs


def main():
    out = ['# LA-6 cycle 2: rounding and unit preferences per commodity']
    la = la_tokens()
    table(la, LA_COMS + ['other', 'bare'], 'Linear A (entries, no totals)', out)
    # site split for the main goods
    lb = lb_tokens()
    lbc = ['GRA', 'HORD', 'FAR', 'OLE', 'VIN', 'NI', 'OLIV', 'CYP', 'AROM', 'VIR', 'MUL', 'OVIS', 'CAP', 'SUS', 'BOS', 'LANA', 'TELA', 'ME±RI', 'AES']
    table(lb, lbc, 'Linear B control (DAMOS)', out)
    txt = '\n'.join(out)
    print(txt)
    open(os.path.join(D, 'la6_units.out'), 'w').write(txt + '\n')


if __name__ == '__main__':
    main()
