#!/usr/bin/env python3
"""LA-6 cycle 3: flows. Where does each commodity sit relative to KU-RO totals and headings?

(a) Zones on tablets with a KU-RO/PO-TO-KU-RO: entries before the first total, the total
    itself, entries after it. For each commodity, number of entries in the 'after' zone.
    Null: zone labels permuted among the entries of each tablet (keeps every tablet's
    composition and its number of after-entries), 10,000 runs.
(b) Which commodity does the KU-RO count? For each total, the commodities of the entries
    before it; does the total match (within 1) the sum of the entries of one commodity, of
    all entries, or of none.
(c) The after-total package: its size against the KU-RO number (levy hypothesis: package
    proportional to the total). Spearman rho, exact permutation P.
(d) Heading commodities: a main commodity logogram in the first line without a number,
    against the commodities quantified in the body of the same tablet. Null: headings
    permuted across tablets that have one, 10,000 runs; statistic = headings whose
    commodity is absent from the body.
"""
import sys, os, json, random, itertools
from collections import defaultdict, Counter
sys.path.insert(0, os.path.dirname(__file__))
from la6_common import la_entries, base_of, D, MAIN

random.seed(63)
R = 10000


def spearman(x, y):
    def rank(v):
        o = sorted(range(len(v)), key=lambda i: v[i]); r = [0] * len(v)
        for k, i in enumerate(o): r[i] = k
        return r
    a, b = rank(x), rank(y)
    n = len(x); ma = sum(a) / n; mb = sum(b) / n
    num = sum((p - ma) * (q - mb) for p, q in zip(a, b))
    den = (sum((p - ma) ** 2 for p in a) * sum((q - mb) ** 2 for q in b)) ** .5
    return num / den if den else 0


def main():
    out = ['# LA-6 cycle 3: flows relative to totals and headings']
    E = la_entries()
    bydoc = defaultdict(list)
    for e in E: bydoc[e['doc']].append(e)
    tot_docs = {d: es for d, es in bydoc.items() if any(e['role'] in ('total', 'grand') for e in es)}
    # (a)
    ents = [(d, e) for d, es in tot_docs.items() for e in es if e['role'] in ('entry', 'post', 'head') and e['com']]
    coms = Counter(c for d, e in ents for c in e['com'])
    obs_after = Counter(c for d, e in ents if e['zone'] == 'after' for c in e['com'])
    obs_before = Counter(c for d, e in ents if e['zone'] == 'before' for c in e['com'])
    perdoc = defaultdict(list)
    for d, e in ents: perdoc[d].append(e)
    ge = Counter(); mean = Counter()
    for _ in range(R):
        cnt = Counter()
        for d, es in perdoc.items():
            z = [e['zone'] for e in es]; random.shuffle(z)
            for e, zz in zip(es, z):
                if zz == 'after':
                    for c in e['com']: cnt[c] += 1
        for c in coms:
            ge[c] += cnt[c] >= obs_after[c]; mean[c] += cnt[c]
    out.append(f'## (a) {len(tot_docs)} tablets with a total; {len(ents)} commodity entries')
    out.append('commodity | before | after | after expected | P(after >= obs)')
    for c, n in coms.most_common():
        if n < 3: continue
        out.append(f'{c} | {obs_before[c]} | {obs_after[c]} | {mean[c]/R:.2f} | {(ge[c]+1)/(R+1):.4f}')
    out.append('after-zone entries:')
    for d, es in sorted(tot_docs.items()):
        aft = [e for e in es if e['zone'] == 'after' and (e['com'] or e['bare'])]
        tot = [e for e in es if e['role'] in ('total', 'grand')][0]
        tv = sum(tot['bare']) + sum(tot['com'].values())
        bef = Counter(c for e in es if e['zone'] == 'before' for c in e['com'])
        out.append(f'  {d}: before {dict(bef)} | total {float(tv):g} {list(tot["com"])} | after ' +
                   '; '.join(f"{e['label']} {({c: round(float(v), 2) for c, v in e['com'].items()})}{' bare ' + str([float(x) for x in e['bare']]) if e['bare'] else ''}" for e in aft))
    # (b)
    out.append('\n## (b) what the first KU-RO counts')
    kinds = Counter()
    for d, es in sorted(tot_docs.items()):
        tot = [e for e in es if e['role'] in ('total', 'grand')][0]
        tv = float(sum(tot['bare']) + sum(tot['com'].values()))
        bef = [e for e in es if e['zone'] == 'before']
        allsum = float(sum(sum(e['com'].values()) + sum(e['bare']) for e in bef))
        per = defaultdict(float)
        for e in bef:
            for c, v in e['com'].items(): per[c] += float(v)
            if e['bare']: per['(bare)'] += float(sum(e['bare']))
        m = [c for c, v in per.items() if abs(v - tv) <= 1]
        k = 'all' if abs(allsum - tv) <= 1 else ('one:' + ','.join(m) if m else 'none')
        kinds[k.split(':')[0]] += 1
        out.append(f'  {d}: total {tv:g}; all-entries {allsum:g}; per commodity {({c: round(v, 2) for c, v in per.items()})} -> {k}')
    out.append(f'  summary {dict(kinds)}')
    # (c) package after total
    out.append('\n## (c) after-total package vs total (levy test)')
    rows = []
    for d, es in sorted(tot_docs.items()):
        tot = [e for e in es if e['role'] in ('total', 'grand')][0]
        tv = float(sum(tot['bare']) + sum(tot['com'].values()))
        aft = defaultdict(float)
        for e in es:
            if e['zone'] == 'after' and e['role'] in ('entry', 'post'):
                for c, v in e['com'].items(): aft[c] += float(v)
        if tv > 0 and any(c in aft for c in ('CYP', 'NI', 'VIN')):
            rows.append((d, tv, aft))
    for d, tv, aft in rows:
        out.append(f'  {d}: total {tv:g}, ' + ', '.join(f'{c} {v:.3g} ({v/tv*100:.1f}%)' for c, v in aft.items()))
    for c in ('CYP', 'NI', 'VIN'):
        xs = [(tv, a[c]) for d, tv, a in rows if c in a]
        if len(xs) >= 4:
            x, y = [p[0] for p in xs], [p[1] for p in xs]
            rho = spearman(x, y)
            perms = list(itertools.permutations(y)) if len(y) <= 8 else None
            if perms:
                p = sum(spearman(x, list(q)) >= rho - 1e-9 for q in perms) / len(perms)
            else:
                p = sum(spearman(x, random.sample(y, len(y))) >= rho - 1e-9 for _ in range(R)) / R
            share = [b / a for a, b in xs]
            out.append(f'  {c}: n={len(xs)} rho={rho:.2f} one-sided P={p:.3f}; share of total min {min(share)*100:.1f}% max {max(share)*100:.1f}%')
    # (d) headings
    C = json.load(open(os.path.join(D, 'corpus.json')))
    heads = []
    for ins in C:
        toks = ins['tokens']
        first = []
        for i, t in enumerate(toks):
            if t['t'] == 'nl': break
            first.append((i, t))
        h = None
        for i, t in first:
            if t['t'] == 'logo':
                b = base_of(t['v'])
                nx = toks[i + 1] if i + 1 < len(toks) else None
                if b in MAIN and (nx is None or nx['t'] != 'num'):
                    h = b
        if h is None: continue
        body = Counter()
        for e in bydoc[ins['id']]:
            for c in e['com']: body[c] += 1
        if body:
            heads.append((ins['id'], h, set(body)))
    absent = sum(1 for d, h, b in heads if h not in b)
    hs = [h for _, h, _ in heads]
    ge = 0; m = 0
    for _ in range(R):
        random.shuffle(hs)
        a = sum(1 for (d, _h, b), h in zip(heads, hs) if h not in b)
        ge += a <= absent; m += a
    out.append(f'\n## (d) heading commodity (no number, first line) vs body: {len(heads)} tablets; heading absent from body {absent} vs null {m/R:.1f}, P(<=)={(ge+1)/(R+1):.4f}')
    for d, h, b in heads:
        out.append(f'  {d}: head {h} | body {sorted(b)}')
    txt = '\n'.join(out)
    print(txt)
    open(os.path.join(D, 'la6_flows.out'), 'w').write(txt + '\n')


if __name__ == '__main__':
    main()
