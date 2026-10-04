#!/usr/bin/env python3
"""LA-6 cycle 5: fixed allotments. Does the same word get the same quantity on different tablets?

For each word (entry label, 2+ signs, totals excluded) seen with a quantity on >= 2 tablets:
  a 'match' = two of its tablets where its first quantity is equal (same class, within 1%).
Statistic: number of words with at least one cross-tablet match; and total matching tablet pairs.
Nulls (2,000 runs):
  N1 shuffle within commodity class across the whole corpus (bare numbers are a class);
  N2 shuffle within tablet (the entry quantities of each tablet permuted among its entries):
     keeps each tablet's own standard amounts; tests whether the *word* carries its amount.
Linear B control: PY Ab (persons/places with MUL, GRA, NI) and KN Fs entries by word: same word, same amount.
"""
import sys, os, math, random, json, re
from collections import defaultdict, Counter
sys.path.insert(0, os.path.dirname(__file__))
from la6_common import la_entries, D, lb_docs
from la6_fingerprint import units

random.seed(66)
R = 2000


def stat(rec):
    """rec: word -> list of (doc, (cls, value))"""
    words = pairs = 0
    det = {}
    for w, l in rec.items():
        bydoc = {}
        for d, q in l:
            bydoc.setdefault(d, q)
        ds = list(bydoc)
        m = 0
        for i in range(len(ds)):
            for j in range(i + 1, len(ds)):
                a, b = bydoc[ds[i]], bydoc[ds[j]]
                if a[0] == b[0] and abs(a[1] - b[1]) <= 0.01 * max(a[1], b[1]):
                    m += 1
        words += m > 0; pairs += m
        if m: det[w] = sorted((d, q[1]) for d, q in bydoc.items())
    return words, pairs, det


def main():
    out = ['# LA-6 cycle 5: fixed allotments (same word, same quantity, different tablets)']
    E = la_entries()
    Q = [e for e in E if e['role'] in ('entry', 'post') and e['label'] and '-' in e['label'] and units(e)]
    ndoc = defaultdict(set)
    for e in Q: ndoc[e['label']].add(e['doc'])
    keepw = {w for w, s in ndoc.items() if len(s) >= 2}
    items = [(e['label'], e['doc'], units(e)[0]) for e in Q]
    rec = defaultdict(list)
    for w, d, q in items:
        if w in keepw: rec[w].append((d, q))
    ow, op, det = stat(rec)
    out.append(f'{len(keepw)} words (2+ signs) on >= 2 tablets')
    out.append(f'observed: {ow} words with a cross-tablet exact repeat; {op} matching tablet pairs')
    for w, l in sorted(det.items()):
        out.append(f'  {w}: {l}')
    # N1
    bycls = defaultdict(list)
    for k, (w, d, q) in enumerate(items): bycls[q[0]].append(k)
    bydoc = defaultdict(list)
    for k, (w, d, q) in enumerate(items): bydoc[d].append(k)
    for name, groups in (('N1 within commodity', bycls), ('N2 within tablet', bydoc)):
        gw = gp = 0; mw = mp = 0
        for _ in range(R):
            qs = [q for _w, _d, q in items]
            for g, ks in groups.items():
                v = [qs[k] for k in ks]; random.shuffle(v)
                for k, x in zip(ks, v): qs[k] = x
            r2 = defaultdict(list)
            for (w, d, _q), q in zip(items, qs):
                if w in keepw: r2[w].append((d, q))
            a, b, _ = stat(r2)
            mw += a; mp += b; gw += a >= ow; gp += b >= op
        out.append(f'{name}: words {mw/R:.2f} (P = {(gw+1)/(R+1):.4f}); pairs {mp/R:.2f} (P = {(gp+1)/(R+1):.4f})')
    # LB control: PY Ab / KN Fs / all LB docs with a single word heading-like
    L = []
    for line in open(os.path.join(D, 'damos_items.jsonl')):
        d = json.loads(line)
        h = d.get('heading') or ''
        L.append((h, d.get('content') or ''))
    docs = lb_docs()
    lrec = []
    for (h, c), doc in zip(L, docs):
        words = [t.strip('[],') for t in c.split() if re.match(r'^[a-z][a-z0-9\-]*[a-z0-9]$', t.strip('[],')) and '-' in t]
        if not words or not doc['com']: continue
        com = sorted(doc['com'].items(), key=lambda x: -x[1])
        # first word of the document, its biggest-valued first commodity (in the order parsed)
        first = next(iter(doc['lines'][0].items()))
        lrec.append((words[0], h, (first[0], float(first[1]))))
    nd = defaultdict(set)
    for w, h, q in lrec: nd[w].add(h)
    kw = {w for w, s in nd.items() if len(s) >= 2}
    r = defaultdict(list)
    for w, h, q in lrec:
        if w in kw: r[w].append((h, q))
    lw, lp, ldet = stat(r)
    items2 = [(w, h, q) for w, h, q in lrec]
    bc = defaultdict(list)
    for k, (w, h, q) in enumerate(items2): bc[q[0]].append(k)
    gw = 0; mw = 0
    for _ in range(R):
        qs = [q for *_x, q in items2]
        for g, ks in bc.items():
            v = [qs[k] for k in ks]; random.shuffle(v)
            for k, x in zip(ks, v): qs[k] = x
        r2 = defaultdict(list)
        for (w, h, _q), q in zip(items2, qs):
            if w in kw: r2[w].append((h, q))
        a, b, _ = stat(r2); mw += a; gw += a >= lw
    out.append(f'\nLinear B control (first word of a document, first commodity amount): {len(kw)} words on >= 2 documents; '
               f'{lw} words repeat an amount vs N1 {mw/R:.2f}, P = {(gw+1)/(R+1):.4f}')
    for w, l in list(sorted(ldet.items()))[:8]:
        out.append(f'  {w}: {l[:5]}')
    txt = '\n'.join(out)
    print(txt)
    open(os.path.join(D, 'la6_allot.out'), 'w').write(txt + '\n')


if __name__ == '__main__':
    main()
