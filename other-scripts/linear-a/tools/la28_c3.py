#!/usr/bin/env python3
"""la28 cycle 3: summary tablets, roles of linking terms, and receipt-only signs.

(a) Summary tablets: each tablet scored by the number of distinct own-site receipt
    terms it carries; per-tablet P against decoy receipt-term sets (type-level site swap,
    2,000), and the corpus maximum against the null maximum. LB analogue (PY, KN, MY).
(b) Roles: when a receipt term appears on a same-site tablet, is it an entry word
    (a number on the same line) more often than other tablet terms of the same kind
    (single sign / word / logogram)? Permutation of the 'receipt term' label among
    tablet term types of the same kind and similar frequency.
(c) Receipt-only signs: component signs (split on '-' and '+') of receipt terms never
    written on ANY tablet of the same site / of any site; vs site swap, and vs LB.
"""
import random, sys, os, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import la28_common as C

R = random.Random(283)
out = []


def say(s):
    print(s, flush=True)
    out.append(s)


def kind(t):
    k, v = t.split(':', 1)
    if k == 'L':
        return 'L'
    return 'S' if '-' not in v else 'W'


def summary(docs, label):
    tsites = {d['site'] for d in docs if d['cls'] == 'T'}
    types = sorted({(d['site'], t) for d in docs if d['cls'] == 'R' and d['site'] in tsites for t in C.doc_terms(d)})
    sites = [s for s, _ in types]
    terms = [t for _, t in types]
    tabs = [d for d in docs if d['cls'] == 'T' and C.doc_terms(d)]
    tt = [(d['id'], d['site'], C.doc_terms(d)) for d in tabs]

    def scores(sitelist):
        by = collections.defaultdict(set)
        for s, t in zip(sitelist, terms):
            by[s].add(t)
        return [len(ts & by[s]) for _, s, ts in tt]
    obs = scores(sites)
    nul = []
    for _ in range(2000):
        s2 = sites[:]
        R.shuffle(s2)
        nul.append(scores(s2))
    mx = max(obs)
    nmx = [max(v) for v in nul]
    say(f'  {label}: max receipt terms on one tablet {mx} vs null max {sum(nmx)/len(nmx):.2f} (P {(1+sum(v>=mx for v in nmx))/2001:.3f}); '
        f'tablets with >= 2 own-site receipt terms {sum(o>=2 for o in obs)} vs null {sum(sum(x>=2 for x in v) for v in nul)/len(nul):.1f} '
        f'(P {(1+sum(sum(x>=2 for x in v)>=sum(o>=2 for o in obs) for v in nul))/2001:.3f})')
    ranked = sorted(range(len(tt)), key=lambda i: -obs[i])[:6]
    by = collections.defaultdict(set)
    for s, t in types:
        by[s].add(t)
    for i in ranked:
        if obs[i] < 2:
            break
        p = (1 + sum(v[i] >= obs[i] for v in nul)) / 2001
        say(f'    {tt[i][0]} ({tt[i][1]}): {obs[i]} receipt terms {sorted(tt[i][2] & by[tt[i][1]])}, per-tablet P {p:.3f} (x{len(tt)} tablets)')


def roles(docs, label):
    tsites = {d['site'] for d in docs if d['cls'] == 'T'}
    rset = {(d['site'], t) for d in docs if d['cls'] == 'R' for t in C.doc_terms(d)}
    occ = collections.defaultdict(lambda: [0, 0])   # (site,term) -> [entry occurrences, all]
    for d in docs:
        if d['cls'] != 'T':
            continue
        for it in d['items']:
            for t in set(it['terms']):
                o = occ[(d['site'], t)]
                o[1] += 1
                o[0] += bool(it['nums'])
    keys = list(occ)
    isr = [k in rset for k in keys]
    stat = lambda lab: sum(occ[k][0] for k, r in zip(keys, lab) if r) / max(1, sum(occ[k][1] for k, r in zip(keys, lab) if r))
    obs = stat(isr)
    # permute receipt label within strata (site, kind, frequency band)
    strata = collections.defaultdict(list)
    for i, k in enumerate(keys):
        f = occ[k][1]
        band = 0 if f == 1 else 1 if f <= 3 else 2 if f <= 10 else 3
        strata[(k[0], kind(k[1]), band)].append(i)
    vals = []
    for _ in range(2000):
        lab = [False] * len(keys)
        for idx in strata.values():
            n = sum(isr[i] for i in idx)
            for i in R.sample(idx, n):
                lab[i] = True
        vals.append(stat(lab))
    m = sum(vals) / len(vals)
    p_hi = (1 + sum(v >= obs for v in vals)) / 2001
    p_lo = (1 + sum(v <= obs for v in vals)) / 2001
    n_r = sum(occ[k][1] for k, r in zip(keys, isr) if r)
    say(f'  {label}: receipt terms on tablets stand on a line with a number in {obs:.2f} of {n_r} occurrences vs matched other terms {m:.2f} '
        f'(P(>=) {p_hi:.3f}, P(<=) {p_lo:.3f})')
    top = sorted(((occ[k][1], k) for k, r in zip(keys, isr) if r), reverse=True)[:12]
    say('    most frequent linking terms (tablet occurrences, with-number share): ' +
        ', '.join(f'{k[1]}@{k[0][:2]} {n} ({occ[k][0]/n:.2f})' for n, k in top))


def comps(t):
    k, v = t.split(':', 1)
    return set(x for x in v.replace("'", '').replace('+', '-').split('-') if x and x != '[?]')


def receipt_only(docs, label):
    tsites = {d['site'] for d in docs if d['cls'] == 'T'}
    tsig = collections.defaultdict(set)
    for d in docs:
        if d['cls'] == 'T':
            for t in C.doc_terms(d):
                tsig[d['site']] |= comps(t)
    allsig = set().union(*tsig.values())
    types = sorted({(d['site'], t) for d in docs if d['cls'] == 'R' and d['site'] in tsites for t in C.doc_terms(d)})
    def cnt(sitelist):
        return sum(1 for s, (_, t) in zip(sitelist, types) if not (comps(t) <= tsig[s]))
    sites = [s for s, _ in types]
    obs = cnt(sites)
    vals = []
    for _ in range(2000):
        s2 = sites[:]
        R.shuffle(s2)
        vals.append(cnt(s2))
    glob = [t for _, t in types if not (comps(t) <= allsig)]
    say(f'  {label}: receipt types with a sign absent from own-site tablets {obs}/{len(types)} vs site swap {sum(vals)/len(vals):.1f} '
        f'(P(>=) {(1+sum(v>=obs for v in vals))/2001:.3f}); with a sign absent from ALL tablets: {len(set(glob))} '
        f'{sorted(set(glob))[:20]}')


if __name__ == '__main__':
    la = C.load_la()
    lb = C.load_lb()
    say('(a) summary tablets')
    summary(la, 'LA')
    summary(lb, 'LB')
    say('(b) roles of linking terms')
    roles(la, 'LA')
    roles(lb, 'LB')
    say('(c) receipt-only signs')
    receipt_only(la, 'LA')
    receipt_only([d for d in la if not (d['cls'] == 'R' and d['support'] != 'Roundel')], 'LA roundels')
    receipt_only([d for d in la if not (d['cls'] == 'R' and d['support'] != 'Nodule')], 'LA nodules')
    receipt_only(lb, 'LB')
    open(os.path.join(C.CK, 'c3.out'), 'w').write('\n'.join(out) + '\n')


def absent_rate(docs, label, reps=200):
    """Reference rates: share of term types with a sign never written on any tablet,
    for (i) other (non-receipt) documents, (ii) tablet halves vs the other half, (iii) random
    tablet sub-samples matched to the receipt type count."""
    tabs = [d for d in docs if d['cls'] == 'T']
    def sig(ds):
        s = set()
        for d in ds:
            for t in C.doc_terms(d):
                s |= comps(t)
        return s
    allsig = sig(tabs)
    rt = {t for d in docs if d['cls'] == 'R' for t in C.doc_terms(d)}
    ot = {t for d in docs if d['cls'] == 'O' for t in C.doc_terms(d)}
    r_rate = sum(1 for t in rt if not comps(t) <= allsig) / len(rt)
    o_rate = sum(1 for t in ot if not comps(t) <= allsig) / max(1, len(ot))
    hv = []
    for _ in range(reps):
        R.shuffle(tabs)
        k = max(1, len(tabs) // 10)
        A, B = tabs[:k], tabs[k:]
        sB = sig(B)
        tA = {t for d in A for t in C.doc_terms(d)} - {t for d in B for t in C.doc_terms(d)}
        tA_all = {t for d in A for t in C.doc_terms(d)}
        hv.append(sum(1 for t in tA_all if not comps(t) <= sB) / max(1, len(tA_all)))
    m = sum(hv) / len(hv)
    p = (1 + sum(v >= r_rate for v in hv)) / (1 + len(hv))
    say(f'  {label}: receipt types with a sign absent from all tablets {r_rate:.3f} ({len(rt)} types); other documents {o_rate:.3f} ({len(ot)}); '
        f'10 % tablet hold-out vs the other 90 % {m:.3f} (P(>= receipt rate) {p:.3f})')


if __name__ == '__main__':
    say('(c2) reference rates for receipt-only signs')
    absent_rate(C.load_la(), 'LA')
    absent_rate(C.load_lb(), 'LB')
    open(os.path.join(C.CK, 'c3.out'), 'w').write('\n'.join(out) + '\n')
