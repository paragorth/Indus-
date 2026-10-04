#!/usr/bin/env python3
"""LA-6 cycle 4: put the words back. Numeric fingerprint of each frequent entry word.

Words: entry labels (the word that opens an entry) seen with a quantity on >= 3 tablets.
Fingerprint: log10 median quantity, residual log quantity (minus the commodity mean),
commodity shares (GRA OLE OLIV VIN NI CYP *304 counted=VIR/*305/*86/HIDE, bare, other),
fraction share, multi-commodity share, after-total share.

(A) Do numbers carry word identity beyond the commodity? F = between-word / within-word
    variance of the residual log quantity. Null: quantities shuffled within commodity
    (bare numbers form their own class) across all entries, 2,000 runs.
    Per word: mean residual and its P under the same null (two-sided).
(B) Same fingerprint = functional synonyms? Euclidean distance on standardised features;
    for each pair of words, P = share of 1,000 label-shuffle runs (entry labels permuted
    across entries carrying quantities) in which that pair is as close. Report the closest
    pairs, whether they co-occur on a tablet, and FDR-style count vs expectation.
(C) Converters: word w whose quantity is a fixed multiple of the entry just before it
    (same tablet). HIT = distinct tablets within +-10% of one ratio; null: the quantities
    shuffled within commodity; 2,000 runs.
"""
import sys, os, math, random
from collections import defaultdict, Counter
sys.path.insert(0, os.path.dirname(__file__))
from la6_common import la_entries, D

random.seed(64)
CLS = ['GRA', 'OLE', 'OLIV', 'VIN', 'NI', 'CYP', '*304', 'counted', 'bare', 'other']
COUNTED = {'VIR', '*305', '*86', 'HIDE'}


def cls(c):
    return c if c in CLS else ('counted' if c in COUNTED else 'other')


def units(e):
    """quantities of an entry as (class, value) list"""
    u = [(cls(c), float(v)) for c, v in e['com'].items() if v > 0]
    u += [('bare', float(v)) for v in e['bare'] if v > 0]
    return u


def main():
    out = ['# LA-6 cycle 4: numeric fingerprints of words']
    E = la_entries()
    Q = [e for e in E if e['role'] in ('entry', 'post', 'total', 'deficit', 'grand') and e['label'] and units(e)]
    tabs = defaultdict(set)
    for e in Q: tabs[e['label']].add(e['doc'])
    words = sorted(w for w in tabs if len(tabs[w]) >= 3)
    out.append(f'{len(Q)} labelled entries with quantities; {len(words)} words on >= 3 tablets')
    # residual log quantities
    allu = [(i, c, v) for i, e in enumerate(Q) for c, v in units(e)]
    pool = defaultdict(list)
    for i, c, v in allu: pool[c].append(math.log10(v))
    cmean = {c: sum(v) / len(v) for c, v in pool.items()}

    def wres(vals):
        # vals: list of (entry index, class, log10 value)
        byw = defaultdict(list)
        for i, c, lv in vals:
            w = Q[i]['label']
            if w in wset: byw[w].append(lv - cmean[c])
        return byw
    wset = set(words)

    def Fstat(byw):
        allr = [x for v in byw.values() for x in v]
        m = sum(allr) / len(allr)
        bss = sum(len(v) * (sum(v) / len(v) - m) ** 2 for v in byw.values())
        wss = sum((x - sum(v) / len(v)) ** 2 for v in byw.values() for x in v)
        k, n = len(byw), len(allr)
        return (bss / (k - 1)) / (wss / (n - k))
    obs_vals = [(i, c, math.log10(v)) for i, c, v in allu]
    byw = wres(obs_vals)
    F = Fstat(byw)
    obs_mean = {w: sum(v) / len(v) for w, v in byw.items()}
    R = 2000
    ge = 0; Fm = 0
    hi = Counter(); lo = Counter()
    idx_by_c = defaultdict(list)
    for k, (i, c, v) in enumerate(obs_vals): idx_by_c[c].append(k)
    for _ in range(R):
        sh = list(obs_vals)
        for c, ks in idx_by_c.items():
            vs = [obs_vals[k][2] for k in ks]; random.shuffle(vs)
            for k, v in zip(ks, vs): sh[k] = (obs_vals[k][0], c, v)
        b2 = wres(sh)
        f2 = Fstat(b2); Fm += f2; ge += f2 >= F
        for w, v in b2.items():
            m = sum(v) / len(v)
            hi[w] += m >= obs_mean[w]; lo[w] += m <= obs_mean[w]
    out.append(f'\n## (A) word identity in the numbers: F = {F:.2f} vs shuffled-within-commodity {Fm/R:.2f}, P = {(ge+1)/(R+1):.4f}')
    out.append('word | tablets | quantities | mean residual log10 (x typical for its commodity) | P two-sided | commodity profile')
    feats = {}
    for w in sorted(words, key=lambda w: obs_mean.get(w, 0)):
        if w not in byw: continue
        es = [e for e in Q if e['label'] == w]
        prof = Counter(c for e in es for c, v in units(e))
        p = min(1, 2 * min(hi[w] + 1, lo[w] + 1) / (R + 1))
        flag = ' *' if p < 0.05 else ''
        out.append(f'{w} | {len(tabs[w])} | {len(byw[w])} | {obs_mean[w]:+.2f} (x{10**obs_mean[w]:.2g}) | {p:.3f}{flag} | {dict(prof.most_common(4))}')
        n = sum(prof.values())
        lv = sorted(math.log10(v) for e in es for c, v in units(e))
        feats[w] = [lv[len(lv) // 2], obs_mean[w]] + [prof[c] / n for c in CLS] + [
            sum(1 for e in es if any(f for _, _, f in e['raw'])) / len(es),
            sum(1 for e in es if len(units(e)) >= 2) / len(es),
            sum(1 for e in es if e['zone'] == 'after') / len(es)]
    # (B) fingerprint distances
    ws = list(feats)
    dim = len(feats[ws[0]])

    def standardise(fe):
        mu = [sum(fe[w][j] for w in fe) / len(fe) for j in range(dim)]
        sd = [max(1e-9, (sum((fe[w][j] - mu[j]) ** 2 for w in fe) / len(fe)) ** .5) for j in range(dim)]
        return {w: [(fe[w][j] - mu[j]) / sd[j] for j in range(dim)] for w in fe}

    def dist(a, b):
        return sum((x - y) ** 2 for x, y in zip(a, b)) ** .5
    Z = standardise(feats)
    pairs = [(ws[i], ws[j]) for i in range(len(ws)) for j in range(i + 1, len(ws))]
    dobs = {p: dist(Z[p[0]], Z[p[1]]) for p in pairs}
    # null: permute labels across entries, recompute features for same word set
    labs = [e['label'] for e in Q]
    RB = 300
    closer = Counter()
    mins = []
    for _ in range(RB):
        random.shuffle(labs)
        grp = defaultdict(list)
        for l, e in zip(labs, Q):
            if l in feats: grp[l].append(e)
        fe = {}
        for w, es in grp.items():
            prof = Counter(c for e in es for c, v in units(e)); n = sum(prof.values())
            lv = sorted(math.log10(v) for e in es for c, v in units(e))
            res = [math.log10(v) - cmean[c] for e in es for c, v in units(e)]
            fe[w] = [lv[len(lv) // 2], sum(res) / len(res)] + [prof[c] / n for c in CLS] + [
                sum(1 for e in es if any(f for _, _, f in e['raw'])) / len(es),
                sum(1 for e in es if len(units(e)) >= 2) / len(es),
                sum(1 for e in es if e['zone'] == 'after') / len(es)]
        Z2 = standardise(fe)
        dd = {p: dist(Z2[p[0]], Z2[p[1]]) for p in pairs if p[0] in Z2 and p[1] in Z2}
        for p, d in dd.items(): closer[p] += d <= dobs[p]
        mins.append(sorted(dd.values())[:10])
    out.append(f'\n## (B) closest fingerprints (label-shuffle null, {RB} runs); P = share of runs with that pair as close')
    best = sorted(pairs, key=lambda p: closer[p])[:20]
    nsig = sum(1 for p in pairs if (closer[p] + 1) / (RB + 1) < 0.01)
    out.append(f'pairs with P < 0.01: {nsig} of {len(pairs)} (expected ~{len(pairs)*0.01:.0f} by chance)')
    for p in best:
        co = len(tabs[p[0]] & tabs[p[1]])
        out.append(f'  {p[0]} ~ {p[1]}: d = {dobs[p]:.2f}, P = {(closer[p]+1)/(RB+1):.4f}, shared tablets {co}')
    # (C) converters
    out.append('\n## (C) converters: q(word) / q(previous entry, same tablet)')
    bydoc = defaultdict(list)
    for e in E: bydoc[e['doc']].append(e)
    rat = defaultdict(list)
    for d, es in bydoc.items():
        prev = None
        for e in es:
            u = units(e)
            if e['label'] and u and prev:
                rat[e['label']].append((d, u[0], prev))
            if u: prev = u[0]
    RC = 2000
    res = []
    for w in words:
        xs = rat.get(w, [])
        if len({d for d, *_ in xs}) < 3: continue

        def hit(rs, docs):
            o = sorted(range(len(rs)), key=lambda i: rs[i]); best = 0; j = 0
            for i in range(len(o)):
                while rs[o[i]] - rs[o[j]] > 2 * math.log(1.1): j += 1
                best = max(best, len({docs[o[k]] for k in range(j, i + 1)}))
            return best
        rs = [math.log(a[1] / b[1]) for d, a, b in xs]
        docs = [d for d, *_ in xs]
        h = hit(rs, docs)
        ge = 0
        for _ in range(RC):
            r2 = [math.log(10 ** random.choice(pool[a[0]]) / 10 ** random.choice(pool[b[0]])) for d, a, b in xs]
            ge += hit(r2, docs) >= h
        res.append((w, len(set(docs)), h, (ge + 1) / (RC + 1), sorted(round(math.exp(r), 2) for r in rs)))
    for w, n, h, p, rr in sorted(res, key=lambda x: x[3]):
        out.append(f'  {w}: tablets {n}, HIT {h}, P = {p:.3f}, ratios {rr[:12]}')
    txt = '\n'.join(out)
    print(txt)
    open(os.path.join(D, 'la6_fingerprint.out'), 'w').write(txt + '\n')


if __name__ == '__main__':
    main()
