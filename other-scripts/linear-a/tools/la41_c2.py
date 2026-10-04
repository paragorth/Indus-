#!/usr/bin/env python3
"""LA-41 cycle 2: the edit profile, and which elements are fixed.

On the robust families of cycle 1 (Linear A, Linear B, planted), items are matched order-free
(exact word, then one-sign variant) and every edit operation is catalogued.
(1) Fixed vs variable. For matched items, rates of identical spelling, identical logogram and
    identical amount are compared with two nulls: (N1) the same word on two unrelated lists of
    the same site (non-family document pairs); (N2) random list pairs of the same site and the
    same item counts, item i against item i. Bootstrap over families for intervals.
(1b) Reverse selection: families found by aligning NUMBER sequences only (>= 4 equal amounts in
    order, amounts >= 2, coverage >= 0.5, FDR 0.1 against a site-number shuffle); then word
    identity at the aligned positions vs N2. Tells whether numbers are ever kept while names change.
"""
import sys, os, json, pickle, random, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from la41_common import *

OUT = os.path.join(HERE, '..', 'loops', 'la41_cycle2.txt')


def load_fams(tag):
    J = json.load(open(os.path.join(CK, f'c1_fam_{tag}.json')))
    return [(r[0], r[1]) for r in J['rob']]


def elem_rates(docs, pairs_list):
    """pairs_list: list of (a, b, matched item pairs). Returns per-family counts."""
    out = []
    for a, b, mp in pairs_list:
        A, B = docs[a]['items'], docs[b]['items']
        c = Counter()
        for i, j in mp:
            x, y = A[i], B[j]
            c['w_n'] += 1; c['w_same'] += x['w'] == y['w']
            if x['logo'] and y['logo']: c['l_n'] += 1; c['l_same'] += x['logo'] == y['logo']
            if x['num'] is not None and y['num'] is not None: c['a_n'] += 1; c['a_same'] += x['num'] == y['num']
        out.append(c)
    return out


def rate(cs, k):
    n = sum(c[k + '_n'] for c in cs); s = sum(c[k + '_same'] for c in cs)
    return s / n if n else float('nan'), s, n


def boot(cs, k, rng, R=2000):
    v = []
    for _ in range(R):
        smp = [rng.choice(cs) for _ in cs]
        r = rate(smp, k)[0]
        if r == r: v.append(r)
    v.sort()
    return v[int(0.025 * len(v))], v[int(0.975 * len(v))]


def null_sameword(docs, fams, rng, maxp=20000):
    """N1: same word (2+ signs, not totals) on two non-family docs of one site."""
    famset = {(min(a, b), max(a, b)) for a, b in fams}
    stems = {}
    occ = defaultdict(list)
    for d in docs:
        for i, it in enumerate(docs[d]['items']):
            if it['w'] and len(it['w']) >= 2 and it['w'] not in TOTW:
                occ[(docs[d]['site'], it['w'])].append((d, i))
    pl = []
    for k, L in occ.items():
        for p in range(len(L)):
            for q in range(p + 1, len(L)):
                a, b = L[p][0], L[q][0]
                if a == b or (min(a, b), max(a, b)) in famset or docs[a]['stem'] == docs[b]['stem']: continue
                pl.append((a, b, [(L[p][1], L[q][1])]))
    rng.shuffle(pl)
    return pl[:maxp]


def null_positional(docs, fams, rng, reps=200):
    """N2: random same-site doc pairs with the same item counts as each family pair."""
    by = defaultdict(list)
    for d in docs: by[(docs[d]['site'], len(docs[d]['items']))].append(d)
    pl = []
    for a, b in fams:
        ka = (docs[a]['site'], len(docs[a]['items'])); kb = (docs[b]['site'], len(docs[b]['items']))
        for _ in range(reps):
            x = rng.choice(by[ka]); y = rng.choice(by[kb])
            if x == y: continue
            n = min(len(docs[x]['items']), len(docs[y]['items']))
            mp = [(i, i) for i in range(n) if docs[x]['items'][i]['w'] and docs[y]['items'][i]['w']]
            pl.append((x, y, mp))
    return pl


def number_families(docs, rng, reps=30):
    def numseq(d):
        return [(i, it['num']) for i, it in enumerate(docs[d]['items']) if it['num'] is not None and (it['val'] or 0) >= 2]

    def lcs_pairs(X, Y):
        n, m = len(X), len(Y)
        H = [[0] * (m + 1) for _ in range(n + 1)]
        for i in range(n):
            for j in range(m):
                H[i + 1][j + 1] = H[i][j] + 1 if X[i][1] == Y[j][1] else max(H[i][j + 1], H[i + 1][j])
        i, j, out = n, m, []
        while i and j:
            if X[i - 1][1] == Y[j - 1][1] and H[i][j] == H[i - 1][j - 1] + 1: out.append((X[i - 1][0], Y[j - 1][0])); i -= 1; j -= 1
            elif H[i - 1][j] >= H[i][j - 1]: i -= 1
            else: j -= 1
        return out[::-1]

    def scan(dd):
        ids = [d for d in dd if len([1 for it in dd[d]['items'] if it['num'] is not None and (it['val'] or 0) >= 2]) >= 4]
        by = defaultdict(list)
        for d in ids: by[dd[d]['site']].append(d)
        res = []
        for s, L in by.items():
            seqs = {d: [(i, it['num']) for i, it in enumerate(dd[d]['items']) if it['num'] is not None and (it['val'] or 0) >= 2] for d in L}
            for p in range(len(L)):
                for q in range(p + 1, len(L)):
                    a, b = L[p], L[q]
                    mp = lcs_pairs(seqs[a], seqs[b])
                    cov = len(mp) / min(len(seqs[a]), len(seqs[b]))
                    if len(mp) >= 4 and cov >= 0.5: res.append((a, b, mp, len(set(n for _, n in seqs[a]))))
        return res

    real = scan(docs)
    nul = []
    for r in range(reps):
        # shuffle amounts among items of the same site (keeps list shapes)
        dd = {}
        pool = defaultdict(list)
        for d in docs:
            for it in docs[d]['items']:
                if it['num'] is not None: pool[docs[d]['site']].append((it['num'], it['val']))
        for s in pool: rng.shuffle(pool[s])
        pos = Counter()
        for d in docs:
            s = docs[d]['site']; items = []
            for it in docs[d]['items']:
                it2 = dict(it)
                if it['num'] is not None:
                    it2['num'], it2['val'] = pool[s][pos[s]]; pos[s] += 1
                items.append(it2)
            dd[d] = dict(docs[d]); dd[d]['items'] = items
        nul.append(len(scan(dd)))
    return real, nul


def main():
    rng = random.Random(41)
    rows = []
    res = {}
    for tag in ('LA', 'LB'):
        docs = la_docs() if tag == 'LA' else lb_docs()
        fams = load_fams(tag)
        fl = [(a, b, set_match(docs[a]['items'], docs[b]['items'])) for a, b in fams]
        F = elem_rates(docs, fl)
        N1 = elem_rates(docs, null_sameword(docs, fams, rng))
        N2 = elem_rates(docs, null_positional(docs, fams, rng))
        # edit catalogue
        ops = Counter(); subs = []; amts = []
        for a, b, mp in fl:
            o, s, am = catalogue(docs[a]['items'], docs[b]['items'], mp)
            # catalogue's drop/add span uses first..last matched; order swaps from mp directly
            o['order_swaps'] = sum(1 for p in range(len(mp)) for q in range(p + 1, len(mp)) if mp[q][1] < mp[p][1])
            ops.update(o); subs += [(a, b) + t for t in s]; amts += [(a, b) + t for t in am]
        res[tag] = dict(ops=dict(ops), subs=[[a, b, '-'.join(x), '-'.join(y), list(e)] for a, b, x, y, e in subs], amts=amts)
        line = []
        for k, nm in (('w', 'spelling'), ('l', 'logogram'), ('a', 'amount')):
            r, s, n = rate(F, k); lo, hi = boot(F, k, rng) if n else (float('nan'),) * 2
            r1 = rate(N1, k); r2 = rate(N2, k)
            line.append(f"{nm} same {s}/{n} = {r:.2f} [{lo:.2f}-{hi:.2f}] vs same-word-unrelated {r1[0]:.2f} ({r1[2]}) vs positional {r2[0]:.2f} ({r2[2]})")
        tot_ops = ', '.join(f"{k} {v}" for k, v in sorted(ops.items()))
        rows.append(f"| LA-41.2{'a' if tag == 'LA' else 'b'} | {tag}: {len(fams)} robust families, order-free item matching; element identity rates vs N1 (same word on two unrelated same-site lists) and N2 (random same-site list pairs of the same item counts, item i vs i); bootstrap over families. | {'; '.join(line)}. Edit catalogue: {tot_ops}. | see 2c |")
        if tag == 'LA':
            real, nul = number_families(docs, rng)
            nm = sum(nul) / len(nul)
            wid = []
            for a, b, mp, nd in real:
                A, B = docs[a]['items'], docs[b]['items']
                ws = [(A[i]['w'], B[j]['w']) for i, j in mp]
                wid.append(sum(1 for x, y in ws if x and y and x == y) / len(ws))
            desc = '; '.join(f"{a}~{b} n{len(mp)} distinct amounts {nd} word-identity {w:.2f}" for (a, b, mp, nd), w in zip(real, wid))
            rows.append(f"| LA-41.2c | LA reverse selection: lists aligned on NUMBERS only (LCS of amounts >= 2; >= 4 equal amounts, coverage >= 0.5, same site) vs amounts shuffled among same-site items (30 reps). | Real {len(real)} pairs vs null mean {nm:.2f} (max {max(nul)}). {desc} | {'numbers carry families' if len(real) > 2 * nm + 2 else 'no number-only families beyond null'} |")
    json.dump(res, open(os.path.join(CK, 'c2.json'), 'w'), indent=0)
    for r in rows: wlog(OUT, r)
    print('\n'.join(rows))


if __name__ == '__main__':
    main()
