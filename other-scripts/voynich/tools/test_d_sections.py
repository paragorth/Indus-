#!/usr/bin/env python3
"""Test (d): section and Currier A/B differences; labels vs running text.
1. Word x section enrichment (one-sided binomial, BH FDR 5%) on running text.
   Null: section labels permuted across pages (section page counts kept), 20x.
2. Currier A vs B: Jensen-Shannon divergence of word distributions at matched
   size (5000 tokens drawn as whole pages), vs within-A and within-B page splits,
   herbal-A vs herbal-B (same illustration type), and real-language yardsticks
   (two halves of one author, two Latin authors, Latin vs Italian).
3. Labels: share of label tokens whose word type occurs in running text, vs
   running-text tokens (leave-one-out); label word length; first-glyph profile."""
import sys, os, math, random
sys.path.insert(0, os.path.dirname(__file__))
from vlib import *
from test_a_language import unitize

SEC = {'H': 'herbal', 'A': 'astro-cosmo', 'Z': 'astro-cosmo', 'C': 'astro-cosmo',
       'B': 'bio', 'P': 'pharma', 'S': 'stars-recipes', 'T': 'text-only'}

def pages_of(lines, key=None):
    pg = defaultdict(list); lab = {}
    for L in lines:
        pg[L['folio']].extend(L['words'])
        if key: lab[L['folio']] = key(L)
    return pg, lab

def enrichment(pg, lab, min_count=10):
    secs = sorted(set(lab.values()))
    tot = Counter(); bys = {s: Counter() for s in secs}
    for f, ws in pg.items():
        tot.update(ws); bys[lab[f]].update(ws)
    N = sum(tot.values()); ns = {s: sum(bys[s].values()) for s in secs}
    tests, pv = [], []
    for w, c in tot.items():
        if c < min_count: continue
        for s in secs:
            k = bys[s][w]
            exp = c * ns[s] / N
            if k <= exp: continue
            pv.append(log_binom_sf(k, c, ns[s] / N)); tests.append((w, s, k, c, round(k / exp, 2)))
    sig = bh_fdr(pv) if pv else []
    hits = [t for t, s_ in zip(tests, sig) if s_]
    return hits, len(pv)

def jsd(c1, c2):
    n1, n2 = sum(c1.values()), sum(c2.values())
    keys = set(c1) | set(c2); d = 0.0
    for k in keys:
        p = c1.get(k, 0) / n1; q = c2.get(k, 0) / n2; m = (p + q) / 2
        if p: d += 0.5 * p * math.log2(p / m)
        if q: d += 0.5 * q * math.log2(q / m)
    return d

def sample_pages(pages, n, rng):
    ps = list(pages); rng.shuffle(ps); out = []
    for p in ps:
        out.extend(p)
        if len(out) >= n: break
    return Counter(out[:n])

def split_jsd(pages, n, rng):
    ps = list(pages); rng.shuffle(ps); h = len(ps) // 2
    return jsd(sample_pages(ps[:h], n, rng), sample_pages(ps[h:], n, rng))

def blocks(words, size=170):
    return [words[i:i + size] for i in range(0, len(words), size)]

def run():
    out = {}
    rng = random.Random(5)
    vz = unitize(load_voynich('ZL3b', ('P',), True))
    pg, lab = pages_of(vz, lambda L: SEC.get(L['illus'], '?'))
    hits, ntests = enrichment(pg, lab)
    # null: permute labels across pages
    nulls = []
    folios = list(pg); labs = [lab[f] for f in folios]
    for s in range(20):
        r = random.Random(100 + s); l2 = labs[:]; r.shuffle(l2)
        h, _ = enrichment(pg, dict(zip(folios, l2))); nulls.append(len(h))
    bysec = Counter(h[1] for h in hits)
    out['section_enrichment'] = {'n_tests': ntests, 'n_sig': len(hits), 'null_mean': sum(nulls) / len(nulls),
                                 'null_max': max(nulls), 'by_section': dict(bysec),
                                 'top': sorted(hits, key=lambda t: -t[2])[:40]}
    print('section enrichment: sig', len(hits), 'of', ntests, 'null mean', sum(nulls) / 20, 'max', max(nulls), dict(bysec))
    for s in sorted(bysec):
        print('  ', s, [(h[0], h[2], h[3], h[4]) for h in sorted([h for h in hits if h[1] == s], key=lambda t: -t[2])[:8]])
    # Currier A vs B (all running text), and within herbal
    pgL, labL = pages_of(vz, lambda L: L['lang'])
    A = [pg[f] for f in pg if labL[f] == 'A']; B = [pg[f] for f in pg if labL[f] == 'B']
    HA = [pg[f] for f in pg if labL[f] == 'A' and lab[f] == 'herbal']
    HB = [pg[f] for f in pg if labL[f] == 'B' and lab[f] == 'herbal']
    hitsAB, nAB = enrichment({f: pg[f] for f in pg if labL[f] in ('A', 'B')}, {f: labL[f] for f in pg if labL[f] in ('A', 'B')})
    out['AB_enrichment'] = {'n_tests': nAB, 'n_sig': len(hitsAB), 'top_A': sorted([h for h in hitsAB if h[1] == 'A'], key=lambda t: -t[2])[:15],
                            'top_B': sorted([h for h in hitsAB if h[1] == 'B'], key=lambda t: -t[2])[:15]}
    print('A/B enriched words', len(hitsAB), 'of', nAB)
    nH = min(sum(map(len, HA)), sum(map(len, HB))) // 1
    def rep(f, k=20):
        v = [f() for _ in range(k)]; return round(sum(v) / k, 4)
    n = 5000
    J = {}
    J['Voynich A vs B'] = rep(lambda: jsd(sample_pages(A, n, rng), sample_pages(B, n, rng)))
    J['Voynich A vs A (page split)'] = rep(lambda: split_jsd(A, n, rng))
    J['Voynich B vs B (page split)'] = rep(lambda: split_jsd(B, n, rng))
    nh = min(2000, nH // 1)
    J[f'Voynich herbal-A vs herbal-B (n={nh})'] = rep(lambda: jsd(sample_pages(HA, nh, rng), sample_pages(HB, nh, rng)))
    J[f'Voynich herbal-A split (n={nh})'] = rep(lambda: split_jsd(HA, nh, rng))
    refs = {k: blocks(words_of(load_ref(k, max_words=20000, skip_frac=0.3 if k in ('Italian-Manzoni', 'Spanish-Cervantes', 'Italian-Dante') else 0))) for k in REFS}
    J['Caesar vs Caesar (block split)'] = rep(lambda: split_jsd(refs['Latin-Caesar'], n, rng))
    J['Manzoni vs Manzoni (block split)'] = rep(lambda: split_jsd(refs['Italian-Manzoni'], n, rng))
    J['Caesar vs Descartes (two Latin authors)'] = rep(lambda: jsd(sample_pages(refs['Latin-Caesar'], n, rng), sample_pages(refs['Latin-Descartes'], n, rng)))
    J['Manzoni vs Dante (two Italian authors)'] = rep(lambda: jsd(sample_pages(refs['Italian-Manzoni'], n, rng), sample_pages(refs['Italian-Dante'], n, rng)))
    J['Caesar vs Manzoni (Latin vs Italian)'] = rep(lambda: jsd(sample_pages(refs['Latin-Caesar'], n, rng), sample_pages(refs['Italian-Manzoni'], n, rng)))
    J['Manzoni vs Cervantes (Italian vs Spanish)'] = rep(lambda: jsd(sample_pages(refs['Italian-Manzoni'], n, rng), sample_pages(refs['Spanish-Cervantes'], n, rng)))
    out['jsd_words'] = J
    for k, v in J.items(): print(f'  JSD {k:45s} {v}')
    # glyph-bigram level A vs B
    def gb(pages):
        c = Counter()
        for p in pages:
            for w in p:
                s = '_' + w + '_'; c.update(zip(s, s[1:]))
        return c
    out['jsd_glyph_bigrams'] = {'A vs B': jsd(gb(A), gb(B)),
                                'A split': jsd(gb(A[::2]), gb(A[1::2])), 'B split': jsd(gb(B[::2]), gb(B[1::2])),
                                'Caesar vs Descartes': jsd(gb(refs['Latin-Caesar']), gb(refs['Latin-Descartes'])),
                                'Caesar vs Manzoni': jsd(gb(refs['Latin-Caesar']), gb(refs['Italian-Manzoni']))}
    print('glyph-bigram JSD', {k: round(v, 4) for k, v in out['jsd_glyph_bigrams'].items()})
    # labels
    allrecs = [L for L in unitize(load_voynich('ZL3b', ('L',), True))]
    labw = [w for L in allrecs for w in L['words']]
    textw = words_of(vz); tc = Counter(textw)
    lab_in = sum(1 for w in labw if tc[w] > 0) / len(labw)
    samp = rng.sample(textw, min(len(labw), len(textw)))
    txt_in = sum(1 for w in samp if tc[w] > 1) / len(samp)
    # length-matched control: text tokens with the same length distribution as labels
    bylen = defaultdict(list)
    for w in textw: bylen[len(w)].append(w)
    lm = [rng.choice(bylen[len(w)]) for w in labw if bylen[len(w)]]
    txt_in_lm = sum(1 for w in lm if tc[w] > 1) / len(lm)
    lab_single = [L['words'][0] for L in allrecs if len(L['words']) == 1]
    fg_l = Counter(w[0] for w in labw); fg_t = Counter(w[0] for w in textw)
    nl, nt = len(labw), len(textw)
    fg = sorted(((g, round(fg_l[g] / nl, 3), round(fg_t[g] / nt, 3)) for g in set(fg_l) | set(fg_t)), key=lambda x: -x[1])[:10]
    out['labels'] = {'label_tokens': nl, 'label_types': len(set(labw)), 'single_word_labels': len(lab_single),
                     'share_label_tokens_type_in_text': lab_in, 'share_text_tokens_type_elsewhere_in_text': txt_in,
                     'share_text_tokens_type_elsewhere_lengthmatched': txt_in_lm,
                     'mean_len_label': sum(map(len, labw)) / nl, 'mean_len_text': sum(map(len, textw)) / nt,
                     'first_glyph_label_vs_text': fg,
                     'label_pages_by_section': dict(Counter(SEC.get(L['illus'], '?') for L in allrecs))}
    print('labels', {k: v for k, v in out['labels'].items()})
    save('test_d_sections', out)

if __name__ == '__main__':
    run()
