"""v78 cycle 3: INVERT THE PROBLEM. Which meaningful writing unit has no neighbour rule?

Cycle 1: Voynich tokens recur at lag 1 exactly as often as at lags 2-3 (spike 0.99), with line-level re-use, while
words of every language avoid their neighbour (spike 0.04-0.19). So ask: at what unit size does a real text lose the
neighbour ban? Real texts are cut into units of different sizes - WORD, SYLLABLE (vowel-nucleus split), LETTER,
and for chant the NEUME GROUP of one syllable - laid out at ~9 units per line (Voynich mean), or on natural lines
(chant phrases), and measured with the cycle-1 decomposition (rate, page, line, adj, lag2, spike) both as plain
units and after the v72 merge code + planted surface + E1c (the Voynich's own pipeline). The Voynich must fall in
the band of one unit size and outside the others; the kill: if no unit size hits rate, spike and line together,
the 'unit size' reading is dropped.
"""
import os, sys, re, json, pickle, random, gzip
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import v78_lib as L, v78_c1 as C1, v78_corpora as CO, v72_lib as V

VOW = set('aeiouyáéíóúàèìòù')


def syllables(w):
    """vowel-nucleus split: V.CV and VC.CV (consonant clusters split before their last consonant)."""
    nuc = [i for i, c in enumerate(w) if c in VOW and (i == 0 or w[i - 1] not in VOW)]
    if len(nuc) <= 1: return [w]
    cuts = []
    for a, b in zip(nuc, nuc[1:]):
        j = a
        while j < len(w) and w[j] in VOW: j += 1
        cons = b - j
        cuts.append(j if cons <= 1 else b - 1)
    out = []; s = 0
    for c in cuts: out.append(w[s:c]); s = c
    out.append(w[s:])
    return [x for x in out if x]


def units(paras, level):
    out = []
    for ws in paras:
        if level == 'word': out.append(ws)
        elif level == 'syll': out.append([s for w in ws for s in syllables(w)])
        elif level == 'letter': out.append([c for w in ws for c in w])
    return out


def sources():
    S = {}
    def from_pages(pages): return [[w for l in p['lines'] for w in l['w']] for p in pages]
    S['LA'] = from_pages(V.isidore_plain())
    S['IT'] = from_pages(V.brumati_plain())
    S['DE'] = from_pages(V.german_plain())
    S['TL'] = CO.tagalog('tl_17479.txt')
    S['MS'] = CO.malay()
    return S


def chant_pages(cap=40000):
    d = json.load(gzip.open(os.path.join(L.ROOT, 'data', 'derived', 'v5_chant.json.gz')))
    pages = []; n = 0; cur = None
    for k, c in enumerate(d):
        if n >= cap: break
        if cur is None or sum(len(l['w']) for l in cur['lines']) >= 160:
            cur = dict(id='ch%04d' % len(pages), sec='m' + str(c.get('mode', '-'))[:1], lang='-', hand='-', quire='-', lines=[])
            pages.append(cur)
        for i, ph in enumerate(c['phrases']):
            ws = [x for x in ph if x]
            if ws: cur['lines'].append(dict(w=ws, ps=(i == 0))); n += len(ws)
    return pages


def main():
    S = sources(); res = {}
    for src, paras in S.items():
        for lev in ('word', 'syll', 'letter'):
            U = units(paras, lev)
            pages = L.plain_pages(U, src.lower(), cap=36000, line_w=9)
            for mode in ('plain', 'surf'):
                P = pages if mode == 'plain' else L.through_surface(pages, 7300 + len(src))
                if mode == 'plain':
                    L._E.clear()
                    saved = L.e1c
                    L.e1c = lambda w: w; C1.L.e1c = L.e1c
                r, ci = C1.summarize(P, B=100)
                if mode == 'plain':
                    L.e1c = saved; C1.L.e1c = saved; L._E.clear()
                key = '%s_%s_%s' % (src, lev, mode); res[key] = dict(r=r, ci=ci)
                print(key, C1.fmt(r, ci), flush=True)
    P = chant_pages()
    for mode in ('plain', 'surf'):
        Q = P if mode == 'plain' else L.through_surface(P, 7399)
        if mode == 'plain':
            saved = L.e1c; L.e1c = lambda w: w; C1.L.e1c = L.e1c
        r, ci = C1.summarize(Q, B=100)
        if mode == 'plain': L.e1c = saved; C1.L.e1c = saved; L._E.clear()
        res['CHANT_neume_' + mode] = dict(r=r, ci=ci); print('CHANT_neume_' + mode, C1.fmt(r, ci), flush=True)
    json.dump(res, open(os.path.join(L.CK, 'c3.json'), 'w'), default=float, indent=1)


if __name__ == '__main__':
    main()
