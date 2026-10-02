#!/usr/bin/env python3
"""Test (a): is the Voynich text language-like?
Statistics at a matched size (first N=20000 running-text words of each text):
Zipf slope, TTR, hapax share, word length mean/sd, h1/h2/h3 of the character
stream (word space counted as a symbol), adjacent-word mutual information.
Voynich in EVA letters and in glyph units (ch, sh, cth, ckh, cph, cfh = 1 unit).
Controls: char shuffle, char bigram/trigram Markov, table-grille, self-citation,
plus six real-language samples. Also word-shuffle baseline for MI."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from vlib import *
import gen

N = 20000

def unitize(lines):
    out = []
    for L in lines:
        nl = dict(L); nl['words'] = [''.join(glyphs(w)) for w in L['words']]; out.append(nl)
    return out

def run():
    res = {}
    texts = {}
    for name in ('ZL3b', 'IT2a'):
        v = load_voynich(name, ltypes=('P',), drop_uncertain=True)
        texts[f'Voynich-{name}-EVA'] = v
        texts[f'Voynich-{name}-glyph'] = unitize(v)
    vA = [L for L in load_voynich('ZL3b', ('P',), True) if L['lang'] == 'A']
    vB = [L for L in load_voynich('ZL3b', ('P',), True) if L['lang'] == 'B']
    texts['Voynich-ZL-A-glyph'] = unitize(vA)
    texts['Voynich-ZL-B-glyph'] = unitize(vB)
    for k in REFS:
        skip = 0.3 if k in ('Italian-Manzoni', 'Spanish-Cervantes', 'Italian-Dante') else 0.0
        texts[k] = load_ref(k, max_words=N + 100, skip_frac=skip)
    base = texts['Voynich-ZL3b-glyph']
    for g, f in gen.GENERATORS.items():
        texts['CTRL-' + g + '(Voynich-glyph)'] = f(base, seed=7)
    texts['CTRL-char_shuffle(Latin-Caesar)'] = gen.char_shuffle(texts['Latin-Caesar'], seed=7)
    for k, lines in texts.items():
        n = N if not k.startswith('Voynich-ZL-') else min(N, len(words_of(lines)))
        ws = words_of(truncate(lines, n))[:n]
        st = basic_stats(ws)
        wsh = words_of(gen.word_shuffle([{'words': ws}], seed=3))
        st['word_bigram_mi_shuffled'] = word_bigram_mi(wsh)
        st['word_bigram_mi_excess'] = st['word_bigram_mi'] - st['word_bigram_mi_shuffled']
        st['wlen_dist'] = [round(x, 4) for x in wlen_dist(ws)]
        res[k] = st
    # full-size Voynich numbers
    for name in ('ZL3b',):
        ws = words_of(texts[f'Voynich-{name}-glyph'])
        res[f'Voynich-{name}-glyph-FULL'] = basic_stats(ws)
    save('test_a_language', res)
    cols = ['tokens', 'types', 'ttr', 'hapax_frac_types', 'zipf_slope', 'wlen_mean', 'wlen_sd', 'alphabet', 'h1', 'h2', 'h3', 'word_bigram_mi_excess']
    print('%-40s' % 'text' + ''.join('%9s' % c[:9] for c in cols))
    for k, st in res.items():
        print('%-40s' % k[:40] + ''.join('%9.3f' % st[c] if isinstance(st.get(c), float) else '%9s' % st.get(c, '') for c in cols))

if __name__ == '__main__':
    run()
