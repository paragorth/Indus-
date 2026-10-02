#!/usr/bin/env python3
"""Test (b): line and paragraph position effects.
Null model = within-line shuffle of word order (keeps every line's words).
For each text:
  * positional MI excess: I(position class; feature) observed minus the mean
    over 20 within-line shuffles, for feature = word type, first glyph, last glyph.
    position class = line-initial / medial / line-final (lines >= 3 words).
  * number of word types significantly enriched line-initially / line-finally
    (exact Bernoulli-sum mean/variance under the shuffle, z-test, BH FDR 5%).
  * mean word length by position (obs vs shuffle expectation = line mean).
Voynich only: paragraph-initial glyph table (gallows) vs other line-initial words,
and line-initial/final glyph enrichments.
References: Gutenberg typeset lines (arbitrary breaks: should give ~0) and Dante
verse lines (real linguistic units)."""
import sys, os, math, random
sys.path.insert(0, os.path.dirname(__file__))
from vlib import *
import gen
from test_a_language import unitize

def pos_class(i, n):
    return 'I' if i == 0 else ('F' if i == n - 1 else 'M')

def mi_pos(lines, feat):
    joint = Counter(); px = Counter(); py = Counter(); n = 0
    for L in lines:
        ws = L['words']
        if len(ws) < 3: continue
        for i, w in enumerate(ws):
            a = pos_class(i, len(ws)); b = feat(w)
            joint[a, b] += 1; px[a] += 1; py[b] += 1; n += 1
    return sum(v / n * math.log2(v * n / (px[a] * py[b])) for (a, b), v in joint.items())

def mi_excess(lines, feat, reps=20):
    obs = mi_pos(lines, feat)
    sh = [mi_pos(gen.within_line_shuffle(lines, seed=s), feat) for s in range(reps)]
    m = sum(sh) / reps; sd = (sum((x - m) ** 2 for x in sh) / reps) ** 0.5
    return {'obs': obs, 'shuffle_mean': m, 'excess': obs - m, 'z': (obs - m) / sd if sd > 0 else None}

def enriched_types(lines, which, min_count=20, feat=lambda w: w):
    obs = Counter(); mu = Counter(); var = Counter(); tot = Counter()
    for L in lines:
        ws = [feat(w) for w in L['words']]
        n = len(ws)
        if n < 3: continue
        tgt = ws[0] if which == 'I' else ws[-1]
        obs[tgt] += 1
        for w, k in Counter(ws).items():
            p = k / n; mu[w] += p; var[w] += p * (1 - p); tot[w] += k
    keys = [w for w in tot if tot[w] >= min_count]
    rows, pv = [], []
    for w in keys:
        z = (obs[w] - mu[w]) / math.sqrt(var[w]) if var[w] > 0 else 0
        p = 0.5 * math.erfc(abs(z) / math.sqrt(2)) * 2
        rows.append((w, tot[w], obs[w], round(mu[w], 1), round(obs[w] / mu[w], 2) if mu[w] else None, round(z, 1))); pv.append(p)
    sig = bh_fdr(pv) if pv else []
    rows = [r for r, s in zip(rows, sig) if s]
    rows.sort(key=lambda r: -abs(r[5]))
    return len(keys), rows

def length_by_pos(lines):
    s = Counter(); c = Counter(); e = Counter()
    for L in lines:
        ws = L['words']
        if len(ws) < 3: continue
        m = sum(len(w) for w in ws) / len(ws)
        for i, w in enumerate(ws):
            a = pos_class(i, len(ws)); s[a] += len(w); c[a] += 1; e[a] += m
    return {a: {'obs': round(s[a] / c[a], 3), 'shuffle_exp': round(e[a] / c[a], 3)} for a in 'IMF'}

def para_initial_glyphs(lines):
    """first glyph of paragraph-first word vs first glyph of other line-first words."""
    pa = Counter(); other = Counter()
    for L in lines:
        g = L['words'][0][0]
        (pa if L['para_start'] else other)[g] += 1
    return pa, other

def summarize(name, lines, out, detail=False):
    r = {}
    r['lines_ge3'] = sum(1 for L in lines if len(L['words']) >= 3)
    r['mi_word'] = mi_excess(lines, lambda w: w)
    r['mi_first_glyph'] = mi_excess(lines, lambda w: w[0])
    r['mi_last_glyph'] = mi_excess(lines, lambda w: w[-1])
    nk, ri = enriched_types(lines, 'I'); _, rf = enriched_types(lines, 'F')
    r['types_tested'] = nk; r['types_sig_initial'] = len(ri); r['types_sig_final'] = len(rf)
    r['length_by_pos'] = length_by_pos(lines)
    if detail:
        r['top_initial_types'] = ri[:15]; r['top_final_types'] = rf[:15]
        _, gi = enriched_types(lines, 'I', 20, lambda w: w[0])
        _, gf = enriched_types(lines, 'F', 20, lambda w: w[-1])
        r['glyph_first_of_line_sig'] = gi; r['glyph_last_of_line_sig'] = gf
        pa, other = para_initial_glyphs(lines)
        npa, no = sum(pa.values()), sum(other.values())
        r['para_initial'] = {'n_paragraph_starts': npa, 'n_other_line_starts': no,
            'gallows_share_para_start': sum(pa[g] for g in 'ktpfKTPF') / npa,
            'gallows_share_other_line_start': sum(other[g] for g in 'ktpfKTPF') / no,
            'p_or_f_para_start': sum(pa[g] for g in 'pfPF') / npa,
            'p_or_f_other_line_start': sum(other[g] for g in 'pfPF') / no,
            'top_para_initial': pa.most_common(8)}
        # p/f gallows anywhere: share of their tokens on paragraph-first lines
        pf_first = pf_all = n_first = n_all = 0
        for L in lines:
            k = sum(w.count('p') + w.count('f') + w.count('P') + w.count('F') for w in L['words'])
            g = sum(len(w) for w in L['words'])
            pf_all += k; n_all += g
            if L['para_start']: pf_first += k; n_first += g
        r['pf_gallows_on_para_first_lines'] = {'share_of_pf_tokens': pf_first / pf_all, 'share_of_all_glyphs': n_first / n_all}
    out[name] = r
    print(f"{name:34s} MIword exc={r['mi_word']['excess']:.4f} (z={r['mi_word']['z']:.1f})  firstglyph exc={r['mi_first_glyph']['excess']:.4f} (z={r['mi_first_glyph']['z']:.1f})  lastglyph exc={r['mi_last_glyph']['excess']:.4f} (z={r['mi_last_glyph']['z']:.1f})  sigI={r['types_sig_initial']} sigF={r['types_sig_final']} /{nk}  len={r['length_by_pos']}")

def run():
    out = {}
    vz = unitize(load_voynich('ZL3b', ('P',), True))
    vt = unitize(load_voynich('IT2a', ('P',), True))
    summarize('Voynich-ZL3b-glyph', vz, out, detail=True)
    summarize('Voynich-IT2a-glyph', vt, out, detail=True)
    summarize('Voynich-ZL-A', [L for L in vz if L['lang'] == 'A'], out)
    summarize('Voynich-ZL-B', [L for L in vz if L['lang'] == 'B'], out)
    for g in ('char_trigram_markov', 'table_grille', 'self_citation'):
        summarize('CTRL-' + g, gen.GENERATORS[g](vz, seed=7), out)
    for k in REFS:
        skip = 0.3 if k in ('Italian-Manzoni', 'Spanish-Cervantes', 'Italian-Dante') else 0.0
        summarize(k, load_ref(k, max_words=35000, skip_frac=skip), out, detail=(k == 'Italian-Dante'))
    save('test_b_position', out)

if __name__ == '__main__' and len(sys.argv) == 1:
    run()

def strip_test(lines, min_len=3):
    """Is a line-initial word 'one extra glyph + an ordinary word'?
    For words of length>=min_len: share whose remainder after dropping the first
    glyph is itself a word type seen >=5 times mid-line (pos 2..n-2).
    Compare line-initial (not paragraph-initial) words with mid-line words.
    Same for line-final words, dropping the last glyph."""
    mid = Counter()
    for L in lines:
        ws = L['words']
        for w in ws[1:-1]:
            mid[w] += 1
    vocab = {w for w, c in mid.items() if c >= 5}
    def share(ws, cut):
        ws = [w for w in ws if len(w) >= min_len]
        return sum(1 for w in ws if cut(w) in vocab) / len(ws), len(ws)
    ini = [L['words'][0] for L in lines if not L['para_start'] and len(L['words']) >= 3]
    fin = [L['words'][-1] for L in lines if len(L['words']) >= 3]
    med = [w for L in lines if len(L['words']) >= 3 for w in L['words'][1:-1]]
    return {
        'initial_dropfirst': share(ini, lambda w: w[1:]), 'medial_dropfirst': share(med, lambda w: w[1:]),
        'final_droplast': share(fin, lambda w: w[:-1]), 'medial_droplast': share(med, lambda w: w[:-1]),
        'initial_whole_in_vocab': share(ini, lambda w: w), 'medial_whole_in_vocab': share(med, lambda w: w),
        'final_whole_in_vocab': share(fin, lambda w: w)}

def run2():
    out = {}
    vz = unitize(load_voynich('ZL3b', ('P',), True))
    vt = unitize(load_voynich('IT2a', ('P',), True))
    sets = {'Voynich-ZL3b': vz, 'Voynich-IT2a': vt}
    for k in REFS:
        skip = 0.3 if k in ('Italian-Manzoni', 'Spanish-Cervantes', 'Italian-Dante') else 0.0
        sets[k] = load_ref(k, max_words=35000, skip_frac=skip)
    for name, lines in sets.items():
        nonpara = [L for L in lines if not L['para_start']]
        r = {'mi_first_glyph_nonpara': mi_excess(nonpara, lambda w: w[0]),
             'strip': strip_test(lines)}
        out[name] = r
        s = r['strip']
        print(f"{name:20s} firstglyph(non-para lines) exc={r['mi_first_glyph_nonpara']['excess']:.4f}  "
              + '  '.join(f"{k}={v[0]:.3f}" for k, v in s.items()))
    save('test_b_position_extra', out)

if __name__ == '__main__' and len(sys.argv) > 1 and sys.argv[1] == 'extra':
    run2()
