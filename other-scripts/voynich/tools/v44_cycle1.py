"""v44 cycle 1: does real glyph order inside words save pen effort, against within-word shuffles, compared with
manuscript languages in a 15th-c. cursive, short gibberish writers and the Codex Seraphinianus?
Controls: planted effort-minimiser (must be found with the true model; the randomised-model null must not move);
randomised cost model (1000 glyph->signature permutations) is the null for every corpus.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import v44_lib as L

FN = 'v44_cycle1.txt'
NP = 1000


def fmt(r):
    s, e = r['shuf'], r['edge']
    return (f"S_shuf {100*s['S']:+.1f}% (null {100*s['null_mean']:+.1f}±{100*s['null_sd']:.1f}, z {s['z']:+.1f}, pct {s['pct']:.3f}); "
            f"S_edge {100*e['S']:+.1f}% (null {100*e['null_mean']:+.1f}±{100*e['null_sd']:.1f}, z {e['z']:+.1f}, pct {e['pct']:.3f})")


def main():
    res = {}
    fsig = {k: tuple(v) for k, v in L.load('sig_font_voynich.json').items()}
    V = {'V_ZL': L.words_of(L.voynich('ZL3b')), 'V_IT': L.words_of(L.voynich('IT2a')),
         'V_A': L.words_of(L.voynich('ZL3b', lang='A')), 'V_B': L.words_of(L.voynich('ZL3b', lang='B'))}
    i = 0
    # --- planted controls first
    w = V['V_ZL']; alph = L.alphabet(w, L.VOYNICH_HAND); S = L.sig_array(L.VOYNICH_HAND, alph); c = L.cost_matrix(S)
    for beta in (0.25, 0.5, 1.0, 2.0):
        pw = L.plant(w, alph, c, beta)
        r = L.summarize(pw, L.VOYNICH_HAND, NP); rf = L.summarize(pw, fsig, NP)
        res[f'plant_V_{beta}'] = dict(hand=r, font=rf)
        i += 1
        L.row(FN, f'V-44.1.{i}', f'Control: planted effort-minimiser, Voynich words reordered ~exp(-{beta} x mean hand-model cost) among 24 orderings; scored with the hand model (and font model)',
              f'hand: {fmt(r)} | font: S_shuf {100*rf["shuf"]["S"]:+.1f}% pct {rf["shuf"]["pct"]:.3f}',
              'detected' if r['shuf']['pct'] > 0.95 else 'missed')
    lw = L.words_of(L.v31_corpus('L_msI_Lat'))
    la = L.alphabet(lw, L.CURSIVA); lc = L.cost_matrix(L.sig_array(L.CURSIVA, la))
    for beta in (0.5, 1.0):
        r = L.summarize(L.plant(lw, la, lc, beta), L.CURSIVA, NP)
        res[f'plant_LA_{beta}'] = r; i += 1
        L.row(FN, f'V-44.1.{i}', f'Control: planted effort-minimiser in manuscript Latin (beta {beta}), cursiva model', fmt(r),
              'detected' if r['shuf']['pct'] > 0.95 else 'missed')
    # --- real corpora
    jobs = [(k, w, L.VOYNICH_HAND, 'Voynich hand ductus') for k, w in V.items()]
    jobs += [(k + '/font', w, fsig, 'Voynich EVA-font automatic') for k, w in V.items() if k in ('V_ZL', 'V_IT')]
    for nm in ('L_msI_Lat', 'L_msG_Bav1', 'L_msG_Bav2', 'L_msG_Alem', 'L_msG_Rip', 'L_msI_Ita', 'L_msC_Old', 'L_Isidore', 'L_pgCaesar', 'L_Latin_Lite', 'L_German_Lite'):
        jobs.append((nm, L.words_of(L.v31_corpus(nm)), L.CURSIVA, '15th-c cursiva'))
    gb = []
    for nm in L.gibberish_names(): gb += L.words_of(L.v31_corpus(nm))
    jobs.append(('GIBB_pool', gb, L.CURSIVA, 'cursiva (37 gibberish writers pooled)'))
    jobs.append(('CS_raw', L.words_of(L.cs_lines()), L.CS_HAND, 'CS cursive from Ponzi images'))
    jobs.append(('CS_col', L.words_of(L.cs_lines(collapse=True)), L.CS_HAND, 'CS cursive, repeats collapsed'))
    for k, w, sig, desc in jobs:
        r = L.summarize(w, sig, NP)
        res[k] = r; i += 1
        L.row(FN, f'V-44.1.{i}', f'{k}: saving of real within-word order vs uniform shuffle and edge-kept shuffle; {desc}; null = 1000 random glyph->signature reassignments ({int(r["ntrans"])} transitions, {r["nglyph"]} glyphs)',
              fmt(r), 'saves effort beyond chance' if r['shuf']['pct'] > 0.975 and r['edge']['pct'] > 0.975 else
              ('costs MORE than chance' if r['shuf']['pct'] < 0.025 and r['edge']['pct'] < 0.025 else 'mixed / null'))
        print(k, fmt(r), flush=True)
    # per gibberish writer
    pw = []
    for nm in L.gibberish_names():
        w = L.words_of(L.v31_corpus(nm))
        if sum(len(x) - 1 for x in w if len(x) > 1) < 1500: continue
        r = L.summarize(w, L.CURSIVA, 300); pw.append((nm, r['shuf']['z'], r['edge']['z'])); res['GW_' + nm] = r
    zs = np.array([p[1] for p in pw]); ze = np.array([p[2] for p in pw])
    i += 1
    L.row(FN, f'V-44.1.{i}', f'Gibberish writers one by one (n={len(pw)} with >=1500 transitions), cursiva model, perm-null z',
          f'z_shuf median {np.median(zs):+.1f} (range {zs.min():+.1f}..{zs.max():+.1f}; {int((zs>1.96).sum())} above +1.96); z_edge median {np.median(ze):+.1f} ({int((ze>1.96).sum())} above +1.96)', 'see cycle verdict')
    L.save('cycle1.json', res)


if __name__ == '__main__':
    main()
