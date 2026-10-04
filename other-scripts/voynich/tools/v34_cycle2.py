"""v34 cycle 2: positive control through the SAME image pipeline. CREMMA Latin pages go
through v18 word alignment + v34 line-end re-measurement exactly like the Voynich pages;
abbreviation marks of the last token come from the raw ALTO transcription.
Also: last-word squeeze (comp) and whole-line density, which share the measurement
coupling in both scripts, so their slopes are compared Latin vs Voynich, not vs zero.
Split-half replication (recto vs verso pages for Voynich; manuscripts halves for Latin).
"""
import sys, os, json
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import v34_lib as V
from v34_cycle1 import table, voy_feats, zl_freq

NP = 2000


def dens(l):
    """whole-line px per glyph relative to page unit (excluding last word)."""
    G = sum(l['g'][:-1]) + 0.6 * (len(l['g']) - 1)
    return (l['x1'][-2] - l['x0'][0]) / G / l['unit']


def lat_img_feats(ok):
    lt = [l['rawtoks'][-1] for l in ok]
    F = {'abbr_last': [V.abbr_count(w) for w in lt],
         'abbr_last_any': [V.abbr_count(w) > 0 for w in lt],
         'letters_last': [V.base_letters(w) for w in lt],
         'abbr_rest_mean': [np.mean([V.abbr_count(w) for w in l['rawtoks'][:-1]] or [0]) for l in ok],
         'comp_last': [l['comp'] for l in ok],
         'dens_rest': [dens(l) for l in ok],
         'nwords': [len(l['words']) for l in ok]}
    F['abbr_last_minus_rest'] = [a - b for a, b in zip(F['abbr_last'], F['abbr_rest_mean'])]
    return F


if __name__ == '__main__':
    out = {}
    la = [l for l in V.latin_img_lines() if l['ok']]
    Fl = lat_img_feats(la)
    table('latin_img', la, Fl, out)
    table('latin_img', la, {k: v for k, v in Fl.items() if k != 'letters_last'}, out,
          covar=np.array([Fl['letters_last']], float), tag='|len')
    vo = [l for l in V.voynich_lines() if l['ok']]
    Fv = voy_feats(vo, zl_freq()); Fv['dens_rest'] = [dens(l) for l in vo]
    keep = ['comp_last', 'dens_rest', 'nglyph_last', 'nwords', 'final_m', 'final_g', 'final_n', 'final_y']
    table('voynich', vo, {k: Fv[k] for k in keep}, out)
    # split halves
    for nm, sel in [('recto', lambda l: l['page'].endswith('r')), ('verso', lambda l: l['page'].endswith('v'))]:
        idx = [i for i, l in enumerate(vo) if sel(l)]
        table('voynich_' + nm, [vo[i] for i in idx], {k: [Fv[k][i] for i in idx] for k in keep}, out)
    mss = sorted(set(l['ms'] for l in la))
    for half, ms_set in [('A', mss[0::2]), ('B', mss[1::2])]:
        idx = [i for i, l in enumerate(la) if l['ms'] in ms_set]
        table('latin_img_' + half, [la[i] for i in idx], {k: [Fl[k][i] for i in idx] for k in ['abbr_last', 'abbr_last_any', 'comp_last', 'dens_rest', 'letters_last']}, out)
    # tight-vs-roomy contrast table for readability (page-wise median split)
    for nm, L, F in [('voynich', vo, Fv), ('latin_img', la, Fl)]:
        med = {}
        for l in L:
            med.setdefault(l['page'], []).append(l['slack'])
        tight = np.array([l['slack'] < np.median(med[l['page']]) for l in L])
        for k in ['comp_last', 'dens_rest'] + (['abbr_last', 'abbr_last_any', 'letters_last'] if nm != 'voynich' else ['final_m', 'final_g', 'nglyph_last']):
            a = np.asarray(F[k], float)
            print(f'{nm:10s} {k:16s} tight {np.nanmean(a[tight]):.3f}  roomy {np.nanmean(a[~tight]):.3f}')
    json.dump(out, open(os.path.join(V.CKPT, 'cycle2.json'), 'w'), indent=1)
