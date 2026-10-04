"""v34 cycle 1: do last words of crowded (tight) lines differ from those of roomy lines?
Tightness = -slack (glyph units left before the page's right margin). Null: tightness
shuffled among lines of the same page (keeps every line-final effect, kills crowding).
Positive control: CREMMA Latin (abbreviation marks of the last word). Negative control:
machine-wrapped printed text (Gutenberg). Same statistic for all three.
"""
import sys, os, json, collections
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import v34_lib as V

NP = 2000


def zl_freq():
    c = collections.Counter()
    for ln in open(os.path.join(V.DATA, 'ZL3b-n.txt'), encoding='utf-8', errors='replace'):
        if ln.startswith('<f'):
            t = ln.split('>', 1)[-1]
            import re
            t = re.sub(r'<[^>]*>', '', t).replace(',', '.')
            for w in t.strip().split('.'):
                if w and '?' not in w:
                    c[w] += 1
    return c


def voy_feats(ok, freq):
    lw = [l['words'][-1] for l in ok]
    G = [V.glyphs(w) for w in lw]
    fin = [g[-1] if g else '' for g in G]
    F = {}
    for gl in ['y', 'n', 'l', 'r', 'm', 'g', 's', 'd', 'o']:
        F['final_' + gl] = [f == gl for f in fin]
    F['final_m_or_g'] = [f in ('m', 'g') for f in fin]
    for suf in ['am', 'aiin', 'ain', 'dy', 'ey', 'ol', 'ar', 'al', 'or']:
        F['ends_' + suf] = [w.endswith(suf) for w in lw]
    F['nglyph_last'] = [len(g) for g in G]
    F['logfreq_last'] = [np.log1p(freq.get(w, 0)) for w in lw]
    F['hapax_last'] = [freq.get(w, 0) <= 1 for w in lw]
    F['comp_last'] = [l['comp'] for l in ok]
    F['nwords'] = [len(l['words']) for l in ok]
    F['commas_line'] = [l['raw'].count(',') for l in ok]
    F['last_space_uncertain'] = [l['raw'].rstrip('.').rsplit('.', 1)[-1].count(',') > 0 for l in ok]
    F['gallows_last'] = [any(c in w for c in 'ktpf') for w in lw]
    return F


def lat_feats(ok):
    lw = [l['words'][-1] for l in ok]
    F = {}
    F['abbr_last'] = [V.abbr_count(w) for w in lw]
    F['abbr_last_any'] = [V.abbr_count(w) > 0 for w in lw]
    F['letters_last'] = [V.base_letters(w) for w in lw]
    F['abbr_rest_mean'] = [np.mean([V.abbr_count(w) for w in l['words'][:-1]]) for l in ok]
    F['abbr_last_minus_rest'] = [a - b for a, b in zip(F['abbr_last'], F['abbr_rest_mean'])]
    F['final_letter_m'] = [w[-1:] == 'm' for w in lw]
    F['nwords'] = [len(l['words']) for l in ok]
    return F


def prn_feats(ok):
    lw = [l['words'][-1] for l in ok]
    F = {'len_last': [len(w) for w in lw], 'next_first_len': [l['next_first'] for l in ok],
         'nwords': [len(l['words']) for l in ok]}
    for c in 'esnatom':
        F['final_' + c] = [w[-1:].lower() == c for w in lw]
    return F


def table(name, ok, F, out, covar=None, tag=''):
    t = [-l['slack'] for l in ok]
    g = [l['page'] for l in ok]
    F = {k: np.nan_to_num(np.asarray(v, float), nan=np.nanmedian(np.asarray(v, float))) for k, v in F.items()}
    r = V.multi_perm(t, F, g, NP, covar=covar)
    out[name + tag] = r
    print(f'== {name}{tag}  n={len(ok)} pages={len(set(g))}')
    for k, v in sorted(r.items(), key=lambda kv: -abs(kv[1]['z'])):
        print(f"  {k:24s} r={v['r']:+.3f} z={v['z']:+.2f} p={v['p']:.4f} pFW={v['p_fw']:.4f} mean={v['mean']:.3f}")


if __name__ == '__main__':
    out = {}
    freq = zl_freq()
    vo = [l for l in V.voynich_lines() if l['ok']]
    Fv = voy_feats(vo, freq)
    table('voynich', vo, Fv, out)
    table('voynich', vo, {k: v for k, v in Fv.items() if k != 'nglyph_last'}, out,
          covar=np.array([Fv['nglyph_last']], float), tag='|len')
    la = [l for l in V.latin_lines() if l['ok']]
    Fl = lat_feats(la)
    table('latin', la, Fl, out)
    table('latin', la, {k: v for k, v in Fl.items() if k != 'letters_last'}, out,
          covar=np.array([Fl['letters_last']], float), tag='|len')
    pr = []
    for fn in ['pg218.txt', 'pg23306.txt', 'pg22367.txt', 'pg45334.txt', 'pg2000.txt']:
        pr += V.print_lines(fn)
    rng = np.random.default_rng(1)
    pr = [pr[i] for i in sorted(rng.choice(len(pr), min(len(pr), 3000), replace=False))]
    table('print', pr, prn_feats(pr), out)
    # Voynich: greedy-wrap signature (roomy line <-> long first word of next line)
    nxt = {}
    for l in V.voynich_lines():
        nxt[(l['page'], l['n'])] = l
    vv = [l for l in vo if (l['page'], l['n'] + 1) in nxt]
    table('voynich_next', vv, {'next_first_glyphs': [len(V.glyphs(nxt[(l['page'], l['n'] + 1)]['words'][0])) for l in vv],
                               'next_second_glyphs': [len(V.glyphs(nxt[(l['page'], l['n'] + 1)]['words'][1])) for l in vv]}, out)
    json.dump(out, open(os.path.join(V.CKPT, 'cycle1.json'), 'w'), indent=1)
