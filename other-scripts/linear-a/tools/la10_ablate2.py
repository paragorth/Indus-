#!/usr/bin/env python3
"""LA-10 length-preserving ablation (cycle 2). python3 la10_ablate2.py run_file.json [...]
The cycle-1 affix ablation fired on the shuffled target because removing affixes from 1-syllable roots shortens words.
Here every null is length-matched: removing prefixes (suffixes) adds the lost syllables to the root (root_len += p x mean
affix syllables), so the test asks whether the material is an AFFIX (a reused, position-fixed string) rather than whether
words need length. Also: vowel dissimilation (harm_copy < 0) and stem consonants (v2) as separate nulls.
Run with the same LA10_V2 setting as the run file. Writes <run>.abl2.json beside it."""
import sys, os, json, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import la10_common as L

def u_of(g): return L.encode(g)

def main(path, reps=12):
    d = json.load(open(path)); full, sd, n, sp = d['target_fp'], d['target_sd'], d['n'], d['spelling']
    L.WDIST = d.get('wdist', 0.3)
    g0 = dict(d['best']); seeds = [777 + i for i in range(reps)]
    def ev(g): return L.evaluate(u_of(g), full, sd, n, seeds, sp)
    base = ev(g0); out = {'_base': base, '_noise': L.evaluate(u_of(g0), full, sd, n, [877 + i for i in range(reps)], sp) - base}
    asyl = 1 + g0['aff_2syl']
    def lenfix(g, p):
        g['root_len'] = min(4.0, g['root_len'] + p * asyl)
    tests = {}
    g = dict(g0); g['pre_p'] = 0.0; lenfix(g, g0['pre_p']); tests['prefixing_lenfix'] = g
    g = dict(g0); g['suf_p'] = 0.0; lenfix(g, g0['suf_p']); tests['suffixing_lenfix'] = g
    g = dict(g0); g['pre_p'] = g['suf_p'] = 0.0; lenfix(g, g0['pre_p'] + g0['suf_p']); tests['all_affixes_lenfix'] = g
    g = dict(g0); g['pre_p'], g['suf_p'], g['pre_n'], g['suf_n'] = g0['suf_p'], g0['pre_p'], g0['suf_n'], g0['pre_n']; tests['swap_pre_suf'] = g
    g = dict(g0); g['harm_copy'] = 0.0; tests['harm_copy'] = g
    g = dict(g0); g['harm_copy'] = abs(g0['harm_copy']); tests['flip_harm_copy'] = g
    g = dict(g0); g['harm_fb'] = 0.0; tests['harm_fb'] = g
    g = dict(g0); g['ocp_id'] = 0.0; tests['ocp_id'] = g
    g = dict(g0); g['ocp_place'] = 0.0; tests['ocp_place'] = g
    g = dict(g0); g['ocp_id'] = g['ocp_place'] = 0.0; tests['ocp'] = g
    g = dict(g0); g['cluster'] = 0.0; tests['clusters'] = g
    g = dict(g0); g['coda_med'] = 0.0; tests['codas'] = g
    g = dict(g0); g['fv_str'] = 0.0; tests['final_vowel'] = g
    g = dict(g0); g['reuse'] = 0.0; tests['reuse'] = g
    g = dict(g0); g['onset0_med'] = 0.0; tests['hiatus'] = g
    if 'stem_c' in g0:
        g = dict(g0); g['stem_c'] = 0.0; tests['stem_c'] = g
    for k, g in tests.items(): out[k] = ev(g) - base
    json.dump(out, open(path.replace('.json', '.abl2.json'), 'w'), indent=1)
    thr = max(15.0, 3 * abs(out['_noise']))
    print(os.path.basename(path), f"base {base:.1f} noise {out['_noise']:+.1f} thr {thr:.1f} | " +
          ' '.join(f"{k} {v:+.1f}{'*' if v > thr else ''}" for k, v in out.items() if not k.startswith('_')))

if __name__ == '__main__':
    for p in sys.argv[1:]: main(p)
