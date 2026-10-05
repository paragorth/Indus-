"""v51 cycle 2: massive random mapping search with held-out re-test.

stage 'pick': from discovery rows (TAG_d), score every mapping by
    gain(c, m) = speech(c@d, m) - speech(c~shuf@d, m)
  where speech = signed sum of metrics standardised over all rows. Writes the
  top-K mapping ids per corpus plus K random mapping ids to v51_ckpt/c2_pick_<c>.txt
stage 'test': from held-out rows (TAG_h), compare held-out gain of top-K vs random-K.
  Also: vowel recovery (for alphabetic controls, are true vowels assigned the
  vowel class more often in top mappings than in all mappings?) and, for the
  Voynich, which glyph ranks the top mappings make vowels.
"""
import sys, os, json, glob
from collections import defaultdict
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v51_lib as V
from v51_analyze import load

MET = sys.argv[3].split(',') if len(sys.argv) > 3 else V.METRICS
CORP = ['V-ZL', 'la', 'de', 'cs', 'eo', 'he', 'G-selfcit', 'G-mk2']
K = 15

def speech_table(rows, ref=None):
    X = np.array([[r[k] for k in MET] for r in rows])
    mu, sd = (X.mean(0), X.std(0) + 1e-9) if ref is None else ref
    s = ((X - mu) / sd * np.array([V.SIGN[k] for k in MET])).sum(1)
    T = defaultdict(list)
    for r, v in zip(rows, s):
        T[(r['corpus'].split('@')[0], r['m'])].append(v)
    return {k: float(np.mean(v)) for k, v in T.items()}, (mu, sd)

def gains(T, c):
    ms = sorted(m for (cc, m) in T if cc == c and (c + '~shuf', m) in T)
    return ms, np.array([T[(c, m)] - T[(c + '~shuf', m)] for m in ms])

def pick(tag):
    rows = load(tag)
    T, ref = speech_table(rows)
    json.dump([list(ref[0]), list(ref[1])], open(os.path.join(V.CKPT, 'c2_ref.json'), 'w'))
    rng = np.random.default_rng(5)
    allids = set()
    for c in CORP:
        ms, g = gains(T, c)
        o = np.argsort(-g)
        top = [ms[i] for i in o[:K]]
        rnd = [int(x) for x in rng.choice([m for m in ms if m not in top], K, replace=False)]
        open(os.path.join(V.CKPT, f'c2_pick_{c}.txt'), 'w').write(json.dumps({'top': top, 'rnd': rnd, 'gain_mean': float(g.mean()), 'gain_top': float(g[o[:K]].mean())}))
        allids |= set(top) | set(rnd)
        print(c, len(ms), 'disc gain mean %.3f sd %.3f top-%d %.3f' % (g.mean(), g.std(), K, g[o[:K]].mean()))
    open(os.path.join(V.CKPT, 'c2_ids.txt'), 'w').write('\n'.join(map(str, sorted(allids))))
    print('ids', len(allids))

VOW = {'la': 'aeiouy', 'de': 'aeiouyäöü', 'eo': 'aeiou', 'it': 'aeiouàèéìòù'}

def test(tagd, tagh):
    ref = json.load(open(os.path.join(V.CKPT, 'c2_ref.json'))); ref = (np.array(ref[0]), np.array(ref[1]))
    Td, _ = speech_table(load(tagd), ref)
    Th, _ = speech_table(load(tagh), ref)
    C = V.corpora(include_nulls=False)
    out = []
    from scipy.stats import mannwhitneyu, spearmanr
    for c in CORP:
        P = json.loads(open(os.path.join(V.CKPT, f'c2_pick_{c}.txt')).read())
        gt = [Th[(c, m)] - Th[(c + '~shuf', m)] for m in P['top'] if (c, m) in Th]
        gr = [Th[(c, m)] - Th[(c + '~shuf', m)] for m in P['rnd'] if (c, m) in Th]
        u = mannwhitneyu(gt, gr, alternative='greater').pvalue if gt and gr else float('nan')
        # does discovery gain predict held-out gain across all re-tested mappings?
        ids = P['top'] + P['rnd']
        dd = [Td[(c, m)] - Td[(c + '~shuf', m)] for m in ids if (c, m) in Th]
        hh = [Th[(c, m)] - Th[(c + '~shuf', m)] for m in ids if (c, m) in Th]
        rho = spearmanr(dd, hh)[0] if len(dd) > 3 else float('nan')
        line = f'{c}: held-out gain top {np.mean(gt):+.3f} vs random {np.mean(gr):+.3f} (MW p {u:.3f}); rho(disc, held) {rho:+.2f}'
        # vowel slots chosen by top mappings
        base = c if not c.startswith('G-') else c
        if base in C:
            ri = V.rank_index(C[base]); inv = {r: u_ for u_, r in ri.items()}
            vt = np.zeros(V.NSLOT); va = np.zeros(V.NSLOT)
            for m in P['top']:
                M = V.random_mapping(np.random.default_rng(1000 + m)); vt += [s['cls'] == 'V' for s in M['slots']]
            vt /= len(P['top'])
            for m in range(800):
                M = V.random_mapping(np.random.default_rng(1000 + m)); va += [s['cls'] == 'V' for s in M['slots']]
            va /= 800
            lift = vt - va
            if c in VOW:
                tv = [ri[x] for x in VOW[c] if x in ri and ri[x] < 20]
                tc = [r for r in range(20) if r not in tv and r in inv]
                line += f' | vowel lift on true vowels {lift[tv].mean():+.3f} vs consonants {lift[tc].mean():+.3f}'
            if c == 'V-ZL':
                o = np.argsort(-lift[:20])
                line += ' | most vowel-like glyphs (top-20 ranks): ' + ' '.join(f'{inv[r]}{lift[r]:+.2f}' for r in o[:6]) + ' / least: ' + ' '.join(f'{inv[r]}{lift[r]:+.2f}' for r in o[-4:])
        out.append(line)
    return '\n'.join(out)

def profile(tag):
    """Pre-registered replication of V-51.1.6 on fresh mappings: per-metric paired z, corpus vs own shuffle."""
    rows = load(tag)
    T = defaultdict(dict)
    for r in rows:
        T[(r['corpus'].split('@')[0], r['m'])] = r
    out = []
    for c in CORP:
        ms = sorted(m for (cc, m) in T if cc == c and (c + '~shuf', m) in T)
        z = []
        for k in V.METRICS:
            d = np.array([V.SIGN[k] * (T[(c, m)][k] - T[(c + '~shuf', m)][k]) for m in ms])
            z.append(d.mean() / (d.std(ddof=1) / np.sqrt(len(d))))
        out.append(f'{c} (n={len(ms)}): ' + ' '.join(f'{k}:{v:+.1f}' for k, v in zip(V.METRICS, z)))
    return '\n'.join(out)

if __name__ == '__main__':
    if sys.argv[1] == 'profile':
        print(profile(sys.argv[2])); sys.exit()
    if sys.argv[1] == 'pick':
        pick(sys.argv[2])
    else:
        print(test(sys.argv[2].split(',')[0], sys.argv[2].split(',')[1]))
