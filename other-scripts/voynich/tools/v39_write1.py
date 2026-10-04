"""v39 cycle 1 rows."""
import numpy as np
import v39_lib as L, v39_report as R

FN = 'v39_cycle1.txt'
D = R.all_results('c1_')
for _k in D: D[_k]['pLI'] = D[_k]['pL'] + D[_k]['pI']
M = R.MEAS
KEYM = ['h21', 'ttr', 'zipf', 'arrow', 'gap', 'pLI', 'pG', 'auc']
R.MEAS.append('pLI')


def g(k, m): return D[k][m]


def zline(cmp, ms=KEYM):
    return '; '.join(f"{m} {R.fmt(c['v'], m)} (null {R.fmt(c['mu'], m)}+-{R.fmt(c['sd'], m)}, z {c['z']:+.1f})"
                     for m, c in cmp.items() if m in ms)


with open(f'{L.LOOPS}/{FN}', 'w') as f:
    f.write('# v39 cycle 1 - COLLAPSE THE TWINS: merge the v35 look-alike pairs and rerun the battery (4 Oct 2026). Tools v39_lib.py, v39_cycle1.py, v39_write1.py; checkpoints data/v39_ckpt/c1_*.json\n')
    f.write('# Battery: h2/h1 (h21), types per 10k tokens (ttr), Zipf slope, v23 word-FREQUENCY arrow survivors /400 probes, v33 gap ratio (6 rules x 2 blocks), v31 classifier P(language) pL / P(generator) pG (trained without the source corpus), v21 forgery AUC (F3 junction forger, ridge).\n')
    f.write('| row | method and control | result | verdict |\n|---|---|---|---|\n')

nullV5 = [k for k in D if k.startswith('V|rand|(5,')]
nullV1 = [k for k in D if k.startswith('V|rand|(1,')]
langs = {'la': 'Latin herbal', 'de': 'Alemannic German', 'it': 'Italian herbal'}
ref = {m: np.mean([g(f'{lg}|cfg|none', m) for lg in langs]) for m in M}

# 1.1 Voynich configs
res = []
for c in ('all5', 'fam', 'gal4'):
    cmp = R.compare(D, f'V|cfg|{c}', nullV5)
    res.append(f"[{c}] " + zline(cmp))
none = 'unmerged: ' + R.line(D, 'V|cfg|none', KEYM)
R_ = R.compare(D, 'V|cfg|all5', nullV5)
L.row(FN, 'V-39.1.1', f'Voynich ZL (202 pages, 34.7k tokens): merge ALL twin pairs (all5: k=t, p=f, ckh=cth, cph=cfh, ch=sh), the whole gallows family (fam: k=t=p=f, ckh=cth=cph=cfh, ch=sh) and the four gallows pairs (gal4). NULL: {len(nullV5)} random merges of 5 pairs, each unit within +-2 frequency ranks of the real twin',
      none + ' // ' + ' // '.join(res),
      'Merging the twins does NOT make the Voynich language-like: P(language or conlang) stays 0.13-0.15 (null 0.12), P(generator) 0.43-0.54 (null 0.58), frequency arrow 0/400, gap ratio 0.94-0.96 (languages 0.46-0.60), forgery AUC 0.92-0.94. What DOES move beyond the null is the split signature itself: types per 10k tokens fall 0.29 -> 0.24 / 0.22 (z -4.8 / -6.7), the Zipf slope steepens (z -5), h2/h1 drops (z -2.7): many word types differ only by a twin, as in a homophone split. Grade A (negative for language-likeness; positive for twin = variant)')
# 1.2 singles
res = []
for c in ('kt', 'pf', 'KT', 'PF', 'CS'):
    cmp = R.compare(D, f'V|cfg|{c}', nullV1)
    res.append(f"[{c}] pL {R.fmt(cmp['pL']['v'],'pL')} (z {cmp['pL']['z']:+.1f}), arrow {int(cmp['arrow']['v'])}, gap {cmp['gap']['v']:.2f} (z {cmp['gap']['z']:+.1f}), ttr {cmp['ttr']['v']:.3f} (z {cmp['ttr']['z']:+.1f}), h21 z {cmp['h21']['z']:+.1f}")
L.row(FN, 'V-39.1.2', f'Each twin pair merged alone vs {len(nullV1)} random single freq-matched merges', ' ; '.join(res),
      'No single pair moves P(language), the arrow or the gap ratio. k=t and ch=sh each collapse word types far beyond frequency-matched merges (ttr z -10.6 / -10.1); p=f, ckh=cth, cph=cfh are too rare to move ttr. Grade A')
# 1.3 planted languages
res = []
for lg, nm in langs.items():
    o, pf, pp = f'{lg}|cfg|none', f"{lg}|plant|('free', 0)", f"{lg}|plant|('pos', 0)"
    if pf not in D: continue
    info = D[pf]['info']
    s = f"{nm} letters {''.join(info['letters'])}: " + ', '.join(
        f"{m} {R.fmt(g(o, m), m)} -> free {R.fmt(g(pf, m), m)} / pos {R.fmt(g(pp, m), m)}" for m in KEYM)
    res.append(s)
L.row(FN, 'V-39.1.3', 'POSITIVE control / sensitivity: real Latin, German, Italian with a planted homophone split (5 letters at the frequency ranks of the merged Voynich twins, each written as two glyphs at the Voynich minor-variant share; free = random choice, pos = chosen by the preceding letter with 15% noise). Merging by the planted key restores the text exactly (checked: token-identical), so the question is how far the split moved it',
      ' // '.join(res),
      'The battery IS sensitive to a twin split: a Voynich-matched split lowers P(language) by 0.10-0.16, raises P(generator) by 0.07-0.16 and raises types per 10k by 0.06-0.09 in all three languages (German gap ratio jumps 0.46 -> 1.65). It does NOT erase the frequency arrow (Latin 47 -> 49, Italian 232 -> 208). Merging by the planted key restores everything exactly. So un-splitting can move a text toward language; the size of the move is ~0.1-0.15 in P(language). Grade A (calibration)')
# 1.4 random merges in languages
res = []
for lg, nm in langs.items():
    ks = [k for k in D if k.startswith(f'{lg}|rand|')]
    if not ks: continue
    v = {m: np.array([g(k, m) for k in ks]) for m in KEYM + ['pL']}
    res.append(f"{nm} (n={len(ks)}): pL {g(f'{lg}|cfg|none','pL'):.2f} -> {v['pL'].mean():.2f}+-{v['pL'].std():.2f} (more language-like in {int((v['pL'] > g(f'{lg}|cfg|none','pL')).sum())}/{len(ks)}), arrow {int(g(f'{lg}|cfg|none','arrow'))} -> {v['arrow'].mean():.0f}, gap {g(f'{lg}|cfg|none','gap'):.2f} -> {v['gap'].mean():.2f}, ttr {g(f'{lg}|cfg|none','ttr'):.3f} -> {v['ttr'].mean():.3f}")
L.row(FN, 'V-39.1.4', 'NEGATIVE control: random merges of 5 freq-matched letter pairs in the real languages (must not make them more language-like)', ' ; '.join(res),
      'Random merges of distinct letters do not make languages more language-like on balance (Latin 8/15, Italian 4/15 raise pL; German 14/15 by +0.02, smaller than the planted effect 0.10-0.16); the gap ratio rises (more generator-like) in German and Italian. Negative control passes. Grade A')
# 1.5 Copiale
kc = [k for k in D if k.startswith('cop|coprand')]
v = {m: np.array([g(k, m) for k in kc]) for m in KEYM}
s = 'cipher: ' + R.line(D, 'cop|cfg|none', KEYM) + ' // merged by the published key: ' + R.line(D, 'copK|cfg|none', KEYM) + \
    f' // {len(kc)} random merges with the key\'s class sizes: ' + ', '.join(f"{m} {R.fmt(v[m].mean(), m)}+-{R.fmt(v[m].std(), m)}" for m in KEYM)
L.row(FN, 'V-39.1.5', 'POSITIVE control, a real homophone cipher: Copiale (HTR transcription, 13k cipher words split at its space symbols) as written, merged by the Knight-Megyesi-Schaefer 2011 key (homophones -> one plaintext unit), and random merges with the same class sizes', s,
      'POSITIVE CONTROL PASSES STRONGLY: the real homophone cipher reads as a generator (P(gen) 0.97), merged by its true key it leaves the generator class (P(gen) 0.07; P(conlang) 0.90, P(language or conlang) 0.92), types per 10k halve (0.58 -> 0.30), gap ratio falls to the language band (0.78 -> 0.61), Zipf steepens to language (-0.77 -> -0.91); random merges of the same class sizes change nothing (P(gen) 0.95). A homophone cipher merged correctly DOES become language-like on this battery. Grade A')
L.row(FN, 'V-39.1.V', 'Cycle-1 verdict', 'Copiale: gen 0.97 -> 0.07 when merged by its key; planted Latin/German/Italian: P(language) -0.10 to -0.16 from a Voynich-matched split; Voynich merged by its twins: P(language or conlang) 0.15 -> 0.13, P(gen) 0.44 -> 0.50, arrow 0 -> 0, gap 0.92 -> 0.94, while ttr / Zipf / h2 move as a split being undone (z -3 to -7)',
      'Collapsing the twins does NOT make the Voynich language-like on any generator-vs-language measure, although the same battery recovers a real homophone cipher (Copiale) and planted splits. The twins behave like a split (merging collapses word types far beyond random merges), but the generator signatures are NOT artefacts of the split. Grade A (negative)')
print(open(f'{L.LOOPS}/{FN}').read())
