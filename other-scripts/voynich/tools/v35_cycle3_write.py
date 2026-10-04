"""v35 cycle 3: which mechanism?  Writes loops/v35_cycle3.txt from data/v35_ckpt/c3.pkl, c3b.pkl, c3_class.pkl, c3_nogal.log."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v28_lib as X, v25_lib as L, v35_lib as V

c3b = X.load('c3b.pkl'); cl = X.load('c3_class.pkl')
def ix(nm, opt=None):
    r = [x for x in c3b if x['name'] == nm and x.get('opt') == opt][0]
    return f"{nm}{'/' + opt if opt else ''} ({r['n']} units, {r['shape']}): top-10 I {r['top']:.2f} vs freq-matched {r['null']:.2f} (p {r['p']:.4f})"
rows = [
 ('V-35.3.1', 'INDISTINGUISHABILITY of look-alike glyphs (new statistic). Corpus split into alternating 400-word blocks; PPMI context rows per half; I(a,b) = cross-half cos(a,b) / sqrt(cos(a,a\') cos(b,b\')), 1 = as alike as a glyph is to itself. Mean I over the 10 most shape-similar pairs; null = random pairs with each glyph replaced by one within +-2 frequency ranks (5,000 draws). Calibration: planted homophone-family cipher (must be ~1), Hangul (featural, distinct jamo)',
  '; '.join(ix(n) for n in ('voy', 'voy_it', 'plant_family', 'hangul', 'tengwar', 'shavian', 'cree', 'borg', 'copiale')) +
  '. Voynich top pairs: cfh/cph 0.93, cph/cth 0.90, f/p 0.94, ch/sh 0.91, cfh/ckh 0.80, ckh/cth 0.64',
  'The Voynich behaves like the PLANTED HOMOPHONE CIPHER (0.81 vs 0.73), not like any featural script (Hangul 0.37, Tengwar 0.26, Shavian 0.29, syllabics 0.17) or real cipher (Borg 0.23, Copiale 0.15). Its most similar glyphs are near-interchangeable in context. Replicates in IT (0.82). Grade A for the measurement, B for the reading'),
 ('V-35.3.2', 'KILL CONTROL: drop the gallows family (k t p f ckh cth cph cfh) and redo 3.1; and redo the v25 Mantel r on the remaining 15 units',
  ix('voy', 'nogallows') + '; ' + ix('voy_it', 'nogallows') + '. Mantel without gallows: ' + open(os.path.join(V.CK, 'c3_nogal.log')).read().strip().replace('\n', ' || '),
  'SPLIT RESULT: interchangeability lives in the gallows family (+ ch/sh); without it I falls to Hangul level (0.34-0.36, p 0.04-0.06). The graded shape-behaviour link does NOT need the gallows (image +0.33 / +0.37 / +0.36, p <= 0.003). Two mechanisms: near-equivalent twins (gallows, ch/sh) plus a featural gradient over the rest. Grade A'),
 ('V-35.3.3', 'KILL CONTROL for a class confound (marks vs letters, tall vs short, finals vs syllables): split each inventory into its 2 top-level shape clusters; partial r given same-cluster, and Mantel p with labels permuted only within clusters',
  '; '.join(f"{nm}: " + ', '.join(f"{k} partial {cl[nm][(k, 'ppmi')][0]:+.2f} / {cl[nm][(k, 'svd')][0]:+.2f} / {cl[nm][(k, 'potts')][0]:+.2f} (within-cluster p {cl[nm][(k, 'ppmi')][2]:.3f} / {cl[nm][(k, 'svd')][2]:.3f} / {cl[nm][(k, 'potts')][2]:.4f}; minor cluster {len(cl[nm][(k, 'split')])} units)" for k in sorted({kk for kk, _ in cl[nm]})) for nm in ('voy', 'tengwar', 'cree', 'shavian', 'borg', 'copiale', 'cherokee', 'deseret')),
  'The Voynich link is not a two-class artefact (+0.27 to +0.38 within clusters). Shavian\'s hand result IS (height class: partial ~0); its image result is not. Grade A'),
 ('V-35.3.V', 'Cycle 3 verdict', 'Invented featural scripts make similar shapes behave SIMILARLY but distinctly; a homophone-family cipher makes them behave IDENTICALLY. The Voynich has both signatures: identical-behaving twins in the gallows / bench family and a graded featural link elsewhere',
  'New, falsifiable reading (grade C): the gallows set (k~t, p~f, and the benched forms) and ch~sh are near-equivalents (homophone-like or free/positional variants of a few underlying signs), while the rest of the inventory is featurally graded. Would support: k/t and p/f substitutable within the same words across the corpus beyond the frequency-matched rate; would kill: a corpus slice (e.g. a single scribe or quire) where k and t contexts diverge as much as Hangul jamo pairs'),
]
hdr = ('# v35 cycle 3 - which mechanism: twins (cipher-like) or graded features (invented-script-like)? (4 Oct 2026). Tools v35_cycle3.py, v35_cycle3b.py, v35_cycle3_write.py.\n'
       '| row | method and control | result | verdict |\n|---|---|---|---|')
L.write_rows(os.path.join(L.LOOPS, 'v35_cycle3.txt'), rows, hdr)
