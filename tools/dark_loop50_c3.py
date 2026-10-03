"""S-DARK-50 cycle 3: calibration of the shape pictoriality index on scripts with known sign classes.

Linear B (Noto Sans Linear B): ideograms (U+10080-100FA, commodity/animal/object signs, the slot analogue of
   Indus heads) vs syllabograms (U+10000-1007F, phonetic). Frequencies from the DAMOS lines in
   data/derived/dark/loop32_corpora/linb_syll.jsonl (syllable values via Unicode names; ideogram abbreviations via
   a fixed table). Test: AUC(ideogram > syllabogram) on the 5-part shape index and on curv+asym, and a
   frequency-matched permutation of the class label (log2 bins, 1,000x) on the signs that have a frequency.
Egyptian (Noto Sans Egyptian Hieroglyphs): the 24 uniliteral phonograms (Gardiner codes, standard list) vs the
   rest; Gardiner classes A-I (people, gods, body, mammals, birds, reptiles, fish, insects) vs Z (strokes and
   geometric figures) as the positive control; no frequencies available, unweighted.
Cuneiform (Noto Sans Cuneiform): CUNEIFORM SIGN vs CUNEIFORM NUMERIC SIGN, unweighted; whole-script
   distribution as the reference for an abstract script.
Indus on the same 5-part scale (from loop50_signs.json + loop50_roles_seq_raw.json): heads vs far middle AUC.
Proto-Elamite: no glyph font or drawings in the repository, so no shape measure; stated as not measurable.
Output: loop50_cycle3.txt, loop50_calib_signs.json
"""
import sys, json, re, math, collections, time
import numpy as np
sys.path.insert(0, 'tools')
from dark_loop50_shape import render_char, measures_mask, add_indices, auc, KEYS5

SP = '/tmp/claude-0/-home-user-Indus-/874df4c7-80d6-5f08-b42c-eea96a214079/scratchpad/'
OUT = 'data/derived/dark/'
UD = {}
for line in open(OUT + 'loop32_corpora/raw/UnicodeData.txt'):
    f = line.split(';'); UD[int(f[0], 16)] = f[1]
np.random.seed(50); T0 = time.time()
rep = []
def P(*a):
    s = ' '.join(str(x) for x in a); print(s); rep.append(s)
P(f'# LOOP 50 cycle 3: calibration on Linear B, Egyptian, cuneiform ({time.strftime("%Y-%m-%dT%H:%M")})')

def script_rows(font, cps, classer):
    rows = []
    for cp in cps:
        name = UD.get(cp, '')
        cls = classer(cp, name)
        if cls is None: continue
        m = measures_mask(render_char(font, chr(cp)))
        if m is None: continue
        m.update(cp=cp, name=name, cls=cls); rows.append(m)
    return add_indices(rows)

def perm_test(rows, cls_pos, cls_neg, measure, nperm=1000):
    sel = [r for r in rows if r['cls'] in (cls_pos, cls_neg) and r.get('freq', 0) > 0]
    if len(sel) < 10: return None
    vals = np.array([r[measure] for r in sel]); lab = np.array([r['cls'] == cls_pos for r in sel])
    fb = np.array([int(math.log2(r['freq'])) for r in sel])
    obs = vals[lab].mean() - vals[~lab].mean(); nulls = []
    bins = {b: np.where(fb == b)[0] for b in set(fb)}
    for _ in range(nperm):
        pl = lab.copy()
        for b, idx in bins.items(): pl[idx] = lab[np.random.permutation(idx)]
        if pl.any() and (~pl).any(): nulls.append(vals[pl].mean() - vals[~pl].mean())
    nl = np.array(nulls); p = (np.sum(np.abs(nl) >= abs(obs)) + 1) / (len(nl) + 1)
    return dict(n=len(sel), npos=int(lab.sum()), obs=float(obs), lo=float(np.percentile(nl, 2.5)), hi=float(np.percentile(nl, 97.5)), p=float(p))

def report(rows, pos, neg, title):
    pr = [r for r in rows if r['cls'] == pos]; ng = [r for r in rows if r['cls'] == neg]
    P(f'  {title}: {len(pr)} {pos} vs {len(ng)} {neg}')
    for k in list(KEYS5) + ['pict_shape5', 'pict_curv']:
        P(f'     AUC({pos} > {neg}) by {k:12s} = {auc([r[k] for r in pr], [r[k] for r in ng]):.3f}   means {np.mean([r[k] for r in pr]):+.3f} / {np.mean([r[k] for r in ng]):+.3f}')
    for k in ('pict_shape5', 'pict_curv', 'perim'):
        t = perm_test(rows, pos, neg, k)
        if t: P(f'     frequency-matched permutation, {k}: n={t["n"]} ({t["npos"]} {pos}), diff {t["obs"]:+.3f}, null [{t["lo"]:+.3f},{t["hi"]:+.3f}], P {t["p"]:.3f}')

# ---------------------------------------------------------------- Linear B
IDEO = {'VIR': 0x10080, 'MUL': 0x10081, 'CERV': 0x10082, 'EQU': 0x10083, 'OVIS': 0x10086, 'CAP': 0x10088, 'SUS': 0x1008A,
        'BOS': 0x1008C, 'GRA': 0x1008E, 'HORD': 0x1008F, 'OLIV': 0x10090, 'AROM': 0x10091, 'CYP': 0x10092, 'OLE': 0x10095,
        'VIN': 0x10096, 'AES': 0x1009A, 'AUR': 0x1009B, 'LANA': 0x1009D, '*146': 0x1009E, '*152': 0x100A1, 'CORNU': 0x100A0,
        'TELA': 0x100A7, 'TUN': 0x100AA, 'ARM': 0x100AB, 'LUNA': 0x100B5, 'ARB': 0x100B7, 'GAL': 0x100C3, 'HAS': 0x100C6,
        'SAG': 0x100C7, 'PUG': 0x100C9, 'BIG': 0x100CC, 'CUR': 0x100CD, 'CAPS': 0x100CE, 'ROTA': 0x100CF, 'JAC': 0x100D8}
syl_cp = {}
for cp, nm in UD.items():
    m = re.match(r'LINEAR B SYLLABLE B\d+ (\w+)$', nm)
    if m: syl_cp[m.group(1).lower()] = cp
    m = re.match(r'LINEAR B SYMBOL B(\d+)$', nm)
    if m: syl_cp['*' + str(int(m.group(1)))] = cp
freq = collections.Counter()
for line in open(OUT + 'loop32_corpora/linb_syll.jsonl'):
    for t in json.loads(line)['seq']:
        t0 = re.sub(r'[̣̀-ͯ]', '', t).split(';')[0].split(':')[0].split('+')[0]
        if t0 in syl_cp: freq[syl_cp[t0]] += 1
        elif t0 in IDEO: freq[IDEO[t0]] += 1
def lb_class(cp, name):
    if 'SYLLABLE' in name or 'SYMBOL' in name: return 'syllabogram'
    if 'IDEOGRAM' in name: return 'ideogram'
    return None
LB = script_rows(SP + 'NotoSansLinearB-Regular.ttf', range(0x10000, 0x100FB), lb_class)
for r in LB: r['freq'] = freq.get(r['cp'], 0)
P(f'\n== Linear B: {len(LB)} glyphs; with a DAMOS frequency: syllabograms {sum(1 for r in LB if r["cls"]=="syllabogram" and r["freq"])}, ideograms {sum(1 for r in LB if r["cls"]=="ideogram" and r["freq"])}')
report(LB, 'ideogram', 'syllabogram', 'Linear B')
# vessel ideograms (B200+) are the most picture-like; named vs unnamed ideograms
named = [r for r in LB if r['cls'] == 'ideogram' and re.search(r'B\d+[FM]? \w', r['name'])]
P(f'  ideograms with a Unicode object name (pictorial by label): {len(named)}, mean pict_shape5 {np.mean([r["pict_shape5"] for r in named]):+.3f}; '
  f'syllabograms {np.mean([r["pict_shape5"] for r in LB if r["cls"]=="syllabogram"]):+.3f}')
fr = sorted([r for r in LB if r['freq'] > 0], key=lambda r: -r['freq'])
P('  commonest 10 syllabograms: ' + ', '.join(f'{r["name"].split()[-1]}:{r["pict_shape5"]:+.1f}' for r in fr if r['cls'] == 'syllabogram')[:200])
P('  commonest 10 ideograms:    ' + ', '.join(f'{" ".join(r["name"].split()[3:]) or r["name"].split()[-1]}:{r["pict_shape5"]:+.1f}' for r in fr if r['cls'] == 'ideogram')[:220])
P(f'  Spearman(pict_shape5, log freq) over signs with freq>0: {np.corrcoef(np.argsort(np.argsort([r["pict_shape5"] for r in fr])), np.argsort(np.argsort([math.log(r["freq"]) for r in fr])))[0,1]:+.3f} (n={len(fr)})')

# ---------------------------------------------------------------- Egyptian
UNILIT = {'G001', 'M017', 'D036', 'G043', 'D058', 'Q003', 'I009', 'G017', 'N035', 'D021', 'O004', 'V028', 'AA001', 'F032',
          'O034', 'S029', 'N037', 'N029', 'V031', 'W011', 'X001', 'V013', 'D046', 'I010', 'Z004'}
def eg_class(cp, name):
    m = re.match(r'EGYPTIAN HIEROGLYPH ([A-Z]+)(\d+)([A-Z]*)$', name)
    if not m: return None
    code = m.group(1) + m.group(2) + m.group(3)
    if code in UNILIT: return 'uniliteral'
    letter = m.group(1)
    if letter in ('A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L'): return 'living'   # people, gods, body, animals
    if letter == 'Z': return 'strokes'
    return 'other'
EG = script_rows(SP + 'NotoSansEgyptianHieroglyphs-Regular.ttf', range(0x13000, 0x13430), eg_class)
P(f'\n== Egyptian: {len(EG)} glyphs; classes {dict(collections.Counter(r["cls"] for r in EG))}')
report(EG, 'living', 'strokes', 'Egyptian positive control')
report(EG, 'uniliteral', 'living', 'Egyptian phonetic vs pictorial-semantic')
report(EG, 'uniliteral', 'strokes', 'Egyptian phonetic vs strokes')
uni = sorted([r for r in EG if r['cls'] == 'uniliteral'], key=lambda r: r['pict_shape5'])
P('  uniliterals by shape index: ' + ', '.join(f'{r["name"].split()[-1]}:{r["pict_shape5"]:+.1f}' for r in uni))

# ---------------------------------------------------------------- cuneiform
def cu_class(cp, name):
    if name.startswith('CUNEIFORM NUMERIC SIGN'): return 'numeric'
    if name.startswith('CUNEIFORM SIGN'): return 'sign'
    return None
CU = script_rows(SP + 'NotoSansCuneiform-Regular.ttf', range(0x12000, 0x12470), cu_class)
P(f'\n== Cuneiform: {len(CU)} glyphs; classes {dict(collections.Counter(r["cls"] for r in CU))}')
report(CU, 'sign', 'numeric', 'Cuneiform signs vs numerals')

# ---------------------------------------------------------------- Indus on the same 5-part scale
S = {m['w']: m for m in json.load(open(OUT + 'loop50_signs.json'))}
IN = [dict(r, cp=r['w'], name=f'W{r["w"]}') for r in S.values()]
add_indices(IN, base=[r for r in IN if not r['numeral']])
ROLES = {r['w']: r for r in json.load(open(OUT + 'loop50_roles_seq_raw.json'))}
for r in IN:
    rr = ROLES.get(r['w']); r['cls'] = rr['modal'] if rr else None; r['freq'] = rr['n'] if rr else 0
P(f'\n== Indus on the same 5-part scale (seq_raw modal roles, >= 5 tokens)')
INN = [r for r in IN if r['cls'] and not r['numeral']]
report(INN, 'CLOSER', 'NAME_FAR', 'Indus heads vs far middle')
report(INN, 'OPENER', 'NAME_FAR', 'Indus openers vs far middle')
report(INN, 'NAME_ADJ', 'NAME_FAR', 'Indus adjacent vs far middle')
hd = sorted([r for r in INN if r['cls'] == 'CLOSER'], key=lambda r: r['pict_shape5'])
P('  heads by shape index: ' + ', '.join(f'W{r["w"]}:{r["pict_shape5"]:+.1f}' for r in hd))
no617 = [r for r in INN if r['w'] != 617]
P(f'  without W617: AUC heads vs far middle pict_shape5 = {auc([r["pict_shape5"] for r in no617 if r["cls"]=="CLOSER"], [r["pict_shape5"] for r in no617 if r["cls"]=="NAME_FAR"]):.3f}, '
  f'pict_curv = {auc([r["pict_curv"] for r in no617 if r["cls"]=="CLOSER"], [r["pict_curv"] for r in no617 if r["cls"]=="NAME_FAR"]):.3f}')
t = perm_test(no617, 'CLOSER', 'NAME_FAR', 'pict_shape5'); P(f'  without W617, frequency-matched permutation pict_shape5: diff {t["obs"]:+.3f} null [{t["lo"]:+.3f},{t["hi"]:+.3f}] P {t["p"]:.3f}')
t = perm_test(no617, 'CLOSER', 'NAME_FAR', 'perim'); P(f'  without W617, frequency-matched permutation perim: diff {t["obs"]:+.3f} null [{t["lo"]:+.3f},{t["hi"]:+.3f}] P {t["p"]:.3f}')
P('\n== Proto-Elamite: no glyph drawings or font in the repository (other-scripts/proto-elamite holds transliterations only); shape pictoriality not measurable here.')
json.dump(dict(linear_b=LB, egyptian=EG, cuneiform=CU), open(OUT + 'loop50_calib_signs.json', 'w'))
P(f'done {time.time()-T0:.0f}s')
open(OUT + 'loop50_cycle3.txt', 'w').write('\n'.join(rep))
