"""Gulf / Near East seals: do Indus signs track the PICTORIAL motif beside them?

Idea under test (counter-intuitive anchor): on Gulf-type and Near-East seals the Indus text sits
beside motifs that are not Indus emblems (scorpion, footprint, antelope, peacock, mating scene,
manger ...). If some signs are logograms with fixed meanings, sign x motif co-occurrence across
independent seals should beat chance, and the same signs at home should prefer the matching
Indus emblem / object class.

Data: data/gulf_seals.csv (Laursen 2010 numbering), sources/notes/{laursen2010,alsindi1999,
parpola1994_neareast}.md, ANCHORS.md section 9 (Gadd 1932 catalogue facts), the Laursen text
(data/derived/laursen-2010-text.txt, footnote 11 and pp. 111-114), and the merged canonical corpus
(seq = strong+probable merges; seq_raw also reported where it matters).

Output: data/derived/strat_gulfmotif.txt. No readings from other people are used, only object facts
(which motif is carved on which seal) and sign identities.
"""
import json, csv, random, math, collections, itertools, sys
random.seed(11)
ROOT = '/home/user/Indus-'
M = json.load(open(f'{ROOT}/data/derived/merged-corpus-canonical.json'))
BR = json.load(open(f'{ROOT}/data/derived/bridge_extended.json'))
out = []
def P(*a):
    s = ' '.join(str(x) for x in a); out.append(s); print(s)

ABROAD = {'Failaka','Hajar','Kish','Luristan','Nippur',"Qala'at al-Bahrain","Ra's al-Junayz",'Susa','Tell Umma',
          'Tello','Tepe Yahya','Ur','Gonur Depe','Salut','Girsu','Karzakan','Saar','Janabiyah','Kalba','Dilmun',
          'Altyn Depe','Shortughai','Miri Qalat'}

def lookup(site, raw):
    for r in M:
        if r['site'] == site and r.get('seq_raw') == raw: return r
    return None

# ---------------------------------------------------------------------------------------------
# 1. The object table. Each row: id, site, format, text source, motif tags, citation.
#    text = list of canonical sign labels (ints = Wells numbers from the corpus 'seq'; strings = signs
#    legible on the object but not in the corpus, e.g. 'crab'). status: full / partial / none.
#    Motif tags are object facts from the cited source, not interpretations of the script.
# ---------------------------------------------------------------------------------------------
def corpus_text(site, raw):
    r = lookup(site, raw)
    if r is None: raise SystemExit(f'not in corpus: {site} {raw}')
    return list(r['seq']), list(r['seq_raw']), r

OBJ = []
def add(oid, site, fmt, stratum, raw, motifs, cite, status='full', text=None, note=''):
    if raw is not None:
        seq, seq_raw, r = corpus_text(site, raw)
    else:
        seq, seq_raw = list(text or []), list(text or [])
    OBJ.append(dict(id=oid, site=site, fmt=fmt, stratum=stratum, seq=seq, seq_raw=seq_raw,
                    motifs=set(motifs), cite=cite, status=status, note=note))

# --- Gulf-type round seals and cylinders found abroad (stratum 'abroad-round') ---
add('L6', "Qala'at al-Bahrain", 'round', 'abroad-round', None, {'antelope'},
    'Laursen 2010 Table 1 no. 6 = Kjaerum 1994 fig. 1726; Parpola 1994 no. 6 (antelope); Laursen n. 11: Kjaerum reads the animal as an antelope with short tail',
    status='partial', text=[91], note='fragment, only the twins legible (gulf_seals.csv)')
add('L7', "Qala'at al-Bahrain", 'round', 'abroad-round', [190,60,55,90,160], {'bull'},
    'Laursen no. 7 = Kjaerum 1994 fig. 1725 = Al-Sindi 1999 no. 279 (large bull, pleated neck); Parpola 1994 no. 5 (bison)')
add('L8', 'Karzakan', 'round', 'abroad-round', [91,31,455,220], {'rhino'},
    'Laursen no. 8 = Srivastava 1991 fig. 55 = Al-Sindi 1999 no. 180: animal is a rhinoceros (Al-Sindi text)')
add('L9', 'Saar', 'round', 'abroad-round', [55,220,91,1,93,31], {'none'},
    'Laursen no. 9 = Al-Sindi 1999 no. 182: lower half empty, no animal (Al-Sindi p. 245; photo)')
add('L10', 'Janabiyah', 'round', 'abroad-round', [71,31,831,55,121,99,55], {'bull'},
    'Laursen no. 10 (fig. 1a): bull')
add('L11', 'Karzakan', 'round', 'abroad-round', [91,32,1,33], {'bull'},
    'Laursen no. 11 (fig. 1b), BBM 14569: bull; pseudo-script after the twins (ANCHORS sec. 10)', note='pseudo-script: only W91 secure')
add('L12', 'Failaka', 'round', 'abroad-round', [716,350,1,90], {'unrecorded'},
    'Laursen no. 12 = Kjaerum 1983 no. 319; motif not recorded in our notes (csv: -)')
add('L13', 'Failaka', 'round', 'abroad-round', [90,817,317], {'bull','scene'},
    'Laursen no. 13 = Kjaerum 1983 no. 279 (Dilmun Type, compass-drilled bull head, extra scene; Laursen n. 12, p. 114)')
add('L14', 'Susa', 'round', 'abroad-round', [365,90,407,604,700], {'bull'}, 'Laursen no. 14 = Amiet 1972 no. 1643: bull')
add('L15', 'Luristan', 'round', 'abroad-round', [91,840,413,831], {'bull'}, 'Laursen no. 15 = Amiet 1973 pl. 23a-b: bull')
add('L16', 'Ur', 'round', 'abroad-round', [528,220,924,340,93], {'bull'},
    'Laursen no. 16 = Gadd 1932 no. 2 (BM 122187): short-horned bull (ANCHORS sec. 9)')
add('L18', 'Ur', 'round', 'abroad-round', [90,405], {'unrecorded'}, 'Laursen no. 18 = Gadd 1932 no. 4 (BM 122188): fragment, fish + 1')
add('L19', 'Ur', 'round', 'abroad-round', [90,1], {'unrecorded'}, 'Laursen no. 19 = Gadd 1932 no. 5 (Penn U.17341): lower half only')
add('L20', 'Ur', 'round', 'abroad-round', None, {'bull','scorpion','footprint'},
    'Laursen no. 20 = Gadd 1932 no. 15, PG-401: crude bull; text band holds the plain man sign, a crab-sign variant, a scorpion and a human footprint (Laursen p. 114)',
    status='partial', text=[90,'crab'])
add('L21', 'Dilmun', 'round', 'abroad-round', [415,803,1,717,354], {'bull'},
    'Laursen no. 21 = Gadd 1932 no. 16 (BM 123208, U.17649): bull, no manger')
add('L22', 'Girsu', 'round', 'abroad-round', [255,13,744,740], {'bull','manger'},
    'Laursen no. 22 = Sarzec & Heuzey pl. 30.3: bull with manger (Laursen p. 111; corpus cult = Trough)')
add('L23', 'Unknown', 'round', 'abroad-round', [467,550,1,740,740], {'bull','manger'},
    'Laursen no. 23 = Gadd 1932 no. 17 (Babylon per Collon 1994): bull with proper manger, Indus-style Decke (Laursen p. 112)')
add('L24', 'Unknown', 'round', 'abroad-round', [190,91,603], {'bull','scene_mating'},
    'Laursen no. 24 = Gadd 1932 no. 18 (BM 123059): mating bull and cow, facing left; twins sign oversized (Laursen n. 11, p. 117)')
add('L26', 'Unknown', 'round', 'abroad-round', [160,384,133,1,90], {'bull'}, 'Laursen no. 26 = Buchanan 1981 no. 1088 / Newell 23: bull')
add('L27', 'Unknown', 'round', 'abroad-round', [725,90,2], {'bird'},
    'Laursen no. 27 = Buchanan 1981 no. 1089 / Newell 876: peacock(?) (Laursen n. 11); corpus symbol Bird')
add('L56', 'Karzakan', 'round', 'abroad-round', [740,740], {'bull','two_animals'},
    'Laursen no. 56 = Al-Sindi 1999 no. 160: two animals upside down to each other, two sign-like marks between them')
add('X1', 'Susa', 'cylinder', 'abroad-round', [924,1,319,31,55,2,150,416], {'bull','manger'},
    'merged corpus: Susa cylinder, symbol Gaur, cult Trough')
add('X2', 'Ur', 'round', 'abroad-round', [415,803,1,328,4,2], {'bull'}, 'merged corpus: Ur round seal, symbol Gaur')
add('X3', 'Ur', 'round', 'abroad-round', [350,90,4], {'bull','manger'}, 'merged corpus: Ur round seal, symbol Gaur, cult Trough')
add('X4', 'Unknown', 'cylinder', 'abroad-round', [32,132,55,90], {'unrecorded'}, 'merged corpus: cylinder, symbol Unknown')
add('X5', 'Unknown', 'cylinder', 'abroad-round', [139,145], {'scene'}, 'merged corpus: stone cylinder, symbol Scene')
add('X6', 'Unknown', 'round', 'abroad-round', [171,35,130,60,17,154,740], {'unrecorded'}, 'merged corpus: round seal, no symbol recorded')
add('X7', 'Kalba', 'round', 'abroad-round', [91], {'unrecorded'}, 'merged corpus: Kalba K4, pseudo-twins (Laursen fig. 10i)', note='pseudo-script')
add('G7', 'Ur', 'cylinder', 'abroad-round', None, {'unicorn','tree'},
    'Gadd 1932 no. 7 (U.11958): cylinder, unicorn bull and tree; cross-hatched fish (+2?) (ANCHORS sec. 9)', status='partial', text=['fish_hatched'])
# --- Square seals / tags found abroad (home format; stratum 'abroad-square') ---
add('K1', 'Kish', 'square', 'abroad-square', [416,840,60,3,220,590,390,740], {'unicorn'}, 'merged corpus Kish, symbol Bull1:L; Parpola 1994 no. 27')
add('K2', 'Kish', 'square', 'abroad-square', [240,643], {'unicorn'}, 'merged corpus Kish, symbol Bull1; Parpola 1994 no. 28')
add('N1', 'Nippur', 'square', 'abroad-square', [151], {'zebu'}, 'merged corpus Nippur, symbol Zebu; Parpola 1994 no. 26')
add('T1', 'Tello', 'square', 'abroad-square', [384,440,390,220], {'tiger','manger'}, 'merged corpus Tello, symbol Tigr, cult Trough')
add('U1', 'Tell Umma', 'tag', 'abroad-square', [127,705,2,4,390], {'unicorn','manger'}, 'Umma bulla, Ashmolean 1931.120; unicorn with cult object (S353)')
add('Go1', 'Gonur Depe', 'square', 'abroad-square', [285,2,235,220,125,590,390,740], {'elephant'}, 'merged corpus Gonur Depe, symbol Elep')
add('Sa1', 'Salut', 'square', 'abroad-square', [595,278,2,4,392], {'bull','manger'}, 'merged corpus Salut, symbol Gaur, cult Trough')
add('Sa2', 'Salut', 'tag', 'abroad-square', [921], {'elephant'}, 'merged corpus Salut tag, symbol Elep')
add('Sh1', 'Shortughai', 'square', 'abroad-square', [3,426], {'rhino'}, 'merged corpus Shortughai, symbol Rhin')
add('Al1', 'Altyn Depe', 'square', 'abroad-square', [415,390], {'unrecorded'}, 'merged corpus Altyn Depe, no symbol')
# --- Gulf-type seals found in the Indus valley (stratum 'home-round') ---
add('L1', 'Chanhu-daro', 'round', 'home-round', [632,390,400,375], {'unicorn'}, 'Laursen no. 1 = CISI C-32, symbol Bull1:W')
add('L2', 'Mohenjo-daro', 'round', 'home-round', [16,405,1,255,324], {'bull'}, 'Laursen no. 2 = M-416, symbol Gaur (western-style bull, Laursen p. 112)')
add('L3', 'Mohenjo-daro', 'round', 'home-round', [820,60,2,832,140,592,55], {'unrecorded'}, 'Laursen no. 3 = M-1369, no symbol recorded')
add('L4', 'Mohenjo-daro', 'round', 'home-round', [140], {'composite'}, 'Laursen no. 4 = M-417, symbol Comp')
add('L5', 'Mohenjo-daro', 'round', 'home-round', [832,256], {'bull'}, 'Laursen no. 5 = M-415, symbol Gaur')

# ---------------------------------------------------------------------------------------------
# 2. Helpers: exact Fisher (hypergeometric), BH-FDR
# ---------------------------------------------------------------------------------------------
def hyper_p(a, b, c, d):
    """one-sided P(X >= a) for table [[a,b],[c,d]] (a = sign & motif)."""
    n1, n2, k, N = a + b, c + d, a + c, a + b + c + d
    lo, hi = max(0, k - n2), min(k, n1)
    def lp(x): return (math.lgamma(n1+1)-math.lgamma(x+1)-math.lgamma(n1-x+1)
                       +math.lgamma(n2+1)-math.lgamma(k-x+1)-math.lgamma(n2-k+x+1)
                       -(math.lgamma(N+1)-math.lgamma(k+1)-math.lgamma(N-k+1)))
    return sum(math.exp(lp(x)) for x in range(a, hi+1))
def bh(ps):
    idx = sorted(range(len(ps)), key=lambda i: ps[i]); q = [0]*len(ps); m = len(ps); prev = 1
    for rank, i in reversed(list(enumerate(idx, 1))):
        prev = min(prev, ps[i]*m/rank); q[i] = prev
    return q

PERSON = {90, 91, 93, 99, 121}
NUMER = {1, 3, 4, 5, 6, 7, 8, 55, 56, 9, 10, 11, 12, 13, 14}
FISH = {220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230}
def classes(seq):
    s = set(seq); c = set()
    if s & PERSON: c.add('PERSON')
    if 91 in s or (121 in s and 99 in s): c.add('TWINS')
    if s & NUMER: c.add('NUMERAL')
    if 55 in s: c.add('TWELVE')
    if s & FISH: c.add('FISH')
    if 740 in s: c.add('JAR')
    return c

# ---------------------------------------------------------------------------------------------
P('# Strategy: sign x pictorial motif on Gulf-type and Near-East seals (counter-intuitive anchor)')
P('# Script: tools/strat_gulfmotif.py. Corpus: merged canonical (seq = strong+probable merges).')
P()
P('## 1. Object table (every row cited; motif = carved picture beside the text, from the cited source)')
P('id | site | format | stratum | text (Wells, reading order) | status | motifs | source')
for o in OBJ:
    P(f"{o['id']} | {o['site']} | {o['fmt']} | {o['stratum']} | {'-'.join(map(str,o['seq']))} | {o['status']} | {','.join(sorted(o['motifs']))} | {o['cite']}" + (f" [{o['note']}]" if o['note'] else ''))
P()
ab = [o for o in OBJ if o['stratum'] == 'abroad-round']
abm = [o for o in ab if 'unrecorded' not in o['motifs']]
P(f'Abroad round/cylinder objects with Indus signs: {len(ab)}; with a recorded motif: {len(abm)}; '
  f'with a motif other than the plain bull: {sum(1 for o in abm if o["motifs"] - {"bull"})}.')
mot = collections.Counter(m for o in abm for m in o['motifs'])
P('Motif counts (abroad round, recorded):', dict(mot))
P('Note: "bull" = short-horned bull / bison / gaur, the standard Gulf-type animal; Laursen (p. 112) says the scorpion,'
  ' footprint, crescent and caprid motifs sit mostly on UNINSCRIBED "local" Bahraini seals, so inscribed seals with these motifs are rare by construction.')
P()

# ---------------------------------------------------------------------------------------------
# 3. (a) sign x motif Fisher + BH + permutation, abroad round seals with recorded motif
# ---------------------------------------------------------------------------------------------
P('## 2. Test (a): sign x motif, abroad round/cylinder seals with recorded motif (n = %d)' % len(abm))
def tokens(o):
    return set(o['seq']) | classes([x for x in o['seq'] if isinstance(x, int)])
motifs_all = sorted(set(m for o in abm for m in o['motifs']))
signs_all = sorted(set(t for o in abm for t in tokens(o)), key=str)
tests = []
for s in signs_all:
    ns = sum(1 for o in abm if s in tokens(o))
    if ns < 2: continue
    for m in motifs_all:
        nm = sum(1 for o in abm if m in o['motifs'])
        if nm < 2: continue
        a = sum(1 for o in abm if s in tokens(o) and m in o['motifs'])
        if a < 2: continue
        b = ns - a; c = nm - a; d = len(abm) - a - b - c
        tests.append((s, m, a, b, c, d, hyper_p(a, b, c, d)))
qs = bh([t[6] for t in tests]) if tests else []
P(f'Pairs tested (sign in >=2 objects, motif on >=2 objects, co-occurrence >=2): {len(tests)}')
P('sign | motif | sign&motif | sign only | motif only | neither | Fisher one-sided p | BH q')
for t, q in sorted(zip(tests, qs), key=lambda z: z[0][6]):
    P(f'{t[0]} | {t[1]} | {t[2]} | {t[3]} | {t[4]} | {t[5]} | {t[6]:.3f} | {q:.3f}')
surv = [t for t, q in zip(tests, qs) if q < 0.1]
P(f'Pairs with BH q < 0.10: {len(surv)}')
# permutation: shuffle motif sets across objects, min p over all pairs
obs_min = min(t[6] for t in tests) if tests else 1.0
mot_sets = [o['motifs'] for o in abm]; tok_sets = [tokens(o) for o in abm]
def min_p(msets):
    best = 1.0
    for s in signs_all:
        idx_s = [i for i, tk in enumerate(tok_sets) if s in tk]
        if len(idx_s) < 2: continue
        for m in motifs_all:
            nm = sum(1 for ms in msets if m in ms)
            if nm < 2: continue
            a = sum(1 for i in idx_s if m in msets[i])
            if a < 2: continue
            p = hyper_p(a, len(idx_s) - a, nm - a, len(abm) - len(idx_s) - nm + a)
            best = min(best, p)
    return best
NPERM = 2000
cnt = 0
for _ in range(NPERM):
    ms = mot_sets[:]; random.shuffle(ms)
    if min_p(ms) <= obs_min: cnt += 1
P(f'Permutation (shuffle motif labels across the {len(abm)} objects, {NPERM}x): smallest observed p = {obs_min:.3f}; '
  f'a smallest p this small or smaller arises in {cnt}/{NPERM} shuffles (P = {cnt/NPERM:.2f}).')
P()
# Non-bull motifs: which signs sit beside them? exhaustive listing (the data are too few for more)
P('### Non-bull motifs and the signs beside them (exhaustive)')
for o in abm:
    nb = o['motifs'] - {'bull'}
    if nb: P(f"  {o['id']} {o['site']}: {','.join(sorted(nb))} <- {'-'.join(map(str,o['seq']))}  classes={','.join(sorted(classes([x for x in o['seq'] if isinstance(x,int)])))}")
P()

# ---------------------------------------------------------------------------------------------
# 4. (c) twins / man signs vs scenes, abroad
# ---------------------------------------------------------------------------------------------
P('## 3. Test (c): person / twins signs vs motif type, abroad round seals (recorded motif)')
def has_person(o): return bool(set(x for x in o['seq'] if isinstance(x, int)) & PERSON)
nonbull = lambda o: bool(o['motifs'] - {'bull'})
a = sum(1 for o in abm if has_person(o) and nonbull(o)); b = sum(1 for o in abm if has_person(o) and not nonbull(o))
c = sum(1 for o in abm if not has_person(o) and nonbull(o)); d = sum(1 for o in abm if not has_person(o) and not nonbull(o))
P(f'person sign present & non-bull motif {a}; person & plain bull {b}; no person & non-bull {c}; no person & plain bull {d}; Fisher one-sided p = {hyper_p(a,b,c,d):.3f}')
a = sum(1 for o in abm if 'TWINS' in classes(o['seq']) and nonbull(o)); b = sum(1 for o in abm if 'TWINS' in classes(o['seq']) and not nonbull(o))
c = sum(1 for o in abm if 'TWINS' not in classes(o['seq']) and nonbull(o)); d = len(abm) - a - b - c
P(f'twins (W91 or 121+99) & non-bull {a}; twins & plain bull {b}; no twins & non-bull {c}; no twins & plain bull {d}; Fisher p = {hyper_p(a,b,c,d):.3f}')
scene = lambda o: bool(o['motifs'] & {'scene', 'scene_mating', 'two_animals'})
a = sum(1 for o in abm if has_person(o) and scene(o)); b = sum(1 for o in abm if has_person(o) and not scene(o))
c = sum(1 for o in abm if not has_person(o) and scene(o)); d = len(abm) - a - b - c
P(f'person & scene (mating, Failaka scene, inverted animals, cylinder scene) {a}; person & no scene {b}; no person & scene {c}; no person & no scene {d}; Fisher p = {hyper_p(a,b,c,d):.3f}')
P('Twins sign and its motif, object by object:')
for o in ab:
    if 'TWINS' in classes([x for x in o['seq'] if isinstance(x, int)]): P(f"  {o['id']} {o['site']}: {','.join(sorted(o['motifs']))}")
P()

# ---------------------------------------------------------------------------------------------
# 5. (d) numeral adjacent to person sign abroad, by motif
# ---------------------------------------------------------------------------------------------
P('## 4. Test (d): numeral next to a person sign (S98b unit) vs motif, abroad round seals')
def num_adj(seq):
    q = [x for x in seq if isinstance(x, int)]
    return [(x, y) for x, y in zip(q, q[1:]) if (x in NUMER and y in PERSON) or (y in NUMER and x in PERSON)]
rows = []
for o in ab:
    pairs = num_adj(o['seq'])
    if has_person(o): rows.append((o, pairs))
with_unit = [(o, p) for o, p in rows if p]; without = [(o, p) for o, p in rows if not p]
P(f'Abroad round texts with a person sign: {len(rows)}; with numeral+person adjacency: {len(with_unit)}')
for o, p in with_unit: P(f"  {o['id']} {o['site']}: {'-'.join(map(str,o['seq']))}  pairs={p}  motif={','.join(sorted(o['motifs']))}")
P('Person sign without an adjacent numeral:')
for o, p in without: P(f"  {o['id']} {o['site']}: {'-'.join(map(str,o['seq']))}  motif={','.join(sorted(o['motifs']))}")
mu = collections.Counter(m for o, p in with_unit for m in o['motifs']); mw = collections.Counter(m for o, p in without for m in o['motifs'])
P('motif counts, numeral+person texts:', dict(mu), '| person without numeral:', dict(mw))
P()

# ---------------------------------------------------------------------------------------------
# 6. (b) home corpus: sign x emblem on home seals, permutation by site
# ---------------------------------------------------------------------------------------------
P('## 5. Test (b): the same signs at home. Sign x emblem on home seals (SEAL* types, Indus-valley sites), site+text dedup')
def emblem_class(sym):
    sym = sym or ''
    if sym.startswith('Bull1') or sym == 'Bull': return 'unicorn'
    if sym in ('Gaur', 'Gavi', 'Bult', 'Bull2', 'Bull3', 'Bison', 'CompBull'): return {'Gaur': 'bull', 'Gavi': 'bull', 'Bult': 'bull', 'Bull2': 'bull', 'Bull3': 'bull', 'CompBull': 'composite'}.get(sym, 'bull')
    if sym == 'Zebu': return 'zebu'
    if sym == 'Elep': return 'elephant'
    if sym == 'Rhin': return 'rhino'
    if sym == 'Tigr' or sym == 'Htgr': return 'tiger'
    if sym == 'Buff': return 'buffalo'
    if sym.startswith('Goat'): return 'goat'
    if sym == 'Hare': return 'hare'
    if sym == 'Fish': return 'fish'
    if sym == 'Bird': return 'bird'
    if sym in ('Scene', 'Anth'): return 'scene/anthropomorph'
    if sym in ('Comp', 'Mult'): return 'composite'
    if sym in ('Phyt', 'Pipal'): return 'plant'
    if sym in ('', '-', '?', 'Unknown'): return None
    return 'other'
home_seals = []; seen = set(); META = {}
IN_TABLE = {(o['site'], tuple(o['seq_raw'])) for o in OBJ if o['stratum'] != 'home-round'}
for r in M:
    if r['site'] in ABROAD or not r['type'].startswith('SEAL'): continue
    # unprovenanced round/cylinder seals are the Gadd/Buchanan Gulf-type pieces: not home
    if r['site'] == 'Unknown' and r['type'] in ('SEAL:C', 'SEAL:R', 'SEAL:CY'): continue
    if (r['site'], tuple(r['seq_raw'])) in IN_TABLE: continue
    s = tuple(r['seq'])
    if not s or (r['site'], s) in seen: continue
    e = emblem_class(r['symbol'])
    if e is None: continue
    seen.add((r['site'], s))
    home_seals.append((r['site'], s, e, r['cult'] or ''))
    META[(r['site'], s)] = (r['cisi'], r['type'], r['symbol'])
ec = collections.Counter(e for _, _, e, _ in home_seals)
P(f'Home seals with a recorded emblem: {len(home_seals)}; emblem classes: {dict(ec.most_common())}')
P('Also "manger" = cult field Trough or S*/R*/H* cult-stand codes; "none" = cult field - with an emblem absent is not recoverable from these fields.')
has_manger = lambda cult: cult not in ('', '-', '?')
# pre-registered pairs mirroring the abroad table
PRE = [  # (sign or class, home emblem class, why)
    ('PERSON', 'bull', 'person signs sit beside the bull abroad'),
    ('TWINS', 'bull', 'twins beside bull abroad (L10, L11, L15, L24)'),
    ('PERSON', 'scene/anthropomorph', 'persons beside scenes abroad (L13, L24)'),
    ('TWINS', 'scene/anthropomorph', 'twins on the mating-scene seal L24'),
    ('PERSON', 'rhino', 'twins beside rhinoceros L8'),
    ('TWINS', 'rhino', 'twins beside rhinoceros L8'),
    ('PERSON', 'bird', 'man beside peacock L27'),
    (90, 'bird', 'man beside peacock L27'),
    ('FISH', 'fish', 'fish sign beside fish emblem?'),
    (220, 'fish', 'plain fish beside fish emblem?'),
    ('FISH', 'rhino', 'fish beside rhinoceros L8'),
    ('JAR', 'bull', 'doubled jar on bull seals L23, L56'),
    (740, 'bull', 'jar on bull seals'),
    (415, 'bull', 'pitchfork opens two bull seals (L21, X2)'),
    (55, 'bull', '12 on bull seals L7, L10, X1'),
    ('TWELVE', 'bull', '12 on bull seals'),
    (1, 'bull', 'single stroke on bull seals'),
    ('NUMERAL', 'bull', 'numerals on bull seals'),
]
def home_table(sig, emb):
    def has(s):
        c = classes(s)
        return (sig in c) if isinstance(sig, str) else (sig in s)
    a = sum(1 for _, s, e, _ in home_seals if has(s) and e == emb); b = sum(1 for _, s, e, _ in home_seals if has(s) and e != emb)
    c = sum(1 for _, s, e, _ in home_seals if not has(s) and e == emb); d = len(home_seals) - a - b - c
    return a, b, c, d
def site_perm(sig, emb, n=2000):
    """shuffle emblem labels within site; P(a_perm >= a_obs)."""
    a0 = home_table(sig, emb)[0]
    def has(s):
        c = classes(s)
        return (sig in c) if isinstance(sig, str) else (sig in s)
    by_site = collections.defaultdict(list)
    for st, s, e, _ in home_seals: by_site[st].append((has(s), e))
    cnt = 0
    for _ in range(n):
        a = 0
        for st, lst in by_site.items():
            es = [e for _, e in lst]; random.shuffle(es)
            a += sum(1 for (h, _), e in zip(lst, es) if h and e == emb)
        if a >= a0: cnt += 1
    return cnt / n
P('sign/class | emblem | sign&emblem | sign other | emblem other | neither | Fisher p (one-sided, enrichment) | site-permutation P | why tested')
home_ps = []
for sig, emb, why in PRE:
    a, b, c, d = home_table(sig, emb)
    p = hyper_p(a, b, c, d); pp = site_perm(sig, emb)
    home_ps.append(p)
    P(f'{sig} | {emb} | {a} | {b} | {c} | {d} | {p:.3f} | {pp:.3f} | {why}')
hq = bh(home_ps)
P('BH q over the %d pre-registered home pairs: ' % len(PRE) + ', '.join(f'{s}x{e}={q:.2f}' for (s, e, _), q in zip(PRE, hq) if q < 0.2) + (' (none < 0.2)' if not any(q < 0.2 for q in hq) else ''))
P()
# Depletion direction too (the person sign may AVOID the bull at home)
P('Depletion check (two-sided view) for the person class and the twins at home, by emblem:')
for sig in ('PERSON', 'TWINS', 90, 91):
    row = []
    for emb in ('unicorn', 'bull', 'zebu', 'elephant', 'rhino', 'tiger', 'buffalo', 'goat', 'scene/anthropomorph', 'composite', 'fish', 'bird'):
        a, b, c, d = home_table(sig, emb)
        exp = (a + b) * (a + c) / len(home_seals) if len(home_seals) else 0
        row.append(f'{emb} {a}/{a+c} (exp {exp:.1f})')
    P(f'  {sig}: ' + '; '.join(row))
P()
P('Home scene/anthropomorph-emblem seals (the one pre-registered home pair below p = 0.05), text by text:')
for st, s, e, cult in home_seals:
    if e == 'scene/anthropomorph':
        ci, ty, sym = META[(st, s)]
        P(f"  {ci} {st} {ty} {sym}: {'-'.join(map(str,s))}" + ('  <- PERSON' if set(s) & PERSON else ''))
P('Home seals with a numeral directly next to a person sign (S98b unit), with emblem:')
for st, s, e, cult in home_seals:
    if num_adj(s):
        ci, ty, sym = META[(st, s)]
        P(f"  {ci} {st} {ty} {sym}: {'-'.join(map(str,s))}  pairs={num_adj(s)}")
# merge-level robustness for the person x scene pair
P('Merge-level robustness, PERSON x scene/anthropomorph on home seals (seq / seq_raw / seq_strong):')
for key in ('seq', 'seq_raw', 'seq_strong'):
    seen2 = set(); a = n_sc = n_p = N = 0
    for r in M:
        if r['site'] in ABROAD or not r['type'].startswith('SEAL'): continue
        if r['site'] == 'Unknown' and r['type'] in ('SEAL:C', 'SEAL:R', 'SEAL:CY'): continue
        if (r['site'], tuple(r['seq_raw'])) in IN_TABLE: continue
        s = tuple(r[key]); e = emblem_class(r['symbol'])
        if not s or e is None or (r['site'], s) in seen2: continue
        seen2.add((r['site'], s)); N += 1
        p_ = bool(set(s) & PERSON); sc_ = e == 'scene/anthropomorph'
        a += p_ and sc_; n_sc += sc_; n_p += p_
    P(f'  {key}: person&scene {a} | scene seals {n_sc} | person seals {n_p} | N {N} | Fisher p = {hyper_p(a, n_p-a, n_sc-a, N-n_p-n_sc+a):.3f}')
P()
# Exploratory: all signs (>=15 home seal texts) x all emblem classes (>=20 seals), Fisher one-sided, BH
P('### Exploratory home scan: every sign with >=15 seal texts x every emblem class with >=20 seals (one-sided enrichment, BH-FDR)')
sc = collections.Counter(x for _, s, _, _ in home_seals for x in set(s))
signs = [x for x, n in sc.items() if n >= 15]; embs = [e for e, n in ec.items() if n >= 20]
T2 = []
for x in signs:
    for e in embs:
        a, b, c, d = home_table(x, e)
        if a >= 3: T2.append((x, e, a, b, c, d, hyper_p(a, b, c, d)))
q2 = bh([t[6] for t in T2])
hits = sorted([(t, q) for t, q in zip(T2, q2) if q < 0.05], key=lambda z: z[0][6])
P(f'{len(T2)} pairs tested; {len(hits)} with BH q < 0.05:')
for t, q in hits[:40]:
    mnum = BR.get(str(t[0]))
    P(f'  W{t[0]} (M{mnum}) x {t[1]}: {t[2]} of {t[2]+t[3]} texts with the sign carry this emblem; emblem base {t[2]+t[4]}/{len(home_seals)}; p = {t[6]:.2e}, q = {q:.3f}')
# site-permutation on the top hits (emblem labels shuffled within site)
P('Site-permutation check on the hits (emblem labels shuffled within site, 2000x):')
for t, q in hits[:15]:
    P(f'  W{t[0]} x {t[1]}: P_perm = {site_perm(t[0], t[1]):.4f}')
P()
# the Gaur-emblem home seals: which signs are enriched there (the abroad animal)
P('### Home bull/gaur-emblem seals: do they share the abroad vocabulary (PERSON, TWINS, 1, 55, 415-803, 740-740)?')
gaur = [s for _, s, e, _ in home_seals if e == 'bull']; others = [s for _, s, e, _ in home_seals if e != 'bull']
def share(lst, f): return sum(1 for s in lst if f(s)), len(lst)
for name, f in [('PERSON', lambda s: 'PERSON' in classes(s)), ('TWINS', lambda s: 'TWINS' in classes(s)), ('W1', lambda s: 1 in s), ('W55', lambda s: 55 in s),
                ('415-803 bigram', lambda s: any(x == 415 and y == 803 for x, y in zip(s, s[1:]))), ('740-740', lambda s: any(x == 740 and y == 740 for x, y in zip(s, s[1:]))),
                ('numeral+person adjacency', lambda s: bool(num_adj(s)))]:
    a, n1 = share(gaur, f); c, n2 = share(others, f)
    P(f'  {name}: gaur seals {a}/{n1} vs other emblems {c}/{n2}; Fisher p(enriched on gaur) = {hyper_p(a, n1-a, c, n2-c):.3f}')
P()

# ---------------------------------------------------------------------------------------------
# 7. Mahadevan mapping of the key signs (CLAUDE.md rule) and frame slots
# ---------------------------------------------------------------------------------------------
P('## 6. Bridge check (W -> M) for the signs in the abroad table and their GRAMMAR.md slot')
SLOT = {817: 'opener', 861: 'opener', 820: 'opener', 2: 'marker (M99)', 60: 'marker (M123)', 31: 'middle-initial marker (M86)',
        740: 'closer (M342)', 400: 'suffix (M176)', 90: 'suffix/person (M1)', 1: 'marker stroke (S234)', 55: 'fixed 12 (M121)'}
ks = sorted(set(x for o in ab for x in o['seq'] if isinstance(x, int)))
P(', '.join(f'W{x}->M{BR.get(str(x))}{" ["+SLOT[x]+"]" if x in SLOT else ""}' for x in ks))
P()

# ---------------------------------------------------------------------------------------------
# 8. Verdict
# ---------------------------------------------------------------------------------------------
P('## 7. Verdict')
P('''(a) NULL. 29 abroad round/cylinder objects carry Indus signs; 23 have a recorded motif; 17 of the 23 show the plain
short-horned bull. Every non-bull ANIMAL motif is attested on exactly one inscribed seal (antelope L6, rhinoceros L8,
peacock L27, scorpion+footprint L20, mating pair L24, empty field L9); only the manger (4) and an unspecified scene (2)
recur. So no sign can co-occur twice with any non-bull animal: the maximum evidence for a 'scorpion word' or
'antelope word' in this material is one object, an upper bound of < 3/1 by the sampling rule. Of 23 testable sign x motif
pairs the best is NUMERAL x manger (4 of 11 numeral texts vs 0 of 12 without; p = 0.037), BH q = 0.51, and a smallest
p this small arises in 30% of label shuffles. Nothing survives.
(c) The twins sign W91 sits beside an antelope, a rhinoceros, an empty field, three bulls and a mating scene; the man
sign W90 beside bulls, a peacock, a scorpion+footprint and the Failaka scene. Person signs show no preference for
non-bull motifs (8 of 15 person texts vs 6 of 8 others, p = 0.93). The twins is therefore NOT a label for the animal
drawn; it is independent of the picture, which fits S275/S314 (a marker of the overseas seal-using community).
(d) The numeral+person unit (S98b) occurs in 8 of 20 abroad person texts (12, 1, 1, 1, 1, 4, 12, 12/1). Its motifs: bull 4,
bull+manger 1, empty field 1, unrecorded 3. The four person texts beside a non-bull animal (antelope, rhino, peacock,
scorpion/footprint) all lack the numeral (0 of 4 vs 5 of 11; one-sided p = 0.15, n.s.). Reported as an upper bound only.
(b) HOME. 18 pre-registered sign x emblem pairs on 1,536 home seals with an emblem (site+text dedup; the Gulf-type
pieces of unknown provenance removed). Two reach p < 0.05, none BH q < 0.1 (q = 0.19):
  - PERSON x scene/anthropomorph emblem: 3 of 12 scene seals carry a person sign (M-1918 90-390, M-304, M-1186, all
    Mohenjo-daro) vs 81 of 1,524 other seals (5.3%); p = 0.024, site-permutation P = 0.022, identical on seq, seq_raw
    and seq_strong. Abroad the matching cases are L13 (Failaka scene, 90-861-317) and L20 (Ur, man sign beside a
    footprint). Rests on Mohenjo-daro alone (3 of 3).
  - W55 '12' x gaur emblem: 4 of 106 gaur seals vs 14 of 1,430 others (p = 0.031, perm P = 0.029). Abroad, 12 stands on
    three bull seals (L7, L10, Susa cylinder X1). The two home seals with a numeral next to a person sign both carry the
    gaur (M-234 Mohenjo-daro; the Dholavira round seal 255-368-1-91-803; 2 of 106 vs 0 of 1,430, p = 0.005, but one is
    itself a Gulf-type round seal, so this is S46 again: the format, not the picture).
  Exploratory scan of 175 sign x emblem pairs (signs >= 15 seal texts, emblems >= 20 seals): 0 at BH q < 0.05. The Indus
  emblem does not select the signs on the text band, at home any more than abroad.
GRADE-C CANDIDATE (one, weak): W90 (M1, standing man) = 'person / human figure', pictorially motivated. Abroad and
at home it is the one sign that appears beside human-figure scenes above chance (3 of 12 home scene seals; Failaka
scene; Ur footprint seal), while the twins W91 does not (0 of 12 at home). It adds a pictorial argument to the B gloss
'man, person' already in WORKING-DICTIONARY.md rather than a new reading.
  Would support: in new or held-out scene/anthropomorph-emblem seals, W90 present in > 15% (base 5.3%); W90 beside
  further human-figure motifs abroad (drinking scene, footprint) on inscribed Gulf seals.
  Would kill: the next 20 scene-emblem seals carrying W90 at the base rate (<= 1 of 20), or W90 turning up as often beside
  non-human motifs on new Gulf-type seals.
NULL RESULTS, plainly: no 'scorpion', 'antelope', 'peacock', 'footprint' or 'bull' sign can be isolated; the Gulf-type
corpus has one inscribed seal per non-bull motif, and at home no sign tracks its emblem (175 pairs, 0 hits). The
hypothesis that Indus signs are picture-labels for the motif beside them is not supported by anything in this material,
and the motif-independence of the twins is positive evidence against it for that sign. Sample caveats: 23 objects,
76% of the home seals come from Mohenjo-daro and Harappa, and the home scene hit is Mohenjo-daro only.

## 8. Proposed STRATEGIES.md row (insert before "## Summary"; S360 is taken, use the next free number)
| S361 | **Counter-intuitive anchor: do Indus signs track the PICTORIAL motif beside them on Gulf-type / Near-East seals (scorpion, footprint, antelope, peacock, manger, mating scene), and the matching emblem at home?** Object table of 29 abroad round/cylinder seals + 10 square/tags abroad + 5 Gulf-type at home, each row cited (gulf_seals.csv, Laursen 2010 pp. 111-114 and n. 11, Al-Sindi 1999, Gadd 1932 via ANCHORS sec. 9, Parpola 1994 facts). (a) sign x motif Fisher + BH + label-shuffle; (b) 18 pre-registered sign x emblem pairs on 1,536 home seals, site-permutation, all three merge levels; exploratory 175 pairs; (c) twins/man vs scenes; (d) numeral+person unit vs motif. `tools/strat_gulfmotif.py`, `data/derived/strat_gulfmotif.txt` | **(a) 23 objects with a motif, 17 plain bull; every non-bull animal occurs on ONE inscribed seal, so no sign can recur beside it (upper bound < 3/1). 23 pairs: best NUMERAL x manger 4/11 vs 0/12, p = 0.037, q = 0.51, shuffle P = 0.30. (c) twins W91 sits beside antelope, rhino, empty field, 3 bulls, mating pair: motif-independent (p = 0.93). (d) numeral+person in 8/20 person texts, never beside a non-bull animal (0/4 vs 5/11, p = 0.15). (b) PERSON x scene/anthropomorph emblem 3/12 vs 81/1,524 (p = 0.024, perm 0.022, q = 0.19; Mohenjo-daro only); W55 x gaur 4/106 vs 14/1,430 (p = 0.031); exploratory 0/175 at q < 0.05.** | **Not supported.** Signs do not label the picture beside them, abroad or at home; the twins is positively independent of the animal. Only W90 keeps weak pictorial company with human-figure scenes (home 3/12 + Failaka scene + Ur footprint seal), which adds a C-grade pictorial argument to its existing B gloss 'man, person'. Would support: W90 > 15% on new scene-emblem seals. Would kill: base rate (<= 1/20) on the next 20. The gaur-emblem/12/numbered-person package recurs at home only on Gulf-format seals (S46: format, not picture). |''')
open(f'{ROOT}/data/derived/strat_gulfmotif.txt', 'w').write('\n'.join(out) + '\n')
