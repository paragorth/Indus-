"""Loop 54 cycle 4: the outside facts behind each entry, recomputed (not cited): for every dictionary sign, share of its texts
by object class (seal / sealing TAG / tablet TAB / pot / other), the abroad share (sites outside the Indus region), the number of
sites, and a Fisher test of the strongest object-class enrichment against all other texts. A chain fitted within site x type
reproduces these by construction, so they are the independent support the re-grading may lean on.
Usage: python3 tools/dark_loop54_c4.py [seq_all]; output loop54_c4_outside.txt"""
import sys, collections, json
from scipy.stats import fisher_exact
from dark_loop54_common import *

level = sys.argv[1] if len(sys.argv) > 1 else 'seq_all'
C = json.load(open(ROOT + 'data/derived/merged-corpus-canonical.json'))
ABROAD = {'Ur', 'Kish', 'Susa', 'Bahrain', 'Failaka', 'Tell Asmar', 'Nippur', 'Umma', 'Tepe Yahya', 'Shahr-i Sokhta', 'Gonur', 'Altyn-depe', 'Lagash', 'Qala\'at al-Bahrain', 'Saar', 'Ras al-Jinz', 'Maysar', 'Tell Abraq', 'Hili', 'Mesopotamia', 'Iran', 'Oman', 'Nineveh', 'Tello', 'Girsu', 'Karzakan', 'Salut', 'Hajar', 'Janabiyah', 'Dilmun', 'Luristan', 'Shortughai', 'Altyn Depe', 'Gonur Depe', "Ra's al-Junayz", 'Kalba', 'Tell Umma', 'Miri Qalat', 'West Asian finds'}
def cls(t):
    o = otype(t)
    return 'SEAL' if o == 'SEAL' else 'TAG' if o == 'TAG' else 'TAB' if o == 'TAB' else 'POT' if o == 'POT' else 'BNGL' if o == 'BNGL' else 'OTHER'
T = [(r['site'], cls(r['type']), tuple(r[level])) for r in C if r[level] and r['site'] != 'Unknown']
# die regime
seen = set(); D = []
for r in C:
    s = tuple(r[level])
    if not s or r['site'] == 'Unknown': continue
    k = (r['site'], otype(r['type']), s) if (otype(r['type']) in ('TAB', 'TAG') or r['type'].startswith('POT:T:s')) else (r['cisi'] or id(r), s)
    if k in seen: continue
    seen.add(k); D.append((r['site'], cls(r['type']), s))
sites_all = collections.Counter(s for s, _, _ in D)
abroad = lambda site: any(a.lower() in site.lower() for a in ABROAD) or site in ('West Asian finds',)
base = collections.Counter(c for _, c, _ in D); N = len(D)
SIGNS = {'W1-5,16-18 short numerals': {1, 3, 4, 5, 16, 17, 18}, 'W31-34 tall': {31, 32, 33, 34}, 'W55': {55}, 'W56': {56}, 'W2': {2}, 'W60': {60},
         'W817/861': {817, 861}, 'W820': {820}, 'W740': {740}, 'W400': {400}, 'W90': {90}, 'W91': {91}, 'W595': {595}, 'W520': {520}, 'W151': {151},
         'W156': {156}, 'W527': {527}, 'W226': {226}, 'W617': {617}, 'W154/158': {154, 158}, 'W700': {700}, 'W176': {176}, 'W100': {100}, 'W760': {760},
         'W923': {923}, 'W390/405': {390, 405, 406}, 'W407': {407}, 'W220': {220}, 'W900': {900}, 'W575': {575}, 'W585': {585}, 'W632': {632},
         'W235/240/233/231 modified fish': {235, 240, 233, 231}, 'W142': {142}, 'W125': {125}, 'W140': {140}, 'W415': {415}, 'W803/806': {803, 806},
         'W550': {550}, 'W798': {798}, 'W705/706': {705, 706}, 'W741': {741}, 'W692': {692}, 'W920': {920}, 'W840': {840}, 'W460': {460}, 'W435': {435},
         'W255': {255}, 'W690': {690}, 'W590': {590}, 'W845': {845}, 'W368': {368}, 'W892': {892}, 'W61': {61}, 'W34 alone': {34}, 'W999': {999},
         'W480': {480}, 'W697': {697}, 'W790': {790}, 'W64': {64}, 'W930': {930}, 'W752': {752}, 'W48': {48}, 'W503': {503}, 'W413': {413}}
out = [f'# Loop 54 cycle 4: outside facts recomputed on the canonical corpus ({level}, die regime, {N} texts, Unknown site dropped). base shares: ' + ', '.join(f'{k} {v / N:.2f}' for k, v in base.most_common()),
       '| sign | texts | sites | abroad share (base ' + f'{sum(1 for s, _, _ in D if abroad(s)) / N:.3f}' + ') | SEAL | TAG | TAB | POT | strongest class enrichment (odds, Fisher P) | MD+H share |', '|---|---|---|---|---|---|---|---|---|---|']
for nm, X in SIGNS.items():
    w = [(s, c) for s, c, q in D if set(q) & X]
    if not w: out.append(f'| {nm} | 0 | | | | | | | | |'); continue
    n = len(w); cc = collections.Counter(c for _, c in w); ss = collections.Counter(s for s, _ in w)
    ab = sum(1 for s, _ in w if abroad(s)) / n
    best = None
    for c in ('SEAL', 'TAG', 'TAB', 'POT', 'BNGL', 'OTHER'):
        a = cc[c]; b = n - a; c2 = base[c] - a; d = N - n - c2
        odds, p = fisher_exact([[a, b], [c2, d]])
        if best is None or p < best[2]: best = (c, odds, p)
    out.append(f"| {nm} | {n} | {len(ss)} | {ab:.3f} | {cc['SEAL'] / n:.2f} | {cc['TAG'] / n:.2f} | {cc['TAB'] / n:.2f} | {cc['POT'] / n:.2f} | {best[0]} {best[1]:.2f}x P={best[2]:.1e} | {(ss['Mohenjo-daro'] + ss['Harappa']) / n:.2f} |")
open(DARK + 'loop54_c4_outside.txt', 'w').write('\n'.join(out) + '\n')
print('\n'.join(out))
