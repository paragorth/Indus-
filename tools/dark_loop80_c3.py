"""Loop 80 cycle 3: do Meluhha entries carry counts in the 3-6 range (the Indus quantity-seal range, S335-S346,
e.g. '2-4-390' = 'of 4 tree-units'), and is wood counted in 4s?
Meluhha side: discrete counts (pieces, animals, persons, garments) and ab-ba/mes wood items from the hand-annotated
third-millennium rows of data/derived/dark/loop80_meluhha.csv, plus an automatic pass (leading N(disz) count of the
Meluhha line or of the line directly above it).
Baseline: every administrative line with a leading discrete count N(disz) / N(asz@c) / N(u)+N(disz) in Ur III, Lagash II,
Old Akkadian and Early OB texts from the same find-spots (Ur, Umma, Girsu, Adab, Irisagrig, Isin), all goods and
{gesz} (wood) lines only; and ab-ba wood lines that are NOT of Meluhha.
Usage: python3 tools/dark_loop80_c3.py ATF CATALOG"""
import re, csv, sys, json, math, random
from collections import Counter
csv.field_size_limit(10**9)
atf, cat = sys.argv[1:3]
SITES = ('Ur (', 'Umma', 'Girsu', 'Adab', 'Irisagrig', 'Isin')
PERS = ('Ur III', 'Lagash II', 'Old Akkadian', 'Early Old Babylonian')
meta = {}
for r in csv.DictReader(open(cat, errors='replace')):
    if r['id_text'].isdigit() and r['period'].startswith(PERS) and r['provenience'].startswith(SITES) and 'Administrative' in r['genre']:
        meta['P%06d' % int(r['id_text'])] = r['provenience'].split(' (')[0]
CNT = re.compile(r"^\s*\d+[a-z]?'?\.\s+((?:\d+\((?:u|disz|asz@c|u@c|disz@t)\)\s*)+)(.*)")
def val(s):
    v = 0
    for n, u in re.findall(r'(\d+)\((u|disz|asz@c|u@c|disz@t)\)', s):
        v += int(n) * (10 if u.startswith('u') else 1)
    return v
MEL = re.compile(r'm[ei]-lu(?:h|-uh)(?!-luh)|m[ei]-lu-ha')
base_all = Counter(); base_gesz = Counter(); base_abba = Counter(); mel_auto = Counter(); mel_lines = []
cur = None; prev = ''
for l in open(atf, errors='replace'):
    if l.startswith('&P'):
        cur = l[1:8] if l[1:8] in meta else None; prev = ''; continue
    if not cur or not re.match(r"^\s*\d+[a-z]?'?\.", l): continue
    c = re.sub(r'[\[\]#?!<>]', '', l)
    if '(asz)' in c or '(barig)' in c or '(ban2)' in c or 'sila3' in c or 'gin2' in c or 'ma-na' in c or 'gur' in c:
        prev = c; continue  # capacity / weight measures are not discrete counts
    m = CNT.match(c)
    if MEL.search(c):
        mm = m or CNT.match(prev)
        if mm: mel_auto[val(mm.group(1))] += 1; mel_lines.append((cur, c.strip()))
    elif m:
        v = val(m.group(1)); rest = m.group(2)
        if 1 <= v <= 60:
            base_all[v] += 1
            if '{gesz}' in rest or rest.startswith('gesz'): base_gesz[v] += 1
            if 'ab-ba' in rest and '{gesz}' in rest: base_abba[v] += 1
    prev = c
# hand-annotated discrete Meluhha counts (unit = piece / animal / person / garment), 3rd millennium
rows = list(csv.DictReader(open('data/derived/dark/loop80_meluhha.csv')))
hand = []; wood = []
for r in rows:
    if not any(k in r['period'] for k in ('Akkadian', 'Lagash', 'Ur III', 'Early Old')): continue
    if r['unit'] in ('piece', 'udu', 'animal', 'person', 'workers', 'tug2'):
        for q in re.split(r';\s*', r['quantity']):
            if q.strip().isdigit(): hand.append(int(q))
        if 'wood' in r['goods']:
            for q in re.split(r';\s*', r['quantity']):
                if q.strip().isdigit(): wood.append(int(q))
def share36(c):
    n = sum(c.values()); k = sum(v for x, v in c.items() if 3 <= x <= 6)
    return k, n, (k / n if n else float('nan'))
def binom_p(k, n, p):  # P(X <= k) and P(X >= k)
    lo = sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(0, k + 1))
    hi = sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k, n + 1))
    return lo, hi
res = {}
for name, c in (('baseline_all', base_all), ('baseline_gesz', base_gesz), ('baseline_abba_nonMeluhha', base_abba)):
    k, n, s = share36(c); res[name] = {'k36': k, 'n': n, 'share36': round(s, 3), 'share1': round(c[1] / n, 3), 'share4': round(c[4] / n, 4)}
hc = Counter(hand); wc = Counter(wood)
for name, c, ref in (('meluhha_hand_discrete', hc, 'baseline_all'), ('meluhha_hand_wood', wc, 'baseline_abba_nonMeluhha'), ('meluhha_auto', mel_auto, 'baseline_all')):
    k, n, s = share36(c); p = res[ref]['share36']; lo, hi = binom_p(k, n, p)
    res[name] = {'counts': dict(sorted(c.items())), 'k36': k, 'n': n, 'share36': round(s, 3), 'vs': ref, 'P_low': round(lo, 4), 'P_high': round(hi, 4),
                 'n_four': c[4], 'share1': round(c[1] / n, 3) if n else None}
res['meluhha_auto_lines'] = mel_lines
json.dump(res, open('data/derived/dark/loop80_c3.json', 'w'), indent=1, ensure_ascii=False)
for k, v in res.items():
    if k != 'meluhha_auto_lines': print(k, v)
for x in mel_lines: print(' ', x)
