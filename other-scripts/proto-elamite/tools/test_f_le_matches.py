#!/usr/bin/env python3
"""Test f: do the PE signs that Desset et al. (2026, Pls. III-V) draw as graphic
ancestors of Linear Elamite syllables behave like phonetic signs inside PE?

Prediction if they were already used phonetically in PE (e.g. to spell names):
 1. enriched inside long entry strings (len >= 3), depleted as whole one-sign entries;
 2. enriched in non-final (non-category) positions of long strings;
 3. they sit next to each other (adjacent pairs both in the set) more than chance.
Control: 2000 random sign sets of the same size, matched on token frequency
(each matched sign replaced by a random sign from the same frequency decile).
Done at base-sign level (variants merged) and with numerals-as-signs ignored."""
import collections, json, os, random, re
from common import load, entries, base, DATA

random.seed(11)
T = load()
E = [e['signs'] for e in entries(T) if 'x' not in e['signs']]
M = json.load(open(os.path.join(DATA, 'le_pe_graphic_matches.json')))['matches']
le = set()
for k, v in M.items():
    if k.startswith('US'):
        continue
    for s in v:
        if '+' in s:
            continue
        le.add(re.match(r'M\d+', s).group(0))
# CDLI pads to 3 digits
le = {('M%03d' % int(s[1:])) for s in le}
freq = collections.Counter(x for s in E for x in s)
le = {s for s in le if freq[s] > 0}
import sys
if '--drop-frame' in sys.argv:
    # drop the frame/category signs found in tests a,b,d (strong slot signs)
    le -= {'M288', 'M387', 'M297', 'M263', 'M371', 'M218', 'M346', 'M009', 'M032', 'M157'}
print('LE-matched base signs present in corpus:', len(le), sorted(le))


def stats(S):
    tok = single = longin = mid = 0
    adj = adjboth = 0
    for s in E:
        for i, x in enumerate(s):
            if x not in S:
                continue
            tok += 1
            if len(s) == 1:
                single += 1
            if len(s) >= 3:
                longin += 1
                if i < len(s) - 1:
                    mid += 1
        if len(s) >= 2:
            for a, b in zip(s, s[1:]):
                if a in S:
                    adj += 1
                    adjboth += b in S
    return {'single': single / tok, 'long': longin / tok, 'nonfinal_long': mid / tok,
            'adj_both': adjboth / max(adj, 1), 'tokens': tok}


obs = stats(le)
# frequency deciles
signs = sorted(freq, key=lambda x: freq[x])
dec = {}
for i, x in enumerate(signs):
    dec[x] = int(10 * i / len(signs))
bydec = collections.defaultdict(list)
for x in signs:
    bydec[dec[x]].append(x)
ctrl = collections.defaultdict(list)
for _ in range(2000):
    R = set()
    for x in le:
        pool = [y for y in bydec[dec[x]] if y not in R]
        R.add(random.choice(pool))
    st = stats(R)
    for k, v in st.items():
        ctrl[k].append(v)
out = {'n_signs': len(le), 'obs': obs, 'ctrl': {}}
for k in ('single', 'long', 'nonfinal_long', 'adj_both'):
    c = ctrl[k]
    m = sum(c) / len(c)
    p_hi = sum(1 for v in c if v >= obs[k]) / len(c)
    p_lo = sum(1 for v in c if v <= obs[k]) / len(c)
    out['ctrl'][k] = {'mean': m, 'p_ge': p_hi, 'p_le': p_lo}
    print('%-14s obs %.3f  ctrl mean %.3f  P(ctrl>=obs) %.3f  P(ctrl<=obs) %.3f' % (k, obs[k], m, p_hi, p_lo))
print('LE-set token count', obs['tokens'])
json.dump(out, open(os.path.join(DATA, 'res_f_le%s.json' % ('_dropframe' if '--drop-frame' in sys.argv else '')), 'w'), indent=1)
