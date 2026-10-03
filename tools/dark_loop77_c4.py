"""Loop 77 cycle 4: (i) POWER: would the Indus group structures have shown a PE-like tablet signature? For each Indus
analogue (DES element set, seq_raw), its group-size profile is refilled with real PE middles: each Indus group of size
k becomes k entries drawn without replacement from one random PE tablet with >= k middles (>= 1 element); the same
within() statistic and stratified permutation null (stratum = middle length) is run; detection = share P < 0.05 over
REPS refills. Negative control: k entries drawn from k different tablets (should detect ~5%).
(ii) ELEMENTS: every Indus non-identical DES pair that shares an element inside a group, for the analogues that came
out above chance (multi-sided seals + sealings, Mohenjo-daro multi-sided objects, Mohenjo-daro same-spot seals):
shared element, its token count in the whole corpus, its position in each text, whether one face text contains the
other (abbreviation), and the Mahadevan number (bridge_extended; proposed entries marked *).
Usage: python3 tools/dark_loop77_c4.py [reps] [nperm]
"""
import sys
REPS = int(sys.argv[1]) if len(sys.argv) > 1 else 100
NPERM = int(sys.argv[2]) if len(sys.argv) > 2 else 200
sys.path.insert(0, '/home/user/Indus-/tools')
from dark_loop77_common import *
R = {}
P(f'== Loop 77 cycle 4: power and elements; reps {REPS}, nperm {NPERM}')
BR = json.load(open(ROOT + 'data/derived/bridge_extended.json')); PROP = json.load(open(ROOT + 'data/derived/dark/bridge_proposals.json'))
def Mn(w):
    if str(w) in BR: return f'W{w}/M{"+".join(map(str, BR[str(w)]))}'
    p = [x['M'] for x in PROP['proposals'] if x['W'] == w]
    return f'W{w}/M{p[0]}*' if p else f'W{w}'

objs, QUAL, FR = load_objects('seq_raw')
M = collapse([o for o in objs.values() if sum(1 for f in o['faces'] if f['seq']) >= 2])
def ms_items(sel):
    return [(o['oid'], f['k'], f['DES'], (o['site'], o['type'], min(len(f['DES']), 5))) for o in M if sel(o) for f in o['faces'] if f['complete'] and f['seq'] and f['DES']]
def textface(o):
    fs = [f for f in o['faces'] if f['complete'] and f['seq']]
    return max(fs, key=lambda f: (len(f['seq']), -f['k'])) if fs else None
def spot_items(site, ot):
    out = []
    for o in objs.values():
        if o['site'] != site or o['ot'] != ot or not o['room']: continue
        f = textface(o)
        if f and f['DES']: out.append(((o['area'], o['block'], o['room']), int(o['oid']), f['DES'], (site, o['type'], min(len(f['DES']), 5))))
    return out
AN = {'multi-sided, all': ms_items(lambda o: True),
      'multi-sided, Harappa tablets': ms_items(lambda o: o['site'] == 'Harappa' and o['ot'] == 'tablet'),
      'multi-sided, seals + sealings': ms_items(lambda o: o['ot'] in ('seal', 'sealing')),
      'Harappa tablets same spot': spot_items('Harappa', 'tablet'),
      'Mohenjo-daro seals same spot': spot_items('Mohenjo-daro', 'seal')}

# ---------------- (i) power ----------------
import dark_loop73_common as L73
E = L73.pe_entries()
tab = collections.defaultdict(list)
for e in E:
    if len(e['mid']) >= 1: tab[e['tablet']].append(e['mid'])
P('\n##### (i) power: Indus group structures refilled with PE middles (same tablet = PE signature; different tablets = null)')
for name, items in AN.items():
    g = collections.Counter(d for d, _, _, _ in items); sizes = [k for k in g.values() if k >= 2]
    if not sizes: continue
    det = {'same': 0, 'diff': 0}; ratios = {'same': [], 'diff': []}; rpairs = []
    r = random.Random(7704)
    for rep in range(REPS):
        for mode in ('same', 'diff'):
            its = []
            for gi, k in enumerate(sizes):
                if mode == 'same':
                    t = r.choice([t for t, L in tab.items() if len(L) >= k]); mids = r.sample(tab[t], k)
                else:
                    ts = r.sample(list(tab), k); mids = [r.choice(tab[t]) for t in ts]
                for j, m in enumerate(mids): its.append((gi, 2 * j, m, min(len(m), 5)))
            res = within('power', its, NPERM, seed=rep, markov=False, quiet=True)
            if res and res['share'].get('perm'):
                ratios[mode].append(res['share']['perm']['ratio'])
                if res['share']['perm']['p'] < 0.05: det[mode] += 1
                if mode == 'same': rpairs.append(res['obs']['rare'] * res['obs']['pairs'])
    med = lambda v: sorted(x for x in v if not math.isnan(x))[len(v) // 2] if v else float('nan')
    P(f'  [{name}] groups {len(sizes)} (sizes {collections.Counter(sizes).most_common(4)}): PE same-tablet detection {det["same"]}/{REPS} '
      f'(median share ratio x{med(ratios["same"]):.2f}, median rare-sharing pairs {med(rpairs):.1f}); different-tablet control {det["diff"]}/{REPS} (x{med(ratios["diff"]):.2f})')
    R[f'power_{name}'] = dict(groups=len(sizes), detect_same=det['same'] / REPS, detect_diff=det['diff'] / REPS,
                              ratio_same=med(ratios['same']), ratio_diff=med(ratios['diff']), rare_pairs_same=med(rpairs))

# ---------------- (ii) elements ----------------
P('\n##### (ii) the Indus pairs that share a DES element inside a group (seq_raw)')
tokc = collections.Counter(x for o in objs.values() for f in o['faces'] for x in f['seq'])
def show(name, items, faces_by):
    g = collections.defaultdict(list)
    for d, p, m, s in items: g[d].append((p, m))
    rows = []
    for d, L in g.items():
        for i in range(len(L)):
            for j in range(i + 1, len(L)):
                if L[i][1] == L[j][1]: continue
                S = set(L[i][1]) & set(L[j][1])
                if not S: continue
                fa = faces_by(d, L[i][0]); fb = faces_by(d, L[j][0])
                def pos(x, f): k = f['seq'].index(x); return f'{k + 1}/{len(f["seq"])}'
                sub = lambda a, b: any(tuple(b[k:k + len(a)]) == tuple(a) for k in range(len(b) - len(a) + 1))
                ab = 'abbrev' if sub(fa['seq'], fb['seq']) or sub(fb['seq'], fa['seq']) else ''
                for x in S:
                    rows.append((d, Mn(x), tokc[x], pos(x, fa), pos(x, fb), '-'.join(map(str, fa['seq'])), '-'.join(map(str, fb['seq'])), ab))
    P(f'  [{name}] {len(rows)} shared-element rows')
    for r_ in rows: P('    ', r_)
    return rows
fb_obj = lambda d, k: next(f for f in objs[d]['faces'] if f['k'] == k)
fb_spot = lambda d, oid: textface(objs[str(oid)])
R['el_ms_seal'] = show('multi-sided seals + sealings', AN['multi-sided, seals + sealings'], fb_obj)
R['el_ms_md'] = show('multi-sided Mohenjo-daro', ms_items(lambda o: o['site'] == 'Mohenjo-daro'), fb_obj)
R['el_md_spot'] = show('Mohenjo-daro seals same spot', AN['Mohenjo-daro seals same spot'], fb_spot)
# positional fixity / rarity of the shared elements vs all DES elements
allpos = collections.defaultdict(list)
for o in objs.values():
    for f in o['faces']:
        for k, x in enumerate(f['seq']):
            if x in f.get('DES', ()): allpos[x].append(k / max(1, len(f['seq']) - 1))
P('\n  shared elements, corpus profile (tokens; mean relative position 0 = first, 1 = last; sd):')
for x in sorted({r_[1] for k in ('el_ms_seal', 'el_ms_md', 'el_md_spot') for r_ in R[k]}):
    w = int(x.split('/')[0][1:]); v = allpos[w]
    if v:
        mu = sum(v) / len(v); sd = (sum((a - mu) ** 2 for a in v) / len(v)) ** 0.5
        P(f'    {x}: tokens {tokc[w]}, DES tokens {len(v)}, position {mu:.2f} +/- {sd:.2f}')
save('loop77_c4', R)
