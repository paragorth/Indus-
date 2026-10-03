"""Loop 77 cycle 3: analogue (c): Lothal sealed deposit.
 (c1) dies impressed on ONE sealing = entries on one document (closest analogue of a PE tablet listing several persons):
      Wells faces (loop71 die assignment, 0 = lost sign dropped from element sets) and IM77 sides (M bridged to W).
      Null: dies permuted across sealings within type x designation length (curveball-free: plain stratified shuffle).
 (c2) distinct dies of one deposit as one group: Lothal warehouse (FT 2005 context W / W1, loop71 catalogue) vs Lothal
      non-warehouse vs the distinct sealing dies of every other site (one group per site); null = dies permuted across
      deposits within designation length (site is the grouping, so it is not a stratum).
 Element sets FULL / MID / DES (dark_loop77_common); three merge levels.
Usage: python3 tools/dark_loop77_c3.py [nperm]
"""
import sys
NPERM = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
sys.path.insert(0, '/home/user/Indus-/tools')
from dark_loop77_common import *
import dark_loop71_common as L71
R = {}
P(f'== Loop 77 cycle 3 (Lothal); nperm {NPERM}')
CAT = {r['cisi']: r for r in csv.DictReader(open(DARK + 'loop71_lothal_sealings.csv'))}
WH = {c for c, r in CAT.items() if r['ft_context'] in ('W', 'W1')}

for LV in ['seq_raw', 'seq_strong', 'seq_all']:
    P(f'\n##### {LV}')
    objs, QUAL, FR = load_objects(LV)
    W = L71.wells_lothal(LV)
    imps = [(f['id'], f['toks']) for k, fs in W.items() for f in fs]
    lab = L71.assign_dies(imps)
    inc = collections.OrderedDict()
    for c, fs in W.items():
        ds = []
        for f in fs:
            if not [x for x in f['toks'] if x]: continue
            d = lab[f['id']]
            if d not in ds: ds.append(d)
        if ds: inc[c] = ds
    def toks(d): return [x for x in (d[1:] if d[0] == 'u' else d) if x]
    def es(d, ES): return element_sets(toks(d), QUAL, FR)[ES] if toks(d) else ()
    multi = {c: ds for c, ds in inc.items() if len(ds) >= 2}
    P(f'  Wells Lothal sealings with a die: {len(inc)}; with >= 2 distinct dies: {len(multi)}; warehouse (W/W1) sealings {sum(1 for c in inc if c in WH)}')
    for ES in ['FULL', 'MID', 'DES']:
        items = [(c, k, es(d, ES), ('sealing', min(len(es(d, ES)), 5))) for c, ds in inc.items() for k, d in enumerate(ds)]
        R[f'{LV}_c1_{ES}'] = within(f'{LV} Lothal dies on one sealing (Wells), {ES}', items, NPERM)
        if ES != 'FULL' and LV == 'seq_raw':
            P('     shared:', shared_elements(items).most_common(8))
    # (c2) deposits: distinct sealing dies per site; Lothal split W/W1 vs other
    dep = collections.defaultdict(set)
    for c, ds in inc.items():
        for d in ds:
            if d[0] == 'u' and len(toks(d)) < 2: continue
            dep['Lothal-warehouse' if c in WH else 'Lothal-other'].add(tuple(toks(d)))
    for o in objs.values():
        if o['ot'] != 'sealing' or o['site'] == 'Lothal': continue
        for f in o['faces']:
            if f['seq'] and len(f['seq']) >= 2: dep[o['site']].add(tuple(f['seq']))
    P('  deposits (distinct dies):', sorted(((k, len(v)) for k, v in dep.items()), key=lambda x: -x[1])[:10])
    for ES in ['FULL', 'MID', 'DES']:
        items = []; i = 0
        for g, S in dep.items():
            for t in sorted(S):
                e = element_sets(list(t), QUAL, FR)[ES]; items.append((g, i, e, min(len(e), 5))); i += 2  # spaced: no adjacency in a die set
        res = within(f'{LV} deposits (sites; Lothal split), {ES}', items, max(200, NPERM // 5), markov=True)
        R[f'{LV}_c2_{ES}'] = res
        # warehouse-only statistic: pairs inside Lothal-warehouse vs the same null
        wi = [x for x in items if x[0] == 'Lothal-warehouse']
        cnt = collections.Counter(x for _, _, m, _ in items for x in set(m))
        fcls = collections.defaultdict(lambda: 'rare'); fcls.update({x: 'rare' if c < 5 else 'mid' if c < 20 else 'freq' for x, c in cnt.items()})
        o = pair_stats([[(p, m) for _, p, m, _ in wi if m]], fcls)
        r = random.Random(773); pool = [m for _, _, m, _ in items if m]; nl = []
        byL = collections.defaultdict(list)
        for m in pool: byL[min(len(m), 5)].append(m)
        for _ in range(NPERM):
            draw = [r.choice(byL[min(len(m), 5)]) for _, _, m, _ in wi if m]
            nl.append(pair_stats([[(k, m) for k, m in enumerate(draw)]], fcls))
        line = f'    Lothal warehouse dies alone ({len([1 for x in wi if x[2]])}): pairs {o["pairs"]}; '
        rr = {}
        for k in ['share', 'rare', 'mid', 'freq']:
            v = [x[k] for x in nl]; mu = sum(v) / len(v); p = (1 + sum(1 for x in v if x >= o[k])) / (1 + len(v))
            rr[k] = dict(obs=o[k], null=mu, p=p); line += f'{k} {o[k]:.3f} vs {mu:.3f} (P {p:.3f}); '
        P(line + '(null = same number of dies drawn from all sites\' sealing dies, length-matched)')
        R[f'{LV}_c2_WH_{ES}'] = rr
        if LV == 'seq_raw' and ES == 'DES':
            P('     warehouse DES shared:', shared_elements(wi).most_common(10))

P('\n##### IM77 Lothal sealings: sides = impressions (loop71 die assignment in M space, then bridged to W)')
T, pp = im77_sides()
BR = json.load(open(ROOT + 'data/derived/bridge_extended.json')); PROP = json.load(open(ROOT + 'data/derived/dark/bridge_proposals.json'))
M2W = {}
for w, ms in BR.items():
    for m in ms: M2W.setdefault(m, []).append(int(w))
for p in PROP['proposals']: M2W.setdefault(p['M'], []).append(p['W'])
M2W = {m: min(v) for m, v in M2W.items()}
raw = collections.defaultdict(lambda: collections.defaultdict(list))
for r in csv.DictReader(open(ROOT + 'data/im77/im77_corpus_lines.csv')):
    if r['site'] != 'Lothal' or r['object_type'] != 'sealing' or r['line'] == '9' or not r['signs_clean'].strip(): continue
    raw[r['text_no']][int(r['side'])].extend(int(x) for x in r['signs_clean'].split())
imps = [(f'{k}.{s}', t) for k, sides in raw.items() for s, t in sides.items()]
lab = L71.assign_dies(imps)
inc = collections.OrderedDict()
for k, sides in raw.items():
    ds = []
    for s, t in sides.items():
        if not [x for x in t if x]: continue
        d = lab[f'{k}.{s}']
        if d not in ds: ds.append(d)
    ds2 = [d for d in ds if not (d[0] == 'u' and any(e != d and L71.fits(list(d[1:]), [x for x in (e[1:] if e[0] == 'u' else e)]) for e in ds))]
    inc[k] = ds2
allseq = [[M2W.get(x, 10000 + x) for x in s['seq']] for L in T.values() for s in L if s['seq']]
QUAL = build_qual(allseq); FR = frame_inventory(QUAL)
def tw(d): return [M2W.get(x, 10000 + x) for x in (d[1:] if d[0] == 'u' else d) if x]
P(f'  IM77 Lothal sealings {len(inc)}, with >= 2 dies {sum(1 for v in inc.values() if len(v) >= 2)}')
for ES in ['FULL', 'MID', 'DES']:
    items = []
    for c, ds in inc.items():
        for k, d in enumerate(ds):
            t = tw(d)
            e = element_sets(t, QUAL, FR)[ES] if t else ()
            # a count side (numeral + M328 = W700) is a field, not a die: drop
            if t and t[-1] == 700 and all(x in NUM or x == 700 for x in t): continue
            items.append((c, k, e, ('sealing', min(len(e), 5))))
    R[f'IM77_c1_{ES}'] = within(f'IM77 Lothal dies on one sealing, {ES}', items, NPERM)
    if ES == 'DES': P('     shared:', shared_elements(items).most_common(8))
save('loop77_c3', R)
