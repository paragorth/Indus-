"""Loop 54 cycle 3: aggregate cycles 1-2 into the re-graded table.
Inputs: loop54_c1_<level>_big.json (3 levels), loop54_c1_<level>_heldout.json (seq_raw, seq_all), loop54_c2_im77.json.
Rules (strict, as set for this loop):
  - an 'adjacency' claim is chain-explained by construction (first-order chain reproduces bigrams in expectation);
  - a 'position' / 'distance' / 'trigram' / 'corpus' claim is 'beyond chain' at MD+H if it lies outside the Markov-1 AND Markov-2
    2.5-97.5% bands at all three merge levels (merge-dependent if only at some);
  - it 'replicates' if it is also outside both bands on the held-out sites (either level) or on the IM77-only set, in the same direction;
  - entry grade: A kept only if some non-adjacency claim is beyond both chains at every level AND replicates;
    A -> B if beyond chains at MD+H but not replicated out of sample (or only at some merge levels);
    A/B -> C if every claim is chain-explained, unless an outside fact is recorded: then B at most;
    B kept if beyond chains (replication not required for B), else C (B- with outside fact);
    C entries and grammar statements: marked chain-explained / beyond chain / untestable only.
Output: loop54_regrade.csv, loop54_c3_table.txt"""
import json, csv, collections
from dark_loop54_common import DARK, fmt

LEV = ['seq_raw', 'seq_strong', 'seq_all']
big = {L: json.load(open(DARK + f'loop54_c1_{L}_big.json')) for L in LEV}
held = {L: json.load(open(DARK + f'loop54_c1_{L}_heldout.json')) for L in ('seq_raw', 'seq_all')}
im = json.load(open(DARK + 'loop54_c2_im77.json'))
n = len(big['seq_all'])
def key(r): return (r['entry'], r['claim'])
def beyond(r):
    """None = untestable; True = outside both M1 and M2 bands; False otherwise. Returns (flag, direction)."""
    m1, m2 = r['M1'], r['M2']
    if m1['inside'] is None or m2['inside'] is None or r['obs'] != r['obs']: return None, ''
    return (not m1['inside']) and (not m2['inside']), m1['dir'] or m2['dir']

rows = []; per_entry = collections.defaultdict(list)
for i in range(n):
    r = big['seq_all'][i]; k = key(r)
    assert all(key(big[L][i]) == k for L in LEV)
    kind = r['kind']
    bs = {L: beyond(big[L][i]) for L in LEV}
    flags = [bs[L][0] for L in LEV]
    if kind == 'adjacency': md = 'adjacency (chain by construction)'
    elif all(f is None for f in flags): md = 'untestable'
    elif all(f for f in flags): md = 'beyond chains (' + bs['seq_all'][1] + ')'
    elif any(f for f in flags): md = 'beyond chains only at ' + '/'.join(L for L in LEV if bs[L][0]) + ' (merge-dependent)'
    else: md = 'chain-explained'
    # held-out sites
    hs = {L: beyond(held[L][i]) for L in held}
    hflag = any(v[0] for v in hs.values() if v[0] is not None) and all(v[1] == bs['seq_all'][1] or not v[0] for v in hs.values())
    hdir = '/'.join(f"{L[4:]}:{'beyond' if v[0] else ('untestable' if v[0] is None else 'inside')}" for L, v in hs.items())
    # im77
    ir = im[i]; ib = beyond(ir)
    iflag = bool(ib[0]) and ib[1] == bs['seq_all'][1]
    idir = 'beyond' if ib[0] else ('untestable/no tokens' if ib[0] is None else 'inside')
    repl = hflag or iflag
    g = r.get('G', {})
    gflag = '' if not g or g.get('inside') is None else ('in' if g['inside'] else g['dir'])
    rec = {'entry': r['entry'], 'grade': r['grade'], 'claim': r['claim'], 'kind': kind, 'obs_all': r['obs'],
           'M1': f"{fmt(r['M1']['med'])} [{fmt(r['M1']['lo'])}-{fmt(r['M1']['hi'])}]", 'M2': f"{fmt(r['M2']['med'])} [{fmt(r['M2']['lo'])}-{fmt(r['M2']['hi'])}]",
           'M1E': ('in' if r['M1E']['inside'] else r['M1E']['dir']) if r['M1E']['inside'] is not None else '-', 'G': gflag,
           'mdh': md, 'obs_raw': big['seq_raw'][i]['obs'], 'obs_strong': big['seq_strong'][i]['obs'],
           'heldout': f"obs {fmt(held['seq_all'][i]['obs'])} vs M1 {fmt(held['seq_all'][i]['M1']['med'])} [{fmt(held['seq_all'][i]['M1']['lo'])}-{fmt(held['seq_all'][i]['M1']['hi'])}] ({hdir})",
           'im77': f"obs {fmt(ir['obs'])} vs M1 {fmt(ir['M1']['med'])} [{fmt(ir['M1']['lo'])}-{fmt(ir['M1']['hi'])}] ({idir})",
           'beyond_all_levels': all(f for f in flags) if kind != 'adjacency' else False, 'beyond_some': any(f for f in flags) if kind != 'adjacency' else False,
           'replicated': repl if kind != 'adjacency' else False, 'outside': r['outside'], 'srow': r['srow']}
    rows.append(rec); per_entry[r['entry']].append(rec)

def propose(entry, recs):
    g = recs[0]['grade']
    base = g.split()[0].strip('/')
    outside = any(x['outside'] for x in recs)
    strong = sorted([x for x in recs if x['beyond_all_levels']], key=lambda x: x['M1E'] == 'in')   # M1E-robust claims first
    some = [x for x in recs if x['beyond_some']]
    rep = sorted([x for x in strong if x['replicated']], key=lambda x: x['M1E'] == 'in')
    if g == 'grammar' or base.startswith('C'):
        if strong: return g if g == 'grammar' else base, 'beyond chains' + (' and replicated' if rep else '') + ': ' + strong[0]['claim']
        if some: return base if base.startswith('C') else g, 'beyond chains at some merge levels only: ' + some[0]['claim']
        return base if base.startswith('C') else g, 'chain-explained' + (' (outside fact: ' + next(x['outside'] for x in recs if x['outside']) + ')' if outside else '')
    if base.startswith('A'):
        if rep: return 'A', 'beats both chains at all levels and replicates: ' + rep[0]['claim']
        if strong: return 'B', 'beats both chains at MD+H but not replicated out of sample (power or absent): ' + strong[0]['claim']
        if some: return 'B', 'beyond chains only at some merge levels: ' + some[0]['claim']
        if outside: return 'B', 'sequence claims chain-explained; kept at B on the outside fact: ' + next(x['outside'] for x in recs if x['outside'])
        return 'C', 'every sequence claim is chain-explained; no outside fact'
    # B
    if strong: return 'B', 'beats both chains' + (' and replicates' if rep else ' (MD+H only)') + ': ' + strong[0]['claim']
    if some: return 'B-', 'beyond chains only at some merge levels: ' + some[0]['claim']
    if outside: return 'B-', 'sequence claims chain-explained; outside fact only: ' + next(x['outside'] for x in recs if x['outside'])
    return 'C', 'every sequence claim is chain-explained; no outside fact'

def propose2(e, recs):
    pg, why = propose(e, recs)
    dec = [x for x in recs if x['claim'] in why]
    if dec and dec[0]['M1E'] == 'in' and dec[0]['kind'] != 'adjacency': why += ' [end-state fact: an order-1 chain with END reproduces it]'
    return pg, why
prop = {e: propose2(e, recs) for e, recs in per_entry.items()}
with open(DARK + 'loop54_regrade.csv', 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['sign', 'current grade', 'claim', 'statistic (obs seq_all; seq_raw; seq_strong)', 'Markov-1 null (med [band])', 'Markov-2 null', 'held-out (sites; IM77-only)', 'proposed grade', 'reason', 'kind', 'MD+H verdict', 'M1E', 'S366 gen', 'outside fact', 'S-rows'])
    for e, recs in per_entry.items():
        pg, why = prop[e]
        for x in recs:
            w.writerow([e, x['grade'], x['claim'], f"{fmt(x['obs_all'])}; {fmt(x['obs_raw'])}; {fmt(x['obs_strong'])}", x['M1'], x['M2'], x['heldout'] + ' | IM77 ' + x['im77'], pg, why, x['kind'], x['mdh'], x['M1E'], x['G'], x['outside'], x['srow']])
out = ['# Loop 54 cycle 3: re-graded table (one line per entry; decisive claim in the reason). Grades: proposed vs current.', '| entry | current | proposed | reason | claims beyond chains at all 3 levels | replicated out of sample |', '|---|---|---|---|---|---|']
changes = []
for e, recs in per_entry.items():
    pg, why = prop[e]; cur = recs[0]['grade']
    nb = sum(x['beyond_all_levels'] for x in recs); nr = sum(x['replicated'] for x in recs if x['beyond_all_levels'])
    out.append(f'| {e} | {cur} | {pg} | {why} | {nb}/{len(recs)} | {nr} |')
    if cur != 'grammar' and pg != cur.split()[0].strip('/').rstrip('-') and not cur.startswith(pg + ' '): changes.append((e, cur, pg, why))
out.append('\n## Grade changes')
for e, cur, pg, why in changes: out.append(f'- {e}: {cur} -> {pg}: {why}')
out.append('\n## GRAMMAR statements')
for e, recs in per_entry.items():
    if recs[0]['grade'] == 'grammar' or e.startswith('GRAMMAR'):
        for x in recs: out.append(f"- {e}: {x['claim']}: obs {fmt(x['obs_all'])} vs M1 {x['M1']} M2 {x['M2']}; {x['mdh']}; held-out {x['heldout']}; IM77 {x['im77']}; S366 gen {x['G'] or '-'}")
open(DARK + 'loop54_c3_table.txt', 'w').write('\n'.join(out) + '\n')
print('\n'.join(out))
