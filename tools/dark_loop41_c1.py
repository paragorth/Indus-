"""Loop 41 cycle 1: (d) rebuild the corpus from the current inscriptions.csv; (a) duplicate / moulded-copy inflation.
Every headline statistic recomputed on canonical vs rebuilt, at seq_raw and seq_all, under four copy regimes:
rows (every object), site_type_text, site_text, die. Output: data/derived/dark/loop41_cycle1.txt"""
import sys, random, json, collections, statistics as st
sys.path.insert(0, '/home/user/Indus-/tools')
from dark_loop41_common import *

out = open(DARK + 'loop41_cycle1.txt', 'w')
def P(*a):
    print(*a); print(*a, file=out); out.flush()

R = rebuild()
Cc = load('canonical')
P(f'# Loop 41 cycle 1: rebuild + duplicate inflation')
P(f'canonical rows {len(Cc)}; rebuilt rows {len(R)} (from {sum(1 for _ in open(ROOT + "data/raw/inscriptions.csv")) - 1} csv rows)')
for lvl in ('seq_raw', 'seq_all'):
    for name, C in (('canonical', Cc), ('rebuilt', R)):
        d = {m: len(dedup(C, lvl, m)) for m in ('rows', 'site_type_text', 'site_text', 'die')}
        P(f'  {name} {lvl}: copies per regime {d}; sign types {len({x for r in C for x in r[lvl]})}; tokens {sum(len(r[lvl]) for r in C)}')
# which rows are new
cc = {r['cisi'] for r in Cc}
new = [r for r in R if r['cisi'] not in cc]
P(f'  rebuilt objects whose cisi is absent from canonical: {len(new)}; by type {collections.Counter(otype(r["type"]) for r in new).most_common(6)}; by site {collections.Counter(r["site"] for r in new).most_common(6)}')
# text-level agreement for shared cisi
byc = collections.defaultdict(set); byr = collections.defaultdict(set)
for r in Cc: byc[r['cisi']].add(tuple(r['seq_raw']))
for r in R: byr[r['cisi']].add(tuple(r['seq_raw']))
same = sum(1 for k in byc if byc[k] == byr.get(k)); P(f'  cisi with identical text sets canonical vs rebuilt: {same}/{len(byc)}')
diff = [k for k in byc if byc[k] != byr.get(k)][:8]
for k in diff: P(f'    changed: {k} canonical {sorted(byc[k])} rebuilt {sorted(byr.get(k, set()))}')

rnd = random.Random(41)
rows = []
for lvl in ('seq_raw', 'seq_all'):
    for name, C in (('canonical', Cc), ('rebuilt', R)):
        for mode in ('rows', 'site_type_text', 'site_text', 'die'):
            T = dedup(C, lvl, mode)
            res = {'corpus': name, 'level': lvl, 'regime': mode, 'n': len(T)}
            # S289 closers
            npass, lst, nmed, nmax = closer_paradigm(T, rnd, nnull=40)
            res['closers'] = npass; res['closers_null_max'] = nmax; res['closer_list'] = lst
            # S347 nesting (distinct site+text in the original); here on the regime's texts
            o, nm, nx, nsmall = nesting(T, rnd, nnull=30)
            res['nest'] = o; res['nest_null'] = nm; res['nest_ratio'] = o / nm if nm else float('nan')
            # S349 held-out
            h = p1p2(T, rnd, nnull=30); res.update({'P1': h['P1'], 'P1_null': h['P1_null'], 'P1x': h['P1'] / h['P1_null'] if h['P1_null'] else float('nan'),
                                                    'P2': h['P2'], 'P2_null': h['P2_null'], 'P2x': h['P2'] / h['P2_null'] if h['P2_null'] else float('nan'), 'n_heldout': h['n_heldout']})
            # S321 name ratio (seals, one per cisi+text already in 'die'; in 'rows' every seal row)
            u, b, ratio, nm_ = name_ratio(T, rnd); res['name_ratio'] = ratio; res['name_n'] = nm_
            # S-DARK-19
            g, ag, pairs, dif = anagram(T); res['anagram_groups'] = g; res['anagram_pairs'] = pairs; res['anagram_diff'] = (dif / pairs) if pairs else float('nan')
            M, fx, fr = fixed_pairs(T); res['pairs_tested'] = M; res['pairs_fixed'] = fx; res['fixed_share'] = fx / M if M else float('nan')
            # S-DARK-16
            nt, gh, ns, dd = ghosts_dead(T); res['ghost'] = gh / nt if nt else float('nan'); res['dead'] = dd / ns if ns else float('nan'); res['n_tags'] = nt
            # S366 closer change
            o, nl, pr = middle_closer_change(T, rnd, nnull=40); res['mid_chg'] = o; res['mid_chg_null'] = nl; res['mid_pairs'] = pr
            # S-DARK-15 W2
            (tw, mj), (etw, emj) = w2_rules(T, rnd, nnull=40); res['w2_twice'] = tw; res['w2_twice_E'] = etw; res['w2_741'] = mj; res['w2_741_E'] = emj
            res.update(frame_rates(T))
            rows.append(res)
            P(f"{name:9s} {lvl:10s} {mode:15s} n={res['n']:5d} | closers {npass} (null max {nmax}) | nest {o:.3f}/{nm:.3f}={res['nest_ratio']:.2f}x | P1 {res['P1']:.3f}/{res['P1_null']:.3f}={res['P1x']:.1f}x P2 {res['P2']:.3f}/{res['P2_null']:.3f}={res['P2x']:.1f}x (held-out n {res['n_heldout']}) | name {ratio:.3f} (n {nm_}) | anagram diff {res['anagram_diff']:.3f} ({pairs} pairs) fixed {fx}/{M}={res['fixed_share']:.2f} | ghost {res['ghost']:.2f} dead {res['dead']:.2f} | mid-chg {res['mid_chg']:.2f} vs {res['mid_chg_null']:.2f} ({pr}) | W2x2 {tw} (E {etw:.1f}) W2+741 {mj} (E {emj:.1f}) | opener {res['opener_first']:.3f} closer-last {res['closer_last']:.3f}")
json.dump(rows, open(DARK + 'loop41_cycle1.json', 'w'), indent=1)
P('\nclosers passing (canonical seq_raw rows):', [r['closer_list'] for r in rows if r['corpus'] == 'canonical' and r['level'] == 'seq_raw' and r['regime'] == 'rows'][0])
P('closers passing (rebuilt seq_raw die):', [r['closer_list'] for r in rows if r['corpus'] == 'rebuilt' and r['level'] == 'seq_raw' and r['regime'] == 'die'][0])
