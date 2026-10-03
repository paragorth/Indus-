"""S-DARK-65 cycle 1: catalogue of every sealing / tag with >= 2 distinct seal impressions.
Usage: python3 tools/dark_loop65_c1.py <seq_raw|seq_strong|seq_all>"""
import sys, collections, json
sys.path.insert(0, '/home/user/Indus-/tools')
from dark_loop65 import *
LV = sys.argv[1] if len(sys.argv) > 1 else 'seq_raw'
objs, S = load_sealings(LV)
seals = seal_texts(objs)
allseqs = [f['seq'] for o in objs.values() for f in o['faces'] if f['seq']]
parse = make_parser(learn_qual(allseqs)); middle = middle_fn(parse)
out = []
def P(*a):
    s = ' '.join(str(x) for x in a); print(s); out.append(s)
P(f'== S-DARK-65 cycle 1 ({LV}): catalogue of multi-impression sealings')
P(f'TAG objects in inscriptions.csv: {len(S)}; with >= 2 recorded faces: {sum(1 for o in S.values() if len(o["faces"]) >= 2)}')
multi_rec = [o for o in S.values() if len(o['faces']) >= 2]
multi = [o for o in S.values() if len(o['imps']) >= 2]
multi_leg1 = [o for o in S.values() if len(o['imps']) + o['illegible'] >= 2 and len(o['imps']) >= 1]
P(f'  after collapsing identical dies (000 wildcard, fragment-in-whole): >= 2 distinct LEGIBLE impressions: {len(multi)};'
  f' >= 2 distinct impressions of which >= 1 legible: {len(multi_leg1)};'
  f' same-die repeats only: {sum(1 for o in multi_rec if len(o["imps"]) + o["illegible"] < 2)}')
P('\n-- by site (objects: all TAG / >=2 faces / >=2 distinct legible / >=2 distinct incl. illegible):')
for site in sorted(set(o['site'] for o in S.values())):
    a = [o for o in S.values() if o['site'] == site]
    P(f'   {site:14s} {len(a):4d} {sum(1 for o in a if len(o["faces"]) >= 2):4d} {sum(1 for o in a if len(o["imps"]) >= 2):4d}'
      f' {sum(1 for o in a if len(o["imps"]) + o["illegible"] >= 2 and o["imps"]):4d}')
P('\n-- by carrier code (TAG sub-code) and Lothal back type (Frenez & Tosi 2005) for the >= 2 distinct legible set:')
P('   carrier:', dict(collections.Counter(o['carrier'] for o in multi)))
P('   Lothal back:', dict(collections.Counter(o['back'] or '?' for o in multi if o['site'] == 'Lothal')))
P('   Lothal context:', dict(collections.Counter(o['ctx'] or '?' for o in multi if o['site'] == 'Lothal')))
# Frenez & Tosi impression counts vs Wells faces at Lothal
backs = lothal_backs()
ft_multi = {k: v for k, v in backs.items() if v['nimp'] >= 2}
w_lothal = {o['cisi']: o for o in S.values() if o['site'] == 'Lothal'}
P(f'\n-- Lothal, Frenez & Tosi Table 1: {len(backs)} sealings, {len(ft_multi)} with >= 2 seal impressions '
  f'({dict(collections.Counter(v["nimp"] for v in ft_multi.values()))}); '
  f'of these Wells records >= 2 faces for {sum(1 for k in ft_multi if k in w_lothal and len(w_lothal[k]["faces"]) >= 2)}, '
  f'1 face for {sum(1 for k in ft_multi if k in w_lothal and len(w_lothal[k]["faces"]) == 1)}, none for {sum(1 for k in ft_multi if k not in w_lothal)}')
P('   F&T >= 2 impressions, Wells faces:', [(k, v['nimp'], len(w_lothal[k]['faces']) if k in w_lothal else 0, v['back'], v['ctx']) for k, v in ft_multi.items()])
P('   bale sealings L-161..170 (elephant, 817-2-48-740): F&T impressions', [(k, backs[k]['nimp'], backs[k]['back']) for k in backs if k in [f'L-{i}' for i in range(161, 171)]])
P('\n-- full list (>= 2 distinct impressions incl. illegible; * = matches a seal text somewhere; head in []; middle in {}):')
for o in sorted(multi_leg1, key=lambda o: (o['site'], o['oid'])):
    P(f'  {o["oid"]:6s} {o["cisi"]:8s} {o["site"]:12s} {o["type"]:7s} sides={o["sides"]:2s} faces={len(o["faces"])} dies={len(o["dies"])} illegible={o["illegible"]}'
      + (f' back={o["back"]}/{o["ctx"]} F&T n={o["nimp_ft"]}' if o['back'] else ''))
    for f in o['imps']:
        m = matches_seal(f, seals)
        P(f'        {"*" if m else " "} {fmt(f["seq"]):40s} [{head_of(f["seq"])}] {{{fmt(middle(f["seq"]))}}} {"complete" if f["complete"] else "fragment"} x{f["ndup"]}'
          + (f' seal at {sorted(seals[tuple(f["seq"])])}' if m == 'exact' else ''))
# recurring impressions across multi-impression sealings
P('\n-- impression texts recurring on >= 2 different multi-impression sealings:')
cnt = collections.Counter(); where = collections.defaultdict(list)
for o in multi_leg1:
    for f in o['imps']:
        cnt[tuple(f['seq'])] += 1; where[tuple(f['seq'])].append(o['cisi'] or o['oid'])
for t, n in cnt.most_common():
    if n >= 2: P(f'   {fmt(t):40s} x{n}  on {where[t]}')
pairs = sum(len(o['imps']) * (len(o['imps']) - 1) // 2 for o in multi)
P(f'\nSUMMARY {LV}: {len(multi)} sealings with >= 2 distinct legible impressions ({pairs} impression pairs), {len(multi_leg1)} incl. one illegible partner;'
  f' sites {dict(collections.Counter(o["site"] for o in multi))}; upper bound on the share of all {len(S)} recorded sealings: {len(multi)}/{len(S)} = {len(multi)/len(S):.1%}')
open(OUT + f'loop65_c1_{LV}.txt', 'w').write('\n'.join(out) + '\n')
