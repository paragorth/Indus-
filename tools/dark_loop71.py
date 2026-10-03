#!/usr/bin/env python3
"""Loop 71, cycle 1: the Lothal sealing catalogue (93 sealings, Frenez & Tosi 2005) joined with Rao 1985 plates,
Wells faces, IM77 sides, die (seal identity) assignment; and two checks of the archive reconstruction.
  T1  Frenez & Tosi assign 18 sealings to the warehouse ONLY because they share a seal with the warehouse group
      (context W1). Independent check: Rao's antiquity (register) numbers. Calibrate on the sealings whose find-spot
      Rao prints (plates): which register range is the GX 10 warehouse lot? Then ask how many W1, W2 and '?'
      sealings fall in that range. Control: the same share for the '?' group (no seal link); Fisher exact.
  T2  Back type, two observers: Rao's reverse description vs Frenez & Tosi's fastening type (same object).
  T3  Recurrent seals recovered from transcriptions (Wells, IM77) vs Frenez & Tosi's 13 recurrent seals / 76
      impressions; impressions per sealing FT vs Wells vs IM77.
Writes data/derived/dark/loop71_lothal_sealings.csv and loop71_c1.txt.
Usage: python3 tools/dark_loop71.py [seq_raw|seq_strong|seq_all]
"""
import sys, csv, json, collections, math
sys.path.insert(0, '/home/user/Indus-/tools')
from dark_loop71_common import *

LV = sys.argv[1] if len(sys.argv) > 1 else 'seq_raw'
LOG = []
def P(*a):
    s = ' '.join(str(x) for x in a); print(s); LOG.append(s)

def fisher2(a, b, c, d):
    """two-sided Fisher exact for [[a,b],[c,d]]"""
    n = a + b + c + d; r1 = a + b; c1 = a + c
    def pr(x): return math.comb(r1, x) * math.comb(n - r1, c1 - x) / math.comb(n, c1)
    p0 = pr(a); lo = max(0, c1 - (n - r1)); hi = min(r1, c1)
    return min(1.0, sum(pr(x) for x in range(lo, hi + 1) if pr(x) <= p0 * (1 + 1e-9)))

FT = {r[0]: dict(nimp=r[1], back=r[2], ctx=r[3]) for r in json.load(open(FTJ))['rows']}
W = wells_lothal(LV)
im = load_im77()
IML = {k: o for k, o in im.items() if o['site'] == 'Lothal' and o['type'] == 'sealing'}

# ---- IM77 link: loop-24 aligner, accepted and unique on both sides ----
pairs = [e for e in json.load(open(L24)) if e['text_no'] in IML and e['accepted']]
cnt = collections.Counter(e['cisi'] for e in pairs)
IMLINK = {e['cisi']: e['text_no'] for e in pairs if cnt[e['cisi']] == 1}

# ---- dies: Wells (W numbers) and IM77 (M numbers), each within Lothal ----
wimps = [(f['id'], f['toks']) for k, fs in W.items() for f in fs]
wlab = assign_dies(wimps)
# IM77 sides with 0 kept (load_im77 drops 0): re-read
from dark_loop37 import IM77 as IM77_PATH
iimps = []
imraw = collections.defaultdict(lambda: collections.defaultdict(list))
for r in csv.DictReader(open(IM77_PATH)):
    if r['site'] != 'Lothal' or r['object_type'] != 'sealing' or r['line'] == '9' or not r['signs_clean'].strip(): continue
    imraw[r['text_no']][int(r['side'])].extend(int(x) for x in r['signs_clean'].split())
for k, sides in imraw.items():
    for s, t in sides.items(): iimps.append((f'{k}.{s}', t))
ilab = assign_dies(iimps)

# ---- Rao plates by antiquity number and by plate code ----
RB_A = {p[1]: p for p in RAO_PLATES if p[1]}
RB_P = {p[0]: p for p in RAO_PLATES}
MANUAL_RAO = {'L-174': 'CLXII-A1', 'L-175': 'CLXII-A2', 'L-173': 'CLXII-F', 'L-215': 'CLXIII-G'}   # from FT descriptions

CLAIMED = {}
for c0, fs0 in W.items():
    pc0 = plate(fs0[0]['exid'])
    if pc0 in RB_P: CLAIMED[pc0] = c0
rows = []
keys = ['L-%d' % i for i in range(124, 217)] + [k for k in W if k not in FT]
for c in keys:
    ft = FT.get(c, {})
    fs = W.get(c, [])
    exid = fs[0]['exid'] if fs else ''
    a = antiq(exid) if fs else None
    pc = plate(exid) if fs else ''
    rp = None; rao_note = ''
    if pc and pc in RB_P:
        rp = RB_P[pc]
        if a and a in RB_A and RB_A[a][0] != pc: rao_note = f'antiquity no. {a} is Rao plate {RB_A[a][0]} but Wells gives plate {pc}; plate followed'
    elif a and a in RB_A:
        if RB_A[a][0] in CLAIMED: rao_note = f'antiquity no. {a} = Rao plate {RB_A[a][0]}, already the plate of {CLAIMED[RB_A[a][0]]} (Wells id clash; not joined)'
        else: rp = RB_A[a]; rao_note = 'joined by antiquity no. only'
    if not rp and c in MANUAL_RAO: rp = RB_P[MANUAL_RAO[c]]; rao_note = 'plate assigned from FT description'
    dies_w = []
    for f in fs:
        d = wlab[f['id']]
        if not [x for x in f['toks'] if x]: dies_w.append('illegible')
        elif d not in dies_w: dies_w.append(d)
    imt = IMLINK.get(c, '')
    dies_i = []
    if imt:
        for s in sorted(imraw.get(imt, {})):
            d = ilab[f'{imt}.{s}']
            if d not in dies_i: dies_i.append(d)
    findspot = (rp[6] + (' ' + rp[7] if rp[7] else '')) if rp else (fs[0]['area'] + ' ' + fs[0]['layer'] + ' ' + fs[0]['depth']).strip() if fs else ''
    src = ['FT2005 Table 1' if ft else '']
    if fs: src.append('Wells inscriptions.csv')
    if rp: src.append('Rao 1985 pp.325-327')
    if imt: src.append('IM77 (loop24 link)')
    if c in FT_NOTES: src.append('FT2005 text')
    rows.append(dict(cisi=c, ft_n_impressions=ft.get('nimp', ''), ft_back_type=ft.get('back', ''), ft_context=ft.get('ctx', ''),
                     ft_note=FT_NOTES.get(c, ''), ft_hub_named='Y' if c in FT_HUB else '',
                     wells_ids=';'.join(f['id'] for f in fs), wells_n_faces=len(fs),
                     wells_impressions=' | '.join('-'.join(map(str, f['toks'])) for f in fs),
                     wells_dies=' | '.join(fmt(d) if d != 'illegible' else d for d in dies_w),
                     wells_emblem=(fs[0]['symbol'] if fs else ''), wells_excavation_id=exid, antiquity_no=a or '',
                     rao_plate=rp[0] if rp else '', rao_n_impressions=rp[2] if rp else '', rao_obverse=rp[3] if rp else '',
                     rao_reverse=rp[4] if rp else '', rao_phase=rp[5] if rp else '', findspot=findspot, rao_note=rao_note,
                     im77_text_no=imt, im77_sides=' | '.join(' '.join(map(str, imraw[imt][s])) for s in sorted(imraw.get(imt, {}))) if imt else '',
                     im77_dies=' | '.join(fmt(d) for d in dies_i), source='; '.join(x for x in src if x)))

# ---- register-lot calibration (T1) ----
P(f'== S-DARK-71 cycle 1 ({LV}): Lothal sealing catalogue + archive checks')
P(f'catalogue: {len(rows)} rows ({sum(1 for r in rows if r["ft_context"])} Frenez & Tosi sealings L-124..L-216, '
  f'{sum(1 for r in rows if r["wells_ids"])} with Wells faces, {sum(1 for r in rows if r["rao_plate"])} with a Rao plate entry, '
  f'{sum(1 for r in rows if r["im77_text_no"])} with a unique IM77 link)')
cal = [(p[1], 'GX 10' in p[6]) for p in RAO_PLATES if p[1]]
gx = sorted(n for n, g in cal if g); ngx = sorted(n for n, g in cal if not g)
P(f'T1 calibration: Rao find-spots with antiquity numbers: GX 10 warehouse lot {len(gx)} -> numbers {gx}')
P(f'   non-GX 10 {len(ngx)} -> {ngx}')
LO, HI = 1826, 2100
inside_gx = sum(1 for n in gx if LO <= n <= HI); inside_ngx = sum(1 for n in ngx if LO <= n <= HI)
P(f'   rule: register no. {LO}-{HI} = GX 10 lot. Calibration: {inside_gx}/{len(gx)} GX 10 numbers inside, {inside_ngx}/{len(ngx)} others inside '
  f'(14586 = "SRG 2, GX 10, unstratified" counted as GX 10 and outside the range)')
by = collections.defaultdict(lambda: [0, 0, 0, []])   # inside, outside, no number
for r in rows:
    if not r['ft_context']: continue
    a = r['antiquity_no']
    g = by[r['ft_context']]
    if a == '': g[2] += 1
    elif LO <= a <= HI: g[0] += 1; g[3].append(r['cisi'])
    else: g[1] += 1
for k in ('W', 'W1', 'W2', 'O', '?'):
    g = by[k]
    P(f'   FT context {k:3s}: register no. in GX 10 lot {g[0]}, outside {g[1]}, no number {g[2]}')
a, b = by['W1'][0], by['W1'][1]; c, d = by['?'][0], by['?'][1]
P(f'   W1 (assigned by shared seals) {a}/{a+b} in the lot vs ? (no seal link) {c}/{c+d}: Fisher P = {fisher2(a, b, c, d):.3f}')
a2, b2 = by['W2'][0], by['W2'][1]
P(f'   W2 (Rao: SRG 3, no recurrent seal) {a2}/{a2+b2} in the lot; O (outside) {by["O"][0]}/{by["O"][0]+by["O"][1]}')
P(f'   "?" sealings that the register places in the warehouse lot: {", ".join(by["?"][3])}')
mis = [r['cisi'] for r in rows if r['ft_context'] in ('?',) and r['rao_plate']]
P(f'   "?" sealings that DO have a Rao plate entry with find-spot: ' + '; '.join(f'{r["cisi"]} = {r["rao_plate"]} {r["findspot"]}' for r in rows if r['cisi'] in mis))
# ---- T2 back type two observers ----
P('T2 back type: Rao reverse vs FT fastening (sealings with both):')
t2 = collections.Counter()
for r in rows:
    if r['rao_reverse'] and r['ft_back_type']:
        rr = r['rao_reverse']
        rk = 'reed/stick/cane' if any(w in rr for w in ('reed', 'stick', 'groove')) else ('plain/flat' if any(w in rr for w in ('plain', 'flat')) else 'other')
        t2[(rk, r['ft_back_type'])] += 1
for (rk, fk), n in sorted(t2.items()): P(f'   Rao {rk:16s} x FT {fk:15s} {n}')
# ---- T3 recurrence ----
P('T3 recurrent dies (seal identities) recovered at Lothal:')
for lab, name, imps_ in ((wlab, 'Wells', wimps), (ilab, 'IM77', iimps)):
    objs_of = collections.defaultdict(set)
    for k, t in imps_:
        d = lab[k]
        if d[0] == 'u': continue
        objs_of[d].add(k.split('.')[0])
    rec = {d: s for d, s in objs_of.items() if len(s) >= 2}
    P(f'   {name}: {len(imps_)} impressions on {len({k.split(".")[0] for k, t in imps_})} sealings; recurrent dies {len(rec)} covering '
      f'{sum(len(s) for s in rec.values())} sealings: ' + '; '.join(f'{fmt(d)} x{len(s)}' for d, s in sorted(rec.items(), key=lambda x: -len(x[1]))))
P(f'   FT 2005 (from photographs): 13 recurrent seals, >= 76 of 130 impressions; hub {FT_SUMMARY["hub_imps"]}, elephant {FT_SUMMARY["elephant_imps"]}; 57-67 seals in all, 25-30 in the warehouse')
nft = sum(r['ft_n_impressions'] or 0 for r in rows if r['ft_context'])
nw = sum(len([d for d in r['wells_dies'].split(' | ') if d and d != 'illegible']) for r in rows if r['ft_context'])
P(f'   impressions: FT {nft}; Wells distinct legible dies on the same sealings {nw}; Wells faces {sum(r["wells_n_faces"] for r in rows if r["ft_context"])}')
multi_ft = [r for r in rows if r['ft_context'] and (r['ft_n_impressions'] or 0) >= 2]
P(f'   FT multi-impression sealings {len(multi_ft)}: Wells records >= 2 faces on {sum(1 for r in multi_ft if r["wells_n_faces"] >= 2)}, '
  f'1 face on {sum(1 for r in multi_ft if r["wells_n_faces"] == 1)}, none on {sum(1 for r in multi_ft if r["wells_n_faces"] == 0)}')
nowells = [r['cisi'] for r in rows if r['ft_context'] and not r['wells_ids']]
P(f'   FT sealings absent from Wells ({len(nowells)}): {", ".join(nowells)}')

cols = list(rows[0].keys())
with open(CAT, 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=cols); w.writeheader(); w.writerows(rows)
P(f'wrote {CAT}')
open(OUT + f'loop71_c1_{LV}.txt', 'w').write('\n'.join(LOG) + '\n')
