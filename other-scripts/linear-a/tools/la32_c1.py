#!/usr/bin/env python3
"""LA-32 cycle 1: build shape and sound matrices; score real one-sign confusions in LA and LB;
modern re-reading control (SigLA vs lineara.xyz: confusions by modern readers, visual by construction)."""
import sys, os, json, unicodedata, pickle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from la32_common import *

rng = np.random.default_rng(32)
out = []
def log(s):
    print(s, flush=True); out.append(s)

# ---------------- LA matrices
lamap = la_signmap()
W = la_words()
la_signs = sorted({s for r in W for s in r['w']} | set(lamap))
la_signs = [s for s in la_signs if s in lamap]
SM_LA, imgs_LA = shape_matrices(la_signs, lamap, FONTS['LA'])
iLA = {s: i for i, s in enumerate(la_signs)}
rows = la21_rows()
rs, RP, nf = rows['LA']
SND_LA = np.full((len(la_signs),) * 2, np.nan)
for a in rs:
    for b in rs:
        if a in iLA and b in iLA and a != b:
            SND_LA[iLA[a], iLA[b]] = RP[rs.index(a), rs.index(b)]
# outside check only: LB values carried by the conventional LA readings
def la_cv(s):
    return lb_cv(s.lower()) if not s.startswith('*') else None
CON_LA = np.full(SND_LA.shape, np.nan); VOW_LA = np.full(SND_LA.shape, np.nan)
for a in la_signs:
    for b in la_signs:
        ca, cb = la_cv(a), la_cv(b)
        if ca and cb and a != b:
            CON_LA[iLA[a], iLA[b]] = float(ca[0] == cb[0]); VOW_LA[iLA[a], iLA[b]] = float(ca[1] == cb[1])
VIS_LA = combine(SM_LA, ['blur', 'chamfer', 'hog', 'prof'])

# ---------------- LB matrices
B = lb_words()
lb_signs = sorted({s for r in B for s in r['w']})
SM_LB, imgs_LB = shape_matrices(lb_signs, LB_VAL, FONTS['LB'])
iLB = {s: i for i, s in enumerate(lb_signs)}
VIS_LB = combine(SM_LB, ['blur', 'chamfer', 'hog', 'prof'])
CON_LB = np.full((len(lb_signs),) * 2, np.nan); VOW_LB = CON_LB.copy()
for a in lb_signs:
    for b in lb_signs:
        ca, cb = lb_cv(a), lb_cv(b)
        if ca and cb and a != b:
            CON_LB[iLB[a], iLB[b]] = float(ca[0] == cb[0]); VOW_LB[iLB[a], iLB[b]] = float(ca[1] == cb[1])
rsb, RPB, nfb = rows['LB']
SND_LB = np.full(CON_LB.shape, np.nan)
for a in rsb:
    for b in rsb:
        al, bl = a.lower(), b.lower()
        if al in iLB and bl in iLB and a != b:
            SND_LB[iLB[al], iLB[bl]] = RPB[rsb.index(a), rsb.index(b)]
pickle.dump(dict(la_signs=la_signs, lb_signs=lb_signs, SM_LA=SM_LA, SM_LB=SM_LB, VIS_LA=VIS_LA, VIS_LB=VIS_LB,
                 SND_LA=SND_LA, SND_LB=SND_LB, CON_LA=CON_LA, VOW_LA=VOW_LA, CON_LB=CON_LB, VOW_LB=VOW_LB, lamap=lamap),
            open(os.path.join(CK, 'mats.pkl'), 'wb'))

def cor_nan(A, Bm):
    iu = np.triu_indices(A.shape[0], 1); a, b = A[iu], Bm[iu]; ok = ~np.isnan(a) & ~np.isnan(b)
    return float(np.corrcoef(a[ok], b[ok])[0, 1]), int(ok.sum())
log('# LA-32 cycle 1: real confusions vs shape and sound')
log(f'LA signs with glyphs {len(la_signs)}; LB signs {len(lb_signs)}; la21 LA row runs {nf}, LB {nfb}')
for k in SM_LB:
    log(f'  LB shape metric {k}: corr with same-consonant {cor_nan(rank_norm(SM_LB[k]), CON_LB)[0]:.3f}, same-vowel {cor_nan(rank_norm(SM_LB[k]), VOW_LB)[0]:.3f}')
log(f'  LB VIS vs CON {cor_nan(VIS_LB, CON_LB)}, VIS vs blind rows {cor_nan(VIS_LB, SND_LB)}, CON vs blind rows {cor_nan(CON_LB, SND_LB)}')
log(f'  LA VIS vs blind rows {cor_nan(VIS_LA, SND_LA)}; LA VIS vs LB-value consonant {cor_nan(VIS_LA, CON_LA)}')
log(f'  metric agreement (LA): ' + ', '.join(f'{a}-{b} {cor_nan(rank_norm(SM_LA[a]), rank_norm(SM_LA[b]))[0]:.2f}' for a, b in [('blur', 'chamfer'), ('blur', 'hog'), ('chamfer', 'prof'), ('hog', 'prof')]))

def block(tag, E, mats, idx):
    for name, M in mats:
        r = score_edges(E, M, idx, nrep=3000, rng=rng)
        log(f'  {tag:34s} {name:10s} {fmt(r)}')
        yield name, r

res = {}
# ---------------- LB real
log('\n## Linear B (KN+PY), one-sign pairs within a site, words >= 3 signs')
for posn in ('all', 'final', 'nonfinal', 'medial'):
    E = one_sign_pairs(B, 3, 'site', posn)
    for sub, f in (('all', lambda e: True), ('samescribe', lambda e: e['samescribe']), ('rare-variant', lambda e: e['rare'])):
        Es = [e for e in E if f(e)]
        res[f'LB_{posn}_{sub}'] = dict(block(f'LB {posn} {sub}', Es, [('VIS', VIS_LB), ('CON', CON_LB), ('VOW', VOW_LB), ('blindrow', SND_LB)], iLB))

# ---------------- LA real
log('\n## Linear A, one-sign pairs, words >= 3 signs (any site) and 2-sign pairs within document/scribe')
WA = la_words()
for posn in ('all', 'final', 'nonfinal'):
    E = one_sign_pairs(WA, 3, None, posn)
    for sub, f in (('all', lambda e: True), ('rare-variant', lambda e: e['rare']), ('same site', None)):
        if sub == 'same site':
            Es = [e for e in one_sign_pairs(WA, 3, 'site', posn)]
        else:
            Es = [e for e in E if f(e)]
        res[f'LA_{posn}_{sub}'] = dict(block(f'LA {posn} {sub}', Es, [('VIS', VIS_LA), ('blindrow', SND_LA), ('LBcon*', CON_LA), ('LBvow*', VOW_LA)], iLA))
E2 = [e for e in one_sign_pairs(WA, 2, 'site', 'all') if e['L'] == 2 and (e['samedoc'] or e['samescribe'])]
res['LA_2sign_local'] = dict(block('LA 2-sign same doc/scribe', E2, [('VIS', VIS_LA), ('blindrow', SND_LA), ('LBcon*', CON_LA), ('LBvow*', VOW_LA)], iLA))

# ---------------- modern readers: SigLA vs lineara.xyz substitutions
ab2t = {}
for s, ch in lamap.items():
    try:
        n = unicodedata.name(ch).split()
    except ValueError:
        continue
    ab = n[3] if len(n) > 3 else ''
    m = re.match(r'^(AB|A)0*(\d+)', ab)
    if m:
        ab2t[m.group(1) + m.group(2)] = s
sig = json.load(open(os.path.join(DATA, 'la22', 'la22_sigla_test.json')))
ES = []
for ev in sig['sub_events_nonsys']:
    a, b = ab2t.get(ev[1]), ab2t.get(ev[2])
    if a and b:
        ES.append(dict(a=a, b=b))
log(f'\n## Modern re-readings (SigLA vs lineara.xyz), mapped {len(ES)}/{len(sig["sub_events_nonsys"])}: ' + ' '.join(f"{e['a']}>{e['b']}" for e in ES))
res['SIGLA'] = dict(block('SigLA re-readings', ES, [('VIS', VIS_LA), ('blindrow', SND_LA), ('LBcon*', CON_LA), ('LBvow*', VOW_LA)], iLA))
ESn = [e for e in ES if '118' not in e['a'] + e['b']]
res['SIGLA_no118'] = dict(block('SigLA re-readings w/o *118', ESn, [('VIS', VIS_LA), ('blindrow', SND_LA), ('LBcon*', CON_LA)], iLA))
json.dump(res, open(os.path.join(CK, 'c1_res.json'), 'w'), indent=1)
open(os.path.join(CK, 'c1_report.txt'), 'w').write('\n'.join(out))
