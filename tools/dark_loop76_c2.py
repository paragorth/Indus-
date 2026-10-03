"""S-DARK-76 cycle 2: WHICH designations recur, and with what? For every designation on >= 2 documents of a kind:
documents, distinct whole texts (frames), heads (closer group, S289), opener register (text-initial opener sign),
sites, areas, copies of one mould / seal; cross-kind presence (the same designation on seals / sealings / tablets).
Lothal team of S-DARK-71 and Harappa tablet series flagged. Share of recurrences that are the SAME whole text
('mould / seal recurrence') vs the designation in a DIFFERENT frame ('designation recurrence').
Usage: python3 tools/dark_loop76_c2.py   (writes data/derived/dark/loop76_c2.txt)"""
import sys; sys.path.insert(0, '/home/user/Indus-/tools')
from dark_loop76_common import *
LOTHAL_TEAM = {(705, 500, 741, 1, 55, 220, 740, 90), (880, 2, 32, 590, 390, 740), (880, 2, 32, 590, 390), (861, 2, 236),
               (817, 2, 48, 740), (617, 2, 240, 220, 233, 520)}
fmt = lambda s: '-'.join(map(str, s))

def table(label, docs, kinds, top=15):
    T = tokens(docs, 2, kinds)
    by = collections.defaultdict(list)
    for i, m, f in T: by[m].append((docs[i], f))
    rec = {m: L for m, L in by.items() if len(L) >= 2}
    ntok = len(T); nrec = sum(len(L) for L in rec.values())
    same_text = sum(1 for L in rec.values() if len(set(f['seq'] for _, f in L)) == 1)
    one_head = sum(1 for L in rec.values() if len(set(f['hgroup'] for _, f in L)) == 1)
    one_site = sum(1 for L in rec.values() if len(set(d['site'] for d, _ in L)) == 1)
    one_area = sum(1 for L in rec.values() if len(set((d['site'], d['area']) for d, _ in L)) == 1)
    tok_same = sum(len(L) for L in rec.values() if len(set(f['seq'] for _, f in L)) == 1)
    P(f'  [{label}] tokens {ntok}; recurring designations {len(rec)} carrying {nrec} tokens ({nrec/ntok if ntok else 0:.2f}); '
      f'all copies ONE whole text {same_text}/{len(rec)} (tokens {tok_same}/{nrec}); one head {one_head}/{len(rec)}; one site {one_site}/{len(rec)}; one site x area {one_area}/{len(rec)}')
    for m, L in sorted(rec.items(), key=lambda x: -len(x[1]))[:top]:
        texts = collections.Counter(fmt(f['seq']) for _, f in L)
        heads = collections.Counter(f['hgroup'] for _, f in L)
        sites = collections.Counter(d['site'] for d, _ in L)
        op = sum(1 for _, f in L if f['opener']) / len(L)
        flag = ' LOTHAL-TEAM' if any(tuple(f['seq']) in LOTHAL_TEAM for _, f in L) else ''
        typ = collections.Counter(d['type'] for d, _ in L)
        P(f'     {fmt(m):22s} x{len(L):3d} texts {len(texts)} {dict(texts.most_common(3))} heads {dict(heads)} opener {op:.2f} sites {dict(sites)} types {dict(typ)}{flag}')
    return dict(tokens=ntok, rec=len(rec), rec_tokens=nrec, same_text=same_text, tok_same=tok_same, one_head=one_head, one_site=one_site, one_area=one_area)

R = {}
for LV in ['seq_raw', 'seq_strong', 'seq_all']:
    for filt in ['loose', 'strict']:
        D0 = wells_docs(LV, filt)
        for lev in ['act', 'die']:
            D = collapse_level(D0, lev)
            P(f'#### Wells {LV} {filt} {lev}')
            top = 15 if (LV == 'seq_raw' and filt == 'loose') else 0
            for k in [('seal',), ('sealing',), ('tablet_m',), ('tablet_i',), ('pot', 'other')]:
                R[f'{LV}_{filt}_{lev}_{"+".join(k)}'] = table(f'{LV}/{filt}/{lev} {"+".join(k)}', D, set(k), top)
        if filt == 'loose':
            # cross-kind presence
            D = collapse_level(D0, 'die')
            kinds = collections.defaultdict(collections.Counter)
            for i, m, f in tokens(D, 2): kinds[m][D[i]['kind']] += 1
            sl = [m for m in kinds if kinds[m]['sealing']]
            on_seal = [m for m in sl if kinds[m]['seal']]
            on_tab = [m for m in sl if kinds[m]['tablet_m'] or kinds[m]['tablet_i']]
            tb = [m for m in kinds if kinds[m]['tablet_m'] or kinds[m]['tablet_i']]
            tb_seal = [m for m in tb if kinds[m]['seal']]
            P(f'  cross-kind ({LV}, die level): sealing designations {len(sl)}, also on a seal {len(on_seal)} ({len(on_seal)/len(sl):.2f}), also on a tablet {len(on_tab)};'
              f' tablet designations {len(tb)}, also on a seal {len(tb_seal)} ({len(tb_seal)/len(tb):.2f}); examples sealing&seal {[fmt(m) for m in on_seal[:8]]}; tablet&seal {[fmt(m) for m in tb_seal[:10]]}')
            # same share for a random seal designation: chance that a seal designation of matched length occurs on another seal
            R[f'{LV}_cross'] = dict(sealing=len(sl), sealing_seal=len(on_seal), tab=len(tb), tab_seal=len(tb_seal))
for filt in ['strict']:
    I = im77_docs(filt)
    P(f'#### IM77 {filt} (act; Lothal / Kalibangan sealings only, since IM77 sealing at Harappa = moulded tablet)')
    Is = [d for d in I if not (d['kind'] == 'sealing' and d['site'] not in ('Lothal', 'Kalibangan'))]
    for k in [('seal',), ('sealing',), ('tablet_m',), ('tablet_c',)]:
        R[f'im77_{"+".join(k)}'] = table(f'IM77 {"+".join(k)}', Is, set(k), 8)
    Ih = [d for d in I if d['kind'] == 'sealing' and d['site'] == 'Harappa']
    R['im77_harappa_sealing'] = table('IM77 Harappa "sealing" (= moulded tablets)', Ih, {'sealing'}, 8)
json.dump(R, open(DARK + 'loop76_c2.json', 'w'), indent=1, default=str)
save('loop76_c2')
