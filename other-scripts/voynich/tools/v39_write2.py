"""v39 cycle 2 rows."""
import v39_lib as L, v39_report as R

FN = 'v39_cycle2.txt'
D = R.all_results('c2_')
for k in D: D[k]['pLI'] = D[k]['pL'] + D[k]['pI']
THR = ['None', '0.9', '0.8', '0.7', '0.6', '0.5']
KM = ['pLI', 'pG', 'arrow', 'gap', 'ttr', 'h21', 'auc']
NM = {'V': 'Voynich ZL', 'VI': 'Voynich IT2a', 'Pla': 'Latin+plant', 'Pde': 'German+plant', 'Pit': 'Italian+plant',
      'la': 'Latin', 'de': 'German', 'it': 'Italian'}


def esc(u):
    return u if u.isascii() and u.isprintable() else "'" + ({chr(0xE000 + i): f'#{i}' for i in range(5)}.get(u, '?'))


def sweep(n):
    out = []
    for t in THR:
        k = f'{n}|{t}'
        if k not in D: continue
        r = D[k]
        out.append(f"thr {t} ({r['nmerged']} merged): " + ', '.join(f"{m} {R.fmt(r[m], m)}" for m in KM))
    return f"{NM[n]}: " + ' ; '.join(out)


with open(f'{L.LOOPS}/{FN}', 'w') as f:
    f.write('# v39 cycle 2 - DATA-DRIVEN merges: union of all unit pairs whose v35 interchangeability index I (discovery half of the pages) >= threshold, battery on the HELD-OUT half (4 Oct 2026). Tools v39_cycle2.py, v39_write2.py; checkpoints data/v39_ckpt/c2_*.json\n')
    f.write('| row | method and control | result | verdict |\n|---|---|---|---|\n')

tops = []
for n in NM:
    k = f'{n}|None'
    if k in D:
        tp = D[k]['top'][:6]
        npl = sum(1 for v, a, b in D[k]['top'] if v >= 0.9)
        tops.append(f"{NM[n]}: " + ', '.join(f"{esc(a)}/{esc(b)} {v:.2f}" for v, a, b in tp) + f" (pairs >= 0.9 among top 12: {npl})")
L.row(FN, 'V-39.2.1', 'Index I for every unit pair (>= 30 tokens) on the discovery half; planted controls: Latin/German/Italian with 5 Voynich-matched free twin splits (planted twin of letter x shown as x/\'#i)',
      ' // '.join(tops),
      'CALIBRATED: in every planted language the 5 planted pairs are the top 5 (I 0.96-1.02) and no real letter pair reaches 0.9 (max 0.69-0.73; Italian rare j/x 0.82). The Voynich has 5 (ZL) / 7 (IT2a) pairs in the planted band, all inside the gallows / bench family plus ch/sh. Grade A')
for n in ('V', 'VI'):
    L.row(FN, f'V-39.2.2{"a" if n == "V" else "b"}', f'{NM[n]}: merge all pairs with I >= threshold (union), battery on the held-out half', sweep(n),
          'see 2.V')
for n in ('Pla', 'Pde', 'Pit'):
    L.row(FN, f'V-39.2.3-{n}', f'POSITIVE control {NM[n]}: same sweep (thr 0.9 merges exactly the planted pairs; lower thresholds also merge real letters)', sweep(n), 'see 2.V')
for n in ('la', 'de', 'it'):
    L.row(FN, f'V-39.2.4-{n}', f'NEGATIVE control {NM[n]} (no plant): same sweep', sweep(n), 'see 2.V')
L.row(FN, 'V-39.2.V', 'Cycle-2 verdict: does merging what the data say is interchangeable make the Voynich language-like?',
      'Planted controls at thr 0.9 (exactly the planted pairs): P(lang or conlang) +0.14 to +0.16 (Latin 0.39 -> 0.54, German 0.45 -> 0.59, Italian 0.27 -> 0.43), P(gen) -0.07 to -0.16, ttr back to the unplanted value. Voynich at thr 0.9: ZL 0.16 -> 0.15 (P(gen) 0.43 -> 0.47), IT2a 0.12 -> 0.11 (0.51 -> 0.59); at 0.8-0.6 ZL pLI 0.20-0.21 but P(gen) rises too (0.49-0.54), IT2a P(gen) 0.67-0.91; at 0.5 both collapse into the generator class (0.91-0.94). Arrow 0-1 (ZL), 3-11 (IT2a, base 7); gap ratio 0.93-1.01 throughout (languages 0.56-0.79). Caveat: index-driven merges of distinct letters can also raise P(lang) in languages (Latin thr 0.6 +0.12, German thr 0.5 +0.21), so P(lang) moves of +-0.1 are within method noise',
      'NO. With the data-driven merges the Voynich moves, if anywhere, TOWARD the generator class, while every planted language moves back to language by the same procedure. The twin structure is real (planted-band I) but its collapse does not uncover a language-like text. Grade A (negative), held out and in two transcriptions')
print(open(f'{L.LOOPS}/{FN}').read())
