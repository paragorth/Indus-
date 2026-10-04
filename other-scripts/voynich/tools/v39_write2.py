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
      ' || '.join(tops),
      'CALIBRATED: in every planted language the 5 planted pairs are the top 5 (I 0.96-1.02) and no real letter pair reaches 0.9 (max ~0.7). The Voynich has 4-5 pairs in the planted band (p/f, cth/cph, k/t, ch/sh, cth/ckh), then the gallows family at 0.8-0.84 and l/r 0.80. IT2a agrees. Grade A')
for n in ('V', 'VI'):
    L.row(FN, f'V-39.2.2{"a" if n == "V" else "b"}', f'{NM[n]}: merge all pairs with I >= threshold (union), battery on the held-out half', sweep(n),
          'see 2.V')
for n in ('Pla', 'Pde', 'Pit'):
    L.row(FN, f'V-39.2.3-{n}', f'POSITIVE control {NM[n]}: same sweep (thr 0.9 merges exactly the planted pairs; lower thresholds also merge real letters)', sweep(n), 'see 2.V')
for n in ('la', 'de', 'it'):
    L.row(FN, f'V-39.2.4-{n}', f'NEGATIVE control {NM[n]} (no plant): same sweep', sweep(n), 'see 2.V')
print(open(f'{L.LOOPS}/{FN}').read())
