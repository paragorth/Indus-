"""Aggregate loop-1 dark-machine cycles into loop1_final.txt with ready-to-paste STRATEGIES rows."""
import json,glob,re,collections,os
os.chdir('/home/user/Indus-')
out=[]
rows=[]
for cyc in (1,2,3,4):
    f=f'data/derived/dark/loop1_cycle{cyc}.txt'; fc=f'data/derived/dark/loop1_cycle{cyc}_control.txt'
    if not os.path.exists(f): out.append(f'cycle {cyc}: not finished'); continue
    L=open(f).read().splitlines(); Lc=open(fc).read().splitlines() if os.path.exists(fc) else []
    hdr=L[0]; fams=re.search(r'families (\S+)',hdr).group(1); seed=re.search(r'seed (\d+)',hdr).group(1)
    g=lambda LL,pat: next((l for l in LL if pat in l),'')
    bonf=g(L,'Bonferroni survivors'); stA=g(L,'stage A'); full=g(L,'full survivors')
    bonfc=g(Lc,'Bonferroni survivors'); stAc=g(Lc,'stage A'); fullc=g(Lc,'full survivors')
    fulls=[l for l in L if l.startswith('FULL ')]; parts=[l for l in L if l.startswith('part ')]
    fullsc=[l for l in Lc if l.startswith('FULL ')]
    out.append(f'=== cycle {cyc} (seed {seed}, families {fams}) ===')
    out+=[bonf,stA,full,'control: '+(bonfc or 'n/a'),'control: '+(stAc or 'n/a'),'control: '+(fullc or 'n/a')]
    out.append(f'FULL survivors ({len(fulls)}):'); out+=['  '+l[:400] for l in fulls]
    out.append(f'partial (corrected, beyond type/length, but failing perm/held-out/levels) {len(parts)}; first 8:'); out+=['  '+l[:300] for l in parts[:8]]
    out.append(g(L,'rediscoveries by'))
    nb=re.search(r'Bonferroni survivors (\d+); BH-FDR\(q=0.05\) survivors (\d+)',bonf); nbc=re.search(r'Bonferroni survivors (\d+); BH-FDR\(q=0.05\) survivors (\d+)',bonfc) if bonfc else None
    na=re.search(r'-> (\d+) are type/length rediscoveries.*; (\d+) carry',stA); nf=re.search(r'levels train and held-out\): (\d+)',full)
    nfc=re.search(r'levels train and held-out\): (\d+)',fullc) if fullc else None
    verdict='no survivor beyond the frame' if nf and nf.group(1)=='0' else f'{nf.group(1) if nf else "?"} survivor(s) to check against GRAMMAR, see loop1_cycle{cyc}.txt'
    rows.append(f"| S-DARK-1.{cyc} | Arrow-in-the-dark, loop 1 cycle {cyc}: 3,000 random (reading x outside fact) hypotheses, reading families {fams}, seed {seed}; statistic I(reading; fact | site), G-test screen, Bonferroni (alpha 1.7e-5) and BH-FDR (q 0.05) over all 3,000 arrows; survivors filtered by a permutation within site x type x length bin (frame-rule rediscoveries), then within-site permutation (2,000), held-out sites (all but Mohenjo-daro and Harappa, perm p<0.01) and seq_raw/seq_strong/seq_all. Control: same run with facts scrambled within site | Bonferroni {nb.group(1) if nb else '?'}, BH {nb.group(2) if nb else '?'}; type/length rediscoveries {na.group(1) if na else '?'}, beyond type+length {na.group(2) if na else '?'}, full survivors {nf.group(1) if nf else '?'}. Scrambled control: Bonferroni {nbc.group(1) if nbc else '?'}, BH {nbc.group(2) if nbc else '?'}, full {nfc.group(1) if nfc else '?'} | {verdict} |")
out.append('\n=== STRATEGIES.md rows (paste before ## Summary) ===')
out+=rows
open('data/derived/dark/loop1_final.txt','w').write('\n'.join(out)+'\n')
print('\n'.join(out))
