"""report for v12 cycle 3 parts A-C (reads data/results/v12/c3abc_*.json)."""
import json, os, math, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v12_lib as V
names = V.VOY + V.POS + V.NEG + ['Gloss-pair']
def ms(xs):
    m = sum(xs) / len(xs); s = math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1)); return m, s
print('C. second-order info I(x_t; x_t-2 | x_t-1) and lag MI, excess over 20 line shuffles (millibits), z')
for n in names:
    r = json.load(open(os.path.join(V.CK, f'c3abc_{n}.json')))
    row = []
    for f in ('first', 'gallows', 'vowel', 'last'):
        for key in ('mi1', 'mi2', 'cmi2'):
            m, s = ms([x[f'{key}_{f}'] for x in r['nulls']]); o = r['obs'][f'{key}_{f}']
            row.append(f"{key}_{f[:3]} {1000*(o-m):6.1f}({(o-m)/max(s,1e-9):5.1f})")
    print(f"{n:13s} " + ' '.join(row))
print('\nA. Hamming-1 neighbours by glyph pair: obs / null mean (z), lag1 and lag2, top 12 by lag-1 count (Voynich ZL, IT)')
for n in ['Voynich-ZL', 'Voynich-IT']:
    r = json.load(open(os.path.join(V.CK, f'c3abc_{n}.json')))
    for lag in (1, 2):
        o = r['obs'][f'ham{lag}']; keys = sorted(o, key=lambda k: -o[k])[:14]
        tot_o = sum(o.values()); tn = [sum(x[f'ham{lag}'].values()) for x in r['nulls']]; m, s = ms(tn)
        print(n, f'lag{lag} total {tot_o} vs {m:.0f} (z {(tot_o-m)/s:.1f})')
        out = []
        for k in keys:
            m, s = ms([x[f'ham{lag}'].get(k, 0) for x in r['nulls']])
            out.append(f"{k} {o[k]}/{m:.0f}({(o[k]-m)/max(s,0.5):.1f})")
        print('   ', ', '.join(out))
print('\nB. phase: excess over shuffle per junction j (word j -> j+1), gallows concordance / Hamming-1 rate / a-o concordance')
for n in names:
    r = json.load(open(os.path.join(V.CK, f'c3abc_{n}.json')))
    line = []
    for f in ('gconc', 'ham', 'vconc'):
        ex = []
        for j in range(8):
            k = f'{f}{j}'
            if k not in r['obs']['phase']: continue
            m, s = ms([x['phase'].get(k, 0) for x in r['nulls']]); ex.append((r['obs']['phase'][k] - m, s))
        if len(ex) < 4: continue
        odd = [e for i, (e, s) in enumerate(ex) if i % 2 == 0]; even = [e for i, (e, s) in enumerate(ex) if i % 2 == 1]
        sd = math.sqrt(sum(s * s for e, s in ex)) / len(ex) * 2
        contrast = sum(odd) / len(odd) - sum(even) / len(even)
        line.append(f"{f} " + ' '.join(f'{1000*e:5.0f}' for e, s in ex) + f" | odd-even {1000*contrast:5.0f} (z {contrast/max(sd,1e-9):4.1f})")
    print(f"{n:13s} " + ' || '.join(line))
print('\nD-prelim. word-type entropy odd / even line positions (obs vs null mean)')
for n in names:
    r = json.load(open(os.path.join(V.CK, f'c3abc_{n}.json')))
    o = r['obs']['par_ent']; m0 = ms([x['par_ent'][0] for x in r['nulls']]); m1 = ms([x['par_ent'][1] for x in r['nulls']])
    print(f"{n:13s} H(odd) {o[0]:.3f} vs {m0[0]:.3f}  H(even) {o[1]:.3f} vs {m1[0]:.3f}  diff {o[0]-o[1]:+.3f} (null {m0[0]-m1[0]:+.3f})")
