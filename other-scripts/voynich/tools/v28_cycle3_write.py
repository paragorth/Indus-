"""write loops/v28_cycle3.txt from c3_sm.pkl, c3_pos.pkl (v28_cycle3.py) and the planted allograph runs."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v28_lib as X, v25_lib as L
sm = X.load('c3_sm.pkl'); pos = X.load('c3_pos.pkl')
rows = []
for k, (n, (o, top)) in enumerate(sm.items(), 1):
    rows.append((f'V-28.3.S{k}', f'SIZE-MATCHED: {n} restricted to its 23 most frequent units (= Voynich v25 inventory size); behaviour from the full set; image combo r averaged over fonts; null 3,000 label permutations (worst p over fonts)',
                 '; '.join(f'{m} r {r:+.2f} (p <= {p:.4f})' for m, (r, p) in o.items()) + f'. Units: {" ".join(top)}', ''))
for k, (n, (o, top, sk)) in enumerate(pos.items(), 1):
    rows.append((f'V-28.3.P{k}', f'POSITIONAL-VARIANT test on {n} ({sk}): Mantel on all pairs; minus known allograph pairs (medieval sets); within pairs that share word positions (overlap >= median) vs complementary pairs; dropping the k most allograph-like pairs (shape-similar x positionally complementary); null = label permutation on the same pair mask',
                 ' || '.join(f'{m}: {s}' for m, s in o) + f'. Most allograph-like pairs: {", ".join(top)}', ''))
hdr = ('# v28 cycle 3 - size-matched comparison and the positional-variant (allograph) explanation.\n'
       '| row | method and control | result | verdict |\n|---|---|---|---|')
L.write_rows(os.path.join(X.LOOPS, 'v28_cycle3.txt'), rows, hdr)
print(len(rows))
