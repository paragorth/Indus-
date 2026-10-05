"""v50 known-effect control set: REAL with the line-initial words permuted among the lines of each page (kills the
v6 left-margin first-glyph succession and any vertical content of line-initial words, keeps everything else).
Writes REAL_FW and REAL_FW_LS1-4."""
import os, sys, json, random, zlib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v50_build as B
import v50_lib as L

pages = json.load(open(os.path.join(L.SETS, 'REAL.json')))
rng = random.Random(4242); out = []
for f, p in pages:
    fw = [ws[0] for ps, ws in p]; rng.shuffle(fw)
    out.append((f, [(ps, [fw[i]] + ws[1:]) for i, (ps, ws) in enumerate(p)]))
B.write_set('REAL_FW', out)
for k in (1, 2, 3, 4): B.write_set(f'REAL_FW_LS{k}', B.line_shuffle(out, 31 * k + 5))
print('ok')
