"""v6 cycle 1: word-index grid. Do columns, diagonals, spirals or boustrophedon beat left-to-right rows?
Corpora: Voynich ZL3b and IT2a paragraphs (ltype P). Controls: Latin (Caesar+Descartes) verbose-encoded and written
into the SAME paragraph shapes along a known order: row (negative = plain prose), col, diag_dr, spiral (positives).
Output data/results/v6_cycle1.json"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vlib
from v6_gridlib import *

REPS = int(os.environ.get('REPS', 30))
out = {}
vz = voynich_paragraphs('ZL3b'); shapes = shapes_of(vz)
print('ZL paragraphs', len(vz), 'words', sum(map(sum, shapes)), 'lines', sum(map(len, shapes)))
corp = {'Voynich-ZL': (vz, units_voy), 'Voynich-IT': (voynich_paragraphs('IT2a'), units_voy)}
lat = encode_verbose(latin_stream())
need = sum(map(sum, shapes)); lat = (lat * 2)[:need]
for o in ('row', 'col', 'diag_dr', 'spiral'):
    corp['Latin-verbose written ' + o] = (build_control(shapes, lat, o), units_plain)

for name, (paras, U) in corp.items():
    F = make_features(paras, U)
    r = compare_orders(paras, F, reps=REPS); out[name] = r
    print('\n==', name)
    print('%-8s' % 'order' + ''.join('%22s' % f for f in ('junction', 'word', 'first-first', 'same-word', 'same-prefix2')))
    for o in ORDERS:
        row = '%-8s' % o
        for f in ('junction', 'word', 'first-first', 'same-word', 'same-prefix2'):
            spec = 'inrow' if o in ('row', 'boustro') else 'line'
            row += '  %+.4f(%5.1f)/%5.1f' % (r[o]['full'][f]['ex'], r[o]['full'][f]['z'], r[o][spec][f]['z'])
        print(row)
json.dump(out, open(os.path.join(vlib.RES, 'v6_cycle1.json'), 'w'), indent=0)
