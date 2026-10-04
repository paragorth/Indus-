"""v38 cycle 3 report: random-restart search (Voynich and Gerard) -> loops/v38_cycle3.txt"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v38_lib import CK, LOOPS

rows = []
for src, rid in [('voynich', 'V-38.22'), ('gerard', 'V-38.23')]:
    s = json.load(open(os.path.join(CK, 'c3_%s.json' % src)))['summary']
    rows.append((rid, 'RANDOM RESTARTS, %s: %d random hypotheses (random visual sub-space of 6 families incl. top-30 PCs of the embeddings, random weights, cosine/Euclidean; random text view: words in a random document-frequency band, glyph/letter n-grams, prefixes or suffixes, binary or tf-idf). Discovery on %d pages (half of the leaves), top 20 re-scored on the %d held-out pages; whole search rerun on %d within-stratum page permutations' % (
        src, s['NH'], s['nA'], s['nB'], s['NNULL']),
        'best discovery r %.3f (null mean %.3f, max %.3f; p %.2f); held-out mean r of top 20 %+.4f (null %+.4f +- %.4f; z %+.1f, p %.3f); families in top 20 %s; text views %s' % (
            s['bestA'], s['null_bestA'][0], s['null_bestA'][1], s['p_bestA'], s['meanB'], s['null_meanB'][0], s['null_meanB'][1], s['zB'], s['p_meanB'],
            s['fam_count'], s['txt_count']),
        ('search finds and replicates' if s['p_meanB'] < 0.05 else 'search finds nothing that replicates')))
with open(os.path.join(LOOPS, 'v38_cycle3.txt'), 'w') as fh:
    fh.write('# v38 cycle 3 - massive random guessing over feature choices, held-out pages\n| id | method and control | result | verdict |\n|---|---|---|---|\n')
    for r in rows:
        fh.write('| %s | %s | %s | %s |\n' % r)
print(open(os.path.join(LOOPS, 'v38_cycle3.txt')).read())
