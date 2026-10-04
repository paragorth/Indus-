"""v46: assemble per-page object counts (derived numbers only) -> data/derived/v46_page_counts.json
bio      : nymphs per page, counted by eye on Yale IIIF scans at 1000 px (best, lo, hi)
zodiac   : nymphs per panel; equal to the ZL3b label loci on every panel (checked by eye on f71r: 10 + 5 = 15)
pharma   : drawn plant fragments + jars per panel = ZL3b label loci (checked by eye: f88r 16 fragments + 3 jars
           vs 15 labels; f99r 31 fragments + 4 jars vs 34 labels); jar counts by eye where seen
stars    : stars per page from the ZL3b star comments (checked by eye: f103r 19 = 19); 8-point stars; dark stars
herbal   : blind automated leaf-blob count (tools/v46_leaves.py) + manual counts on a random 10-page sample
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from collections import Counter, defaultdict
HERE = os.path.dirname(os.path.abspath(__file__)); DATA = os.path.join(os.path.dirname(HERE), 'data')

BIO = {  # folio: (best, lo, hi), counted by eye
    'f75r': (14, 12, 16), 'f75v': (28, 24, 33), 'f76r': (0, 0, 0), 'f76v': (5, 5, 5), 'f77r': (4, 4, 4),
    'f77v': (7, 7, 7), 'f78r': (17, 15, 19), 'f78v': (10, 9, 10), 'f79r': (7, 7, 7), 'f79v': (4, 4, 5),
    'f80r': (14, 13, 15), 'f80v': (10, 9, 11), 'f81r': (13, 13, 13), 'f81v': (16, 16, 16), 'f82r': (14, 12, 16),
    'f82v': (8, 8, 8), 'f83r': (5, 5, 5), 'f83v': (4, 4, 4), 'f84r': (35, 30, 40), 'f84v': (17, 12, 22)}
HERB_MANUAL = {'f93v': 6, 'f15v': 4, 'f35r': 1, 'f13v': 20, 'f46r': 22, 'f8r': 2, 'f25r': 12, 'f49r': 2,
               'f41v': 6, 'f20r': 40, 'f9v': 24, 'f33v': 4}

def main(leaves_json):
    recs = json.load(open(os.path.join(DATA, 'derived', 'ZL3b_lines.json')))
    lab = Counter(r['folio'] for r in recs if r['ltype'] == 'L')
    illus = {}
    for r in recs: illus.setdefault(r['folio'], r['illus'])
    from v32_lib import load_q20
    st = defaultdict(Counter)
    for u in load_q20():
        if u['star']:
            st[u['f']]['stars'] += 1; st[u['f']]['p8'] += (u['star']['pts'] or 0) >= 8; st[u['f']]['dark'] += u['star']['dark']
    leaves = json.load(open(leaves_json))
    out = []
    for f, (b, lo, hi) in BIO.items():
        out.append(dict(id=f, sec='bio', n=b, lo=lo, hi=hi, src='eye'))
    for f in lab:
        if illus.get(f) == 'Z':
            out.append(dict(id=f, sec='zodiac', n=lab[f], lo=lab[f], hi=lab[f], src='labels=eye'))
        if illus.get(f) == 'P':
            out.append(dict(id=f, sec='pharma', n=lab[f], lo=lab[f] - 2, hi=lab[f] + 3, src='labels~eye'))
    for f, c in st.items():
        out.append(dict(id=f, sec='stars', n=c['stars'], lo=c['stars'], hi=c['stars'], n8=c['p8'], dark=c['dark'], src='ZL comments=eye'))
    for f, c in leaves.items():
        if illus.get(f) == 'H':
            out.append(dict(id=f, sec='herbal', n=c['leaves'], lo=None, hi=None, green_regions=c['green_regions'],
                            paint_blobs=c['paint_blobs'], manual=HERB_MANUAL.get(f), src='auto'))
    json.dump(out, open(os.path.join(DATA, 'derived', 'v46_page_counts.json'), 'w'), indent=0)
    print(Counter(o['sec'] for o in out))

if __name__ == '__main__':
    main(sys.argv[1])
