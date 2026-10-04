"""LA-31 'the winds carried the words': shared sites, paths and per-site data layers.

No readings or sound values are used: only sign identities, word types, logograms, numbers and
find-sites. Coordinates are approximate (site gazetteers), sufficient at the 1-5 km scale.
"""
import os, sys, json, re, collections, math
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
DATA = os.path.join(HERE, '..', 'data')
OUT = os.path.join(DATA, 'la31'); CKPT = os.path.join(DATA, 'la31_ckpt')
SCR = os.environ.get('LA31_SCRATCH', '/tmp/claude-0/-home-user-Indus-/874df4c7-80d6-5f08-b42c-eea96a214079/scratchpad/la31')
for p in (OUT, CKPT):
    os.makedirs(p, exist_ok=True)

# code: (corpus site name, lat, lon). Linear A find-sites with >= 5 syllabic signs and a known location.
SITES = {
    'HT': ('Haghia Triada', 35.059, 24.792), 'PH': ('Phaistos', 35.051, 24.814),
    'KH': ('Khania', 35.517, 24.018), 'KN': ('Knossos', 35.298, 25.163),
    'ZA': ('Zakros', 35.098, 26.261), 'PK': ('Palaikastro', 35.195, 26.276),
    'MA': ('Malia', 35.293, 25.492), 'IO': ('Iouktas', 35.226, 25.116),
    'AR': ('Arkhalkhori', 35.130, 25.270), 'PE': ('Petras', 35.205, 26.112),
    'SY': ('Syme', 35.050, 25.470), 'TY': ('Tylissos', 35.300, 25.020),
    'GO': ('Gournia', 35.107, 25.792), 'PY': ('Pyrgos', 35.007, 25.583),
    'AP': ('Apodoulou', 35.167, 24.661), 'KO': ('Kophinas', 34.965, 25.115),
    'TL': ('Troullos', 35.240, 25.160), 'PL': ('Platanos', 35.030, 24.960),
    'PS': ('Psykhro', 35.163, 25.453), 'VR': ('Vrysinas', 35.298, 24.456),
    'PO': ('Poros Herakleiou', 35.340, 25.155), 'PR': ('Prassa', 35.315, 25.205),
    'SK': ('Skoteino Cave', 35.307, 25.395),
    'TH': ('Thera', 36.352, 25.404), 'KE': ('Kea', 37.660, 24.330),
    'ML': ('Milos', 36.755, 24.513), 'MI': ('Miletos', 37.530, 27.280),
    'SA': ('Samothrace', 40.475, 25.530), 'TR': ('Troy', 39.957, 26.239),
}
CRETE = [c for c in SITES if SITES[c][1] < 35.6]
NAME2CODE = {v[0]: k for k, v in SITES.items()}


def hav(a, b):
    la1, lo1, la2, lo2 = map(math.radians, (a[0], a[1], b[0], b[1]))
    h = math.sin((la2 - la1) / 2) ** 2 + math.cos(la1) * math.cos(la2) * math.sin((lo2 - lo1) / 2) ** 2
    return 2 * 6371 * math.asin(math.sqrt(h))


def euclid(codes):
    return np.array([[hav(SITES[a][1:], SITES[b][1:]) for b in codes] for a in codes])


def _logo(s):
    return bool(re.fullmatch(r'\*[4-9]\d\d.*', s)) or s in ('VS', 'VAS') or bool(re.search(r'[a-z]', s))


SUPPORT = {'tablet': 'tablet', 'lames (short thin tablet)': 'tablet', '3-sided bar': 'tablet',
           '4-sided bar': 'tablet', 'nodule': 'nodule', 'sealing': 'nodule', 'label': 'nodule',
           'roundel': 'roundel', 'clay vessel': 'vessel', 'inked inscription': 'vessel',
           'graffito': 'vessel'}


def load_docs():
    """One record per document at an analysed site: word types (2+ syllabic signs), syllabic
    sign tokens, logogram tokens, and entry-structure counts."""
    from la15_common import load_la
    keep = {d['id'] for d in load_la()}          # drops joins whose parts are also listed
    C = json.load(open(os.path.join(DATA, 'corpus.json')))
    docs = []
    for d in C:
        if d['site'] not in NAME2CODE or d['id'] not in keep:
            continue
        words, signs, logos = [], [], []
        nnum = nfrac = nword = nline = nwn = 0
        toks = d['tokens']
        for k, t in enumerate(toks):
            if t['t'] == 'word':
                ss = t['s']
                syl = [s for s in ss if not _logo(s)]
                logos += [s for s in ss if _logo(s)]
                signs += syl
                if len(syl) >= 2 and len(syl) == len(ss):
                    words.append('-'.join(ss))
                nword += 1
                if k + 1 < len(toks) and toks[k + 1]['t'] == 'num':
                    nwn += 1
            elif t['t'] == 'logo':
                logos.append(t['v'])
            elif t['t'] == 'num':
                nnum += 1; nfrac += bool(t.get('frac'))
            elif t['t'] == 'nl':
                nline += 1
        docs.append(dict(id=d['id'], site=NAME2CODE[d['site']], support=SUPPORT.get(d['support'].lower(), 'object'), words=words,
                         signs=signs, logos=logos,
                         feat=np.array([nword, nnum, nfrac, len(logos), nline + 1, nwn,
                                        sum(len(w.split('-')) for w in words), len(words)], float)))
    return docs
