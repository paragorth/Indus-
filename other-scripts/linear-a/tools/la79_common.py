"""LA-79 'counterfactual excavation histories'.

la78 showed that, for Linear A, the order of discovery is a regime break: what fits the
published past best predicts later finds worst, and random-document CV rewards that.
la79 re-runs earlier Linear A claims (and thousands of same-form decoy claims) under every
counterfactual excavation history: each subset of six site groups plays 'what was dug
first' (train), the rest plays 'what was found later' (test), plus the three real
publication cuts (1950/1976/1988) and random-document splits (the shuffled-history control).
Publication years: la78_common.pub_year (GORILA introductions, concordance, source URLs).
Only sign identities, positions, numbers, fraction letters, logograms and sites are used.
"""
import json, os, sys, re, collections, hashlib, itertools
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, '..', 'data')
CK = os.path.join(DATA, 'la79_ckpt')
os.makedirs(CK, exist_ok=True)
sys.path.insert(0, HERE)
import la78_common as L78  # noqa: E402
from la15_common import _logo  # noqa: E402

GROUPS = ['HT', 'KH', 'ZA', 'PH', 'KN', 'OTH']
SITE2G = {'Haghia Triada': 'HT', 'Khania': 'KH', 'Zakros': 'ZA', 'Phaistos': 'PH', 'Knossos': 'KN'}


def norm(s):
    return s.replace('₂', '2').replace('₃', '3')


def load(status=('read', 'damaged')):
    C = json.load(open(os.path.join(DATA, 'corpus_ra.json')))
    meta = {d['id']: d for d in L78.load()}
    docs = []
    for d in C:
        m = meta[d['id']]
        toks = [t for t in d['tokens'] if t['t'] not in ('nl', 'div') and t.get('st') in status]
        words, logos, fracs = [], [], []
        cur = None
        for i, t in enumerate(toks):
            if t['t'] == 'word':
                syl = tuple(norm(s) for s in t['s'] if not _logo(s))
                if syl:
                    nxt = toks[i + 1]['t'] if i + 1 < len(toks) else 'end'
                    words.append(dict(s=syl, read=t['st'] == 'read', nxt=nxt))
            elif t['t'] == 'logo':
                v = t.get('v', '').split('+')[0].strip('[]*') if t.get('v') else ''
                v = t.get('v', '').split('+')[0].rstrip('[')
                logos.append(v)
                cur = v
            elif t['t'] == 'num':
                fr = t.get('frac') or []
                if fr:
                    fracs.append((tuple(fr), cur))
        sup = (d['support'] or '?').lower()
        docs.append(dict(id=d['id'], site=d['site'] or '?', g=SITE2G.get(d['site'], 'OTH'), sup=sup,
                         year=m['year'], words=words, logos=logos, fracs=fracs))
    return docs


def splits(docs, n_random=40, seed=0):
    """List of (name, kind, train_idx, test_idx)."""
    rng = np.random.default_rng(seed)
    g = np.array([d['g'] for d in docs])
    y = np.array([d['year'] for d in docs])
    out = []
    for cut in (1950, 1976, 1988):
        out.append(('T%d' % cut, 'time', np.where(y <= cut)[0], np.where(y > cut)[0]))
    for k in range(1, len(GROUPS)):
        for tr in itertools.combinations(GROUPS, k):
            trm = np.isin(g, tr)
            out.append(('S:' + '+'.join(tr), 'site', np.where(trm)[0], np.where(~trm)[0]))
    # shuffled history: random document splits of the same sizes as the site splits / time cuts
    sizes = [len(s[3]) for s in out]
    n = len(docs)
    for r in range(n_random):
        m = sizes[r % len(sizes)]
        p = rng.permutation(n)
        out.append(('R%d' % r, 'rand', np.sort(p[m:]), np.sort(p[:m])))
    return out


def sha(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()
