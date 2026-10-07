"""LA-84 common loader: sign-groups with SigLA wear classes, strata and a familiarity index.

Classes (corpus_ra.json, words with >= 2 signs, documents SigLA covers):
  READ  status 'read', no flag except 'cont'
  WORN  only 'worn' (SigLA stipple >= 0.20 of a sign box; physical surface damage measured from the drawing)
  VAR   SigLA reads another sign ('variant') and no edge / part / bridge flag  -- held out of every fit
Edge-truncated groups (edgeL, edgeR, part, bridge) are left out: truncation changes familiarity mechanically.
Familiarity of a token in document d: the same sign tuple occurs in another document (exact), or only a
same-length tuple differing in one sign occurs in another document (near).
"""
import json, os, collections, hashlib

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, '..', 'data')
CK = os.path.join(DATA, 'la84_ckpt')
os.makedirs(CK, exist_ok=True)
SITE2G = {'Haghia Triada': 'HT', 'Khania': 'KH', 'Zakros': 'ZA', 'Phaistos': 'PH', 'Knossos': 'KN'}
EDGE = {'edgeL', 'edgeR', 'part', 'bridge', 'erased', 'illegible'}


def norm(s):
    return s.replace('₂', '2').replace('₃', '3')


def load(path=None):
    C = json.load(open(path or os.path.join(DATA, 'corpus_ra.json')))
    lex = collections.defaultdict(set)          # tuple -> docs
    toks = []
    for d in C:
        for t in d['tokens']:
            if t['t'] != 'word' or t.get('st') not in ('read', 'damaged'):
                continue
            s = tuple(norm(x) for x in t['s'])
            lex[s].add(d['id'])
            if len(s) < 2 or not d.get('sigla'):
                continue
            fl = set(t.get('fl', [])) - {'cont'}
            if fl & EDGE:
                cls = None
            elif t['st'] == 'read' and not fl:
                cls = 'READ'
            elif fl == {'worn'}:
                cls = 'WORN'
            elif 'variant' in fl:
                cls = 'VAR'
            else:
                cls = 'OTHER'
            if cls is None:
                continue
            g = 'HT' if d['site'] == 'Haghia Triada' else 'X'
            sup = 'T' if (d['support'] or '').lower() == 'tablet' else 'O'
            L = min(len(s), 4)
            toks.append(dict(doc=d['id'], s=s, cls=cls, stratum=(g, sup, L), site=SITE2G.get(d['site'], 'OTH')))
    return toks, lex


class Lex:
    def __init__(self, lex):
        self.lex = lex
        self.nb = collections.defaultdict(list)     # (L, i, masked) -> [(sign, tuple)]
        for w in lex:
            for i in range(len(w)):
                self.nb[(len(w), i, w[:i] + ('_',) + w[i + 1:])].append((w[i], w))
        sc = collections.Counter()
        for w, ds in lex.items():
            for x in w:
                sc[x] += len(ds)
        self.signs = list(sc)
        self.sfreq = [sc[x] for x in self.signs]

    def count(self, w, doc):
        ds = self.lex.get(w)
        if not ds:
            return 0
        return len(ds) - (doc in ds)

    def near(self, w, doc):
        for i in range(len(w)):
            for x, v in self.nb.get((len(w), i, w[:i] + ('_',) + w[i + 1:]), ()):
                if v != w and self.count(v, doc) > 0:
                    return True
        return False

    def stats(self, items):
        """items: list of (tuple, doc). Returns exact rate, near-only rate, mean log1p count."""
        import math
        n = len(items)
        ex = nr = lc = 0.0
        for w, doc in items:
            c = self.count(w, doc)
            if c > 0:
                ex += 1
                lc += math.log1p(c)
            elif self.near(w, doc):
                nr += 1
        return (ex / n, nr / n, lc / n)


def sha_file(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()
