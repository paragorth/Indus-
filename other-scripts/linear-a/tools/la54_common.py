#!/usr/bin/env python3
"""LA-54 shared code: READ WHAT WAS NOT WRITTEN.

Many Linear A documents list words and numbers with no commodity sign at all: the clerk left out what
everyone in the room knew. We invert the problem. Every document that DOES show its commodity becomes a
training example with its commodity signs deleted; thousands of random classifiers on random feature sets
(number sizes, fraction signs, rounding, words, site, support, layout) are scored on held-out documents
with the commodity masked, survivors are pooled, and the pooled model fills in the commodity of every
document that never wrote one.

doc = {'id','site','support','label', 'nums': [(value, [frac signs])], 'words': [str], 'lines': int,
       'ntok_lines': [int], 'numfirst': float}
Commodity signs are removed from every document before features are built. No sound values are used:
word identities are opaque strings.
Controls: Linear B (DAMOS) and Ur III (CDLI) documents with their commodity signs/words deleted.
"""
import json, os, re, sys, math, random, hashlib, collections
import os as _os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    _os.environ.setdefault(_v, "1")
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, '..', 'data')
CK = os.path.join(D, 'la54_ckpt')
os.makedirs(CK, exist_ok=True)
sys.path.insert(0, HERE)


def seed(name):
    return int(hashlib.sha256(name.encode()).hexdigest()[:8], 16)


# ------------------------------------------------------------------ Linear A
LA_CLASSES = ['GRA', 'VIN', 'OLE', 'OLIV', 'CYP', 'VIR', 'NI', 'OTH']


def la_base(v):
    s = re.sub(r"[\[\]'\"?]", '', v).lstrip('*')
    b = s.split('+')[0]
    b = re.sub(r'(?<=[A-Z])[a-z]+$', '', b)
    if b in ('GRA', 'VIN', 'OLE', 'OLIV', 'CYP', 'VIR', 'NI'):
        return b
    return 'OTH'


def dominant(bases, classes):
    if not bases:
        return None
    c = collections.Counter(bases)
    top = max(c.values())
    for b in bases:  # first written among the most frequent
        if c[b] == top:
            return b


def la_docs():
    C = json.load(open(os.path.join(D, 'corpus.json')))
    out = []
    for ins in C:
        nums, words, bases = [], [], []
        lines = [[]]
        for t in ins['tokens']:
            if t['t'] == 'nl':
                lines.append([])
            elif t['t'] == 'word':
                w = '-'.join(t['s'])
                words.append(w); lines[-1].append('W')
            elif t['t'] == 'logo':
                bases.append(la_base(t['v']))  # deleted from the document
            elif t['t'] == 'num':
                nums.append((float(t['v']), list(t['frac']))); lines[-1].append('N')
        if not nums:
            continue
        lines = [l for l in lines if l]
        out.append(mk(ins['id'], ins['site'] or '?', ins['support'], dominant(bases, LA_CLASSES), nums, words, lines,
                      bases=bases))
    return out


def mk(i, site, support, label, nums, words, lines, bases=None):
    nf = [1.0 if l and l[0] == 'N' else 0.0 for l in lines]
    return {'id': i, 'site': site, 'support': support, 'label': label, 'nums': nums, 'words': words,
            'lines': len(lines), 'ntok_lines': [len(l) for l in lines], 'numfirst': float(np.mean(nf)) if nf else 0.0,
            'bases': bases or []}


# ------------------------------------------------------------------ Linear B
LB_MAP = {'VIR': 'PEOPLE', 'MUL': 'PEOPLE', 'GRA': 'GRAIN', 'HORD': 'GRAIN', 'FAR': 'GRAIN', 'LANA': 'WOOL',
          'OLE': 'OIL', 'VIN': 'WINE', 'OVIS': 'ANIMAL', 'CAP': 'ANIMAL', 'SUS': 'ANIMAL', 'BOS': 'ANIMAL',
          'EQU': 'ANIMAL', 'TELA': 'CLOTH', 'TUN': 'CLOTH', 'AES': 'METAL', 'AUR': 'METAL', 'NI': 'FIG',
          'OLIV': 'OLIVE', 'AROM': 'SPICE', 'CROC': 'SPICE', 'CYP': 'SPICE', '*146': 'CLOTH'}


def lb_base(v):
    b = v.split(';')[0].split('+')[0]
    b = re.sub(r'^(OVIS|CAP|SUS|BOS)[mf]?$', r'\1', b)
    return LB_MAP.get(b, 'OTH')


def lb_docs():
    import la53_common as C53
    out = []
    for d in C53.lb_docs():
        nums, words, bases, lines = [], [], [], []
        for l in d['lines']:
            L = []
            for t in l:
                if t[0] == 'L':
                    bases.append(lb_base(t[1]))
                elif t[0] == 'W':
                    words.append(t[1]); L.append('W')
                elif t[0] == 'N':
                    v = t[1]; fr = []
                    if abs(v - round(v)) > 1e-6:
                        fr = ['f%d' % (int(round((v - math.floor(v)) * 72)) % 12)]  # fractional part as 'sign'
                    nums.append((float(math.floor(v)), fr)); L.append('N')
            if L: lines.append(L)
        if nums and bases:
            out.append(mk(d['id'], d['site'], d['support'], dominant(bases, None), nums, words, lines, bases))
    return out


# ------------------------------------------------------------------ Ur III
UR_COMM = [('GRAIN', ('sze', 'zi3', 'dabin', 'zi3-sig15', 'esza', 'ziz2', 'gig')),
           ('BEER_BREAD', ('kasz', 'ninda', 'kasz-saga', 'kasz-du')),
           ('SHEEP_GOAT', ('udu', 'sila4', 'masz2', 'u8', 'ud5', 'gukkal', 'kir11', 'masz2-gal', 'udu-niga')),
           ('CATTLE', ('gu4', 'ab2', 'amar', 'gu4-niga', 'ansze')),
           ('OIL', ('i3', 'i3-gesz', 'i3-nun', 'i3-szah2')),
           ('SILVER', ('ku3-babbar', 'ku3', 'ku3-sig17')),
           ('TEXTILE', ('siki', 'tug2', 'gada')),
           ('LABOUR', ('gurusz', 'geme2', 'erin2', 'dumu-gi7'))]
UR_ROOT = {}
for c, ws in UR_COMM:
    for w in ws:
        UR_ROOT[w] = c


def ur_core(w):
    c = re.sub(r'\{[^}]*\}', '', w)
    c = re.sub(r'-(ra|sze3|ke4|ka|ta|a|bi|e|kam|ak|hi-a)$', '', c)
    return c


def ur_docs():
    import la53_common as C53
    out = []
    for d in C53.ur3_docs():
        nums, words, bases, lines = [], [], [], []
        for l in d['lines']:
            L = []
            for t in l:
                if t[0] == 'W':
                    c = ur_core(t[1])
                    if c in UR_ROOT or c.split('-')[0] in UR_ROOT and c.split('-')[0] in ('udu', 'gu4', 'sze', 'kasz', 'i3', 'siki', 'tug2', 'ku3'):
                        bases.append(UR_ROOT.get(c, UR_ROOT.get(c.split('-')[0])))  # commodity word deleted
                        continue
                    words.append(t[1]); L.append('W')
                elif t[0] == 'N':
                    v = t[1]; fr = []
                    if abs(v - round(v)) > 1e-6:
                        fr = ['f%d' % (int(round((v - math.floor(v)) * 60)) % 12)]
                    nums.append((float(math.floor(v)), fr)); L.append('N')
            if L: lines.append(L)
        if nums and bases:
            out.append(mk(d['id'], d['site'], d['support'], dominant(bases, None), nums, words, lines, bases))
    return out


def sample_like(docs, n, rng, min_class=5):
    lab = [d for d in docs if d['label']]
    rng.shuffle(lab)
    s = lab[:n]
    c = collections.Counter(d['label'] for d in s)
    for d in s:
        if c[d['label']] < min_class:
            d['label'] = 'OTH'
    return s


# ------------------------------------------------------------------ features
GROUPS = ['NUM', 'FRAC', 'ROUND', 'WORD', 'SITE', 'SUPP', 'LAYOUT']


class Featurizer:
    def __init__(self, docs, nword=80, nsite=8, nsupp=5):
        wc = collections.Counter(w for d in docs for w in set(d['words']))
        self.vocab = [w for w, k in wc.most_common(nword) if k >= 3]
        fc = collections.Counter(f for d in docs for v, fr in d['nums'] for f in fr)
        self.fracs = [f for f, k in fc.most_common(16) if k >= 2]
        self.sites = [s for s, _ in collections.Counter(d['site'] for d in docs).most_common(nsite)]
        self.supps = [s for s, _ in collections.Counter(d['support'] for d in docs).most_common(nsupp)]
        self.names, self.group = [], []
        for g, names in self._spec():
            for n in names:
                self.names.append(n); self.group.append(g)
        self.group = np.array(self.group)

    def _spec(self):
        return [('NUM', ['n_num', 'lmean', 'lmed', 'lmax', 'lmin', 'lsum', 'lsd', 'sh1', 'sh10', 'sh50', 'sh100', 'sh_lt5']),
                ('FRAC', ['sh_frac', 'n_fracsign'] + ['fr_' + f for f in self.fracs]),
                ('ROUND', ['r5', 'r10', 'r100', 'last1', 'odd']),
                ('WORD', ['n_words', 'wlen', 'sh_single', 'w_per_num'] + ['w_' + w for w in self.vocab]),
                ('SITE', ['s_' + s for s in self.sites] + ['s_other']),
                ('SUPP', ['p_' + s for s in self.supps] + ['p_other']),
                ('LAYOUT', ['n_lines', 'tok_line', 'numfirst', 'maxline'])]

    def row(self, d):
        v = np.array([x for x, _ in d['nums']], float)
        lv = np.log1p(v)
        big = v[v >= 10]
        nfr = [fr for _, fr in d['nums']]
        allf = [f for fr in nfr for f in fr]
        ws = d['words']
        r = [len(v), lv.mean(), np.median(lv), lv.max(), lv.min(), np.log1p(v.sum()), lv.std(), (v == 1).mean(),
             (v >= 10).mean(), (v >= 50).mean(), (v >= 100).mean(), (v < 5).mean()]
        r += [np.mean([1.0 if fr else 0.0 for fr in nfr]), len(set(allf))] + [allf.count(f) / len(v) for f in self.fracs]
        r += [(big % 5 == 0).mean() if len(big) else 0.0, (big % 10 == 0).mean() if len(big) else 0.0,
              (v[v >= 100] % 100 == 0).mean() if (v >= 100).any() else 0.0,
              (v % 10 == 1).mean(), (v % 2 == 1).mean()]
        sw = set(ws)
        r += [len(ws), np.mean([w.count('-') + 1 for w in ws]) if ws else 0.0,
              np.mean([1.0 if '-' not in w else 0.0 for w in ws]) if ws else 0.0, len(ws) / len(v)]
        r += [1.0 if w in sw else 0.0 for w in self.vocab]
        r += [1.0 if d['site'] == s else 0.0 for s in self.sites] + [0.0 if d['site'] in self.sites else 1.0]
        r += [1.0 if d['support'] == s else 0.0 for s in self.supps] + [0.0 if d['support'] in self.supps else 1.0]
        r += [d['lines'], np.mean(d['ntok_lines']) if d['ntok_lines'] else 0.0, d['numfirst'],
              max(d['ntok_lines']) if d['ntok_lines'] else 0.0]
        return r

    def X(self, docs):
        return np.nan_to_num(np.array([self.row(d) for d in docs], float))


# ------------------------------------------------------------------ nulls / plants
def shuffle_numbers(docs, rng):
    """Numbers shuffled across documents: every doc keeps its count of numbers, values (with their fraction
    signs) come from the pooled corpus."""
    pool = [x for d in docs for x in d['nums']]
    rng.shuffle(pool)
    out, k = [], 0
    for d in docs:
        n = len(d['nums'])
        out.append(dict(d, nums=pool[k:k + n])); k += n
    return out


def plant(docs, rng, n_plant=25, factor=(1.6, 2.4), wordsig=False):
    """Planted implicit commodity PLANT: n_plant labelled documents (random classes) relabelled; their
    integers rescaled by a document-level factor (a different measuring habit). Nothing else marks them."""
    lab = [i for i, d in enumerate(docs) if d['label']]
    pick = set(rng.sample(lab, n_plant))
    out = []
    for i, d in enumerate(docs):
        if i in pick:
            f = rng.uniform(*factor)
            nums = [(float(max(1, round(v * f))) if v > 0 else v, fr) for v, fr in d['nums']]
            words = list(d['words'])
            if wordsig and rng.random() < 0.5:
                words = words + ['PLANTWORD']
            out.append(dict(d, label='PLANT', nums=nums, words=words))
        else:
            out.append(d)
    return out
