"""S-DARK-34: THE CHECKSUM HUNT. Is any element of a seal a deterministic function of the others?
Targets: first sign, last sign, closer identity, connective/marker, suffix, emblem class, material, boss class,
shape class. Predictors (sklearn): decision tree, histogram gradient boosting, k-nearest neighbours, trained on
Mohenjo-daro + Harappa seals, scored on (a) seals from every other site and (b) the 324 IM77-only texts of
S-DARK-27.3 (loop27_sets.json 'new', mapped M -> W through the completed bridge; emblem via im77_field_symbols).
Features = everything else on the object: bag of the other signs, the neighbours of the target position, text
length, object sub-type, facts (emblem/material/boss/shape when they are not the target) and ARITHMETIC
features (sum of sign indices mod k, k = 2..13; sum of S204 numeral values; counts of fish/numerals/strokes)
so that a true checksum would be learnable.
Baselines: modal class; slot-grammar baseline = modal class given the frame class of the rest of the text
(opener present, connective present, suffix present, jar-closer present, numeral present, length bucket).
Planted controls (same pipeline, same hold-outs): (P1) target := class[(sum of other sign indices) mod k]
(a true checksum); (P2) target := lookup[first other sign] (a class sign fixed by content); (P3) labels shuffled
within the training set (must fall to baseline). Three merge levels.
Usage: python3 tools/dark_loop34.py <cycle 1|2|3> <seq_raw|seq_strong|seq_all>
"""
import json, csv, sys, random, collections, math, re
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.inspection import permutation_importance
from scipy.stats import binomtest

SP = '/tmp/claude-0/-home-user-Indus-/874df4c7-80d6-5f08-b42c-eea96a214079/scratchpad/'
CY = int(sys.argv[1]) if len(sys.argv) > 1 else 1
LV = sys.argv[2] if len(sys.argv) > 2 else 'seq_raw'
rnd = random.Random(34); np.random.seed(34)

# ------------------------------------------------------------------ data
C = json.load(open('data/derived/merged-corpus-canonical.json'))
CSV = {r['cisi']: r for r in csv.DictReader(open('data/raw/inscriptions.csv'))}
SETS = json.load(open('data/derived/dark/loop27_sets.json'))
BR = SETS['bridge']                      # W -> [M], completed bridge (S-DARK-27.3)
MERGE = {}
for x in C:
    for a, b in zip(x['seq_raw'], x[LV]): MERGE[a] = b

OPEN = {817, 861, 820, 920, 692}; MARK = {2, 60}; SUF = {400, 90}
CLOSERS = [740, 520, 151, 156, 527, 226, 617, 154, 158, 236, 700, 342]
FISH = {220, 235, 240, 233, 231}
SHORT = (set(range(1, 8)) | set(range(12, 21)) | set(range(25, 30))) - {2}
TALL = set(range(32, 38)) | {39}
VAL = {**{i: i for i in range(1, 8)}, **{i: i - 10 for i in range(12, 21)},
       **{i: i - 20 for i in range(25, 30)}, **{i: i - 30 for i in range(32, 38)}, 39: 9}
NUMER = SHORT | TALL

def emblem_class(sym):
    s = (sym or '').split(':')[0]
    if s in ('None',): return 'none'
    if s in ('-', ''): return None
    return {'Bull1': 'unicorn', 'Gaur': 'shorthorn', 'Zebu': 'zebu', 'Elep': 'elephant', 'Tigr': 'tiger',
            'Rhin': 'rhino', 'Buff': 'buffalo', 'Goat': 'goat', 'Gavi': 'gharial'}.get(s, 'other')

def emblem_class_im77(desc):
    d = desc.lower()
    if 'no field symbol' in d: return 'none'
    if 'uncertain' in d or 'illegible' in d: return None
    for k, v in (('unicorn', 'unicorn'), ('short-horned', 'shorthorn'), ('humped', 'zebu'), ('elephant', 'elephant'),
                 ('tiger', 'tiger'), ('rhinoceros', 'rhino'), ('buffalo', 'buffalo'), ('goat', 'goat'), ('gharial', 'gharial')):
        if k in d: return v
    return 'other'

def boss_class(b):
    if b in ('-', ''): return None
    if b.startswith('PBSG'): return 'PBSG'
    if b.startswith('PB'): return 'PB'
    if b.startswith('PN') or b.startswith('P'): return 'P'
    if b.startswith('N'): return 'N'
    return 'other'

def shape_class(s):
    if s in ('-', ''): return None
    return s if s in ('square', 'rectangular', 'circular', 'cylindrical', 'prism', 'cuboid-convex') else 'other'

def material_class(m):
    m = m.lower()
    if m in ('-', ''): return None
    return m if m in ('steatite', 'faience', 'clay', 'terracotta', 'copper', 'ivory', 'stoneware') else 'other'

def otype(t):
    return t.split(':')[0]

OBJ = []
for r in C:
    s = r[LV]
    if not s or len(s) < 2 or r['complete'] != 'Y' or r['dir.'].strip() == '-': continue
    cr = CSV.get(r['cisi'], {})
    OBJ.append(dict(id=r['cisi'], site=r['site'], type=r['type'], ot=otype(r['type']), seq=list(s),
                    emblem=emblem_class(r.get('symbol')), material=material_class(r.get('material', '-')),
                    boss=boss_class(cr.get('boss', '-')), shape=shape_class(r.get('shape', '-')),
                    big=r['site'] in ('Mohenjo-daro', 'Harappa'), src='wells'))

# IM77-only texts in W space
WFREQ = collections.Counter(t for o in OBJ for t in o['seq'])
M2W = {}
for w, ms in BR.items():
    for m in ms:
        if m not in M2W or WFREQ[int(w)] > WFREQ[M2W[m]]: M2W[m] = int(w)
FSD = {r['fs80']: r['description'] for r in csv.DictReader(open('data/im77/im77_field_symbols.csv'))}
IML = collections.defaultdict(list)
for r in csv.DictReader(open('data/im77/im77_corpus_lines.csv')): IML[(r['text_no'], r['side'])].append(r)
NEW = []
unk = 0; tot = 0
for tn, side in SETS['new']:
    L = sorted(IML.get((tn, side), []), key=lambda r: int(r['line']))
    if not L: continue
    ms = []
    for r in L: ms += [int(x) for x in r['signs_clean'].split()]
    if not ms or 0 in ms or len(ms) < 2: continue
    seq = []
    for m in ms:
        tot += 1
        if m in M2W: seq.append(MERGE.get(M2W[m], M2W[m]))
        else: seq.append(-m); unk += 1
    ot = {'seal': 'SEAL', 'sealing': 'TAG', 'miniature tablet': 'TAB', 'copper tablet': 'TAB', 'pottery graffito': 'POT'}.get(L[0]['object_type'], 'MISC')
    NEW.append(dict(id=f'IM{tn}:{side}', site='IM77-new:' + L[0]['site'], type=ot, ot=ot, seq=seq,
                    emblem=emblem_class_im77(FSD.get(L[0]['fs80'], '')), material=None, boss=None, shape=None, big=False, src='im77'))
print(f'== S-DARK-34 cycle {CY} level {LV}: Wells objects {len(OBJ)}; IM77-only texts {len(NEW)} '
      f'(unmapped M tokens {unk}/{tot}); IM77 types {collections.Counter(o["ot"] for o in NEW).most_common()}')

# ------------------------------------------------------------------ targets
def closer_of(seq):
    s = seq[:]
    while len(s) > 1 and s[-1] in SUF: s.pop()
    return s[-1] if s[-1] in CLOSERS else 'none'

def frame_class(seq, exclude_pos=None):
    s = [t for i, t in enumerate(seq) if i != exclude_pos]
    if not s: return 'empty'
    return '|'.join([str(int(s[0] in OPEN)), str(int(any(t in MARK for t in s[:3]))), str(int(s[-1] in SUF)),
                     str(int(740 in s)), str(int(any(t in NUMER for t in s))), str(min(len(s), 7))])

TARGETS = {
    # name: (getter -> (label, excluded position or None), applicable filter)
    'first_sign': (lambda o: (o['seq'][0], 0), lambda o: True),
    'last_sign': (lambda o: (o['seq'][-1], len(o['seq']) - 1), lambda o: True),
    'closer': (lambda o: (closer_of(o['seq']), None), lambda o: True),
    'second_sign': (lambda o: (o['seq'][1], 1), lambda o: len(o['seq']) >= 3),
    'penult_sign': (lambda o: (o['seq'][-2], len(o['seq']) - 2), lambda o: len(o['seq']) >= 3),
    'emblem': (lambda o: (o['emblem'], None), lambda o: o['emblem'] is not None),
    'material': (lambda o: (o['material'], None), lambda o: o['material'] is not None),
    'boss': (lambda o: (o['boss'], None), lambda o: o['boss'] is not None),
    'shape': (lambda o: (o['shape'], None), lambda o: o['shape'] is not None),
}

# ------------------------------------------------------------------ features
class Vec:
    def __init__(self, train_objs, target, use_facts=True):
        self.target = target; self.use_facts = use_facts
        cnt = collections.Counter(t for o in train_objs for t in o['seq'])
        self.signs = [s for s, _ in cnt.most_common(200)]
        self.sidx = {s: i for i, s in enumerate(self.signs)}
        ctx = [s for s, _ in cnt.most_common(60)]
        self.cidx = {s: i for i, s in enumerate(ctx)}
        self.facts = [f for f in ('emblem', 'material', 'boss', 'shape') if f != target] if use_facts else []
        self.fvals = {f: sorted({str(o[f]) for o in train_objs}) for f in self.facts}
        self.types = sorted({o['type'] for o in train_objs})
        self.names = ([f'count_W{s}' for s in self.signs] + ['count_other', 'length', 'n_fish', 'n_numeral', 'numeral_value_sum',
                      'n_distinct', 'has_opener', 'has_marker', 'has_suffix'] +
                      [f'sumW_mod{k}' for k in range(2, 14)] + [f'sumidx_mod{k}' for k in range(2, 14)] +
                      [f'prev_W{s}' for s in ctx] + ['prev_other', 'prev_none'] + [f'next_W{s}' for s in ctx] + ['next_other', 'next_none'] +
                      [f'first_W{s}' for s in ctx] + ['first_other'] + [f'last_W{s}' for s in ctx] + ['last_other'] +
                      [f'type={t}' for t in self.types] + [f'{f}={v}' for f in self.facts for v in self.fvals[f]])
    def row(self, o, excl):
        seq = o['seq']; rest = [t for i, t in enumerate(seq) if i != excl]
        v = []
        c = collections.Counter(rest)
        v += [c.get(s, 0) for s in self.signs]
        v.append(sum(n for s, n in c.items() if s not in self.sidx))
        v += [len(rest), sum(1 for t in rest if t in FISH), sum(1 for t in rest if t in NUMER),
              sum(VAL.get(t, 0) for t in rest if t in NUMER), len(set(rest)),
              int(bool(rest) and rest[0] in OPEN), int(any(t in MARK for t in rest[:3])), int(bool(rest) and rest[-1] in SUF)]
        sw = sum(abs(t) for t in rest); si = sum(self.sidx.get(t, 200) for t in rest)
        v += [sw % k for k in range(2, 14)] + [si % k for k in range(2, 14)]
        for nb in ((seq[excl - 1] if excl is not None and excl > 0 else None),
                   (seq[excl + 1] if excl is not None and excl + 1 < len(seq) else None)):
            oh = [0] * (len(self.cidx) + 2)
            if nb is None: oh[-1] = 1
            elif nb in self.cidx: oh[self.cidx[nb]] = 1
            else: oh[-2] = 1
            v += oh
        for end in (rest[0] if rest else None, rest[-1] if rest else None):
            oh = [0] * (len(self.cidx) + 1)
            if end in self.cidx: oh[self.cidx[end]] = 1
            else: oh[-1] = 1
            v += oh
        v += [int(o['type'] == t) for t in self.types]
        for f in self.facts: v += [int(str(o[f]) == val) for val in self.fvals[f]]
        return v
    def matrix(self, objs, excls):
        return np.array([self.row(o, e) for o, e in zip(objs, excls)], dtype=float)

def labels_for(objs, tname, topk=12):
    get, _ = TARGETS[tname]
    labs, excls = [], []
    for o in objs:
        l, e = get(o); labs.append(str(l)); excls.append(e)
    return labs, excls

def reduce_labels(train_labs, other_labs_list, topk=12):
    keep = {l for l, _ in collections.Counter(train_labs).most_common(topk)}
    f = lambda L: [l if l in keep else 'OTHER' for l in L]
    return f(train_labs), [f(L) for L in other_labs_list]

def fit_models(X, y):
    ms = {}
    ms['tree'] = DecisionTreeClassifier(max_depth=8, min_samples_leaf=3, random_state=0).fit(X, y)
    ms['gbm'] = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.08, max_depth=6, min_samples_leaf=5, random_state=0).fit(X, y)
    Xb = (X > 0).astype(float)
    ms['knn'] = KNeighborsClassifier(n_neighbors=5, metric='cosine').fit(Xb, y)
    return ms

def predict(ms, name, X):
    return ms[name].predict((X > 0).astype(float) if name == 'knn' else X)

def acc(a, b): return float(np.mean(np.array(a) == np.array(b))) if len(a) else float('nan')

def slot_baseline(train_objs, train_labs, train_excl, test_objs, test_excl):
    tab = collections.defaultdict(collections.Counter)
    for o, l, e in zip(train_objs, train_labs, train_excl): tab[frame_class(o['seq'], e)][l] += 1
    modal = collections.Counter(train_labs).most_common(1)[0][0]
    return [tab[frame_class(o['seq'], e)].most_common(1)[0][0] if tab[frame_class(o['seq'], e)] else modal
            for o, e in zip(test_objs, test_excl)], modal

def evaluate(tname, train, tests, use_facts, plant=None, shuffle=False, report_feats=True, log=print):
    """tests: dict name -> list of objects. plant: None | 'checksum' | 'classsign'. Returns dict of results."""
    train = [o for o in train if TARGETS[tname][1](o)]
    tests = {k: [o for o in v if TARGETS[tname][1](o)] for k, v in tests.items()}
    ytr, etr = labels_for(train, tname)
    ytes = {k: labels_for(v, tname) for k, v in tests.items()}
    ytr, others = reduce_labels(ytr, [ytes[k][0] for k in tests])
    ytes = {k: (others[i], ytes[k][1]) for i, k in enumerate(tests)}
    classes = sorted(set(ytr))
    if plant:  # overwrite labels with a deterministic function of the rest of the object
        kk = min(8, len(classes))
        def f(o, e):
            rest = [t for i, t in enumerate(o['seq']) if i != e]
            if plant == 'checksum': return classes[sum(abs(t) for t in rest) % kk]
            return classes[(abs(rest[0]) * 7919) % kk if rest else 0]   # class sign: fixed by the first other sign
        ytr = [f(o, e) for o, e in zip(train, etr)]
        ytes = {k: ([f(o, e) for o, e in zip(tests[k], ytes[k][1])], ytes[k][1]) for k in tests}
    if shuffle:
        ytr = ytr[:]; rnd.shuffle(ytr)
    V = Vec(train, tname, use_facts)
    Xtr = V.matrix(train, etr)
    ms = fit_models(Xtr, ytr)
    out = dict(target=tname, n_train=len(train), n_classes=len(set(ytr)), plant=plant, shuffle=shuffle, facts=use_facts)
    modal = collections.Counter(ytr).most_common(1)[0][0]
    out['train_modal_share'] = ytr.count(modal) / len(ytr)
    for k, objs in tests.items():
        if not objs: continue
        y, e = ytes[k]
        X = V.matrix(objs, e)
        res = {m: acc(predict(ms, m, X), y) for m in ms}
        sb, _ = slot_baseline(train, ytr, etr, objs, e)
        res['modal'] = acc([modal] * len(y), y); res['slot'] = acc(sb, y); res['n'] = len(y)
        best = max(('tree', 'gbm', 'knn'), key=lambda m: res[m])
        res['best'] = best
        # binomial test: best model vs the slot baseline rate
        nhit = int(round(res[best] * len(y)))
        res['p_vs_slot'] = binomtest(nhit, len(y), max(res['slot'], 1e-9), alternative='greater').pvalue if res['slot'] < 1 else 1.0
        out[k] = res
    if report_feats and not shuffle:
        try:
            pi = permutation_importance(ms['gbm'], Xtr, ytr, n_repeats=3, random_state=0, max_samples=min(600, len(ytr)))
            top = np.argsort(-pi.importances_mean)[:5]
            out['top_features'] = [(V.names[i], round(float(pi.importances_mean[i]), 3)) for i in top if pi.importances_mean[i] > 0]
        except Exception as ex:
            out['top_features'] = str(ex)
    return out, (ms, V, ytr, etr, train)

def fmt(res):
    s = f"{res['target']:12s} n={res['n_train']:4d} k={res['n_classes']:2d}"
    if res['plant']: s += f" PLANT={res['plant']}"
    if res['shuffle']: s += ' SHUFFLED'
    for k in ('sites', 'im77'):
        if k in res:
            r = res[k]
            s += (f" | {k} n={r['n']}: tree {r['tree']:.2f} gbm {r['gbm']:.2f} knn {r['knn']:.2f} | modal {r['modal']:.2f} slot {r['slot']:.2f}"
                  f" | best={r['best']} p={r['p_vs_slot']:.3f}")
    if res.get('top_features'): s += f" | feats {res['top_features']}"
    return s
