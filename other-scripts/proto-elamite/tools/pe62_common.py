"""pe62 READ WHAT THE CLERK LEFT OUT: shared library.

Invert the problem of unmarked entries.  Train thousands of random classifiers on Proto-Elamite entries
whose class sign IS written (class sign deleted before featurising), using everything else (numerals,
rounding, the name string, position, neighbouring entries, tablet size and system, header, museum batch);
judge them on held-out tablets with the class masked; then fill in every unmarked entry and test what
changes.  Same pipeline on proto-cuneiform and Ur III with the commodity/class signs deleted (controls),
with planted implicit classes, label-shuffle, within-tablet shuffle and number-shuffle nulls.

Entry record (all corpora):
  {'tab','idx','pos','n_ent','surf','name':[signs without class signs],'nums':[[n,code]],'sys':str,
   'val_a','val_b' (log2 values under two value maps), 'label': class or None, 'raw_signs': [...]}
Tablet record: {'id','batch','museum','site','header','tsys','has_total','h','w','entries':[...]}
"""
import os, sys, re, json, math, hashlib, random, csv
from collections import Counter, defaultdict
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
PEROOT = os.path.abspath(os.path.join(HERE, '..'))
DATA = os.path.join(PEROOT, 'data')
LOOPS = os.path.join(PEROOT, 'loops')
CK = os.path.join(DATA, 'pe62_ckpt')
os.makedirs(CK, exist_ok=True)
SCR = '/tmp/claude-0/-home-user-Indus-/874df4c7-80d6-5f08-b42c-eea96a214079/scratchpad'

import pe59_lib as L59  # noqa: E402


def hsplit(key, tag, frac):
    return (int(hashlib.sha256((tag + key).encode()).hexdigest(), 16) % 1000) < frac * 1000


def lg(x):
    return math.log2(float(x)) if x is not None and x > 0 else -1.0


# ------------------------------------------------------------------ catalogue (batch, museum, size)
def catalogue(period_prefix):
    fn = os.path.join(CK, 'cat_%s.json' % re.sub(r'\W', '', period_prefix))
    if os.path.exists(fn):
        return json.load(open(fn))
    csv.field_size_limit(10 ** 9)
    out = {}
    for r in csv.DictReader(open(os.path.join(SCR, 'cdli_cat.csv'), errors='replace')):
        if not r['period'].startswith(period_prefix) or not r['id_text'].isdigit():
            continue
        pub = r['primary_publication'].split(',')[0].strip()
        mus = (r['museum_no'].split(' ')[0] or 'unk')
        def f(x):
            try:
                return float(x)
            except ValueError:
                return None
        out['P%06d' % int(r['id_text'])] = {'batch': pub, 'museum': mus, 'h': f(r['height']), 'w': f(r['width']),
                                            'prov': r['provenience'].split(' (')[0]}
    json.dump(out, open(fn, 'w'))
    return out


# ------------------------------------------------------------------ Proto-Elamite
PE_CLASS_ORDER = ['ALLOT', 'FRACLINE', 'MEASURED', 'COUNTED', 'PERSON']


def pe_class_sets():
    R = L59.pe_roles()['sets']
    return {'ALLOT': {'M288'}, 'FRACLINE': set(R['FRACLINE']), 'MEASURED': set(R['MEASURED']),
            'COUNTED': set(R['COUNTED']), 'PERSON': set(R['PERSON'])}


def pe_sysfeat(nums):
    if not nums or nums[0][1] == 'n':
        return 'NONE', None, None
    cls = L59.ncls(nums)
    cap, cnt = L59.pe_maps()
    a = L59.value(nums, cap)
    b = L59.value(nums, cnt['dec2'])
    return cls, a, b


def build_pe():
    """Entries of PE tablets.  label = class of the FINAL sign (pe59 line_role); class signs anywhere in the
    entry are deleted from the name string.  Unmarked = name string but no class sign (OTHER), or bare number."""
    fn = os.path.join(CK, 'pe_entries.json')
    if os.path.exists(fn):
        return json.load(open(fn))
    T = L59.build_pe()
    cat = catalogue('Proto-Elamite')
    CS = pe_class_sets()
    allcls = set().union(*CS.values())
    roles = L59.pe_roles()
    tabs = []
    for t in T:
        c = cat.get(t['id'], {})
        H = [l for l in t['lines'] if l['role'] == 'H']
        hdr = H[0]['signs'][0] if H and H[0]['signs'] else 'NONE'
        E = [l for l in t['lines'] if l['role'] == 'E']
        tot = [l for l in t['lines'] if l['role'] == 'T']
        tab = {'id': t['id'], 'batch': c.get('batch', 'unk'), 'museum': c.get('museum', 'unk'), 'site': t['site'],
               'header': hdr, 'hrole': L59.header_role(hdr, roles) if hdr != 'NONE' else 'NONE',
               'tsys': L59.tablet_type(t) or 'AMBT', 'has_total': bool(tot), 'h': c.get('h'), 'w': c.get('w'),
               'n_lines': len(t['lines']), 'entries': [],
               'totals': [{'signs': l['signs'], 'nums': l['nums']} for l in tot]}
        for i, l in enumerate(E):
            sg = l['signs']
            lab = None
            if sg:
                f = sg[-1]
                for k in PE_CLASS_ORDER:
                    if f in CS[k]:
                        lab = k
                        break
            name = [s for s in sg if s not in allcls]
            kind = 'LAB' if lab else ('BARE' if not sg else ('IN' if any(s in allcls for s in sg) else 'NAME'))
            cls, a, b = pe_sysfeat(l['nums'])
            tab['entries'].append({'idx': i, 'n_ent': len(E), 'surf': l['surf'], 'name': name, 'nums': l['nums'],
                                   'sys': cls, 'va': lg(a), 'vb': lg(b), 'label': lab, 'kind': kind,
                                   'raw': sg, 'clean': l['clean']})
        tabs.append(tab)
    json.dump(tabs, open(fn, 'w'))
    return tabs


# ------------------------------------------------------------------ proto-cuneiform control
PC_CLASSES = {
    'GRAIN': {'SZE', 'ZIZ2', '|SZE+NAM2|', 'SZE~a'},
    'PRODUCT': {'GAR', 'NINDA2', 'KU6', 'KASZ', 'KU6~a', 'DUG'},
    'ANIMAL': {'UDU', 'U8', 'UDUNITA', 'SILA4', 'MASZ', 'MASZ2', 'UD5', 'GU4', 'AB2', 'ANSZE', 'SZAH2', 'AMAR', 'KISZ'},
    'PERSON': {'SAL', 'KUR', 'ERIM', 'SAG', 'GURUSZ', 'KUR2'},
    'GOODS': {'TUG2', 'SIG2', 'URUDU', 'GADA'},
}
PC_ORDER = ['GRAIN', 'PRODUCT', 'ANIMAL', 'PERSON', 'GOODS']


def pc_sysfeat(nums):
    if not nums or nums[0][1] == 'n':
        return 'NONE', None, None
    cls = L59.ncls(nums)
    cap, cnt = L59.pc_maps()
    a = L59.value(nums, cap)
    b = L59.value(nums, cnt['S'])
    return cls, a, b


def build_pc():
    fn = os.path.join(CK, 'pc_entries.json')
    if os.path.exists(fn):
        return json.load(open(fn))
    T = L59.build_pc()
    cat = catalogue('Uruk')
    allcls = set().union(*PC_CLASSES.values())
    tabs = []
    for t in T:
        c = cat.get(t['id'], {})
        H = [l for l in t['lines'] if l['role'] == 'H']
        hdr = H[0]['signs'][0] if H and H[0]['signs'] else 'NONE'
        E = [l for l in t['lines'] if l['role'] == 'E']
        if len(E) < 2:
            continue
        tab = {'id': t['id'], 'batch': c.get('batch', 'unk'), 'museum': c.get('museum', 'unk'), 'site': t['site'],
               'header': hdr, 'hrole': 'H_X' if hdr != 'NONE' else 'NONE', 'tsys': L59.tablet_type(t) or 'AMBT',
               'has_total': any(l['role'] == 'T' for l in t['lines']), 'h': c.get('h'), 'w': c.get('w'),
               'n_lines': len(t['lines']), 'entries': [], 'totals': []}
        for i, l in enumerate(E):
            sg = l['signs']
            lab = None
            for k in PC_ORDER:
                if any(s in PC_CLASSES[k] for s in sg):
                    lab = k
                    break
            name = [s for s in sg if s not in allcls]
            kind = 'LAB' if lab else ('BARE' if not sg else 'NAME')
            cls, a, b = pc_sysfeat(l['nums'])
            tab['entries'].append({'idx': i, 'n_ent': len(E), 'surf': l['surf'], 'name': name, 'nums': l['nums'],
                                   'sys': cls, 'va': lg(a), 'vb': lg(b), 'label': lab, 'kind': kind, 'raw': sg,
                                   'clean': l['clean']})
        tabs.append(tab)
    json.dump(tabs, open(fn, 'w'))
    return tabs


# ------------------------------------------------------------------ Ur III control
U3_CLASSES = {
    'SMALLCATTLE': {'udu', 'u8', 'sila4', 'masz2', 'ud5', 'kir11', 'masz', 'udu-nita2', 'u8-sila4', 'gukkal', 'sila4-nita2',
                    'masz2-gal', 'masz2-nita2', 'kun-gid2'},
    'CATTLE': {'gu4', 'ab2', 'amar', 'gu4-niga', 'anse', 'dusu2', 'szah2'},
    'GRAIN': {'sze', 'zi3', 'dabin', 'kasz', 'ninda', 'zi3-sig15', 'esza', 'dida', 'kasz-saga', 'kasz-du', 'zu2-lum',
              'ninda-ba'},
    'OIL': {'i3', 'i3-giszc', 'i3-nun', 'i3-szah2', 'i3-gisz'},
    'PERSON': {'gurusz', 'geme2', 'dumu', 'erin2', 'dumu-nita2', 'dumu-munus', 'ug3-il2', 'szu-gi4'},
    'TEXTILE': {'tug2', 'sig2', 'gu2', 'gada', 'kusz'},
    'METAL': {'ku3-babbar', 'urudu', 'an-na', 'ku3-sig17', 'zabar'},
}
U3_ORDER = ['SMALLCATTLE', 'CATTLE', 'PERSON', 'GRAIN', 'OIL', 'TEXTILE', 'METAL']
U3_UNITS = {'gur': 'CAP', 'sila3': 'CAP', 'gin2': 'WT', 'ma-na': 'WT', 'gu2': 'WT', 'sar': 'AREA', 'iku': 'AREA',
            'bur3': 'AREA', 'esze3': 'AREA', 'sze': None}
U3_NUM = {'disz': 1, 'u': 10, 'gesz2': 60, "gesz'u": 600, 'szar2': 3600, 'asz': 1, 'ban2': 10, 'barig': 60,
          'bur3': 18, 'esze3': 6, 'iku': 1}


def u3_clean(w):
    return re.sub(r'[#!?*<>\[\]]', '', w)


def build_u3(max_tabs=4000):
    fn = os.path.join(CK, 'u3_entries.json')
    if os.path.exists(fn):
        return json.load(open(fn))
    cat = {}
    csv.field_size_limit(10 ** 9)
    for r in csv.DictReader(open(os.path.join(SCR, 'cdli_cat.csv'), errors='replace')):
        if r['period'].startswith('Ur III') and r['genre'] == 'Administrative' and r['id_text'].isdigit():
            def f(x):
                try:
                    return float(x)
                except ValueError:
                    return None
            cat['P%06d' % int(r['id_text'])] = {'batch': r['primary_publication'].split(',')[0].strip(),
                                                'museum': r['museum_no'].split(' ')[0] or 'unk',
                                                'prov': r['provenience'].split(' (')[0], 'h': f(r['height']),
                                                'w': f(r['width'])}
    cls_of = {}
    for k in U3_ORDER:
        for w in U3_CLASSES[k]:
            cls_of.setdefault(w, k)
    raw = {}
    cur = None
    surf = 'obverse'
    for line in open(os.path.join(SCR, 'cdli.atf'), encoding='utf-8', errors='replace'):
        line = line.rstrip('\n')
        if line.startswith('&P'):
            pid = line[1:8]
            cur = pid if pid in cat else None
            if cur:
                raw[cur] = []
            surf = 'obverse'
            continue
        if cur is None:
            continue
        if line.startswith('@'):
            s = line[1:].split()[0] if line[1:].split() else ''
            if s in ('obverse', 'reverse', 'left', 'right', 'top', 'bottom', 'edge'):
                surf = s
            elif s in ('seal', 'envelope'):
                cur = None
            continue
        m = re.match(r"^\d+[a-z']*\.\s+(.*)$", line)
        if m:
            raw[cur].append((surf, m.group(1)))
    pids = sorted(raw, key=lambda p: hashlib.sha256(('u3' + p).encode()).hexdigest())
    tabs = []
    for pid in pids:
        lines = raw[pid]
        if len(lines) < 3:
            continue
        E = []
        hdr = 'NONE'
        tot = False
        for surf, txt in lines:
            toks = [u3_clean(w) for w in txt.split()]
            toks = [w for w in toks if w]
            if any(w in ('szu-nigin2', 'szunigin', 'szu-nigin') for w in toks):
                tot = True
                continue
            nums = []
            i = 0
            while i < len(toks):
                mm = re.match(r"^(\d+)(?:/(\d+))?\(([a-z0-9']+)\)$", toks[i])
                if not mm or mm.group(2):
                    break
                nums.append([int(mm.group(1)), mm.group(3)])
                i += 1
            if not nums:
                if not E and toks and hdr == 'NONE':
                    hdr = toks[0]
                continue
            if any(c not in U3_NUM for _, c in nums):
                continue
            rest = toks[i:]
            if any(w in ('x', '...') or 'x' == w for w in rest) or '...' in txt:
                continue
            sysf = 'CNT'
            name = []
            lab = None
            for w in rest:
                if w in U3_UNITS and U3_UNITS[w]:
                    sysf = U3_UNITS[w]
                    continue
                if w in cls_of:
                    if lab is None:
                        lab = cls_of[w]
                    continue
                name.append(w)
            if lab == 'GRAIN' and sysf != 'CAP' and 'sze' in rest and sysf == 'CNT':
                pass
            v = sum(n * U3_NUM[c] for n, c in nums)
            E.append({'surf': surf, 'name': name[:6], 'nums': nums, 'sys': sysf, 'va': lg(v),
                      'vb': lg(len(nums)), 'label': lab, 'kind': 'LAB' if lab else ('BARE' if not name else 'NAME'),
                      'raw': rest, 'clean': True})
        if len(E) < 2:
            continue
        for i, e in enumerate(E):
            e['idx'] = i
            e['n_ent'] = len(E)
        c = cat[pid]
        tabs.append({'id': pid, 'batch': c['batch'], 'museum': c['museum'], 'site': c['prov'], 'header': hdr,
                     'hrole': 'H_X' if hdr != 'NONE' else 'NONE', 'tsys': 'AMBT', 'has_total': tot, 'h': c['h'],
                     'w': c['w'], 'n_lines': len(lines), 'entries': E, 'totals': []})
        if len(tabs) >= max_tabs:
            break
    json.dump(tabs, open(fn, 'w'))
    return tabs


def match_size(tabs, n_lab, tag):
    """Subsample tablets (deterministic hash order) until n_lab labelled entries: PE-size control."""
    order = sorted(tabs, key=lambda t: hashlib.sha256((tag + t['id']).encode()).hexdigest())
    out, k = [], 0
    for t in order:
        if k >= n_lab:
            break
        out.append(t)
        k += sum(1 for e in t['entries'] if e['label'])
    return out


# ------------------------------------------------------------------ features
GROUPS = ['num', 'round', 'name', 'first', 'pos', 'nbr', 'tabsz', 'tsys', 'hdr', 'batch']
NUMBER_GROUPS = {'num', 'round', 'tsys'}


def bin_(x, w=1.0):
    return int(math.floor(x / w)) if x is not None else -9


def entry_feats(t, e, vis_labels):
    """Feature dict grouped by prefix.  vis_labels: per-entry visible class (None if unmarked or the target)."""
    f = {}
    # num: numeral system and size
    f['num=sys:' + e['sys']] = 1
    for n, c in e['nums']:
        f['num=code:' + c] = 1
    f['num=va:%d' % bin_(e['va'])] = 1
    f['num=vb:%d' % bin_(e['vb'])] = 1
    f['num=vac'] = max(e['va'], -1) / 10.0
    # round: rounding and digit shape
    nd = sum(n for n, c in e['nums'] if isinstance(n, int))
    f['round=ncodes:%d' % min(len(e['nums']), 4)] = 1
    f['round=ndig:%d' % min(nd, 12)] = 1
    f['round=single1'] = 1 if (len(e['nums']) == 1 and e['nums'][0][0] == 1) else 0
    if e['nums']:
        f['round=low:' + e['nums'][-1][1]] = 1
        f['round=high:' + e['nums'][0][1]] = 1
    # name string
    for s in set(e['name']):
        f['name=' + s] = 1
    f['name=len:%d' % min(len(e['name']), 5)] = 1
    if e['name']:
        f['first=F:' + e['name'][0]] = 1
        f['first=L:' + e['name'][-1]] = 1
    # position
    n = max(e['n_ent'], 1)
    f['pos=rel:%d' % int(4 * e['idx'] / n)] = 1
    f['pos=surf:' + e['surf']] = 1
    f['pos=first'] = 1 if e['idx'] == 0 else 0
    f['pos=last'] = 1 if e['idx'] == n - 1 else 0
    # neighbours (visible classes of other entries)
    i = e['idx']
    prv = vis_labels[i - 1] if i > 0 else 'EDGE'
    nxt = vis_labels[i + 1] if i + 1 < len(vis_labels) else 'EDGE'
    f['nbr=prev:%s' % (prv or 'UNK')] = 1
    f['nbr=next:%s' % (nxt or 'UNK')] = 1
    cnt = Counter(v for j, v in enumerate(vis_labels) if j != i and v)
    tot = sum(cnt.values())
    for k, v in cnt.items():
        f['nbr=share:' + k] = v / tot
    f['nbr=nvis:%d' % min(tot, 6)] = 1
    # tablet size
    f['tabsz=nent:%d' % min(int(math.log2(n + 1)), 6)] = 1
    f['tabsz=h:%d' % bin_(t['h'], 10) if t['h'] else 'tabsz=h:na'] = 1
    f['tabsz=w:%d' % bin_(t['w'], 10) if t['w'] else 'tabsz=w:na'] = 1
    # tablet system (derived from numerals of every line)
    f['tsys=' + t['tsys']] = 1
    f['tsys=tot:%d' % int(t['has_total'])] = 1
    # header
    f['hdr=' + t['hrole']] = 1
    f['hdr=s:' + t['header']] = 1
    # batch / museum / site
    f['batch=' + t['batch']] = 1
    f['batch=m:' + t['museum']] = 1
    f['batch=site:' + t['site']] = 1
    return f


def featurise(tabs, label_override=None):
    """Return list of (tab_index, entry_index, featdict, label, kind).  label_override: dict (ti,ei)->label used
    BOTH as the target and as the visible neighbour label (for shuffles of the labelled layer)."""
    rows = []
    for ti, t in enumerate(tabs):
        labs = [label_override.get((ti, ei), e['label']) if label_override else e['label']
                for ei, e in enumerate(t['entries'])]
        for ei, e in enumerate(t['entries']):
            vis = list(labs)
            vis[ei] = None
            rows.append((ti, ei, entry_feats(t, e, vis), labs[ei], e['kind']))
    return rows


class Design:
    """Sparse design matrix with group column masks."""

    def __init__(self, rows):
        from sklearn.feature_extraction import DictVectorizer
        self.dv = DictVectorizer()
        self.X = self.dv.fit_transform([r[2] for r in rows]).tocsr()
        names = self.dv.get_feature_names_out()
        self.gcols = {g: np.array([i for i, nm in enumerate(names) if nm.split('=')[0] == g]) for g in GROUPS}
        self.names = names


# ------------------------------------------------------------------ random classifier search
def sample_hyp(rng, allowed=GROUPS):
    while True:
        gs = [g for g in allowed if rng.random() < 0.5]
        if gs:
            break
    fam = rng.choice(['lr', 'lr', 'nb', 'et'])
    hp = {'fam': fam, 'groups': gs, 'scale': {g: round(2 ** rng.uniform(-1.5, 1.5), 3) for g in gs}}
    if fam == 'lr':
        hp['C'] = round(10 ** rng.uniform(-2, 1), 4)
        hp['balanced'] = rng.random() < 0.3
    elif fam == 'nb':
        hp['alpha'] = round(10 ** rng.uniform(-2, 0.5), 4)
    else:
        hp['n'] = rng.choice([30, 60])
        hp['mf'] = rng.choice(['sqrt', 0.1, 0.3])
        hp['leaf'] = rng.choice([1, 2, 5])
    return hp


def make_X(D, hp):
    import scipy.sparse as sp
    cols = []
    for g in hp['groups']:
        if len(D.gcols[g]):
            cols.append(D.X[:, D.gcols[g]] * hp['scale'][g])
    return sp.hstack(cols).tocsr()


def fit_pred(hp, Xtr, ytr, Xte, classes, seed=0):
    import warnings
    warnings.filterwarnings('ignore')
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.ensemble import ExtraTreesClassifier
    if hp['fam'] == 'lr':
        m = LogisticRegression(C=hp['C'], solver='newton-cg', max_iter=60, tol=1e-3,
                               class_weight='balanced' if hp['balanced'] else None)
    elif hp['fam'] == 'nb':
        m = MultinomialNB(alpha=hp['alpha'])
        Xtr = abs(Xtr); Xte = abs(Xte)
    else:
        m = ExtraTreesClassifier(n_estimators=hp['n'], max_features=hp['mf'], min_samples_leaf=hp['leaf'],
                                 random_state=seed, n_jobs=1)
    m.fit(Xtr, ytr)
    P = np.full((Xte.shape[0], len(classes)), 1e-4)
    pr = m.predict_proba(Xte)
    for j, c in enumerate(m.classes_):
        P[:, classes.index(c)] = pr[:, j]
    P = np.clip(P, 1e-4, None)
    return P / P.sum(1, keepdims=True)


def score(P, y, classes, prior):
    yi = np.array([classes.index(v) for v in y])
    acc = float((P.argmax(1) == yi).mean())
    ce = float(-np.log2(P[np.arange(len(yi)), yi]).mean())
    h = float(-np.log2(np.array([prior[c] for c in y])).mean())
    bacc = float(np.mean([(P.argmax(1)[yi == k] == k).mean() for k in range(len(classes)) if (yi == k).any()]))
    return {'acc': acc, 'bacc': bacc, 'bits': h - ce}


def search(D, rows, tr_idx, ho_idx, ytr_all, classes, n_hyp, rng, allowed=GROUPS, topk=15, folds=3, log=None):
    """Random search on the training tablets (grouped CV), ensemble of the top-k, scored on held-out rows.
    ytr_all: labels for every row (training labels may be shuffled; held-out labels are true)."""
    tr_idx = np.array(tr_idx); ho_idx = np.array(ho_idx)
    tabs_tr = sorted({rows[i][0] for i in tr_idx})
    fold_of = {t: hash_fold(t, folds) for t in tabs_tr}
    fid = np.array([fold_of[rows[i][0]] for i in tr_idx])
    ytr = [ytr_all[i] for i in tr_idx]
    prior = Counter(ytr)
    prior = {c: (prior[c] + 1) / (len(ytr) + len(classes)) for c in classes}
    res = []
    res_order = []
    for h in range(n_hyp):
        hp = sample_hyp(rng, allowed)
        X = make_X(D, hp)
        P = np.zeros((len(tr_idx), len(classes)))
        for k in range(folds):
            a = fid != k
            b = fid == k
            P[b] = fit_pred(hp, X[tr_idx[a]], [ytr[i] for i in np.where(a)[0]], X[tr_idx[b]], classes, seed=h)
        s = score(P, ytr, classes, prior)
        res.append((s['bits'], hp, s))
        res_order.append((s['bits'], hp, s))
        if log and h % 50 == 0:
            log('hyp %d best %.3f' % (h, max(r[0] for r in res)))
    res.sort(key=lambda r: -r[0])
    top = res[:topk]
    Pho = np.zeros((len(ho_idx), len(classes)))
    for _, hp, _ in top:
        X = make_X(D, hp)
        Pho += fit_pred(hp, X[tr_idx], ytr, X[ho_idx], classes)
    Pho /= len(top)
    out150 = None
    if len(res) > 150:
        sub = sorted(res_order[:150], key=lambda r: -r[0])[:topk]
        out150 = ensemble_predict(D, sub, tr_idx, ytr, ho_idx, classes)
    return {'P_ho150': out150, 'top': [(r[0], r[1], r[2]) for r in top], 'all_bits': [r[0] for r in res],
            'P_ho': Pho, 'prior': prior}


def ensemble_predict(D, top, tr_idx, ytr, pred_idx, classes):
    P = np.zeros((len(pred_idx), len(classes)))
    for _, hp, _ in top:
        X = make_X(D, hp)
        P += fit_pred(hp, X[np.array(tr_idx)], ytr, X[np.array(pred_idx)], classes)
    return P / len(top)


def hash_fold(key, k):
    return int(hashlib.sha256(('pe62f' + str(key)).encode()).hexdigest(), 16) % k


def jdump(obj, fn):
    def conv(o):
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(type(o))
    json.dump(obj, open(fn, 'w'), default=conv)
