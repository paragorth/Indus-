"""S-DARK-67: ARE THE MOHENJO-DARO COPPER TABLETS A KEY (a legend table binding one text to one image)?
Cycle 1: copper-tablet table (text x image, copies, dies, find-spots) in IM77 (fs80 image side) and Wells
         (symbol field + image-sign faces); binding tested with an image-shuffle null (1,000x) and against the seal
         baseline (identical-text seal pairs sharing their emblem); one-to-one / one-to-many / many-to-many at die level.
Cycle 2: the bound texts read against the frame (S310/S331 parser); the same texts / closing units on seals and other
         tablets and the emblems there (S325/S329/S-DARK-34 independence as control).
Cycle 3: copper image set vs seal emblem set; for shared animals, does the copper text agree with that emblem's seal
         statistics (head, closing unit, sign set)? null = emblem labels shuffled among seals.
Cycle 4: Wells vs IM77 label agreement on aligned objects; all three merge levels; leave-one-die-out image -> text
         prediction with hit rates; the prediction for a new copper tablet.
Usage: python3 tools/dark_loop67.py <1|2|3|4> [nperm]
"""
import sys, os, re, json, csv, math, random, collections, itertools

ROOT = '/home/user/Indus-/'
DARK = ROOT + 'data/derived/dark/'
CY = int(sys.argv[1]) if len(sys.argv) > 1 else 1
NP = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
rnd = random.Random(67)
LOG = []
def P(*a):
    s = ' '.join(str(x) for x in a); print(s); LOG.append(s)

# ---------------- image classes ----------------
def fs_class(fs):
    fs = int(fs)
    if fs == 0: return None
    if 11 <= fs <= 19: return 'unicorn'
    if 22 <= fs <= 23: return 'longhorn-bull'
    if 31 <= fs <= 32: return 'zebu'
    if 41 <= fs <= 50: return 'shorthorn-bull'
    if 61 <= fs <= 65: return 'buffalo'
    if 71 <= fs <= 82: return 'elephant'
    if 91 <= fs <= 95: return 'tiger'
    if 101 <= fs <= 104: return 'horned-tiger'
    if 111 <= fs <= 121: return 'rhino'
    if 131 <= fs <= 139: return 'goat-antelope'
    if 141 <= fs <= 150: return 'ox-antelope'
    if fs in (163, 171): return 'hare'
    if 181 <= fs <= 221: return 'animal-row'
    if fs == 290: return 'two-headed'
    if 231 <= fs <= 340: return 'composite'
    if fs in (351, 352): return 'uncertain-bovine'
    if 361 <= fs <= 369: return 'gharial'
    if 371 <= fs <= 460: return 'object/tree'
    if 470 <= fs <= 590: return 'personage'
    if 600 <= fs <= 810: return 'scene'
    if fs == 860: return 'endless-knot'
    if fs == 950: return 'double-axe'
    if 821 <= fs <= 988: return 'symbol'
    if fs == 999: return None
    return 'other'

WSYM = {'Hare': 'hare', 'Elep': 'elephant', 'Anth': 'personage', 'Comp': 'composite', 'Gaur': 'gaur', 'Loop': 'endless-knot',
        'Goat:8': 'goat-antelope', 'Bull1:W': 'unicorn', 'Rhin': 'rhino', 'Tigr': 'tiger', 'Bull': 'bull', 'Buff': 'buffalo',
        'Hgls': 'other', 'Unknown': None, 'Othr': 'Othr', 'None': None, '-': None, '': None}
# Wells single-sign faces that occur only on copper tablets: image drawn as a 'sign'
WIMG = {749, 753, 777, 781, 782, 841, 957}

def wsym_seal(sym):
    """coarse emblem class for Wells seal symbols"""
    if not sym or sym in ('-', 'None', 'Unknown', 'Othr'): return None
    s = sym.split(':')[0]
    # Wells/CISI codes: Bull1 = one-horned bull (unicorn); Bull = bull unspecified; Bull2/Bull3 other bulls
    m = {'Unic': 'unicorn', 'Bull1': 'unicorn', 'Bull2': 'bull2', 'Bull3': 'bull3', 'Zebu': 'zebu', 'Elep': 'elephant', 'Rhin': 'rhino',
         'Tigr': 'tiger', 'Buff': 'buffalo', 'Goat': 'goat-antelope', 'Gaur': 'gaur', 'Hare': 'hare', 'Anth': 'personage', 'Comp': 'composite',
         'Ghar': 'gharial', 'Bull': 'bull', 'Loop': 'endless-knot'}
    return m.get(s, s.lower())

# ---------------- frame parser (S310/S331, as tools/dark_loop58.py) ----------------
OPEN = {817, 861, 820, 920, 692}; MARK = {2, 60}; MJAR = {741, 742, 745}; SUF = {400, 90}
CL = [740, 520, 151, 156, 527, 226, 617, 154, 158, 236, 700]
FISH = {235, 240, 233, 231, 220}; NUM = {1, 3, 4, 5, 16, 17, 18, 31, 32, 33, 34, 55, 56}

def build_qual(seqs):
    left = collections.defaultdict(collections.Counter)
    for s in seqs:
        s = list(s)
        while len(s) > 1 and s[-1] in SUF: s.pop()
        if len(s) >= 2 and s[-1] in CL: left[s[-1]][s[-2]] += 1
    Q = {}
    for c, cnt in left.items():
        tot = sum(cnt.values()); acc = 0; q = set()
        for a, n in cnt.most_common():
            if acc / tot >= 0.6: break
            q.add(a); acc += n
        Q[c] = q
    return Q

def parse(s, QUAL):
    lab = ['NAME'] * len(s); i = 0; j = len(s)
    if s[0] in OPEN:
        lab[0] = 'OPENER'; i = 1
        if len(s) > 1 and s[1] in MARK:
            lab[1] = 'MARKER'; i = 2
            if s[0] == 920 and len(s) > 2 and s[2] in MJAR: lab[2] = 'MARKER'; i = 3
    while j - 1 > i and s[j - 1] in SUF and j >= 2 and (s[j - 2] in CL or s[j - 2] in SUF): lab[j - 1] = 'SUFFIX'; j -= 1
    if j - 1 >= i and s[j - 1] in CL:
        c = s[j - 1]; lab[j - 1] = 'CLOSER'; j -= 1
        if c == 520:
            if j - 2 >= i and s[j - 1] == 33 and s[j - 2] in (705, 706): lab[j - 1] = lab[j - 2] = 'TITLE'; j -= 2
            while j - 1 >= i and s[j - 1] in FISH: lab[j - 1] = 'TITLE'; j -= 1
        elif c == 740:
            if j - 1 >= i and s[j - 1] == 100: lab[j - 1] = 'TITLE'; j -= 1
            if j - 1 >= i and s[j - 1] in QUAL.get(c, ()): lab[j - 1] = 'TITLE'; j -= 1
        elif j - 1 >= i and s[j - 1] in QUAL.get(c, ()): lab[j - 1] = 'TITLE'; j -= 1
        if j - 1 >= i and s[j - 1] in NUM and lab[j] == 'TITLE': lab[j - 1] = 'TITLE'; j -= 1
    for k in range(i, j - 1):
        if s[k] in NUM and lab[k] == 'NAME' and lab[k + 1] == 'NAME': lab[k] = lab[k + 1] = 'COUNT'
    for k in range(i, j):
        if s[k] in NUM and lab[k] == 'NAME': lab[k] = 'COUNT'
    return lab

# ---------------- loaders ----------------
def load_im77():
    rows = list(csv.DictReader(open(ROOT + 'data/im77/im77_corpus_lines.csv')))
    by = collections.defaultdict(list)
    for r in rows: by[r['text_no']].append(r)
    objs = []
    for tn, ls in by.items():
        ls.sort(key=lambda r: (int(r['side']), int(r['line'])))
        sides = collections.defaultdict(list); fs = set()
        for r in ls:
            if r['fs80'] not in ('0', '999'): fs.add(int(r['fs80']))
            if r['line'] == '9' or not r['signs_clean'].strip(): continue
            sides[r['side']].append(r)
        texts = []; imgsign = None
        for sd, lr in sides.items():
            seq = [int(x) for x in ' '.join(r['signs_clean'] for r in lr).split()]
            if len(seq) == 1 and lr[0]['direction_code'] == '3' and len(sides) > 1 and not fs:
                imgsign = seq[0]; continue
            texts.append(dict(side=sd, seq=seq, complete=0 not in seq))
        r0 = ls[0]
        o = dict(id=tn, site=r0['site'], ot=r0['object_type'], fs=sorted(fs), texts=texts, imgsign=imgsign,
                 locus=r0['locus'], level=r0['level'])
        o['img'] = ('fs%d' % o['fs'][0]) if o['fs'] else (('Msign%d' % imgsign) if imgsign else None)
        o['cls'] = fs_class(o['fs'][0]) if o['fs'] else (('Msign%d' % imgsign) if imgsign else None)
        objs.append(o)
    return objs

def load_wells():
    C = json.load(open(ROOT + 'data/derived/merged-corpus-canonical.json'))
    rows = list(csv.DictReader(open(ROOT + 'data/raw/inscriptions.csv')))
    g = lambda r, k: (r.get(k) or '')
    def parsed(r): return [x for x in (int(y) for y in re.findall(r'\d{3}', r['text'])) if x != 0]
    j = 0; faces = []
    for c in C:
        while rows[j]['cisi'] != c['cisi']: j += 1
        r = rows[j]; j += 1
        raw = c['seq_raw']
        ok = list(reversed(parsed(r))) == raw
        txt = g(r, 'text')
        f = dict(id=g(r, 'id'), obj=g(r, 'id').split('.')[0], cisi=c['cisi'], site=c['site'], type=c['type'], symbol=g(r, 'symbol'),
                 area=g(r, 'area-section'), block=g(r, 'block-house'), depth=g(r, 'depth'), material=g(r, 'material'),
                 complete=(g(r, 'complete') == 'Y') and '000' not in txt, text_ok=ok, txt=txt,
                 seq_raw=list(c['seq_raw']), seq_strong=list(c['seq_strong']), seq_all=list(c['seq_all']))
        faces.append(f)
    return faces

def wells_copper(faces, level='seq_raw'):
    """group TAB:C faces into objects with text faces + image label"""
    by = collections.defaultdict(list)
    for f in faces:
        if f['type'] == 'TAB:C': by[f['obj']].append(f)
    objs = []
    for ob, fs in by.items():
        texts = []; img = None; imgsign = None
        for f in fs:
            s = f[level]
            if len(s) == 1 and s[0] in WIMG: imgsign = s[0]; continue
            texts.append(dict(seq=s, complete=f['complete'], id=f['id']))
        sym = fs[0]['symbol']
        cls = WSYM.get(sym, sym)
        if imgsign is not None: img = 'W%d' % imgsign; cls = 'Wsign%d' % imgsign
        elif cls: img = sym
        o = dict(id=ob, cisi=fs[0]['cisi'], site=fs[0]['site'], texts=texts, img=img, cls=cls, symbol=sym,
                 area=fs[0]['area'], block=fs[0]['block'], depth=fs[0]['depth'])
        objs.append(o)
    return objs

def m2w_map():
    BR = json.load(open(ROOT + 'data/derived/bridge_extended.json'))
    m2w = {}; w2m = {}
    for w, ms in BR.items():
        w2m[int(w)] = [int(m) for m in ms]
        for m in ms: m2w.setdefault(int(m), int(w))
    prop = set()
    for p in json.load(open(DARK + 'bridge_proposals.json'))['proposals']:
        if int(p['M']) not in m2w: m2w[int(p['M'])] = int(p['W']); prop.add(int(p['M']))
        w2m.setdefault(int(p['W']), []).append(int(p['M']))
    return m2w, w2m, prop

# ---------------- statistics ----------------
def entropy(cnt):
    n = sum(cnt.values()); return -sum(v / n * math.log2(v / n) for v in cnt.values() if v)

def mi(pairs):
    cx = collections.Counter(a for a, b in pairs); cy = collections.Counter(b for a, b in pairs); cxy = collections.Counter(pairs)
    return entropy(cx) + entropy(cy) - entropy(cxy)

def cond_entropy(pairs):
    """H(image | text)"""
    cx = collections.Counter(a for a, b in pairs); cxy = collections.Counter(pairs)
    return entropy(cxy) - entropy(cx)

def pair_agree(items):
    """items: list of (text, image). among pairs of items with identical text, share with same image"""
    by = collections.defaultdict(list)
    for t, im in items: by[t].append(im)
    same = tot = 0
    for t, ims in by.items():
        for a, b in itertools.combinations(ims, 2):
            tot += 1; same += (a == b)
    return same, tot

def shuffle_test(items, stat, nperm):
    obs = stat(items)
    texts = [t for t, im in items]; ims = [im for t, im in items]
    ge = 0; vals = []
    for _ in range(nperm):
        rnd.shuffle(ims)
        v = stat(list(zip(texts, ims))); vals.append(v); ge += (v >= obs)
    vals.sort()
    return obs, (ge + 1) / (nperm + 1), sum(vals) / len(vals), vals[int(0.025 * nperm)], vals[int(0.975 * nperm) - 1]

def fmt(seq): return '-'.join(str(x) for x in seq)

def binding_table(items, label):
    """items: (text tuple, image, objid). prints the text x image table and the die-level binding type"""
    by_t = collections.defaultdict(collections.Counter); by_i = collections.defaultdict(collections.Counter)
    for t, im, oid in items: by_t[t][im] += 1; by_i[im][t] += 1
    P(f'  [{label}] {len(items)} tablets with complete text and known image; {len(by_t)} distinct texts; {len(by_i)} distinct images; '
      f'{len(set((t, im) for t, im, _ in items))} distinct text x image pairs')
    for t, c in sorted(by_t.items(), key=lambda kv: -sum(kv[1].values())):
        P(f'    {fmt(t):45s} n={sum(c.values()):3d}  images: ' + ', '.join(f'{im} x{n}' for im, n in c.most_common()))
    t_multi = [t for t, c in by_t.items() if len(c) > 1]
    i_multi = [im for im, c in by_i.items() if len(c) > 1]
    P(f'  texts with >1 image: {len(t_multi)}/{len(by_t)} ({", ".join(fmt(t) for t in t_multi)})')
    P(f'  images with >1 text: {len(i_multi)}/{len(by_i)} (' + '; '.join(f'{im}: ' + ' | '.join(fmt(t) for t in by_i[im]) for im in i_multi) + ')')
    kind = 'one-to-one' if not t_multi and not i_multi else 'one-to-many (text -> several images)' if t_multi and not i_multi else \
           'one-to-many (image -> several texts)' if i_multi and not t_multi else 'many-to-many'
    P(f'  die-level binding type: {kind}')
    return by_t, by_i

# ====================================================================================================
def cycle1(nperm):
    P('# S-DARK-67.1  copper-tablet table and text-image binding')
    IM = load_im77(); W = load_wells()
    cu = [o for o in IM if o['ot'] == 'copper tablet']
    P(f'IM77: {len(cu)} copper tablets (all Mohenjodaro); with an image side (fs80): {sum(1 for o in cu if o["fs"])}; '
      f'with an image drawn as a single "sign" face: {sum(1 for o in cu if o["imgsign"])}; one side only: {sum(1 for o in cu if len(o["texts"]) + bool(o["fs"]) + bool(o["imgsign"]) == 1)}')
    P('  fs80 codes on copper tablets: ' + ', '.join(f'fs{k}={v}' for k, v in collections.Counter(f for o in cu for f in o['fs']).most_common()))
    P('  multi-text-side tablets: ' + '; '.join(f'{o["id"]}: ' + ' || '.join(fmt(t["seq"]) for t in o['texts']) + (f' [img {o["img"]}]' if o['img'] else '') for o in cu if len(o['texts']) > 1))
    # items: tablets with image and one complete text (if two text sides, use the longest complete one, note it)
    def items_of(objs, fine=True):
        it = []
        for o in objs:
            if not o['img']: continue
            ts = [t for t in o['texts'] if t['complete']]
            if not ts: continue
            t = max(ts, key=lambda t: len(t['seq']))
            it.append((tuple(t['seq']), o['img'] if fine else o['cls'], o['id']))
        return it
    it_fine = items_of(cu, True); it_cls = items_of(cu, False)
    P('\n## IM77 text x image (fine fs80 code)')
    by_t, by_i = binding_table(it_fine, 'IM77 fine')
    P('\n## IM77 text x image (coarse animal class)')
    binding_table(it_cls, 'IM77 class')
    # dies and find-spots
    P('\n## find-spots per text (IM77 locus = MIC/FEM locus code, level = depth code)')
    byt = collections.defaultdict(list)
    for o in cu:
        for t in o['texts']:
            if t['complete']: byt[tuple(t['seq'])].append((o['locus'], o['level'], o['img']))
    for t, L in sorted(byt.items(), key=lambda kv: -len(kv[1])):
        if len(L) >= 2:
            P(f'    {fmt(t):40s} n={len(L):2d} loci: ' + ', '.join(f'{k}x{v}' for k, v in collections.Counter(l for l, _, _ in L).most_common()))
    # binding test
    P('\n## binding test: image labels shuffled among tablets (%d x)' % nperm)
    for lab, it in (('fine', it_fine), ('class', it_cls)):
        pairs = [(t, im) for t, im, _ in it]
        o, p, mu, lo, hi = shuffle_test(pairs, mi, nperm)
        P(f'  IM77 {lab}: MI(text;image) = {o:.3f} bits, null {mu:.3f} [{lo:.3f},{hi:.3f}], P = {p:.4f}; H(image)={entropy(collections.Counter(im for _, im in pairs)):.3f}, H(text)={entropy(collections.Counter(t for t, _ in pairs)):.3f}')
        o, p, mu, lo, hi = shuffle_test(pairs, lambda ps: -cond_entropy(ps), nperm)
        P(f'  IM77 {lab}: H(image|text) = {-o:.3f} bits, null {-mu:.3f}, P = {p:.4f}')
        s, n = pair_agree(pairs)
        o2, p2, mu2, lo2, hi2 = shuffle_test(pairs, lambda ps: pair_agree(ps)[0] / max(1, pair_agree(ps)[1]), nperm)
        P(f'  IM77 {lab}: identical-text tablet pairs sharing the image: {s}/{n} = {s / max(1, n):.3f}, null {mu2:.3f} [{lo2:.3f},{hi2:.3f}], P = {p2:.4f}')
    # seal baseline, IM77 seals with fs and >= 2 copies of a text
    P('\n## seal baseline (S-DARK-34 / S325): identical-text SEAL pairs sharing the emblem, IM77 seals with a field symbol')
    for site in ('Mohenjodaro', 'all'):
        seals = [o for o in IM if o['ot'] == 'seal' and o['fs'] and (site == 'all' or o['site'] == site)]
        for lab, key in (('fine', 'img'), ('class', 'cls')):
            pairs = []
            for o in seals:
                ts = [t for t in o['texts'] if t['complete']]
                if ts: pairs.append((tuple(max(ts, key=lambda t: len(t['seq']))['seq']), o[key]))
            cnt = collections.Counter(t for t, _ in pairs)
            dup = [(t, im) for t, im in pairs if cnt[t] >= 2]
            s, n = pair_agree(dup)
            o2, p2, mu2, lo2, hi2 = shuffle_test(dup, lambda ps: pair_agree(ps)[0] / max(1, pair_agree(ps)[1]), nperm)
            o3, p3, mu3, lo3, hi3 = shuffle_test(dup, mi, nperm)
            P(f'  seals {site} {lab}: {len(seals)} seals, {len(dup)} seals whose text has >= 2 copies ({len(set(t for t, _ in dup))} texts): pairs sharing emblem {s}/{n} = {s / max(1, n):.3f}, '
              f'null {mu2:.3f} [{lo2:.3f},{hi2:.3f}], P = {p2:.4f}; MI {o3:.3f} vs null {mu3:.3f}, P = {p3:.4f}')
    # Wells side
    P('\n## Wells (inscriptions.csv TAB:C): symbol field + image-sign faces (W749/753/777/781/782/841/957 occur only on copper tablets)')
    for level in ('seq_raw', 'seq_strong', 'seq_all'):
        wc = wells_copper(W, level)
        P(f'  level {level}: {len(wc)} objects; image known {sum(1 for o in wc if o["img"] and o["cls"] != "Othr")}, "Othr" {sum(1 for o in wc if o["cls"] == "Othr")}, none {sum(1 for o in wc if not o["img"])}')
        it = []
        for o in wc:
            if not o['img'] or o['cls'] == 'Othr': continue
            ts = [t for t in o['texts'] if t['complete']]
            if not ts: continue
            it.append((tuple(max(ts, key=lambda t: len(t['seq']))['seq']), o['cls'], o['id']))
        if level == 'seq_raw':
            binding_table(it, 'Wells raw, Othr excluded')
            it2 = []
            for o in wc:
                if not o['img']: continue
                ts = [t for t in o['texts'] if t['complete']]
                if ts: it2.append((tuple(max(ts, key=lambda t: len(t['seq']))['seq']), o['cls'], o['id']))
            P('  (with Othr as a label)'); binding_table(it2, 'Wells raw, Othr kept')
        pairs = [(t, im) for t, im, _ in it]
        o, p, mu, lo, hi = shuffle_test(pairs, mi, nperm)
        s, n = pair_agree(pairs)
        o2, p2, mu2, lo2, hi2 = shuffle_test(pairs, lambda ps: pair_agree(ps)[0] / max(1, pair_agree(ps)[1]), nperm)
        P(f'  Wells {level}: {len(pairs)} tablets, MI = {o:.3f} vs null {mu:.3f}, P = {p:.4f}; identical-text pairs sharing image {s}/{n} = {s / max(1, n):.3f} vs null {mu2:.3f}, P = {p2:.4f}')
    # Wells seal baseline
    seals = [f for f in W if f['type'].startswith('SEAL') and f['site'] == 'Mohenjo-daro' and f['complete'] and wsym_seal(f['symbol'])]
    for level in ('seq_raw', 'seq_all'):
        pairs = [(tuple(f[level]), wsym_seal(f['symbol'])) for f in seals]
        cnt = collections.Counter(t for t, _ in pairs); dup = [(t, im) for t, im in pairs if cnt[t] >= 2]
        s, n = pair_agree(dup)
        o2, p2, mu2, lo2, hi2 = shuffle_test(dup, lambda ps: pair_agree(ps)[0] / max(1, pair_agree(ps)[1]), nperm)
        P(f'  Wells MD seals {level}: {len(seals)} seals with emblem, {len(dup)} with a duplicated text: pairs sharing emblem {s}/{n} = {s / max(1, n):.3f} vs null {mu2:.3f}, P = {p2:.4f}')
    # Harappa moulded tablets with an image (IM77 miniature tablets with fs) as a second baseline
    ht = [o for o in IM if o['ot'] == 'miniature tablet' and o['fs']]
    pairs = []
    for o in ht:
        ts = [t for t in o['texts'] if t['complete']]
        if ts: pairs.append((tuple(max(ts, key=lambda t: len(t['seq']))['seq']), o['cls']))
    if pairs:
        s, n = pair_agree(pairs)
        o2, p2, mu2, lo2, hi2 = shuffle_test(pairs, lambda ps: pair_agree(ps)[0] / max(1, pair_agree(ps)[1]), nperm)
        o3, p3, mu3, lo3, hi3 = shuffle_test(pairs, mi, nperm)
        P(f'  Harappa miniature tablets with a field symbol (IM77): {len(pairs)} tablets, images {collections.Counter(im for _, im in pairs).most_common(6)}; identical-text pairs sharing image {s}/{n} = {s / max(1, n):.3f} vs null {mu2:.3f}, P = {p2:.4f}; MI {o3:.3f} vs {mu3:.3f}, P = {p3:.4f}')
    return IM, W

# ====================================================================================================
def cycle2(nperm):
    P('# S-DARK-67.2  reading the bound texts against the frame; the same texts elsewhere')
    IM = load_im77(); W = load_wells()
    m2w, w2m, prop = m2w_map()
    # frame parser trained on all complete Wells texts (seq_all)
    allseq = [tuple(f['seq_all']) for f in W if f['complete'] and f['seq_all']]
    QUAL = build_qual(allseq)
    wc = wells_copper(W, 'seq_all')
    # distinct copper texts (Wells), with copies and image labels
    by_t = collections.defaultdict(list)
    for o in wc:
        for t in o['texts']:
            if t['complete'] and len(t['seq']) >= 1: by_t[tuple(t['seq'])].append(o)
    P(f'Wells copper: {len(by_t)} distinct complete texts (seq_all) on {sum(len(v) for v in by_t.values())} faces')
    # index of all non-copper faces by text and by closing unit
    others = [f for f in W if f['type'] != 'TAB:C' and f['complete'] and f['seq_all']]
    idx_exact = collections.defaultdict(list)
    idx_last2 = collections.defaultdict(list); idx_last3 = collections.defaultdict(list)
    for f in others:
        s = tuple(f['seq_all']); idx_exact[s].append(f)
        if len(s) >= 2: idx_last2[s[-2:]].append(f)
        if len(s) >= 3: idx_last3[s[-3:]].append(f)
    def otype(t): return t.split(':')[0]
    def emb(f): return wsym_seal(f['symbol']) or (f['symbol'] if f['symbol'] not in ('', '-', 'None') else '?')
    P('\n## each copper text: frame parse, class, copies, images, and occurrences elsewhere (Wells seq_all)')
    classes = collections.Counter(); summary = []
    for t, objs in sorted(by_t.items(), key=lambda kv: -len(kv[1])):
        lab = parse(list(t), QUAL)
        roles = set(lab)
        if roles <= {'OPENER', 'MARKER', 'CLOSER', 'SUFFIX'}: cls = 'frame-only'
        elif roles <= {'OPENER', 'MARKER', 'CLOSER', 'SUFFIX', 'TITLE', 'COUNT'}: cls = 'designation-only (title/count + closer)'
        elif 'CLOSER' not in roles and 'OPENER' not in roles: cls = 'name-only (no frame)'
        else: cls = 'full (name + frame)'
        classes[cls] += len(objs)
        ims = collections.Counter(o['cls'] or 'none' for o in objs)
        ex = idx_exact.get(t, [])
        l2 = idx_last2.get(t[-2:], []) if len(t) >= 2 else []
        l3 = idx_last3.get(t[-3:], []) if len(t) >= 3 else []
        P(f'  {fmt(t):42s} n={len(objs):2d} imgs={dict(ims)}')
        P(f'      parse: {" ".join(f"{s}:{l[:2]}" for s, l in zip(t, lab))}  -> {cls}')
        if ex: P(f'      EXACT elsewhere: {len(ex)}: ' + ', '.join(f'{f["cisi"]}({otype(f["type"])},{f["site"][:4]},{emb(f)})' for f in ex[:12]))
        else: P('      exact elsewhere: 0')
        if len(t) >= 3:
            P(f'      last-3 unit {fmt(t[-3:])} elsewhere: {len(l3)} ' + (f'types {dict(collections.Counter(otype(f["type"]) for f in l3))} emblems {dict(collections.Counter(emb(f) for f in l3))}' if l3 else ''))
        if len(t) >= 2:
            c2 = collections.Counter(emb(f) for f in l2 if otype(f['type']) == 'SEAL')
            P(f'      last-2 unit {fmt(t[-2:])} elsewhere: {len(l2)} types {dict(collections.Counter(otype(f["type"]) for f in l2))}; seal emblems {dict(c2.most_common(6))}')
        summary.append((t, len(objs), ims, cls, len(ex), len(l3), len(l2)))
    P('\n## class totals (faces): ' + ', '.join(f'{k}: {v}' for k, v in classes.most_common()))
    P('## distinct texts: ' + ', '.join(f'{k}: {v}' for k, v in collections.Counter(s[3] for s in summary).most_common()))
    # how many copper texts recur exactly on seals / other tablets?
    nex = sum(1 for s in summary if s[4] > 0)
    P(f'## copper texts found exactly on a non-copper object: {nex}/{len(summary)}; with last-3 unit elsewhere: {sum(1 for s in summary if s[5] > 0)}/{sum(1 for s in summary if len(s[0]) >= 3)}; last-2 unit elsewhere: {sum(1 for s in summary if s[6] > 0)}/{sum(1 for s in summary if len(s[0]) >= 2)}')
    # control: random seal texts of the same lengths - how often do they recur exactly elsewhere, and their closing unit?
    seals_md = [tuple(f['seq_all']) for f in others if otype(f['type']) == 'SEAL' and f['site'] == 'Mohenjo-daro']
    cnt_all = collections.Counter(tuple(f['seq_all']) for f in others)
    bylen = collections.defaultdict(list)
    for s in seals_md: bylen[len(s)].append(s)
    rec = []
    for _ in range(nperm):
        k = 0
        for t, n, ims, cls, ex, l3, l2 in summary:
            pool = bylen.get(len(t)) or seals_md
            s = rnd.choice(pool)
            k += (cnt_all[s] - 1 > 0)
        rec.append(k)
    rec.sort()
    P(f'## control: MD seal texts of the same lengths recur exactly elsewhere {sum(rec) / len(rec):.1f} of {len(summary)} [{rec[int(0.025 * nperm)]},{rec[int(0.975 * nperm) - 1]}] (copper: {nex})')
    # IM77 side: exact M-sequence matches of copper texts on seals / sealings / tablets with their field symbols
    P('\n## IM77: copper texts found exactly on other objects, with the field symbol there')
    cu = [o for o in IM if o['ot'] == 'copper tablet']
    cutexts = collections.Counter(tuple(t['seq']) for o in cu for t in o['texts'] if t['complete'])
    idx = collections.defaultdict(list)
    for o in IM:
        if o['ot'] == 'copper tablet': continue
        for t in o['texts']:
            if t['complete']: idx[tuple(t['seq'])].append(o)
    nm = 0
    for t, n in cutexts.most_common():
        hits = idx.get(t, [])
        if hits:
            nm += 1
            P(f'  {fmt(t):40s} copper x{n}: elsewhere {len(hits)}: ' + ', '.join(f'{o["id"]}({o["ot"][:6]},{o["site"][:5]},{o["cls"] or "no-fs"})' for o in hits[:10]))
    P(f'  IM77: {nm}/{len(cutexts)} distinct copper texts recur exactly on a non-copper object')
    # closing-unit (last 2 M signs) emblem distribution on IM77 seals vs copper image
    P('\n## IM77: for each copper text, the seal emblems of seals ending in the same last-2 unit (is the copper image over-represented?)')
    seals = [o for o in IM if o['ot'] == 'seal' and o['fs']]
    seal_items = []
    for o in seals:
        ts = [t for t in o['texts'] if t['complete']]
        if ts: seal_items.append((tuple(max(ts, key=lambda t: len(t['seq']))['seq']), o['cls']))
    base = collections.Counter(c for _, c in seal_items)
    cu_items = []
    for o in cu:
        if not o['cls']: continue
        for t in o['texts']:
            if t['complete'] and len(t['seq']) >= 2: cu_items.append((tuple(t['seq']), o['cls']))
    seen = set(); tot_obs = 0; tot_exp = 0.0
    for t, c in sorted(set(cu_items), key=lambda x: x[0]):
        if (t[-2:], c) in seen: continue
        seen.add((t[-2:], c))
        hits = [cl for s, cl in seal_items if s[-2:] == t[-2:]]
        if not hits: continue
        k = sum(1 for cl in hits if cl == c); exp = len(hits) * base[c] / max(1, sum(base.values()))
        tot_obs += k; tot_exp += exp
        P(f'  unit {fmt(t[-2:]):10s} copper image {c:15s}: seals ending so {len(hits):3d}, with that emblem {k} (expected {exp:.1f}); top emblems {collections.Counter(hits).most_common(3)}')
    P(f'  total: copper-image emblem among same-unit seals {tot_obs} vs expected {tot_exp:.1f} (base rates)')
    return summary

# ====================================================================================================
def cycle3(nperm):
    P('# S-DARK-67.3  the image set vs the seal emblem set; copper text vs emblem-specific seal statistics')
    IM = load_im77(); W = load_wells()
    cu = [o for o in IM if o['ot'] == 'copper tablet']
    seals = [o for o in IM if o['ot'] == 'seal' and o['fs']]
    cu_cls = collections.Counter(o['cls'] for o in cu if o['cls'])
    seal_cls = collections.Counter(o['cls'] for o in seals if o['cls'])
    P('## IM77 copper image classes: ' + ', '.join(f'{k} {v}' for k, v in cu_cls.most_common()))
    P('## IM77 seal emblem classes: ' + ', '.join(f'{k} {v}' for k, v in seal_cls.most_common()))
    shared = [c for c in cu_cls if c in seal_cls]
    P(f'## shared classes: {shared}; copper-only: {[c for c in cu_cls if c not in seal_cls]}; '
      f'seal classes absent from copper: {[c for c in seal_cls if c not in cu_cls and seal_cls[c] >= 5]}')
    P(f'## copper tablets whose image class is a seal emblem class: {sum(cu_cls[c] for c in shared)}/{sum(cu_cls.values())}; '
      f'unicorn share: copper {cu_cls.get("unicorn", 0)}/{sum(cu_cls.values())} vs seals {seal_cls.get("unicorn", 0)}/{sum(seal_cls.values())}')
    # Wells side
    wc = wells_copper(W, 'seq_raw')
    wsy = collections.Counter(o['cls'] for o in wc if o['cls'])
    wse = collections.Counter(wsym_seal(f['symbol']) for f in W if f['type'].startswith('SEAL') and wsym_seal(f['symbol']))
    P('## Wells copper labels: ' + ', '.join(f'{k} {v}' for k, v in wsy.most_common()))
    P('## Wells seal emblems: ' + ', '.join(f'{k} {v}' for k, v in wse.most_common(14)))
    # per shared class: copper text(s) vs emblem-specific seal statistics
    M_HEAD = {342: 'jar', 343: 'mjar', 344: 'mjar', 345: 'mjar', 211: 'arrow', 12: 'W151', 15: 'W156', 254: 'box527', 60: 'W226',
              245: 'W615/617', 66: 'W236', 328: 'W700', 252: 'W595', 169: 'tree169'}
    seal_items = []
    for o in seals:
        ts = [t for t in o['texts'] if t['complete']]
        if ts and o['cls']: seal_items.append((tuple(max(ts, key=lambda t: len(t['seq']))['seq']), o['cls']))
    P(f'\n## emblem-specific seal statistics ({len(seal_items)} IM77 seals with complete text + emblem); null = emblem labels shuffled {nperm}x')
    def head(s): return M_HEAD.get(s[-1], 'other')
    cu_by_cls = collections.defaultdict(collections.Counter)
    for o in cu:
        if not o['cls']: continue
        for t in o['texts']:
            if t['complete'] and len(t['seq']) >= 2: cu_by_cls[o['cls']][tuple(t['seq'])] += 1
    results = []
    for c in sorted(cu_by_cls, key=lambda c: -sum(cu_by_cls[c].values())):
        ns = sum(1 for _, cl in seal_items if cl == c)
        for t, n in cu_by_cls[c].most_common():
            h = head(t); u2 = t[-2:]; sset = set(t)
            def stat(items, c=c, h=h, u2=u2, sset=sset):
                grp = [s for s, cl in items if cl == c]
                if not grp: return (0, 0, 0)
                fh = sum(1 for s in grp if head(s) == h) / len(grp)
                fu = sum(1 for s in grp if s[-2:] == u2) / len(grp)
                fj = sum(len(sset & set(s)) / len(sset | set(s)) for s in grp) / len(grp)
                return (fh, fu, fj)
            obs = stat(seal_items)
            texts = [s for s, _ in seal_items]; labs = [cl for _, cl in seal_items]
            ge = [0, 0, 0]; mus = [0.0, 0.0, 0.0]
            for _ in range(nperm):
                rnd.shuffle(labs)
                v = stat(list(zip(texts, labs)))
                for k in range(3): ge[k] += (v[k] >= obs[k]); mus[k] += v[k] / nperm
            ps = [(g + 1) / (nperm + 1) for g in ge]
            P(f'  {c:15s} seals={ns:4d} copper text {fmt(t):38s} x{n}: head {h:9s} on {c} seals {obs[0]:.2f} (null {mus[0]:.2f}, P={ps[0]:.3f}); '
              f'last-2 {fmt(u2)} {obs[1]:.3f} (null {mus[1]:.3f}, P={ps[1]:.3f}); sign-set Jaccard {obs[2]:.3f} (null {mus[2]:.3f}, P={ps[2]:.3f})')
            results.append((c, t, n, ns, obs, ps))
    # aggregate over shared animals with >= 5 seals: how many of the 3 statistics beat the null at P < 0.05, vs expected
    k = sum(1 for r in results if r[3] >= 5 for p in r[5] if p < 0.05); m = sum(3 for r in results if r[3] >= 5)
    P(f'## aggregate: {k}/{m} statistics at P < 0.05 among copper texts whose image class has >= 5 seals (expected {0.05 * m:.1f})')
    # does the copper text for animal X occur (exactly or by last-2 unit) on X-seals more than on other seals? pooled
    tot_on = tot_off = n_on = n_off = 0
    for c, t, n, ns, obs, ps in results:
        if ns < 5: continue
        on = [s for s, cl in seal_items if cl == c]; off = [s for s, cl in seal_items if cl != c]
        tot_on += sum(1 for s in on if s[-2:] == t[-2:]); n_on += len(on)
        tot_off += sum(1 for s in off if s[-2:] == t[-2:]); n_off += len(off)
    P(f'## pooled last-2 unit: on same-emblem seals {tot_on}/{n_on} = {tot_on / max(1, n_on):.3f}; on other seals {tot_off}/{n_off} = {tot_off / max(1, n_off):.3f}')
    # Wells: W930/M393 style check - any sign of the copper text enriched on seals of that emblem? (Wells symbols)
    P('\n## Wells seals: for the Wells-labelled copper classes (hare, elephant, rhino, tiger, goat, buffalo, personage), copper-text signs on seals with that emblem vs others')
    wseals = [(tuple(f['seq_all']), wsym_seal(f['symbol'])) for f in W if f['type'].startswith('SEAL') and f['complete'] and wsym_seal(f['symbol'])]
    wc = wells_copper(W, 'seq_all')
    wcu = collections.defaultdict(collections.Counter)
    for o in wc:
        if o['cls'] and o['cls'] != 'Othr' and not str(o['cls']).startswith('Wsign'):
            for t in o['texts']:
                if t['complete'] and len(t['seq']) >= 2: wcu[o['cls']][tuple(t['seq'])] += 1
    for c in wcu:
        on = [s for s, cl in wseals if cl == c]; off = [s for s, cl in wseals if cl != c]
        if len(on) < 3: P(f'  {c}: only {len(on)} Wells seals with that emblem'); continue
        for t, n in wcu[c].most_common(3):
            sset = set(t)
            fon = sum(len(sset & set(s)) for s in on) / len(on); foff = sum(len(sset & set(s)) for s in off) / len(off)
            labs = [cl for _, cl in wseals]; texts = [s for s, _ in wseals]; ge = 0
            for _ in range(nperm):
                rnd.shuffle(labs)
                on2 = [s for s, cl in zip(texts, labs) if cl == c]
                ge += (sum(len(sset & set(s)) for s in on2) / len(on2) >= fon)
            P(f'  {c:14s} ({len(on)} seals) copper {fmt(t):36s} x{n}: shared signs per seal {fon:.2f} on {c} seals vs {foff:.2f} others, P = {(ge + 1) / (nperm + 1):.3f}')

# ====================================================================================================
def cycle4(nperm):
    P('# S-DARK-67.4  Wells vs IM77 on the same objects, merge levels, leave-one-die-out prediction')
    IM = load_im77(); W = load_wells()
    pairs = json.load(open(DARK + 'loop24_pairs.json'))
    cu = {o['id']: o for o in IM if o['ot'] == 'copper tablet'}
    wc_raw = {o['cisi']: o for o in wells_copper(W, 'seq_raw')}
    # alignment via loop24 pairs (text matched; accept 'ok' and 'symbol differ')
    agree = disagree = unk = 0; rows = []
    for p in pairs:
        if p['wells']['type'] != 'TAB:C' or p['reason'] not in ('ok', 'symbol differ'): continue
        o = cu.get(p['text_no']); w = wc_raw.get(p['cisi'])
        if not o or not w: continue
        wl = w['cls']; il = o['cls']
        if not wl or not il or wl == 'Othr' or str(wl).startswith('Wsign'): unk += 1; rows.append((p['cisi'], p['text_no'], wl, il, 'unk')); continue
        same = (wl == il) or (wl == 'gaur' and il in ('ox-antelope',)) or (wl == 'bull' and 'bull' in il) or (wl == 'composite' and il in ('two-headed', 'composite'))
        agree += same; disagree += (not same); rows.append((p['cisi'], p['text_no'], wl, il, 'same' if same else 'DIFF'))
    P(f'## aligned copper objects (loop24 pairs, text-matched): {len(rows)}; both labelled: {agree + disagree}; agree {agree}, disagree {disagree}; one side unlabelled/Othr/sign-image {unk}')
    for r in rows:
        if r[4] == 'DIFF': P(f'    {r[0]} <-> IM77 {r[1]}: Wells {r[2]} vs IM77 {r[3]}')
    P('  Wells image-sign faces <-> IM77 field symbols on aligned objects: ' + '; '.join(f'{r[2]}={r[3]}' for r in rows if str(r[2]).startswith('Wsign') and r[3]))
    P('  Wells "Othr" <-> IM77: ' + str(collections.Counter(r[3] for r in rows if r[2] == 'Othr')))
    # merge levels: dies and binding at each level (Wells)
    P('\n## merge levels (Wells): distinct texts (dies) among complete copper faces and binding')
    for level in ('seq_raw', 'seq_strong', 'seq_all'):
        wc = wells_copper(W, level)
        texts = [tuple(t['seq']) for o in wc for t in o['texts'] if t['complete']]
        it = []
        for o in wc:
            if not o['img'] or o['cls'] == 'Othr': continue
            ts = [t for t in o['texts'] if t['complete']]
            if ts: it.append((tuple(max(ts, key=lambda t: len(t['seq']))['seq']), o['cls']))
        o1, p1, mu1, lo1, hi1 = shuffle_test(it, mi, nperm)
        by_t = collections.defaultdict(set); by_i = collections.defaultdict(set)
        for t, im in it: by_t[t].add(im); by_i[im].add(t)
        P(f'  {level}: {len(texts)} complete faces, {len(set(texts))} distinct texts; labelled {len(it)}: {len(by_t)} texts x {len(by_i)} images, {len(set(it))} pairs; '
          f'texts>1 image {sum(1 for v in by_t.values() if len(v) > 1)}, images>1 text {sum(1 for v in by_i.values() if len(v) > 1)}; MI {o1:.3f} vs null {mu1:.3f} P={p1:.4f}')
    # leave-one-die-out prediction (IM77 and Wells): die = distinct raw text string with its image
    def ed1(a, b):
        if a == b: return True
        if abs(len(a) - len(b)) > 1: return False
        if len(a) == len(b): return sum(x != y for x, y in zip(a, b)) == 1
        s, l = (a, b) if len(a) < len(b) else (b, a)
        for k in range(len(l)):
            if l[:k] + l[k + 1:] == s: return True
        return False
    def loo(dies, label):
        """dies: list of (text, image, copies). predict text from image using the other dies (copy-weighted majority)."""
        hits = collections.Counter(); n = 0; pred_table = {}
        for i, (t, im, n_i) in enumerate(dies):
            others = [(t2, im2, n2) for j, (t2, im2, n2) in enumerate(dies) if j != i and im2 == im]
            n += 1
            if not others: hits['no-other-die-with-image'] += 1; continue
            cnt = collections.Counter()
            for t2, im2, n2 in others: cnt[t2] += n2
            pred = cnt.most_common(1)[0][0]
            hits['exact'] += (pred == t); hits['ed<=1'] += ed1(pred, t)
            hits['same head (last sign)'] += (pred[-1] == t[-1]); hits['same last-2'] += (pred[-2:] == t[-2:])
            hits['any other die shares last-2'] += any(t2[-2:] == t[-2:] for t2, _, _ in others)
        P(f'  {label}: {n} dies; ' + ', '.join(f'{k} {v}/{n} = {v / n:.2f}' for k, v in hits.items()))
        return hits, n
    def loo_null(dies, label, nperm):
        ims = [im for _, im, _ in dies]; acc = collections.Counter()
        for _ in range(nperm):
            rnd.shuffle(ims)
            d2 = [(t, im, n) for (t, _, n), im in zip(dies, ims)]
            for i, (t, im, n_i) in enumerate(d2):
                others = [(t2, n2) for j, (t2, im2, n2) in enumerate(d2) if j != i and im2 == im]
                if not others: continue
                cnt = collections.Counter()
                for t2, n2 in others: cnt[t2] += n2
                pred = cnt.most_common(1)[0][0]
                acc['exact'] += (pred == t); acc['ed<=1'] += ed1(pred, t); acc['same head (last sign)'] += (pred[-1] == t[-1]); acc['same last-2'] += (pred[-2:] == t[-2:])
        P(f'  {label} null (images shuffled among dies, {nperm}x): ' + ', '.join(f'{k} {v / nperm:.2f}/{len(dies)}' for k, v in acc.items()))
    for label, src in (('IM77 fine fs80', 'imf'), ('IM77 class', 'imc'), ('Wells class (Othr excluded)', 'w')):
        dcount = collections.Counter()
        if src.startswith('im'):
            for o in cu.values():
                if not o['img']: continue
                for t in o['texts']:
                    if t['complete'] and len(t['seq']) >= 2: dcount[(tuple(t['seq']), o['img'] if src == 'imf' else o['cls'])] += 1
        else:
            for o in wells_copper(W, 'seq_raw'):
                if not o['img'] or o['cls'] == 'Othr': continue
                for t in o['texts']:
                    if t['complete'] and len(t['seq']) >= 2: dcount[(tuple(t['seq']), o['cls'])] += 1
        dies = [(t, im, n) for (t, im), n in dcount.items()]
        loo(dies, label); loo_null(dies, label, min(nperm, 200))
    # the prediction table image -> text (IM77 class, copy-weighted)
    P('\n## PREDICTION for a newly found Mohenjo-daro copper tablet (IM77 numbering; Wells in brackets where aligned)')
    by_i = collections.defaultdict(collections.Counter)
    for o in cu.values():
        if not o['cls']: continue
        for t in o['texts']:
            if t['complete'] and len(t['seq']) >= 2: by_i[o['cls']][tuple(t['seq'])] += 1
    for im, c in sorted(by_i.items(), key=lambda kv: -sum(kv[1].values())):
        tot = sum(c.values()); top, n = c.most_common(1)[0]
        P(f'  image {im:15s} ({tot:2d} tablets, {len(c)} dies): predict text {fmt(top):40s} (holds for {n}/{tot} = {n / tot:.2f} of known tablets; other dies: {", ".join(fmt(t) for t, _ in c.most_common()[1:])})')
    # overall hit rate if a new tablet is a copy (tablet-level, leave-one-tablet-out) vs a new die
    tabs = [(tuple(max([t for t in o['texts'] if t['complete']], key=lambda t: len(t['seq']))['seq']), o['cls']) for o in cu.values()
            if o['cls'] and any(t['complete'] for t in o['texts'])]
    hit = 0
    for i, (t, im) in enumerate(tabs):
        cnt = collections.Counter(t2 for j, (t2, im2) in enumerate(tabs) if j != i and im2 == im)
        hit += bool(cnt) and cnt.most_common(1)[0][0] == t
    P(f'  leave-one-TABLET-out exact hit (a new tablet that is another copy): {hit}/{len(tabs)} = {hit / len(tabs):.2f}')

# ====================================================================================================
if __name__ == '__main__':
    {1: cycle1, 2: cycle2, 3: cycle3, 4: cycle4}[CY](NP)
    open(DARK + f'loop67_cycle{CY}_log.txt', 'w').write('\n'.join(LOG) + '\n')
