"""Loop 73 common: Proto-Elamite entry middles and header designations as a calibration for the Indus middle.

PE entry = [optional prefix] + middle + [final class sign] + numeral (other-scripts/proto-elamite FINDINGS test a).
Frame sets are FIXED from test a (res_a_slots.json, z >= 3, computed before this loop):
  PREFIX = initial-biased signs, CLASS = final-biased signs.
Middle = clean entry (no x, no lacuna) with leading PREFIX signs (up to 2) and one trailing CLASS sign removed.
Header designation = signs of the lines before the first numeral line, minus the leading opener sign
(the commonest header-initial signs: M157, M327 family, M136, ...).
Indus middle = S-DARK-26 / 56 definition (tools/dark_loop56.py: NAME + COUNT residue of the S310/S331 parser).
"""
import sys, os, json, collections, math, random, re
ROOT = '/home/user/Indus-/'
sys.path.insert(0, ROOT + 'other-scripts/proto-elamite/tools')
sys.path.insert(0, ROOT + 'tools')
os.chdir(ROOT)
import common as PEC
_argv = sys.argv; sys.argv = ['x', '0']
import dark_loop56 as D56
sys.argv = _argv
DARK = ROOT + 'data/derived/dark/'

SLOTS = json.load(open(ROOT + 'other-scripts/proto-elamite/data/res_a_slots.json'))['rows']
PREFIX = {r['sign'] for r in SLOTS if r['z_init'] >= 3}
CLASS = {r['sign'] for r in SLOTS if r['z_final'] >= 3}
OPENERS = {'M157', '|M327+M342|', 'M327', 'M136', '|M327+X|', '|M136+M365|', '|M377+M320+M377|', 'M005', 'M305', 'M247'}

LOG = []
def P(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); LOG.append(s)

def numtok(nums):
    return 'NUM:' + '+'.join(f'{n}{c}' for n, c in nums)

def pe_entries():
    T = PEC.load()
    out = []
    for e in PEC.entries(T, require_clean=True):
        s = list(e['signs'])
        pre = []
        while len(s) > 1 and s[0] in PREFIX and len(pre) < 2:
            pre.append(s.pop(0))
        cls = None
        if s and s[-1] in CLASS:
            cls = s.pop()
        e2 = dict(tablet=e['tablet'], prov=e['prov'], signs=tuple(e['signs']), prefix=tuple(pre), mid=tuple(s), cls=cls,
                  num=numtok(e['numerals']), system=e['system'])
        out.append(e2)
    return out

def pe_tablet_lines():
    """tablet -> list of entries in order (clean and unclean marked) for within-tablet tests."""
    by = collections.defaultdict(list)
    for e in pe_entries(): by[e['tablet']].append(e)
    return by

def pe_headers():
    T = PEC.load(); out = []
    for t in T:
        pre = []
        for l in t['lines']:
            if l['numerals']: break
            pre.append(l)
        if not pre: continue
        sg = [PEC.base(x) for l in pre for x in l['signs'] if PEC.is_sign(x)]
        if not sg or any(l['lacuna'] for l in pre) or any(x == 'x' for l in pre for x in l['signs']): continue
        op = sg[0] if sg[0] in OPENERS else None
        des = tuple(sg[1:]) if op else tuple(sg)
        out.append(dict(tablet=t['id'], prov=t['provenience'], opener=op, des=des, full=tuple(sg)))
    return out

# ---------------- Indus ----------------
def indus_objs(LV):
    return D56.load_indus(LV) if LV != 'im77' else D56.im77_objects()

def indus_parts(o):
    """prefix (opener+marker), mid (NAME+COUNT), head (TITLE+CLOSER+SUFFIX) from the parse labels."""
    pre = tuple(a for a, l in zip(o['seq'], o['lab']) if l in ('OPENER', 'MARKER'))
    head = tuple(a for a, l in zip(o['seq'], o['lab']) if l in ('TITLE', 'CLOSER', 'SUFFIX'))
    return pre, o['mid'], head

# ---------------- comparators ----------------
def jl(path): return [tuple(json.loads(l)['seq']) for l in open(path)]
def ur3_names_dedup(): return [s for s in jl(DARK + 'loop56_corpora/ur3_names_dedup.jsonl') if s and s[0] != '($']
def linb_names_dedup(): return [s for s in jl(DARK + 'loop56_corpora/linb_personnel_dedup.jsonl')]
UR3_TITLE = {'dumu','dub-sar','lugal','arad2','arad2-zu','arad','lu2','ensi2','ugula','nu-banda3','gudu4','dam-gar3','sanga','sukkal','szabra','agrig','sipa','lunga','simug','aszgab','nagar','kuruszda','szagina','ra2-gab','gal5-la2','muhaldim','kiszib3','dam','szesz','nin','ama','ab-ba','lu2-kin-gi4-a','i3-du8','gudu4-abzu','ensi','sagi','gal','nar','azlag2','ma2-lah5','szu-i','ad-kup4','asz-gab','bahar2','szitim','engar','mu'}
def ur3_names_tokens():
    """one per distinct legend (loop32 ur3_words): owner name = first word, as syllable tuple (repeats across legends kept)."""
    out = []
    for s in jl(DARK + 'loop32_corpora/ur3_words.jsonl'):
        w = s[0]
        if w in UR3_TITLE or w.startswith('_') or 'x' in w.split('-') or '...' in w or '$' in w: continue
        t = tuple(x for x in w.split('-') if x)
        if t: out.append(t)
    return out
def ur3_legends_labeled():
    """legend = name word (NAME syllables) + other words (TITLE syllables)."""
    out = []
    for s in jl(DARK + 'loop32_corpora/ur3_words.jsonl'):
        if any('x' in w.split('-') or '...' in w or '$' in w for w in s): continue
        if s[0] in UR3_TITLE: continue
        toks = []; labs = []
        for i, w in enumerate(s):
            for x in w.replace('_', '').split('-'):
                if x: toks.append(x); labs.append('DES' if i == 0 else 'FRAME')
        if toks: out.append((tuple(toks), tuple(labs)))
    return out

def linb_lines():
    """DAMOS personnel-series lines: (tablet, name tuple, frame tokens) where the line starts with a sign group followed by VIR/MUL or an ideogram + number."""
    items = [json.loads(l) for l in open(ROOT + 'other-scripts/linear-a/data/damos_items.jsonl')]
    SER = ('As','B','An','Jn','Ap','Ad','Ae','Cn','Da','Db','Dc','Dd','De','Df','Dg','Dk','Dl','Dm','Dn','Dq','Dv','Ea','Eb','En','Eo','Ep','Es','V','Vc','Ak','Ai','Na','Nn','Ma','Sc','Xd','Ce','Vd','D','Dh')
    out = []
    for it in items:
        h = it.get('heading', '')
        m = re.match(r'(KN|PY|TH|MY)\s+([A-Z][a-z]?)(?:\(\d+\))?\s', h)
        if not m or m.group(2) not in SER: continue
        site = m.group(1)
        for ln in it['content'].split('\n'):
            toks = [t.strip(",'/") for t in ln.split()]
            toks = [t for t in toks if t and not re.match(r'^\.[0-9AaBbv]+[ab]?$', t)]
            if not toks or any(ch in ''.join(toks) for ch in '[]?') or re.search('[\u0323]', ''.join(toks)): continue
            if not (re.fullmatch(r'[a-z0-9*-]+', toks[0]) and '-' in toks[0]): continue
            name = tuple(toks[0].split('-'))
            frame = []
            for t in toks[1:]:
                if re.fullmatch(r'[0-9]+', t): frame.append('NUM')
                elif re.fullmatch(r'[a-z0-9*-]+', t): frame += t.split('-')
                elif re.fullmatch(r'[A-Z]+[0-9]*', t): frame.append(t)
            if any(f in ('VIR', 'MUL') or f.isupper() for f in frame) and 'NUM' in frame:
                out.append(dict(tablet=h.split('(')[0].strip(), site=site, name=name, frame=tuple(frame)))
    return out

# ---------------- statistics ----------------
H = D56.H; gini = D56.gini; zipf = D56.zipf
def heaps_ids(texts, r, norders=10):
    n = len(texts); grid = sorted(set(int(round(n * 0.05 * 1.25 ** i)) for i in range(40) if n * 0.05 * 1.25 ** i <= n) | {n})
    acc = collections.defaultdict(float)
    for _ in range(norders):
        idx = list(range(n)); r.shuffle(idx); seen = set(); gi = 0; m = 0
        for i in idx:
            seen.add(texts[i]); m += 1
            if gi < len(grid) and m == grid[gi]: acc[m] += len(seen); gi += 1
    xs = [math.log(m) for m in grid if m >= max(10, n * 0.05)]; ys = [math.log(acc[m] / norders) for m in grid if m >= max(10, n * 0.05)]
    if len(xs) < 3: return float('nan')
    mx = sum(xs) / len(xs); my = sum(ys) / len(ys)
    return sum((a - mx) * (b - my) for a, b in zip(xs, ys)) / sum((a - mx) ** 2 for a in xs)

def freq_metrics(texts, r):
    """S-DARK-53 identifier-frequency statistics on a token list of designations (repeats kept)."""
    cnt = collections.Counter(texts); n = len(texts); D = len(cnt)
    f1 = sum(1 for v in cnt.values() if v == 1); f2 = sum(1 for v in cnt.values() if v == 2)
    ch = D + (f1 * f1 / (2 * f2) if f2 else f1 * (f1 - 1) / 2)
    return dict(uniq=D / n, gini_id=gini(list(cnt.values())), zipf_id=zipf(list(cnt.values())), heaps_id=heaps_ids(texts, r),
                gt_new=f1 / n, chao_ratio=ch / n, top10=sum(v for _, v in cnt.most_common(10)) / n)

def edge_metrics(names, r, nrand=20):
    """S-DARK-56.3d closed-slot test on deduplicated names (>= 2 elements): top-10 initial / final coverage minus
    frequency-matched random strings (iid from the element unigram, same lengths); normalised final entropy."""
    def cov(ns):
        ini = collections.Counter(n[0] for n in ns); fin = collections.Counter(n[-1] for n in ns); N = len(ns)
        hf = H(fin) / math.log2(len(fin)) if len(fin) > 1 else 0
        return sum(v for _, v in ini.most_common(10)) / N, sum(v for _, v in fin.most_common(10)) / N, hf
    i0, f0, h0 = cov(names)
    pool = [a for n in names for a in n]; rs = []
    for _ in range(nrand):
        rs.append(cov([tuple(r.choice(pool) for _ in range(len(n))) for n in names]))
    ri = sum(x[0] for x in rs) / nrand; rf = sum(x[1] for x in rs) / nrand; rh = sum(x[2] for x in rs) / nrand
    return dict(top10_init=i0, top10_fin=f0, init_ex=i0 - ri, fin_ex=f0 - rf, fin_H=h0, fin_H_rand=rh)

def bind_metrics(names, r, nperm=200):
    """S-DARK-61.2 slot binding inside the designation: penultimate -> final element; MI excess (bits and share of
    H(final)) over finals permuted among names; loyalty of penultimates (>= 5 tokens) to one final (>= 80%)."""
    pairs = [(n[-2], n[-1]) for n in names if len(n) >= 2]
    def mi(pp):
        a = collections.Counter(x for x, _ in pp); b = collections.Counter(y for _, y in pp); j = collections.Counter(pp); N = len(pp)
        return sum(v / N * math.log2(v * N / (a[x] * b[y])) for (x, y), v in j.items())
    def loyal(pp):
        by = collections.defaultdict(collections.Counter)
        for x, y in pp: by[x][y] += 1
        el = [c for c in by.values() if sum(c.values()) >= 5]
        return sum(1 for c in el if c.most_common(1)[0][1] / sum(c.values()) >= 0.8) / len(el) if el else float('nan')
    o = mi(pairs); lo = loyal(pairs); ys = [y for _, y in pairs]; nm = []; nl = []
    for _ in range(nperm):
        r.shuffle(ys); pp = [(x, y) for (x, _), y in zip(pairs, ys)]; nm.append(mi(pp)); nl.append(loyal(pp))
    hf = H(collections.Counter(y for _, y in pairs))
    mu = sum(nm) / nperm
    nlv = [v for v in nl if not math.isnan(v)]
    return dict(bind_mi_ex=o - mu, bind_share=(o - mu) / hf if hf else float('nan'), loyal=lo, loyal_null=sum(nlv) / len(nlv) if nlv else float('nan'))

def q(v, p):
    v = sorted(x for x in v if not (isinstance(x, float) and math.isnan(x)))
    return v[min(len(v) - 1, int(p * len(v)))] if v else float('nan')

def save(name, obj):
    json.dump(obj, open(DARK + name + '.json', 'w'), indent=1, default=str)
    open(DARK + name + '_log.txt', 'w').write('\n'.join(LOG) + '\n')

# Linear B personnel names WITH tablet id (same rules as tools/dark_loop56_prep.py linb_names, which drops the tablet)
import unicodedata
_LB_FIRST = {'Da','Db','Dc','Dd','De','Df','Dg','Dh','Dk','Dl','Dm','Dn','Dp','Dq','Dv','D','Cn','Ea','Eb','En','Eo','Ep','Es','Sc','Jn','Nn','Ma','Na','Nc','Ne','Ng'}
_LB_LIST = {'As','B','An','Ap','Ai','Ak','Ad','Ae','V','Vc','Vd','Xd','Ce'}
_LB_OCC = {'do-e-ro','do-e-ra','ka-ke-u','te-ko-to','ku-wa','ko-wa','ko-wo','pe-di-ra','ka-ko','ta-ra-si-ja','ko-to-na','ki-ti-me-na','ke-ke-me-na','o-na-to','e-ke','to-so','to-sa','pa-ro','o-pe-ro','a-pu-do-si','me-no','e-ke-qe','e-ke-si','o-na-te-re','te-re-ta','ka-ma','ki-ti-je-si','pe-ma','o-pe-ro-sa','to-so-de','a-ke-ro','ra-wa-ke-ta','e-qe-ta','i-je-re-ja','i-je-re-u','ku-ru-so'}
def _lb_ok(w):
    if not re.fullmatch(r'[a-z0-9-]+', w): return False
    if '-' not in w and len(w) < 2: return False
    return w not in ('vac','vacat','vest','lat','inf','sup','mut','deest')
def _strip_diac(w): return ''.join(c for c in unicodedata.normalize('NFD', w) if unicodedata.category(c) != 'Mn')
def linb_names_tab():
    items = [json.loads(l) for l in open(ROOT + 'other-scripts/linear-a/data/damos_items.jsonl')]
    out = []
    for it in items:
        h = it.get('heading', '')
        m = re.match(r'(KN|PY|TH|MY|TI)\s+([A-Z][a-z]?)(?:\(\d+\))?\s', h)
        if not m: continue
        site, ser = m.group(1), m.group(2)
        if ser not in _LB_FIRST and ser not in _LB_LIST: continue
        for ln in it['content'].split('\n'):
            ln = _strip_diac(ln)
            if not re.match(r'^\.?[0-9AaBb]', ln.strip()) and not ln.startswith(' '): continue
            toks = [t for t in ln.split() if not re.match(r'^\.?[0-9AaBbv]+[ab]?$', t) and t not in (',', '/', "'", '[', ']')]
            words = [t.strip(",'/") for t in toks]
            words = [w for w in words if _lb_ok(w) and w not in _LB_OCC and not re.fullmatch(r'[0-9]+', w)]
            if not words: continue
            if ser in _LB_FIRST: words = words[:1]
            for w in words:
                if '-' in w and len(w.split('-')) <= 7:
                    out.append(dict(tablet=h.split('(')[0].strip() if '(' in h else h, site=site, series=ser, name=tuple(w.split('-'))))
    return out
