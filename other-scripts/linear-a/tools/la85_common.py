#!/usr/bin/env python3
"""la85 common: value-free document forms for Linear A (corpus_ra_v2, 'rd'), Linear B (DAMOS: KN, PY, TH)
and Ur III (CDLI, via la57_common._cdli), a library of value-free document features, and random statistics.

Value-free: a document is a list of lines; a line is a list of tokens
  ('W', key, nsigns)   syllabic sign-group (key is opaque; only used for repetition inside the corpus)
  ('G', key)           logogram / ideogram-like token (class only; identity never compared across systems)
  ('N', value, frac)   number (frac = carries a fraction or a sub-unit)
No sign value, no logogram identity and no reading crosses from one system to another.
"""
import json, os, re, sys, math, random, hashlib, unicodedata
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
D = os.path.join(HERE, '..', 'data')
CK = os.path.join(D, 'la85_ckpt')
os.makedirs(CK, exist_ok=True)

LA_GROUP = {'Haghia Triada': 'HT', 'Khania': 'KH', 'Zakros': 'ZA', 'Phaistos': 'PH'}


def seed(name):
    return int(hashlib.sha256(name.encode()).hexdigest()[:8], 16)


def la_docs(min_lines=2):
    import la83_parse
    out = []
    for ins in la83_parse.load('rd'):
        if ins.get('superseded_by') or ins['support'] not in ('Tablet', 'Lames (short thin tablet)'):
            continue
        lines, cur = [], []
        for t in ins['tokens']:
            if t['t'] == 'word':
                cur.append(('W', '-'.join(t['s']), len(t['s'])))
            elif t['t'] == 'logo':
                cur.append(('G', t['v']))
            elif t['t'] == 'num':
                cur.append(('N', float(t['v']), bool(t['frac'])))
            elif t['t'] == 'nl':
                if cur:
                    lines.append(cur); cur = []
        if cur:
            lines.append(cur)
        if len(lines) >= min_lines and any(x[0] == 'N' for l in lines for x in l):
            out.append({'id': ins['id'], 'sys': 'LA', 'site': ins['site'],
                        'grp': LA_GROUP.get(ins['site'], 'OT'), 'lines': lines})
    return out


def _strip(t):
    return ''.join(ch for ch in unicodedata.normalize('NFD', t) if unicodedata.category(ch) != 'Mn')


def lb_docs(sites=('KN', 'PY', 'TH'), min_lines=2):
    WORD = re.compile(r'^[a-z0-9*]+(-[a-z0-9*]+)*$')
    MEAS = set('TVZSMNPQ')
    out = []
    for line in open(os.path.join(D, 'damos_items.jsonl')):
        d = json.loads(line)
        h = d.get('heading') or ''
        m = re.match(r'^([A-Z]{2,3})\s+([A-Z][a-z]?\d?)', h)
        if not m or m.group(1) not in sites:
            continue
        lines = []
        for ln in (d.get('content') or '').split('\n'):
            cur, meas = [], None
            for t in _strip(ln).split():
                s = re.sub(r'[\[\]⟦⟧?!,⌞⌟\'"]', '', t)
                if not s or s.startswith('.') or s in ('/', 'vac', 'vac.', 'v.', 'r.', 'v.↓', 'lat.', 'inf.', 'sup.'):
                    continue
                b = s.split('+')[0]
                if s in MEAS:
                    meas = s
                elif s.isdigit():
                    v = float(int(s))
                    if cur and cur[-1][0] == 'N' and meas:
                        cur[-1] = ('N', cur[-1][1], True)
                    else:
                        cur.append(('N', v, bool(meas)))
                    meas = None
                elif re.fullmatch(r'\*1\d\d', b) or (b.isupper() and len(b) >= 2 and re.fullmatch(r'[A-Z±*0-9]+', b)):
                    cur.append(('G', b))
                elif WORD.match(s) and '-' in s:
                    cur.append(('W', s, s.count('-') + 1))
            if cur:
                lines.append(cur)
        if len(lines) >= min_lines and any(x[0] == 'N' for l in lines for x in l):
            out.append({'id': h, 'damos_id': d['id'], 'sys': 'LB', 'site': m.group(1), 'grp': m.group(1),
                        'series': m.group(2), 'lines': lines})
    return out


def ur3_docs(nmax=3000, min_lines=2):
    p = os.path.join(CK, 'ur3_docs.json')
    if os.path.exists(p):
        return json.load(open(p))
    import la57_common as C
    raw = C._cdli(lambda x: x.startswith('Ur III'), nmax, 'UR3')
    out = []
    for d in raw:
        lines, cur = [], []
        for x in d['toks']:
            if x[0] == 'L':
                if cur:
                    lines.append(cur); cur = []
            elif x[0] == 'N':
                if x[1] is not None:
                    cur.append(('N', float(x[1]), bool(x[2])))
            else:
                w = x[1]
                cur.append(('W', w, w.count('-') + 1))
        if cur:
            lines.append(cur)
        if len(lines) >= min_lines and any(t[0] == 'N' for l in lines for t in l):
            out.append({'id': d['id'], 'sys': 'UR3', 'site': d['site'], 'grp': 'UR3', 'lines': lines})
    json.dump(out, open(p, 'w'))
    return out


# ------------------------------------------------------------------ value-free document features
FEATS = ['nlines', 'ntok', 'nnum', 'nword', 'nlogo', 'fracshare', 'one', 'ge10', 'ge100', 'lmean', 'maxsum',
         'posmax', 'lasttotal', 'lastmax', 'linenum', 'linestartW', 'line2num', 'ttr', 'logoshare', 'logo_before_N',
         'word_before_N', 'wlen', 'wlen2', 'wlen4', 'dig0', 'desc', 'repnum', 'tokline', 'numline', 'headfree',
         'distinctnum', 'numfirst', 'lastline_num', 'cv_num', 'wordline']


def doc_features(lines):
    toks = [t for l in lines for t in l]
    nums = [t[1] for t in toks if t[0] == 'N']
    nf = [t[2] for t in toks if t[0] == 'N']
    words = [t for t in toks if t[0] == 'W']
    logos = [t for t in toks if t[0] == 'G']
    nan = float('nan')
    f = {}
    f['nlines'] = len(lines); f['ntok'] = len(toks); f['nnum'] = len(nums); f['nword'] = len(words)
    f['nlogo'] = len(logos)
    f['fracshare'] = sum(nf) / len(nf) if nf else nan
    f['one'] = sum(1 for v in nums if v == 1) / len(nums) if nums else nan
    f['ge10'] = sum(1 for v in nums if v >= 10) / len(nums) if nums else nan
    f['ge100'] = sum(1 for v in nums if v >= 100) / len(nums) if nums else nan
    f['lmean'] = float(np.mean(np.log1p(nums))) if nums else nan
    s = sum(nums)
    f['maxsum'] = max(nums) / s if nums and s > 0 else nan
    f['posmax'] = (nums.index(max(nums)) / (len(nums) - 1)) if len(nums) >= 2 else nan
    if len(nums) >= 3:
        f['lasttotal'] = 1.0 if abs(nums[-1] - sum(nums[:-1])) <= 0.01 * max(1, nums[-1]) else 0.0
        f['lastmax'] = 1.0 if nums[-1] >= max(nums[:-1]) else 0.0
        f['desc'] = sum(1 for a, b in zip(nums, nums[1:]) if b < a) / (len(nums) - 1)
        f['cv_num'] = float(np.std(nums) / np.mean(nums)) if np.mean(nums) > 0 else nan
    else:
        f['lasttotal'] = f['lastmax'] = f['desc'] = f['cv_num'] = nan
    f['linenum'] = sum(1 for l in lines if any(t[0] == 'N' for t in l)) / len(lines)
    f['linestartW'] = sum(1 for l in lines if l[0][0] == 'W') / len(lines)
    f['line2num'] = sum(1 for l in lines if sum(t[0] == 'N' for t in l) >= 2) / len(lines)
    f['ttr'] = len(set(w[1] for w in words)) / len(words) if words else nan
    tt = len(words) + len(logos)
    f['logoshare'] = len(logos) / tt if tt else nan
    pre = [toks[i - 1][0] for i in range(1, len(toks)) if toks[i][0] == 'N']
    f['logo_before_N'] = pre.count('G') / len(pre) if pre else nan
    f['word_before_N'] = pre.count('W') / len(pre) if pre else nan
    wl = [w[2] for w in words]
    f['wlen'] = float(np.mean(wl)) if wl else nan
    f['wlen2'] = sum(1 for x in wl if x == 2) / len(wl) if wl else nan
    f['wlen4'] = sum(1 for x in wl if x >= 4) / len(wl) if wl else nan
    iv = [int(v) for v in nums if v >= 10 and abs(v - round(v)) < 1e-9]
    f['dig0'] = sum(1 for v in iv if v % 10 == 0) / len(iv) if iv else nan
    f['repnum'] = 1 - len(set(nums)) / len(nums) if len(nums) >= 2 else nan
    f['tokline'] = len(toks) / len(lines)
    f['numline'] = len(nums) / len(lines)
    f['headfree'] = 1.0 if not any(t[0] == 'N' for t in lines[0]) else 0.0
    f['distinctnum'] = len(set(nums)) if nums else nan
    f['numfirst'] = sum(1 for l in lines if l[0][0] == 'N') / len(lines)
    f['lastline_num'] = 1.0 if any(t[0] == 'N' for t in lines[-1]) else 0.0
    f['wordline'] = len(words) / len(lines)
    return [f[k] for k in FEATS]


def feature_matrix(docs):
    return np.array([doc_features(d['lines']) for d in docs], dtype=float)


# ------------------------------------------------------------------ random statistics
AGG = ['mean', 'median', 'q', 'gt', 'corr', 'condmean']


def random_stats(n, rng, Xref):
    """Random statistic specs. Thresholds come from the REFERENCE matrix (Linear A pooled) only."""
    F = Xref.shape[1]
    specs = []
    while len(specs) < n:
        a = rng.choice(AGG); f = rng.randrange(F); tr = rng.choice(['id', 'log', 'sqrt'])
        col = Xref[:, f]; col = col[~np.isnan(col)]
        if len(col) < 30 or np.std(col) == 0:
            continue
        sp = {'agg': a, 'f': f, 'tr': tr}
        if a == 'q':
            sp['q'] = rng.choice([0.1, 0.25, 0.75, 0.9])
        if a == 'gt':
            sp['t'] = float(np.quantile(col, rng.choice([0.2, 0.35, 0.5, 0.65, 0.8])))
        if a in ('corr', 'condmean'):
            g = rng.randrange(F)
            if g == f:
                continue
            gc = Xref[:, g]; gc = gc[~np.isnan(gc)]
            if len(gc) < 30 or np.std(gc) == 0:
                continue
            sp['g'] = g
            if a == 'condmean':
                sp['side'] = rng.choice([1, -1]); sp['gt'] = float(np.median(gc))
        specs.append(sp)
    return specs


def _tr(x, tr):
    if tr == 'log':
        return np.log1p(np.abs(x)) * np.sign(x)
    if tr == 'sqrt':
        return np.sqrt(np.abs(x)) * np.sign(x)
    return x


def eval_stat(sp, X):
    x = _tr(X[:, sp['f']], sp['tr'])
    a = sp['agg']
    if a == 'condmean':
        g = X[:, sp['g']]
        m = (g > sp['gt']) if sp['side'] > 0 else (g <= sp['gt'])
        x = x[m & ~np.isnan(g)]
    if a == 'corr':
        g = X[:, sp['g']]
        ok = ~np.isnan(x) & ~np.isnan(g)
        if ok.sum() < 8:
            return float('nan')
        xr = np.argsort(np.argsort(x[ok])); gr = np.argsort(np.argsort(g[ok]))
        if np.std(xr) == 0 or np.std(gr) == 0:
            return float('nan')
        return float(np.corrcoef(xr, gr)[0, 1])
    x = x[~np.isnan(x)]
    if len(x) < 5:
        return float('nan')
    if a in ('mean', 'condmean'):
        return float(np.mean(x))
    if a == 'median':
        return float(np.median(x))
    if a == 'q':
        return float(np.quantile(x, sp['q']))
    if a == 'gt':
        return float(np.mean(x > _tr(np.array([sp['t']]), sp['tr'])[0]))
    return float('nan')


def eval_all(specs, X):
    return np.array([eval_stat(s, X) for s in specs])
