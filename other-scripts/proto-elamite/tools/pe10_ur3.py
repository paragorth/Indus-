"""pe10 positive control: Ur III year formulas.  The same year is written in long and
short forms (known abbreviation).  If 'squeeze' is a real force, the formula should be
shorter on tablets whose other text leaves less room (more non-year lines per cm of height),
comparing tablets dated to the SAME year.  Null: permute crowding within year.
Also extracts (year, formula string, squeezed flag) for the pair-mining positive control.
Source: CDLI bulk ATF + catalogue (scratchpad; not committed).
"""
import csv, re, json, os, sys
import numpy as np
from collections import defaultdict, Counter
from pe10_common import SCRATCH, CKPT

rng = np.random.default_rng(3)


def load_cat():
    csv.field_size_limit(10 ** 9)
    cat = {}
    for row in csv.DictReader(open(os.path.join(SCRATCH, 'cdli_cat.csv'), encoding='utf-8')):
        if not row['period'].startswith('Ur III'):
            continue
        d = row['date_of_origin'].strip()
        m = re.match(r'^([^.]+)\.(\d\d)\.', d)
        if not m or m.group(2) == '00' or '?' in d.split('.')[1]:
            continue
        try:
            h = float(row['height']); w = float(row['width'])
        except Exception:
            continue
        if h <= 0 or w <= 0:
            continue
        cat['P%06d' % int(row['id_text'])] = {'year': m.group(1) + '.' + m.group(2), 'h': h, 'w': w,
                                             'prov': row['provenience'][:12]}
    return cat


def graphemes(s):
    s = re.sub(r'[\[\]#?!<>]', '', s)
    return [g for g in re.split(r'[\s\-.]+', s) if g and g != '...']


def parse(cat):
    out = []
    cur = None
    face = None
    for raw in open(os.path.join(SCRATCH, 'cdli.atf'), encoding='utf-8', errors='replace'):
        line = raw.rstrip()
        if line.startswith('&P'):
            if cur:
                out.append(cur)
            pid = line[1:8]
            cur = {'id': pid, 'lines': []} if pid in cat else None
            face = None
            continue
        if cur is None:
            continue
        if line.startswith('@'):
            tag = line[1:].split()[0] if line[1:].split() else ''
            if tag in ('obverse', 'reverse', 'left', 'bottom', 'top', 'right', 'edge', 'seal', 'envelope'):
                face = tag
            continue
        if line.startswith('$') and 'broken' in line:
            cur['broken'] = True
        m = re.match(r"^(\d+'?)\.\s+(.*)$", line)
        if m and face:
            cur['lines'].append((face, m.group(2)))
    if cur:
        out.append(cur)
    return out


def year_block(t):
    L = [(f, s) for f, s in t['lines'] if f not in ('seal', 'envelope')]
    idx = None
    for i, (f, s) in enumerate(L):
        if re.match(r'^\[?mu[\s\]]', s) and not s.startswith('mu-'):
            idx = i
    if idx is None:
        return None
    blk = L[idx:]
    # stop if a later line starts a new section that's clearly not the year (rare); keep simple
    txt = ' '.join(s for _, s in blk)
    if '...' in txt or '[' in txt or 'x' in txt.split():
        return None
    return {'n_year_lines': len(blk), 'faces': [f for f, _ in blk], 'g': graphemes(txt), 'txt': txt,
            'n_other': idx}


if __name__ == '__main__':
    cat = load_cat()
    tabs = parse(cat)
    rows = []
    for t in tabs:
        if t.get('broken'):
            continue
        yb = year_block(t)
        if yb is None or yb['n_other'] < 2:
            continue
        c = cat[t['id']]
        rows.append({'id': t['id'], 'year': c['year'], 'h': c['h'], 'w': c['w'],
                     'len': len(yb['g']), 'crowd': yb['n_other'] / (c['h'] / 10), 'n_other': yb['n_other'],
                     'edge': any(f in ('left', 'bottom', 'top', 'right', 'edge') for f in yb['faces']),
                     'txt': yb['txt']})
    by = defaultdict(list)
    for r in rows:
        by[r['year']].append(r)
    keep = [g for g in by.values() if len(g) >= 10 and len({r['len'] for r in g}) > 1]
    R = [r for g in keep for r in g]
    print('tablets', len(rows), 'in years with >=10 and variable length', len(R), 'years', len(keep))
    # within-year demeaned correlation
    gid = np.array([i for i, g in enumerate(keep) for _ in g])
    L = np.array([r['len'] for r in R], float)
    X = np.array([r['crowd'] for r in R], float)
    H = np.array([r['h'] for r in R], float)
    def dm(v):
        o = v.copy()
        for i in range(len(keep)):
            k = gid == i; o[k] = v[k] - v[k].mean()
        return o
    Ld = dm(L)
    def stat(x):
        return float(np.corrcoef(dm(x), Ld)[0, 1])
    obs = stat(X); obsH = stat(H)
    nul = []; nulH = []
    for _ in range(1000):
        x = X.copy(); hh = H.copy()
        for i in range(len(keep)):
            k = np.where(gid == i)[0]; p = rng.permutation(k)
            x[k] = X[p]; hh[k] = H[p]
        nul.append(stat(x)); nulH.append(stat(hh))
    nul = np.array(nul); nulH = np.array(nulH)
    # edge placement: formula on an edge vs not, same year
    E = np.array([r['edge'] for r in R])
    de = Ld[E].mean() - Ld[~E].mean() if E.any() else float('nan')
    nule = []
    for _ in range(1000):
        e = E.copy()
        for i in range(len(keep)):
            k = np.where(gid == i)[0]; e[k] = E[rng.permutation(k)]
        nule.append(Ld[e].mean() - Ld[~e].mean())
    nule = np.array(nule)
    res = {'n_tablets': len(R), 'n_years': len(keep),
           'r_crowding_vs_len_within_year': obs, 'null_sd': float(nul.std()), 'z': obs / nul.std(),
           'r_height_vs_len_within_year': obsH, 'z_height': obsH / nulH.std(),
           'edge_minus_face_len': float(de), 'n_edge': int(E.sum()), 'z_edge': float(de / nule.std())}
    print(json.dumps(res, indent=1))
    json.dump({'res': res, 'rows': rows}, open(os.path.join(CKPT, 'ur3_years.json'), 'w'))
