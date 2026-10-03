"""S-DARK-50 cycle 2: pictoriality against slot role, frequency controlled.

Roles per token from the S310/S331 frame parser (as in dark_loop26/19/22): OPENER, MARKER, CLOSER (= head),
TITLE (adjacent qualifier selected by the closer, S303), NAME (middle), COUNT (numeral), SUFFIX.  NAME tokens
directly before the closer are split off as NAME_ADJ; the rest are NAME_FAR (non-adjacent middle).
Each sign with >= MINTOK tokens gets its modal role (share >= 0.5, else MIXED).
Measures from cycle 1 (loop50_signs.json): pict_shape (6-part z), pict_curv (curv + asym only, least complexity-
loaded), label (dossier keyword rule, 84 signs).
Null: role labels permuted among signs inside log2-frequency bins (1,000x); statistic = mean measure per role.
Also an OLS with log f as covariate (role dummies vs NAME_FAR) and a token-weighted version.
Usage: python3 tools/dark_loop50_c2.py LEVEL [NPERM]
"""
import sys, json, math, collections, random
import numpy as np

LV = sys.argv[1]; NP = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
MINTOK = 5
OUT = 'data/derived/dark/'
rnd = random.Random(50); np.random.seed(50)
C = json.load(open('data/derived/merged-corpus-canonical.json'))
S = {m['w']: m for m in json.load(open(OUT + 'loop50_signs.json'))}

# curvature-only index, z over non-numeral signs
base = [m for m in S.values() if not m['numeral']]
for k in ('curv', 'asym'):
    mu = np.mean([m[k] for m in base]); sd = np.std([m[k] for m in base]) + 1e-9
    for m in S.values(): m['z_' + k] = (m[k] - mu) / sd
for m in S.values(): m['pict_curv'] = (m['z_curv'] + m['z_asym']) / 2

def otype(t):
    t = t.split(':')[0]
    return {'SEAL': 'seal', 'TAB': 'tablet', 'POT': 'pot', 'TAG': 'sealing'}.get(t, 'other')
OBJ = []
for r in C:
    s = r[LV]
    if not s or len(s) < 2 or r['complete'] != 'Y' or r['dir.'].strip() == '-': continue
    OBJ.append(dict(site=r['site'], ot=otype(r['type']), seq=list(s)))

OPEN = {817, 861, 820, 920, 692}; MARK = {2, 60}; MJAR = {741, 742, 745}; SUF = {400, 90}
CL = [740, 520, 151, 156, 527, 226, 617, 154, 158, 236, 700]
FISH = {235, 240, 233, 231, 220}; NUM = {1, 3, 4, 5, 16, 17, 18, 31, 32, 33, 34, 55, 56}
left = collections.defaultdict(collections.Counter)
for o in OBJ:
    s = o['seq'][:]
    while len(s) > 1 and s[-1] in SUF: s.pop()
    if len(s) >= 2 and s[-1] in CL: left[s[-1]][s[-2]] += 1
QUAL = {}
for c, cnt in left.items():
    tot = sum(cnt.values()); acc = 0; q = set()
    for a, n in cnt.most_common():
        if acc / tot >= 0.6: break
        q.add(a); acc += n
    QUAL[c] = q

def parse(s):
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
    # split NAME directly before the closer/title block
    if 'CLOSER' in lab:
        jc = lab.index('CLOSER')
        k = jc - 1
        while k >= 0 and lab[k] == 'TITLE': k -= 1
        if k >= 0 and lab[k] == 'NAME': lab[k] = 'NAME_ADJ'
    return ['NAME_FAR' if l == 'NAME' else l for l in lab]

ROLE_TOK = collections.defaultdict(collections.Counter)
for o in OBJ:
    for a, l in zip(o['seq'], parse(o['seq'])): ROLE_TOK[a][l] += 1
ROLES = ['OPENER', 'MARKER', 'CLOSER', 'TITLE', 'NAME_ADJ', 'NAME_FAR', 'COUNT', 'SUFFIX']

rows = []
for w, cnt in ROLE_TOK.items():
    n = sum(cnt.values())
    if n < MINTOK or w not in S: continue
    role, k = cnt.most_common(1)[0]
    modal = role if k / n >= 0.5 else 'MIXED'
    rows.append(dict(w=w, n=n, modal=modal, shares={r: cnt[r] / n for r in ROLES}, numeral=S[w]['numeral'],
                     pict_shape=S[w]['pict_shape'], pict_curv=S[w]['pict_curv'], label=S[w]['label'], label3=S[w]['label3'],
                     perim=S[w]['perim'], holes=S[w]['holes'], strokes=S[w]['strokes']))
json.dump(rows, open(OUT + f'loop50_roles_{LV}.json', 'w'))

rep = []
def P(*a):
    s = ' '.join(str(x) for x in a); print(s); rep.append(s)
P(f'# LOOP 50 cycle 2, level {LV}: pictoriality by slot role ({len(OBJ)} complete direction-recorded texts, {len(rows)} signs with >= {MINTOK} tokens)')
P('modal role counts:', dict(collections.Counter(r['modal'] for r in rows)))

def test(sel, measure, title, nperm=NP, by='modal'):
    """sel: list of rows; measure key; null = permute the role labels inside log2-frequency bins"""
    vals = np.array([r[measure] for r in sel], float); ok = ~np.isnan(vals)
    sel = [r for r, o in zip(sel, ok) if o]; vals = vals[ok]
    roles = np.array([r[by] for r in sel]); fbin = np.array([int(math.log2(r['n'])) for r in sel])
    groups = sorted(set(roles), key=lambda g: ROLES.index(g) if g in ROLES else 99)
    obs = {g: vals[roles == g].mean() for g in groups}
    nulls = {g: [] for g in groups}
    bins = {b: np.where(fbin == b)[0] for b in set(fbin)}
    for _ in range(nperm):
        pr = roles.copy()
        for b, idx in bins.items():
            pr[idx] = roles[np.random.permutation(idx)]
        for g in groups: nulls[g].append(vals[pr == g].mean())
    P(f'  {title}: measure {measure}, n = {len(sel)}, overall mean {vals.mean():+.3f}')
    for g in groups:
        nl = np.array(nulls[g]); z = (obs[g] - nl.mean()) / (nl.std() + 1e-9)
        p = (np.sum(np.abs(nl - nl.mean()) >= abs(obs[g] - nl.mean())) + 1) / (nperm + 1)
        P(f'     {g:9s} n={np.sum(roles==g):3d}  mean {obs[g]:+.3f}  null {nl.mean():+.3f} [{np.percentile(nl,2.5):+.3f},{np.percentile(nl,97.5):+.3f}]  z {z:+.2f}  P {p:.3f}')
    # head vs far middle difference
    if 'CLOSER' in groups and 'NAME_FAR' in groups:
        d = obs['CLOSER'] - obs['NAME_FAR']; dn = np.array(nulls['CLOSER']) - np.array(nulls['NAME_FAR'])
        p = (np.sum(np.abs(dn - dn.mean()) >= abs(d - dn.mean())) + 1) / (nperm + 1)
        P(f'     head - far middle: {d:+.3f}, null {dn.mean():+.3f} [{np.percentile(dn,2.5):+.3f},{np.percentile(dn,97.5):+.3f}], P {p:.3f}')
    if 'OPENER' in groups and 'NAME_FAR' in groups:
        fr = [g for g in ('OPENER', 'MARKER', 'CLOSER', 'SUFFIX') if g in groups]
        frame = np.mean([obs[g] for g in fr]); fn = np.mean([nulls[g] for g in fr], axis=0)
        d = frame - obs['NAME_FAR']; dn = fn - np.array(nulls['NAME_FAR'])
        p = (np.sum(np.abs(dn - dn.mean()) >= abs(d - dn.mean())) + 1) / (nperm + 1)
        P(f'     frame slots ({"+".join(fr)}) - far middle: {d:+.3f}, null [{np.percentile(dn,2.5):+.3f},{np.percentile(dn,97.5):+.3f}], P {p:.3f}')

NONNUM = [r for r in rows if not r['numeral']]
P(f'\n== A. shape index, numerals excluded (n={len(NONNUM)})')
test(NONNUM, 'pict_shape', 'all non-numeral signs')
test(NONNUM, 'pict_curv', 'all non-numeral signs')
P(f'\n== B. shape index, numerals included (COUNT role is pure strokes by construction)')
test(rows, 'pict_shape', 'all signs')
P(f'\n== C. dossier label (1 pictorial / 0 geometric), numerals excluded')
LAB = [r for r in NONNUM if r['label3'] in ('pictorial', 'geometric')]
test(LAB, 'label', 'labelled signs', by='modal')
P(f'\n== D. complexity alone (perimetric), numerals excluded')
test(NONNUM, 'perim', 'all non-numeral signs')

# OLS: measure ~ log n + role dummies (ref NAME_FAR), numerals excluded
def ols(sel, measure):
    sel = [r for r in sel if not math.isnan(r[measure]) and r['modal'] != 'MIXED']
    groups = [g for g in ROLES if g != 'NAME_FAR' and any(r['modal'] == g for r in sel)]
    X = np.array([[1.0, math.log(r['n'])] + [1.0 if r['modal'] == g else 0.0 for g in groups] for r in sel])
    y = np.array([r[measure] for r in sel])
    beta, res, rk, sv = np.linalg.lstsq(X, y, rcond=None)
    yhat = X @ beta; s2 = ((y - yhat) ** 2).sum() / max(len(y) - X.shape[1], 1)
    cov = s2 * np.linalg.pinv(X.T @ X); se = np.sqrt(np.diag(cov))
    P(f'  OLS {measure} ~ log n + role (ref NAME_FAR), n={len(sel)}: log n {beta[1]:+.3f} (se {se[1]:.3f})')
    for i, g in enumerate(groups): P(f'     {g:9s} {beta[2+i]:+.3f} (se {se[2+i]:.3f}, t {beta[2+i]/se[2+i]:+.2f})')
P('\n== E. OLS with log frequency as covariate (numerals excluded)')
ols(NONNUM, 'pict_shape'); ols(NONNUM, 'pict_curv'); ols(NONNUM, 'perim')

# token-weighted picture: mean measure of tokens in each role
P('\n== F. token-weighted means (every token of every sign, numerals excluded)')
for meas in ('pict_shape', 'pict_curv'):
    acc = collections.defaultdict(list)
    for r in NONNUM:
        for g in ROLES: acc[g] += [r[meas]] * int(round(r['shares'][g] * r['n']))
    P('  ' + meas + ': ' + '  '.join(f'{g} {np.mean(v):+.2f} (n={len(v)})' for g, v in acc.items() if v))

# list per role
P('\n== G. signs by modal role (non-numeral), shape index and label')
for g in ROLES + ['MIXED']:
    sel = sorted([r for r in NONNUM if r['modal'] == g], key=lambda r: -r['n'])
    if not sel: continue
    P(f'  {g} ({len(sel)}): ' + ', '.join(f'W{r["w"]}:{r["pict_shape"]:+.1f}{"P" if r["label3"]=="pictorial" else "G" if r["label3"]=="geometric" else ""}' for r in sel[:25]))
open(OUT + f'loop50_cycle2_{LV}.txt', 'w').write('\n'.join(rep))
