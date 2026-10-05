"""pe50 cycle 3: does the counted administration fit the town?

A  Survival-rate scale check (Ur III): small tablet (2-4 numbered lines) 'found' in a big tablet
   (>= 10 numbered lines) when every numbered line of the small one occurs verbatim on the big one.
   Fraction found must fall in proportion to the thinning q (q = 1, .5, .2, .1); null = numbered
   lines redealt among big tablets.  Then PE: same statistic.
B  Spelling merges: PE MID2 strings merged (i) as unordered sign sets, (ii) by one-sign edits
   (substitution or deletion, union-find), (iii) by dropping variant marks (already base signs).
   Chao2 floor recomputed.  Control: Ur III OFF names merged the same way (sign = ATF syllable).
C  Workforce and herds: per-tablet counts on the rationed-unit lines (pe grade B class
   M388, M054, M124, M003, |M370+M388|), with the counting units N14 = 10 and three value sets
   for the large units (N45 100 / N34 300; N45 100 / N34 60; N45 600 / N34 3600).  Herd tablets:
   summed animal counts.
D  Person-equivalents: chain PE string floor / (Ur III ENT2-per-OFF ratio at PE size) x (Ur III
   OFF full/Chao2 multiplier).  Cross-archive control: calibrate on Drehem, predict Umma's full
   official count from a PE-sized Umma sample, and the reverse.
Output data/pe50_ckpt/cycle3.json
"""
import json, os, sys, random, collections, itertools, re
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from pe50_lib import chao2  # noqa
from common import load  # noqa
CK = os.path.join(HERE, '..', 'data', 'pe50_ckpt')
D = json.load(open(os.path.join(CK, 'caps.json')))
CAPS, META = D['caps'], D['meta']
U = json.load(open(os.path.join(CK, 'ur3_caps.json')))
rng = random.Random(50)
out = {}


# ---------- A ----------
def found_frac(tabs, rng, null=False):
    small = [t for t in tabs if 2 <= len(t['entries']) <= 4]
    big = [t for t in tabs if len(t['entries']) >= 10]
    if null:
        pool = [e for t in big for e in t['entries']]
        rng.shuffle(pool)
        k = 0
        nb = []
        for t in big:
            nb.append(pool[k:k + len(t['entries'])])
            k += len(t['entries'])
        bigsets = [set(x) for x in nb]
    else:
        bigsets = [set(t['entries']) for t in big]
    idx = collections.defaultdict(set)
    for j, s in enumerate(bigsets):
        for e in s:
            idx[e].add(j)
    f = 0
    for t in small:
        es = set(t['entries'])
        cand = None
        for e in es:
            cand = idx.get(e, set()) if cand is None else cand & idx.get(e, set())
            if not cand:
                break
        f += bool(cand)
    return f, len(small), len(big)


A = {}
for a in ('DREHEM', 'UMMA'):
    tabs = U[a]
    A[a] = {}
    for q in (1.0, 0.5, 0.2, 0.1):
        reps = 1 if q == 1.0 else 3
        v = []
        for r in range(reps):
            s = tabs if q == 1.0 else [t for t in tabs if rng.random() < q]
            f, ns, nb = found_frac(s, rng)
            fn, _, _ = found_frac(s, rng, null=True)
            v.append((f, ns, nb, fn))
        A[a][q] = v
        print('A', a, q, v, flush=True)
# PE version: entries as (signs, numerals) strings
T = load()
pet = []
for t in T:
    ents = []
    for l in t['lines']:
        if l['numerals'] and l['signs']:
            ents.append(json.dumps([l['signs'], l['numerals']]))
    pet.append({'entries': ents})
f, ns, nb = found_frac(pet, rng)
fn = [found_frac(pet, rng, null=True)[0] for _ in range(20)]
A['PE'] = {'found': f, 'small': ns, 'big': nb, 'null': fn}
print('A PE', f, ns, nb, fn, flush=True)
out['A'] = A


# ---------- B ----------
def merge_edit(strings):
    parent = {s: s for s in strings}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    keys = collections.defaultdict(list)
    for s in strings:
        keys[('del',) + s].append(s)
        if len(s) < 3:
            continue
        for i in range(len(s)):
            keys[('sub',) + s[:i] + ('*',) + s[i + 1:]].append(s)
            keys[('del',) + s[:i] + s[i + 1:]].append(s)
    for k, L in keys.items():
        for x in L[1:]:
            a, b = find(L[0]), find(x)
            if a != b:
                parent[a] = b
    return {s: find(s) for s in strings}


def merged_inc(inc, how):
    allS = sorted({e for s in inc for e in s})
    if how == 'set':
        m = {s: tuple(sorted(set(s))) for s in allS}
    elif how == 'edit':
        m = merge_edit(allS)
    else:
        m = {s: s for s in allS}
    return [{m[e] for e in s} for s in inc]


B = {}
pe_inc = [{tuple(json.loads(e)) for e in s} for s in CAPS['PE']['MID2'] if s]
for how in ('none', 'set', 'edit'):
    I = merged_inc(pe_inc, how)
    S = len(set().union(*I))
    B['PE_' + how] = (S, chao2(I))
    print('B PE', how, S, chao2(I), flush=True)
for a in ('DREHEM', 'UMMA'):
    full = [{tuple(re.split(r'[-.]', n)) for n in t['names']} for t in U[a] if t['names']]
    Sf = len(set().union(*full))
    for how in ('none', 'set', 'edit'):
        If = merged_inc(full, how)
        Sm = len(set().union(*If))
        sub = rng.sample(If, 607)
        B[a + '_' + how] = (Sf, Sm, chao2(sub))
        print('B', a, how, Sf, Sm, chao2(sub), flush=True)
out['B'] = B

# ---------- C ----------
RAT = {'M388', 'M054', 'M124', 'M003', '|M370+M388|'}
HERD = {'M362', 'M362~a', 'M367', 'M367~a', 'M346', 'M346~a', 'M006', 'M006@g'}
VS = {'v100_300': {'N01': 1, 'N14': 10, 'N45': 100, 'N34': 300},
      'v100_60': {'N01': 1, 'N14': 10, 'N45': 100, 'N34': 60},
      'v600_3600': {'N01': 1, 'N14': 10, 'N45': 600, 'N34': 3600}}
C = {}
for vn, tab in VS.items():
    rat, herd, allc = [], [], []
    for t in T:
        r = h = a = 0
        for i, l in enumerate(t['lines']):
            if not l['numerals'] or not l['signs']:
                continue
            if any(c not in tab or not isinstance(n, int) for n, c in l['numerals']):
                continue
            v = sum(n * tab[c] for n, c in l['numerals'])
            last = l['signs'][-1].split('~')[0] if l['signs'][-1] != '|M370+M388|' else l['signs'][-1]
            if l['surface'] == 'obverse':
                a += v
                if last in RAT or l['signs'][-1] in RAT:
                    r += v
                if l['signs'][-1] in HERD or any(s.startswith('|M362+') for s in l['signs']):
                    h += v
        rat.append((r, t['id'], t['designation']))
        herd.append((h, t['id'], t['designation']))
        allc.append(a)
    rat.sort(reverse=True)
    herd.sort(reverse=True)
    C[vn] = {'rat_top': rat[:12], 'rat_n': sum(1 for x in rat if x[0] > 0),
             'rat_q': np.percentile([x[0] for x in rat if x[0] > 0], [50, 90, 99]).tolist(),
             'herd_top': herd[:8], 'all_q': np.percentile([x for x in allc if x > 0], [50, 90, 99, 100]).tolist()}
    print('C', vn, C[vn]['rat_top'][:6], C[vn]['rat_q'], C[vn]['herd_top'][:4], flush=True)
out['C'] = C


# ---------- D ----------
def chao_sample(inc, n, reps=10):
    v = []
    for _ in range(reps):
        v.append(chao2(rng.sample(inc, n))[0])
    return float(np.median(v))


Dd = {}
for a in ('DREHEM', 'UMMA'):
    ent = [set(s) for s in CAPS[a]['ENT2'] if s]
    off = [set(s) for s in CAPS[a]['OFF'] if s]
    Dd[a] = {'ent_chao': chao_sample(ent, 607), 'off_chao': chao_sample(off, 607),
             'off_full': len(set().union(*off)), 'ent_full': len(set().union(*ent))}
for a, b in (('DREHEM', 'UMMA'), ('UMMA', 'DREHEM')):
    ratio = Dd[a]['ent_chao'] / Dd[a]['off_chao']
    mult = Dd[a]['off_full'] / Dd[a]['off_chao']
    pred = Dd[b]['ent_chao'] / ratio * mult
    Dd['pred_%s_from_%s' % (b, a)] = {'pred_off_full': pred, 'true_off_full': Dd[b]['off_full'],
                                      'ratio': ratio, 'mult': mult}
pe_chao = chao2(pe_inc)
for a in ('DREHEM', 'UMMA'):
    ratio = Dd[a]['ent_chao'] / Dd[a]['off_chao']
    mult = Dd[a]['off_full'] / Dd[a]['off_chao']
    Dd['PE_from_' + a] = {'floor': pe_chao[1] / ratio, 'point': pe_chao[0] / ratio, 'calibrated': pe_chao[0] / ratio * mult,
                          'calibrated_lo': pe_chao[1] / ratio * mult}
print('D', json.dumps(Dd, indent=1), flush=True)
out['D'] = Dd
# persons-ever at Susa: residents (area x density) x (1 + duration / career) x adult share
mc = []
r2 = np.random.default_rng(5)
for _ in range(20000):
    area = r2.uniform(5, 11)          # < 11 ha (Carter 1998); lower end assumed
    dens = r2.uniform(100, 200)       # Kramer 1982 ~120/ha; up to 200/ha
    dur = r2.uniform(100, 150)        # ca. 3050-2900 BCE (Englund 1998)
    car = r2.uniform(15, 30)
    adult = r2.uniform(0.4, 0.6)
    mc.append((area * dens, area * dens * (1 + dur / car) * adult))
mc = np.array(mc)
out['town'] = {'residents': np.percentile(mc[:, 0], [2.5, 50, 97.5]).tolist(),
               'adults_ever': np.percentile(mc[:, 1], [2.5, 50, 97.5]).tolist()}
print('town', out['town'])
json.dump(out, open(os.path.join(CK, 'cycle3.json'), 'w'), default=float)
