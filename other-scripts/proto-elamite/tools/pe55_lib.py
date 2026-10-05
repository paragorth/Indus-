"""pe55 THE NUMBERS MUST OBEY AGRONOMY AND PHYSIOLOGY.

A count (people, animals, or an area) written next to a capacity quantity gives a ratio.  If the
pairing is an activity (rationing, fodder, sowing, harvest), the ratio in litres per unit must sit
in the physical range of that activity.  Hypothesis = (u litres per base capacity unit, A hectares
per 'area' count unit, activity label per key sign).  Thousands of random hypotheses are scored by
how many pairs land in physical windows at once; survivors are tested on held-out tablets.

Physical constants (litres of hulled barley; bulk density 0.6-0.7 kg/l, energy ~3.5 kcal/g):
  person-day      1.2 l   (0.5-2.5)  FAO/WHO/UNU 2004 energy needs 2,000-3,200 kcal/d adult,
                                    children less; 2,600 kcal / (3,520 kcal/kg x 0.65 kg/l) = 1.14 l
  smallstock-day  0.8 l   (0.3-2.1)  barley to sheep 150-750 g/head/day as supplement (CSIRO EA9930403,
                                    EA9890029; ~500 g optimum for lambs) up to ~1.3 kg/d grain-fattening
  largestock-day  4.5 l   (1.7-12)   working ox / donkey concentrate 1-5 kg/d (10 g/kg body weight on
                                    250-400 kg oxen, Edinburgh draught-ox studies; R5198 DFID reports),
                                    up to ~7 kg/d when grain-fattened
  x30 (month) and x360 (year) issue periods for each of the above
  area-seed       120 l/ha (45-280)  barley seed 30-180 kg/ha traditional, 120-200 kg/ha drilled rainfed
                                    trials in Iran (ICARDA/FAO AGRIS 1977-78 season; Shiraz and Razi
                                    univ. trials)
  area-yield     1500 l/ha (450-4500) rainfed barley 0.3-3 t/ha; Iran rainfed mean 2.1 t/ha
  (beer: grain-equivalent of a daily beer issue overlaps person-day; wool has no capacity pairing)
No sign reading from anyone is used.  Ur III nouns are used only as the truth of the control.
"""
import json, os, re, sys, math
from collections import Counter, defaultdict
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, '..', 'data')
CK = os.path.join(DATA, 'pe55_ckpt')
os.makedirs(CK, exist_ok=True)
SCRATCH = '/tmp/claude-0/-home-user-Indus-/874df4c7-80d6-5f08-b42c-eea96a214079/scratchpad'
sys.path.insert(0, HERE)

# activity: (median litres per unit [per ha for area], log-sd, kind)
ACT = {}
for nm, med, sd in [('person', 1.2, 0.40), ('small', 0.8, 0.6), ('large', 4.5, 0.6)]:
    for per, f in [('d', 1), ('m', 30), ('y', 360)]:
        ACT[f'{nm}-{per}'] = (med * f, sd, 'head')
ACT['area-seed'] = (120.0, 0.45, 'area')
ACT['area-yield'] = (1500.0, 0.55, 'area')
ACTS = list(ACT)
NA = len(ACTS)
MU = np.log(np.array([ACT[a][0] for a in ACTS]))
SD = np.array([ACT[a][1] for a in ACTS])
ISAREA = np.array([ACT[a][2] == 'area' for a in ACTS])
ROBUST = True
BG_SD = 2.0          # background: log-normal centred on the key's own median, sd 2 (unit-free)
LOGU = (math.log(0.01), math.log(100.0))     # prior on litres per base capacity unit
LOGA = (math.log(0.01), math.log(10.0))      # prior on hectares per area count unit
COARSE = {'person': 'person', 'small': 'stock', 'large': 'stock', 'area': 'area'}


def coarse(a):
    return 'none' if a is None or a == 'none' else COARSE[a.split('-')[0]]


# ---------------------------------------------------------------- pair extraction
def adjacent_pairs(tabs, key_mode='cnt'):
    """tabs: [{'id','Q':[{'sys','v','fin','line'}]}] -> list of pair dicts.
    A pair = a count line and a capacity line on neighbouring lines (next line preferred)."""
    P = []
    for t in tabs:
        Q = sorted(t['Q'], key=lambda q: q['line'])
        byl = {q['line']: q for q in Q}
        used = set()
        for q in Q:
            if not q['sys'].startswith('CNT') or float(q['v']) <= 0:
                continue
            for d in (1, -1):
                o = byl.get(q['line'] + d)
                if o is not None and o['sys'].startswith('CAP') and float(o['v']) > 0 and o['line'] not in used:
                    used.add(o['line'])
                    P.append({'tab': t['id'], 'c': float(q['v']), 'q': float(o['v']), 'kc': q['fin'],
                              'kq': o['fin'], 'truth': q.get('truth'), 'line': q['line']})
                    break
    return P


def build_arrays(P, key='kc', min_n=3, keys=None):
    if keys is None:
        cnt = Counter(p[key] for p in P)
        keys = sorted(k for k, n in cnt.items() if n >= min_n and k not in ('-', None))
    ki = {k: i for i, k in enumerate(keys)}
    sel = [p for p in P if p[key] in ki]
    lr = np.array([math.log(p['q'] / p['c']) for p in sel])
    kk = np.array([ki[p[key]] for p in sel], int)
    tab = np.array([p['tab'] for p in sel])
    return dict(keys=keys, lr=lr, k=kk, tab=tab, P=sel)


# ---------------------------------------------------------------- scoring
def key_medians(D):
    K = len(D['keys'])
    med = np.zeros(K)
    for i in range(K):
        m = D['k'] == i
        med[i] = np.median(D['lr'][m]) if m.any() else 0.0
    return med


def gains(D, logu, loga, med=None):
    """G[h, key, act] = sum over the key's pairs of log f_act(r) - log f_bg(r), for each hypothesis h
    (logu, loga arrays of length H).  Background f_bg is unit-free (centred on the key's median)."""
    if med is None:
        med = key_medians(D)
    lr, k = D['lr'], D['k']
    K = len(D['keys'])
    H = len(logu)
    # r in litres per unit (head) or per ha (area): log r = lr + logu (- loga)
    if H > 300:
        return np.concatenate([gains(D, logu[i:i + 300], loga[i:i + 300], med) for i in range(0, H, 300)])
    x = lr[None, :, None] + logu[:, None, None] - np.where(ISAREA, 1, 0)[None, None, :] * loga[:, None, None]
    lf = -0.5 * ((x - MU[None, None, :]) / SD[None, None, :]) ** 2 - np.log(SD)[None, None, :]
    lb = -0.5 * ((lr - med[k]) / BG_SD) ** 2 - math.log(BG_SD)
    g = lf - lb[None, :, None]
    if ROBUST:
        # each pair is activity (prob w) or junk (background); best of a small w grid per key
        rho = np.exp(np.clip(g, -50, 50))
        oh = np.zeros((len(lr), K))
        oh[np.arange(len(lr)), k] = 1
        Gs = [np.einsum('hpa,pk->hka', np.log(w * rho + 1 - w), oh) for w in (0.3, 0.6, 0.9)]
        return np.maximum(np.maximum(Gs[0], Gs[1]), Gs[2])
    oh = np.zeros((len(lr), K))
    oh[np.arange(len(lr)), k] = 1
    G = np.einsum('hpa,pk->hka', g, oh)
    return G


def hits(D, logu, loga, labels):
    """Fraction of pairs whose ratio lies inside +-1.64 sd (90%) of its key's activity window."""
    lr, k = D['lr'], D['k']
    tot, ok = 0, 0
    for i, a in enumerate(labels):
        if a is None:
            continue
        m = k == i
        j = ACTS.index(a)
        x = lr[m] + logu - (loga if ISAREA[j] else 0)
        ok += int((np.abs(x - MU[j]) <= 1.64 * SD[j]).sum())
        tot += int(m.sum())
    return ok, tot


def best_labels(G, allow_area=True):
    """Per hypothesis: per-key best activity (or none if gain <= 0).  Returns score[h], lab[h,key]."""
    Gm = G.copy()
    if not allow_area:
        Gm[:, :, ISAREA] = -1e9
    b = Gm.argmax(2)
    v = Gm.max(2)
    lab = np.where(v > 0, b, -1)
    return np.maximum(v, 0).sum(1), lab


def score_labels(G, lab):
    """G[h,key,act], lab[key] (fixed, -1 = none) -> score per hypothesis."""
    s = np.zeros(G.shape[0])
    for i, a in enumerate(lab):
        if a >= 0:
            s += G[:, i, a]
    return s


def random_search(D, rng, n_hyp=5000, allow_area=True, n_rand_lab=5000):
    """Massive random guessing: n_hyp random (u, A); labels either drawn at random (pure guessing)
    or set to the per-key best for that (u, A).  Returns the top hypotheses."""
    logu = rng.uniform(*LOGU, n_hyp)
    loga = rng.uniform(*LOGA, n_hyp)
    med = key_medians(D)
    G = gains(D, logu, loga, med)
    s_best, lab_best = best_labels(G, allow_area)
    # pure random label sets scored at random units (the brief's random guessing; kept as a pool)
    K = len(D['keys'])
    choices = np.arange(-1, NA if allow_area else int((~ISAREA).sum()))
    rl = rng.choice(choices, size=(n_rand_lab, K))
    hh = rng.integers(0, n_hyp, n_rand_lab)
    s_rand = np.array([sum(G[h, i, a] for i, a in enumerate(rl[j]) if a >= 0) for j, h in enumerate(hh)])
    return dict(logu=logu, loga=loga, s_best=s_best, lab_best=lab_best, s_rand=s_rand, rl=rl, hh=hh,
                med=med)


def apply_frozen(D, logu, loga, lab, keys_train):
    """Score held-out D with a frozen hypothesis (keys mapped by name)."""
    ki = {k: i for i, k in enumerate(D['keys'])}
    med = key_medians(D)
    G = gains(D, np.array([logu]), np.array([loga]), med)[0]
    s = 0.0
    n = 0
    for i, a in enumerate(lab):
        if a >= 0 and keys_train[i] in ki:
            s += G[ki[keys_train[i]], a]
            n += int((D['k'] == ki[keys_train[i]]).sum())
    return s, n


def split_tabs(D, rng, frac=0.5):
    t = np.unique(D['tab'])
    rng.shuffle(t)
    A = set(t[:int(len(t) * frac)])
    return A


def subset(D, tabset, keep=True):
    m = np.array([(x in tabset) == keep for x in D['tab']])
    return dict(keys=D['keys'], lr=D['lr'][m], k=D['k'][m], tab=D['tab'][m],
                P=[p for p, mm in zip(D['P'], m) if mm])


def shuffle_q(D, rng):
    """Null 1: capacity values re-dealt across pairs (numbers shuffled across lines)."""
    P = [dict(p) for p in D['P']]
    qs = [p['q'] for p in P]
    rng.shuffle(qs)
    for p, q in zip(P, qs):
        p['q'] = q
    lr = np.array([math.log(p['q'] / p['c']) for p in P])
    return dict(keys=D['keys'], lr=lr, k=D['k'].copy(), tab=D['tab'].copy(), P=P)


# ---------------------------------------------------------------- data sources
def pe_pairs(capset='apriori'):
    from pe27_common import pe_tablets
    return adjacent_pairs(pe_tablets(capset))


UR_TRUTH = {'gurusz': 'person', 'geme2': 'person', 'erin2': 'person', 'dumu': 'person', 'lu2': 'person',
            'szu-gi4': 'person', 'dumu-munus': 'person', 'munus': 'person', 'nita2': 'person', 'ug3-IL2': 'person',
            'kinkin2': 'person',
            'udu': 'stock', 'u8': 'stock', 'masz2': 'stock', 'ud5': 'stock', 'sila4': 'stock', 'udu-nita2': 'stock',
            'gukkal': 'stock', 'masz': 'stock', 'kir11': 'stock',
            'gu4': 'stock', 'ab2': 'stock', 'amar': 'stock', 'ansze': 'stock', 'dusu2': 'stock'}


def ur3_pairs():
    """Ur III count/capacity neighbours (pe27 extraction), nouns made opaque as U###."""
    d = json.load(open(os.path.join(DATA, 'pe27_ckpt', 'ur3_tabs.json')))
    for t in d:
        for q in t['Q']:
            q['v'] = float(eval(str(q['v']))) if '/' in str(q['v']) else float(q['v'])
            if q['sys'] == 'CNT':
                q['truth'] = UR_TRUTH.get(q['fin'], 'other')
    P = adjacent_pairs(d)
    return P


def opaque(P, seed=7):
    rng = np.random.default_rng(seed)
    names = sorted({p['kc'] for p in P} | {p['kq'] for p in P})
    perm = rng.permutation(len(names))
    m = {n: 'U%03d' % perm[i] for i, n in enumerate(names)}
    out = []
    for p in P:
        p = dict(p)
        p['truth_name'] = p['kc']
        p['kc'] = m[p['kc']]
        p['kq'] = m[p['kq']]
        out.append(p)
    return out, m


AREA_U = {"sar2": 64800, "bur'u": 180, 'bur3': 18, 'esze3': 6, 'iku': 1}
NUMT = re.compile(r"^(\d+(?:/\d+)?)\(([a-z0-9']+)(?:@[a-z])?\)$")


def ur3_area_tabs(rebuild=False):
    """Ur III field texts: area lines (iku; truth 1 iku = 0.36 ha) and neighbouring capacity lines
    (sila; truth ~1 l).  Truth label of the tablet: 'seed' if 'numun' occurs, else 'other'."""
    fn = os.path.join(CK, 'ur3_area.json')
    if os.path.exists(fn) and not rebuild:
        return json.load(open(fn))
    import csv
    from pe27_common import ur_cap, _toks
    csv.field_size_limit(10 ** 9)
    keep = set()
    for row in csv.DictReader(open(os.path.join(SCRATCH, 'cdli_cat.csv'), encoding='utf-8')):
        if row['period'].startswith('Ur III'):
            keep.add('P%06d' % int(row['id_text']))
    texts = defaultdict(list)
    cur = None
    for raw in open(os.path.join(SCRATCH, 'cdli.atf'), encoding='utf-8', errors='replace'):
        if raw.startswith('&P'):
            cur = raw[1:8] if raw[1:8] in keep else None
            continue
        if cur and raw[:1].isdigit():
            m = re.match(r"^\d+'?\.\s+(.*)$", raw.rstrip())
            if m:
                texts[cur].append(m.group(1))
    out = []
    for pid, lines in texts.items():
        full = ' '.join(lines)
        if 'GAN2' not in full:
            continue
        lab = 'seed' if 'numun' in full else 'other'
        Q = []
        for i, body in enumerate(lines):
            if '...' in body or '[' in body or ' x ' in f' {body} ' or '-ta' in body:
                continue
            tk = _toks(body)
            if 'GAN2' in tk:
                j = tk.index('GAN2')
                v, ok = 0.0, True
                for t in tk[:j]:
                    m = NUMT.match(t)
                    if not m or m.group(2) not in AREA_U:
                        ok = False
                        break
                    v += eval(m.group(1)) * AREA_U[m.group(2)]
                if ok and v > 0 and j > 0:
                    rest = [t for t in tk[j + 1:] if not NUMT.match(t)]
                    Q.append({'sys': 'CNT', 'v': v, 'fin': 'GAN2', 'line': i, 'truth': 'area', 'field': rest[0] if rest else '-'})
                continue
            c = ur_cap(tk[1:] if tk and tk[0] in ('sze-bi', 'sze', 'ziz2-bi') else tk)
            if c:
                noun = [t for t in tk if not NUMT.match(t)]
                Q.append({'sys': 'CAP', 'v': float(c), 'fin': noun[0] if noun else '-', 'line': i})
        if {'CNT', 'CAP'} <= {q['sys'] for q in Q}:
            out.append({'id': pid, 'lab': lab, 'Q': Q})
    json.dump(out, open(fn, 'w'))
    return out


def lab_name(a):
    return 'none' if a < 0 else ACTS[a]
