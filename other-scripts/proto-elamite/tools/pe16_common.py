"""pe16 'hunger is the ruler': shared data + physiology + likelihood engine.

Idea: a per-person allotment = (need in litres/day) x (period D days) / (unit size u litres),
so the recorded amount in base units a = m_c * D / u * scatter.  Human and animal biology
gives m_c (litres of grain per day per recipient class) from outside the corpus.
Only theta = log(D/u) is identified by amounts alone; D and u separate only with an
outside yardstick (here: measured vessel volumes, used as a check).

Data: PE capacity entries (pe_corpus.json); Ur III distributive ration lines (CDLI bulk
ATF in the scratchpad, cached as JSON in data/pe16_ckpt/); proto-cuneiform (pe2_pc_corpus).
"""
import json, os, re, sys, math
from collections import Counter, defaultdict
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import load, base, is_sign, C_CODES  # noqa

DATA = os.path.join(HERE, '..', 'data')
CK = os.path.join(DATA, 'pe16_ckpt')
os.makedirs(CK, exist_ok=True)
SCRATCH = '/tmp/claude-0/-home-user-Indus-/874df4c7-80d6-5f08-b42c-eea96a214079/scratchpad'

# ---------------------------------------------------------------- PE capacity values
# a-priori set fixed in attack_arith (from the writing of the signs, before any totals),
# in units of N39C.  Alternatives = other orderings that the totals could not exclude.
VSETS = {
    'apriori': {'N39C': 1, 'N30D': 2, 'N30C': 4, 'N24': 12, 'N39B': 24, 'N01': 120, 'N14': 720},
    'alt_a':   {'N39C': 1, 'N30D': 2, 'N30C': 4, 'N24': 8, 'N39B': 24, 'N01': 120, 'N14': 720},
    'alt_b':   {'N39C': 1, 'N30D': 2, 'N30C': 4, 'N24': 12, 'N39B': 24, 'N01': 144, 'N14': 1440},
    'alt_c':   {'N39C': 1, 'N30D': 3, 'N30C': 6, 'N24': 12, 'N39B': 24, 'N01': 120, 'N14': 1200},
    'alt_d':   {'N39C': 1, 'N30D': 2, 'N30C': 6, 'N24': 12, 'N39B': 60, 'N01': 300, 'N14': 1800},
}
CAP_CLASS = {'M297', 'M002', 'M036', 'M243', 'M106', 'M075', 'M010', 'M379', 'M050'}


def cap_value(numerals, vs):
    tot = 0
    for n, c in numerals:
        c0 = c.split('@')[0]
        if not isinstance(n, int) or c0 not in vs:
            return None
        tot += n * vs[c0]
    return tot if tot > 0 else None


def pe_capacity_entries(vs_name='apriori'):
    """Every PE line with signs and a numeral group made only of capacity-system codes,
    containing at least one diagnostic C code OR headed by a capacity class sign on a
    tablet that has C codes.  Returns dicts: tablet, idx, signs, label (final base sign),
    value (N39C units), codes."""
    vs = VSETS[vs_name]
    out = []
    for t in load():
        tab_c = any({c.split('@')[0] for _, c in l['numerals']} & C_CODES for l in t['lines'])
        if not tab_c:
            continue
        ents = [l for l in t['lines'] if l['numerals'] and any(is_sign(s) for s in l['signs'])]
        for i, l in enumerate(ents):
            codes = {c.split('@')[0] for _, c in l['numerals']}
            sg = [base(s) for s in l['signs'] if is_sign(s)]
            if not ((codes & C_CODES) or (sg and sg[-1] in CAP_CLASS)):
                continue
            v = cap_value(l['numerals'], vs)
            if v is None:
                continue
            out.append({'tab': t['id'], 'idx': i, 'signs': sg, 'label': sg[-1], 'value': v,
                        'raw': l['raw'], 'codes': sorted(codes), 'n_signs': len(sg),
                        'nums': l['numerals'], 'damaged': l.get('damaged', False)})
    return out


def pe_record_entries(vs_name='apriori', min_rep=3):
    """Capacity lines that sit inside fixed two-line records: tablets where one entry string
    recurs at every second position >= min_rep times.  Returns main + companion capacity
    lines with role tags."""
    vs = VSETS[vs_name]
    out = []
    for t in load():
        ents = [l for l in t['lines'] if l['numerals'] and any(is_sign(s) for s in l['signs'])]
        S = [tuple(base(s) for s in l['signs'] if is_sign(s)) for l in ents]
        cnt = Counter(S[i] for i in range(len(S) - 2) if S[i] == S[i + 2] and S[i + 1] != S[i])
        if not cnt:
            continue
        comp, k = cnt.most_common(1)[0]
        if k < min_rep:
            continue
        for i, l in enumerate(ents):
            v = cap_value(l['numerals'], vs)
            codes = {c.split('@')[0] for _, c in l['numerals']}
            role = 'comp' if S[i] == comp else 'main'
            out.append({'tab': t['id'], 'idx': i, 'role': role, 'signs': list(S[i]),
                        'label': (S[i][-1] if S[i] else '?'), 'value': v, 'raw': l['raw'],
                        'capacity': bool(codes & C_CODES), 'nums': l['numerals']})
    return out


# ---------------------------------------------------------------- Ur III control
UR3_UNITS = {'gur': 300, 'barig': 60, 'ban2': 10}
FRAC = {'1/2': 0.5, '1/3': 1 / 3, '2/3': 2 / 3, '5/6': 5 / 6}


def _num(s):
    s = s.strip()
    if s in FRAC:
        return FRAC[s]
    try:
        return float(s)
    except ValueError:
        return None


def parse_ur3_amount(txt):
    """'1(ban2) 5(disz) sila3' -> 15 (sila).  Returns None if unparseable."""
    txt = re.sub(r'[\[\]#?!<>]', '', txt)
    tot = 0.0
    toks = txt.split()
    i = 0
    seen = False
    while i < len(toks):
        m = re.match(r'^([\d/]+)\((barig|ban2|disz|asz|u)\)$', toks[i])
        if not m:
            return None
        n = _num(m.group(1))
        if n is None:
            return None
        unit = m.group(2)
        if unit in ('barig', 'ban2'):
            tot += n * UR3_UNITS[unit]; seen = True; i += 1
        else:  # disz/asz/u followed by sila3 / gin2 / gur
            mult = 10 if unit == 'u' else 1
            val = n * mult
            # collect following u/disz
            j = i + 1
            while j < len(toks) and re.match(r'^[\d/]+\((disz|u|asz)\)$', toks[j]):
                m2 = re.match(r'^([\d/]+)\((disz|u|asz)\)$', toks[j])
                n2 = _num(m2.group(1))
                if n2 is None:
                    return None
                val += n2 * (10 if m2.group(2) == 'u' else 1)
                j += 1
            if j >= len(toks):
                return None
            w = toks[j]
            if w.startswith('sila3'):
                tot += val
            elif w.startswith('gin2'):
                tot += val / 60.0
            elif w.startswith('gur'):
                tot += val * 300
            else:
                return None
            seen = True
            i = j + 1
    return tot if seen and tot > 0 else None


UR3_LINE = re.compile(r'^\d+\'?\.\s+(?P<count>(?:[\d/]+\((?:disz|u|gesz2|asz)\)\s+(?:la2\s+)?)+)'
                      r'(?P<noun>.+?)\s+(?P<amt>(?:[\d/]+\((?:barig|ban2|disz|u|asz)\)\s*(?:sila3|gin2|gur)?\s*)+?)'
                      r'(?:sila3|gin2|gur)?-ta\b')


def ur3_class(noun):
    n = re.sub(r'[\[\]#?!<>]', '', noun)
    w = n.split()
    for k, cls in (('gurusz', 'man'), ('geme2', 'woman'), ('dumu', 'child'), ('lu2', 'man'),
                   ('udu', 'sheep'), ('ud5', 'goat'), ('u8', 'sheep'),
                   ('gu4', 'cattle'), ('ab2', 'cattle'), ('ansze', 'equid'), ('dusu2', 'equid'),
                   ('sila4', 'lamb'), ('amar', 'calf')):
        if k in w:
            return cls, k
    return None, w[-1] if w else '?'


def ur3_entries(rebuild=False):
    fn = os.path.join(CK, 'ur3_ta.json')
    if os.path.exists(fn) and not rebuild:
        return json.load(open(fn))
    import csv
    csv.field_size_limit(10 ** 9)
    keep = {}
    for row in csv.DictReader(open(os.path.join(SCRATCH, 'cdli_cat.csv'), encoding='utf-8')):
        if row['period'].startswith('Ur III'):
            keep['P%06d' % int(row['id_text'])] = row['provenience'][:20]
    out = []
    cur = None
    for raw in open(os.path.join(SCRATCH, 'cdli.atf'), encoding='utf-8', errors='replace'):
        if raw.startswith('&P'):
            pid = raw[1:8]
            cur = pid if pid in keep else None
            continue
        if cur is None or not raw[:1].isdigit() or '-ta' not in raw:
            continue
        line = raw.rstrip()
        m = re.match(r'^\d+\'?\.\s+(.*)$', line)
        if not m:
            continue
        body = m.group(1)
        # amount = maximal run of numeral tokens (+ unit words) immediately before '-ta'
        mm = re.search(r'((?:[\d/]+\((?:barig|ban2|disz|u|asz)\)\s*(?:sila3|gin2|gur)?\s*)+?(?:sila3|gin2|gur)?)-ta\b', body)
        if not mm:
            continue
        amt_txt = mm.group(1)
        amt_txt = re.sub(r'\s+', ' ', amt_txt).strip()
        # the amount must contain a capacity unit word or barig/ban2
        if not re.search(r'barig|ban2|sila3|gur', amt_txt + body[mm.end() - 3:mm.end()]):
            continue
        a = parse_ur3_amount(amt_txt if re.search(r'(sila3|gin2|gur)$', amt_txt) or
                             re.search(r'(barig|ban2)\)$', amt_txt) else amt_txt + ' sila3')
        if a is None:
            continue
        pre = body[:mm.start()].strip()
        cls, noun = ur3_class(pre)
        if cls is None:
            continue
        out.append({'tab': cur, 'prov': keep[cur], 'cls': cls, 'noun': noun, 'value': a,
                    'raw': body})
    json.dump(out, open(fn, 'w'))
    return out


# ---------------------------------------------------------------- physiology (outside yardstick)
# Litres of grain per day.  Inputs are biology, not readings of any text:
#  human energy need (kcal/day) by class, share of energy from grain f, an allotment factor g
#  (rations also feed dependants / pay; broad prior, calibrated on Ur III), barley energy
#  density E (kcal/kg) and bulk density rho (kg/l).  Animals: grain/concentrate fed per day
#  (kg), broad log-normal priors from animal-husbandry ranges.
HUMAN_KCAL = {'man': (3000, 400), 'woman': (2300, 300), 'child': (1500, 350)}
ANIMAL_KG = {'sheep': (0.4, 0.5), 'equid': (2.0, 0.45), 'cattle': (3.0, 0.5)}  # median, log-sd
CLASSES = ['man', 'woman', 'child', 'sheep', 'equid', 'cattle']


def draw_physiology(J, rng, classes=CLASSES):
    """Return [J, C] litres/day per class + [J] scatter sigma."""
    E = rng.uniform(3200, 3500, J)
    rho = rng.uniform(0.58, 0.70, J)
    f = rng.uniform(0.6, 0.95, J)
    g = np.exp(rng.normal(np.log(1.3), 0.35, J))
    out = np.zeros((J, len(classes)))
    for k, c in enumerate(classes):
        if c in HUMAN_KCAL:
            mu, sd = HUMAN_KCAL[c]
            kcal = np.clip(rng.normal(mu, sd, J), 600, None)
            out[:, k] = kcal * f * g / (E * rho)
        else:
            med, lsd = ANIMAL_KG[c]
            out[:, k] = np.exp(rng.normal(np.log(med), lsd, J)) / rho
    sig = rng.uniform(0.15, 0.6, J)
    return out, sig


# ---------------------------------------------------------------- likelihood engine
TH = np.linspace(np.log(1e-4), np.log(1e5), 461)   # theta = log(D/u)


def loglik_grid(values, labels, M, sig, eps=0.05, pi=None, rng=None, mode='label',
                kprior=None):
    """values: amounts in base units; labels: group ids (class shared within a label).
    M: [J, C] litres/day; sig: [J].  Returns [J, G] log-likelihood over theta grid.
    mode 'label': class latent per label; 'entry': per entry.
    kprior: None (k=1) or (ks, logw) = multiplicities with log prior weights."""
    la = np.log(np.asarray(values, float))
    span = la.max() - la.min() + 2.0
    lu = -np.log(span)  # uniform outlier density on log scale
    lab = np.asarray(labels)
    uniq, inv = np.unique(lab, return_inverse=True)
    J, C = M.shape
    G = len(TH)
    out = np.zeros((J, G))
    if rng is None:
        rng = np.random.default_rng(0)
    for j in range(J):
        p = pi[j] if pi is not None else rng.dirichlet(np.ones(C))
        mu = np.log(M[j])[None, :, None] + TH[None, None, :]   # [1, C, G]
        x = la[:, None, None]
        if kprior is None:
            z = (x - mu) / sig[j]
            lg = -0.5 * z * z - np.log(sig[j] * np.sqrt(2 * np.pi))   # [n, C, G]
        else:
            ks, lw = kprior
            lgk = []
            for kk, w in zip(ks, lw):
                z = (x - mu - np.log(kk)) / sig[j]
                lgk.append(-0.5 * z * z - np.log(sig[j] * np.sqrt(2 * np.pi)) + w)
            lg = np.logaddexp.reduce(np.stack(lgk), axis=0)
        lg = np.logaddexp(lg + np.log(1 - eps), np.log(eps) + lu)
        if mode == 'entry':
            tot = np.logaddexp.reduce(lg + np.log(p)[None, :, None], axis=1).sum(0)
        else:
            S = np.zeros((len(uniq), C, G))
            np.add.at(S, inv, lg)
            tot = np.logaddexp.reduce(S + np.log(p)[None, :, None], axis=1).sum(0)
        out[j] = tot
    return out


def theta_post(LL):
    """Marginal posterior over theta (flat prior on grid) and log evidence."""
    J = LL.shape[0]
    lm = np.logaddexp.reduce(LL, axis=0) - np.log(J)
    ev = np.logaddexp.reduce(lm) - np.log(len(TH))
    p = np.exp(lm - lm.max())
    p /= p.sum()
    return p, ev


def summarize(p):
    c = np.cumsum(p)
    q = lambda a: float(np.exp(TH[np.searchsorted(c, a)]))
    return {'map': float(np.exp(TH[np.argmax(p)])), 'q05': q(0.05), 'q50': q(0.5), 'q95': q(0.95)}


# ---------------------------------------------------------------- v2 engine: period per tablet
LU = np.linspace(np.log(1e-4), np.log(1e4), 241)    # log u (litres per base unit)
PERIODS = (1.0, 30.0, 360.0)


def prep(entries, label_key='label'):
    """entries -> (log a [n], group index [n], tablet-of-group [g], n tablets)."""
    la = np.log(np.array([e['value'] for e in entries], float))
    gk = [(e['tab'], e[label_key]) for e in entries]
    gu = {k: i for i, k in enumerate(dict.fromkeys(gk))}
    gi = np.array([gu[k] for k in gk])
    tabs = {t: i for i, t in enumerate(dict.fromkeys(k[0] for k in gu))}
    gt = np.array([tabs[k[0]] for k in gu])
    return la, gi, gt, len(tabs)


def loglik_v2(la, gi, gt, T, M, sig, rng, periods=PERIODS, wD=None, eps=0.05, kprior=None,
              per_tab=False):
    """[J, G] log-likelihood over log u.  Class latent per (tablet, label) group; period latent
    per tablet (prior wD, default uniform); entries scatter log-normally (sigma) around
    log(m_c * D / u) (+ log k for multiplicity prior kprior=(ks, logw)); eps outliers."""
    J, C = M.shape
    G = len(LU)
    P = len(periods)
    lD = np.log(np.asarray(periods))
    if wD is None:
        wD = np.full(P, 1.0 / P)
    lw = np.log(wD)
    span = la.max() - la.min() + 2.0
    lo = np.log(eps) - np.log(span)
    ng = gi.max() + 1
    out = np.zeros((J, G), np.float64)
    tabpost = np.zeros((T, P)) if per_tab else None
    x = la.astype(np.float32)[:, None, None, None]
    for j in range(J):
        pi = rng.dirichlet(np.ones(C))
        mu = (np.log(M[j])[None, :, None, None] + lD[None, None, None, :]
              - LU[None, None, :, None]).astype(np.float32)          # [1, C, G, P]
        s = np.float32(sig[j])
        if kprior is None:
            z = (x - mu) / s
            f = -0.5 * z * z - np.float32(np.log(s * np.sqrt(2 * np.pi)))
        else:
            ks, lwk = kprior
            f = None
            for kk, wk in zip(ks, lwk):
                z = (x - mu - np.float32(np.log(kk))) / s
                fk = -0.5 * z * z - np.float32(np.log(s * np.sqrt(2 * np.pi))) + np.float32(wk)
                f = fk if f is None else np.logaddexp(f, fk)
        f = np.logaddexp(f + np.float32(np.log(1 - eps)), np.float32(lo))  # [n, C, G, P]
        S = np.zeros((ng,) + f.shape[1:], np.float32)
        np.add.at(S, gi, f)
        S = np.logaddexp.reduce(S + np.log(pi).astype(np.float32)[None, :, None, None], axis=1)  # [ng, G, P]
        TT = np.zeros((T, G, P), np.float32)
        np.add.at(TT, gt, S)
        tot = np.logaddexp.reduce(TT + lw[None, None, :].astype(np.float32), axis=2)   # [T, G]
        out[j] = tot.sum(0)
        if per_tab:
            tabpost += 0  # filled by caller via tab_period_post
    return out


def post_u(LL):
    J = LL.shape[0]
    lm = np.logaddexp.reduce(LL, axis=0) - np.log(J)
    ev = float(np.logaddexp.reduce(lm) - np.log(len(LU)))
    p = np.exp(lm - lm.max()); p /= p.sum()
    return p, ev


def summ_u(p):
    c = np.cumsum(p)
    q = lambda a: float(np.exp(LU[min(np.searchsorted(c, a), len(LU) - 1)]))
    return {'map': float(np.exp(LU[np.argmax(p)])), 'q05': q(0.05), 'q25': q(0.25),
            'q50': q(0.5), 'q75': q(0.75), 'q95': q(0.95)}


def random_ruler(rng, C=6):
    """A nonsense yardstick of the same shape: C random class needs (log-uniform 0.05-10 l/day
    medians) and a random 3-period calendar {1, a, b} (a, b log-uniform 2-1000 days)."""
    med = np.exp(rng.uniform(np.log(0.05), np.log(10), C))
    a, b = np.sort(np.exp(rng.uniform(np.log(2), np.log(1000), 2)))
    return med, (1.0, float(a), float(b))


def draw_ruler(J, rng, med):
    """Nuisance draws around a ruler's class medians (log-sd 0.3), same sigma prior."""
    M = med[None, :] * np.exp(rng.normal(0, 0.3, (J, len(med))))
    return M, rng.uniform(0.15, 0.6, J)
