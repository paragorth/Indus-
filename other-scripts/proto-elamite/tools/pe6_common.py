"""pe6: grow a fake Susa and make it write tablets.  Shared code.

An agent-based early-state economy (offices, named units = workers/herds, goods, rations, deliveries)
writes synthetic tablets with a small set of scribal practices.  Every sign in the simulation is a ROLE
with no shape.  The simulator is fitted to the real Proto-Elamite corpus by approximate Bayesian
computation (rejection + local-linear regression) on a battery of summary statistics; real signs are then
mapped to roles by a classifier trained on posterior simulations (posterior assignment).

Common representation of a corpus (real or simulated):
  tablet = dict(hdr=tuple|None, ent=[(signs tuple, sys, val)], tot=(signs, sys, val)|None, frag=bool)
  sys in {'S' counted, 'C' capacity, 'F' counted with fraction, 'O' other/unknown}
  val = float (D3 counting values / a-priori notation capacity values in N39C units, FINDINGS attack 2) or None
Simulated signs are strings '<ROLE>:<k>'; 'x' marks an illegible token.  Nothing here interprets a sign.
"""
import bisect, json, math, os, sys, random
from collections import Counter, defaultdict
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
PEDATA = os.path.join(HERE, '..', 'data')
ROLES = ['OPEN', 'PRE', 'CLS_S', 'CLS_C', 'CLS_F', 'CLS_O', 'SYL']

# ---------------------------------------------------------------------------------------------- real corpus
def load_real(site=None):
    from common import load, base, is_sign
    from pe5_common import pe_system, pe_value, VSETS, CSETS
    FR = {'N02', 'N08', 'N08A', 'N8A', 'N8B'}
    T = load()
    out = []
    for t in T:
        if site and not t['provenience'].startswith(site):
            continue
        L = t['lines']
        if not L:
            continue
        sg = lambda l: tuple('x' if s in ('x', 'X', 'n') else base(s) for s in l['signs']
                             if is_sign(s) or s in ('x', 'X', 'n'))
        hdr = None
        start = 0
        if not L[0]['numerals'] and sg(L[0]):
            hdr = sg(L[0]); start = 1
        nobv = sum(1 for l in L if l['surface'] == 'obverse' and l['numerals'])
        off = [l for l in L if l['surface'] != 'obverse' and l['numerals']]
        totl = off[0] if (len(off) == 1 and nobv >= 1) else None

        def rec(l):
            s = pe_system(l['numerals'])
            if s is None:
                return (sg(l), 'O', None)
            if s == 'S' and any(c in FR for _, c in l['numerals']):
                s2 = 'F'
            else:
                s2 = s
            v = pe_value(l['numerals'], s, VSETS['D3'], CSETS['NOT'])
            return (sg(l), s2, v)
        ent = [rec(l) for l in L[start:] if l['numerals'] and l is not totl]
        tot = rec(totl) if totl is not None else None
        frag = any(l['lacuna'] for l in L)
        out.append(dict(id=t['id'], hdr=hdr, ent=ent, tot=tot, frag=frag))
    return out


# ---------------------------------------------------------------------------------------------- priors
# name: (kind, lo, hi)   kind 'u' uniform, 'l' log-uniform, 'i' integer uniform, 'li' log-uniform integer
PRIOR = [
    ('n_off', 'li', 2, 60),        # offices
    ('w_mean', 'l', 2, 150),       # named units (workers / herds) per office
    ('a_off', 'u', 0.0, 1.5),      # Zipf exponent of office tablet output
    ('n_syl', 'li', 20, 1500),      # signs used to spell unit names
    ('a_syl', 'u', 0.3, 1.6),
    ('name_len', 'u', 1.0, 4.0),   # mean signs per name
    ('p_oneoff', 'u', 0.0, 1.0),   # a named entry is a one-off name (not from the office roster)
    ('p_poly', 'u', 0.0, 0.6),     # share of opener/prefix/class signs ALSO used as name signs (polyvalence)
    ('n_open', 'li', 1, 40),
    ('a_open', 'u', 0.3, 2.5),
    ('p_open_off', 'u', 0.0, 1.0), # header opener is the office's own opener (else global Zipf draw)
    ('p_hdr', 'u', 0.1, 0.95),
    ('p_hdr_name', 'u', 0.0, 0.8),
    ('p_hdr_open', 'u', 0.2, 1.0), # header starts with an opener (else it is a bare name) # header also names the office/unit
    ('n_pre', 'li', 1, 25),
    ('a_pre', 'u', 0.3, 2.0),
    ('p_pre', 'u', 0.0, 0.5),      # entry carries a qualifier prefix
    ('n_cs', 'li', 2, 60),         # counted goods (class signs)
    ('n_cc', 'li', 1, 30),         # capacity goods
    ('n_cf', 'i', 0, 6),           # fraction-taking counted goods
    ('n_co', 'i', 0, 8),           # goods in other notation systems
    ('a_goods', 'u', 0.3, 2.0),
    ('k_goods', 'li', 1, 12),      # goods handled per office
    ('p_named', 'u', 0.0, 1.0),    # tablet lists named units (else items)
    ('p_cls', 'u', 0.0, 1.0),      # named entry also carries the class sign
    ('p_mix', 'u', 0.0, 0.6),      # entry good differs from tablet main good
    ('ent_mean', 'l', 1.0, 20.0),  # entries per tablet
    ('k_ent', 'l', 0.2, 10.0),     # negative-binomial dispersion
    ('cnt_mu', 'l', 1.0, 100.0),   # median counted quantity
    ('cnt_sd', 'u', 0.2, 2.0),     # log-sd
    ('rat_mu', 'l', 1.0, 3000.0),  # median capacity quantity (N39C units of the a-priori notation set)
    ('rat_sd', 'u', 0.2, 2.5),
    ('p_bare', 'u', 0.0, 0.5),     # entry written as a bare numeral under the previous designation
    ('p_tot', 'u', 0.0, 1.0),
    ('p_totsign', 'u', 0.0, 1.0),
    ('p_err', 'u', 0.0, 0.8),      # written total differs from the sum
    ('p_break', 'u', 0.1, 0.9),
    ('p_x', 'u', 0.03, 0.25),
]
PNAMES = [p[0] for p in PRIOR]


def draw_prior(rng):
    th = {}
    for n, k, lo, hi in PRIOR:
        if k == 'u':
            th[n] = rng.uniform(lo, hi)
        elif k == 'l':
            th[n] = math.exp(rng.uniform(math.log(lo), math.log(hi)))
        elif k == 'i':
            th[n] = rng.randint(lo, hi)
        elif k == 'li':
            th[n] = int(round(math.exp(rng.uniform(math.log(lo), math.log(hi + 0.49)))))
            th[n] = min(max(th[n], lo), hi)
    return th


def to_unbounded(th):
    v = []
    for n, k, lo, hi in PRIOR:
        x = th[n]
        if k in ('l', 'li'):
            u = (math.log(max(x, lo)) - math.log(lo)) / (math.log(hi + (0.49 if k == 'li' else 0)) - math.log(lo))
        else:
            u = (x - lo + (0.5 if k == 'i' else 0)) / (hi - lo + (1 if k == 'i' else 0))
        u = min(max(u, 1e-4), 1 - 1e-4)
        v.append(math.log(u / (1 - u)))
    return np.array(v)


def from_unbounded(v):
    th = {}
    for (n, k, lo, hi), z in zip(PRIOR, v):
        u = 1 / (1 + math.exp(-z))
        if k in ('l', 'li'):
            x = math.exp(math.log(lo) + u * (math.log(hi + (0.49 if k == 'li' else 0)) - math.log(lo)))
            th[n] = min(max(int(round(x)), lo), hi) if k == 'li' else x
        elif k == 'i':
            th[n] = min(max(int(math.floor(lo + u * (hi - lo + 1))), lo), hi)
        else:
            th[n] = lo + u * (hi - lo)
    return th


# ---------------------------------------------------------------------------------------------- simulator
def zipf_w(n, a):
    w = np.arange(1, n + 1, dtype=float) ** (-a)
    return w / w.sum()


def simulate(th, seed, n_tab=1585):
    """Return (corpus, role_of_sign).  seed: int (python + numpy generators)."""
    r = random.Random(seed); g = np.random.default_rng(seed)
    role = {}
    def mk(rl, n):
        ids = [f'{rl}:{i}' for i in range(n)]
        for x in ids:
            role[x] = rl
        return ids
    OPEN = mk('OPEN', th['n_open']); PRE = mk('PRE', th['n_pre']); SYL = mk('SYL', th['n_syl'])
    goods = []   # (class sign, system)
    for rl, n, sy in (('CLS_S', th['n_cs'], 'S'), ('CLS_C', th['n_cc'], 'C'),
                      ('CLS_F', th['n_cf'], 'F'), ('CLS_O', th['n_co'], 'O')):
        for x in mk(rl, n):
            goods.append((x, sy))
    G = len(goods)
    perm = g.permutation(G)                         # goods popularity independent of system
    gw = np.empty(G); gw[perm] = zipf_w(G, th['a_goods'])
    cum = lambda w: list(np.cumsum(w))
    SYLp = [SYL[i] for i in g.permutation(len(SYL))]
    nonsyl = OPEN + PRE + [x for x, _ in goods]
    npoly = int(round(th['p_poly'] * len(nonsyl)))
    for x in g.choice(len(nonsyl), size=npoly, replace=False) if npoly else []:
        SYLp.insert(int(g.integers(len(SYLp) + 1)), nonsyl[x])   # same shape, second use in names
    sylc = cum(zipf_w(len(SYLp), th['a_syl']))
    opc = cum(zipf_w(len(OPEN), th['a_open'])); prc = cum(zipf_w(len(PRE), th['a_pre']))
    pick = lambda c: min(bisect.bisect_left(c, r.random() * c[-1]), len(c) - 1)
    p_geo = 1.0 / th['name_len']

    def name():
        L = 1
        if p_geo < 1:
            while r.random() > p_geo and L < 8:
                L += 1
        return tuple(SYLp[pick(sylc)] for _ in range(L))

    offices = []
    for o in range(th['n_off']):
        W = max(1, int(r.lognormvariate(math.log(th['w_mean']), 0.5)))
        roster = [name() for _ in range(W)]
        k = min(G, max(1, int(round(th['k_goods'] * r.uniform(0.5, 1.5)))))
        port = [int(x) for x in g.choice(G, size=k, replace=False, p=gw)]
        offices.append(dict(roster=roster, port=port, opener=OPEN[pick(opc)], oname=name()))
    offc = cum(zipf_w(th['n_off'], th['a_off']))
    m = th['ent_mean']; kk = th['k_ent']
    if m > 1.0:
        NE = 1 + g.negative_binomial(kk, kk / (kk + (m - 1)), size=n_tab)
    else:
        NE = np.ones(n_tab, int)
    NE = np.minimum(NE, 80)
    lmc, sdc = math.log(th['cnt_mu']), th['cnt_sd']
    lmr, sdr = math.log(th['rat_mu']), th['rat_sd']
    px = th['p_x']; pb = th['p_bare']
    corpus = []
    for ti in range(n_tab):
        off = offices[pick(offc)]
        port = off['port']
        g0 = port[r.randrange(len(port))]
        ne = int(NE[ti])
        named = r.random() < th['p_named']
        if named:
            R = off['roster']
            idx = r.sample(range(len(R)), ne) if ne <= len(R) else [r.randrange(len(R)) for _ in range(ne)]
        ent = []
        for j in range(ne):
            gi = g0 if (len(port) == 1 or r.random() >= th['p_mix']) else port[r.randrange(len(port))]
            cs, sy = goods[gi]
            des = []
            if j > 0 and r.random() < pb:
                pass                                   # bare numeral sub-line (designation implied)
            else:
                if r.random() < th['p_pre']:
                    des.append(PRE[pick(prc)])
                if named:
                    des.extend(name() if r.random() < th['p_oneoff'] else R[idx[j]])
                    if r.random() < th['p_cls']:
                        des.append(cs)
                else:
                    des.append(cs)
            if sy == 'S' or sy == 'F':
                v = max(1, int(round(r.lognormvariate(lmc, sdc))))
                if sy == 'F' and r.random() < 0.6:
                    v += 0.5
            elif sy == 'C':
                v = max(1, int(round(r.lognormvariate(lmr, sdr))))
            else:
                v = None
            ent.append((tuple(des), sy, v))
        tot = None
        if ne >= 2 and r.random() < th['p_tot']:
            ds = Counter(e[1] for e in ent).most_common(1)[0][0]
            v = None if ds == 'O' else sum(e[2] for e in ent if e[1] == ds)
            if v is not None and r.random() < th['p_err']:
                v = max(1, v + r.choice((-1, 1)) * max(1, int(round(v * r.uniform(0.02, 0.3)))))
            ts = (goods[g0][0],) if r.random() < th['p_totsign'] else ()
            tot = (ts, ds, v)
        hdr = None
        if r.random() < th['p_hdr']:
            op = off['opener'] if r.random() < th['p_open_off'] else OPEN[pick(opc)]
            if r.random() < th['p_hdr_open']:
                hdr = (op,) + (off['oname'] if r.random() < th['p_hdr_name'] else ())
            else:
                hdr = off['oname'] if r.random() < 0.5 else name()
        frag = False
        if r.random() < th['p_break']:
            frag = True
            keep = max(1, int(math.ceil(len(ent) * r.uniform(0.2, 1.0))))
            if keep < len(ent):
                ent = ent[:keep]; tot = None
            if r.random() < 0.3:
                hdr = None
            if r.random() < 0.3:
                tot = None
        rr = r.random
        dmg = lambda sg: tuple('x' if rr() < px else x for x in sg)
        ent = [(dmg(sg), sy, v) for sg, sy, v in ent]
        if hdr:
            hdr = dmg(hdr)
        if tot:
            tot = (dmg(tot[0]), tot[1], tot[2])
        corpus.append(dict(hdr=hdr, ent=ent, tot=tot, frag=frag))
    return corpus, role


# ---------------------------------------------------------------------------------------------- statistics
STATS = ['L_mean', 'L_one', 'L_ge10', 'L_med',
         'E_len1', 'E_len2', 'E_len3', 'E_len4', 'E_nosign', 'E_fincon', 'E_inicon', 'E_finbias', 'E_inibias',
         'N_S', 'N_C', 'N_F', 'N_O', 'N_mixtab', 'N_MIfin', 'N_Smu', 'N_Ssd', 'N_Cmu', 'N_Csd', 'N_S1',
         'T_share', 'T_signed', 'T_headcls', 'T_adds',
         'H_share', 'H_len', 'H_top', 'H_div', 'H_excl', 'H_MIsys',
         'D_uniq', 'D_multi', 'D_types', 'D_zipf', 'D_hapax', 'D_finrep', 'D_inrep',
         'X_x', 'X_frag']
GROUPS = {'L': 'tablet length', 'E': 'entry structure', 'N': 'number-system mix', 'T': 'totals',
          'H': 'header/entry relations', 'D': 'designation recurrence', 'X': 'damage (nuisance)'}


def mi_bits(pairs):
    n = len(pairs)
    if n == 0:
        return 0.0
    a = Counter(p[0] for p in pairs); b = Counter(p[1] for p in pairs); ab = Counter(pairs)
    return sum(c / n * math.log2(c * n / (a[x] * b[y])) for (x, y), c in ab.items())


def stats(C):
    nt = len(C)
    ne = np.array([len(t['ent']) for t in C])
    s = {}
    s['L_mean'] = ne.mean(); s['L_one'] = (ne == 1).mean(); s['L_ge10'] = (ne >= 10).mean()
    s['L_med'] = float(np.median(ne))
    E = [e for t in C for e in t['ent']]
    ln = np.array([len(e[0]) for e in E])
    s['E_len1'] = (ln == 1).mean(); s['E_len2'] = (ln == 2).mean(); s['E_len3'] = (ln == 3).mean()
    s['E_len4'] = (ln >= 4).mean(); s['E_nosign'] = (ln == 0).mean()
    fin = Counter(); ini = Counter(); tok = Counter()
    for sg, _, _ in E:
        if len(sg) >= 2:
            for x in sg:
                if x != 'x':
                    tok[x] += 1
            if sg[-1] != 'x': fin[sg[-1]] += 1
            if sg[0] != 'x': ini[sg[0]] += 1
    nf = sum(fin.values()) or 1; ni = sum(ini.values()) or 1
    s['E_fincon'] = sum(c for _, c in fin.most_common(15)) / nf
    s['E_inicon'] = sum(c for _, c in ini.most_common(10)) / ni
    big = [x for x in tok if tok[x] >= 20]
    s['E_finbias'] = np.mean([fin[x] / tok[x] >= 0.6 for x in big]) if big else 0.0
    s['E_inibias'] = np.mean([ini[x] / tok[x] >= 0.6 for x in big]) if big else 0.0
    sy = Counter(e[1] for e in E); n = len(E) or 1
    for k in 'SCFO':
        s['N_' + k] = sy[k] / n
    mix = [t for t in C if len(t['ent']) >= 3]
    s['N_mixtab'] = np.mean([len({e[1] for e in t['ent']} & {'S', 'C'}) == 2 for t in mix]) if mix else 0.0
    fc = Counter(e[0][-1] for e in E if e[0] and e[0][-1] != 'x')
    s['N_MIfin'] = mi_bits([(e[0][-1], e[1]) for e in E if e[0] and e[0][-1] != 'x' and fc[e[0][-1]] >= 10])
    Sv = np.log10([e[2] for e in E if e[1] == 'S' and e[2] and e[2] > 0] or [1])
    Cv = np.log10([e[2] for e in E if e[1] == 'C' and e[2] and e[2] > 0] or [1])
    s['N_Smu'] = Sv.mean(); s['N_Ssd'] = Sv.std(); s['N_Cmu'] = Cv.mean(); s['N_Csd'] = Cv.std()
    s['N_S1'] = (Sv == 0).mean()
    t2 = [t for t in C if len(t['ent']) >= 2]
    tots = [t for t in t2 if t['tot']]
    s['T_share'] = len(tots) / (len(t2) or 1)
    s['T_signed'] = np.mean([len(t['tot'][0]) > 0 for t in tots]) if tots else 0.0
    hc = []
    for t in tots:
        if t['tot'][0] and t['tot'][0][0] != 'x':
            f = Counter(e[0][-1] for e in t['ent'] if e[0] and e[0][-1] != 'x')
            if f:
                hc.append(t['tot'][0][0] == f.most_common(1)[0][0])
    s['T_headcls'] = np.mean(hc) if hc else 0.0
    ad = []
    for t in tots:
        if t['frag'] or t['tot'][1] != 'S' or t['tot'][2] is None:
            continue
        vv = [e[2] for e in t['ent'] if e[1] == 'S']
        if len(vv) == len(t['ent']) and all(v is not None for v in vv):
            ad.append(abs(sum(vv) - t['tot'][2]) < 1e-6)
    s['T_adds'] = np.mean(ad) if ad else 0.0
    H = [t for t in C if t['hdr']]
    s['H_share'] = len(H) / nt
    s['H_len'] = np.mean([len(t['hdr']) for t in H]) if H else 0.0
    hf = Counter(t['hdr'][0] for t in H if t['hdr'][0] != 'x')
    nh = sum(hf.values()) or 1
    s['H_top'] = hf.most_common(1)[0][1] / nh if hf else 0.0
    s['H_div'] = len(hf) / nh
    etok = Counter(x for e in E for x in e[0])
    s['H_excl'] = sum(c * c / (c + etok[x]) for x, c in hf.items()) / nh if hf else 0.0
    pr = []
    for t in H:
        if t['hdr'][0] != 'x' and hf[t['hdr'][0]] >= 5 and t['ent']:
            pr.append((t['hdr'][0], Counter(e[1] for e in t['ent']).most_common(1)[0][0]))
    s['H_MIsys'] = mi_bits(pr)
    strs = defaultdict(set); stok = Counter()
    for i, t in enumerate(C):
        for sg, _, _ in t['ent']:
            if len(sg) >= 2 and 'x' not in sg:
                strs[sg].add(i); stok[sg] += 1
    nst = sum(stok.values()) or 1
    s['D_uniq'] = sum(1 for v in stok.values() if v == 1) / nst
    s['D_multi'] = sum(1 for v in strs.values() if len(v) >= 2) / (len(strs) or 1)
    allt = Counter()
    for t in C:
        for sg in ([t['hdr']] if t['hdr'] else []) + [e[0] for e in t['ent']] + ([t['tot'][0]] if t['tot'] else []):
            for x in sg:
                allt[x] += 1
    xs = allt.pop('x', 0)
    s['D_types'] = math.log10(len(allt) or 1)
    c = np.array(sorted(allt.values(), reverse=True)[:100], float)
    s['D_zipf'] = float(np.polyfit(np.log(np.arange(1, len(c) + 1)), np.log(c), 1)[0]) if len(c) >= 3 else 0.0
    s['D_hapax'] = sum(1 for v in allt.values() if v == 1) / (len(allt) or 1)
    fr = []; ir = []
    for t in C:
        if len(t['ent']) >= 3:
            f = Counter(e[0][-1] for e in t['ent'] if e[0] and e[0][-1] != 'x')
            if f:
                fr.append(f.most_common(1)[0][1] / len(t['ent']))
            m2 = Counter(e[0] for e in t['ent'] if len(e[0]) >= 2 and 'x' not in e[0])
            ir.append(sum(v for v in m2.values() if v >= 2) / len(t['ent']))
    s['D_finrep'] = np.mean(fr) if fr else 0.0
    s['D_inrep'] = np.mean(ir) if ir else 0.0
    s['X_x'] = xs / ((sum(allt.values()) + xs) or 1)
    s['X_frag'] = np.mean([t['frag'] for t in C])
    return np.array([float(s[k]) for k in STATS])


# ---------------------------------------------------------------------------------------------- per-sign features
FEATS = ['logf', 'hdr', 'hdr1', 'ini', 'fin', 'mid', 'single', 'tot', 'sysS', 'sysC', 'sysF', 'sysO',
         'disp', 'elen', 'rnb', 'lnb', 'tabsys_pure', 'nb_multi']


def sign_features(C, min_n=15):
    n = Counter(); f = defaultdict(lambda: np.zeros(len(FEATS)))
    tabs = defaultdict(set); rnb = defaultdict(set); lnb = defaultdict(set)
    total = 0
    I = {k: i for i, k in enumerate(FEATS)}
    for ti, t in enumerate(C):
        tsys = Counter(e[1] for e in t['ent'])
        pure = (tsys.most_common(1)[0][1] / sum(tsys.values())) if tsys else 0.0
        if t['hdr']:
            for j, x in enumerate(t['hdr']):
                if x == 'x': continue
                n[x] += 1; total += 1; v = f[x]; v[I['hdr']] += 1
                if j == 0: v[I['hdr1']] += 1
                tabs[x].add(ti); v[I['tabsys_pure']] += pure
        for sg, sy, _ in t['ent']:
            L = len(sg)
            for j, x in enumerate(sg):
                if x == 'x': continue
                n[x] += 1; total += 1; v = f[x]
                tabs[x].add(ti)
                if L == 1: v[I['single']] += 1
                else:
                    v[I['nb_multi']] += 1
                    if j == 0: v[I['ini']] += 1
                    elif j == L - 1: v[I['fin']] += 1
                    else: v[I['mid']] += 1
                v[I['sys' + sy]] += 1
                v[I['elen']] += L
                v[I['tabsys_pure']] += pure
                if j + 1 < L: rnb[x].add(sg[j + 1])
                if j > 0: lnb[x].add(sg[j - 1])
        if t['tot']:
            for x in t['tot'][0]:
                if x == 'x': continue
                n[x] += 1; total += 1; f[x][I['tot']] += 1; tabs[x].add(ti)
    out = {}
    for x, c in n.items():
        if c < min_n:
            continue
        v = f[x] / c
        v[I['logf']] = math.log(c / total)
        v[I['disp']] = len(tabs[x]) / c
        v[I['rnb']] = len(rnb[x]) / c
        v[I['lnb']] = len(lnb[x]) / c
        out[x] = v
    return out


def rng_for(seed):
    return np.random.default_rng(seed)


# ---------------------------------------------------------------------------------------------- shuffles
def shuffle_global(C, seed):
    """All sign tokens permuted over every slot of the corpus (skeleton, numerals, x kept)."""
    r = random.Random(seed)
    pool = [x for t in C for sg in ([t['hdr']] if t['hdr'] else []) + [e[0] for e in t['ent']] +
            ([t['tot'][0]] if t['tot'] else []) for x in sg if x != 'x']
    r.shuffle(pool); it = iter(pool)
    rep = lambda sg: tuple(x if x == 'x' else next(it) for x in sg)
    out = []
    for t in C:
        h = rep(t['hdr']) if t['hdr'] else None
        e = [(rep(s), sy, v) for s, sy, v in t['ent']]
        to = (rep(t['tot'][0]), t['tot'][1], t['tot'][2]) if t['tot'] else None
        out.append(dict(hdr=h, ent=e, tot=to, frag=t['frag']))
    return out


def shuffle_within_line(C, seed):
    r = random.Random(seed)
    def rep(sg):
        l = list(sg); r.shuffle(l); return tuple(l)
    out = []
    for t in C:
        h = rep(t['hdr']) if t['hdr'] else None
        e = [(rep(s), sy, v) for s, sy, v in t['ent']]
        to = (rep(t['tot'][0]), t['tot'][1], t['tot'][2]) if t['tot'] else None
        out.append(dict(hdr=h, ent=e, tot=to, frag=t['frag']))
    return out


# ---------------------------------------------------------------------------------------------- ABC
def abc_fit(bank_th, bank_s, sobs, frac=0.01, use=None, adjust=True):
    """bank_th: (N,P) unbounded params; bank_s: (N,K) stats.  Returns accepted adjusted params (unbounded),
    accepted indices, distances, scale."""
    K = bank_s.shape[1]
    use = np.arange(K) if use is None else np.asarray(use)
    S = bank_s[:, use]; so = sobs[use]
    med = np.median(S, 0)
    mad = np.median(np.abs(S - med), 0) * 1.4826
    mad[mad < 1e-9] = S.std(0)[mad < 1e-9] + 1e-9
    d = np.sqrt((((S - so) / mad) ** 2).sum(1))
    nacc = max(30, int(frac * len(d)))
    idx = np.argsort(d)[:nacc]
    th = bank_th[idx]
    if not adjust:
        return th, idx, d[idx], mad
    h = d[idx].max()
    w = 1 - (d[idx] / h) ** 2
    X = (S[idx] - so) / mad
    X1 = np.hstack([np.ones((nacc, 1)), X])
    W = np.diag(w)
    lam = 1e-3 * np.eye(X1.shape[1]); lam[0, 0] = 0
    beta = np.linalg.solve(X1.T @ W @ X1 + lam * nacc, X1.T @ W @ th)
    adj = th - X @ beta[1:]
    return adj, idx, d[idx], mad
