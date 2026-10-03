"""Loop 17 (arrow-in-the-dark): decode the INSTITUTION, not the words.

Agent-based generator of a credential corpus (authorities, offices, commodities, holders, re-issue as tablets,
travelling seals) laid over the real excavation skeleton (site, object class, find area, completeness of every
object), fitted to the real corpus by approximate Bayesian computation on a vector of summary statistics
(nesting, cross-site run sharing, middle uniqueness vs bigram, stock-text copy distribution and area clustering,
seal vs tablet locality of stock texts, closer inventory and Zipf slope, quantity-seal share, sealing->seal
cross-site matches, text-length distribution by object class, sign richness).

Usage:
  python3 tools/dark_loop17.py real              # observed statistic vectors for seq_raw / seq_strong / seq_all
  python3 tools/dark_loop17.py pilot N           # N prior simulations, prior-predictive coverage of the observed stats
  python3 tools/dark_loop17.py bank N OUT.npz    # N prior simulations -> reference table (multiprocessing)
  python3 tools/dark_loop17.py fit BANK.npz      # rejection + local-linear regression ABC on all three seq levels
  python3 tools/dark_loop17.py control BANK.npz  # recovery of known parameters from synthetic corpora (same bank)
No model identifiers, no interpretation of signs: everything is counted.
"""
import json, sys, os, math, collections, random, time
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORPUS = os.path.join(HERE, 'data/derived/merged-corpus-canonical.json')
SHORT = (set(range(1, 8)) | set(range(12, 21)) | set(range(25, 30))) - {2}
TALL = set(range(32, 38)) | {39}
REAL_NUM = SHORT | TALL

def cls_of(ty):
    if ty.startswith('SEAL'): return 'SEAL'
    if ty.startswith('TAB'): return 'TAB'
    if ty.startswith('TAG'): return 'TAG'
    if ty.startswith('POT'): return 'POT'
    return 'TAB'          # MISC, BNGL, ROD, IMPL ... treated as tablet-like tokens

def load_real(seqkey='seq'):
    C = json.load(open(CORPUS))
    objs = []
    for t in C:
        if t['site'] == 'Unknown' or t['complete'] != 'Y' or not t[seqkey]: continue
        area = t['area-section'] if t['area-section'] not in ('--', '-', '') else None
        objs.append(dict(site=t['site'], cls=cls_of(t['type']), area=area, seq=tuple(t[seqkey])))
    return objs

# ---------------------------------------------------------------------------------------------- statistics
def zipf_slope(counts, top):
    c = sorted(counts, reverse=True)[:top]
    if len(c) < 3: return 0.0
    x = np.log(np.arange(1, len(c) + 1)); y = np.log(np.array(c, float))
    return float(np.polyfit(x, y, 1)[0])

def frame_classes(texts):
    """data-driven frame: opener class (initial-prone), marker class (2nd after opener), final class (final-prone)."""
    n = collections.Counter(); ini = collections.Counter(); fin = collections.Counter()
    for s in texts:
        if len(s) < 2: continue
        for x in s: n[x] += 1
        ini[s[0]] += 1; fin[s[-1]] += 1
    OPEN = {x for x in n if n[x] >= 10 and ini[x] / n[x] >= 0.6}
    FINAL = {x for x in n if n[x] >= 10 and fin[x] / n[x] >= 0.5}
    sec = collections.Counter()
    for s in texts:
        if len(s) >= 3 and s[0] in OPEN: sec[s[1]] += 1
    MARK = {x for x in sec if n[x] >= 10 and sec[x] / n[x] >= 0.5}
    return OPEN, MARK, FINAL

def middle_of(s, OPEN, MARK, FINAL):
    i, j = 0, len(s)
    if s and s[0] in OPEN:
        i = 1
        if len(s) > 1 and s[1] in MARK: i = 2
    k = 0
    while j - 1 > i and s[j - 1] in FINAL and k < 2: j -= 1; k += 1
    return s[i:j]

def bigram_uniqueness_ratio(mids, rng):
    mids = [m for m in mids if len(m) >= 1]
    if len(mids) < 20: return 1.0
    uniq = len([m for m, c in collections.Counter(mids).items() if c == 1]) / len(mids)
    start = collections.Counter(m[0] for m in mids)
    trans = collections.defaultdict(collections.Counter)
    for m in mids:
        for a, b in zip(m, m[1:]): trans[a][b] += 1
    lens = [len(m) for m in mids]
    sk, sw = zip(*start.items()); sw = np.array(sw, float); sw /= sw.sum()
    tk = {a: (list(c.keys()), np.cumsum(np.array(list(c.values()), float) / sum(c.values()))) for a, c in trans.items()}
    allk = list(set(x for m in mids for x in m))
    out = []
    firsts = rng.choice(len(sk), size=len(mids), p=sw)
    us = rng.random(sum(lens)); ui = 0
    for L, f in zip(lens, firsts):
        cur = sk[f]; m = [cur]
        for _ in range(L - 1):
            if cur in tk:
                ks, cp = tk[cur]; cur = ks[min(len(ks) - 1, int(np.searchsorted(cp, us[ui])))]
            else:
                cur = allk[int(us[ui] * len(allk))]
            ui += 1; m.append(cur)
        out.append(tuple(m))
    u2 = len([m for m, c in collections.Counter(out).items() if c == 1]) / len(out)
    return uniq / max(u2, 1e-6)

STAT_NAMES = ['nest_all', 'nest_seal', 'xsite3', 'miduniq_ratio', 'mid_nonempty', 'stock_tab_per1000', 'stock_seal_per1000',
              'zipf_copy', 'topcopy_share', 'area_ratio', 'loc_tab', 'loc_seal', 'closer_units_per1000', 'n_final_signs',
              'zipf_final', 'top_final_share', 'qty_seal', 'open_seal', 'tag_match', 'tag_cross', 'len_seal_mean', 'len_seal_sd',
              'len_tab_mean', 'len_tab_sd', 'len_tag_mean', 'len_pot_mean', 'pot_len1', 'tab_in_seal', 'tab_in_seal_same',
              'seal_dup_same', 'signs_per1000', 'top10_share', 'rep_within', 'tab_suffix_share']

def summary(objs, is_num, rng):
    """objs: list of dict(site, cls, area, seq). Returns np.array of STAT_NAMES."""
    texts = [o['seq'] for o in objs]
    by = collections.defaultdict(list)
    for o in objs: by[o['cls']].append(o)
    # 1-2 nesting
    distinct = set(texts)
    subs = set()
    for t in distinct:
        for n in (3, 4, 5):
            if len(t) > n:
                for i in range(len(t) - n + 1): subs.add(t[i:i + n])
    short = [t for t in distinct if 3 <= len(t) <= 5]
    nest_all = sum(t in subs for t in short) / max(1, len(short))
    sshort = {o['seq'] for o in by['SEAL'] if 3 <= len(o['seq']) <= 5}
    nest_seal = sum(t in subs for t in sshort) / max(1, len(sshort))
    # 3 cross-site 3-run sharing
    run_sites = collections.defaultdict(set)
    st = {(o['site'], o['seq']) for o in objs if len(o['seq']) >= 3}
    for site, t in st:
        for i in range(len(t) - 2): run_sites[t[i:i + 3]].add(site)
    xs = 0
    for site, t in st:
        if any(len(run_sites[t[i:i + 3]] - {site}) > 0 for i in range(len(t) - 2)): xs += 1
    xsite3 = xs / max(1, len(st))
    # 4-5 middles
    OPEN, MARK, FINAL = frame_classes(texts)
    mids = [middle_of(t, OPEN, MARK, FINAL) for t in texts if len(t) >= 3]
    mid_nonempty = sum(len(m) >= 1 for m in mids) / max(1, len(mids))
    miduniq = bigram_uniqueness_ratio(mids, rng)
    # 6-9 stock texts
    cnt_all = collections.Counter(texts)
    cnt_tab = collections.Counter(o['seq'] for o in by['TAB']); cnt_seal = collections.Counter(o['seq'] for o in by['SEAL'])
    stock_tab = [t for t, c in cnt_tab.items() if c >= 3]; stock_seal = [t for t, c in cnt_seal.items() if c >= 3]
    stock_all = [c for t, c in cnt_all.items() if c >= 3]
    zc = zipf_slope(stock_all, 30)
    topcopy = max(cnt_all.values()) / len(texts)
    # 10 area clustering of same-text pairs within site (known areas), vs area labels permuted within site x cls
    def same_area_rate(ob):
        g = collections.defaultdict(list)
        for o in ob:
            if o['area'] is not None and len(o['seq']) >= 2: g[(o['site'], o['seq'])].append(o['area'])
        same = tot = 0
        for k, ar in g.items():
            if len(ar) < 2: continue
            c = collections.Counter(ar); n = len(ar)
            tot += n * (n - 1) / 2; same += sum(v * (v - 1) / 2 for v in c.values())
        return same, tot
    same, tot = same_area_rate(objs)
    if tot >= 5:
        nulls = []
        grp = collections.defaultdict(list)
        for i, o in enumerate(objs):
            if o['area'] is not None and len(o['seq']) >= 2: grp[(o['site'], o['cls'])].append(i)
        light = [dict(site=o['site'], seq=o['seq'], area=o['area']) for o in objs]
        for _ in range(2):
            for idx in grp.values():
                ar = [light[i]['area'] for i in idx]; rng.shuffle(ar)
                for i, a in zip(idx, ar): light[i]['area'] = a
            s2, t2 = same_area_rate(light); nulls.append(s2 / max(1, t2))
        area_ratio = (same / tot) / max(1e-3, np.mean(nulls))
    else:
        area_ratio = 1.0
    # 11-12 locality
    def single_site(stock, ob):
        if not stock: return 0.5
        sites = collections.defaultdict(set)
        for o in ob: sites[o['seq']].add(o['site'])
        return sum(len(sites[t]) == 1 for t in stock) / len(stock)
    loc_tab = single_site(stock_tab, by['TAB']); loc_seal = single_site(stock_seal, by['SEAL'])
    # 13-16 closers on seals
    seals2 = [o['seq'] for o in by['SEAL'] if len(o['seq']) >= 2]
    units = collections.Counter(t[-2:] for t in seals2); fin = collections.Counter(t[-1] for t in seals2)
    closer_units = 1000 * sum(c >= 3 for c in units.values()) / max(1, len(by['SEAL']))
    n_final = sum(c >= 3 for c in fin.values())
    zipf_final = zipf_slope(fin.values(), 10)
    top_final = fin.most_common(1)[0][1] / max(1, len(seals2)) if seals2 else 0
    qty_seal = sum(any(x in is_num for x in t) for t in seals2) / max(1, len(seals2))
    open_seal = sum(t[0] in OPEN for t in seals2) / max(1, len(seals2))
    # 19-20 tags vs seals
    seal_sites = collections.defaultdict(set)
    for o in by['SEAL']: seal_sites[o['seq']].add(o['site'])
    tags3 = [o for o in by['TAG'] if len(o['seq']) >= 3]
    m = [o for o in tags3 if o['seq'] in seal_sites]
    tag_match = len(m) / max(1, len(tags3))
    cross = sum(o['site'] not in seal_sites[o['seq']] for o in m)
    tag_cross = (cross + 0.5) / (len(m) + 1)
    # 21-27 lengths
    def ms(cls):
        L = [len(o['seq']) for o in by[cls]]
        return (float(np.mean(L)), float(np.std(L))) if L else (0.0, 0.0)
    ls = ms('SEAL'); lt = ms('TAB'); lg = ms('TAG'); lp = ms('POT')
    pot1 = sum(len(o['seq']) == 1 for o in by['POT']) / max(1, len(by['POT']))
    # 28-30 tablet re-issue and seal duplicates
    tabd = {o['seq'] for o in by['TAB'] if len(o['seq']) >= 2}
    tab_in_seal = sum(t in seal_sites for t in tabd) / max(1, len(tabd))
    tabsite = collections.defaultdict(set)
    for o in by['TAB']: tabsite[o['seq']].add(o['site'])
    tis = [t for t in tabd if t in seal_sites]
    tab_in_seal_same = (sum(len(tabsite[t] & seal_sites[t]) > 0 for t in tis) + 0.5) / (len(tis) + 1)
    sc = collections.Counter((o['site'], o['seq']) for o in by['SEAL'] if len(o['seq']) >= 3)
    seal_dup_same = sum(c for c in sc.values() if c >= 2) / max(1, sum(sc.values()))
    # 31-33 richness, repetition
    tok = collections.Counter(x for t in texts for x in t); ntok = sum(tok.values())
    signs_per1000 = 1000 * len(tok) / max(1, ntok)
    top10 = sum(c for _, c in tok.most_common(10)) / max(1, ntok)
    rep = sum(len(set(t)) < len(t) for t in texts if len(t) >= 2) / max(1, sum(len(t) >= 2 for t in texts))
    # 34 tablet suffix: share of tablet texts (len>=2) whose last sign is tablet-enriched (final-class sign >=3x commoner final on tablets than on seals)
    tabs2 = [o['seq'] for o in by['TAB'] if len(o['seq']) >= 2]
    ft = collections.Counter(t[-1] for t in tabs2)
    enr = {x for x in ft if ft[x] / max(1, len(tabs2)) >= 3 * (fin[x] / max(1, len(seals2))) and ft[x] >= 5}
    tab_suffix = sum(t[-1] in enr for t in tabs2) / max(1, len(tabs2))
    v = [nest_all, nest_seal, xsite3, miduniq, mid_nonempty, 1000 * len(stock_tab) / max(1, len(by['TAB'])),
         1000 * len(stock_seal) / max(1, len(by['SEAL'])), zc, topcopy, area_ratio, loc_tab, loc_seal, closer_units, n_final,
         zipf_final, top_final, qty_seal, open_seal, tag_match, tag_cross, ls[0], ls[1], lt[0], lt[1], lg[0], lp[0], pot1,
         tab_in_seal, tab_in_seal_same, seal_dup_same, signs_per1000, top10, rep, tab_suffix]
    return np.array(v, float)

# ---------------------------------------------------------------------------------------------- generator
PARAMS = ['A', 'O', 'zO', 'C', 'H', 'k', 'p_mid', 'lam_mid', 'V_mid', 'z_mid', 'p_share', 'p_open', 'p_comm', 'p_off', 'p_ext',
          'p_reissue', 'z_copy', 'p_batch', 'p_travel', 'p_suf', 'p_potcred']
# prior: (type, lo, hi); 'int' uniform integer, 'log' log-uniform, 'u' uniform
PRIOR = {'A': ('int', 1, 8), 'O': ('log', 3, 400), 'zO': ('u', 0.3, 2.0), 'C': ('log', 1, 40), 'H': ('log', 50, 60000),
         'k': ('u', 1.0, 6.0), 'p_mid': ('u', 0, 1), 'lam_mid': ('u', 0, 2.5), 'V_mid': ('log', 30, 3000), 'z_mid': ('u', 0.3, 1.3), 'p_share': ('u', 0, 1),
         'p_open': ('u', 0, 1), 'p_comm': ('u', 0, 1), 'p_off': ('u', 0, 1), 'p_ext': ('u', 0, 1), 'p_reissue': ('u', 0, 1),
         'z_copy': ('u', 0.3, 2.5), 'p_batch': ('u', 0, 1), 'p_travel': ('u', 0, 1), 'p_suf': ('u', 0, 1), 'p_potcred': ('u', 0, 1)}

def draw_prior(rng):
    th = {}
    for p in PARAMS:
        kind, lo, hi = PRIOR[p]
        if kind == 'int': th[p] = int(rng.integers(lo, hi + 1))
        elif kind == 'log': th[p] = float(math.exp(rng.uniform(math.log(lo), math.log(hi))))
        else: th[p] = float(rng.uniform(lo, hi))
    return th

def to_z(th):
    """parameters -> unbounded working scale (log / logit / linear) for regression"""
    z = []
    for p in PARAMS:
        kind, lo, hi = PRIOR[p]; v = th[p]
        if kind == 'log': z.append(math.log(v))
        elif kind == 'u' and lo == 0 and hi == 1: v = min(max(v, 1e-3), 1 - 1e-3); z.append(math.log(v / (1 - v)))
        else: z.append(float(v))
    return np.array(z)

def from_z(z):
    th = {}
    for p, v in zip(PARAMS, z):
        kind, lo, hi = PRIOR[p]
        if kind == 'log': th[p] = float(math.exp(v))
        elif kind == 'u' and lo == 0 and hi == 1: th[p] = float(1 / (1 + math.exp(-v)))
        else: th[p] = float(v)
    return th

def zipf_p(n, z):
    w = np.arange(1, n + 1, dtype=float) ** (-z); return w / w.sum()

class Skeleton:
    """the real excavation design: per site, counts of complete objects by class and their find-area labels"""
    def __init__(self, objs):
        self.objs = objs
        self.sites = sorted({o['site'] for o in objs})
        self.site_w = {}; self.areas = {}
        for s in self.sites:
            so = [o for o in objs if o['site'] == s]
            self.site_w[s] = len([o for o in so if o['cls'] == 'SEAL']) + 0.5
            ac = collections.Counter(o['area'] for o in so if o['area'] is not None)
            if ac:
                ks = list(ac.keys()); w = np.array(list(ac.values()), float); self.areas[s] = (ks, w / w.sum())
            else:
                self.areas[s] = (['a0'], np.array([1.0]))
        wmax = max(self.site_w.values()); self.site_rel = {s: self.site_w[s] / wmax for s in self.sites}
        self.cum_areas = {s: np.cumsum(self.areas[s][1]) for s in self.sites}
        self.idx = collections.defaultdict(list)
        for i, o in enumerate(objs): self.idx[(o['site'], o['cls'])].append(i)

# sign id layout (abstract signs; identity only matters)
def sim_is_num(): return set(range(20, 33))

def simulate(th, skel, seed):
    rng = np.random.default_rng(seed); U = rng.random
    A = int(round(th['A'])); O = max(1, int(round(th['O']))); C = max(1, int(round(th['C'])))
    V = max(5, int(round(th['V_mid']))); k = max(1.0, th['k'])
    cA = np.cumsum(zipf_p(A, 1.0)); cO = np.cumsum(zipf_p(O, th['zO'])); cC = np.cumsum(zipf_p(C, 1.0)); cV = np.cumsum(zipf_p(V, th['z_mid']))
    pick = lambda cum: int(min(len(cum) - 1, np.searchsorted(cum, U())))
    OPEN_ID = [1 + a for a in range(A)]; CONN_ID = [10 + (a % 2) for a in range(A)]
    NUM_ID = {v: 19 + v for v in range(1, 9)}; TALL_ID = {2: 30, 3: 31, 4: 32}
    VOUCH = 40; SUF = 41
    FIN = [50 + i for i in range(12)]; cFIN = np.cumsum(zipf_p(12, 1.0))
    off_final = [FIN[pick(cFIN)] for _ in range(O)]
    off_title = [100 + o if U() < 0.6 else None for o in range(O)]
    mode = [int(np.searchsorted([0.25, 0.70, 1.0], U())) + 2 for _ in range(C)]
    cQ = np.cumsum(zipf_p(30, 1.0))
    DEV = [-1, 0, 1, 2]; cDEV = np.cumsum([0.2, 0.4, 0.25, 0.15]); cTALL = np.cumsum([0.3, 0.4, 0.3])

    def fresh_mid():
        L = 1 + rng.poisson(th['lam_mid'])
        return tuple(1000 + pick(cV) for _ in range(L))

    def office():
        o = pick(cO); return tuple(x for x in (off_title[o], off_final[o]) if x is not None)

    def credential(hmid):
        a = pick(cA); parts = []
        if U() < th['p_open']: parts += [OPEN_ID[a], CONN_ID[a]]
        if U() < th['p_mid']: parts += list(hmid if U() < th['p_share'] else fresh_mid())
        if U() < th['p_comm']:
            c = pick(cC); n = min(8, max(1, mode[c] + DEV[int(np.searchsorted(cDEV, U()))])); parts += [NUM_ID[n], 500 + c]
        if U() < th['p_ext']: parts.append(600 + pick(cQ))
        if U() < th['p_off'] or not parts: parts += list(office())
        return tuple(parts)

    # lazily generated credentials: site s has N_s = round(H * rel_s * k) credentials, credential i belongs to holder i // k
    N = {s: max(1, int(round(th['H'] * skel.site_rel[s] * k))) for s in skel.sites}
    cum_pop = {s: np.cumsum(zipf_p(N[s], th['z_copy'])) for s in skel.sites}
    cred = {}; holder = {}
    def get_cred(s, i):
        key = (s, i)
        if key not in cred:
            h = (s, int(i // k))
            if h not in holder:
                holder[h] = (fresh_mid(), rand_area(s))
            cred[key] = (credential(holder[h][0]), holder[h][1])
        return cred[key]
    rand_area = lambda s: skel.areas[s][0][int(min(len(skel.areas[s][0]) - 1, np.searchsorted(skel.cum_areas[s], U())))]
    site_list = skel.sites; sw = np.array([skel.site_w[s] for s in site_list]); csw = np.cumsum(sw / sw.sum())
    tall = lambda: TALL_ID[2 + int(np.searchsorted(cTALL, U()))]

    out = []
    for o in skel.objs:
        s = o['site']; known = o['area'] is not None; area = None
        if o['cls'] == 'SEAL':
            seq, area = get_cred(s, int(rng.integers(N[s])))
        elif o['cls'] == 'TAB':
            if U() < th['p_reissue']:
                seq, ha = get_cred(s, pick(cum_pop[s]))
                if U() < th['p_suf']: seq = seq + (SUF,)
                area = ha if U() < th['p_batch'] else rand_area(s)
            else:
                if U() < 0.5: seq = (tall(), VOUCH)
                else:
                    seq = office()
                    if U() < th['p_suf']: seq = seq + (SUF,)
                area = rand_area(s)
        elif o['cls'] == 'TAG':
            if U() < th['p_travel'] and len(site_list) > 1:
                while True:
                    s2 = site_list[int(min(len(site_list) - 1, np.searchsorted(csw, U())))]
                    if s2 != s: break
                seq, _ = get_cred(s2, pick(cum_pop[s2]))
            else:
                seq, _ = get_cred(s, pick(cum_pop[s]))
            area = rand_area(s)
        else:  # POT
            u = U()
            if u < th['p_potcred']: seq, _ = get_cred(s, pick(cum_pop[s]))
            elif u < th['p_potcred'] + (1 - th['p_potcred']) / 2: seq = (tall(),) if U() < 0.8 else (NUM_ID[1],)
            else:
                oo = office(); seq = oo[-1:] if U() < 0.6 else oo
            area = rand_area(s)
        out.append(dict(site=s, cls=o['cls'], area=(area if known else None), seq=seq))
    return out

def sim_stats(th, skel, seed):
    objs = simulate(th, skel, seed)
    return summary(objs, sim_is_num(), np.random.default_rng(seed + 1))

# ---------------------------------------------------------------------------------------------- ABC
def abc_fit(bank_theta, bank_stats, obs, n_acc=400, use=None):
    """rejection + local-linear regression adjustment (Beaumont et al. 2002) on the working scale."""
    S = bank_stats.copy(); ok = np.all(np.isfinite(S), axis=1)
    S = S[ok]; T = bank_theta[ok]
    if use is not None: S = S[:, use]; o = obs[use]
    else: o = obs
    scale = np.median(np.abs(S - np.median(S, axis=0)), axis=0) * 1.4826 + 1e-9
    d = np.sqrt((((S - o) / scale) ** 2).sum(axis=1))
    order = np.argsort(d); acc = order[:n_acc]; h = d[acc[-1]]
    w = 1 - (d[acc] / h) ** 2
    X = np.c_[np.ones(n_acc), (S[acc] - o) / scale]
    W = np.diag(w)
    try:
        beta = np.linalg.solve(X.T @ W @ X + 1e-6 * np.eye(X.shape[1]), X.T @ W @ T[acc])
        adj = T[acc] - X[:, 1:] @ beta[1:]
    except np.linalg.LinAlgError:
        adj = T[acc]
    return dict(acc=acc, w=w, rej=T[acc], adj=adj, dist=d[acc], h=h)

def wquant(x, w, q):
    i = np.argsort(x); cw = np.cumsum(w[i]) / w.sum(); return float(np.interp(q, cw, x[i]))

def describe(post, w):
    lines = []
    for j, p in enumerate(PARAMS):
        kind, lo, hi = PRIOR[p]
        z = post[:, j]
        med = wquant(z, w, 0.5); l5 = wquant(z, w, 0.05); u95 = wquant(z, w, 0.95)
        def back(v):
            if kind == 'log': return math.exp(v)
            if kind == 'u' and lo == 0 and hi == 1: return 1 / (1 + math.exp(-v))
            return v
        # prior 90% width on the working scale for shrinkage
        if kind == 'log': pw = 0.9 * (math.log(hi) - math.log(lo))
        elif kind == 'u' and lo == 0 and hi == 1: pw = math.log(0.95 / 0.05) - math.log(0.05 / 0.95)
        else: pw = 0.9 * (hi - lo)
        shrink = (u95 - l5) / pw
        lines.append((p, back(med), back(l5), back(u95), shrink))
    return lines

def fmt_post(lines):
    s = ['%-10s %10s %10s %10s %8s' % ('param', 'median', '5%', '95%', 'CI/prior')]
    for p, m, l, u, sh in lines:
        s.append('%-10s %10.3g %10.3g %10.3g %8.2f' % (p, m, l, u, sh))
    return '\n'.join(s)

def main():
    cmd = sys.argv[1]
    if cmd == 'real':
        for key in ('seq_raw', 'seq_strong', 'seq_all'):
            objs = load_real(key); v = summary(objs, REAL_NUM, np.random.default_rng(1))
            print(key, len(objs))
            for n, x in zip(STAT_NAMES, v): print('  %-22s %.4f' % (n, x))
        return
    objs = load_real('seq'); skel = Skeleton(objs)
    if cmd == 'pilot':
        N = int(sys.argv[2]); rng = np.random.default_rng(17)
        obs = summary(objs, REAL_NUM, np.random.default_rng(1))
        t0 = time.time(); ths = []; S = []
        for i in range(N):
            th = draw_prior(rng); ths.append(th); S.append(sim_stats(th, skel, 1000 + i))
        S = np.array(S); print('time per sim %.3f s' % ((time.time() - t0) / N))
        print('%-22s %9s %9s %9s %9s %6s' % ('stat', 'obs', 'prior5%', 'prior50%', 'prior95%', 'cover'))
        for j, n in enumerate(STAT_NAMES):
            q = np.nanpercentile(S[:, j], [5, 50, 95]); cov = np.nanmin(S[:, j]) <= obs[j] <= np.nanmax(S[:, j])
            print('%-22s %9.3f %9.3f %9.3f %9.3f %6s' % (n, obs[j], q[0], q[1], q[2], 'yes' if cov else 'NO'))
        return
    if cmd == 'bank':
        N = int(sys.argv[2]); out = sys.argv[3]; seed0 = int(sys.argv[4]) if len(sys.argv) > 4 else 0
        import multiprocessing as mp
        rng = np.random.default_rng(170 + seed0); ths = [draw_prior(rng) for _ in range(N)]
        with mp.Pool(mp.cpu_count()) as pool:
            res = pool.starmap(sim_stats, [(th, skel, 10 ** 6 * (seed0 + 1) + i) for i, th in enumerate(ths)], chunksize=20)
        T = np.array([to_z(th) for th in ths]); S = np.array(res)
        np.savez(out, theta=T, stats=S, params=np.array(PARAMS), stat_names=np.array(STAT_NAMES))
        print('saved', out, T.shape, S.shape)
        return
    if cmd in ('fit', 'control'):
        b = np.load(sys.argv[2]); T = b['theta']; S = b['stats']
        if cmd == 'fit':
            for key in ('seq_raw', 'seq_strong', 'seq_all'):
                ob = load_real(key); obs = summary(ob, REAL_NUM, np.random.default_rng(1))
                r = abc_fit(T, S, obs)
                print('=== observed', key, 'n_acc', len(r['acc']), 'h', round(float(r['h']), 2))
                print('-- rejection posterior'); print(fmt_post(describe(r['rej'], r['w'])))
                print('-- regression-adjusted posterior'); print(fmt_post(describe(r['adj'], r['w'])))
        else:
            rng = np.random.default_rng(99); n_test = int(sys.argv[3]) if len(sys.argv) > 3 else 50
            hits = np.zeros(len(PARAMS)); hits_r = np.zeros(len(PARAMS)); shr = []; shr_r = []
            ids = rng.choice(len(T), size=n_test, replace=False)
            for i in ids:
                mask = np.ones(len(T), bool); mask[i] = False
                r = abc_fit(T[mask], S[mask], S[i])
                for mode, key in ((r['adj'], hits), (r['rej'], hits_r)):
                    for j in range(len(PARAMS)):
                        l5 = wquant(mode[:, j], r['w'], 0.05); u95 = wquant(mode[:, j], r['w'], 0.95)
                        key[j] += l5 <= T[i, j] <= u95
                shr.append([x[4] for x in describe(r['adj'], r['w'])]); shr_r.append([x[4] for x in describe(r['rej'], r['w'])])
            print('coverage of 90%% CI over %d pseudo-observed corpora (leave-one-out from the bank):' % n_test)
            print('%-10s %8s %8s %10s %10s' % ('param', 'cov_adj', 'cov_rej', 'CIw/prior_adj', 'CIw/prior_rej'))
            for j, p in enumerate(PARAMS):
                print('%-10s %8.2f %8.2f %10.2f %10.2f' % (p, hits[j] / n_test, hits_r[j] / n_test, np.mean([s[j] for s in shr]), np.mean([s[j] for s in shr_r])))

if __name__ == '__main__':
    main()
