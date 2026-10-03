"""S-DARK-48: A CENTRALIZATION INDEX. How uniform is the Indus system across cities, measured against writing
systems of known administration (Ur III seal legends by city, Linear B by palace, proto-cuneiform by site,
Latin EDH by region, Proto-Elamite Susa vs others) and two synthetic anchors (one shared generator = perfectly
central; independent generators = fully local)?

Design (identical code for every corpus):
  texts collapsed to one per distinct text per site x type (reuse component uses the uncollapsed objects);
  for a pair of groups (A, B) draw n texts from each (SUB draws) and compute the divergence components;
  null = WITHIN-SITE: two disjoint random halves of n texts from the same site (NPERM draws), for A and for B;
  report between as (a) excess over within (bits or units, bias-corrected) and (b) multiple of within.
Components:
  uni    Jensen-Shannon divergence (bits) of the sign unigram distributions
  bi     JSD of the bigram distributions
  rho    1 - Spearman rank correlation of sign frequencies (signs with >= 5 pooled tokens)
  init   JSD of the text-initial sign distribution (generic 'opener menu')
  fin    JSD of the text-final sign distribution (generic 'closer menu')
  midinv JSD of the non-edge sign distribution (generic 'middle inventory')
  len    JSD of the length distribution
  top30  JSD of the distribution over the 30 pooled-commonest signs ('frame vocabulary')
  rest   JSD of the distribution over all other signs ('content vocabulary')
  frozen 1 - Jaccard of the frozen-pair sets (adjacent >= 3 times, adjacent in >= 90% of co-occurrences)
  reuse  share of distinct texts (>= 2 signs) of one group attested in the other (mean of both directions),
         on uncollapsed objects; reported as within / between (> 1 = more local)
Indus / IM77 only (frame parser of S310/S331, tools/dark_loop37.py):
  opener JSD of the opener menu (OPEN signs or none), closer JSD of the closer menu (CL signs or none),
  rates  mean |difference| of opener / marker / closer / suffix / title rates,
  name   JSD of NAME-labelled tokens, num JSD of COUNT-labelled numeral signs
Ur III: title = JSD over title words, name = JSD over non-title words. Linear B: ideo = JSD over ideograms/NUM, syl = JSD over syllabic signs.
Usage: python3 tools/dark_loop48.py <cycle 1|2|3> <seq_raw|seq_strong|seq_all> [nperm] [nsub]
"""
import sys, json, csv, collections, random, math, os, re
import numpy as np
from scipy.stats import spearmanr
sys.path.insert(0, '/home/user/Indus-/tools')
from dark_loop37 import OPEN, MARK, MJAR, SUF, CL, FISH, NUM, learn_qual, make_parser

ROOT = '/home/user/Indus-/'
DARK = ROOT + 'data/derived/dark/'
C48 = DARK + 'loop48_corpora/'
CY = int(sys.argv[1]) if len(sys.argv) > 1 else 1
LV = sys.argv[2] if len(sys.argv) > 2 else 'seq_raw'
NPERM = int(sys.argv[3]) if len(sys.argv) > 3 else 1000
NSUB = int(sys.argv[4]) if len(sys.argv) > 4 else 200
rng = np.random.default_rng(48 + CY)
GENERIC = ['uni', 'bi', 'rho', 'init', 'fin', 'midinv', 'len', 'top30', 'rest', 'frozen']
UR3_TITLES = {'dub-sar', 'arad2', 'arad2-zu', 'arad', 'ensi2', 'ugula', 'nu-banda3', 'gudu4', 'dam-gar3', 'lunga', 'sipa', 'szabra',
              'sagi', 'kuruszda', 'aga3-us2', 'muhaldim', 'nu-banda3-gu4', 'sanga', 'szagina', 'sukkal', 'ra2-gaba', 'nar', 'simug',
              'aszgab', 'nagar', 'ad-kup4', 'szu-i', 'ma2-lah5', 'engar', 'ma2-gal', 'gal5-la2-gal', 'kas4', 'sza13-dub-ba', 'lu2-{d}inanna',
              'dumu', 'lugal', 'kal-ga', 'ki-ag2', 'dingir', 'kur-kur-ra', 'ki-en-gi', 'ki-uri', 'lugal-bi', 'uri5{ki}-ma', 'nita', 'an-ub-da', 'limmu2-ba'}

# ------------------------------------------------------------------ loaders
def otype(t):
    return {'SEAL': 'seal', 'TAB': 'tablet', 'POT': 'pot', 'TAG': 'sealing'}.get(t.split(':')[0], 'other')

def load_wells(level):
    C = json.load(open(ROOT + 'data/derived/merged-corpus-canonical.json'))
    T = []
    for r in C:
        s = r[level]
        if not s or len(s) < 2 or r['complete'] != 'Y' or r['dir.'].strip() == '-': continue
        T.append(dict(site=r['site'], type=otype(r['type']), seq=tuple(s), area=r['area-section'].strip()))
    return T

def load_im77():
    rows = list(csv.DictReader(open(ROOT + 'data/im77/im77_corpus_lines.csv')))
    objs = collections.OrderedDict()
    for r in rows:
        if r['line'] == '9' or not r['signs_clean'].strip(): continue
        o = objs.setdefault((r['text_no'], r['side']), dict(site=r['site'], ot=r['object_type'], seq=[]))
        o['seq'].extend(int(x) for x in r['signs_clean'].split() if x != '0')
    smap = {'Mohenjodaro': 'Mohenjo-daro', 'Harappa': 'Harappa', 'Lothal': 'Lothal', 'Kalibangan': 'Kalibangan', 'Chanhudaro': 'Chanhu-daro'}
    tmap = {'seal': 'seal', 'sealing': 'sealing', 'miniature tablet': 'tablet', 'copper tablet': 'tablet', 'pottery graffito': 'pot'}
    return [dict(site=smap.get(o['site'], 'other'), type=tmap.get(o['ot'], 'other'), seq=tuple(o['seq']), area='')
            for o in objs.values() if len(o['seq']) >= 2]

def load_ref(name):
    return [dict(site=d['site'], type=d['type'], seq=tuple(d['seq']), area='') for d in map(json.loads, open(C48 + name + '.jsonl'))]

def collapse(T):
    seen = set(); out = []
    for t in T:
        k = (t['site'], t['type'], t['seq'])
        if k in seen: continue
        seen.add(k); out.append(t)
    return out

# ------------------------------------------------------------------ Indus frame labels
MCL = {342: 'C740', 211: 'C520', 12: 'C151', 15: 'C156', 254: 'C527', 60: 'C226', 245: 'C617', 328: 'C700', 66: 'C236'}
MOPEN = {267, 391, 293, 150}; MMARK = {99, 100, 123}; MMJAR = {343, 344, 345, 346}; MSUF = {176, 1}
MNUM = {97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130}

def im77_parser():
    def parse(s):
        n = len(s); i = 0; j = n; lab = ['NAME'] * n
        if s[0] in MOPEN:
            lab[0] = 'OPENER'; i = 1
            if n > 1 and s[1] in MMARK:
                lab[1] = 'MARKER'; i = 2
                if s[0] == 293 and n > 2 and s[2] in MMJAR: lab[2] = 'MARKER'; i = 3
        while j - 1 > i and s[j - 1] in MSUF and (s[j - 2] in MCL or s[j - 2] in MSUF): lab[j - 1] = 'SUFFIX'; j -= 1
        if j - 1 >= i and s[j - 1] in MCL: lab[j - 1] = 'CLOSER'; j -= 1
        for k in range(i, j):
            if s[k] in MNUM: lab[k] = 'COUNT'
        return lab
    return parse

def add_labels(T, parse):
    for t in T: t['lab'] = parse(list(t['seq']))

# ------------------------------------------------------------------ encoding
class Corpus:
    """collapsed texts encoded as int arrays; kind = 'indus' | 'im77' | 'ur3' | 'linb' | None"""
    def __init__(self, T, kind=None, raw=None):
        self.T = T; self.kind = kind; self.raw = raw if raw is not None else T
        vocab = {}
        for t in T:
            for s in t['seq']: vocab.setdefault(s, len(vocab))
        self.V = len(vocab); self.vocab = vocab
        self.enc = [np.array([vocab[s] for s in t['seq']], dtype=np.int64) for t in T]
        self.lens = np.array([len(e) for e in self.enc])
        # class membership per token for the frame / content split
        self.cls = []
        for t in T:
            if kind in ('indus', 'im77'):
                self.cls.append(t['lab'])
            elif kind == 'ur3':
                self.cls.append(['TITLE' if w in UR3_TITLES else 'NAME' for w in t['seq']])
            elif kind == 'linb':
                self.cls.append(['IDEO' if (w == 'NUM' or re.fullmatch(r'[A-Z][A-Z0-9*+]*', w)) else 'SYL' for w in t['seq']])
            else:
                self.cls.append(None)
        self.site = np.array([t['site'] for t in T]); self.type = np.array([t['type'] for t in T])
        self.area = np.array([t.get('area', '') for t in T])

def jsd(p, q):
    """Jensen-Shannon divergence in bits between two count vectors"""
    p = p / p.sum(); q = q / q.sum(); m = (p + q) / 2
    def kl(a, b):
        mask = a > 0
        return float(np.sum(a[mask] * np.log2(a[mask] / b[mask])))
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)

def counts(arrs, V):
    if not arrs: return np.zeros(V)
    return np.bincount(np.concatenate(arrs), minlength=V).astype(float)

def bigram_counts(arrs, V):
    bs = [a[:-1] * V + a[1:] for a in arrs if len(a) > 1]
    if not bs: return collections.Counter()
    return collections.Counter(np.concatenate(bs).tolist())

def jsd_counter(c1, c2):
    keys = list(set(c1) | set(c2))
    if not keys: return 0.0
    p = np.array([c1.get(k, 0) for k in keys], float); q = np.array([c2.get(k, 0) for k in keys], float)
    if p.sum() == 0 or q.sum() == 0: return 0.0
    return jsd(p, q)

def frozen_set(arrs, V):
    adj = collections.Counter(); co = collections.Counter()
    for a in arrs:
        u = set(a.tolist())
        for x in range(len(a) - 1): adj[(int(a[x]), int(a[x + 1]))] += 1
        ul = sorted(u)
        for i in range(len(ul)):
            for j in range(len(ul)):
                if i != j: co[(ul[i], ul[j])] += 1
    return {k for k, v in adj.items() if v >= 3 and co[k] > 0 and v / co[k] >= 0.9}

def class_counts(C, idx, labels, V):
    arrs = []
    for i in idx:
        lab = C.cls[i]
        if lab is None: continue
        m = np.array([l in labels for l in lab])
        if m.any(): arrs.append(C.enc[i][m])
    return counts(arrs, V)

def components(C, ia, ib, top30):
    """all divergence components between index sets ia and ib of corpus C"""
    A = [C.enc[i] for i in ia]; B = [C.enc[i] for i in ib]; V = C.V
    ua = counts(A, V); ub = counts(B, V)
    out = {'uni': jsd(ua, ub), 'bi': jsd_counter(bigram_counts(A, V), bigram_counts(B, V))}
    pooled = ua + ub; m = pooled >= 5
    if m.sum() >= 5:
        r = spearmanr(ua[m], ub[m]).correlation
        out['rho'] = 1 - (0.0 if np.isnan(r) else r)
    else: out['rho'] = float('nan')
    out['init'] = jsd(np.bincount([a[0] for a in A], minlength=V).astype(float), np.bincount([b[0] for b in B], minlength=V).astype(float))
    out['fin'] = jsd(np.bincount([a[-1] for a in A], minlength=V).astype(float), np.bincount([b[-1] for b in B], minlength=V).astype(float))
    ma = [a[1:-1] for a in A if len(a) > 2]; mb = [b[1:-1] for b in B if len(b) > 2]
    out['midinv'] = jsd(counts(ma, V), counts(mb, V)) if ma and mb else float('nan')
    la = np.bincount(np.minimum([len(a) for a in A], 12), minlength=13).astype(float)
    lb = np.bincount(np.minimum([len(b) for b in B], 12), minlength=13).astype(float)
    out['len'] = jsd(la, lb)
    t30 = np.zeros(V, bool); t30[top30] = True
    out['top30'] = jsd(ua[t30], ub[t30]) if ua[t30].sum() and ub[t30].sum() else float('nan')
    out['rest'] = jsd(ua[~t30], ub[~t30]) if ua[~t30].sum() and ub[~t30].sum() else float('nan')
    fa = frozen_set(A, V); fb = frozen_set(B, V)
    out['frozen'] = 1 - len(fa & fb) / len(fa | fb) if (fa | fb) else float('nan')
    if C.kind in ('indus', 'im77'):
        def menu(idx, lab, pos):
            c = collections.Counter()
            for i in idx:
                L = C.cls[i]; s = C.T[i]['seq']
                k = [j for j, l in enumerate(L) if l == lab]
                c[s[k[0]] if k else 'none'] += 1
            return c
        out['opener'] = jsd_counter(menu(ia, 'OPENER', 0), menu(ib, 'OPENER', 0))
        out['closer'] = jsd_counter(menu(ia, 'CLOSER', -1), menu(ib, 'CLOSER', -1))
        def rates(idx):
            r = []
            for lab in ('OPENER', 'MARKER', 'CLOSER', 'SUFFIX', 'TITLE'):
                r.append(np.mean([lab in C.cls[i] for i in idx]))
            return np.array(r)
        out['rates'] = float(np.mean(np.abs(rates(ia) - rates(ib))))
        na = class_counts(C, ia, {'NAME'}, V); nb = class_counts(C, ib, {'NAME'}, V)
        out['name'] = jsd(na, nb) if na.sum() and nb.sum() else float('nan')
        qa = class_counts(C, ia, {'COUNT'}, V); qb = class_counts(C, ib, {'COUNT'}, V)
        out['num'] = jsd(qa, qb) if qa.sum() and qb.sum() else float('nan')
        ta = class_counts(C, ia, {'TITLE'}, V); tb = class_counts(C, ib, {'TITLE'}, V)
        out['title'] = jsd(ta, tb) if ta.sum() and tb.sum() else float('nan')
    elif C.kind == 'ur3':
        ta = class_counts(C, ia, {'TITLE'}, V); tb = class_counts(C, ib, {'TITLE'}, V)
        out['title'] = jsd(ta, tb) if ta.sum() and tb.sum() else float('nan')
        na = class_counts(C, ia, {'NAME'}, V); nb = class_counts(C, ib, {'NAME'}, V)
        out['name'] = jsd(na, nb) if na.sum() and nb.sum() else float('nan')
    elif C.kind == 'linb':
        ta = class_counts(C, ia, {'IDEO'}, V); tb = class_counts(C, ib, {'IDEO'}, V)
        out['ideo'] = jsd(ta, tb) if ta.sum() and tb.sum() else float('nan')
        na = class_counts(C, ia, {'SYL'}, V); nb = class_counts(C, ib, {'SYL'}, V)
        out['syl'] = jsd(na, nb) if na.sum() and nb.sum() else float('nan')
    return out

def reuse_share(RA, RB):
    """RA, RB: lists of raw (uncollapsed) seqs. mean over directions of share of distinct texts attested in the other group"""
    sa = set(RA); sb = set(RB)
    if not sa or not sb: return float('nan')
    return 0.5 * (len(sa & sb) / len(sa) + len(sa & sb) / len(sb))

# ------------------------------------------------------------------ the pair test
def pair_test(C, idxA, idxB, n, nperm, nsub, rawA=None, rawB=None, label=''):
    """idxA/idxB: index arrays into C (collapsed). n texts per side. Returns dict of per-component
    between mean, within mean (pooled over the sites that allow a split), excess, ratio, P"""
    top30 = np.argsort(-(counts([C.enc[i] for i in np.concatenate([idxA, idxB])], C.V)))[:30]
    between = collections.defaultdict(list)
    for _ in range(nsub):
        a = rng.choice(idxA, n, replace=False); b = rng.choice(idxB, n, replace=False)
        for k, v in components(C, a, b, top30).items(): between[k].append(v)
    within = collections.defaultdict(list)
    wsites = []
    for idx, nm in ((idxA, 'A'), (idxB, 'B')):
        if len(idx) >= 2 * n:
            wsites.append(nm)
            for _ in range(nperm):
                p = rng.permutation(idx)
                for k, v in components(C, p[:n], p[n:2 * n], top30).items(): within[k].append(v)
    res = {'n': n, 'nA': len(idxA), 'nB': len(idxB), 'within_from': ''.join(wsites), 'comp': {}}
    for k in between:
        bt = np.array(between[k], float); wt = np.array(within.get(k, [np.nan]), float)
        bm = float(np.nanmean(bt)); wm = float(np.nanmean(wt)); wsd = float(np.nanstd(wt))
        P = float((np.sum(wt >= np.nanmedian(bt)) + 1) / (len(wt) + 1)) if len(wt) > 1 else float('nan')
        res['comp'][k] = dict(between=bm, within=wm, within_sd=wsd, excess=bm - wm, ratio=(bm / wm if wm > 0 else float('nan')),
                              z=((bm - wm) / wsd if wsd > 0 else float('nan')), P=P)
    if rawA is not None and rawB is not None:
        # reuse on uncollapsed objects, n objects per side
        bt = []; wt = []
        rA = list(rawA); rB = list(rawB)
        for _ in range(nsub):
            a = rng.choice(len(rA), min(n, len(rA)), replace=False); b = rng.choice(len(rB), min(n, len(rB)), replace=False)
            bt.append(reuse_share([rA[i] for i in a], [rB[i] for i in b]))
        for R in (rA, rB):
            if len(R) >= 2 * n:
                for _ in range(nperm):
                    p = rng.permutation(len(R))
                    wt.append(reuse_share([R[i] for i in p[:n]], [R[i] for i in p[n:2 * n]]))
        bm = float(np.nanmean(bt)); wm = float(np.nanmean(wt)) if wt else float('nan')
        eps = 0.5 / n   # half a shared text per n: keeps the ratio finite when no text is shared
        res['comp']['reuse'] = dict(between=bm, within=wm, within_sd=float(np.nanstd(wt)) if wt else float('nan'),
                                    excess=wm - bm, ratio=((wm + eps) / (bm + eps)),
                                    z=float('nan'), P=float((np.sum(np.array(wt) <= np.median(bt)) + 1) / (len(wt) + 1)) if wt else float('nan'))
    return res

def fmt(res, keys=None):
    keys = keys or list(res['comp'])
    lines = []
    for k in keys:
        if k not in res['comp']: continue
        c = res['comp'][k]
        lines.append(f"    {k:7s} between {c['between']:.4f}  within {c['within']:.4f} (sd {c['within_sd']:.4f})  excess {c['excess']:+.4f}  x{c['ratio']:.2f}  z {c['z']:+.1f}  P {c['P']:.3f}")
    return '\n'.join(lines)

def groups(C, by='site', types=None, min_n=1):
    """index arrays per group label (site or area), optionally restricted to object types"""
    sel = np.ones(len(C.T), bool) if types is None else np.isin(C.type, list(types))
    lab = C.site if by == 'site' else np.array([f'{s}|{a}' for s, a in zip(C.site, C.area)])
    G = {}
    for g in sorted(set(lab[sel])):
        idx = np.where(sel & (lab == g))[0]
        if len(idx) >= min_n: G[g] = idx
    return G

def raw_groups(T, by='site', types=None):
    G = collections.defaultdict(list)
    for t in T:
        if types is not None and t['type'] not in types: continue
        G[t['site'] if by == 'site' else f"{t['site']}|{t['area']}"].append(t['seq'])
    return G

# ------------------------------------------------------------------ synthetic anchors
def fit_chain(seqs, V):
    """order-1 chain with START/END over encoded arrays"""
    S = V; E = V + 1
    M = np.ones((V + 2, V + 2)) * 0.01
    for a in seqs:
        prev = S
        for x in a: M[prev, x] += 1; prev = x
        M[prev, E] += 1
    M[:, S] = 0; M[E, :] = 0
    return M / M.sum(1, keepdims=True).clip(1e-12)

def gen_chain(M, V, n, rng_, maxlen=14):
    S = V; E = V + 1; out = []
    while len(out) < n:
        s = []; cur = S
        for _ in range(maxlen):
            cur = rng_.choice(V + 2, p=M[cur])
            if cur == E: break
            s.append(int(cur))
        if len(s) >= 2: out.append(tuple(s))
    return out

def synthetic_corpora(base_texts, sizes, V):
    """base_texts: encoded int arrays. returns dict name -> list of text dicts with 'site' labels"""
    r = np.random.default_rng(4848)
    M = fit_chain(base_texts, V)
    out = {}
    # central: one generator shared by all sites
    T = []
    for s, n in sizes.items():
        T += [dict(site=s, type='x', seq=q, area='') for q in gen_chain(M, V, n, r)]
    out['SYN_central'] = T
    # local-ranks: same inventory, each site draws its own random relabelling of the shared chain (independent frequency ranks)
    T = []
    for s, n in sizes.items():
        perm = r.permutation(V)
        T += [dict(site=s, type='x', seq=tuple(int(perm[x]) for x in q), area='') for q in gen_chain(M, V, n, r)]
    out['SYN_local'] = T
    # half-local: each site's generator = 50/50 mixture of the shared chain and its own relabelled chain
    T = []
    for s, n in sizes.items():
        perm = r.permutation(V)
        qs = gen_chain(M, V, n, r)
        T += [dict(site=s, type='x', seq=(tuple(int(perm[x]) for x in q) if r.random() < 0.5 else q), area='') for q in qs]
    out['SYN_half'] = T
    # local-rates: shared inventory and shared bigram structure, but each site's unigram frequencies are independently perturbed (log-normal sd 1)
    T = []
    for s, n in sizes.items():
        w = np.exp(r.normal(0, 1.0, V + 2)); Ms = M * w[None, :]; Ms[:, V] = 0; Ms = Ms / Ms.sum(1, keepdims=True).clip(1e-12)
        T += [dict(site=s, type='x', seq=q, area='') for q in gen_chain(Ms, V, n, r)]
    out['SYN_perturbed'] = T
    return out

# ------------------------------------------------------------------ cycles
def run_pairs(C, G, n, out, label, raw=None, keys=None, pairs=None, nperm=NPERM, nsub=NSUB):
    names = list(G)
    pairs = pairs or [(a, b) for i, a in enumerate(names) for b in names[i + 1:]]
    R = {}
    for a, b in pairs:
        if min(len(G[a]), len(G[b])) < n: continue
        ra = raw.get(a) if raw else None; rb = raw.get(b) if raw else None
        res = pair_test(C, G[a], G[b], n, nperm, nsub, ra, rb)
        R[f'{a} vs {b}'] = res
        out.append(f"\n  [{label}] {a} ({len(G[a])}) vs {b} ({len(G[b])}), n = {n} per side, within from {res['within_from'] or 'none'}")
        out.append(fmt(res, keys))
    return R

def summary_table(R, keys, out, label):
    out.append(f"\n  SUMMARY {label}: ratio (between/within) per component; reuse = within/between")
    out.append('  ' + f"{'pair':44s}" + ''.join(f'{k:>8s}' for k in keys))
    for p, res in R.items():
        out.append('  ' + f'{p[:44]:44s}' + ''.join(f"{res['comp'][k]['ratio']:8.2f}" if k in res['comp'] and np.isfinite(res['comp'][k]['ratio']) else f"{'-':>8s}" for k in keys))
    out.append('  ' + f"{'excess (bits / units)':44s}")
    for p, res in R.items():
        out.append('  ' + f'{p[:44]:44s}' + ''.join(f"{res['comp'][k]['excess']:8.3f}" if k in res['comp'] and np.isfinite(res['comp'][k]['excess']) else f"{'-':>8s}" for k in keys))

INDUS_KEYS = GENERIC + ['reuse', 'opener', 'closer', 'rates', 'title', 'name', 'num']

def cycle1():
    out = [f'# S-DARK-48 cycle 1, level {LV}, nperm {NPERM}, nsub {NSUB}: Indus between-city divergence as a multiple of within-city']
    Tw = load_wells(LV)
    parse = make_parser(learn_qual([t['seq'] for t in Tw]))
    add_labels(Tw, parse)
    Tc = collapse(Tw)
    out.append(f'Wells complete texts >= 2 signs: {len(Tw)}; collapsed one per distinct text per site x type: {len(Tc)}')
    C = Corpus(Tc, 'indus')
    R = {}
    for types, tl in ((('seal',), 'seals'), (None, 'all types')):
        G = groups(C, 'site', types, min_n=40); raw = raw_groups(Tw, 'site', types)
        out.append(f'\n== {tl}: groups {dict((str(g), len(i)) for g, i in G.items())}')
        for n in (200, 30):
            big = {g: i for g, i in G.items() if len(i) >= n}
            if len(big) < 2: continue
            out.append(f'\n-- {tl}, n = {n}')
            keys = INDUS_KEYS
            Rn = run_pairs(C, big, n, out, f'{tl} n={n}', raw, keys)
            summary_table(Rn, keys, out, f'{tl} n={n}')
            R[f'{tl}|n{n}'] = Rn
    # area-sections inside Mohenjo-daro and Harappa (seals + all), n = 30
    for types, tl in ((('seal',), 'seals'), (None, 'all types')):
        G = groups(C, 'area', types, min_n=60); raw = raw_groups(Tw, 'area', types)
        G = {g: i for g, i in G.items() if not g.endswith('|--') and not g.endswith('|') and g.split('|')[0] in ('Mohenjo-daro', 'Harappa')}
        out.append(f'\n== area-sections ({tl}, >= 60 collapsed texts): {dict((str(g), len(i)) for g, i in G.items())}')
        if len(G) >= 2:
            Rn = run_pairs(C, G, 30, out, f'sections {tl} n=30', raw, INDUS_KEYS)
            summary_table(Rn, INDUS_KEYS, out, f'sections {tl} n=30 (same-city pairs vs cross-city pairs)')
            R[f'sections|{tl}'] = Rn
            # same-city vs cross-city mean ratio
            for k in INDUS_KEYS:
                same = [r['comp'][k]['ratio'] for p, r in Rn.items() if k in r['comp'] and p.split(' vs ')[0].split('|')[0] == p.split(' vs ')[1].split('|')[0]]
                cross = [r['comp'][k]['ratio'] for p, r in Rn.items() if k in r['comp'] and p.split(' vs ')[0].split('|')[0] != p.split(' vs ')[1].split('|')[0]]
                if same and cross:
                    out.append(f"  {k:7s} mean ratio same-city sections {np.nanmean(same):.2f} ({len(same)} pairs)  cross-city sections {np.nanmean(cross):.2f} ({len(cross)} pairs)")
    json.dump(R, open(DARK + f'loop48_c1_{LV}.json', 'w'), indent=0, default=float)
    open(DARK + f'loop48_c1_{LV}.txt', 'w').write('\n'.join(out) + '\n')
    print('\n'.join(out))

def cycle2():
    out = [f'# S-DARK-48 cycle 2, nperm {NPERM}, nsub {NSUB}: the calibration ladder (reference corpora with provenance + synthetic anchors), n = 200']
    n = 200; R = {}
    refs = [('ur3_words', 'ur3', ['Umma', 'Girsu', 'Puzriš-Dagan', 'Nippur', 'Garšana', 'Ur'], None),
            ('ur3_syll', 'ur3s', ['Umma', 'Girsu', 'Puzriš-Dagan', 'Nippur', 'Garšana', 'Ur'], None),
            ('linb_syll', 'linb', ['KN', 'PY', 'TH', 'MY'], None),
            ('linb_words', 'linbw', ['KN', 'PY', 'TH', 'MY'], None),
            ('proto_cuneiform', None, ['Uruk', 'Umma', 'Ur', 'mod.Jemdet Nasr', 'Larsa'], None),
            ('proto_elamite', None, ['Susa', 'other(Tepe Yahya)', 'Anšan'], None),
            ('latin_edh', None, ['Lazio', 'Roma', 'Napoli', 'Campania', 'Puglia', "L'Aquila", 'Umbria', 'Abruzzo', 'Udine'], None)]
    for name, kind, sites, types in refs:
        T = load_ref(name); T = [t for t in T if t['site'] in sites]
        Tc = collapse(T)
        C = Corpus(Tc, 'ur3' if kind in ('ur3',) else ('linb' if kind == 'linb' else None))
        G = groups(C, 'site', None, min_n=n); raw = raw_groups(T, 'site')
        out.append(f'\n== {name}: {len(T)} texts, {len(Tc)} collapsed; groups >= {n}: {dict((str(g), len(i)) for g, i in G.items())}')
        keys = GENERIC + ['reuse'] + (['title', 'name'] if C.kind == 'ur3' else []) + (['ideo', 'syl'] if C.kind == 'linb' else [])
        # Proto-Elamite small sites: n = 30 design
        nn = n if len(G) >= 2 else 30
        if nn != n:
            G = groups(C, 'site', None, min_n=nn); out.append(f'   (small sites: n = {nn}; groups {dict((str(g), len(i)) for g, i in G.items())})')
        Rn = run_pairs(C, G, nn, out, name, raw, keys)
        summary_table(Rn, keys, out, name)
        R[name] = Rn
    # synthetic anchors fitted on Indus seals (MD + H, seq_raw), sizes = MD / H seal counts
    Tw = load_wells('seq_raw'); parse = make_parser(learn_qual([t['seq'] for t in Tw])); add_labels(Tw, parse)
    Tc = [t for t in collapse(Tw) if t['type'] == 'seal' and t['site'] in ('Mohenjo-daro', 'Harappa')]
    Cb = Corpus(Tc, 'indus')
    sizes = {'MD': int(np.sum(Cb.site == 'Mohenjo-daro')), 'H': int(np.sum(Cb.site == 'Harappa'))}
    out.append(f'\n== synthetic anchors: order-1 chain fitted on {len(Tc)} MD + H seal texts (seq_raw), sites of size {sizes}')
    for sname, T in synthetic_corpora(Cb.enc, sizes, Cb.V).items():
        Tc2 = collapse(T); C = Corpus(Tc2, None)
        G = groups(C, 'site', None, min_n=n); raw = raw_groups(T, 'site')
        out.append(f'\n== {sname}: {len(T)} texts, {len(Tc2)} collapsed; groups {dict((str(g), len(i)) for g, i in G.items())}')
        Rn = run_pairs(C, G, n, out, sname, raw, GENERIC + ['reuse'])
        summary_table(Rn, GENERIC + ['reuse'], out, sname)
        R[sname] = Rn
    # ladder: mean ratio of the generic components per corpus
    out.append('\n== LADDER (mean over pairs; generic components; ratio between/within at n = 200 unless noted)')
    out.append('  ' + f"{'corpus':18s}" + ''.join(f'{k:>8s}' for k in GENERIC + ['reuse']) + f"{'pairs':>7s}")
    for name, Rn in R.items():
        if not Rn: continue
        row = []
        for k in GENERIC + ['reuse']:
            v = [r['comp'][k]['ratio'] for r in Rn.values() if k in r['comp'] and np.isfinite(r['comp'][k]['ratio'])]
            row.append(f'{np.mean(v):8.2f}' if v else f"{'-':>8s}")
        out.append('  ' + f'{name:18s}' + ''.join(row) + f'{len(Rn):7d}')
    out.append('  excess (bits; reuse = within - between share)')
    for name, Rn in R.items():
        if not Rn: continue
        row = []
        for k in GENERIC + ['reuse']:
            v = [r['comp'][k]['excess'] for r in Rn.values() if k in r['comp'] and np.isfinite(r['comp'][k]['excess'])]
            row.append(f'{np.mean(v):8.3f}' if v else f"{'-':>8s}")
        out.append('  ' + f'{name:18s}' + ''.join(row))
    json.dump(R, open(DARK + 'loop48_c2.json', 'w'), indent=0, default=float)
    open(DARK + 'loop48_c2.txt', 'w').write('\n'.join(out) + '\n')
    print('\n'.join(out))

def cycle3():
    out = [f'# S-DARK-48 cycle 3, level {LV}, nperm {NPERM}, nsub {NSUB}: IM77 replication and the held-out small-site prediction']
    R = {}
    # IM77 (M space, own parser)
    Ti = load_im77(); add_labels(Ti, im77_parser()); Tic = collapse(Ti)
    C = Corpus(Tic, 'im77')
    out.append(f'IM77 texts >= 2 signs: {len(Ti)}; collapsed: {len(Tic)}')
    for types, tl in ((('seal',), 'seals'), (None, 'all types')):
        G = groups(C, 'site', types, min_n=40); raw = raw_groups(Ti, 'site', types)
        out.append(f'\n== IM77 {tl}: groups {dict((str(g), len(i)) for g, i in G.items())}')
        for n in (200, 25):
            big = {g: i for g, i in G.items() if len(i) >= n}
            if len(big) < 2: continue
            out.append(f'\n-- IM77 {tl}, n = {n}')
            Rn = run_pairs(C, big, n, out, f'IM77 {tl} n={n}', raw, INDUS_KEYS)
            summary_table(Rn, INDUS_KEYS, out, f'IM77 {tl} n={n}')
            R[f'IM77|{tl}|n{n}'] = Rn
    # Wells held-out small sites vs Mohenjo-daro at n = 25: is their frame / closer-menu divergence inside the MD-H band?
    Tw = load_wells(LV); parse = make_parser(learn_qual([t['seq'] for t in Tw])); add_labels(Tw, parse); Tc = collapse(Tw)
    Cw = Corpus(Tc, 'indus')
    for types, tl in ((('seal',), 'seals'), (None, 'all types')):
        G = groups(Cw, 'site', types, min_n=25); raw = raw_groups(Tw, 'site', types)
        out.append(f'\n== Wells {tl} n = 25, every site with >= 25 collapsed texts vs Mohenjo-daro and vs Harappa: {dict((str(g), len(i)) for g, i in G.items())}')
        pairs = [('Mohenjo-daro', g) for g in G if g != 'Mohenjo-daro'] + [('Harappa', g) for g in G if g not in ('Mohenjo-daro', 'Harappa')]
        Rn = run_pairs(Cw, G, 25, out, f'Wells {tl} n=25', raw, INDUS_KEYS, pairs=pairs)
        summary_table(Rn, INDUS_KEYS, out, f'Wells {tl} n=25')
        R[f'Wells heldout|{tl}'] = Rn
        # band test: MD-H ratio as the reference; small-site ratios vs the MD-H within-null distribution
        ref = Rn.get('Mohenjo-daro vs Harappa')
        if ref:
            out.append(f'\n  BAND TEST ({tl}): a held-out site is "inside the band" if its between-MD median does not exceed the within-MD 97.5th percentile'
                       ' and its ratio is <= the MD-H ratio x 1.5')
            for k in ('opener', 'closer', 'rates', 'fin', 'init', 'uni', 'name', 'reuse'):
                row = []
                for p, r in Rn.items():
                    if not p.startswith('Mohenjo-daro vs') or 'Harappa' in p or k not in r['comp']: continue
                    c = r['comp'][k]; rc = ref['comp'][k]
                    inside = (c['P'] > 0.025) and (c['ratio'] <= rc['ratio'] * 1.5 if np.isfinite(rc['ratio']) else True)
                    row.append(f"{p.split(' vs ')[1][:11]} x{c['ratio']:.2f} P{c['P']:.2f} {'IN' if inside else 'OUT'}")
                out.append(f"    {k:7s} MD-H x{ref['comp'][k]['ratio']:.2f} | " + '; '.join(row))
    json.dump(R, open(DARK + f'loop48_c3_{LV}.json', 'w'), indent=0, default=float)
    open(DARK + f'loop48_c3_{LV}.txt', 'w').write('\n'.join(out) + '\n')
    print('\n'.join(out))

if __name__ == '__main__':
    {1: cycle1, 2: cycle2, 3: cycle3}[CY]()
