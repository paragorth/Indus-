"""S-DARK-45: WHO OWNS THE MIDDLE INVENTORY? Is the set of middle elements partitioned by head (closer),
by site, by object type, or shared by all?
Cycle 1  element x {head, site, type} mutual information and 'loyalty' (>= 80% of an element's tokens under one
         label), adjacent qualifier (the sign directly before the closer, S303) kept apart from the rest of the
         middle; nulls: labels permuted among texts within strata (1,000x); S366 chain-model null (heads permuted
         among texts with the same adjacent sign).
Cycle 2  block structure: spectral co-clustering of the element x head and element x site matrices; block purity
         and diagonal concentration vs the permutation null; controls Ur III legends (name syllables x profession)
         and Proto-Elamite (signs x final numeral class).
Cycle 3  replication on held-out sites and IM77 (M space, bridge_extended + loop 27 proposals), and the concrete
         prediction on the 324 IM77-only new texts: does a head-loyal element keep its head there?
Usage: python3 tools/dark_loop45.py <1|2|3> <seq_raw|seq_strong|seq_all> [nperm]
Texts: data/derived/merged-corpus-canonical.json (older build, S-DARK-23 caution), complete, direction recorded,
>= 2 signs; collapsed to one per distinct text per site x object type (S-DARK-13).
"""
import sys, json, csv, collections, random, math
import numpy as np
sys.path.insert(0, '/home/user/Indus-/tools')
from dark_loop37 import OPEN, MARK, MJAR, SUF, CL, FISH, NUM, learn_qual, make_parser, pval

ROOT = '/home/user/Indus-/'
DARK = ROOT + 'data/derived/dark/'
CY = int(sys.argv[1]) if len(sys.argv) > 1 else 1
LV = sys.argv[2] if len(sys.argv) > 2 else 'seq_raw'
NP = int(sys.argv[3]) if len(sys.argv) > 3 else 1000
rnd = random.Random(45 + CY)
nprnd = np.random.default_rng(45 + CY)
BIG = ('Mohenjo-daro', 'Harappa')
SITES = ('Mohenjo-daro', 'Harappa', 'Lothal', 'Kalibangan', 'Dholavira', 'Chanhu-daro')
MINTOK = 5

def otype(t):
    t0 = t.split(':')[0]
    return {'SEAL': 'seal', 'TAB': 'tablet', 'POT': 'pot', 'TAG': 'sealing'}.get(t0, 'other')

def sgroup(s): return s if s in SITES else 'other'

# ------------------------------------------------------------------ texts
def load_wells(level):
    C = json.load(open(ROOT + 'data/derived/merged-corpus-canonical.json'))
    T = []
    for r in C:
        s = r[level]
        if not s or len(s) < 2 or r['complete'] != 'Y' or r['dir.'].strip() == '-': continue
        T.append(dict(id=r['cisi'], site=sgroup(r['site']), rawsite=r['site'], ot=otype(r['type']), seq=list(s)))
    return T

def collapse(T):
    """one per distinct text per site x object type"""
    seen = set(); out = []
    for t in T:
        k = (t['site'], t['ot'], tuple(t['seq']))
        if k in seen: continue
        seen.add(k); out.append(t)
    return out

def parse_all(T, parse, closers):
    """adds head, adj (sign directly before the closer or None), mid (non-adjacent middle tokens), qual (TITLE tokens)"""
    for t in T:
        s = t['seq']; lab = parse(s)
        cl = [i for i, l in enumerate(lab) if l == 'CLOSER']
        if cl:
            ci = cl[0]; t['head'] = 'C%d' % s[ci]
            midpos = [i for i in range(ci) if lab[i] in ('NAME', 'COUNT', 'TITLE')]
            adjpos = ci - 1 if ci - 1 >= 0 and lab[ci - 1] in ('NAME', 'COUNT', 'TITLE') else None
        else:
            t['head'] = 'none'
            midpos = [i for i, l in enumerate(lab) if l in ('NAME', 'COUNT', 'TITLE')]
            adjpos = None
        t['adj'] = s[adjpos] if adjpos is not None else None
        t['mid'] = [s[i] for i in midpos if i != adjpos]
        t['allmid'] = [s[i] for i in midpos]
        t['lab'] = lab
    return T

# ------------------------------------------------------------------ element tables
def tokens(T, field, labfield):
    """list of (element, label, text index)"""
    out = []
    for i, t in enumerate(T):
        for e in t[field]: out.append((e, t[labfield], i))
    return out

def mi_bits(x, y):
    """mutual information in bits of two integer-coded arrays"""
    nx, ny = x.max() + 1, y.max() + 1
    joint = np.bincount(x * ny + y, minlength=nx * ny).reshape(nx, ny).astype(float)
    n = joint.sum(); px = joint.sum(1) / n; py = joint.sum(0) / n
    p = joint / n
    nz = p > 0
    return float((p[nz] * np.log2(p[nz] / (px[:, None] * py[None, :])[nz])).sum()), float(-(px[px > 0] * np.log2(px[px > 0])).sum())

def loyalty(x, y, mintok=MINTOK, cut=0.8):
    """share of elements (>= mintok tokens) with >= cut of tokens under one label; also share seen under >= 2 labels"""
    nx, ny = x.max() + 1, y.max() + 1
    joint = np.bincount(x * ny + y, minlength=nx * ny).reshape(nx, ny)
    tot = joint.sum(1); ok = tot >= mintok
    if ok.sum() == 0: return float('nan'), float('nan'), 0, 0
    mx = joint[ok].max(1) / tot[ok]
    nlab = (joint[ok] > 0).sum(1)
    dom = joint.sum(0).argmax()
    minor = (mx >= cut) & (joint[ok].argmax(1) != dom)
    return float((mx >= cut).mean()), float((nlab >= 2).mean()), int(ok.sum()), int(minor.sum())

def permute_labels(T, labfield, strata, nrep, extra=None):
    """yield label arrays (per text) with labels permuted among texts of the same stratum"""
    groups = collections.defaultdict(list)
    for i, t in enumerate(T): groups[strata(t)].append(i)
    base = [t[labfield] for t in T]
    for _ in range(nrep):
        lab = list(base)
        for idx in groups.values():
            vals = [base[i] for i in idx]; rnd.shuffle(vals)
            for i, v in zip(idx, vals): lab[i] = v
        yield lab

def lenbin(t): return min(len(t['allmid']), 5)

def analyse(T, field, labfield, strata_list, nrep, out, tag):
    """MI, H, loyalty for element tokens in `field` against text label `labfield`, with one or more permutation nulls"""
    toks = tokens(T, field, labfield)
    if len(toks) < 50:
        out.append(f'  {tag}: only {len(toks)} tokens, skipped'); return None
    elems = sorted(set(e for e, _, _ in toks)); eid = {e: i for i, e in enumerate(elems)}
    labs = sorted(set(t[labfield] for t in T)); lid = {l: i for i, l in enumerate(labs)}
    x = np.array([eid[e] for e, _, _ in toks]); ti = np.array([i for _, _, i in toks])
    y = np.array([lid[l] for _, l, _ in toks])
    mi, hx = mi_bits(x, y); loy, shared, ne, minor = loyalty(x, y)
    # label entropy for normalisation
    _, hy = mi_bits(y, x)
    res = dict(tag=tag, ntok=len(toks), nelem=len(elems), nelem_tested=ne, nlab=len(labs), MI=mi, H_elem=hx, H_lab=hy,
               MI_over_Hlab=mi / hy if hy else float('nan'), loyal=loy, shared=shared, loyal_minor=minor, nulls={})
    line = f'  {tag}: {len(toks)} tokens, {len(elems)} elements ({ne} with >= {MINTOK} tokens), {len(labs)} labels; MI {mi:.4f} bits (= {100*mi/hy:.1f}% of H(label) {hy:.3f}); loyal {loy:.3f} (loyal to a NON-dominant label: {minor}); seen under >= 2 labels {shared:.3f}'
    out.append(line)
    for sname, strata in strata_list:
        nm, nl, ns, nmin = [], [], [], []
        for lab in permute_labels(T, labfield, strata, nrep):
            yp = np.array([lid[lab[i]] for i in ti])
            m, _ = mi_bits(x, yp); l, s, _, mn = loyalty(x, yp)
            nm.append(m); nl.append(l); ns.append(s); nmin.append(mn)
        res['nulls'][sname] = dict(MI_mean=float(np.mean(nm)), MI_sd=float(np.std(nm)), MI_p=pval(mi, nm),
                                   loyal_mean=float(np.mean(nl)), loyal_p=pval(loy, nl),
                                   shared_mean=float(np.mean(ns)), shared_p_lo=pval(shared, ns, 'lo'),
                                   minor_mean=float(np.mean(nmin)), minor_p=pval(minor, nmin))
        out.append(f'     null [{sname}] ({nrep}x): MI {np.mean(nm):.4f} +- {np.std(nm):.4f} (P = {pval(mi, nm):.3f}; excess {mi-np.mean(nm):+.4f} bits = {100*(mi-np.mean(nm))/hy:+.1f}% of H(label)); loyal {np.mean(nl):.3f} (P = {pval(loy, nl):.3f}); non-dominant loyal {np.mean(nmin):.2f} (P = {pval(minor, nmin):.3f}); >= 2 labels {np.mean(ns):.3f} (P_lo = {pval(shared, ns, "lo"):.3f})')
    return res

def loyal_elements(T, field, labfield, mintok=MINTOK, cut=0.8):
    c = collections.defaultdict(collections.Counter)
    for t in T:
        for e in t[field]: c[e][t[labfield]] += 1
    out = {}
    for e, cnt in c.items():
        n = sum(cnt.values())
        if n >= mintok:
            l, k = cnt.most_common(1)[0]
            if k / n >= cut: out[e] = (l, k, n)
    return out

# ------------------------------------------------------------------ IM77
MCL = {342: 'C740', 211: 'C520', 12: 'C151', 15: 'C156', 254: 'C527', 60: 'C226', 245: 'C617', 328: 'C700', 66: 'C236'}
MOPEN = {267, 391, 293, 150}; MMARK = {99, 100, 123}; MMJAR = {343, 344, 345, 346}; MSUF = {176, 1}

def load_im77():
    rows = list(csv.DictReader(open(ROOT + 'data/im77/im77_corpus_lines.csv')))
    objs = collections.OrderedDict()
    for r in rows:
        if r['line'] == '9' or not r['signs_clean'].strip(): continue
        key = (r['text_no'], r['side'])
        o = objs.setdefault(key, dict(id=r['text_no'] + '.' + r['side'], site=r['site'], ot=r['object_type'], seq=[], doubt=False))
        o['seq'].extend(int(x) for x in r['signs_clean'].split() if x != '0')
        if r['doubtful_positions'].strip(): o['doubt'] = True
    T = []
    smap = {'Mohenjodaro': 'Mohenjo-daro', 'Harappa': 'Harappa', 'Lothal': 'Lothal', 'Kalibangan': 'Kalibangan', 'Chanhudaro': 'Chanhu-daro'}
    tmap = {'seal': 'seal', 'sealing': 'sealing', 'miniature tablet': 'tablet', 'copper tablet': 'tablet', 'pottery graffito': 'pot'}
    for key, o in objs.items():
        if len(o['seq']) < 2: continue
        T.append(dict(id=o['id'], key=key, site=smap.get(o['site'], 'other'), rawsite=o['site'], ot=tmap.get(o['ot'], 'other'), seq=o['seq']))
    return T

def parse_im77(T):
    for t in T:
        s = t['seq']; n = len(s); i = 0; j = n
        lab = ['NAME'] * n
        if s[0] in MOPEN:
            lab[0] = 'OPENER'; i = 1
            if n > 1 and s[1] in MMARK:
                lab[1] = 'MARKER'; i = 2
                if s[0] == 293 and n > 2 and s[2] in MMJAR: lab[2] = 'MARKER'; i = 3
        while j - 1 > i and s[j - 1] in MSUF and (s[j - 2] in MCL or s[j - 2] in MSUF): lab[j - 1] = 'SUFFIX'; j -= 1
        if j - 1 >= i and s[j - 1] in MCL:
            lab[j - 1] = 'CLOSER'; ci = j - 1; t['head'] = MCL[s[ci]]
            midpos = [k for k in range(i, ci)]
            adjpos = ci - 1 if ci - 1 >= i else None
        else:
            t['head'] = 'none'; midpos = list(range(i, j)); adjpos = None
        t['adj'] = s[adjpos] if adjpos is not None else None
        t['mid'] = [s[k] for k in midpos if k != adjpos]
        t['allmid'] = [s[k] for k in midpos]
        t['lab'] = lab
    return T

def w2m_map():
    """Wells sign -> single M sign where the bridge (+ proposals) is unambiguous"""
    B = json.load(open(ROOT + 'data/derived/bridge_extended.json'))
    P = json.load(open(DARK + 'bridge_proposals.json'))['proposals']
    m = {}; src = {}
    for w, v in B.items():
        if len(v) == 1: m[int(w)] = v[0]; src[int(w)] = 'bridge'
    for r in P:
        if r['W'] not in m: m[r['W']] = r['M']; src[r['W']] = 'proposal'
    return m, src

# ------------------------------------------------------------------ controls
UR3_TITLES = {'dub-sar', 'arad2', 'arad2-zu', 'arad', 'ensi2', 'ugula', 'nu-banda3', 'gudu4', 'dam-gar3', 'lunga', 'sipa', 'szabra',
              'sagi', 'kuruszda', 'aga3-us2', 'muhaldim', 'nu-banda3-gu4', 'sanga', 'szagina', 'sukkal', 'ra2-gaba', 'nar', 'simug',
              'aszgab', 'nagar', 'ad-kup4', 'szu-i', 'ma2-lah5', 'engar', 'ma2-gal', 'gal5-la2-gal', 'kas4', 'sza13-dub-ba', 'lu2-{d}inanna'}
def load_ur3():
    """Ur III seal legends (one per distinct legend): elements = syllables of the owner name (word 0),
    head = first profession word (title); 'none' if the legend names no profession"""
    T = []
    for l in open(DARK + 'loop32_corpora/ur3_words.jsonl'):
        s = json.loads(l)['seq']
        if len(s) < 2 or s[0] in UR3_TITLES or s[0] == 'dumu' or 'x' in s[0]: continue
        name = [x for x in s[0].replace('{', '-').replace('}', '-').split('-') if x]
        heads = [w for w in s[1:] if w in UR3_TITLES]
        T.append(dict(id=str(len(T)), site='ur3', ot='legend', seq=s, head=heads[0] if heads else 'none', adj=None,
                      mid=name, allmid=name))
    return T

def load_pe():
    T = []
    for l in open(DARK + 'loop32_corpora/proto_elamite.jsonl'):
        s = json.loads(l)['seq']
        if len(s) < 2: continue
        if s[-1].startswith('N') or s[-1][0].isdigit():
            head = s[-1]; mid = s[:-1]
        else: head = 'none'; mid = s
        T.append(dict(id=str(len(T)), site='pe', ot='entry', seq=s, head=head, adj=None, mid=mid, allmid=mid))
    return T

# ------------------------------------------------------------------ cycle 1
def cycle1(T, out, nrep, label='Wells', chain_null=True):
    out.append(f'\n== {label}: {len(T)} collapsed texts; heads {dict(collections.Counter(t["head"] for t in T).most_common())}')
    out.append(f'   sites {dict(collections.Counter(t["site"] for t in T).most_common())}; types {dict(collections.Counter(t["ot"] for t in T).most_common())}')
    R = {}
    S_len_type = ('len x type', lambda t: (lenbin(t), t['ot']))
    S_len_type_site = ('len x type x site', lambda t: (lenbin(t), t['ot'], t['site']))
    S_len_type_head = ('len x type x head', lambda t: (lenbin(t), t['ot'], t['head']))
    S_chain = ('S366 chain: len x type x adjacent sign', lambda t: (lenbin(t), t['ot'], t['adj']))
    S_len_site = ('len x site', lambda t: (lenbin(t), t['site']))
    for field, fname in (('mid', 'NON-ADJACENT middle elements'), ('adjq', 'ADJACENT qualifier (sign before the closer)'), ('allmid', 'all middle tokens')):
        if field == 'adjq':
            for t in T: t['adjq'] = [t['adj']] if t['adj'] is not None else []
        out.append(f'\n-- {fname}')
        heads_strata = [S_len_type, S_len_type_site] + ([S_chain] if chain_null and field == 'mid' else [])
        R[(field, 'head')] = analyse(T, field, 'head', heads_strata, nrep, out, f'{field} x HEAD')
        if len(set(t['site'] for t in T)) > 1:
            R[(field, 'site')] = analyse(T, field, 'site', [S_len_type, S_len_type_head], nrep, out, f'{field} x SITE')
        if len(set(t['ot'] for t in T)) > 1:
            R[(field, 'ot')] = analyse(T, field, 'ot', [S_len_site, S_len_type_head if False else ('len x site x head', lambda t: (lenbin(t), t['site'], t['head']))], nrep, out, f'{field} x TYPE')
    return R

def main1():
    out = [f'# S-DARK-45 cycle 1, level {LV}, nperm {NP}: who owns the middle inventory? element x head / site / type']
    T = load_wells(LV); n0 = len(T); T = collapse(T)
    out.append(f'Wells complete texts >= 2 signs: {n0}; collapsed to one per distinct text per site x type: {len(T)}')
    parse = make_parser(learn_qual([t['seq'] for t in T]))
    parse_all(T, parse, CL)
    R = {}
    R['all'] = cycle1(T, out, NP, 'Wells all sites')
    TH = [t for t in T if t['head'] != 'none']
    R['closed'] = cycle1(TH, out, NP, 'Wells, texts WITH a closer only')
    TS = [t for t in T if t['ot'] == 'seal' and t['head'] != 'none']
    R['seals_closed'] = cycle1(TS, out, NP, 'Wells seals with a closer')
    TB = [t for t in T if t['site'] in BIG]
    R['big'] = cycle1(TB, out, NP, 'Mohenjo-daro + Harappa')
    # list the loyal non-adjacent elements
    out.append('\n== loyal NON-ADJACENT elements (>= 5 tokens, >= 80% under one label), all sites, texts with a closer')
    for labf in ('head', 'site', 'ot'):
        L = loyal_elements(TH if labf == 'head' else T, 'mid', labf)
        items = sorted(L.items(), key=lambda kv: -kv[1][2])
        out.append(f'  {labf}: {len(L)} loyal elements: ' + ', '.join(f'W{e}->{l} {k}/{n}' for e, (l, k, n) in items[:60]))
    json.dump({k: {'%s x %s' % kk: v for kk, v in d.items()} for k, d in R.items()}, open(DARK + f'loop45_c1_{LV}.json', 'w'), indent=1, default=str)
    open(DARK + f'loop45_c1_{LV}.txt', 'w').write('\n'.join(out) + '\n')
    print('\n'.join(out))

# ------------------------------------------------------------------ cycle 2: block models
def matrix(T, field, labfield, mintok=MINTOK, minlab=20):
    c = collections.defaultdict(collections.Counter); lc = collections.Counter()
    for t in T:
        lc[t[labfield]] += 1
        for e in t[field]: c[e][t[labfield]] += 1
    labs = [l for l, n in lc.most_common() if n >= minlab]
    elems = [e for e, cnt in c.items() if sum(cnt[l] for l in labs) >= mintok]
    M = np.array([[c[e][l] for l in labs] for e in elems], dtype=float)
    return M, elems, labs

def cocluster(M, k):
    from sklearn.cluster import SpectralCoclustering
    with np.errstate(all='ignore'):
        model = SpectralCoclustering(n_clusters=k, random_state=0, n_init=5)
        model.fit(M + 1e-9)
    return model.row_labels_, model.column_labels_

def block_stats(M, rl, cl):
    """diagonal concentration = share of tokens inside blocks where row cluster == column cluster;
    row purity = share of each element's tokens under the columns of its own block; column purity likewise"""
    k = max(rl.max(), cl.max()) + 1
    n = M.sum(); diag = 0.0
    for b in range(k):
        diag += M[np.ix_(rl == b, cl == b)].sum()
    rowpur = np.array([M[i, cl == rl[i]].sum() / M[i].sum() for i in range(M.shape[0]) if M[i].sum() > 0])
    colpur = np.array([M[rl == cl[j], j].sum() / M[:, j].sum() for j in range(M.shape[1]) if M[:, j].sum() > 0])
    # expected under independence
    exp = sum((M.sum(1)[rl == b].sum() * M.sum(0)[cl == b].sum()) for b in range(k)) / n
    return diag / n, exp / n, float(rowpur.mean()), float(colpur.mean())

def best_k(M, kmax):
    """pick k by the largest gain of diagonal concentration over its independence expectation"""
    best = None
    for k in range(2, min(kmax, M.shape[1]) + 1):
        try: rl, cl = cocluster(M, k)
        except Exception: continue
        d, e, rp, cp = block_stats(M, rl, cl)
        if best is None or d - e > best[1]: best = (k, d - e, d, e, rp, cp, rl, cl)
    return best

def blocks_report(T, field, labfield, out, nrep, tag, kmax=8, strata=None):
    M, elems, labs = matrix(T, field, labfield)
    if M.shape[0] < 10 or M.shape[1] < 2:
        out.append(f'  {tag}: matrix {M.shape}, too small'); return None
    b = best_k(M, kmax)
    if b is None: out.append(f'  {tag}: co-clustering failed'); return None
    k, gain, d, e, rp, cp, rl, cl = b
    out.append(f'  {tag}: matrix {M.shape[0]} elements x {M.shape[1]} labels ({int(M.sum())} tokens); best k = {k}; diagonal concentration {d:.3f} vs independence {e:.3f} (gain {gain:+.3f}); element purity {rp:.3f}; label purity {cp:.3f}')
    # block composition
    for bi in range(k):
        L = [labs[j] for j in range(len(labs)) if cl[j] == bi]
        E = [elems[i] for i in range(len(elems)) if rl[i] == bi]
        rows = M[rl == bi]
        share = rows.sum() / M.sum()
        out.append(f'     block {bi}: labels {L}; {len(E)} elements ({100*share:.1f}% of tokens), e.g. {E[:12]}')
    # null: labels permuted among texts within strata, re-cluster at the same k
    if strata is None: strata = lambda t: (lenbin(t), t['ot'])
    ng, nrp = [], []
    for lab in permute_labels(T, labfield, strata, nrep):
        c = collections.defaultdict(collections.Counter)
        for i, t in enumerate(T):
            for e in t[field]: c[e][lab[i]] += 1
        Mp = np.array([[c[e][l] for l in labs] for e in elems], dtype=float)
        try:
            rlp, clp = cocluster(Mp, k); dp, ep, rpp, cpp = block_stats(Mp, rlp, clp)
        except Exception: continue
        ng.append(dp - ep); nrp.append(rpp)
    if ng:
        out.append(f'     null ({len(ng)}x, labels permuted within len x type, same k): gain {np.mean(ng):+.3f} +- {np.std(ng):.3f} (P = {pval(gain, ng):.3f}); element purity {np.mean(nrp):.3f} (P = {pval(rp, nrp):.3f})')
    return dict(tag=tag, shape=M.shape, k=k, gain=gain, diag=d, indep=e, elem_purity=rp, label_purity=cp,
                null_gain=float(np.mean(ng)) if ng else None, null_gain_p=pval(gain, ng) if ng else None,
                null_purity=float(np.mean(nrp)) if ng else None, null_purity_p=pval(rp, nrp) if ng else None,
                blocks=[dict(labels=[labs[j] for j in range(len(labs)) if cl[j] == bi], n_elem=int((rl == bi).sum())) for bi in range(k)])

def alignment(T, out, nrep, tag):
    """do element blocks found on heads coincide with element blocks found on sites? adjusted mutual information of the two row partitions"""
    from sklearn.metrics import adjusted_mutual_info_score as ami
    Mh, eh, lh = matrix(T, 'mid', 'head'); Ms, es, ls = matrix(T, 'mid', 'site')
    common = [e for e in eh if e in set(es)]
    if len(common) < 10 or Mh.shape[1] < 2 or Ms.shape[1] < 2: return
    bh = best_k(Mh, 8); bs = best_k(Ms, 8)
    if bh is None or bs is None: return
    ih = {e: i for i, e in enumerate(eh)}; i_s = {e: i for i, e in enumerate(es)}
    a = ami([bh[6][ih[e]] for e in common], [bs[6][i_s[e]] for e in common])
    out.append(f'  {tag}: head-blocks vs site-blocks over {len(common)} shared elements: adjusted MI {a:.3f} (0 = unrelated partitions)')

def main2():
    out = [f'# S-DARK-45 cycle 2, level {LV}, nperm {NP}: block structure of element x head and element x site']
    T = collapse(load_wells(LV)); parse = make_parser(learn_qual([t['seq'] for t in T])); parse_all(T, parse, CL)
    TH = [t for t in T if t['head'] != 'none']
    R = {}
    out.append(f'\n== Wells, texts with a closer ({len(TH)}), non-adjacent middle elements')
    R['head'] = blocks_report(TH, 'mid', 'head', out, NP, 'mid x HEAD')
    R['head_adj'] = blocks_report([dict(t, adjq=[t['adj']]) for t in TH if t['adj'] is not None], 'adjq', 'head', out, NP, 'ADJACENT qualifier x HEAD (positive control, S303)')
    R['site'] = blocks_report(T, 'mid', 'site', out, NP, 'mid x SITE (all texts)')
    R['site_closed'] = blocks_report(TH, 'mid', 'site', out, NP, 'mid x SITE (texts with a closer)')
    R['type'] = blocks_report(T, 'mid', 'ot', out, NP, 'mid x TYPE (all texts)', strata=lambda t: (lenbin(t), t['site']))
    R['head_big'] = blocks_report([t for t in TH if t['site'] in BIG], 'mid', 'head', out, NP, 'mid x HEAD, Mohenjo-daro + Harappa')
    R['site_seals'] = blocks_report([t for t in T if t['ot'] == 'seal'], 'mid', 'site', out, NP, 'mid x SITE, seals only')
    alignment(TH, out, NP, 'Wells')
    out.append('\n== controls')
    U = load_ur3()
    out.append(f'  Ur III legends: {len(U)} distinct legends with an owner name; heads {dict(collections.Counter(t["head"] for t in U).most_common(12))}')
    UH = [t for t in U if t['head'] != 'none']
    R['ur3'] = blocks_report(UH, 'mid', 'head', out, NP, 'Ur III name syllables x PROFESSION (legends with a profession)', strata=lambda t: (lenbin(t),))
    R['ur3_all'] = blocks_report(U, 'mid', 'head', out, NP, 'Ur III name syllables x PROFESSION (all, incl. none)', strata=lambda t: (lenbin(t),))
    # also Ur III whole-name words x profession (does a NAME pick a profession?)
    for t in U: t['word'] = [t['seq'][0]]
    R['ur3_word'] = blocks_report(UH, 'word', 'head', out, NP, 'Ur III owner NAME (whole word) x PROFESSION', strata=lambda t: (lenbin(t),))
    P = load_pe(); PH = [t for t in P if t['head'] != 'none']
    out.append(f'  Proto-Elamite entries: {len(P)}; with a final numeral class {len(PH)}; heads {dict(collections.Counter(t["head"] for t in PH).most_common(8))}')
    R['pe'] = blocks_report(PH, 'mid', 'head', out, NP, 'Proto-Elamite signs x FINAL numeral class', strata=lambda t: (lenbin(t),))
    # MI/loyalty for the controls with the same statistics as cycle 1
    out.append('\n== cycle-1 statistics on the controls (non-adjacent = whole name / all non-final signs)')
    analyse(UH, 'mid', 'head', [('len', lambda t: (lenbin(t),))], NP, out, 'Ur III syllables x PROFESSION')
    analyse(UH, 'word', 'head', [('len', lambda t: (lenbin(t),))], NP, out, 'Ur III NAME x PROFESSION')
    analyse(PH, 'mid', 'head', [('len', lambda t: (lenbin(t),))], NP, out, 'Proto-Elamite signs x FINAL class')
    json.dump(R, open(DARK + f'loop45_c2_{LV}.json', 'w'), indent=1, default=str)
    open(DARK + f'loop45_c2_{LV}.txt', 'w').write('\n'.join(out) + '\n')
    print('\n'.join(out))

# ------------------------------------------------------------------ cycle 3: replication + prediction
def main3():
    out = [f'# S-DARK-45 cycle 3, level {LV}, nperm {NP}: held-out sites, IM77, and the prediction on the 324 new texts']
    T = collapse(load_wells(LV)); parse = make_parser(learn_qual([t['seq'] for t in T])); parse_all(T, parse, CL)
    TB = [t for t in T if t['site'] in BIG and t['head'] != 'none']
    TO = [t for t in T if t['site'] not in BIG]
    out.append(f'\n== held-out Wells sites (not Mohenjo-daro / Harappa): {len(TO)} collapsed texts')
    cycle1(TO, out, NP, 'held-out sites', chain_null=True)
    cycle1([t for t in TO if t['head'] != 'none'], out, NP, 'held-out sites, texts with a closer')
    # loyal elements fitted on MD+H, tested on held-out
    LH = loyal_elements(TB, 'mid', 'head')
    out.append(f'\n== head-loyal non-adjacent elements fitted on Mohenjo-daro + Harappa (>= 5 tokens, >= 80% one head): {len(LH)}')
    out.append('   ' + ', '.join(f'W{e}->{l} {k}/{n}' for e, (l, k, n) in sorted(LH.items(), key=lambda kv: -kv[1][2])))
    def test_loyal(L, TT, name, labf='head', wmap=None):
        hits = tot = 0; detail = collections.Counter(); heads = collections.Counter(t[labf] for t in TT if t[labf] != 'none')
        nh = sum(heads.values()); exp = 0.0
        for t in TT:
            if t[labf] == 'none': continue
            for e in t['mid']:
                key = wmap.get(e) if wmap else e
                if key in L:
                    tot += 1; ok = (L[key][0] == t[labf]); hits += ok; detail[(key, L[key][0], t[labf])] += 1
                    exp += heads[L[key][0]] / nh
        ub = 3 / tot if tot else float('nan')
        out.append(f'   {name}: loyal-element tokens in texts with a {labf}: {tot}; same {labf} as fitted {hits} ({hits/tot:.2f} if tot else -) vs base-rate expectation {exp:.1f} ({exp/tot:.2f}); 0-of-n bound < {ub:.2f}' if tot else f'   {name}: no loyal-element tokens')
        if tot:
            # binomial-ish permutation: labels permuted among the texts
            null = []
            for lab in permute_labels(TT, labf, lambda t: (lenbin(t), t['ot']), NP):
                h = 0
                for i, t in enumerate(TT):
                    if lab[i] == 'none': continue
                    for e in t['mid']:
                        key = wmap.get(e) if wmap else e
                        if key in L and L[key][0] == lab[i]: h += 1
                null.append(h)
            out.append(f'      null ({NP}x, {labf} permuted within len x type): {np.mean(null):.1f} +- {np.std(null):.1f}, P = {pval(hits, null):.3f}')
            out.append('      cases: ' + '; '.join(f'{k[0]} fitted {k[1]} seen {k[2]} x{v}' for k, v in detail.most_common(40)))
        return hits, tot
    test_loyal(LH, [t for t in TO if t['head'] != 'none'], 'held-out Wells sites')
    # IM77
    I = parse_im77(load_im77()); I = collapse(I)
    out.append(f'\n== IM77 (M space; closers {sorted(MCL)}, openers {sorted(MOPEN)}): {len(I)} collapsed texts >= 2 signs')
    IH = [t for t in I if t['head'] != 'none']
    cycle1(IH, out, NP, 'IM77 texts with a closer')
    cycle1(I, out, NP, 'IM77 all texts', chain_null=False)
    # map Wells loyal elements to M
    wm, src = w2m_map()
    LHm = {}
    for e, v in LH.items():
        if e in wm: LHm[wm[e]] = v + (e, src[e])
    out.append(f'\n== Wells head-loyal elements mapped to M: {len(LHm)} of {len(LH)} (' + ', '.join(f'W{v[3]}=M{m} ({v[4]}) -> {v[0]}' for m, v in LHm.items()) + ')')
    newkeys = set(tuple(x) for x in json.load(open(DARK + 'loop27_sets.json'))['new'])
    IN = [t for t in I if t['key'] in newkeys]
    out.append(f'   324 IM77-only new texts present after collapse and >= 2 signs: {len(IN)}; with a closer {sum(t["head"] != "none" for t in IN)}')
    test_loyal(LHm, [t for t in IN if t['head'] != 'none'], 'IM77 NEW texts (prediction)')
    test_loyal(LHm, IH, 'IM77 all texts with a closer (70% overlap with Wells, transcription-robust only)')
    # with the IM77-loyal set from IM77 non-new texts, tested on new texts (within-corpus, truly held out)
    I_old = [t for t in IH if t['key'] not in newkeys]
    LI = loyal_elements(I_old, 'mid', 'head')
    out.append(f'\n== loyal elements fitted on IM77 non-new texts ({len(I_old)}): {len(LI)}: ' + ', '.join(f'M{e}->{l} {k}/{n}' for e, (l, k, n) in sorted(LI.items(), key=lambda kv: -kv[1][2])))
    test_loyal(LI, [t for t in IN if t['head'] != 'none'], 'IM77 NEW texts, IM77-fitted loyal set')
    # site-loyal elements -> do they appear at their site in the new texts?
    LS = loyal_elements([t for t in T if t['site'] in BIG], 'mid', 'site')
    LSm = {wm[e]: v for e, v in LS.items() if e in wm}
    out.append(f'\n== site-loyal elements (MD vs Harappa, Wells): {len(LS)}, mapped {len(LSm)}: ' + ', '.join(f'M{m}->{v[0]} {v[1]}/{v[2]}' for m, v in LSm.items()))
    test_loyal(LSm, [t for t in IN if t['site'] in BIG], 'IM77 NEW texts at MD/Harappa', labf='site')
    test_loyal(LS, [t for t in TO], 'held-out sites cannot test MD/H site loyalty (expected all misses, listed for completeness)', labf='site')
    open(DARK + f'loop45_c3_{LV}.txt', 'w').write('\n'.join(out) + '\n')
    print('\n'.join(out))

if __name__ == '__main__':
    {1: main1, 2: main2, 3: main3}[CY]()
