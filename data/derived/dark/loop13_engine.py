"""Loop 13 (arrows in the dark): TRADE-BALANCE COMPLEMENTARITY between a site's seals (what it issues) and its clay
sealings (TAG*, what arrives). Cycles:
  1  vocabulary complementarity seals vs sealings per site (null: sealing site labels permuted, stratified by length class)
  2  counted items / commodity classes on sealings vs seals per site (null: seal/sealing label permuted within site)
  3  site x site flow matrix from exact + edit-distance-1 seal->sealing matches; reciprocity / hierarchy vs degree-
     preserving rewiring and vs a seal-site-label permutation null
  4  pots: pot-label vocabulary at a site vs that site's sealings and seals; Lothal fastening type x sign class
Usage: python3 loop13_engine.py <cycle> [nperm]
"""
import json, collections, random, sys, math
import numpy as np

C = json.load(open('/home/user/Indus-/data/derived/merged-corpus-canonical.json'))
NUM = {1, 3, 4, 5, 16, 17, 18, 31, 32, 33, 34}
TREE = {390, 405, 407}
BRACKET = {900}
FISH = {220, 240, 235, 233, 231, 226}
JAR = {740}
ARROW = {520}
LEVELS = ['seq_raw', 'seq_strong', 'seq_all']
rnd = random.Random(13)
np.random.seed(13)

def otype(r):
    t = r['type']
    if t.startswith('TAG'): return 'TAG'
    if t.startswith('SEAL'): return 'SEAL'
    if t.startswith('POT'): return 'POT'
    if t.startswith('TAB'): return 'TAB'
    return 'OTHER'

def records(level, types, minlen=1, dedup=False):
    seen = set(); out = []
    for r in C:
        s = r.get(level) or []
        if len(s) < minlen or otype(r) not in types: continue
        key = (r['site'], otype(r), tuple(s))
        if dedup and key in seen: continue
        seen.add(key)
        out.append((r['site'], otype(r), tuple(s), r['cisi'], r['type']))
    return out

def pstr(obs, null, two=True):
    null = np.asarray(null)
    lo = (np.sum(null <= obs) + 1) / (len(null) + 1)
    hi = (np.sum(null >= obs) + 1) / (len(null) + 1)
    return lo, hi

def cosine(a, b):
    ka = set(a) | set(b)
    if not a or not b: return float('nan')
    va = np.array([a.get(k, 0) for k in ka], float); vb = np.array([b.get(k, 0) for k in ka], float)
    return float(va @ vb / (np.linalg.norm(va) * np.linalg.norm(vb)))

def jacc(a, b):
    A, B = set(a), set(b)
    return len(A & B) / len(A | B) if A | B else float('nan')

def lenclass(s):
    return min(len(s), 3)

# ---------------------------------------------------------------- cycle 1
def cycle1(nperm, dedup, level, minseal=5, mintag=5, out=sys.stdout):
    R = records(level, {'SEAL', 'TAG'}, dedup=dedup)
    seals = [r for r in R if r[1] == 'SEAL']; tags = [r for r in R if r[1] == 'TAG']
    sites = sorted({s for s, *_ in seals} & {s for s, *_ in tags})
    sites = [s for s in sites if sum(1 for r in seals if r[0] == s) >= minseal and sum(1 for r in tags if r[0] == s) >= mintag]
    sealvoc = {s: collections.Counter(x for r in seals if r[0] == s for x in r[2]) for s in sites}
    sealclos = {s: collections.Counter(r[2][-1] for r in seals if r[0] == s) for s in sites}
    def stats(tag_sites):
        cos = {}; jac = {}; clo = {}
        for s in sites:
            tv = collections.Counter(x for r, st in zip(tags, tag_sites) if st == s for x in r[2])
            tc = collections.Counter(r[2][-1] for r, st in zip(tags, tag_sites) if st == s)
            cos[s] = cosine(sealvoc[s], tv); jac[s] = jacc(sealvoc[s], tv); clo[s] = cosine(sealclos[s], tc)
        return cos, jac, clo
    obs_sites = [r[0] for r in tags]
    ocos, ojac, oclo = stats(obs_sites)
    # null: permute sealing site labels within length class (keeps per-site counts per length class)
    strata = collections.defaultdict(list)
    for i, r in enumerate(tags): strata[lenclass(r[2])].append(i)
    nulls = {k: {s: [] for s in sites} for k in ('cos', 'jac', 'clo')}
    for _ in range(nperm):
        perm = list(obs_sites)
        for idx in strata.values():
            vals = [obs_sites[i] for i in idx]; rnd.shuffle(vals)
            for i, v in zip(idx, vals): perm[i] = v
        c, j, cl = stats(perm)
        for s in sites:
            nulls['cos'][s].append(c[s]); nulls['jac'][s].append(j[s]); nulls['clo'][s].append(cl[s])
    print(f'\n== cycle 1  level={level} dedup={dedup}  seals={len(seals)} sealings={len(tags)}  nperm={nperm}', file=out)
    print(f'{"site":14s} {"nSeal":>5s} {"nTag":>4s} | {"cos":>5s} {"null":>5s} {"p_lo":>6s} {"p_hi":>6s} | {"jacc":>5s} {"null":>5s} {"p_lo":>6s} | {"closer-cos":>9s} {"null":>5s} {"p_lo":>6s} {"p_hi":>6s}', file=out)
    zsum = {'cos': 0, 'jac': 0, 'clo': 0}; zcount = 0
    rows = {}
    for s in sites:
        ns = sum(1 for r in seals if r[0] == s); nt = sum(1 for r in tags if r[0] == s)
        line = f'{s:14s} {ns:5d} {nt:4d}'
        for k, o in (('cos', ocos), ('jac', ojac), ('clo', oclo)):
            nl = np.array(nulls[k][s]); lo, hi = pstr(o[s], nl)
            sd = nl.std() or 1e-9; zsum[k] += (o[s] - nl.mean()) / sd
            if k == 'jac': line += f' | {o[s]:5.3f} {nl.mean():5.3f} {lo:6.3f}'
            else: line += f' | {o[s]:5.3f} {nl.mean():5.3f} {lo:6.3f} {hi:6.3f}'
            rows[(s, k)] = (o[s], nl.mean(), lo, hi)
        zcount += 1
        print(line, file=out)
    print('  summed z (observed - null)/sd over sites: ' + ', '.join(f'{k}={v:+.2f}' for k, v in zsum.items()) +
          '   (negative = sealing vocabulary LESS like own seals than other sites\' sealings = complementarity)', file=out)
    return rows, zsum

# ---------------------------------------------------------------- cycle 2
def feats(s):
    s = list(s)
    f = {}
    f['has_tree'] = any(x in TREE for x in s)
    f['has_bracket'] = any(x in BRACKET for x in s)
    f['has_fish'] = any(x in FISH for x in s)
    f['N_tree'] = any(s[i] in NUM and s[i + 1] in TREE for i in range(len(s) - 1))
    f['N_bracket'] = any(s[i] in NUM and s[i + 1] in BRACKET for i in range(len(s) - 1))
    f['N_fish'] = any(s[i] in NUM and s[i + 1] in FISH for i in range(len(s) - 1))
    f['any_numeral'] = any(x in NUM for x in s)
    f['jar_final'] = s[-1] in JAR
    f['arrow_final'] = s[-1] in ARROW
    f['tree_final'] = s[-1] in TREE
    f['opener_first'] = s[0] in {817, 861, 820, 920, 692}
    f['len'] = len(s)
    return f
FEATS = ['has_tree', 'has_bracket', 'has_fish', 'N_tree', 'N_bracket', 'N_fish', 'any_numeral', 'jar_final', 'arrow_final', 'tree_final', 'opener_first', 'len']

def cycle2(nperm, dedup, level, sites_main=('Lothal', 'Kalibangan', 'Harappa'), out=sys.stdout, minn=5):
    R = records(level, {'SEAL', 'TAG'}, dedup=dedup)
    allsites = sorted({r[0] for r in R})
    sites = [s for s in allsites if sum(1 for r in R if r[0] == s and r[1] == 'SEAL') >= minn and sum(1 for r in R if r[0] == s and r[1] == 'TAG') >= minn]
    print(f'\n== cycle 2  level={level} dedup={dedup}  sites={sites}  nperm={nperm}', file=out)
    F = {r: feats(r[2]) for r in R}
    res = {}
    for s in sites:
        rs = [r for r in R if r[0] == s]; lab = [r[1] for r in rs]
        nS = lab.count('SEAL'); nT = lab.count('TAG')
        def diff(labels):
            d = {}
            for f in FEATS:
                vt = [F[r][f] for r, l in zip(rs, labels) if l == 'TAG']; vs = [F[r][f] for r, l in zip(rs, labels) if l == 'SEAL']
                d[f] = np.mean(vt) - np.mean(vs)
            return d
        od = diff(lab)
        nul = {f: [] for f in FEATS}
        for _ in range(nperm):
            p = list(lab); rnd.shuffle(p)
            dd = diff(p)
            for f in FEATS: nul[f].append(dd[f])
        for f in FEATS:
            lo, hi = pstr(od[f], nul[f]); res[(s, f)] = (od[f], 2 * min(lo, hi), nS, nT)
    # print table: rows = feature, cols = sites, cell = diff (TAG - SEAL) with p
    hdr = f'{"feature (TAG mean - SEAL mean)":30s}' + ''.join(f'{s[:12]:>22s}' for s in sites)
    print(hdr, file=out)
    ntests = len(sites) * len(FEATS)
    consistent = []
    for f in FEATS:
        line = f'{f:30s}'
        signs = []
        for s in sites:
            d, p, nS, nT = res[(s, f)]
            star = '*' if p * ntests < 0.05 else ('+' if p < 0.05 else ' ')
            line += f'{d:+9.3f} p={p:6.3f}{star}   '
            signs.append(np.sign(d))
        main = [np.sign(res[(s, f)][0]) for s in sites_main if s in sites]
        same = len(main) == len(sites_main) and len(set(main)) == 1 and main[0] != 0
        others = [np.sign(res[(s, f)][0]) for s in sites if s not in sites_main]
        held = 'held-out same' if same and others and all(o == main[0] for o in others) else ('held-out mixed' if same else '')
        if same: consistent.append((f, main[0], held))
        print(line + ('  <<< same sign at L,K,H ' + held if same else ''), file=out)
    print(f'  sites n (seal,tag): ' + ', '.join(f'{s}=({res[(s, FEATS[0])][2]},{res[(s, FEATS[0])][3]})' for s in sites), file=out)
    print(f'  * = Bonferroni over {ntests} tests;  + = raw p<0.05', file=out)
    return res, consistent

# ---------------------------------------------------------------- cycle 3
def lev1(a, b):
    """True if Levenshtein distance between tuples a,b is <= 1."""
    if a == b: return True
    la, lb = len(a), len(b)
    if abs(la - lb) > 1: return False
    if la == lb:
        return sum(x != y for x, y in zip(a, b)) == 1
    if la > lb: a, b = b, a; la, lb = lb, la
    i = j = 0; skipped = False
    while i < la and j < lb:
        if a[i] == b[j]: i += 1; j += 1
        elif skipped: return False
        else: skipped = True; j += 1
    return True

def flow_matrix(tags, seals, seal_sites, near=True, minlen=3):
    """Return Counter[(seal_site, tag_site)] weights, one sealing distributes weight 1 among its matched seal sites."""
    exact = collections.defaultdict(set)
    for (st, _, s, *_), ss in zip(seals, seal_sites):
        if len(s) >= minlen: exact[s].add(ss)
    bylen = collections.defaultdict(list)
    for (st, _, s, *_), ss in zip(seals, seal_sites):
        if len(s) >= minlen: bylen[len(s)].append((s, ss))
    M = collections.Counter(); nmatch = 0; nexact = 0
    for (ts, _, t, *_) in tags:
        if len(t) < minlen: continue
        hits = set(exact.get(t, ()))
        if hits: nexact += 1
        if near and not hits:
            for L in (len(t) - 1, len(t), len(t) + 1):
                for s, ss in bylen.get(L, ()):
                    if lev1(t, s): hits.add(ss)
        if hits:
            nmatch += 1
            for h in hits: M[(h, ts)] += 1 / len(hits)
    return M, nmatch, nexact

def graph_stats(M, sites):
    """reciprocity (weighted), out-degree concentration (Gini of out-strength), share of cross-site flow, hub share"""
    cross = {k: v for k, v in M.items() if k[0] != k[1]}
    tot = sum(cross.values()) or 1e-9
    recip = sum(min(v, cross.get((b, a), 0)) for (a, b), v in cross.items()) / tot
    outs = np.array([sum(v for (a, b), v in cross.items() if a == s) for s in sites])
    ins = np.array([sum(v for (a, b), v in cross.items() if b == s) for s in sites])
    def gini(x):
        x = np.sort(np.asarray(x, float)); n = len(x)
        if x.sum() == 0: return float('nan')
        return float((2 * np.sum((np.arange(1, n + 1)) * x) / (n * x.sum())) - (n + 1) / n)
    hub = outs.max() / tot
    # hierarchy: correlation between out-strength and in-strength across sites (negative = sources are not sinks)
    oi = float(np.corrcoef(outs, ins)[0, 1]) if outs.std() > 0 and ins.std() > 0 else float('nan')
    return dict(total=tot, recip=recip, gini_out=gini(outs), hub_share=hub, out_in_corr=oi, self=sum(v for k, v in M.items() if k[0] == k[1]))

def rewire(M, nswap, sites):
    """degree-preserving rewiring of the weighted directed multigraph (edges as unit stubs by rounding weights*4)."""
    edges = []
    for (a, b), v in M.items():
        if a == b: continue
        edges += [(a, b)] * int(round(v * 4))
    edges = list(edges)
    for _ in range(nswap):
        i, j = rnd.randrange(len(edges)), rnd.randrange(len(edges))
        (a, b), (c, d) = edges[i], edges[j]
        if a == d or c == b: continue
        edges[i], edges[j] = (a, d), (c, b)
    Mn = collections.Counter()
    for e in edges: Mn[e] += 0.25
    return Mn

def cycle3(nperm, level, out=sys.stdout, minlen=3):
    seals = records(level, {'SEAL'}, minlen=minlen)
    tags = records(level, {'TAG'}, minlen=minlen)
    sites = sorted({r[0] for r in seals} | {r[0] for r in tags})
    print(f'\n== cycle 3  level={level}  seals(>= {minlen})={len(seals)} sealings={len(tags)}  nperm={nperm}', file=out)
    for near in (False, True):
        M, nm, ne = flow_matrix(tags, seals, [r[0] for r in seals], near=near, minlen=minlen)
        print(f'\n-- near={near}: sealings matched {nm} (exact {ne})', file=out)
        cells = sorted(M.items(), key=lambda kv: -kv[1])
        print('  flow cells (seal-site -> sealing-site : weight):', file=out)
        for (a, b), v in cells: print(f'    {a:14s} -> {b:14s} {v:5.2f}' + ('   (self)' if a == b else ''), file=out)
        gs = graph_stats(M, sites)
        print('  observed: ' + ', '.join(f'{k}={v:.3f}' for k, v in gs.items()), file=out)
        # null A: seal site labels permuted among seals of the same length (texts are pan-Indus; where would matches fall by seal-count alone?)
        bylen = collections.defaultdict(list)
        for i, r in enumerate(seals): bylen[len(r[2])].append(i)
        nullA = collections.defaultdict(list); cellA = collections.defaultdict(list)
        for _ in range(nperm):
            ss = [r[0] for r in seals]
            for idx in bylen.values():
                v = [ss[i] for i in idx]; rnd.shuffle(v)
                for i, x in zip(idx, v): ss[i] = x
            Mp, _, _ = flow_matrix(tags, seals, ss, near=near, minlen=minlen)
            g = graph_stats(Mp, sites)
            for k, v in g.items(): nullA[k].append(v)
            for key in set(M) | set(Mp): cellA[key].append(Mp.get(key, 0))
        print('  null A (seal site labels permuted within length): ' + ', '.join(
            f'{k}={np.nanmean(nullA[k]):.3f} [p_lo={pstr(gs[k], np.nan_to_num(nullA[k], nan=-9))[0]:.3f} p_hi={pstr(gs[k], np.nan_to_num(nullA[k], nan=-9))[1]:.3f}]' for k in gs), file=out)
        print('  cells vs null A (obs, null mean, p_hi):', file=out)
        for key, v in cells:
            nl = np.array(cellA[key]); lo, hi = pstr(v, nl)
            print(f'    {key[0]:14s} -> {key[1]:14s} obs={v:5.2f} null={nl.mean():5.2f} p_hi={hi:.3f} p_lo={lo:.3f}', file=out)
        # null B: degree-preserving rewiring of the observed cross-site graph
        nullB = collections.defaultdict(list)
        for _ in range(nperm):
            Mr = rewire(M, 200, sites); g = graph_stats(Mr, sites)
            for k, v in g.items(): nullB[k].append(v)
        print('  null B (degree-preserving rewiring): ' + ', '.join(
            f'{k}={np.nanmean(nullB[k]):.3f} [p_lo={pstr(gs[k], np.nan_to_num(nullB[k], nan=-9))[0]:.3f} p_hi={pstr(gs[k], np.nan_to_num(nullB[k], nan=-9))[1]:.3f}]' for k in ('recip', 'gini_out', 'hub_share', 'out_in_corr')), file=out)
        # held-out: drop Mohenjo-daro seals and Lothal bale sealings; what is left?
        seals2 = [r for r in seals if r[0] != 'Mohenjo-daro']
        M2, nm2, ne2 = flow_matrix(tags, seals2, [r[0] for r in seals2], near=near, minlen=minlen)
        print(f'  held-out (no Mohenjo-daro seals): matched {nm2} (exact {ne2}); cells: ' + '; '.join(f'{a}->{b} {v:.2f}' for (a, b), v in sorted(M2.items(), key=lambda kv: -kv[1])), file=out)

# ---------------------------------------------------------------- cycle 4
def cycle4(nperm, dedup, level, out=sys.stdout, minn=5):
    R = records(level, {'SEAL', 'TAG', 'POT'}, dedup=dedup)
    sites = sorted({r[0] for r in R})
    sites = [s for s in sites if all(sum(1 for r in R if r[0] == s and r[1] == t) >= minn for t in ('SEAL', 'TAG', 'POT'))]
    print(f'\n== cycle 4  level={level} dedup={dedup}  sites={sites}  nperm={nperm}', file=out)
    print(f'{"site":14s} {"nPot":>4s} {"nTag":>4s} {"nSeal":>5s} | {"cos(pot,tag)":>12s} {"cos(pot,seal)":>13s} {"diff":>7s} {"null":>7s} {"p_hi":>6s} {"p_lo":>6s} | {"jac(pot,tag)":>12s} {"jac(pot,seal)":>13s} {"diff":>7s} {"p_hi":>6s}', file=out)
    zs = []
    for s in sites:
        rs = [r for r in R if r[0] == s]
        pots = [r for r in rs if r[1] == 'POT']; st = [r for r in rs if r[1] in ('SEAL', 'TAG')]
        pv = collections.Counter(x for r in pots for x in r[2])
        def d(labels):
            tv = collections.Counter(x for r, l in zip(st, labels) if l == 'TAG' for x in r[2])
            sv = collections.Counter(x for r, l in zip(st, labels) if l == 'SEAL' for x in r[2])
            return cosine(pv, tv), cosine(pv, sv), jacc(pv, tv), jacc(pv, sv)
        lab = [r[1] for r in st]
        ct, cs, jt, js = d(lab); od = ct - cs; ojd = jt - js
        nul = []; nulj = []
        for _ in range(nperm):
            p = list(lab); rnd.shuffle(p); a, b, c, e = d(p); nul.append(a - b); nulj.append(c - e)
        lo, hi = pstr(od, nul); jlo, jhi = pstr(ojd, nulj)
        nul = np.array(nul); zs.append((od - nul.mean()) / (nul.std() or 1e-9))
        print(f'{s:14s} {len(pots):4d} {lab.count("TAG"):4d} {lab.count("SEAL"):5d} | {ct:12.3f} {cs:13.3f} {od:+7.3f} {nul.mean():+7.3f} {hi:6.3f} {lo:6.3f} | {jt:12.3f} {js:13.3f} {ojd:+7.3f} {jhi:6.3f}', file=out)
    print(f'  summed z over sites (pot closer to sealings than to seals, beyond the size-matched null): {sum(zs):+.2f}', file=out)
    # commodity classes on pots vs sealings vs seals per site
    print('\n  share of texts with tree / bracket / fish / numeral / jar-final, by site and object type:', file=out)
    for s in sites:
        for t in ('POT', 'TAG', 'SEAL'):
            rs = [r for r in R if r[0] == s and r[1] == t]
            fs = [feats(r[2]) for r in rs]
            print(f'    {s:14s} {t:4s} n={len(rs):4d} ' + ' '.join(f'{f}={np.mean([x[f] for x in fs]):.2f}' for f in ('has_tree', 'has_bracket', 'has_fish', 'any_numeral', 'jar_final', 'opener_first')) + f' meanlen={np.mean([x["len"] for x in fs]):.2f}', file=out)

def cycle4b(nperm, level, out=sys.stdout):
    """Lothal fastening type (Frenez & Tosi 2005) x sign class of the sealing text."""
    FT = json.load(open('/home/user/Indus-/data/derived/lothal-sealings-frenez-tosi2005.json'))['rows']
    ft = {r[0]: (r[2], r[3]) for r in FT}
    rows = []
    for r in C:
        if r['site'] == 'Lothal' and otype(r) == 'TAG' and r['cisi'] in ft and r.get(level):
            rows.append((r['cisi'], ft[r['cisi']][0], ft[r['cisi']][1], tuple(r[level])))
    print(f'\n== cycle 4b  Lothal sealings with fastening type: {len(rows)} (level {level})', file=out)
    grp = collections.defaultdict(list)
    for c, f, ctx, s in rows: grp[f].append(s)
    for f, L in sorted(grp.items(), key=lambda kv: -len(kv[1])):
        fs = [feats(s) for s in L]
        texts = collections.Counter(L)
        print(f'  {f:16s} n={len(L):3d} distinct={len(texts):3d} ' + ' '.join(f'{k}={np.mean([x[k] for x in fs]):.2f}' for k in ('has_tree', 'has_fish', 'any_numeral', 'jar_final', 'arrow_final', 'opener_first')) + f' meanlen={np.mean([x["len"] for x in fs]):.1f}', file=out)
        print('      texts: ' + '; '.join(f'{"-".join(map(str, t))} x{n}' for t, n in texts.most_common(4)), file=out)
    # permutation: mutual information between fastening type and (jar_final, has_fish, any_numeral, opener_first), texts dedup'd per type
    lab = [f for _, f, _, _ in rows]; sig = [feats(s) for *_, s in rows]
    def mi(labels, key):
        n = len(labels); j = collections.Counter(zip(labels, [x[key] for x in sig])); a = collections.Counter(labels); b = collections.Counter(x[key] for x in sig)
        return sum(v / n * math.log((v / n) / ((a[l] / n) * (b[k] / n))) for (l, k), v in j.items())
    for key in ('jar_final', 'has_fish', 'any_numeral', 'opener_first', 'has_tree'):
        o = mi(lab, key); nul = []
        for _ in range(nperm):
            p = list(lab); rnd.shuffle(p); nul.append(mi(p, key))
        print(f'  MI(fastening, {key}) = {o:.3f}  null={np.mean(nul):.3f}  p_hi={pstr(o, nul)[1]:.3f}  (note: 10 identical wooden-box sealings L-161..170 inflate this)', file=out)
    # dedup version: one row per (fastening, text)
    dd = {}
    for c, f, ctx, s in rows: dd[(f, s)] = 1
    lab2 = [f for f, s in dd]; sig2 = [feats(s) for f, s in dd]
    def mi2(labels, key):
        n = len(labels); j = collections.Counter(zip(labels, [x[key] for x in sig2])); a = collections.Counter(labels); b = collections.Counter(x[key] for x in sig2)
        return sum(v / n * math.log((v / n) / ((a[l] / n) * (b[k] / n))) for (l, k), v in j.items())
    for key in ('jar_final', 'has_fish', 'any_numeral', 'opener_first', 'has_tree'):
        o = mi2(lab2, key); nul = []
        for _ in range(nperm):
            p = list(lab2); rnd.shuffle(p); nul.append(mi2(p, key))
        print(f'  dedup (n={len(lab2)}) MI(fastening, {key}) = {o:.3f}  null={np.mean(nul):.3f}  p_hi={pstr(o, nul)[1]:.3f}', file=out)

if __name__ == '__main__':
    cyc = sys.argv[1]; nperm = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
    if cyc == '1':
        for level in LEVELS:
            for dedup in (False, True):
                cycle1(nperm, dedup, level)
    elif cyc == '2':
        for level in LEVELS:
            for dedup in (False, True):
                cycle2(nperm, dedup, level)
    elif cyc == '3':
        for level in LEVELS:
            cycle3(nperm, level)
    elif cyc == '4':
        for level in LEVELS:
            for dedup in (False, True):
                cycle4(nperm, dedup, level)
        cycle4b(nperm, 'seq_raw')
