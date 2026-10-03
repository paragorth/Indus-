"""S-DARK-84: RARE-ELEMENT LOCALITY. Are rare designation elements confined to one site, as rare name words are confined to one
town (family names, local deities), or spread across sites like the entries of one central inventory?

Statistic (identical for every corpus): texts collapsed to one per distinct text per group x type (moulded / stock copies count
once); element = Indus non-adjacent middle sign (S-DARK-64 parser; IM77: loop64 M-space parser) or name word / surname / name
element / code suffix; element types carried by 2-10 collapsed texts; CONFINED = all its texts in one group (site, city, state,
province); CONFINED-MINOR = confined to a group other than the largest. Null: group labels permuted among texts within
object type x length (min 7), NP x. O/E = observed / null mean; P one-sided (excess).
Designs: 'native' (all groups of the corpus) and 'matched' (comparators subsampled to the Indus group-size vector, R replicates,
NP2 permutations each; median O/E reported with the replicate range).

Usage: python3 tools/dark_loop84.py <cycle 1|2|3> [NP]
  1: Indus Wells (merged-corpus-canonical.json, older build) seals and tablets, seq_raw / seq_strong / seq_all; IM77 seals, tablets.
  2: comparators: Ur III name words by city, US surnames by state (FCC licensees), Latin name elements by province (EDH),
     FCC call-sign suffix by state (designed code, central, no geographic field), synthetic central code; native + matched.
  3: robustness: corpus v2 (merged-corpus-canonical.v2.json); token bands 2 / 3-5 / 6-10; Wells-IM77 agreement of confined
     elements through the bridge; MD + Harappa only vs third sites.
"""
import json, sys, random, collections, math, csv, gzip, os, glob
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/'
DK = ROOT + 'data/derived/dark/'
CY = int(sys.argv[1]); NP = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
OUT = []
def P(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); OUT.append(s)

# ------------------------------------------------------------------ core statistic
def prep(texts, lo=2, hi=10):
    """texts: list of (group, stratum, elementset). returns structure for fast permutation"""
    gnames = sorted({t[0] for t in texts}); gi = {g: i for i, g in enumerate(gnames)}
    lab = np.array([gi[t[0]] for t in texts], dtype=np.int32)
    gsize = collections.Counter(t[0] for t in texts); dom = gi[gsize.most_common(1)[0][0]]
    occ = collections.defaultdict(list)
    for i, t in enumerate(texts):
        for e in t[2]: occ[e].append(i)
    els = [e for e, ix in occ.items() if lo <= len(ix) <= hi]
    if not els: return None
    idx = np.array([occ[e] + [occ[e][0]] * (hi - len(occ[e])) for e in els], dtype=np.int64)
    ntok = np.array([len(occ[e]) for e in els])
    strata = collections.defaultdict(list)
    for i, t in enumerate(texts): strata[t[1]].append(i)
    strata = [np.array(v) for v in strata.values() if len(v) > 1]
    return dict(lab=lab, dom=dom, idx=idx, ntok=ntok, els=els, strata=strata, gnames=gnames, gsize=gsize, n=len(texts))

def conf_stats(L, idx, dom, ntok):
    M = L[idx]; c = M.max(1) == M.min(1); cm = c & (M[:, 0] != dom)
    b = {'2': ntok == 2, '3-5': (ntok >= 3) & (ntok <= 5), '6-10': ntok >= 6}
    return c.sum(), cm.sum(), {k: c[v].sum() for k, v in b.items()}, c

def run(S, nperm, rng):
    lab, idx, dom, ntok = S['lab'], S['idx'], S['dom'], S['ntok']
    oc, ocm, ob, cvec = conf_stats(lab, idx, dom, ntok)
    nc = np.empty(nperm); ncm = np.empty(nperm); nb = collections.defaultdict(list)
    for p in range(nperm):
        L = lab.copy()
        for ix in S['strata']: L[ix] = lab[rng.permutation(ix)]
        a, b, bb, _ = conf_stats(L, idx, dom, ntok)
        nc[p] = a; ncm[p] = b
        for k, v in bb.items(): nb[k].append(v)
    E = len(idx)
    r = dict(E=E, obs=int(oc), exp=nc.mean(), oe=oc / nc.mean() if nc.mean() > 0 else float('nan'),
             P=(1 + (nc >= oc).sum()) / (1 + nperm), obs_m=int(ocm), exp_m=ncm.mean(),
             oe_m=ocm / ncm.mean() if ncm.mean() > 0 else float('nan'), P_m=(1 + (ncm >= ocm).sum()) / (1 + nperm),
             bands={k: (int(ob[k]), float(np.mean(nb[k])), int(((ntok == 2) if k == '2' else ((ntok >= 3) & (ntok <= 5)) if k == '3-5' else (ntok >= 6)).sum())) for k in ob},
             cvec=cvec)
    return r

def fmt(r, label):
    s = (f'{label}: {r["E"]} element types with 2-10 texts; confined to one group {r["obs"]} ({r["obs"] / r["E"]:.3f}) vs {r["exp"]:.1f} '
         f'({r["exp"] / r["E"]:.3f}) -> O/E {r["oe"]:.2f}, P={r["P"]:.3f}; confined to a MINOR group {r["obs_m"]} vs {r["exp_m"]:.1f} -> O/E {r["oe_m"]:.2f}, P={r["P_m"]:.3f}')
    s += '; bands ' + ', '.join(f'{k}: {o}/{n} vs {e:.1f} (x{(o / e if e else float("nan")):.2f})' for k, (o, e, n) in r['bands'].items())
    return s

def collapse(objs):
    """objs: list of (group, type, seq, elements, length). one per distinct (group, type, seq)"""
    seen = set(); T = []
    for g, typ, seq, el, L in objs:
        k = (g, typ, tuple(seq))
        if k in seen or not el: continue
        seen.add(k); T.append((g, (typ, min(L, 7)), frozenset(el)))
    return T

def matched(groups, sizes, R, nperm, rng, label):
    """groups: list of lists of texts (group-name, stratum, els) ordered by preference; sizes: Indus size vector (desc).
    group k is subsampled to sizes[k] (or kept whole if smaller). R replicates."""
    res = []
    for r_ in range(R):
        T = []
        for g, n in zip(groups, sizes):
            T += [g[i] for i in rng.choice(len(g), size=min(n, len(g)), replace=False)]
        S = prep(T)
        if S is None: continue
        res.append(run(S, nperm, rng))
    if not res: return None
    q = lambda k: np.percentile([x[k] for x in res if x[k] == x[k]], [50, 10, 90])
    oe, oem, E, fr, frx = q('oe'), q('oe_m'), q('E'), q('obs'), q('exp')
    Pm = np.median([x['P'] for x in res]); Pmm = np.median([x['P_m'] for x in res])
    P(f'{label} [matched, {len(res)} reps x {nperm} perms, sizes {sizes}]: element types {E[0]:.0f}; confined {fr[0]:.0f} vs {frx[0]:.1f}; '
      f'O/E median {oe[0]:.2f} [10-90%: {oe[1]:.2f}-{oe[2]:.2f}], median P={Pm:.3f}; MINOR O/E {oem[0]:.2f} [{oem[1]:.2f}-{oem[2]:.2f}], median P={Pmm:.3f}')
    return dict(oe=oe[0], lo=oe[1], hi=oe[2], oem=oem[0], P=Pm, Pm=Pmm)

# ------------------------------------------------------------------ Indus loaders (S-DARK-64 parser)
def load_wells(level, corpus='merged-corpus-canonical.json'):
    src = open(ROOT + 'tools/dark_loop64.py').read().replace("'data/derived/merged-corpus-canonical.json'", repr('data/derived/' + corpus))
    old = sys.argv; sys.argv = ['x', '0', level, '1']
    g = {'__name__': 'x', '__file__': ROOT + 'tools/dark_loop64.py'}
    import io, contextlib
    with contextlib.redirect_stdout(io.StringIO()): exec(compile(src, 'dl64', 'exec'), g)
    sys.argv = old
    return g['OBJ'], g
def indus_texts(OBJ, oc, sites=None):
    objs = [(o['site'], o['typ'], o['seq'], o['el'], o['L']) for o in OBJ if o['oc'] == oc and o['site'] not in ('Unknown',)
            and (sites is None or o['site'] in sites)]
    return collapse(objs)
def load_im77_objs():
    _, g = load_wells('seq_raw')
    return g['load_im77']()

def sizevec(T, minn=20):
    c = collections.Counter(t[0] for t in T)
    return [n for g, n in c.most_common() if n >= minn], [g for g, n in c.most_common() if n >= minn]

# ------------------------------------------------------------------ comparator loaders (one text per distinct legend / person per group)
UR3_TITLES = {'dub-sar', 'arad2', 'arad2-zu', 'arad', 'ensi2', 'ugula', 'nu-banda3', 'gudu4', 'dam-gar3', 'lunga', 'sipa', 'szabra',
              'dumu', 'lugal', 'kiszib3', 'sukkal', 'ra2-gaba', 'szu-i', 'szagina', 'nu-banda3-gu4', 'ka-guru7', 'sanga', 'szandana',
              'i3-du8', 'muhaldim', 'szesz', 'dam', 'mu-ni', 'ki-ag2', 'ir11-zu', 'ir11', 'nin', 'szar2-ra-ab-du', 'sa12-du5'}
def load_ur3():
    T = [json.loads(l) for l in open(DK + 'loop48_corpora/ur3_words.jsonl')]
    objs = []
    for t in T:
        if t['site'] == 'uncertain': continue
        els = [w for w in t['seq'] if w not in UR3_TITLES and 'x' not in w.replace('uszurx', '')]
        objs.append((t['site'], 'legend', t['seq'], els, len(t['seq'])))
    return collapse(objs)
def load_fcc(states, rng, cap=None, unit='surname'):
    G = collections.defaultdict(list)
    for l in gzip.open(DK + 'loop68_corpora/fcc_amateur_callsign_state_surname.tsv.gz', 'rt'):
        f = l.rstrip('\n').split('\t')
        if f[0] == 'call_sign' or len(f) < 3: continue
        cs, st, sn = f[0].strip().upper(), f[1].strip().upper(), f[2].strip().upper()
        if st not in states or not sn: continue
        if unit == 'surname': G[st].append(sn)
        else:
            import re
            m = re.fullmatch(r'[A-Z]{1,2}[0-9]([A-Z]{1,3})', cs)
            if m: G[st].append(m.group(1))
    out = {}
    for st in states:
        v = G[st]
        if cap and len(v) > cap: v = [v[i] for i in rng.choice(len(v), cap, replace=False)]
        out[st] = [(st, ('x', 1), frozenset([e])) for e in v]   # person = text; one element; persons are distinct by construction
    return out
def load_edh():
    sys.path.insert(0, ROOT + 'tools')
    old = sys.argv; sys.argv = ['x', '0']
    import importlib.util
    spec = importlib.util.spec_from_file_location('dl68', ROOT + 'tools/dark_loop68.py')
    src = open(ROOT + 'tools/dark_loop68.py').read()
    # only the loader functions are needed; cut before the cycle dispatch
    g = {'__name__': 'dl68', '__file__': ROOT + 'tools/dark_loop68.py'}
    cut = src.find('\nif __name__')
    exec(compile(src if cut < 0 else src[:cut], 'dl68', 'exec'), g)
    sys.argv = old
    G = g['load_edh_persons'](100)
    out = {}
    for pv, L in G.items():
        seen = set(); T = []
        for p in L:
            if p['hd'] + '|' + '|'.join(p['els']) in seen: continue
            seen.add(p['hd'] + '|' + '|'.join(p['els']))
            T.append((pv, ('person', len(p['els'])), frozenset(p['els'])))
        out[pv] = T
    return out

# ======================================================================= cycle 1: Indus
rng = np.random.default_rng(84)
RES = {}
if CY == 1:
    P(f'== S-DARK-84 cycle 1: Indus designation elements, site locality (NP={NP}); merged-corpus-canonical.json (older build, S-DARK-23)')
    for lv in ('seq_raw', 'seq_strong', 'seq_all'):
        OBJ, _ = load_wells(lv)
        for oc in ('seal', 'tablet'):
            T = indus_texts(OBJ, oc)
            c = collections.Counter(t[0] for t in T)
            S = prep(T); r = run(S, NP, rng)
            P(fmt(r, f'Wells {lv} {oc}s, all named sites ({len(T)} collapsed texts with >= 1 element, {len(c)} sites; {dict(c.most_common(6))})'))
            sz, gs = sizevec(T)
            T2 = [t for t in T if t[0] in gs]; r2 = run(prep(T2), NP, rng)
            P(fmt(r2, f'Wells {lv} {oc}s, sites >= 20 texts {gs} sizes {sz}'))
            RES[f'W_{lv}_{oc}'] = {k: v for k, v in r.items() if k != 'cvec'}; RES[f'W_{lv}_{oc}_20'] = {k: v for k, v in r2.items() if k != 'cvec'}
            RES[f'W_{lv}_{oc}_20']['sizes'] = sz
    IM = load_im77_objs()
    for oc in ('seal', 'tablet'):
        objs = [(o['site'], o['typ'], o['seq'], set(o['ell']) - {0}, o['L']) for o in IM if o['oc'] == oc and o['site']]
        T = collapse(objs); c = collections.Counter(t[0] for t in T)
        r = run(prep(T), NP, rng)
        P(fmt(r, f'IM77 {oc}s (M numbers; seal / sealing+miniature+copper tablet), all sites ({len(T)} texts, {dict(c.most_common(6))})'))
        sz, gs = sizevec(T); r2 = run(prep([t for t in T if t[0] in gs]), NP, rng)
        P(fmt(r2, f'IM77 {oc}s, sites >= 20 texts {gs} sizes {sz}'))
        RES[f'IM_{oc}'] = {k: v for k, v in r.items() if k != 'cvec'}; RES[f'IM_{oc}_20'] = {k: v for k, v in r2.items() if k != 'cvec'}
        RES[f'IM_{oc}_20']['sizes'] = sz

# ======================================================================= cycle 2: comparators
US6 = ['OH', 'PA', 'NY', 'MI', 'IL', 'IN']            # 6 contiguous states, centres ~150-1,100 km apart (Indus sites ~130-1,000 km)
UR6 = ['Umma', 'Girsu', 'Puzriš-Dagan', 'Nippur', 'Garšana', 'Ur']
LAT6 = ['Rom', 'LaC', 'Etr', 'VeH', 'ApC', 'Aem']     # Italian regiones; replaced by the largest available if missing
if CY == 2:
    W = json.load(open(DK + 'loop84_c1.json'))
    P(f'== S-DARK-84 cycle 2: comparators through the identical statistic (native NP={NP}; matched 30 reps x 200 perms)')
    seal_sz = W['W_seq_raw_seal_20']['sizes']; tab_sz = W['W_seq_raw_tablet_20']['sizes']
    P(f'Indus seal size vector (Wells seq_raw, sites >= 20): {seal_sz}; tablets: {tab_sz}')
    # --- Ur III
    U = load_ur3(); c = collections.Counter(t[0] for t in U)
    r = run(prep(U), NP, rng); P(fmt(r, f'Ur III name words by city, native ({len(U)} distinct legends, {dict(c.most_common(8))})'))
    RES['ur3_native'] = {k: v for k, v in r.items() if k != 'cvec'}
    UG = [[t for t in U if t[0] == g] for g in UR6]
    RES['ur3_seal'] = matched(UG, seal_sz, 30, 200, rng, 'Ur III name words by city')
    RES['ur3_tab'] = matched(UG, tab_sz, 30, 200, rng, 'Ur III name words by city (tablet sizes)')
    # --- US surnames (FCC licensees) and call-sign suffixes (designed code)
    for unit in ('surname', 'suffix'):
        G = load_fcc(US6, rng, cap=3000, unit=unit)
        T = [t for st in US6 for t in G[st]]
        r = run(prep(T), NP, rng); P(fmt(r, f'US {unit}s by state, native {US6} (3,000 licensees per state)'))
        RES[f'us_{unit}_native'] = {k: v for k, v in r.items() if k != 'cvec'}
        G = load_fcc(US6, rng, cap=None, unit=unit)
        RES[f'us_{unit}_seal'] = matched([G[s] for s in US6], seal_sz, 30, 200, rng, f'US {unit}s by state')
        RES[f'us_{unit}_tab'] = matched([G[s] for s in US6], tab_sz, 30, 200, rng, f'US {unit}s by state (tablet sizes)')
    # --- Latin
    E = load_edh(); provs = sorted(E, key=lambda g: -len(E[g]))
    P('EDH provinces (persons, >= 100): ' + ', '.join(f'{p} {len(E[p])}' for p in provs[:30]))
    big = [p for p in provs if len(E[p]) >= 400]
    T = [t for p in big for t in E[p]]
    r = run(prep(T), NP, rng); P(fmt(r, f'Latin name elements by province, native ({len(big)} provinces >= 400 persons, {len(T)} persons)'))
    RES['lat_native'] = {k: v for k, v in r.items() if k != 'cvec'}
    lat = [p for p in LAT6 if p in E and len(E[p]) >= 100]
    lat += [p for p in provs if p not in lat][:6 - len(lat)]
    P(f'Latin matched groups: {lat} ({[len(E[p]) for p in lat]})')
    RES['lat_seal'] = matched([E[p] for p in lat], seal_sz, 30, 200, rng, 'Latin name elements by province (Italy)')
    RES['lat_tab'] = matched([E[p] for p in lat], tab_sz, 30, 200, rng, 'Latin name elements by province (tablet sizes)')
    # --- synthetic central code: Indus seal texts (seq_raw, sites >= 20), each element replaced by a draw from the pooled element distribution
    OBJ, _ = load_wells('seq_raw'); T = indus_texts(OBJ, 'seal'); sz, gs = sizevec(T); T = [t for t in T if t[0] in gs]
    pool = [e for t in T for e in t[2]]
    syn = []
    for _ in range(30):
        TT = [(g, s, frozenset(pool[i] for i in rng.integers(0, len(pool), len(el)))) for g, s, el in T]
        syn.append(run(prep(TT), 200, rng)['oe'])
    P(f'Synthetic central code (Indus seal lengths and site sizes, elements drawn from one pooled inventory): O/E median {np.median(syn):.2f} [10-90%: {np.percentile(syn, 10):.2f}-{np.percentile(syn, 90):.2f}]')
    RES['syn_central'] = dict(oe=float(np.median(syn)))

# ======================================================================= cycle 3: robustness
if CY == 3:
    P(f'== S-DARK-84 cycle 3: robustness (NP={NP})')
    # (a) corpus v2
    for lv in ('seq_raw', 'seq_strong', 'seq_all'):
        OBJ, _ = load_wells(lv, 'merged-corpus-canonical.v2.json')
        for oc in ('seal', 'tablet'):
            T = indus_texts(OBJ, oc); r = run(prep(T), NP, rng)
            P(fmt(r, f'v2 {lv} {oc}s all sites ({len(T)} texts)'))
            RES[f'v2_{lv}_{oc}'] = {k: v for k, v in r.items() if k != 'cvec'}
    # (b) which elements are confined, Wells seq_raw seals; Wells vs IM77 agreement through the strict bridge
    OBJ, _ = load_wells('seq_raw'); T = indus_texts(OBJ, 'seal'); S = prep(T); r = run(S, NP, rng)
    bridge = {int(k): v for k, v in json.load(open(ROOT + 'data/derived/bridge_extended.json')).items()}
    IM = load_im77_objs()
    TI = collapse([(o['site'], o['typ'], o['seq'], set(o['ell']) - {0}, o['L']) for o in IM if o['oc'] == 'seal' and o['site']])
    occI = collections.defaultdict(collections.Counter)
    for g, s, el in TI:
        for e in el: occI[e][g] += 1
    conf = [(S['els'][i], S['gnames'][S['lab'][S['idx'][i, 0]]], int(S['ntok'][i])) for i in np.where(r['cvec'])[0]]
    minor = [x for x in conf if x[1] != 'Mohenjo-daro']
    agree = tested = 0; lines = []
    for e, site, n in conf:
        if e not in bridge or len(bridge[e]) != 1: continue
        m = bridge[e][0]; tested += 1
        cI = occI.get(m, collections.Counter())
        ok = len(cI) == 1 and site in cI
        agree += ok
        if site != 'Mohenjo-daro': lines.append(f'W{e}->M{m} {site} n={n}; IM77 {dict(cI)}')
    P(f'Wells seq_raw seals: {len(conf)} confined elements ({len(minor)} at a minor site: ' + ', '.join(f'W{e} {s} {n}' for e, s, n in minor[:30]) + ')')
    P(f'  bridge-mapped (strict one-to-one) confined elements: {tested}; also confined to the same site in IM77 (all M tokens, any count): {agree}')
    for l in lines: P('   ' + l)
    # (c) MD + Harappa only (two-site test), and third sites only (MD + H removed)
    for lv in ('seq_raw', 'seq_strong', 'seq_all'):
        OBJ, _ = load_wells(lv)
        T = indus_texts(OBJ, 'seal', {'Mohenjo-daro', 'Harappa'}); r = run(prep(T), NP, rng)
        P(fmt(r, f'{lv} seals, MD + Harappa only'))
        T = [t for t in indus_texts(OBJ, 'seal') if t[0] not in ('Mohenjo-daro', 'Harappa')]
        r = run(prep(T), NP, rng); P(fmt(r, f'{lv} seals, third sites only ({len(T)} texts)'))
    # (d) Ur III matched to MD + H two-site design, US surnames likewise
    W = json.load(open(DK + 'loop84_c1.json'))
    sz2 = W['W_seq_raw_seal_20']['sizes'][:2]
    U = load_ur3(); UG = [[t for t in U if t[0] == g] for g in UR6]
    matched(UG, sz2, 30, 200, rng, 'Ur III, two-city design (MD + H sizes)')
    G = load_fcc(US6, rng, unit='surname'); matched([G[s] for s in US6], sz2, 30, 200, rng, 'US surnames, two-state design (MD + H sizes)')
    G = load_fcc(US6, rng, unit='suffix'); matched([G[s] for s in US6], sz2, 30, 200, rng, 'FCC suffix, two-state design (MD + H sizes)')

def clean(o):
    if isinstance(o, dict): return {k: clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [clean(v) for v in o]
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, (np.floating,)): return float(o)
    return o
json.dump(clean(RES), open(DK + f'loop84_c{CY}.json', 'w'), indent=1)
open(DK + f'loop84_c{CY}_log.txt', 'w').write('\n'.join(OUT) + '\n')
