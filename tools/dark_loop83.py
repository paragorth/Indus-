"""S-DARK-83: NEAR-COLLISION AVOIDANCE IN THE DESIGNATION SLOT (loop79 frontier item 1).
An issued register keeps identifiers distinguishable: designations should avoid near-identical neighbours, so pairs at
edit distance 1 (one substitution, same length) should be RARER than a generator of the same strings predicts.
Freely chosen names have no such constraint.
Statistic: e1 = number of unordered pairs of distinct same-length strings (>= 2 signs) that differ in exactly one position;
rate = e1 / number of same-length pairs. O/E = observed rate / mean null rate.
Nulls: (c) calibrated chain (S-DARK-74.5 SChain: interpolated Markov-1, lambda by 2-fold held-out likelihood), sampled at
exact length to the same number of DISTINCT strings per length; (f) frequency-matched iid strings (element unigram,
lengths kept, deduplicated). CI for an observed corpus = obs / null 97.5% .. obs / null 2.5% (null-draw interval).
Positive control: planted registers = calibrated-chain strings generated with rejection of every candidate within edit
distance 1 of an accepted string with probability p_rej (1.0 / 0.7 / 0.4), then pushed through the same test.
Indus middle = S-DARK-26/56 parser (NAME+COUNT; also NAME-only), objects deduplicated per site x object type x text,
then one per distinct middle (within each site for the within-site tests).
Usage: python3 tools/dark_loop83.py <cycle> [nnull]
  1 = pooled Indus (seq_raw / seq_strong / seq_all / IM77), mid + name, chain + freq nulls, planted controls,
      what the substituted signs are
  2 = within site (Mohenjo-daro, Harappa, held-out sites pooled with pairs counted only inside one site), seals only,
      cross-site (MD x Harappa pairs against chains fitted to each site)
  3 = the ladder: identical code on logographic given names, PE middles, Ur III and Linear B names (pooled and per
      city / palace), designed and issued codes, uniform IDs; length-matched to the Indus middle at the Indus n
  4 = robustness: by length, by substituted position (edge vs interior), numerals-only substitutions removed,
      lambda sensitivity, MLE Markov-1 null
"""
import sys, os, json, random, collections, math
ROOT = '/home/user/Indus-/'
os.chdir(ROOT); sys.path.insert(0, ROOT + 'tools')
CY = int(sys.argv[1]); NN = int(sys.argv[2]) if len(sys.argv) > 2 else 100
_argv = sys.argv; sys.argv = ['x', '0']
import dark_loop56 as L56
sys.argv = _argv
# SChain / Chain / chain_names from dark_loop74 without running its cycles
_src = open(ROOT + 'tools/dark_loop74.py').read()
_a = _src.index("BOS, EOS = '^', '$'"); _b = _src.index("NULLS = ['shuffle'")
_ns = dict(collections=collections, random=random, math=math)
exec(_src[_a:_b], _ns)
SChain, Chain, chain_names = _ns['SChain'], _ns['Chain'], _ns['chain_names']

DARK = ROOT + 'data/derived/dark/'
OUT = []
def P(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); OUT.append(s)
def save(tag=''):
    open(DARK + f'loop83_c{CY}{tag}_log.txt', 'w').write('\n'.join(OUT) + '\n')
def q(v, p):
    v = sorted(x for x in v if x == x); return v[min(len(v) - 1, int(p * len(v)))] if v else float('nan')
def mean(v):
    v = [x for x in v if x == x]; return sum(v) / len(v) if v else float('nan')
LEVELS = ['seq_raw', 'seq_strong', 'seq_all', 'im77']
FR = L56.NUM | L56.OPEN | L56.MARK | L56.MJAR | L56.SUF | set(L56.CL)

# ------------------------------------------------------------------ statistic
def e1_groups(names):
    """wildcard buckets: (len, pos, string with pos blanked) -> list of strings."""
    B = collections.defaultdict(list)
    for s in names:
        for i in range(len(s)): B[(len(s), i, s[:i] + s[i + 1:])].append(s)
    return B
def e1_pairs(names):
    names = sorted(set(n for n in names if len(n) >= 2))
    B = e1_groups(names)
    pairs = []
    for (L, i, _), v in B.items():
        for x in range(len(v)):
            for y in range(x + 1, len(v)): pairs.append((v[x], v[y], i))
    return pairs
def e1_stat(names):
    names = sorted(set(n for n in names if len(n) >= 2))
    nl = collections.Counter(len(n) for n in names)
    tot = sum(c * (c - 1) // 2 for c in nl.values())
    B = e1_groups(names)
    e1 = sum(len(v) * (len(v) - 1) // 2 for v in B.values())
    nb = set()
    for v in B.values():
        if len(v) > 1: nb.update(v)
    return dict(n=len(names), e1=e1, tot=tot, rate=e1 / tot if tot else float('nan'), nb=len(nb) / len(names) if names else float('nan'))
def e1_within(groups):
    """groups: list of name lists; pairs counted only inside a group."""
    e1 = tot = n = nbn = 0
    for g in groups:
        d = e1_stat(g); e1 += d['e1']; tot += d['tot']; n += d['n']; nbn += d['nb'] * d['n'] if d['n'] else 0
    return dict(n=n, e1=e1, tot=tot, rate=e1 / tot if tot else float('nan'), nb=nbn / n if n else float('nan'))
def e1_cross(X, Y):
    X = sorted(set(X)); Y = set(Y)
    e1 = 0
    BY = e1_groups(sorted(Y))
    for s in X:
        for i in range(len(s)):
            e1 += sum(1 for t in BY.get((len(s), i, s[:i] + s[i + 1:]), ()) if t != s)
    nx = collections.Counter(len(s) for s in X); ny = collections.Counter(len(s) for s in Y)
    tot = sum(nx[L] * ny[L] for L in nx)
    return dict(e1=e1, tot=tot, rate=e1 / tot if tot else float('nan'))

# ------------------------------------------------------------------ nulls
def freq_null(names, r):
    names = sorted(set(names)); pool = [a for n in names for a in n]
    need = collections.Counter(len(n) for n in names); out = []
    for L, c in need.items():
        got = set(); tries = 0
        while len(got) < c and tries < 60 * c:
            got.add(tuple(r.choice(pool) for _ in range(L))); tries += 1
        out += sorted(got)
    return out
def chain_null(names, r, ch=None, lens=None):
    return chain_names(sorted(set(names)), 's', r, ch, lens)[0]
def planted(names, r, prej, ch=None):
    """registry generator: calibrated-chain strings, a candidate within edit 1 of an accepted string rejected with prob prej."""
    names = sorted(set(names)); ch = ch or SChain(names, random.Random(83))
    need = collections.Counter(len(n) for n in names); out = []
    for L, c in need.items():
        acc = set(); keys = set(); tries = 0
        while len(acc) < c and tries < 200 * c:
            tries += 1; s = ch.sample(L, r)
            if s is None or s in acc: continue
            ks = [(i, s[:i] + s[i + 1:]) for i in range(L)]
            if any(k in keys for k in ks) and r.random() < prej: continue
            acc.add(s); keys.update(ks)
        out += sorted(acc)
    return out
def oe(obs, nulls, key='rate'):
    v = [d[key] for d in nulls]; mu = mean(v)
    lo, hi = q(v, .025), q(v, .975)
    p_lo = (sum(1 for x in v if x <= obs) + 1) / (len(v) + 1)
    return dict(oe=obs / mu if mu else float('nan'), ci=(obs / hi if hi else float('nan'), obs / lo if lo else float('inf')), null=mu, p_lo=p_lo)
def fmt(label, o, d):
    return (f'  {label:42s} n={d["n"]:5d} e1={d["e1"]:5d} rate={d["rate"]*1000:.3f}/1000 nb={d["nb"]:.3f} | '
            + ' | '.join(f'{k}: O/E {v["oe"]:.2f} [{v["ci"][0]:.2f},{v["ci"][1]:.2f}] P(<=)={v["p_lo"]:.3f}' for k, v in o.items()))
def test(label, names, nn=NN, seed=8300, extra=None, with_freq=True):
    names = sorted(set(n for n in names if len(n) >= 2))
    ch = SChain(names, random.Random(83)); obs = e1_stat(names)
    cn = [e1_stat(chain_null(names, random.Random(seed + b), ch)) for b in range(nn)]
    o = {f'chain(lam {ch.lam:.2f})': oe(obs['rate'], cn)}
    if with_freq:
        fn = [e1_stat(freq_null(names, random.Random(seed + 5000 + b))) for b in range(nn)]
        o['freq'] = oe(obs['rate'], fn)
    P(fmt(label, o, obs))
    return dict(obs=obs, oe={k.split('(')[0]: v for k, v in o.items()}, lam=ch.lam)
def test_within(label, groups, nn=NN, seed=8400, minfit=150):
    """groups: dict name -> list; chain refit within each group with >= minfit strings, else a chain fitted on the pooled small groups."""
    groups = {k: sorted(set(n for n in v if len(n) >= 2)) for k, v in groups.items()}
    groups = {k: v for k, v in groups.items() if len(v) >= 2}
    small = sorted(set(n for k, v in groups.items() if len(v) < minfit for n in v))
    chs = {k: SChain(v, random.Random(83)) for k, v in groups.items() if len(v) >= minfit}
    chsm = SChain(small, random.Random(83)) if small else None
    obs = e1_within(list(groups.values())); nulls = []
    for b in range(nn):
        r = random.Random(seed + b); gs = []
        for k, v in groups.items():
            ch = chs.get(k, chsm); gs.append(chain_names(v, 's', r, ch)[0])
        nulls.append(e1_within(gs))
    lam = ','.join(f'{k[:6]}:{c.lam:.1f}' for k, c in chs.items()) + (f',small:{chsm.lam:.1f}' if chsm else '')
    o = {f'within-chain({lam})': oe(obs['rate'], nulls)}
    P(fmt(label, o, obs))
    return dict(obs=obs, oe=list(o.values())[0])

# ------------------------------------------------------------------ data
def indus_objs(LV):
    return L56.load_indus(LV) if LV != 'im77' else L56.im77_objects()
def dist(objs, field='mid', sub=None):
    return sorted(set(o[field] for o in objs if len(o[field]) >= 2 and (sub is None or sub(o))))
def jl(path): return [tuple(json.loads(l)['seq']) for l in open(path)]

R = {}
# ================================================================== cycle 1
if CY == 1:
    P(f'##### S-DARK-83 cycle 1: pooled Indus designations, edit-distance-1 pairs vs calibrated chain and freq-random ({NN} draws)')
    for LV in LEVELS:
        objs = indus_objs(LV)
        for field in ('mid', 'name'):
            R[f'{LV}_{field}'] = test(f'Indus {LV} {field}', dist(objs, field))
        R[f'{LV}_seals'] = test(f'Indus {LV} mid, seals only', dist(objs, 'mid', lambda o: o['ot'] == 'seal'))
    # planted registers on the Indus seq_raw strings
    base = dist(indus_objs('seq_raw'), 'mid'); ch = SChain(base, random.Random(83))
    P('  -- positive control: planted registers on the Indus seq_raw middle (same lengths and n), 5 replicates each')
    for prej in (1.0, 0.7, 0.4, 0.0):
        v = []
        for k in range(5):
            pl = planted(base, random.Random(830 + k), prej, ch)
            v.append(test(f'planted p_rej={prej} rep {k}', pl, nn=max(20, NN // 4), with_freq=False)['oe']['chain']['oe'])
        P(f'  == planted p_rej={prej}: chain O/E mean {mean(v):.2f} range {min(v):.2f}-{max(v):.2f}')
        R[f'planted_{prej}'] = v
    # what is substituted
    for LV in ('seq_raw', 'seq_all'):
        objs = indus_objs(LV); names = dist(objs, 'mid'); pr = e1_pairs(names)
        sub = collections.Counter(); kind = collections.Counter(); posk = collections.Counter()
        for s, t, i in pr:
            a, b = sorted((s[i], t[i]), key=str); sub[(a, b)] += 1
            ka = 'num' if a in L56.NUM else 'frame' if a in FR else 'el'; kb = 'num' if b in L56.NUM else 'frame' if b in FR else 'el'
            kind['-'.join(sorted((ka, kb)))] += 1
            L = len(s); posk['initial' if i == 0 else 'final' if i == L - 1 else 'interior'] += 1
        P(f'  {LV}: {len(pr)} edit-1 pairs; substitution kinds {dict(kind)}; position {dict(posk)}')
        P('    top substituted sign pairs: ' + ', '.join(f'{a}/{b} x{n}' for (a, b), n in sub.most_common(15)))
    json.dump(R, open(DARK + 'loop83_c1.json', 'w'), indent=1, default=str)
    save()

# ================================================================== cycle 2
if CY == 2:
    P(f'##### S-DARK-83 cycle 2: within site and across sites ({NN} draws)')
    for LV in LEVELS:
        objs = indus_objs(LV)
        P(f' -- {LV}')
        for field in ('mid', 'name'):
            for site in ('Mohenjo-daro', 'Harappa'):
                R[f'{LV}_{field}_{site}'] = test(f'{LV} {field} {site} (chain refit in site)', dist(objs, field, lambda o, s=site: o['site'] == s))
            ho = collections.defaultdict(list)
            for o in objs:
                if not o['big'] and len(o[field]) >= 2: ho[o['site']].append(o[field])
            R[f'{LV}_{field}_heldout_within'] = test_within(f'{LV} {field} held-out sites, pairs within one site', ho)
            R[f'{LV}_{field}_heldout_pooled'] = test(f'{LV} {field} held-out sites pooled as one set', sorted(set(n for v in ho.values() for n in v)))
            R[f'{LV}_{field}_seals_MD'] = test(f'{LV} {field} MD seals only', dist(objs, field, lambda o: o['site'] == 'Mohenjo-daro' and o['ot'] == 'seal'))
        # cross-site MD x Harappa
        MD = dist(objs, 'mid', lambda o: o['site'] == 'Mohenjo-daro'); HA = dist(objs, 'mid', lambda o: o['site'] == 'Harappa')
        cMD = SChain(MD, random.Random(83)); cHA = SChain(HA, random.Random(83))
        obs = e1_cross(MD, HA); nl = []
        for b in range(NN):
            r = random.Random(8500 + b); nl.append(e1_cross(chain_null(MD, r, cMD), chain_null(HA, r, cHA)))
        # pooled-chain cross null: both sites drawn from ONE chain fitted on MD+HA (a shared pan-Indus generator)
        cP = SChain(sorted(set(MD) | set(HA)), random.Random(83)); nl2 = []
        for b in range(NN):
            r = random.Random(8600 + b); nl2.append(e1_cross(chain_null(MD, r, cP), chain_null(HA, r, cP)))
        o1 = oe(obs['rate'], nl); o2 = oe(obs['rate'], nl2)
        exact = len(set(MD) & set(HA))
        P(f'  {LV} CROSS MD x Harappa: e1={obs["e1"]} rate={obs["rate"]*1e3:.3f}/1000 (identical middles shared {exact}) | per-site chains O/E {o1["oe"]:.2f} [{o1["ci"][0]:.2f},{o1["ci"][1]:.2f}] P(<=)={o1["p_lo"]:.3f} | one pooled chain O/E {o2["oe"]:.2f} [{o2["ci"][0]:.2f},{o2["ci"][1]:.2f}] P(<=)={o2["p_lo"]:.3f}')
        R[f'{LV}_cross'] = dict(obs=obs, sep=o1, pooled=o2)
    json.dump(R, open(DARK + 'loop83_c2.json', 'w'), indent=1, default=str)
    save()

# ================================================================== cycle 3
if CY == 3:
    P(f'##### S-DARK-83 cycle 3: the ladder. Every corpus length-matched to the Indus seq_raw middle at the Indus n (5 draws x {max(20, NN // 4)} chain draws), same code')
    base = dist(indus_objs('seq_raw'), 'mid'); L0 = [len(n) for n in base]; N0 = len(base)
    C63 = DARK + 'loop63_corpora/'; C56 = DARK + 'loop56_corpora/'; C32 = DARK + 'loop32_corpora/'
    import dark_loop73_common as C73
    pe = sorted(set(e['mid'] for e in C73.pe_entries() if len(e['mid']) >= 2))
    def ur3_by_city():
        by = collections.defaultdict(set)
        for l in open(DARK + 'loop48_corpora/ur3_words.jsonl'):
            d = json.loads(l); w = d['seq'][0]
            if w in C73.UR3_TITLE or w.startswith('_') or 'x' in w.split('-') or '...' in w or '$' in w: continue
            t = tuple(x for x in w.split('-') if x)
            if len(t) >= 2: by[d['site']].add(t)
        return {k: sorted(v) for k, v in by.items()}
    U3 = ur3_by_city()
    pools = [
        ('LOGO jp_given', jl(C63 + 'jp_given.jsonl')), ('LOGO jp_person_given', jl(C63 + 'jp_person_given.jsonl')),
        ('LOGO cn_ancient', jl(C63 + 'cn_ancient.jsonl')), ('LOGO cn_given', jl(C63 + 'cn_given.jsonl')),
        ('LOGO vi_given', jl(C63 + 'vi_given.jsonl')), ('LOGO ko_given', jl(C63 + 'ko_given.jsonl')),
        ('PE middle', pe),
        ('NAME ur3 owners (all)', jl(C56 + 'ur3_names_dedup.jsonl')), ('NAME ur3 Umma', U3.get('Umma', [])), ('NAME ur3 Girsu', U3.get('Girsu', [])),
        ('NAME ur3 Puzris-Dagan', U3.get('Puzriš-Dagan', [])), ('NAME ur3 Nippur', U3.get('Nippur', [])),
        ('NAME linB personnel (all)', jl(C56 + 'linb_personnel_dedup.jsonl')), ('NAME linB KN', jl(C56 + 'linb_personnel_KN.jsonl')), ('NAME linB PY', jl(C56 + 'linb_personnel_PY.jsonl')),
        ('NAME OB owners', jl(C56 + 'ob_names_dedup.jsonl')), ('NAME Latin', jl(C56 + 'latin_names_dedup.jsonl')),
        ('CODE aircraft reg (issued)', jl(C32 + 'aircraft_reg.jsonl')), ('CODE HTS', jl(C32 + 'hts.jsonl')), ('CODE ICD-10', jl(C32 + 'icd10.jsonl')),
        ('CODE Unicode names', jl(C32 + 'unicode_names.jsonl')),
    ]
    r0 = random.Random(31)
    pools.append(('SYNTH uniform IDs k=617', [tuple(r0.randrange(617) for _ in range(L)) for L in L0 * 3]))
    pools.append(('SYNTH uniform IDs k=36', [tuple(r0.randrange(36) for _ in range(L)) for L in L0 * 3]))
    nn = max(20, NN // 4)
    for lab, pool in pools:
        pool = sorted(set(tuple(s) for s in pool if len(s) >= 2 and s[0] != '($'))
        vals = []; nbs = []; fr = []
        for b in range(5):
            r = random.Random(8700 + b)
            sub, short = L56.length_match(pool, L0, N0, r)
            if len(pool) <= N0: sub = pool; short = N0 - len(pool)
            P(f'  [{lab} draw {b}: pool {len(pool)}, drawn {len(set(sub))}, length shortfall {short}]')
            t = test(f'{lab} d{b}', sub, nn=nn, seed=8800 + 100 * b)
            vals.append(t['oe']['chain']['oe']); fr.append(t['oe']['freq']['oe']); nbs.append(t['obs']['nb'])
            if len(pool) <= N0: break
        R[lab] = dict(chain=vals, freq=fr, nb=nbs, pool=len(pool))
        P(f'  == {lab}: chain O/E mean {mean(vals):.2f} [{min(vals):.2f}-{max(vals):.2f} over draws]; freq O/E {mean(fr):.2f}; nb share {mean(nbs):.3f}')
    json.dump(R, open(DARK + 'loop83_c3.json', 'w'), indent=1, default=str)
    save()

# ================================================================== cycle 4
if CY == 4:
    P(f'##### S-DARK-83 cycle 4: robustness ({NN} draws)')
    for LV in LEVELS:
        objs = indus_objs(LV); names = dist(objs, 'mid'); ch = SChain(names, random.Random(83))
        nulls = [chain_null(names, random.Random(8900 + b), ch) for b in range(NN)]
        def bylen(ns, L): return [n for n in ns if len(n) == L]
        line = f'  {LV} by length (chain fitted on all, nulls same draws):'
        for L in (2, 3, 4, 5):
            o = e1_stat(bylen(names, L)); nl = [e1_stat(bylen(x, L)) for x in nulls]; z = oe(o['rate'], nl)
            line += f' L{L} n={o["n"]} e1={o["e1"]} O/E {z["oe"]:.2f} [{z["ci"][0]:.2f},{z["ci"][1]:.2f}];'
        P(line)
        # by substituted position
        def bypos(ns):
            c = collections.Counter(); B = e1_groups(sorted(set(n for n in ns if len(n) >= 2)))
            for (L, i, _), v in B.items():
                k = 'edge' if i in (0, L - 1) else 'interior'; c[k] += len(v) * (len(v) - 1) // 2
            return c
        o = bypos(names); nl = [bypos(x) for x in nulls]
        P(f'  {LV} by substituted position: ' + '; '.join(f'{k} obs {o[k]} null {mean([d[k] for d in nl]):.1f} O/E {o[k] / max(1e-9, mean([d[k] for d in nl])):.2f}' for k in ('edge', 'interior')))
        # without numeral-only substitutions (count differences)
        def nonum(ns):
            pr = e1_pairs(ns); return sum(1 for s, t, i in pr if not (s[i] in L56.NUM and t[i] in L56.NUM))
        o = nonum(names); nl = [nonum(x) for x in nulls]
        P(f'  {LV} pairs excluding numeral-for-numeral substitutions: obs {o} null {mean(nl):.1f} [{q(nl, .025)},{q(nl, .975)}] O/E {o / mean(nl):.2f}')
        # lambda sensitivity and MLE chain
        for lam in (0.0, 0.3, 0.6, 0.9):
            c2 = SChain(names, random.Random(83), lam=lam); obs = e1_stat(names)
            nl = [e1_stat(chain_null(names, random.Random(9000 + b), c2)) for b in range(max(20, NN // 4))]
            z = oe(obs['rate'], nl); P(f'  {LV} fixed lambda {lam}: O/E {z["oe"]:.2f} [{z["ci"][0]:.2f},{z["ci"][1]:.2f}]')
        c1 = Chain(names, 1); obs = e1_stat(names)
        nl = [e1_stat(chain_names(names, 1, random.Random(9100 + b), c1)[0]) for b in range(max(20, NN // 4))]
        z = oe(obs['rate'], nl); P(f'  {LV} MLE Markov-1: O/E {z["oe"]:.2f} [{z["ci"][0]:.2f},{z["ci"][1]:.2f}]')
    save()
