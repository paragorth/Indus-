"""S-DARK-58: THE MINIMAL TEXT. When only one or two signs are written, which part of the message survives?
Cycle 1: tabulate 1- and 2-sign texts (complete, no lost sign, one copy per die) by object type, site, emblem;
         classify every sign by its slot role in long texts (>= 4 signs; frame parser of S310/S331 as in
         tools/dark_loop53.py); role mix of minimal texts vs two nulls (signs drawn by overall token frequency and by
         frequency in texts >= 4; 1,000x), per object type (seal, tablet, sealing, pot, bangle), seq_raw/strong/all.
Cycle 2: do 2-sign texts obey the frame? role-pattern table, attested-bigram share vs null, recurring pairs vs the frozen
         pairs of GRAMMAR.md / S-DARK-26; do 1-sign seals with a head sign match sealing (TAG) texts more than longer seals,
         matched on site.
Cycle 3 (dark_loop58_calib.py): the same position-class code on Ur III legends, Linear B, Latin EDH and Indus.
Cycle 4: held-out sites and IM77 (bridge + proposals) at all three levels; the prediction for new finds.
Usage: python3 tools/dark_loop58.py <1|2|4> [seq_raw|seq_strong|seq_all] [nperm]
"""
import sys, os, re, json, csv, math, random, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dark_loop41_common as L41

ROOT = '/home/user/Indus-/'
DARK = ROOT + 'data/derived/dark/'
CY = int(sys.argv[1]); LV = sys.argv[2] if len(sys.argv) > 2 else 'seq_all'; NP = int(sys.argv[3]) if len(sys.argv) > 3 else 1000
rnd = random.Random(58)
LOG = []
def P(*a):
    s = ' '.join(str(x) for x in a); print(s); LOG.append(s)

# ---------------- frame parser (S310 parse_all.py + S331 openers; identical to tools/dark_loop53.py) ----------------
OPEN = {817, 861, 820, 920, 692}; MARK = {2, 60}; MJAR = {741, 742, 745}; SUF = {400, 90}
CL = [740, 520, 151, 156, 527, 226, 617, 154, 158, 236, 700]
FISH = {235, 240, 233, 231, 220}; NUM = {1, 3, 4, 5, 16, 17, 18, 31, 32, 33, 34, 55, 56}
ROLES = ['OPENER', 'MARKER', 'COUNT', 'TITLE', 'CLOSER', 'SUFFIX', 'NAME', 'UNSEEN']

def otype(t):
    t = t.split(':')[0]
    return {'SEAL': 'seal', 'TAB': 'tablet', 'POT': 'pot', 'TAG': 'sealing', 'BNGL': 'bangle'}.get(t, 'other')

def build_qual(seqs):
    left = collections.defaultdict(collections.Counter)
    for s in seqs:
        s = list(s)
        while len(s) > 1 and s[-1] in SUF: s.pop()
        if len(s) >= 2 and s[-1] in CL: left[s[-1]][s[-2]] += 1
    Q = {}
    for c, cnt in left.items():
        tot = sum(cnt.values()); acc = 0; q = set()
        for a, n in cnt.most_common():
            if acc / tot >= 0.6: break
            q.add(a); acc += n
        Q[c] = q
    return Q

def parse(s, QUAL):
    lab = ['NAME'] * len(s); i = 0; j = len(s)
    if s[0] in OPEN:
        lab[0] = 'OPENER'; i = 1
        if len(s) > 1 and s[1] in MARK:
            lab[1] = 'MARKER'; i = 2
            if s[0] == 920 and len(s) > 2 and s[2] in MJAR: lab[2] = 'MARKER'; i = 3
    while j - 1 > i and s[j - 1] in SUF and j >= 2 and (s[j - 2] in CL or s[j - 2] in SUF): lab[j - 1] = 'SUFFIX'; j -= 1
    if j - 1 >= i and s[j - 1] in CL:
        c = s[j - 1]; lab[j - 1] = 'CLOSER'; j -= 1
        if c == 520:
            if j - 2 >= i and s[j - 1] == 33 and s[j - 2] in (705, 706): lab[j - 1] = lab[j - 2] = 'TITLE'; j -= 2
            while j - 1 >= i and s[j - 1] in FISH: lab[j - 1] = 'TITLE'; j -= 1
        elif c == 740:
            if j - 1 >= i and s[j - 1] == 100: lab[j - 1] = 'TITLE'; j -= 1
            if j - 1 >= i and s[j - 1] in QUAL.get(c, ()): lab[j - 1] = 'TITLE'; j -= 1
        elif j - 1 >= i and s[j - 1] in QUAL.get(c, ()): lab[j - 1] = 'TITLE'; j -= 1
        if j - 1 >= i and s[j - 1] in NUM and lab[j] == 'TITLE': lab[j - 1] = 'TITLE'; j -= 1
    for k in range(i, j - 1):
        if s[k] in NUM and lab[k] == 'NAME' and lab[k + 1] == 'NAME': lab[k] = lab[k + 1] = 'COUNT'
    for k in range(i, j):
        if s[k] in NUM and lab[k] == 'NAME': lab[k] = 'COUNT'
    return lab

# ---------------- corpus ----------------
def load_corpus(level):
    """current inscriptions.csv rebuilt (loop41); one copy per die; drop texts with a lost sign (000) or marked incomplete."""
    strong, allm = L41.load_levels()
    C = []
    for r in csv.DictReader(open(ROOT + 'data/raw/inscriptions.csv')):
        s = L41.parse_text(r['text'])
        if not s: continue
        C.append(dict(r, seq_raw=s, seq_strong=[strong.get(x, x) for x in s], seq_all=[allm.get(x, x) for x in s]))
    rows = []
    for r, s in L41.dedup(C, level, 'die'):
        rows.append(dict(r=r, seq=list(s), ot=otype(r['type']), site=r['site'], emb=(r['symbol'] or '-').split(':')[0],
                         clean=(r['complete'] == 'Y' and '000' not in r['text'] and not r['text'].startswith(']') and not r['text'].endswith('[')),
                         big=r['site'] in ('Mohenjo-daro', 'Harappa')))
    return rows

def sign_roles(rows, minlen=4):
    """role of each sign = majority label among its tokens in complete texts of >= minlen signs."""
    longs = [x['seq'] for x in rows if x['clean'] and len(x['seq']) >= minlen]
    QUAL = build_qual(longs)
    dist = collections.defaultdict(collections.Counter)
    for s in longs:
        for a, l in zip(s, parse(s, QUAL)): dist[a][l] += 1
    role = {a: c.most_common(1)[0][0] for a, c in dist.items()}
    return role, dist, QUAL, longs

def role_of(a, role): return role.get(a, 'UNSEEN')

def mix(signs, role):
    c = collections.Counter(role_of(a, role) for a in signs)
    return c

def wilson(k, n, z=1.96):
    if n == 0: return (0, 0, 1)
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0, c - h), min(1, c + h)

def null_mix(n, pool, role, nperm):
    """draw n signs from pool (list of tokens) nperm times; return per-role list of counts."""
    out = collections.defaultdict(list)
    for _ in range(nperm):
        c = mix(rnd.choices(pool, k=n), role)
        for r in ROLES: out[r].append(c.get(r, 0))
    return out

def compare(obs_signs, pools, role, label, nperm):
    n = len(obs_signs); obs = mix(obs_signs, role)
    P(f'  -- {label}: n = {n} sign tokens')
    P(f'     role       obs   share  | null-all mean [2.5,97.5] P | null-long mean [2.5,97.5] P   (P = two-sided permutation, over/under marked)')
    res = {}
    for r in ROLES:
        line = f'     {r:8s} {obs.get(r,0):5d} {obs.get(r,0)/max(n,1):6.2f}'
        res[r] = {'obs': obs.get(r, 0), 'n': n}
        for pname, pool in pools:
            nm = null_mix(n, pool, role, nperm)[r]
            nm.sort(); m = sum(nm) / len(nm); o = obs.get(r, 0)
            phi = sum(1 for v in nm if v >= o) / len(nm); plo = sum(1 for v in nm if v <= o) / len(nm)
            p2 = min(1.0, 2 * min(phi, plo)); arrow = ('OVER' if o > m else 'under') if p2 < 0.05 else ''
            line += f' | {m:6.1f} [{nm[int(0.025*len(nm))]:3d},{nm[int(0.975*len(nm))-1]:3d}] {p2:.3f} {arrow:5s}'
            res[r][pname] = dict(mean=m, lo=nm[int(0.025 * len(nm))], hi=nm[int(0.975 * len(nm)) - 1], p=p2)
        P(line)
    return res

def name(a): return f'W{a}'

# ---------------- cycle 1 ----------------
def cycle1(rows, nperm):
    role, dist, QUAL, longs = sign_roles(rows)
    P(f'== S-DARK-58 cycle 1, level {LV}: minimal texts, roles from {len(longs)} complete texts of >= 4 signs; {len(role)} signs get a role')
    rc = collections.Counter(role.values()); P('   role inventory (signs by majority role):', dict(rc))
    P('   closer-class signs:', ' '.join(name(a) for a in sorted(role) if role[a] == 'CLOSER'))
    P('   title/qualifier signs:', ' '.join(name(a) for a in sorted(role) if role[a] == 'TITLE'))
    P('   opener/marker/suffix:', ' '.join(f'{name(a)}:{role[a][:3]}' for a in sorted(role) if role[a] in ('OPENER', 'MARKER', 'SUFFIX')))
    one = [x for x in rows if x['clean'] and len(x['seq']) == 1]
    two = [x for x in rows if x['clean'] and len(x['seq']) == 2]
    P(f'   1-sign texts (clean, one per die): {len(one)}; 2-sign: {len(two)}; (before cleaning: '
      f'{sum(1 for x in rows if len(x["seq"])==1)} and {sum(1 for x in rows if len(x["seq"])==2)} rows incl. fragments)')
    for lab, S in (('1-sign', one), ('2-sign', two)):
        P(f'   {lab} by object type:', dict(collections.Counter(x['ot'] for x in S).most_common()))
        P(f'   {lab} by site:', dict(collections.Counter(x['site'] for x in S).most_common(10)))
        P(f'   {lab} seals by emblem:', dict(collections.Counter(x['emb'] for x in S if x['ot'] == 'seal').most_common(10)))
    P('   1-sign texts, commonest signs by object type (sign:role count):')
    for ot in ('seal', 'tablet', 'sealing', 'pot', 'bangle', 'other'):
        c = collections.Counter(x['seq'][0] for x in one if x['ot'] == ot)
        P(f'     {ot:8s} n={sum(c.values()):3d}: ' + ' '.join(f'{name(a)}:{role_of(a,role)[:3]} x{k}' for a, k in c.most_common(14)))
    # pools
    pool_all = [a for x in rows if x['clean'] for a in x['seq']]
    pool_long = [a for s in longs for a in s]
    pools = [('null-all', pool_all), ('null-long', pool_long)]
    OUT = {'level': LV, 'roles': {str(k): v for k, v in role.items()}, 'results': {}}
    for lab, S in (('1-sign', one), ('2-sign', two)):
        P(f' == {lab} texts: role mix vs nulls (pools: all clean tokens n={len(pool_all)}; tokens of texts >=4 n={len(pool_long)})')
        for ot in ('ALL', 'seal', 'tablet', 'sealing', 'pot', 'bangle'):
            sub = [x for x in S if ot == 'ALL' or x['ot'] == ot]
            if len(sub) < 5: P(f'  -- {lab} {ot}: n = {len(sub)} texts, too few'); continue
            signs = [a for x in sub for a in x['seq']]
            OUT['results'][f'{lab}/{ot}'] = compare(signs, pools, role, f'{lab} {ot} ({len(sub)} texts)', nperm)
        # within-type pool as a third null (type-specific vocabulary)
        P(f'  -- {lab}: within-object-type null (pool = clean tokens of the same object type, texts >= 3)')
        for ot in ('seal', 'tablet', 'sealing', 'pot', 'bangle'):
            sub = [x for x in S if x['ot'] == ot]
            pool = [a for x in rows if x['clean'] and x['ot'] == ot and len(x['seq']) >= 3 for a in x['seq']]
            if len(sub) < 5 or len(pool) < 50: continue
            signs = [a for x in sub for a in x['seq']]
            OUT['results'][f'{lab}/{ot}/within'] = compare(signs, [('null-type', pool)], role, f'{lab} {ot} within-type', nperm)
    # which single-sign seals are heads, which middles, which openers (list)
    P('   1-sign SEALS, every sign with role:', ' '.join(f'{name(x["seq"][0])}:{role_of(x["seq"][0],role)[:4]}' for x in one if x['ot'] == 'seal'))
    json.dump(OUT, open(DARK + f'loop58_c1_{LV}.json', 'w'), indent=0)
    return role, QUAL, longs, one, two

# ---------------- cycle 2 ----------------
FROZEN = {(33, 700), (34, 700), (32, 700), (3, 156), (31, 156), (3, 861), (33, 520), (32, 226), (17, 585), (17, 575), (32, 877),
          (33, 923), (33, 520), (590, 390), (590, 405), (435, 690), (840, 32), (806, 154), (806, 158), (550, 527), (555, 527),
          (142, 617), (176, 740), (100, 740), (760, 740), (220, 520), (240, 520), (3, 390), (16, 390), (4, 390), (5, 405)}
def cycle2(rows, nperm):
    role, dist, QUAL, longs = sign_roles(rows)
    one = [x for x in rows if x['clean'] and len(x['seq']) == 1]
    two = [x for x in rows if x['clean'] and len(x['seq']) == 2]
    P(f'== S-DARK-58 cycle 2, level {LV}: do 2-sign texts obey the frame? {len(two)} clean 2-sign texts')
    RANK = {'OPENER': 0, 'MARKER': 1, 'COUNT': 2, 'NAME': 3, 'UNSEEN': 3, 'TITLE': 4, 'CLOSER': 5, 'SUFFIX': 6}
    big = collections.Counter(); big_ot = collections.defaultdict(collections.Counter)
    for s in longs:
        for a, b in zip(s, s[1:]): big[(a, b)] += 1
    bigset = set(big)
    pat = collections.defaultdict(collections.Counter)
    ok_order = collections.Counter(); att = collections.Counter(); n_ot = collections.Counter()
    for x in two:
        a, b = x['seq']; ra, rb = role_of(a, role), role_of(b, role)
        pat[x['ot']][f'{ra[:4]}+{rb[:4]}'] += 1; n_ot[x['ot']] += 1
        if RANK[ra] <= RANK[rb]: ok_order[x['ot']] += 1
        if (a, b) in bigset: att[x['ot']] += 1
    for ot in ('seal', 'tablet', 'sealing', 'pot', 'bangle', 'other'):
        if n_ot[ot] == 0: continue
        P(f'   {ot:8s} n={n_ot[ot]:4d}: frame order kept {ok_order[ot]/n_ot[ot]:.2f}; pair attested as a bigram in long texts {att[ot]/n_ot[ot]:.2f}; '
          f'patterns ' + ', '.join(f'{k} {v}' for k, v in pat[ot].most_common(8)))
    # null: random pairs drawn by frequency (all tokens) and by long-text tokens; same stats
    pool_all = [a for x in rows if x['clean'] for a in x['seq']]; pool_long = [a for s in longs for a in s]
    for pname, pool in (('null-all', pool_all), ('null-long', pool_long)):
        o1 = []; o2 = []
        for _ in range(nperm):
            prs = [(rnd.choice(pool), rnd.choice(pool)) for _ in two]
            o1.append(sum(RANK[role_of(a, role)] <= RANK[role_of(b, role)] for a, b in prs) / len(prs))
            o2.append(sum((a, b) in bigset for a, b in prs) / len(prs))
        o1.sort(); o2.sort(); n = len(two)
        P(f'   {pname}: frame order {sum(o1)/len(o1):.2f} [{o1[int(0.025*nperm)]:.2f},{o1[int(0.975*nperm)-1]:.2f}] vs obs {sum(ok_order.values())/n:.2f}; '
          f'attested bigram {sum(o2)/len(o2):.2f} [{o2[int(0.025*nperm)]:.2f},{o2[int(0.975*nperm)-1]:.2f}] vs obs {sum(att.values())/n:.2f}')
    # also a shuffled-order null: reverse each observed pair (order test only)
    rev = sum(RANK[role_of(b, role)] <= RANK[role_of(a, role)] for x in two for a, b in [x['seq']]) / len(two)
    P(f'   reversed-pair control: frame order kept if read backwards {rev:.2f} (obs forward {sum(ok_order.values())/len(two):.2f})')
    # recurring pairs
    cnt = collections.Counter(tuple(x['seq']) for x in two)
    rec = [(p, k) for p, k in cnt.most_common() if k >= 2]
    P(f'   distinct 2-sign texts {len(cnt)}; recurring (>= 2 dies) {len(rec)} covering {sum(k for _,k in rec)} dies; singletons {sum(1 for k in cnt.values() if k==1)}')
    P('   recurring pairs (pair x dies [roles] frozen? sites):')
    for p, k in rec[:40]:
        sites = collections.Counter(x['site'][:4] for x in two if tuple(x['seq']) == p)
        ots = collections.Counter(x['ot'][:3] for x in two if tuple(x['seq']) == p)
        P(f'     {name(p[0])}-{name(p[1])} x{k} [{role_of(p[0],role)[:4]}+{role_of(p[1],role)[:4]}] {"FROZEN" if p in FROZEN else "":6s} bigram-in-long {big.get(p,0)} {dict(ots)} {dict(sites.most_common(3))}')
    fro = sum(k for p, k in rec if p in FROZEN); P(f'   share of recurring dies that are GRAMMAR.md/S-DARK-26 frozen pairs: {fro}/{sum(k for _,k in rec)}')
    # 2-sign seals specifically: what are they
    seals2 = [x for x in two if x['ot'] == 'seal']
    c = collections.Counter(tuple(x['seq']) for x in seals2)
    P(f'   2-sign SEALS n={len(seals2)}: top ' + ' '.join(f'{name(a)}-{name(b)}x{k}' for (a, b), k in c.most_common(20)))
    headed = sum(1 for x in seals2 if role_of(x['seq'][1], role) == 'CLOSER'); opened = sum(1 for x in seals2 if role_of(x['seq'][0], role) == 'OPENER')
    P(f'   2-sign seals ending in a closer-class sign {headed}/{len(seals2)}; starting with an opener {opened}/{len(seals2)}; both-middle '
      f'{sum(1 for x in seals2 if all(role_of(a,role) in ("NAME","UNSEEN") for a in x["seq"]))}')
    # ---- impressions: does a 1-sign seal with a head sign match sealing texts more than longer seals, matched on site
    P(' == seal texts matched by exact text on sealings (TAG) and on tablets, by seal text length')
    tagtexts = collections.defaultdict(set); tabtexts = collections.defaultdict(set)
    for x in rows:
        if x['ot'] == 'sealing': tagtexts[x['site']].add(tuple(x['seq'])); tagtexts['ANY'].add(tuple(x['seq']))
        if x['ot'] == 'tablet': tabtexts[x['site']].add(tuple(x['seq'])); tabtexts['ANY'].add(tuple(x['seq']))
    seals = [x for x in rows if x['clean'] and x['ot'] == 'seal']
    def lclass(x):
        n = len(x['seq'])
        if n == 1: return '1-head' if role_of(x['seq'][0], role) == 'CLOSER' else '1-other'
        return {2: '2', 3: '3'}.get(n, '4+')
    tab = collections.defaultdict(lambda: [0, 0, 0, 0, 0])
    for x in seals:
        k = lclass(x); t = tuple(x['seq']); tab[k][0] += 1
        tab[k][1] += t in tagtexts[x['site']]; tab[k][2] += t in tagtexts['ANY']; tab[k][3] += t in tabtexts[x['site']]; tab[k][4] += t in tabtexts['ANY']
    P('   class     seals  TAG-same-site  TAG-any  TAB-same-site  TAB-any')
    for k in ('1-head', '1-other', '2', '3', '4+'):
        n, a, b, c_, d = tab[k]
        P(f'   {k:8s} {n:5d}  {a:4d} ({a/max(n,1):.3f})  {b:4d} ({b/max(n,1):.3f})  {c_:4d} ({c_/max(n,1):.3f})  {d:4d} ({d/max(n,1):.3f})')
    # site-matched: only sites with >= 5 sealings
    P('   same-site TAG match restricted to sites with >= 5 distinct sealing texts (Lothal, Dholavira, Kalibangan, MD, Harappa):')
    good = {s for s, v in tagtexts.items() if s != 'ANY' and len(v) >= 5}
    for k in ('1-head', '1-other', '2', '3', '4+'):
        sub = [x for x in seals if lclass(x) == k and x['site'] in good]
        m = sum(tuple(x['seq']) in tagtexts[x['site']] for x in sub)
        p, lo, hi = wilson(m, len(sub))
        P(f'     {k:8s} {m}/{len(sub)} = {p:.3f} [{lo:.3f},{hi:.3f}]')
    # the lone-head seals: what sign, where, matched?
    P('   1-sign seals with a head sign:', ' '.join(f'{name(x["seq"][0])}@{x["site"][:4]}{"*" if tuple(x["seq"]) in tagtexts[x["site"]] else ""}' for x in seals if lclass(x) == '1-head'))
    # one-sign sealings: are they heads?
    tags1 = [x for x in rows if x['clean'] and x['ot'] == 'sealing' and len(x['seq']) == 1]
    P(f'   1-sign sealings (clean) n={len(tags1)}: ' + ' '.join(f'{name(x["seq"][0])}:{role_of(x["seq"][0],role)[:4]}@{x["site"][:4]}' for x in tags1))
    json.dump({'level': LV, 'recurring': [[list(p), k] for p, k in rec], 'match': {k: v for k, v in tab.items()}}, open(DARK + f'loop58_c2_{LV}.json', 'w'), indent=0)

# ---------------- cycle 4: held-out sites and IM77 ----------------
def load_im77():
    rows = collections.defaultdict(list)
    for r in csv.DictReader(open(ROOT + 'data/im77/im77_corpus_lines.csv')):
        if r['line'] == '9' or r['direction_code'] == '9': continue
        rows[r['text_no']].append(r)
    texts = []
    for tn, ls in rows.items():
        ls.sort(key=lambda r: (int(r['side']), int(r['line'])))
        toks = [x for r in ls for x in r['signs_clean'].split()]
        clean = '0' not in toks and all(r['direction_code'] != '5' for r in ls)
        s = [int(x) for x in toks if x != '0']
        if not s: continue
        ot = {'seal': 'seal', 'sealing': 'sealing', 'miniature tablet': 'tablet', 'copper tablet': 'tablet', 'pottery graffito': 'pot'}.get(ls[0]['object_type'], 'other')
        texts.append(dict(tn=tn, seq=s, ot=ot, site=ls[0]['site'], clean=clean, fs=ls[0]['fs80']))
    return texts

def m_to_w():
    """Mahadevan -> Wells map: bridge_extended (W -> [M]) inverted, then S-DARK-27 proposals (flagged)."""
    BR = json.load(open(ROOT + 'data/derived/bridge_extended.json'))
    m2w = {}; prop = set()
    for w, ms in BR.items():
        for m in ms: m2w.setdefault(int(m), int(w))
    PR = json.load(open(DARK + 'bridge_proposals.json'))['proposals']
    for p in PR:
        if int(p['M']) not in m2w: m2w[int(p['M'])] = int(p['W']); prop.add(int(p['M']))
    return m2w, prop

def cycle4(rows, nperm):
    role, dist, QUAL, longs = sign_roles(rows)
    P(f'== S-DARK-58 cycle 4, level {LV}: held-out sites and IM77; roles fitted on all Wells long texts (as cycle 1)')
    # (a) held-out: sites outside the IM77 coverage (loop41 heldout_split), roles fitted on home only
    home = [x for x in rows if x['site'] in ('Mohenjo-daro', 'Harappa', 'Chanhu-daro', 'Lothal', 'Kalibangan')]
    held = [x for x in rows if x not in home]
    roleH, _, _, longsH = sign_roles(home)
    P(f'   roles refitted on home sites only ({len(longsH)} long texts); held-out sites: {dict(collections.Counter(x["site"] for x in held).most_common(8))}')
    pool_all = [a for x in home if x['clean'] for a in x['seq']]; pool_long = [a for s in longsH for a in s]
    for lab, n in (('1-sign', 1), ('2-sign', 2)):
        for ot in ('ALL', 'seal', 'tablet', 'pot'):
            sub = [x for x in held if x['clean'] and len(x['seq']) == n and (ot == 'ALL' or x['ot'] == ot)]
            if len(sub) < 5: P(f'  -- held-out {lab} {ot}: n = {len(sub)}, too few'); continue
            compare([a for x in sub for a in x['seq']], [('null-all', pool_all), ('null-long', pool_long)], roleH, f'HELD-OUT {lab} {ot} ({len(sub)} texts)', nperm)
    # home-only 1-sign seals for the prediction
    # (b) IM77 via bridge
    T = load_im77(); m2w, prop = m_to_w()
    def conv(s): return [m2w.get(a, 100000 + a) for a in s]   # unmapped M signs become unique >100000 ids (role UNSEEN/NAME)
    I = [dict(seq=conv(x['seq']), mseq=x['seq'], ot=x['ot'], site=x['site'], clean=x['clean'], tn=x['tn']) for x in T]
    # dedup: one copy per (site, ot, text) for tablets/sealings/pots; seals by text_no
    seen = set(); J = []
    for x in I:
        k = (x['site'], x['ot'], tuple(x['seq'])) if x['ot'] != 'seal' else (x['tn'],)
        if k in seen: continue
        seen.add(k); J.append(x)
    roleI, distI, _, longsI = sign_roles(J)
    mapped = sum(1 for s in longsI for a in s if a < 100000) / max(1, sum(len(s) for s in longsI))
    P(f'   IM77: {len(J)} texts after dedup, {len(longsI)} long; {mapped:.2f} of long-text tokens map to Wells (bridge {len(m2w)-len(prop)} + {len(prop)} proposed)')
    # agreement of IM77-fitted roles with Wells roles on mapped signs
    agree = [(a, roleI[a], role.get(a)) for a in roleI if a < 100000 and a in role]
    P(f'   role agreement IM77-fit vs Wells-fit on {len(agree)} shared signs: {sum(1 for a,b,c in agree if b==c)/max(1,len(agree)):.2f}; disagreements: '
      + ' '.join(f'{name(a)}:{b[:3]}/{c[:3]}' for a, b, c in agree if b != c)[:600])
    pool_all = [a for x in J if x['clean'] for a in x['seq']]; pool_long = [a for s in longsI for a in s]
    for lab, n in (('1-sign', 1), ('2-sign', 2)):
        for ot in ('ALL', 'seal', 'tablet', 'sealing', 'pot'):
            sub = [x for x in J if x['clean'] and len(x['seq']) == n and (ot == 'ALL' or x['ot'] == ot)]
            if len(sub) < 5: P(f'  -- IM77 {lab} {ot}: n = {len(sub)}, too few'); continue
            signs = [a for x in sub for a in x['seq']]
            P(f'     (IM77 {lab} {ot}: {sum(1 for a in signs if a in prop or (a < 100000 and any(m2w.get(m)==a for m in prop)))} tokens rest on a proposed bridge entry; '
              f'{sum(1 for a in signs if a >= 100000)} unmapped)')
            compare(signs, [('null-all', pool_all), ('null-long', pool_long)], roleI, f'IM77 {lab} {ot} ({len(sub)} texts), IM77-fitted roles', nperm)
    one_s = [x for x in J if x['clean'] and len(x['seq']) == 1 and x['ot'] == 'seal']
    P('   IM77 1-sign seals (M number : IM77-fitted role):', ' '.join(f'M{x["mseq"][0]}:{role_of(x["seq"][0],roleI)[:4]}' for x in one_s))
    # (c) the prediction: P(new 1-sign seal carries a head-class sign), with Wilson CI, Wells home / held-out / IM77
    P(' == prediction for new finds: a new complete 1-sign SEAL carries a closer-class (head) sign')
    for lab, S, R in (('Wells all sites', [x for x in rows if x['clean'] and x['ot'] == 'seal' and len(x['seq']) == 1], role),
                      ('Wells home', [x for x in home if x['clean'] and x['ot'] == 'seal' and len(x['seq']) == 1], roleH),
                      ('Wells held-out', [x for x in held if x['clean'] and x['ot'] == 'seal' and len(x['seq']) == 1], roleH),
                      ('IM77', one_s, roleI)):
        k = sum(1 for x in S if role_of(x['seq'][0], R) == 'CLOSER'); p, lo, hi = wilson(k, len(S))
        k2 = sum(1 for x in S if role_of(x['seq'][0], R) in ('CLOSER', 'TITLE')); p2, lo2, hi2 = wilson(k2, len(S))
        k3 = sum(1 for x in S if role_of(x['seq'][0], R) in ('NAME', 'UNSEEN')); p3, lo3, hi3 = wilson(k3, len(S))
        k4 = sum(1 for x in S if role_of(x['seq'][0], R) == 'OPENER'); p4, lo4, hi4 = wilson(k4, len(S))
        P(f'   {lab:15s} n={len(S):3d}: head {k}/{len(S)} = {p:.2f} [{lo:.2f},{hi:.2f}]; head-or-title {p2:.2f} [{lo2:.2f},{hi2:.2f}]; '
          f'middle/unseen {p3:.2f} [{lo3:.2f},{hi3:.2f}]; opener {p4:.2f} [{lo4:.2f},{hi4:.2f}]')
    # same for tablets and pots (1-sign)
    for ot in ('tablet', 'pot', 'sealing'):
        S = [x for x in rows if x['clean'] and x['ot'] == ot and len(x['seq']) == 1]
        k = sum(1 for x in S if role_of(x['seq'][0], role) == 'CLOSER'); kc = sum(1 for x in S if role_of(x['seq'][0], role) == 'COUNT')
        km = sum(1 for x in S if role_of(x['seq'][0], role) in ('NAME', 'UNSEEN'))
        p, lo, hi = wilson(k, len(S)); pc, loc, hic = wilson(kc, len(S)); pm, lom, him = wilson(km, len(S))
        P(f'   Wells {ot:8s} n={len(S):3d}: head {p:.2f} [{lo:.2f},{hi:.2f}]; numeral {pc:.2f} [{loc:.2f},{hic:.2f}]; middle/unseen {pm:.2f} [{lom:.2f},{him:.2f}]')

if __name__ == '__main__':
    rows = load_corpus(LV)
    if CY == 1: cycle1(rows, NP)
    elif CY == 2: cycle2(rows, NP)
    elif CY == 4: cycle4(rows, NP)
    open(DARK + f'loop58_c{CY}_{LV}.txt', 'w').write('\n'.join(LOG) + '\n')
