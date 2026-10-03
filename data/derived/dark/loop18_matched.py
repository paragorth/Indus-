"""Loop 18: the S321/S322 'not names' claim re-done at MATCHED granularity.
Indus: data/derived/merged-corpus-canonical.json, one row per object (NOT deduplicated: 5,369 rows, 3,599 distinct site+text).
Ur III: CDLI ATF dump, @seal sections on Ur III tablets (one impression per section) + Ur III seal objects.
Granularities: (1) object level both sides; (2) distinct texts / distinct legends; (3) one physical seal (Indus SEAL* rows,
one per cisi+text; Ur III distinct legends, and separately CDLI physical seal objects).
Statistic: share of unique strings / bigram-null share (S321), size-matched, 50 draws; Zipf slope of unit frequencies;
person-level homonymy (distinct seals per name / distinct seals per Indus middle). Planted control with known homonymy.
Usage: python3 loop18_matched.py <cycle 1|2|3>
"""
import sys, re, csv, json, random, collections, statistics as st, math
SP = '/tmp/claude-0/-home-user-Indus-/874df4c7-80d6-5f08-b42c-eea96a214079/scratchpad/'
HERE = '/home/user/Indus-/data/derived/dark/'
CYCLE = sys.argv[1] if len(sys.argv) > 1 else '1'
OUT = open(HERE + f'loop18_cycle{CYCLE}.txt', 'w')
def P(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); OUT.write(s + '\n'); OUT.flush()

# ---------------- Ur III legends with P-number ----------------
STOP = {'dub', 'dumu', 'arad2', 'ir3', 'ir11', 'dam', 'szabra', 'ensi2', 'sukkal', 'kiszib3', 'kiszib', 'sanga', 'nu', 'gudu4',
        'sipa', 'nar', 'ugula', 'lugal', 'szagina', 'nin', 'gal'}
def load_ur3():
    cache = HERE + 'loop18_ur3_impressions.json'
    try:
        return json.load(open(cache))
    except Exception:
        pass
    csv.field_size_limit(10 ** 9)
    per = {}
    for r in csv.DictReader(open(SP + 'cdli_cat.csv', errors='ignore')):
        if 'Ur III' in (r.get('period') or ''):
            per[r['id_text'].zfill(6)] = (r.get('object_type', '') or '').lower()
    def clean(t): return re.sub(r'[#?!\[\]<>*]|~\w+', '', t)
    NUM = re.compile(r'^(\d+|n)\((N|n|disz|u|asz|gesz|barig|ban|szar)', re.I)
    out = []; pid = None; in_seal = False; cur = []; lines = []
    def flush():
        nonlocal cur, lines
        if pid in per and cur:
            out.append({'pid': pid, 'obj': per[pid], 'legend': cur, 'line1': lines[0] if lines else []})
        cur = []; lines = []
    for line in open(SP + 'cdli.atf', errors='ignore'):
        if line.startswith('&P'):
            flush(); pid = line[2:8]; in_seal = False; continue
        if pid not in per: continue
        if line.startswith('@seal'): flush(); in_seal = True; continue
        if line.startswith('@'):
            if not line.startswith(('@column', '@surface', '@face')):
                if not in_seal: flush()
            if line.startswith(('@obverse', '@reverse', '@envelope', '@tablet', '@left', '@right', '@top', '@bottom')):
                if in_seal: flush()
                in_seal = False
            continue
        seal_obj = per[pid].startswith('seal (')
        if not (in_seal or seal_obj): continue
        m = re.match(r"^\d+'?\.\s+(.*)", line)
        if not m: continue
        body = re.sub(r'\{[^}]*\}', '', m.group(1))
        words = []
        for w in body.split():
            w = clean(w)
            if not w or NUM.match(w) or re.match(r'^(x|\.\.\.|\d.*|\(.*)$', w): continue
            signs = [s for s in re.split(r'[-.]', w.lower()) if s and s not in ('x', '...')]
            if signs and all(re.match(r'^[a-z0-9]+$', s) for s in signs): words.append(signs)
        if words:
            cur.extend(words); lines.append(words)
    flush()
    out = [l for l in out if 2 <= sum(len(w) for w in l['legend']) <= 14]
    json.dump(out, open(cache, 'w'))
    return out

def ur3_name(rec, mode='line1'):
    """name = syllables of line 1 (first word(s) before a title/kin word); mode 'legend' = S322 rule on the whole legend."""
    words = rec['line1'] if mode == 'line1' else rec['legend']
    n = []
    for w in words:
        if w[0] in STOP or any(s in STOP for s in w[:1]): break
        n.extend(w)
    if 2 <= len(n) <= 6: return tuple(n)
    return None

# ---------------- Indus middles ----------------
OP = {817, 861, 820, 825}; MARK = {2, 60}
CLOSE = {740, 520, 151, 156, 527, 526, 226, 617, 154, 158, 700, 236, 705, 706, 704}; SUF = {400, 90, 93}
def middle(seq):
    s = list(seq)
    if s and s[-1] in SUF: s = s[:-1]
    if s and s[-1] in CLOSE: s = s[:-1]
    if s and s[0] in OP: s = s[1:]
    if s and s[0] in MARK: s = s[1:]
    return tuple(s) if len(s) >= 2 else None

def load_indus():
    return json.load(open('/home/user/Indus-/data/derived/merged-corpus-canonical.json'))

# ---------------- statistics ----------------
def uniq(ms):
    if not ms: return float('nan')
    c = collections.Counter(ms); return sum(1 for m in ms if c[m] == 1) / len(ms)
def bigram(ms):
    b = collections.defaultdict(collections.Counter)
    for m in ms:
        p = 'S'
        for c in m: b[p][c] += 1; p = c
    return b
def gen(b, n, rnd):
    out = []; p = 'S'
    for _ in range(n):
        src = b[p] if b[p] else b['S']; ks, ws = zip(*src.items()); c = rnd.choices(ks, ws)[0]; out.append(c); p = c
    return tuple(out)
def uniq_ratio(ms, n, rnd, draws=50):
    """size-matched: sample n strings, uniqueness vs bigram trained on the sample, same lengths. Returns (obs, null, ratio)."""
    rs = []; ns = []
    for _ in range(draws):
        sub = rnd.sample(ms, min(n, len(ms)))
        rs.append(uniq(sub)); b = bigram(sub)
        ns.append(uniq([gen(b, len(m), rnd) for m in sub]))
    return st.mean(rs), st.mean(ns), st.mean(rs) / st.mean(ns), st.pstdev([r / q for r, q in zip(rs, ns)])
def zipf_slope(ms, top=None):
    c = collections.Counter(u for m in ms for u in m)
    f = sorted(c.values(), reverse=True)
    if top: f = f[:top]
    xs = [math.log(i + 1) for i in range(len(f))]; ys = [math.log(v) for v in f]
    mx = st.mean(xs); my = st.mean(ys)
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs), len(f)
def homonymy(ms):
    """ms = one string per distinct seal. Returns dict: share of seals sharing their string with another seal, mean seals per
    distinct string, max, number of strings on >= 2 seals."""
    c = collections.Counter(ms)
    return {'n_seals': len(ms), 'n_distinct': len(c), 'share_shared': sum(v for v in c.values() if v > 1) / len(ms),
            'seals_per_string': len(ms) / len(c), 'max': c.most_common(1)[0][1] if c else 0,
            'strings_ge2': sum(1 for v in c.values() if v > 1), 'top': c.most_common(5)}
def homonymy_matched(ms, n, rnd, draws=50):
    """size-matched homonymy with its bigram null: share_shared obs / null and seals_per_string obs / null."""
    ss = []; sn = []; ps = []; pn = []
    for _ in range(draws):
        sub = rnd.sample(ms, min(n, len(ms))); h = homonymy(sub); b = bigram(sub)
        g = [gen(b, len(m), rnd) for m in sub]; hn = homonymy(g)
        ss.append(h['share_shared']); sn.append(hn['share_shared']); ps.append(h['seals_per_string']); pn.append(hn['seals_per_string'])
    return st.mean(ss), st.mean(sn), st.mean(ps), st.mean(pn)

def fmt_u(t): return f'unique {t[0]:.3f} vs bigram {t[1]:.3f}; ratio {t[2]:.3f} (sd {t[3]:.3f})'

# =====================================================================================================================
def indus_sets(d, level='seq_raw', sites=None, seals_only_types=('SEAL',)):
    """Return dict of granularity -> list of middles.
    object: every row (all object types) with a middle; object_seal: every SEAL* row (incl. duplicate rows);
    text: distinct (site, text); seal_phys: SEAL* rows deduplicated by (cisi, text) when cisi is a real id."""
    rows = [r for r in d if (sites is None or r['site'] in sites)]
    obj = [middle(r[level]) for r in rows]; obj = [m for m in obj if m]
    objseal = [middle(r[level]) for r in rows if r['type'].startswith(seals_only_types)]; objseal = [m for m in objseal if m]
    seen = set(); text = []
    for r in rows:
        k = (r['site'], tuple(r[level]))
        if k in seen: continue
        seen.add(k); m = middle(r[level])
        if m: text.append(m)
    seen = set(); phys = []
    for r in rows:
        if not r['type'].startswith(seals_only_types): continue
        k = (r['cisi'], tuple(r[level])) if r['cisi'] not in ('-', '', None) else ('row', id(r))
        if k in seen: continue
        seen.add(k); m = middle(r[level])
        if m: phys.append(m)
    return {'object': obj, 'object_seal': objseal, 'text': text, 'seal_phys': phys}

def ur3_sets(U, mode='line1'):
    """impression: every @seal section / seal object (one per tablet impression); legend: distinct full legends;
    sealobj: CDLI physical seal objects only (object_type == seal), one per object."""
    imp = [ur3_name(r, mode) for r in U]; imp = [n for n in imp if n]
    seen = set(); leg = []
    for r in U:
        k = tuple(tuple(w) for w in r['legend'])
        if k in seen: continue
        seen.add(k); n = ur3_name(r, mode)
        if n: leg.append(n)
    seen = set(); sobj = []
    for r in U:
        if not r['obj'].startswith('seal ('): continue
        if r['pid'] in seen: continue
        seen.add(r['pid']); n = ur3_name(r, mode)
        if n: sobj.append(n)
    return {'impression': imp, 'legend': leg, 'sealobj': sobj}

def cycle1():
    import datetime
    P(f'# loop18 cycle 1: matched granularity, uniqueness ratio and Zipf slope ({datetime.datetime.now().isoformat(timespec="minutes")})')
    d = load_indus(); U = load_ur3()
    P(f'Ur III records: {len(U)} impressions/sections; distinct legends {len({tuple(tuple(w) for w in r["legend"]) for r in U})}; '
      f'seal objects {len({r["pid"] for r in U if r["obj"].startswith("seal (")})}; tablets {len({r["pid"] for r in U if not r["obj"].startswith("seal (")})}')
    I = indus_sets(d); R = ur3_sets(U)
    for k, v in I.items(): P(f'Indus {k:12s} n={len(v)} distinct={len(set(v))}')
    for k, v in R.items(): P(f'Ur III {k:12s} n={len(v)} distinct={len(set(v))}')
    rnd = random.Random(18)
    pairs = [('(1) object level', 'object_seal', 'impression'), ('(1b) object level, all Indus objects', 'object', 'impression'),
             ('(2) deduplicated', 'text', 'legend'), ('(3) one physical seal', 'seal_phys', 'legend'),
             ('(3b) one physical seal, CDLI seal objects', 'seal_phys', 'sealobj')]
    res = {}
    for label, ik, rk in pairs:
        im = I[ik]; rm = R[rk]; n = min(len(im), len(rm))
        if n < 50: P(f'\n## {label}: skipped, n={n}'); continue
        ui = uniq_ratio(im, n, rnd); ur = uniq_ratio(rm, n, rnd)
        zi = zipf_slope(im); zr = zipf_slope(rm); zi100 = zipf_slope(im, 100); zr100 = zipf_slope(rm, 100)
        P(f'\n## {label}: matched n={n}')
        P(f'  Indus  {ik:12s} {fmt_u(ui)}; Zipf slope all {zi[0]:.3f} ({zi[1]} units), top100 {zi100[0]:.3f}')
        P(f'  Ur III {rk:12s} {fmt_u(ur)}; Zipf slope all {zr[0]:.3f} ({zr[1]} units), top100 {zr100[0]:.3f}')
        res[label] = {'n': n, 'indus': ui, 'ur3': ur, 'zipf_indus': zi, 'zipf_ur3': zr}
    # unmatched full-size for reference
    P('\n## full-size (own n) for reference')
    for k, v in I.items(): P(f'  Indus {k:12s} {fmt_u(uniq_ratio(v, len(v), rnd, 20))}')
    for k, v in R.items(): P(f'  Ur III {k:12s} {fmt_u(uniq_ratio(v, len(v), rnd, 20))}')
    json.dump(res, open(HERE + 'loop18_cycle1.json', 'w'), default=str)

def planted(rnd, n_names=3000, n_seals=8000, zipf_a=1.0, imp_mean=3.0, base=None):
    """Planted name population: a stock of n_names strings generated from a syllable bigram (trained on `base`), Zipf-weighted
    (exponent zipf_a); n_seals seals each with one owner name drawn from the stock (known homonymy); each seal impressed
    k times, k ~ 1 + geometric with mean imp_mean (heavy right tail via mixture: 5% of seals impressed 20x more)."""
    b = bigram(base); L = [len(m) for m in base]
    stock = list({gen(b, rnd.choice(L), rnd) for _ in range(n_names * 2)})[:n_names]
    w = [1 / (i + 1) ** zipf_a for i in range(len(stock))]
    seals = rnd.choices(stock, w, k=n_seals)
    imps = []
    for s in seals:
        k = 1 + int(rnd.expovariate(1 / (imp_mean - 1))) if imp_mean > 1 else 1
        if rnd.random() < 0.05: k *= 20
        imps.extend([s] * k)
    return {'impression': imps, 'legend': list(dict.fromkeys(seals)) if False else seals, 'persons': seals}

def cycle2():
    import datetime
    P(f'# loop18 cycle 2: person-level homonymy and planted control ({datetime.datetime.now().isoformat(timespec="minutes")})')
    d = load_indus(); U = load_ur3(); I = indus_sets(d); R = ur3_sets(U); rnd = random.Random(182)
    P('\n## Ur III: distinct seals (= distinct legends) per owner name')
    for rk in ('legend', 'sealobj'):
        h = homonymy(R[rk]); P(f'  {rk:10s} ' + json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in h.items()}, default=str))
    P('  (impression level, for contrast) ' + json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in homonymy(R['impression']).items()}, default=str))
    P('\n## Indus: distinct seals per middle (SEAL* rows, one per cisi+text)')
    for ik in ('seal_phys', 'object_seal', 'text'):
        h = homonymy(I[ik]); P(f'  {ik:12s} ' + json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in h.items()}, default=str))
    n = len(I['seal_phys'])
    P(f'\n## size-matched homonymy (n={n}, 50 draws): share of seals whose string recurs on another seal, obs vs bigram null; seals per distinct string obs vs null')
    for side, ms in (('Indus seal_phys', I['seal_phys']), ('Indus text', I['text']), ('Ur III legend', R['legend']), ('Ur III sealobj', R['sealobj']),
                     ('Ur III impression', R['impression'])):
        a, b_, c, e = homonymy_matched(ms, n, rnd)
        P(f'  {side:18s} shared {a:.3f} vs null {b_:.3f} (ratio {a / b_ if b_ else float("nan"):.2f}); seals/string {c:.3f} vs null {e:.3f}')
    # planted control
    P('\n## planted control: 3,000-name Zipf stock, 8,000 seals, impressions geometric mean 3 (+5% heavy users x20)')
    for a in (0.6, 1.0, 1.3):
        pl = planted(rnd, zipf_a=a, base=R['legend'])
        distinct_seals = pl['persons']  # one entry per seal; homonymy = same name on different seals
        for lab, ms in (('impressions', pl['impression']), ('seals (one per seal)', distinct_seals), ('distinct strings', list(set(distinct_seals)))):
            u = uniq_ratio(ms, n, rnd, 20); h = homonymy(ms)
            P(f'  zipf a={a} {lab:22s} n={len(ms)}: {fmt_u(u)}; share_shared {h["share_shared"]:.3f}; seals/string {h["seals_per_string"]:.2f}; max {h["max"]}')
    P('  NOTE: deduplicating by distinct string forces uniqueness to 1.0 by construction; the person-level test must count SEALS, not strings.')

def cycle3():
    import datetime
    P(f'# loop18 cycle 3: replication by site, sign level, name rule ({datetime.datetime.now().isoformat(timespec="minutes")})')
    d = load_indus(); U = load_ur3(); rnd = random.Random(183)
    R1 = ur3_sets(U, 'line1'); R2 = ur3_sets(U, 'legend')
    P('\n## Ur III name rule: line 1 vs S322 whole-legend rule (distinct legends), own-size and n=1700 matched')
    for lab, R in (('line1', R1), ('S322 rule', R2)):
        for rk in ('impression', 'legend', 'sealobj'):
            u = uniq_ratio(R[rk], 1700, rnd); h = homonymy(R[rk])
            P(f'  {lab:9s} {rk:10s} n={len(R[rk])}: {fmt_u(u)}; share_shared {h["share_shared"]:.3f}; seals/string {h["seals_per_string"]:.2f}')
    P('\n## Indus by sign level and site (seal_phys and text; n matched to the smaller side, Ur III legend sampled to the same n)')
    for level in ('seq_raw', 'seq_strong', 'seq_all'):
        for sites, slab in ((None, 'all'), ({'Mohenjo-daro'}, 'MD'), ({'Harappa'}, 'Harappa'), ({'Lothal', 'Kalibangan', 'Dholavira', 'Chanhu-daro'}, 'Lothal+Kalibangan+Dholavira+Chanhu')):
            I = indus_sets(d, level, sites)
            for ik in ('seal_phys', 'text', 'object_seal'):
                ms = I[ik]
                if len(ms) < 60: P(f'  {level} {slab:36s} {ik:12s} n={len(ms)} too small'); continue
                n = len(ms); ui = uniq_ratio(ms, n, rnd, 30); ur = uniq_ratio(R1['legend'], n, rnd, 30); hi = homonymy(ms)
                P(f'  {level} {slab:36s} {ik:12s} n={n}: Indus ratio {ui[2]:.3f} (unique {ui[0]:.3f}/{ui[1]:.3f}); Ur III legend at n={n}: ratio {ur[2]:.3f}; '
                  f'Indus share_shared {hi["share_shared"]:.3f}, seals/string {hi["seals_per_string"]:.3f}; Zipf {zipf_slope(ms)[0]:.3f}')
    # Indus sealings vs seals: do the same middles recur on impressions (TAG) as on seals?
    P('\n## Indus impressions (TAG* sealings) vs seals: share of sealing middles that also occur on a seal (any site), seq_raw')
    sm = set(indus_sets(d)['seal_phys'])
    tags = [middle(r['seq_raw']) for r in d if r['type'].startswith('TAG')]; tags = [m for m in tags if m]
    P(f'  sealings with middle n={len(tags)}, distinct {len(set(tags))}, on a known seal {sum(1 for m in tags if m in sm)} ({sum(1 for m in tags if m in sm)/len(tags):.3f}); '
      f'homonymy among sealings: {json.dumps({k: (round(v,3) if isinstance(v,float) else v) for k,v in homonymy(tags).items()}, default=str)}')

if __name__ == '__main__':
    {'1': cycle1, '2': cycle2, '3': cycle3}[CYCLE]()
