"""S-DARK-24: transcriber disagreement as data.
Align Wells (merged-corpus-canonical, seq_raw) and Mahadevan IM77 readings object by object:
within each site, bridge-aware edit distance; accept unique best matches below a cost threshold;
chance control = the same procedure against the wrong site's Wells objects. Then record every
aligned position (agree / substitution / insertion / deletion / lost-sign) with metadata.
Writes data/derived/dark/loop24_pairs.json (one record per aligned object pair)."""
import json, csv, collections, re, itertools, random, sys
R = '/home/user/Indus-/'
OUT = R + 'data/derived/dark/'
bridge = {int(k): v for k, v in json.load(open(R + 'data/derived/bridge_extended.json')).items()}
inv = collections.defaultdict(set)
for w, ms in bridge.items():
    for m in ms: inv[m].add(w)

# ---------- Wells objects with lost-sign (000) positions and metadata ----------
ins = list(csv.DictReader(open(R + 'data/raw/inscriptions.csv')))
def parse(t): return [int(k) for k in re.findall(r'\d{3}', t)]
meta = collections.defaultdict(list)
for r in ins:
    p = parse(r['text'])
    meta[(r['cisi'], tuple(k for k in p if k))].append(r)
    meta[(r['cisi'], tuple(k for k in p[::-1] if k))].append(r)
corpus = json.load(open(R + 'data/derived/merged-corpus-canonical.json'))
W = []
for i, x in enumerate(corpus):
    s = tuple(x['seq_raw'])
    rs = meta.get((x['cisi'], s), [])
    r = rs[0] if rs else None
    full = s
    if r:
        p = parse(r['text'])
        if tuple(k for k in p if k) == s: full = tuple(p)
        else: full = tuple(p[::-1])
    W.append(dict(idx=i, cisi=x['cisi'], site=x['site'], type=x['type'], seq=s, full=full,
                  symbol=x['symbol'], material=x['material'],
                  condition=(r['condition'].strip().capitalize() if r else '?'),
                  preservation=(r['preservation'].strip() if r else '?'),
                  complete=x['complete'], nlines=(r['text'].count('/') + 1 if r else 1),
                  lines=(r['text'] if r else '')))
site_map = {'MD': 'Mohenjo-daro', 'HP': 'Harappa', 'LL': 'Lothal', 'KB': 'Kalibangan', 'CD': 'Chanhu-daro'}
five = set(site_map.values())
def wsite(code):
    if code in site_map: return [w for w in W if w['site'] == site_map[code]]
    return [w for w in W if w['site'] not in five]

# ---------- IM77 texts ----------
lines = collections.defaultdict(list)
for r in csv.DictReader(open(R + 'data/im77/im77_corpus_lines.csv')):
    if r['line'] == '9': continue
    lines[r['text_no']].append(r)
IM = []
for tn, rs in lines.items():
    rs.sort(key=lambda r: (int(r['side']), int(r['line'])))
    seq, doubt, lineid = [], [], []
    for li, r in enumerate(rs):
        toks = r['signs_clean'].split()
        dp = set(int(k) for k in r['doubtful_positions'].split(',') if k.strip().isdigit()) if r['doubtful_positions'] else set()
        for j, t in enumerate(toks):
            seq.append(int(t)); doubt.append((j + 1) in dp); lineid.append(li)
    if not seq: continue
    r0 = rs[0]
    IM.append(dict(text_no=tn, site_code=r0['site_code'], seq=seq, doubt=doubt, lineid=lineid, nlines=len(rs),
                   fs80=int(r0['fs80']), object_type=r0['object_type'], source=r0['source']))

# ---------- alignment ----------
UNB = 0.5      # cost of aligning an unbridged Wells sign to any M sign (unknown correspondence)
LOST = 0.25    # cost of aligning a lost sign (Wells 000 / M 0) to any sign
def sub_cost(w, m, corr):
    if w == 0 or m == 0: return LOST
    c = corr.get(w)
    if c is None: return UNB
    return 0.0 if m in c else 1.0
def align(ws, ms, corr):
    n, k = len(ws), len(ms)
    D = [[0.0] * (k + 1) for _ in range(n + 1)]
    for i in range(1, n + 1): D[i][0] = D[i-1][0] + (0.5 if ws[i-1] == 0 else 1.0)
    for j in range(1, k + 1): D[0][j] = D[0][j-1] + (0.5 if ms[j-1] == 0 else 1.0)
    for i in range(1, n + 1):
        wi = ws[i-1]
        for j in range(1, k + 1):
            mj = ms[j-1]
            D[i][j] = min(D[i-1][j-1] + sub_cost(wi, mj, corr),
                          D[i-1][j] + (0.5 if wi == 0 else 1.0),
                          D[i][j-1] + (0.5 if mj == 0 else 1.0))
    # traceback
    i, j, ops = n, k, []
    while i > 0 or j > 0:
        if i > 0 and j > 0 and abs(D[i][j] - (D[i-1][j-1] + sub_cost(ws[i-1], ms[j-1], corr))) < 1e-9:
            ops.append(('S', ws[i-1], ms[j-1], i-1, j-1)); i -= 1; j -= 1
        elif i > 0 and abs(D[i][j] - (D[i-1][j] + (0.5 if ws[i-1] == 0 else 1.0))) < 1e-9:
            ops.append(('W', ws[i-1], None, i-1, None)); i -= 1   # Wells-only sign (M omits)
        else:
            ops.append(('M', None, ms[j-1], None, j-1)); j -= 1    # M-only sign (Wells omits)
    return D[n][k], ops[::-1]

def mclasses(w, corr):
    return corr.get(w, set())

def run(corr, null=False, seed=0, verbose=True):
    """corr: W sign -> set of M signs. Returns accepted pair records.
    null=True: match each IM77 text against a wrong site's Wells objects (chance level of near matches)."""
    results = []
    codes = ['MD', 'HP', 'LL', 'KB', 'CD', 'OS', 'WA']
    rng = random.Random(seed)
    for code in codes:
        ims = [t for t in IM if t['site_code'] == code]
        if null:
            others = [c for c in codes if c != code]
            # pool of Wells objects from other sites, same size as the home pool
            home = len(wsite(code)); pool = [w for c in others for w in wsite(c)]
            rng.shuffle(pool); ws = pool[:max(home, 200)]
        else:
            ws = wsite(code)
        # inverted index on M classes
        index = collections.defaultdict(set)
        for wi, w in enumerate(ws):
            for s in set(w['seq']):
                for m in mclasses(s, corr): index[m].add(wi)
                if s not in corr: index['U'].add(wi)
        for t in IM77_iter(ims):
            ms = [m for m in t['seq'] if m]
            if len(ms) < 2: continue
            cnt = collections.Counter()
            for m in set(ms):
                for wi in index.get(m, ()): cnt[wi] += 1
            need = 1 if len(ms) <= 3 else 2
            cands = [wi for wi, c in cnt.items() if c >= need]
            # allow unbridged-heavy Wells texts in as candidates when length is similar
            for wi in index.get('U', ()):
                if abs(len(ws[wi]['seq']) - len(ms)) <= 1 and sum(1 for s in ws[wi]['seq'] if s not in corr) >= len(ws[wi]['seq']) / 2: cands.append(wi)
            cands = set(cands)
            best = []
            for wi in cands:
                w = ws[wi]
                if abs(len(w['full']) - len(t['seq'])) > 3: continue
                c, ops = align(list(w['full']), t['seq'], corr)
                # two-line Wells texts: also try the other line order
                if w['nlines'] == 2 and '/' in w['lines']:
                    a, b = w['lines'].split('/')
                    pa, pb = parse(a), parse(b)
                    f2 = tuple(pb + pa) if tuple(pa + pb) == w['full'] else tuple(pa + pb)
                    if f2 != w['full']:
                        c2, ops2 = align(list(f2), t['seq'], corr)
                        if c2 < c: c, ops = c2, ops2; ops = [('L',) ] + ops
                best.append((c, wi, ops))
            if not best: continue
            best.sort(key=lambda z: z[0])
            c, wi, ops = best[0]
            second = best[1][0] if len(best) > 1 else 99
            L = max(len(t['seq']), len(ws[wi]['full']))
            results.append(dict(text_no=t['text_no'], site_code=code, cisi=ws[wi]['cisi'], widx=ws[wi]['idx'],
                                cost=c, second=second, L=L, ops=ops, wells=ws[wi], im=t))
    return results

def IM77_iter(ims): return ims

def accept(rec, slack=0.15):
    L = rec['L']
    return L >= 3 and rec['cost'] <= 1.0 + slack * L and rec['second'] >= rec['cost'] + 1.0

if __name__ == '__main__':
    corr = {w: set(v) for w, v in bridge.items()}
    res = run(corr)
    nul = run(corr, null=True)
    def summarize(rs, label):
        acc = [r for r in rs if accept(r)]
        exact = sum(1 for r in acc if r['cost'] == 0)
        print(f'{label}: IM77 texts with a candidate {len(rs)}; accepted {len(acc)}; cost 0: {exact}; cost>0: {len(acc)-exact}')
        byc = collections.Counter(min(int(r["cost"] * 2) / 2, 4) for r in acc)
        print('   cost histogram', sorted(byc.items()))
        return acc
    acc = summarize(res, 'HOME'); nacc = summarize(nul, 'WRONG-SITE NULL')
    # uniqueness: one Wells object claimed by several IM77 texts
    cl = collections.Counter(r['widx'] for r in acc)
    print('Wells objects claimed by >1 IM77 text:', sum(1 for v in cl.values() if v > 1))
    json.dump([dict(text_no=r['text_no'], site_code=r['site_code'], cisi=r['cisi'], widx=r['widx'], cost=r['cost'],
                    second=r['second'], L=r['L'], ops=r['ops'],
                    wells=dict(seq=list(r['wells']['full']), type=r['wells']['type'], symbol=r['wells']['symbol'],
                               material=r['wells']['material'], condition=r['wells']['condition'],
                               preservation=r['wells']['preservation'], complete=r['wells']['complete'],
                               nlines=r['wells']['nlines'], site=r['wells']['site']),
                    im=dict(seq=r['im']['seq'], doubt=r['im']['doubt'], lineid=r['im']['lineid'], nlines=r['im']['nlines'],
                            fs80=r['im']['fs80'], object_type=r['im']['object_type'], source=r['im']['source']))
               for r in res], open(OUT + 'loop24_pairs_raw.json', 'w'))
    json.dump([dict(text_no=r['text_no'], cisi=r['cisi'], cost=r['cost'], second=r['second'], L=r['L']) for r in nul],
              open(OUT + 'loop24_pairs_null.json', 'w'))
