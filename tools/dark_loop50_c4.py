"""S-DARK-50 cycle 4: replication on IM77 through the bridge, and the S-DARK-27 rare-sign prediction.

(a) IM77 texts (data/im77/im77_corpus_lines.csv, lines joined in reading order, sign 0 = break -> text dropped,
    >= 2 signs) parsed with the same frame parser as cycle 2, with every Wells class set mapped to Mahadevan
    numbers through data/derived/bridge_extended.json (+ the S-DARK-27 proposals for classes the bridge lacks).
    Each Mahadevan sign gets its modal role; its shape index is the cycle-1 index of the Wells sign(s) bridged to it
    (mean when several).  Same frequency-matched permutation (1,000x) as cycle 2; run with bridge_extended only
    and with bridge_extended + proposals.
(b) Prediction: the S-DARK-27 newly bridged signs (data/derived/dark/bridge_proposals.json) are rare in Wells
    (no glyph-based test was ever run on them).  For each proposal, the IM77 position profile of its M sign
    (share of tokens text-final before an optional suffix = head position; share in the middle) and the Wells glyph
    shape index.  The arrow predicts: proposals that live in the middle are abstract (index below the corpus
    median), those in head position pictorial.  Counts with rule-of-three upper bounds.
Usage: python3 tools/dark_loop50_c4.py [NPERM]
"""
import sys, json, csv, math, collections
import numpy as np

NP = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
OUT = 'data/derived/dark/'
np.random.seed(50)
S = {m['w']: m for m in json.load(open(OUT + 'loop50_signs.json'))}
base = [m for m in S.values() if not m['numeral']]
for k in ('curv', 'asym'):
    mu = np.mean([m[k] for m in base]); sd = np.std([m[k] for m in base]) + 1e-9
    for m in S.values(): m['z_' + k] = (m[k] - mu) / sd
for m in S.values(): m['pict_curv'] = (m['z_curv'] + m['z_asym']) / 2
BR = {int(k): v for k, v in json.load(open('data/derived/bridge_extended.json')).items()}
PROPS = json.load(open(OUT + 'bridge_proposals.json'))['proposals']
rep = []
def P(*a):
    s = ' '.join(str(x) for x in a); print(s); rep.append(s)
P('# LOOP 50 cycle 4: IM77 replication through the bridge, and the S-DARK-27 rare-sign prediction')

rows = list(csv.DictReader(open('data/im77/im77_corpus_lines.csv')))
by = collections.defaultdict(list)
for r in rows: by[(r['text_no'], r['side'])].append(r)
TXT = []
for k, ls in by.items():
    ls.sort(key=lambda r: int(r['line'] or 0)); seq = []; bad = False
    for r in ls:
        for x in r['signs_clean'].split():
            if not x.isdigit(): continue
            if x == '0': bad = True
            else: seq.append(int(x))
    if bad or len(seq) < 2: continue
    TXT.append(seq)
P(f'IM77 texts used: {len(TXT)} (sign 0 dropped texts excluded, >= 2 signs)')

def inv_bridge(with_props):
    inv = collections.defaultdict(set)
    for w, ms in BR.items():
        for m in ms: inv[m].add(w)
    if with_props:
        for p in PROPS:
            inv[p['M']].add(p['W'])
    return inv

def mapset(ws, inv_w2m):
    out = set()
    for w in ws: out |= set(inv_w2m.get(w, []))
    return out

def run(with_props, tag):
    w2m = {w: set(ms) for w, ms in BR.items()}
    if with_props:
        for p in PROPS: w2m.setdefault(p['W'], set()).add(p['M'])
    m2w = inv_bridge(with_props)
    OPEN = mapset({817, 861, 820, 920, 692}, w2m); MARK = mapset({2, 60}, w2m); MJAR = mapset({741, 742, 745}, w2m)
    SUF = mapset({400, 90}, w2m); CL = list(mapset({740, 520, 151, 156, 527, 226, 617, 154, 158, 236, 700}, w2m))
    FISH = mapset({235, 240, 233, 231, 220}, w2m); NUM = mapset({1, 3, 4, 5, 16, 17, 18, 31, 32, 33, 34, 55, 56}, w2m)
    W920 = mapset({920}, w2m); W520 = mapset({520}, w2m); W740 = mapset({740}, w2m); W33 = mapset({33}, w2m); W705 = mapset({705, 706}, w2m); W100 = mapset({100}, w2m)
    P(f'\n== {tag}: classes in M numbers: OPEN {sorted(OPEN)} MARK {sorted(MARK)} SUF {sorted(SUF)} CL {sorted(CL)} NUM {len(NUM)} FISH {sorted(FISH)}')
    left = collections.defaultdict(collections.Counter)
    for s in TXT:
        s = s[:]
        while len(s) > 1 and s[-1] in SUF: s.pop()
        if len(s) >= 2 and s[-1] in CL: left[s[-1]][s[-2]] += 1
    QUAL = {}
    for c, cnt in left.items():
        tot = sum(cnt.values()); acc = 0; q = set()
        for a, n in cnt.most_common():
            if acc / tot >= 0.6: break
            q.add(a); acc += n
        QUAL[c] = q
    def parse(s):
        lab = ['NAME'] * len(s); i = 0; j = len(s)
        if s[0] in OPEN:
            lab[0] = 'OPENER'; i = 1
            if len(s) > 1 and s[1] in MARK:
                lab[1] = 'MARKER'; i = 2
                if s[0] in W920 and len(s) > 2 and s[2] in MJAR: lab[2] = 'MARKER'; i = 3
        while j - 1 > i and s[j - 1] in SUF and j >= 2 and (s[j - 2] in CL or s[j - 2] in SUF): lab[j - 1] = 'SUFFIX'; j -= 1
        if j - 1 >= i and s[j - 1] in CL:
            c = s[j - 1]; lab[j - 1] = 'CLOSER'; j -= 1
            if c in W520:
                if j - 2 >= i and s[j - 1] in W33 and s[j - 2] in W705: lab[j - 1] = lab[j - 2] = 'TITLE'; j -= 2
                while j - 1 >= i and s[j - 1] in FISH: lab[j - 1] = 'TITLE'; j -= 1
            elif c in W740:
                if j - 1 >= i and s[j - 1] in W100: lab[j - 1] = 'TITLE'; j -= 1
                if j - 1 >= i and s[j - 1] in QUAL.get(c, ()): lab[j - 1] = 'TITLE'; j -= 1
            elif j - 1 >= i and s[j - 1] in QUAL.get(c, ()): lab[j - 1] = 'TITLE'; j -= 1
            if j - 1 >= i and s[j - 1] in NUM and lab[j] == 'TITLE': lab[j - 1] = 'TITLE'; j -= 1
        for k in range(i, j - 1):
            if s[k] in NUM and lab[k] == 'NAME' and lab[k + 1] == 'NAME': lab[k] = lab[k + 1] = 'COUNT'
        for k in range(i, j):
            if s[k] in NUM and lab[k] == 'NAME': lab[k] = 'COUNT'
        if 'CLOSER' in lab:
            jc = lab.index('CLOSER'); k = jc - 1
            while k >= 0 and lab[k] == 'TITLE': k -= 1
            if k >= 0 and lab[k] == 'NAME': lab[k] = 'NAME_ADJ'
        return ['NAME_FAR' if l == 'NAME' else l for l in lab]
    RT = collections.defaultdict(collections.Counter)
    for s in TXT:
        for a, l in zip(s, parse(s)): RT[a][l] += 1
    ROLES = ['OPENER', 'MARKER', 'CLOSER', 'TITLE', 'NAME_ADJ', 'NAME_FAR', 'COUNT', 'SUFFIX']
    recs = []
    for m, cnt in RT.items():
        n = sum(cnt.values())
        if n < 5: continue
        ws = [w for w in m2w.get(m, []) if w in S]
        if not ws: continue
        role, k = cnt.most_common(1)[0]; modal = role if k / n >= 0.5 else 'MIXED'
        recs.append(dict(m=m, n=n, modal=modal, ws=ws, numeral=all(S[w]['numeral'] for w in ws),
                         pict_shape=float(np.mean([S[w]['pict_shape'] for w in ws])), pict_curv=float(np.mean([S[w]['pict_curv'] for w in ws])),
                         perim=float(np.mean([S[w]['perim'] for w in ws]))))
    P(f'  Mahadevan signs with >= 5 tokens and a bridged glyph: {len(recs)} of {sum(1 for c in RT.values() if sum(c.values())>=5)}; modal roles {dict(collections.Counter(r["modal"] for r in recs))}')
    NN = [r for r in recs if not r['numeral']]
    for meas in ('pict_shape', 'pict_curv', 'perim'):
        vals = np.array([r[meas] for r in NN]); roles = np.array([r['modal'] for r in NN]); fb = np.array([int(math.log2(r['n'])) for r in NN])
        groups = [g for g in ROLES + ['MIXED'] if (roles == g).any()]
        obs = {g: vals[roles == g].mean() for g in groups}; nulls = {g: [] for g in groups}
        bins = {b: np.where(fb == b)[0] for b in set(fb)}
        for _ in range(NP):
            pr = roles.copy()
            for b, idx in bins.items(): pr[idx] = roles[np.random.permutation(idx)]
            for g in groups: nulls[g].append(vals[pr == g].mean())
        P(f'  measure {meas} (non-numeral n={len(NN)}):')
        for g in groups:
            nl = np.array(nulls[g]); p = (np.sum(np.abs(nl - nl.mean()) >= abs(obs[g] - nl.mean())) + 1) / (NP + 1)
            P(f'     {g:9s} n={int((roles==g).sum()):3d} mean {obs[g]:+.3f} null [{np.percentile(nl,2.5):+.3f},{np.percentile(nl,97.5):+.3f}] P {p:.3f}')
        if 'CLOSER' in groups and 'NAME_FAR' in groups:
            d = obs['CLOSER'] - obs['NAME_FAR']; dn = np.array(nulls['CLOSER']) - np.array(nulls['NAME_FAR'])
            p = (np.sum(np.abs(dn - dn.mean()) >= abs(d - dn.mean())) + 1) / (NP + 1)
            P(f'     head - far middle {d:+.3f}, null [{np.percentile(dn,2.5):+.3f},{np.percentile(dn,97.5):+.3f}], P {p:.3f}')
            # without the W617 / M245 outlier
            sel = [r for r in NN if 617 not in r['ws']]
            v2 = np.array([r[meas] for r in sel]); r2 = np.array([r['modal'] for r in sel])
            P(f'     without W617/M245: head mean {v2[r2=="CLOSER"].mean():+.3f} (n={int((r2=="CLOSER").sum())}) vs far middle {v2[r2=="NAME_FAR"].mean():+.3f}')
    hd = sorted([r for r in NN if r['modal'] == 'CLOSER'], key=lambda r: r['pict_shape'])
    P('  heads: ' + ', '.join(f'M{r["m"]}(W{"/".join(map(str,r["ws"]))}):{r["pict_shape"]:+.1f}' for r in hd))
    return RT, SUF, CL

RT, SUF, CL = run(False, 'bridge_extended only')
run(True, 'bridge_extended + S-DARK-27 proposals')

# ---------------------------------------------------------------- (b) prediction on the S-DARK-27 proposals
P('\n== (b) S-DARK-27 proposals: position of the M sign in IM77 vs the Wells glyph shape index')
pos = collections.defaultdict(lambda: collections.Counter())
for s in TXT:
    t = s[:]
    while len(t) > 1 and t[-1] in SUF: t.pop()
    for i, a in enumerate(t):
        pos[a]['final' if i == len(t) - 1 else ('first' if i == 0 else 'middle')] += 1
median = float(np.median([m['pict_shape'] for m in base]))
P(f'  corpus median shape index (non-numeral Wells signs) {median:+.3f}; proposals {len(PROPS)}')
tab = collections.Counter(); lines = []
for p in PROPS:
    w, m = p['W'], p['M']
    if w not in S: continue
    c = pos.get(m); n = sum(c.values()) if c else 0
    if n == 0: continue
    ps = S[w]['pict_shape']; cls = 'pictorial' if ps > median else 'abstract'
    slot = 'head' if c['final'] / n >= 0.5 and n >= 3 else ('middle' if (c['middle'] + c['first']) / n >= 0.5 else 'mixed')
    tab[(slot, cls)] += 1
    lines.append((slot, cls, w, m, n, c['final'], c['middle'], c['first'], ps, S[w]['numeral']))
P('  cross-table (slot of the M sign in IM77 x glyph class of its Wells partner):')
for slot in ('head', 'middle', 'mixed'):
    a, b = tab[(slot, 'pictorial')], tab[(slot, 'abstract')]
    P(f'     {slot:7s} pictorial {a:3d}  abstract {b:3d}  share pictorial {a/max(a+b,1):.2f}')
mid = [l for l in lines if l[0] == 'middle']; hed = [l for l in lines if l[0] == 'head']
nm = len(mid); na = sum(1 for l in mid if l[1] == 'abstract')
P(f'  prediction "middle -> abstract": {na}/{nm} = {na/max(nm,1):.2f} (binomial vs 0.5: two-sided P {2*min(sum(math.comb(nm,k) for k in range(na,nm+1)), sum(math.comb(nm,k) for k in range(0,na+1)))/2**nm if nm else float("nan"):.3f})')
P(f'  prediction "head -> pictorial": {sum(1 for l in hed if l[1]=="pictorial")}/{len(hed)}; heads among proposals are rare (upper bound on head-slot proposals < 3/{len(lines)} if 0)')
nonnum_mid = [l for l in mid if not l[9]]
P(f'  middle proposals excluding stroke numerals: abstract {sum(1 for l in nonnum_mid if l[1]=="abstract")}/{len(nonnum_mid)}')
P('  head-slot proposals: ' + '; '.join(f'W{l[2]}->M{l[3]} n={l[4]} final {l[5]} idx {l[8]:+.1f}' for l in hed))
P('  10 most abstract middle proposals: ' + '; '.join(f'W{l[2]}->M{l[3]}:{l[8]:+.1f}' for l in sorted(mid, key=lambda l: l[8])[:10]))
P('  10 most pictorial middle proposals: ' + '; '.join(f'W{l[2]}->M{l[3]}:{l[8]:+.1f}' for l in sorted(mid, key=lambda l: -l[8])[:10]))
# comparison: Wells-attested rare middle signs (5-10 tokens, NAME_FAR modal) share above median
R = json.load(open(OUT + 'loop50_roles_seq_raw.json'))
rare_mid = [r for r in R if r['modal'] == 'NAME_FAR' and 5 <= r['n'] <= 10 and not r['numeral']]
P(f'  reference: Wells rare middle signs (5-10 tokens, NAME_FAR): abstract {sum(1 for r in rare_mid if r["pict_shape"]<=median)}/{len(rare_mid)}')
open(OUT + 'loop50_cycle4.txt', 'w').write('\n'.join(rep))
