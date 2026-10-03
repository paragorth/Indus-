"""S-DARK-31 cycle 3: compositionality of derived signs in the embedding.
Embeddings (PPMI + SVD, dim 30) on all complete Wells texts (every site; seq_raw keeps ligatures and marked forms apart),
vocabulary = every sign with >= MINTOK tokens. Derivation edges from loop 28 (data/derived/dark/loop28_edges_all.json:
strokes / enclosure / roof / doubling / ligature found blind from the glyphs, plus the 'family' proxy = same Wells 20-block and
glyph_sim_fine >= 0.6), and the two documented attached-mark families (plain fish -> marked fish, jar -> marked jar).
Tests, each against frequency-matched random pairs (2,000x):
 (a) ligature additivity: cos(E_derived, E_base + E_partner) vs cos(E_derived, E_x + E_y) for random x, y; and vs the parts alone;
 (b) operator consistency: mean pairwise cosine of the offsets E_derived - E_base within one derivation type (and sub-type),
     i.e. is 'add strokes' / 'enclose' / 'double' one direction in the space;
 (c) derived ~ base: cos(E_derived, E_base) vs random, i.e. does a derived sign behave like its base (an allograph) or not.
Usage: python3 tools/dark_loop31_c3.py [seq_raw|seq_strong|seq_all] [MINTOK]
"""
import json, sys, random, collections, math, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dark_loop31 as D
HERE = D.HERE; OUT = D.OUT
LV = sys.argv[1] if len(sys.argv) > 1 else 'seq_raw'; MINTOK = int(sys.argv[2]) if len(sys.argv) > 2 else 4
rnd = random.Random(313)
fo = open(OUT + f'loop31_c3_{LV}.txt', 'w')
def log(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); fo.write(s + '\n'); fo.flush()

objs = D.load_indus(LV, sites=None)
freq = collections.Counter(x for o in objs for x in o['seq'])
vocab = [s for s, n in freq.most_common() if n >= MINTOK]
vocab, E = D.embed(objs, vocab=vocab, dim=30)
VI = {s: i for i, s in enumerate(vocab)}
log(f'== S-DARK-31 cycle 3, level {LV}: {len(objs)} complete Wells texts (all sites), vocab {len(vocab)} signs with >= {MINTOK} tokens')

edges = json.load(open(os.path.join(HERE, 'data/derived/dark/loop28_edges_all.json')))
fam = {'plain fish -> marked fish': [(220, 233), (220, 240), (220, 235), (220, 231), (220, 226)],
       'jar -> marked jar': [(740, 741), (740, 742), (740, 745), (740, 744)]}
fbin = {s: int(math.log2(freq[s])) for s in vocab}
by_bin = collections.defaultdict(list)
for s in vocab: by_bin[fbin[s]].append(s)
def matched(a):
    pool = by_bin[fbin[a]]
    return rnd.choice(pool) if len(pool) > 1 else rnd.choice(vocab)
def matched_pair(a, b):
    for _ in range(50):
        x, y = matched(a), matched(b)
        if x != y: return x, y
    return rnd.sample(vocab, 2)
def unit(v): return v / (np.linalg.norm(v) + 1e-9)
def offcos(prs):
    O = np.array([E[VI[d]] - E[VI[b]] for b, d in prs]); On = O / (np.linalg.norm(O, axis=1, keepdims=True) + 1e-9)
    cs = On @ On.T; m = len(prs); return float((cs.sum() - m) / (m * (m - 1)))

res = {}
# ---------------- (a) ligatures ----------------
log('\n-- (a) ligature additivity: cos(derived, base + partner) vs random part pairs (2,000x)')
lig = [e for e in edges if e['type'] == 'ligature' and e['base'] in VI and e['derived'] in VI and e.get('partner') in VI]
rows = []
for e in lig:
    b, p, d = e['base'], e['partner'], e['derived']
    real = float(E[VI[d]] @ unit(E[VI[b]] + E[VI[p]]))
    cb = float(E[VI[d]] @ E[VI[b]]); cp = float(E[VI[d]] @ E[VI[p]])
    null = []
    for _ in range(2000):
        x, y = matched_pair(b, p)
        if d in (x, y): continue
        null.append(float(E[VI[d]] @ unit(E[VI[x]] + E[VI[y]])))
    P = (sum(1 for v in null if v >= real) + 1) / (len(null) + 1)
    rows.append(dict(base=b, partner=p, derived=d, n=(freq[b], freq[p], freq[d]), cos_sum=real, cos_base=cb, cos_partner=cp, null_med=float(np.median(null)), P=P, conf=e['conf']))
    log(f'  {D.fmt_sign(d)} = {D.fmt_sign(b)} + {D.fmt_sign(p)} (tokens {freq[d]}/{freq[b]}/{freq[p]}, {e["conf"]}): cos(sum) {real:.2f} vs null {np.median(null):.2f} P {P:.3f}; cos(base) {cb:.2f}, cos(partner) {cp:.2f}')
if rows:
    hits = sum(1 for r in rows if r['P'] < 0.05); beats = sum(1 for r in rows if r['cos_sum'] > max(r['cos_base'], r['cos_partner']))
    log(f'  {len(rows)} ligatures testable: {hits} with cos(sum) above the 95% null (expected {0.05*len(rows):.1f}); the sum beats both parts alone in {beats}')
res['ligatures'] = rows

# ---------------- (b) operator consistency and (c) derived ~ base ----------------
log('\n-- (b) operator consistency: mean pairwise offset cosine (derived - base) within one derivation type vs frequency-matched random pairs; (c) cos(derived, base) vs random')
groups = collections.defaultdict(list)
for e in edges:
    if e['base'] in VI and e['derived'] in VI and e['base'] != e['derived']:
        groups[e['type']].append((e['base'], e['derived']))
        sub = e.get('sub') or ''
        if e['type'] == 'strokes': sub = f"{e.get('k','?')} stroke(s)"
        groups[f"{e['type']}/{sub}"].append((e['base'], e['derived']))
for k, prs in fam.items():
    groups[k] = [(b, d) for b, d in prs if b in VI and d in VI]
rowsB = {}
for k in sorted(groups, key=lambda k: (-len(groups[k]), k)):
    prs = []
    seen = set()
    for b, d in groups[k]:
        if (b, d) not in seen: seen.add((b, d)); prs.append((b, d))
    if len(prs) < 2: continue
    real = offcos(prs); sim = float(np.mean([E[VI[d]] @ E[VI[b]] for b, d in prs]))
    null = []; nsim = []
    for _ in range(2000):
        rp = [matched_pair(b, d) for b, d in prs]
        null.append(offcos(rp)); nsim.append(float(np.mean([E[VI[y]] @ E[VI[x]] for x, y in rp])))
    P = (sum(1 for v in null if v >= real) + 1) / 2001; Ps = (sum(1 for v in nsim if v >= sim) + 1) / 2001
    rowsB[k] = dict(n=len(prs), offcos=real, null_med=float(np.median(null)), null95=float(np.percentile(null, 95)), P=P, sim=sim, sim_null=float(np.median(nsim)), P_sim=Ps)
    log(f'  {k}: n = {len(prs)} edges; offset cos {real:.2f} (null median {np.median(null):.2f}, 95% {np.percentile(null,95):.2f}, P = {P:.4f}); cos(derived, base) {sim:.2f} vs null {np.median(nsim):.2f} (P = {Ps:.4f})')
    if len(prs) <= 12:
        log('     edges: ' + ', '.join(f'{D.fmt_sign(b)}->{D.fmt_sign(d)} ({freq[b]}/{freq[d]}, cos {float(E[VI[d]]@E[VI[b]]):.2f})' for b, d in prs))
res['operators'] = rowsB

# ---------------- (d) does a 'marked' sign sit between base and the mark's class? the fish mark as a probe on other bases ----------------
log('\n-- (d) the fish-mark direction (mean of marked - plain fish) applied to other bases that have marked variants in loop 28: is the predicted NN the marked variant?')
fishprs = [(b, d) for b, d in fam['plain fish -> marked fish'] if b in VI and d in VI and d != 226]
if len(fishprs) >= 2:
    o = np.mean([E[VI[d]] - E[VI[b]] for b, d in fishprs], 0)
    hit = 0; tested = 0; ranks = []
    for k in ('strokes', 'enclosure', 'roof', 'doubling', 'jar -> marked jar'):
        for b, d in groups.get(k, []):
            if b == 220: continue
            q = unit(E[VI[b]] + o); sc = E @ q; sc[VI[b]] = -2
            order = np.argsort(-sc); rank = int(np.where(order == VI[d])[0][0]) + 1
            ranks.append(rank); tested += 1; hit += rank <= 5
    null_hit = []
    for _ in range(500):
        h = 0
        for k in ('strokes', 'enclosure', 'roof', 'doubling', 'jar -> marked jar'):
            for b, d in groups.get(k, []):
                if b == 220: continue
                x = matched(d)
                q = unit(E[VI[b]] + o); sc = E @ q; sc[VI[b]] = -2
                order = np.argsort(-sc); h += (int(np.where(order == VI[x])[0][0]) + 1) <= 5 if x != b else 0
        null_hit.append(h)
    log(f'  {tested} base->derived edges: derived sign within the top 5 of base + fish-mark in {hit} (random derived sign: median {np.median(null_hit):.0f}, 95% {np.percentile(null_hit,95):.0f}); median rank {np.median(ranks):.0f} of {len(vocab)}')
    res['fishmark_probe'] = dict(tested=tested, hit=hit, null_med=float(np.median(null_hit)), null95=float(np.percentile(null_hit, 95)), median_rank=float(np.median(ranks)))
json.dump(res, open(OUT + f'loop31_c3_{LV}.json', 'w'), default=str)
log('done')
