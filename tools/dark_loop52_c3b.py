"""Loop 52, cycle 3b: repairs to the two outside tests of cycle 3.
(b') Testability of the Mahadevan lumps: a lump can only be found by E1/E2 if both members have >= 8 tokens.
(c') The 324 'certainly new' texts contain 0 unseen M signs because loop 21 sends any text with an unbridged sign to
     'indeterminate'. Unbiased version: unbridged-M-sign token rate on IM77 objects Wells read too (overlap: a gap in the
     bridge, not a new sign) vs on the objects Wells never read (new + indeterminate + strict-without-match), by site.
     Also: M signs seen only on non-overlap objects, and the Good-Turing expectation at that token count.
Writes data/derived/dark/loop52_cycle3b.txt.
"""
import json, collections, csv
import numpy as np

ROOT = '/home/user/Indus-/'
OUT = ROOT + 'data/derived/dark/'
corpus = json.load(open(ROOT + 'data/derived/merged-corpus-canonical.json'))
tok = collections.Counter(s for r in corpus for s in r['seq_raw'])
bridge = json.load(open(ROOT + 'data/derived/bridge_extended.json'))
props = json.load(open(OUT + 'bridge_proposals.json'))['proposals']
sets27 = json.load(open(OUT + 'loop27_sets.json'))
merges = json.load(open(OUT + 'loop52_merges.json'))
out = ['\n## (b\') testability of the lumps']
lumps = collections.defaultdict(set)
for w, ms in bridge.items():
    for mm in ms: lumps[mm].add(int(w))
for p in props: lumps[p['M']].add(p['W'])
lumps = {mm: {w for w in ws if tok[w] > 0} for mm, ws in lumps.items()}
lumps = {mm: ws for mm, ws in lumps.items() if len(ws) >= 2}
testable = {mm: ws for mm, ws in lumps.items() if sum(1 for w in ws if tok[w] >= 8) >= 2}
pairs = {tuple(sorted((r['a'], r['b']))): r for r in merges['pairs']}


def principled_noE4(r):
    e = r['ev']; return e['E3'] and (e['E1'] or e['E2']) and not r['veto']


found = {mm: ws for mm, ws in testable.items() if any(principled_noE4(pairs[tuple(sorted((a, b)))]) for a in ws for b in ws if a < b and tuple(sorted((a, b))) in pairs)}
e3only = {mm: ws for mm, ws in testable.items() if mm not in found and any(pairs[tuple(sorted((a, b)))]['ev']['E3'] for a in ws for b in ws if a < b and tuple(sorted((a, b))) in pairs)}
out.append('  lumps %d; testable (>= 2 members with >= 8 tokens) %d; found by the graph without E4 (E3 & (E1|E2)) %d; shape-linked but no usage witness %d; untestable (rare) %d' % (
    len(lumps), len(testable), len(found), len(e3only), len(lumps) - len(testable)))
out.append('  testable but not found: ' + '; '.join('M%d=%s' % (mm, sorted(ws)) for mm, ws in sorted(testable.items()) if mm not in found))

# (c')
rows = list(csv.DictReader(open(ROOT + 'data/im77/im77_corpus_lines.csv')))
texts = collections.defaultdict(list); site = {}
for r in rows:
    k = (r['text_no'], r['side']); texts[k].extend(int(s) for s in r['signs_clean'].split() if s != '0'); site[k] = r['site_code']
overlap = {tuple(k) for k in sets27['overlap']}
covered = set()
for w, ms in bridge.items(): covered |= set(ms)
for p in props: covered.add(p['M'])
out.append('\n## (c\') unbridged Mahadevan signs (no Wells counterpart in bridge + proposals, %d M signs covered): overlap objects vs objects Wells never read' % len(covered))
grp = {'overlap (Wells read the same object)': [], 'not overlap (Wells never matched it)': []}
for k, t in texts.items():
    if not t: continue
    grp['overlap (Wells read the same object)' if k in overlap else 'not overlap (Wells never matched it)'].append((k, t))
seen_types = {}
for g, items in grp.items():
    ntok = sum(len(t) for k, t in items); un = sum(1 for k, t in items for s in t if s not in covered)
    types = set(s for k, t in items for s in t); untypes = {s for s in types if s not in covered}
    seen_types[g] = types
    out.append('  %s: %d texts, %d tokens, %d M types; unbridged tokens %d (%.1f%%), unbridged types %d' % (g, len(items), ntok, len(types), un, 100 * un / ntok, len(untypes)))
ov, no = seen_types['overlap (Wells read the same object)'], seen_types['not overlap (Wells never matched it)']
only_new = no - ov
items = grp['not overlap (Wells never matched it)']
ntok = sum(len(t) for k, t in items)
cnt_new = collections.Counter(s for k, t in items for s in t if s in only_new)
out.append('  M types seen only on non-overlap objects: %d (%d tokens of %d = %.1f%%); of them unbridged %d' % (len(only_new), sum(cnt_new.values()), ntok, 100 * sum(cnt_new.values()) / ntok, len(only_new - covered)))
# Good-Turing from the overlap sample: f1/N
ovitems = grp['overlap (Wells read the same object)']
c = collections.Counter(s for k, t in ovitems for s in t); N = sum(c.values()); f1 = sum(1 for v in c.values() if v == 1)
out.append('  Good-Turing from the overlap sample: f1/N = %d/%d = %.3f -> %.0f new-sign tokens expected in %d tokens; observed %d (%.1f%%). S309 predicted 1.7%% -> %.0f' % (
    f1, N, f1 / N, f1 / N * ntok, ntok, sum(cnt_new.values()), 100 * sum(cnt_new.values()) / ntok, 0.017 * ntok))
# control: the same split sizes drawn at random from the whole IM77 corpus (texts shuffled; test = %d tokens, train = the rest)
allitems = [(k, t) for k, t in texts.items() if t]
rng = np.random.default_rng(7); hold = []
keys = list(range(len(allitems)))
for _ in range(300):
    rng.shuffle(keys); cut = []; n = 0
    for i in keys:
        if n >= ntok: break
        cut.append(i); n += len(allitems[i][1])
    cs = set(cut); tr = set(s for i, (k, t) in enumerate(allitems) if i not in cs for s in t)
    te = [s for i in cut for s in allitems[i][1]]
    hold.append((sum(1 for s in te if s not in tr), len({s for s in te if s not in tr})))
h = np.array(hold)
out.append('  control, a random %d-token set of IM77 texts against the remaining %d tokens: %.1f +/- %.1f new tokens (%.1f%%), %.1f +/- %.1f new types; observed 265 / 123 is z = %.1f / %.1f' % (
    ntok, N, h[:, 0].mean(), h[:, 0].std(), 100 * h[:, 0].mean() / ntok, h[:, 1].mean(), h[:, 1].std(), (sum(cnt_new.values()) - h[:, 0].mean()) / h[:, 0].std(), (len(only_new) - h[:, 1].mean()) / h[:, 1].std()))
# site-stratified control: same number of texts per site
by_s = collections.defaultdict(list)
for k, t in allitems: by_s[site[k]].append((k, t))
need = collections.Counter(site[k] for k, t in items); hold2 = []
for _ in range(300):
    cut = []
    for sc, n in need.items():
        pool = by_s[sc]; idx = rng.permutation(len(pool))[:n]; cut.extend(pool[i] for i in idx)
    ck = {k for k, t in cut}; tr = set(s for k, t in allitems if k not in ck for s in t)
    te = [s for k, t in cut for s in t]
    hold2.append((sum(1 for s in te if s not in tr), len({s for s in te if s not in tr}), len(te)))
h2 = np.array(hold2)
out.append('  site-stratified control (same text count per site): %.1f +/- %.1f new tokens of %.0f (%.1f%%), %.1f +/- %.1f new types; observed z = %.1f / %.1f' % (
    h2[:, 0].mean(), h2[:, 0].std(), h2[:, 2].mean(), 100 * h2[:, 0].mean() / h2[:, 2].mean(), h2[:, 1].mean(), h2[:, 1].std(), (sum(cnt_new.values()) - h2[:, 0].mean()) / h2[:, 0].std(), (len(only_new) - h2[:, 1].mean()) / h2[:, 1].std()))
by_site = collections.defaultdict(lambda: [0, 0])
for k, t in items:
    by_site[site[k]][0] += len(t); by_site[site[k]][1] += sum(1 for s in t if s in only_new)
out.append('  by site (non-overlap tokens, new-sign tokens): ' + ', '.join('%s %d/%d' % (s, v[1], v[0]) for s, v in sorted(by_site.items(), key=lambda x: -x[1][0])))
out.append('  the new-only M signs: ' + ', '.join('M%d x%d' % (s, n) for s, n in cnt_new.most_common()))
open(OUT + 'loop52_cycle3b.txt', 'w').write('\n'.join(out) + '\n')
print('\n'.join(out))
