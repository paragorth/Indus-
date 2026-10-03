"""Loop 46 cycle 4: replication of the hands existence test on IM77 (Mahadevan 1977) variant distinctions.

Variant sets in IM77 = (a) Mahadevan's own splits of Wells merge classes (M162|M169 tree, M387|M389, M12|M15, M1|M3) and
(b) every Wells sign that the bridge maps to several Mahadevan signs (Mahadevan draws variants Wells lumps), keeping
sets with >= 2 members of >= 20 tokens each. Texts = IM77 text numbers (all lines of an object joined, sign 0 dropped).
Test as cycle 1: pairs of sets co-occurring in >= 15 texts, G = sum N x MI, null = labels permuted within
site_code x object_type (1,000x) and within site x object_type x fs80 field symbol; per-pair log-odds vs own null.
Also within-text consistency and identical-text copy agreement (cycle 3 analogue) in IM77.
Usage: python3 tools/dark_loop46_c4.py
"""
import sys, json, csv, collections, itertools
import numpy as np
sys.path.insert(0, '/home/user/Indus-/tools')
from dark_loop46_common import *

NPERM = 1000; MINPAIR = 15; MINTOK = 20
rng = np.random.default_rng(464)
lines = ['# loop 46 cycle 4: IM77 replication of the hands test']

rows = list(csv.DictReader(open(ROOT + 'data/im77/im77_corpus_lines.csv')))
texts = collections.OrderedDict()
for r in rows:
    t = texts.setdefault(r['text_no'], dict(site=r['site_code'], otype=r['object_type'], fs=r['fs80'], level=r['level'], locus=r['locus'], seq=[], doubt=[]))
    signs = [int(s) for s in r['signs_clean'].split()] if r['signs_clean'].strip() else []
    dpos = {int(x) for x in r['doubtful_positions'].split()} if r['doubtful_positions'].strip() else set()
    for k, s in enumerate(signs):
        if s != 0:
            t['seq'].append(s); t['doubt'].append((k + 1) in dpos)
T = list(texts.values())
tok = collections.Counter(s for t in T for s in t['seq'])
lines.append(f'IM77 texts {len(T)}, tokens {sum(tok.values())}')

# variant sets
bridge = json.load(open(ROOT + 'data/derived/bridge_extended.json'))
sets = {}
sets['tree 162|169 (W390|405)'] = [162, 169]
sets['387|389 (W803|806)'] = [387, 389]
sets['12|15 (W151|156)'] = [12, 15]
sets['1|3 (W90|93)'] = [1, 3]
for w, ms in bridge.items():
    if len(ms) >= 2:
        big = [m for m in ms if tok[m] >= MINTOK]
        if len(big) >= 2 and not any(set(big) == set(v) for v in sets.values()):
            sets[f'W{w}->M{"|".join(map(str, big))}'] = big
lines.append('candidate sets (M tokens): ' + '; '.join(f'{k}: ' + '/'.join(f'{m}x{tok[m]}' for m in v) for k, v in sets.items()))
sets = {k: v for k, v in sets.items() if sum(1 for m in v if tok[m] >= MINTOK) >= 2}
# head = most frequent member; label 1 = any other member
heads = {k: max(v, key=lambda m: tok[m]) for k, v in sets.items()}
m2set = {m: k for k, v in sets.items() for m in v}
labels = []; multi = []; incons = []
for i, t in enumerate(T):
    d = collections.defaultdict(list)
    for s in t['seq']:
        if s in m2set:
            d[m2set[s]].append(s)
    lab = {}
    for k, fs in d.items():
        if len(fs) >= 2:
            multi.append((i, k, fs))
            if len(set(fs)) > 1:
                incons.append((i, k, fs))
        nh = sum(1 for f in fs if f != heads[k])
        lab[k] = 1 if nh * 2 > len(fs) else (0 if nh * 2 < len(fs) else (1 if fs[0] != heads[k] else 0))
    labels.append(lab)
keys = sorted(sets)
cls_idx = {k: np.array([i for i in range(len(T)) if k in labels[i]]) for k in keys}
cls_lab = {k: np.array([labels[i][k] for i in cls_idx[k]]) for k in keys}
pairs = []
for a, b in itertools.combinations(keys, 2):
    shared = sorted(set(cls_idx[a].tolist()) & set(cls_idx[b].tolist()))
    if len(shared) >= MINPAIR:
        pa = {t: j for j, t in enumerate(cls_idx[a].tolist())}; pb = {t: j for j, t in enumerate(cls_idx[b].tolist())}
        pairs.append((a, b, np.array([pa[t] for t in shared]), np.array([pb[t] for t in shared])))
lines.append(f'texts with >= 2 sets: {sum(1 for l in labels if len(l) >= 2)}; pairs with >= {MINPAIR} shared texts: {len(pairs)}')


def stats(lab):
    g = 0.0; lo = []
    for a, b, ia, ib in pairs:
        x = lab[a][ia]; y = lab[b][ib]; g += len(x) * mi_bits(x, y); lo.append(log_odds(x, y))
    return g, np.array(lo)


G, LO = stats(cls_lab)
summary = {}
for nname, keyf in {'N1 site x object type': lambda t: (t['site'], t['otype']),
                    'N2 site x object type x field symbol': lambda t: (t['site'], t['otype'], t['fs'])}.items():
    st_codes = {k: np.array([hash(keyf(T[i])) for i in cls_idx[k]]) for k in keys}
    nullG = np.zeros(NPERM); nullLO = np.zeros((NPERM, len(pairs)))
    for p in range(NPERM):
        g, lo = stats({k: permute_within(cls_lab[k], st_codes[k], rng) for k in keys}); nullG[p] = g; nullLO[p] = lo
    P = (np.sum(nullG >= G) + 1) / (NPERM + 1); z = (G - nullG.mean()) / (nullG.std() + 1e-12)
    pp = [(np.sum(np.abs(nullLO[:, j]) >= abs(LO[j])) + 1) / (NPERM + 1) for j in range(len(pairs))]
    nsig = sum(1 for q in pp if q < 0.05)
    lines.append(f'{nname}: G={G:.2f} null {nullG.mean():.2f}+/-{nullG.std():.2f} z={z:+.2f} P={P:.3f}; pairs beyond own null: {nsig}/{len(pairs)} (exp {0.05 * len(pairs):.1f})')
    summary[nname] = dict(G=G, null=float(nullG.mean()), sd=float(nullG.std()), z=float(z), P=float(P), nsig=nsig, perpair=pp)
lines.append('pair table: A x B n n11/n10/n01/n00 LO p(N1) p(N2)')
for j, (a, b, ia, ib) in enumerate(pairs):
    x = cls_lab[a][ia]; y = cls_lab[b][ib]
    n11 = int(np.sum((x == 1) & (y == 1))); n10 = int(np.sum((x == 1) & (y == 0))); n01 = int(np.sum((x == 0) & (y == 1))); n00 = int(np.sum((x == 0) & (y == 0)))
    lines.append(f'  [{a}] x [{b}] n={len(x)} {n11}/{n10}/{n01}/{n00} LO={LO[j]:+.2f} ' + ' '.join(f"{summary[n]['perpair'][j]:.3f}" for n in summary))
# robust subset only (the three/four Mahadevan splits)
rob = [k for k in keys if k in ('tree 162|169 (W390|405)', '387|389 (W803|806)', '12|15 (W151|156)', '1|3 (W90|93)')]
rp = [(a, b, ia, ib) for (a, b, ia, ib) in pairs if a in rob and b in rob]
lines.append(f'robust (Mahadevan-split) pairs with >= {MINPAIR} shared texts: {[(a, b, len(ia)) for a, b, ia, ib in rp]}')
# within-text consistency
same = sum(1 for (_, k, fs) in multi if len(set(fs)) == 1)
lines.append(f'within-text consistency: {len(multi)} texts with >= 2 tokens of one set, same form in {same}; inconsistent: ' + '; '.join(f'{list(texts)[i]} {k} {fs}' for i, k, fs in incons[:15]))
# identical-text copies (same seq with set members replaced by heads)
norm = lambda t: tuple(heads[m2set[s]] if s in m2set else s for s in t['seq'])
groups = collections.defaultdict(list)
for i, t in enumerate(T):
    if len(t['seq']) >= 2 and labels[i]:
        groups[norm(t)].append(i)
cp = []
for v in groups.values():
    if len(v) >= 2:
        for a, b in itertools.combinations(v, 2):
            sh = sorted(set(labels[a]) & set(labels[b]))
            if sh:
                cp.append((a, b, sh))
st = {k: np.array([hash((T[i]['site'], T[i]['otype'])) for i in cls_idx[k]]) for k in keys}
pos = {k: {t: j for j, t in enumerate(cls_idx[k].tolist())} for k in keys}
def agree(lab):
    return np.mean([all(lab[k][pos[k][a]] == lab[k][pos[k][b]] for k in sh) for a, b, sh in cp]) if cp else float('nan')
obs = agree(cls_lab)
null = np.array([agree({k: permute_within(cls_lab[k], st[k], rng) for k in keys}) for _ in range(NPERM)])
lines.append(f'identical-text copy pairs (IM77, >= 2 signs, sharing a set): {len(cp)}; all forms agree {obs:.3f} vs null {null.mean():.3f}+/-{null.std():.3f} P(>=)={(np.sum(null >= obs) + 1) / (NPERM + 1):.3f}')
dis = [(a, b, sh) for a, b, sh in cp if not all(cls_lab[k][pos[k][a]] == cls_lab[k][pos[k][b]] for k in sh)]
lines.append('disagreeing copies: ' + '; '.join(f"{list(texts)[a]}({T[a]['site']}/{T[a]['otype']}) {'-'.join(map(str, T[a]['seq']))} vs {list(texts)[b]}({T[b]['site']}/{T[b]['otype']}) {'-'.join(map(str, T[b]['seq']))}" for a, b, sh in dis[:30]))
summary['copies'] = dict(n=len(cp), obs=float(obs), null=float(null.mean()), sd=float(null.std()))
json.dump(summary, open(OUT + 'loop46_cycle4.json', 'w'), indent=1, default=float)
open(OUT + 'loop46_cycle4_log.txt', 'w').write('\n'.join(lines) + '\n')
print('\n'.join(lines))

# ---- follow-up: are the significant pairs carried by repeated formulae? by-site breakdown, text listing, dedup rerun
lines.append('\n## follow-up: by-site breakdown and deduplication (one text per distinct normalised sequence per site)')
for (a, b, ia, ib) in pairs:
    x = cls_lab[a][ia]; y = cls_lab[b][ib]
    if abs(log_odds(x, y)) < 1.5:
        continue
    shared = [cls_idx[a][j] for j in ia]
    bysite = collections.Counter((T[i]['site'], T[i]['otype'], labels[i][a], labels[i][b]) for i in shared)
    lines.append(f'[{a}] x [{b}]: ' + '; '.join(f'{s}/{o} a={la} b={lb} x{n}' for (s, o, la, lb), n in sorted(bysite.items())))
    if len(shared) <= 60:
        lines.append('   texts: ' + '; '.join(f"{list(texts)[i]}({T[i]['site']}/{T[i]['otype'][:4]}) {'-'.join(map(str, T[i]['seq']))}" for i in shared))
# dedup: keep the first text of each (normalised sequence, site)
seen = set(); keep = []
for i, t in enumerate(T):
    key = (norm(t), t['site'])
    if key in seen:
        continue
    seen.add(key); keep.append(i)
keepset = set(keep)
cls_idx_d = {k: np.array([i for i in cls_idx[k] if i in keepset]) for k in keys}
cls_lab_d = {k: np.array([labels[i][k] for i in cls_idx_d[k]]) for k in keys}
pairs_d = []
for a, b in itertools.combinations(keys, 2):
    shared = sorted(set(cls_idx_d[a].tolist()) & set(cls_idx_d[b].tolist()))
    if len(shared) >= MINPAIR:
        pa = {t: j for j, t in enumerate(cls_idx_d[a].tolist())}; pb = {t: j for j, t in enumerate(cls_idx_d[b].tolist())}
        pairs_d.append((a, b, np.array([pa[t] for t in shared]), np.array([pb[t] for t in shared])))
def stats_d(lab):
    g = 0.0; lo = []
    for a, b, ia, ib in pairs_d:
        x = lab[a][ia]; y = lab[b][ib]; g += len(x) * mi_bits(x, y); lo.append(log_odds(x, y))
    return g, np.array(lo)
Gd, LOd = stats_d(cls_lab_d)
st_d = {k: np.array([hash((T[i]['site'], T[i]['otype'])) for i in cls_idx_d[k]]) for k in keys}
nullG = np.zeros(NPERM); nullLO = np.zeros((NPERM, len(pairs_d)))
for p in range(NPERM):
    g, lo = stats_d({k: permute_within(cls_lab_d[k], st_d[k], rng) for k in keys}); nullG[p] = g; nullLO[p] = lo
P = (np.sum(nullG >= Gd) + 1) / (NPERM + 1); z = (Gd - nullG.mean()) / (nullG.std() + 1e-12)
pp = [(np.sum(np.abs(nullLO[:, j]) >= abs(LOd[j])) + 1) / (NPERM + 1) for j in range(len(pairs_d))]
lines.append(f'DEDUP ({len(keep)} of {len(T)} texts kept): pairs {len(pairs_d)}; G={Gd:.2f} null {nullG.mean():.2f}+/-{nullG.std():.2f} z={z:+.2f} P={P:.3f}; pairs beyond own null {sum(1 for q in pp if q < 0.05)}/{len(pairs_d)}')
for j, (a, b, ia, ib) in enumerate(pairs_d):
    x = cls_lab_d[a][ia]; y = cls_lab_d[b][ib]
    n11 = int(np.sum((x == 1) & (y == 1))); n10 = int(np.sum((x == 1) & (y == 0))); n01 = int(np.sum((x == 0) & (y == 1))); n00 = int(np.sum((x == 0) & (y == 0)))
    lines.append(f'  [{a}] x [{b}] n={len(x)} {n11}/{n10}/{n01}/{n00} LO={LOd[j]:+.2f} p={pp[j]:.3f}')
summary['dedup'] = dict(G=Gd, null=float(nullG.mean()), z=float(z), P=float(P))
json.dump(summary, open(OUT + 'loop46_cycle4.json', 'w'), indent=1, default=float)
open(OUT + 'loop46_cycle4_log.txt', 'w').write('\n'.join(lines) + '\n')
print('\n'.join(lines[lines.index('\n## follow-up: by-site breakdown and deduplication (one text per distinct normalised sequence per site)'):]))
