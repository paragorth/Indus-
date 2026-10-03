"""Loop 46 cycle 5: is the variant form set by the neighbouring sign (contextual allograph) rather than by a carver?

For each usable class (Wells, level all; and the Mahadevan sets of cycle 4): MI(form, previous sign) and MI(form, next sign)
at the merged (seq_all) level vs form labels permuted within site x type (1,000x); the top contexts listed.
Then the cycle-1 / cycle-4 pair test rerun with a CONTEXT null: each class's labels permuted within
site x type x (previous sign, next sign) of the token, so a form bound to its neighbours cannot produce co-selection.
If the 803|806 x 390|405 pair vanishes under the context null, it is a spelling rule of the formula, not a hand.
Usage: python3 tools/dark_loop46_c5.py
"""
import sys, json, csv, collections, itertools
import numpy as np
sys.path.insert(0, '/home/user/Indus-/tools')
from dark_loop46_common import *

NPERM = 1000
rng = np.random.default_rng(465)
lines = ['# loop 46 cycle 5: contextual allographs vs hands']
summary = {}


def mi_cat(a, b):
    n = len(a); c = collections.Counter(zip(a, b)); ca = collections.Counter(a); cb = collections.Counter(b)
    return sum(v / n * np.log2(v * n / (ca[x] * cb[y])) for (x, y), v in c.items())


def context_tests(name, toks, strat_of, MINPAIR):
    """toks: list of dicts per token: text, cls, form(0/1), prev, next (merged signs, 'B' boundary)."""
    out = [f'\n## {name}']
    classes = sorted({t['cls'] for t in toks})
    # per class: MI with prev and next
    for c in classes:
        tt = [t for t in toks if t['cls'] == c]
        if len(tt) < 30:
            continue
        f = np.array([t['form'] for t in tt]); st = np.array([hash(strat_of(t)) for t in tt])
        for side in ('prev', 'next'):
            ctx = [str(t[side]) for t in tt]
            obs = mi_cat(f.tolist(), ctx)
            null = np.array([mi_cat(permute_within(f, st, rng).tolist(), ctx) for _ in range(300)])
            P = (np.sum(null >= obs) + 1) / 301
            # top contexts by non-head share
            tab = collections.defaultdict(lambda: [0, 0])
            for t in tt:
                tab[str(t[side])][t['form']] += 1
            top = sorted(((k, v) for k, v in tab.items() if sum(v) >= 8), key=lambda kv: -sum(kv[1]))[:8]
            out.append(f'  {c} form x {side}: MI {obs:.3f} vs null {null.mean():.3f}+/-{null.std():.3f} P={P:.3f}; ' +
                       ' '.join(f'{k}:{v[1]}/{sum(v)}' for k, v in top))
            summary[f'{name}|{c}|{side}'] = dict(MI=obs, null=float(null.mean()), P=float(P))
    # pair test with context null: text-level labels (first token of the class in the text)
    bytext = collections.defaultdict(dict)
    for t in toks:
        if t['cls'] not in bytext[t['text']]:
            bytext[t['text']][t['cls']] = t
    texts = sorted(bytext)
    cls_idx = {c: [x for x in texts if c in bytext[x]] for c in classes}
    cls_lab = {c: np.array([bytext[x][c]['form'] for x in cls_idx[c]]) for c in classes}
    pairs = []
    for a, b in itertools.combinations(classes, 2):
        shared = sorted(set(cls_idx[a]) & set(cls_idx[b]))
        if len(shared) >= MINPAIR:
            pa = {x: j for j, x in enumerate(cls_idx[a])}; pb = {x: j for j, x in enumerate(cls_idx[b])}
            pairs.append((a, b, np.array([pa[x] for x in shared]), np.array([pb[x] for x in shared])))

    def stats(lab):
        g = 0.0; lo = []
        for a, b, ia, ib in pairs:
            x = lab[a][ia]; y = lab[b][ib]; g += len(x) * mi_bits(x, y); lo.append(log_odds(x, y))
        return g, np.array(lo)
    G, LO = stats(cls_lab)
    for nname, keyf in {'site x type': lambda t: strat_of(t),
                        'site x type x prev': lambda t: (strat_of(t), t['prev']),
                        'site x type x next': lambda t: (strat_of(t), t['next']),
                        'site x type x prev x next': lambda t: (strat_of(t), t['prev'], t['next'])}.items():
        st = {c: np.array([hash(keyf(bytext[x][c])) for x in cls_idx[c]]) for c in classes}
        nullG = np.zeros(NPERM); nullLO = np.zeros((NPERM, len(pairs)))
        for p in range(NPERM):
            g, lo = stats({c: permute_within(cls_lab[c], st[c], rng) for c in classes}); nullG[p] = g; nullLO[p] = lo
        P = (np.sum(nullG >= G) + 1) / (NPERM + 1); z = (G - nullG.mean()) / (nullG.std() + 1e-12)
        pp = [(np.sum(np.abs(nullLO[:, j]) >= abs(LO[j])) + 1) / (NPERM + 1) for j in range(len(pairs))]
        # number of free (movable) labels per class under this null
        free = {c: int(sum(n for n in collections.Counter(st[c].tolist()).values() if n > 1)) for c in classes}
        out.append(f'  pair test, null {nname}: G={G:.2f} null {nullG.mean():.2f}+/-{nullG.std():.2f} z={z:+.2f} P={P:.3f}; '
                   f'pairs beyond own null {sum(1 for q in pp if q < 0.05)}/{len(pairs)}; per pair: ' +
                   ' '.join(f'{a}x{b}:{q:.3f}' for (a, b, _, _), q in zip(pairs, pp)) + f'; labels movable {free}')
        summary[f'{name}|pairs|{nname}'] = dict(G=G, null=float(nullG.mean()), z=float(z), P=float(P), perpair={f'{a}x{b}': q for (a, b, _, _), q in zip(pairs, pp)})
    lines.extend(out)


# ---- Wells
corpus = load_corpus()
classes = load_classes('all'); use = usable_classes(corpus, classes)
form2head = {f: h for h, fs in classes.items() for f in fs if h in use}
toks = []
for i, r in enumerate(corpus):
    raw = r['seq_raw']; mer = r['seq_all']
    if len(raw) != len(mer):
        continue
    for k, s in enumerate(raw):
        if s in form2head:
            h = form2head[s]
            toks.append(dict(text=i, cls=f'W{h}', form=int(s != h), prev=mer[k - 1] if k > 0 else 'B', next=mer[k + 1] if k + 1 < len(mer) else 'B'))
context_tests('Wells seq_all classes (form 1 = non-head)', toks, lambda t: (corpus[t['text']]['site'], corpus[t['text']]['type']), 20)

# ---- IM77
rows = list(csv.DictReader(open(ROOT + 'data/im77/im77_corpus_lines.csv')))
texts = collections.OrderedDict()
for r in rows:
    t = texts.setdefault(r['text_no'], dict(site=r['site_code'], otype=r['object_type'], seq=[]))
    for s in (r['signs_clean'].split() if r['signs_clean'].strip() else []):
        if int(s) != 0:
            t['seq'].append(int(s))
T = list(texts.values())
sets = {'M162|169': [162, 169], 'M387|389': [387, 389], 'M12|15': [12, 15], 'M99|100': [99, 100], 'M97|98': [97, 98], 'M102|103': [102, 103], 'M347|358': [347, 358], 'M261|373': [261, 373]}
tok = collections.Counter(s for t in T for s in t['seq'])
heads = {k: max(v, key=lambda m: tok[m]) for k, v in sets.items()}
m2set = {m: k for k, v in sets.items() for m in v}
toks = []
for i, t in enumerate(T):
    mer = [heads[m2set[s]] if s in m2set else s for s in t['seq']]
    for k, s in enumerate(t['seq']):
        if s in m2set:
            toks.append(dict(text=i, cls=m2set[s], form=int(s != heads[m2set[s]]), prev=mer[k - 1] if k > 0 else 'B', next=mer[k + 1] if k + 1 < len(mer) else 'B'))
context_tests('IM77 Mahadevan sets (form 1 = rarer member)', toks, lambda t: (T[t['text']]['site'], T[t['text']]['otype']), 15)

json.dump(summary, open(OUT + 'loop46_cycle5.json', 'w'), indent=1, default=float)
open(OUT + 'loop46_cycle5_log.txt', 'w').write('\n'.join(lines) + '\n')
print('\n'.join(lines))
