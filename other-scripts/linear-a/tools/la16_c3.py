#!/usr/bin/env python3
"""LA-16 cycle 3: infer hands from private spellings alone.
For two documents A, B: a CONFLICT is a word in A and a word in B that differ by exactly one sign (and neither
doc has the other's form); an AGREEMENT is a word type shared by A and B. If scribes have private spellings,
conflicts should mark DIFFERENT hands (P(same hand | conflict) < base), agreements the same hand.
(a) Rates within site, real vs labels permuted within site (5,000).
(b) Leave-one-out nearest-neighbour hand recovery from conflicts only (neighbour = doc with most agreements-minus-
    conflicts; a conflict-only score that predicts 'not this hand'), vs permuted labels.
(c) Unattributed documents: assign to the hand whose documents they agree with and do not conflict with; report
    how many get a unique assignment (cannot be checked; counts only). Held-out check: hide 20% of attributed docs.
LB control: KN + PY with DAMOS hands. Planted: in LA, HT Scribe 9 rewrites one mid-frequency sign (rate 1.0).
"""
import sys, os, json, collections, random, copy
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from la16_common import *
from la16_c1 import plant

say = say_to(os.path.join(OUT, 'c3_report.txt'))


def doc_links(docs):
    """-> agree[(a,b)] = #shared types, conf[(a,b)] = #one-sign conflicts, for doc index pairs a<b."""
    T = sorted({w for d in docs for w in d['words']}); ix = {w: i for i, w in enumerate(T)}
    _, P = build_pairs(T)
    occ = collections.defaultdict(set)
    for j, d in enumerate(docs):
        for w in set(d['words']): occ[ix[w]].add(j)
    agree = collections.Counter(); conf = collections.Counter()
    for t, ds in occ.items():
        ds = sorted(ds)
        for a in range(len(ds)):
            for b in range(a + 1, len(ds)): agree[(ds[a], ds[b])] += 1
    sets = [set(ix[w] for w in d['words']) for d in docs]
    for i, j, x, y in P:
        for a in occ[i]:
            for b in occ[j]:
                if a == b: continue
                if j in sets[a] or i in sets[b]: continue
                conf[(min(a, b), max(a, b))] += 1
    return agree, conf


def rates(docs, agree, conf, labs):
    site = [d['site'] for d in docs]
    def r(keys):
        k = [p for p in keys if site[p[0]] == site[p[1]]]
        if not k: return float('nan'), 0
        return sum(labs[a] == labs[b] for a, b in k) / len(k), len(k)
    co = [p for p in conf if p not in agree]; ag = [p for p in agree if p not in conf]
    both = [p for p in conf if p in agree]
    return r(co), r(ag), r(both)


def base_rate(docs, labs):
    by = collections.defaultdict(list)
    for j, d in enumerate(docs): by[d['site']].append(j)
    s = n = 0
    for ii in by.values():
        c = collections.Counter(labs[i] for i in ii); m = len(ii)
        s += sum(v * (v - 1) / 2 for v in c.values()); n += m * (m - 1) / 2
    return s / n


def perm_within_site(docs, labs, rnd):
    out = list(labs); by = collections.defaultdict(list)
    for j, d in enumerate(docs): by[d['site']].append(j)
    for ii in by.values():
        v = [out[i] for i in ii]; rnd.shuffle(v)
        for i, x in zip(ii, v): out[i] = x
    return out


def loo_nn(docs, agree, conf, labs, use='both'):
    sc = collections.defaultdict(dict)
    for (a, b), v in agree.items():
        if use in ('both', 'agree'): sc[a][b] = sc[a].get(b, 0) + v; sc[b][a] = sc[b].get(a, 0) + v
    for (a, b), v in conf.items():
        if use in ('both', 'conf'): sc[a][b] = sc[a].get(b, 0) - v; sc[b][a] = sc[b].get(a, 0) - v
    hit = tot = 0
    sitec = collections.defaultdict(collections.Counter)
    for b in range(len(docs)): sitec[docs[b]['site']][labs[b]] += 1
    for a in range(len(docs)):
        if not sc[a]: continue
        if use == 'conf':
            # conflict-only: vote AGAINST hands of conflicting docs; predict the most common same-site hand not voted against
            bad = {labs[b] for b, v in sc[a].items() if v < 0}
            if not bad: continue
            cand = sitec[docs[a]['site']].copy(); cand[labs[a]] -= 1
            cand = {h: c for h, c in cand.items() if c > 0 and h not in bad}
            if not cand: continue
            pred = max(cand.items(), key=lambda kv: kv[1])[0]
        else:
            b, v = max(sc[a].items(), key=lambda kv: kv[1])
            if v <= 0: continue
            pred = labs[b]
        tot += 1; hit += pred == labs[a]
    return hit, tot


def conflict_only_baseline(docs, labs, mask):
    """majority same-site hand ignoring any information (for the conflict-only vote)."""
    hit = tot = 0
    sitec = collections.defaultdict(collections.Counter)
    for b in range(len(docs)): sitec[docs[b]['site']][labs[b]] += 1
    for a in mask:
        cand = sitec[docs[a]['site']].copy(); cand[labs[a]] -= 1
        cand = {h: c for h, c in cand.items() if c > 0}
        if not cand: continue
        tot += 1; hit += max(cand.items(), key=lambda kv: kv[1])[0] == labs[a]
    return hit, tot


def battery(tag, docs, rnd, nperm=2000):
    docs = [d for d in docs if d.get('hand')]
    labs = [d['hand'] for d in docs]
    agree, conf = doc_links(docs)
    (pc, nc), (pa, na), (pb, nb) = rates(docs, agree, conf, labs)
    br = base_rate(docs, labs)
    say(f'\n== {tag}: docs {len(docs)} hands {len(set(labs))}; same-site doc pairs with conflict-only {nc}, agreement-only {na}, both {nb}')
    say(f'   P(same hand): base (same site) {br:.3f} | conflict-only {pc:.3f} | agreement-only {pa:.3f} | both {pb:.3f}')
    N = []
    for _ in range(nperm):
        l2 = perm_within_site(docs, labs, rnd); N.append(rates(docs, agree, conf, l2))
    for k, nm, real in ((0, 'conflict-only', pc), (1, 'agreement-only', pa), (2, 'both', pb)):
        xs = np.array([n[k][0] for n in N]); xs = xs[~np.isnan(xs)]
        if len(xs) == 0: continue
        say(f'   {nm}: real {real:.3f} null {xs.mean():.3f} [{np.percentile(xs,2.5):.3f},{np.percentile(xs,97.5):.3f}] P(<=) {(xs <= real).mean():.4f} P(>=) {(xs >= real).mean():.4f}')
    out = dict(base=br, conf=pc, agree=pa, both=pb, nc=nc, na=na)
    for use in ('agree', 'both', 'conf'):
        h, t = loo_nn(docs, agree, conf, labs, use)
        Nn = [loo_nn(docs, agree, conf, perm_within_site(docs, labs, rnd), use) for _ in range(int(os.environ.get('NLOO', 200)))]
        acc = np.array([x[0] / max(x[1], 1) for x in Nn])
        say(f'   LOO hand recovery using {use}: {h}/{t} = {h/max(t,1):.3f}; permuted-labels {acc.mean():.3f} 95% {np.percentile(acc,95):.3f} P(>=) {(acc >= h/max(t,1)).mean():.3f}')
        out['loo_' + use] = (h, t, float(acc.mean()), float((acc >= h / max(t, 1)).mean()))
    hb, tb = conflict_only_baseline(docs, labs, range(len(docs)))
    say(f'   majority-hand-at-site baseline: {hb}/{tb} = {hb/max(tb,1):.3f}')
    return out


def unattributed(tag, docs, rnd):
    """Assign unattributed docs to a hand by agreements minus conflicts; held-out check on 20 % hidden attributed."""
    agree, conf = doc_links(docs)
    sc = collections.defaultdict(collections.Counter)
    for (a, b), v in agree.items(): sc[a][b] += v; sc[b][a] += v
    for (a, b), v in conf.items(): sc[a][b] -= v; sc[b][a] -= v
    att = [i for i, d in enumerate(docs) if d.get('hand')]
    def assign(hide):
        hid = set(hide); res = {}
        for a in range(len(docs)):
            if docs[a].get('hand') and a not in hid: continue
            votes = collections.Counter()
            for b, v in sc[a].items():
                if docs[b].get('hand') and b not in hid: votes[docs[b]['hand']] += v
            if votes:
                (h1, v1), *rest = votes.most_common(2)
                if v1 > 0 and (not rest or rest[0][1] < v1): res[a] = h1
        return res
    accs = []; covs = []
    for r in range(50):
        hide = rnd.sample(att, len(att) // 5); res = assign(hide)
        got = [a for a in hide if a in res]
        accs.append(sum(res[a] == docs[a]['hand'] for a in got) / max(len(got), 1)); covs.append(len(got) / len(hide))
    res = assign([])
    un = [a for a in range(len(docs)) if not docs[a].get('hand')]
    say(f'   {tag}: held-out 20% attributed: coverage {np.mean(covs):.3f}, accuracy {np.mean(accs):.3f} (50 splits)')
    say(f'   {tag}: unattributed docs {len(un)}, uniquely assigned {sum(1 for a in un if a in res)}; by hand: ' +
        ', '.join(f'{h} {n}' for h, n in collections.Counter(res[a] for a in un if a in res).most_common(8)))
    return dict(acc=float(np.mean(accs)), cov=float(np.mean(covs)), assigned={docs[a]['id']: res[a] for a in un if a in res})


def main():
    rnd = random.Random(3)
    res = {}
    lb = lb_corpus(('KN', 'PY'))
    res['LB'] = battery('LB KN+PY', lb, rnd, nperm=1000)
    res['LB_un'] = unattributed('LB', lb, rnd)
    la = la_corpus()
    res['LA'] = battery('LA hands', la, rnd)
    res['LA_un'] = unattributed('LA', la, rnd)
    # planted private spelling in HT Scribe 9 (several signs at once, rate 1.0): conflicts should now flag the hand
    cnt = collections.Counter(s for d in la if d['hand'] == 'HT Scribe 9' for w in d['words'] for s in w)
    for draw in range(3):
        xs = rnd.sample([s for s, c in cnt.items() if c >= 3], 3)
        D = la
        allc = [s for s in collections.Counter(s for d in la for w in d['words'] for s in w)]
        for x in xs: D = plant(D, 'HT Scribe 9', x, rnd.choice(allc), 1.0, rnd)
        res[f'LA_plant{draw}'] = battery(f'LA planted HT9 private spellings {xs}', D, rnd, nperm=1000)
    json.dump(res, open(os.path.join(OUT, 'c3.json'), 'w'), default=str, indent=1)


if __name__ == '__main__':
    main()
