"""pe23 helpers shared by cycles (separation, planting, value shuffles)."""
import math
from pe23_common import *

GS = 60


def groups_for(entries, tabset, ngroups, rng, gs=GS):
    pool = [e for e in entries if e['tab'] in tabset]
    tabs = sorted({e['tab'] for e in pool})
    bt = defaultdict(list)
    for e in pool:
        bt[e['tab']].append(e)
    out = []
    for _ in range(ngroups):
        order = rng.permutation(len(tabs))
        g = []
        for i in order:
            g += bt[tabs[i]][:10]
            if len(g) >= gs:
                break
        if len(g) >= gs // 2:
            out.append(g[:gs])
    return out


def vec(g, den, rng):
    f, eff, z = effects([e['val'] for e in g], [e['tab'] for e in g], den, rng, reps=40)
    return [0.0 if np.isnan(eff[k]) else eff[k] for k in FEATS]


def separation(labelled, classes, rng, tag, gs=GS, den_of=None):
    den_of = den_of or (lambda g: ur_den(g[0]['cls']))
    alltabs = sorted({e['tab'] for e in labelled})
    rng.shuffle(alltabs)
    A, B = set(alltabs[::2]), set(alltabs[1::2])
    X = {h: [] for h in 'AB'}
    Y = {h: [] for h in 'AB'}
    for c in classes:
        ents = [e for e in labelled if e['lab'] == c]
        for h, ts in (('A', A), ('B', B)):
            for g in groups_for(ents, ts, 30, rng, gs):
                X[h].append(vec(g, den_of(g), rng))
                Y[h].append(c)
    XA, XB = np.array(X['A']), np.array(X['B'])
    mu, sd = XA.mean(0), XA.std(0) + 1e-9
    XA, XB = (XA - mu) / sd, (XB - mu) / sd
    YA = np.array(Y['A'])
    cents = {c: XA[YA == c].mean(0) for c in classes if (YA == c).any()}
    pred = [min(cents, key=lambda c: ((x - cents[c]) ** 2).sum()) for x in XB]
    acc = float(np.mean([p == y for p, y in zip(pred, Y['B'])]))
    conf = Counter(zip(Y['B'], pred))
    print(tag, 'acc', round(acc, 3), 'chance', round(1 / len(classes), 3), 'nB', len(pred))
    return acc, {f'{a}->{b}': n for (a, b), n in conf.items()}, {c: list(map(float, cents[c])) for c in cents}, (mu.tolist(), sd.tolist())


def plant(ents, signs, p, rng, key='final'):
    out = []
    for e in ents:
        e = dict(e)
        if e[key] in signs and e['val'] >= 10 and rng.random() < p:
            den = pe_den(e['sys'], e.get('vset', 'B'))
            hi = max(d for d in den if d <= e['val'])
            e['val'] = int(max(1, round(e['val'] / hi)) * hi)
        out.append(e)
    return out


def shuffle_values(ents, rng):
    ents = [dict(e) for e in ents]
    st = defaultdict(list)
    for i, e in enumerate(ents):
        st[(e['sys'], int(math.log2(max(1, e['val']))))].append(i)
    for idx in st.values():
        vals = [ents[i]['val'] for i in idx]
        rng.shuffle(vals)
        for i, v in zip(idx, vals):
            ents[i]['val'] = v
    return ents
