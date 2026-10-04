"""v45 cycle 5: dissect the 'which alternative' (PIT) stream.

Cycle 3 found that the Voynich PIT stream (the rank-position of each chosen word among the rule model's
alternatives, as an arithmetic decoder would read it) has more sequential structure than its generators
(lag-1 MI excess 0.0064-0.0069 vs 0.0014-0.0031), at the level of a planted code (0.0080). Where does it live?
  - z of lag-1 MI against 200 within-page shuffles; then against 200 within-LINE shuffles (keeps each line's
    mix of probable / improbable choices): if the excess survives, choices depend on their neighbour; if it
    dies, whole lines are written in a 'probable' or 'improbable' mode.
  - Line-level heterogeneity: variance of line-mean u (in-table words), z against within-page shuffles.
  - Which lines: mean u and OOV share by line role (paragraph-first, body, last line) and word position.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v45_lib import *

FN = 'v45_cycle5.txt'
NS = 200


def load(name):
    C = get_corpus(name); R = residual(name)
    lines = []
    for p in C:
        if p['id'] not in R: continue
        cur = None; key = None
        recs = R[p['id']]['recs']
        npara = defaultdict(int)
        for r in recs: npara[r[1]] = max(npara[r[1]], r[2])
        for r in recs:
            k = (r[1], r[2])
            if k != key:
                cur = dict(page=p['id'], role='pf' if r[2] == 0 else ('last' if r[2] == npara[r[1]] else 'body'), u=[], b=[])
                lines.append(cur); key = k
            u = r[8]
            cur['u'].append(u); cur['b'].append(8 if u < 0 else min(7, int(u * 8)))
    return lines


def mi(pairs):
    n = len(pairs); a = Counter(x for x, _ in pairs); b = Counter(y for _, y in pairs); ab = Counter(pairs)
    return sum(v / n * math.log2(v * n / (a[x] * b[y])) for (x, y), v in ab.items())


def lag1(lines):
    return [(L['b'][i], L['b'][i + 1]) for L in lines for i in range(len(L['b']) - 1)]


def shuf_page(lines, rng):
    bypage = defaultdict(list)
    for L in lines: bypage[L['page']].append(L)
    out = []
    for ls in bypage.values():
        pool = [x for L in ls for x in zip(L['b'], L['u'])]; rng.shuffle(pool); i = 0
        for L in ls:
            n = len(L['b']); seg = pool[i:i + n]; i += n
            out.append(dict(L, b=[x for x, _ in seg], u=[y for _, y in seg]))
    return out


def shuf_line(lines, rng):
    out = []
    for L in lines:
        z = list(zip(L['b'], L['u'])); rng.shuffle(z)
        out.append(dict(L, b=[x for x, _ in z], u=[y for _, y in z]))
    return out


def linevar(lines):
    m = [np.mean([u for u in L['u'] if u >= 0]) for L in lines if sum(u >= 0 for u in L['u']) >= 4]
    return float(np.var(m))


def zof(x, null):
    null = np.array(null); return float((x - null.mean()) / (null.std() + 1e-12))


if __name__ == '__main__':
    rng = random.Random(455)
    names = sys.argv[1:] or ['V', 'VI', 'GEN0', 'GEN1', 'PL', 'GEw2000', 'LAw', 'LAl']
    out = {}
    for nm in names:
        L = load(nm)
        m0 = mi(lag1(L)); v0 = linevar(L)
        np_ = [shuf_page(L, rng) for _ in range(NS)]
        mp = [mi(lag1(x)) for x in np_]; vp = [linevar(x) for x in np_]
        ml = [mi(lag1(shuf_line(L, rng))) for _ in range(NS)]
        role = {}
        for r in ('pf', 'body', 'last'):
            us = [u for x in L if x['role'] == r for u in x['u']]
            role[r] = dict(u=float(np.mean([u for u in us if u >= 0])), oov=float(np.mean([u < 0 for u in us])), n=len(us))
        pos = {}
        for k in range(6):
            us = [x['u'][k] for x in L if len(x['u']) > k]
            pos[k] = float(np.mean([u for u in us if u >= 0]))
        res = dict(mi=m0, z_page=zof(m0, mp), ex_page=m0 - float(np.mean(mp)), z_line=zof(m0, ml), ex_line=m0 - float(np.mean(ml)),
                   linevar=v0, z_linevar=zof(v0, vp), linevar_ratio=v0 / float(np.mean(vp)), role=role, pos=pos)
        out[nm] = res; jsave('c5_' + '_'.join(names) + '.json', out)
        print(nm, json.dumps(res), flush=True)
