"""v68 cycle 3: read the fitted 'pitch line' as music, on the held-out half only.

Uses the blind smooth-kernel assignment from v68_c2b (fitted on the training half: entry class s(w), exit class
e(w) on a 1-D line of K positions).  On the other half:
  STEP   per adjacent word pair, -|s(next) - e(prev)|, minus the same for pairs re-drawn inside the line
         (within-line shuffle, 200 draws).  By section (Voynich illustration code x Currier language) and by
         mode (chant).
  FINAL  for paragraphs / pieces with >= 3 lines: does a line's LAST word end on the class the other lines of
         its unit end on (leave-one-out modal final), more than a random mid-line word matches its own
         position-mates?  Chant must show the modal final; the null is the same test on a mid-line position.
  MODE   MI(section or mode; unit's modal final class) vs MI(section; modal class of a mid-line word), with a
         permutation null over units.
usage: python3 v68_c3.py corpus K
"""
import json, math, os, random, sys
import numpy as np
from collections import Counter, defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v68_c2 as C2
import v68_lib as V
import v53_lib as L53


def units_of(name):
    """held-out half as units: list of (stratum, [lines of words])."""
    if name.startswith('chant'):
        pieces = V.chant_units()
        src, meta = [], []
        for p in pieces:
            for l in p['lines']:
                if len(src) >= 9000:
                    break
                src.append(l); meta.append((p['id'], p['mode']))
        enc = {'rep': 'abs', 'fold': False, 'group': 'one', 'trunc': 0, 'rank': True, 'p2': 0.3, 'seed': 77,
               'sympermute': True}
        coded = V.encode(src, enc)
        lf = [(i // 25, meta[i], l) for i, l in enumerate(coded)]
    elif name.split('_')[0] in ('LA', 'DE'):
        src = V.lang_lines({'LA': 'LA_ency', 'DE': 'DE_herb'}[name.split('_')[0]])
        enc = {'rep': 'abs', 'fold': False, 'group': 'one', 'trunc': 0, 'rank': True, 'p2': 0.3, 'seed': 78,
               'sympermute': True}
        coded = V.encode(src, enc, symmap=list(range(26)))
        lf = [(i // 25, ('u%d' % (i // 5), 'all'), l) for i, l in enumerate(coded)]
    else:
        vz = L53.load_voynich('ZL3b' if name.startswith('ZL') else 'IT2a')
        recs = [r for r in json.load(open(os.path.join(V.DATA, 'derived', ('ZL3b' if name.startswith('ZL') else 'IT2a') + '_lines.json'))) if r['ltype'] == 'P']
        # rebuild with paragraph ids and section, same word filtering as v53 load
        lf, pid = [], 0
        for r in recs:
            ws = [L53.vglyphs(w) for w, u in zip(r['words'], r['uncertain']) if not u and '?' not in w]
            ws = [w for w in ws if w]
            if r.get('para_start'):
                pid += 1
            if ws:
                lf.append((r['folio'], ('%s.%d' % (r['folio'], pid), '%s%s' % (r['illus'], r.get('lang') or '-')), ws))
    fol = []
    for f, _, _ in lf:
        if f not in fol:
            fol.append(f)
    A = set(fol[0::2])
    test = [(m, l) for f, m, l in lf if f not in A]
    U = defaultdict(list); strat = {}
    for (uid, st), l in test:
        U[uid].append(l); strat[uid] = st
    return [(strat[u], U[u]) for u in U]


def main(name, K):
    asg = json.load(open(os.path.join(V.CK, 'c2b_assign_%s_K%d.json' % (name, K))))
    ce, cs = asg['ce'], asg['cs']
    E = lambda w: ce.get(w, ce.get('E' + w[-1]))
    S = lambda w: cs.get(w, cs.get('S' + w[0]))
    rng = random.Random(5)
    units = units_of(name)
    res = {'name': name, 'K': K, 'beta': asg['beta'], 'n_units': len(units)}
    # STEP by stratum
    st_obs, st_null = defaultdict(list), defaultdict(list)
    for st, lines in units:
        for l in lines:
            ok = [w for w in l if E(w) is not None and S(w) is not None]
            if len(ok) < 3:
                continue
            obs = np.mean([-abs(S(b) - E(a)) for a, b in zip(ok, ok[1:])])
            nul = []
            for _ in range(20):
                p = ok[:]; rng.shuffle(p)
                nul.append(np.mean([-abs(S(b) - E(a)) for a, b in zip(p, p[1:])]))
            st_obs[st].append(obs); st_null[st].append(np.mean(nul))
    step = {}
    allo, alln = [], []
    for st in st_obs:
        d = np.array(st_obs[st]) - np.array(st_null[st])
        allo.extend(d)
        if len(d) >= 20:
            step[st] = (round(float(d.mean()), 4), round(float(d.mean() / (d.std() / math.sqrt(len(d)) + 1e-9)), 2), len(d))
    allo = np.array(allo)
    res['step_all'] = (float(allo.mean()), float(allo.mean() / (allo.std() / math.sqrt(len(allo)))), len(allo))
    res['step_by_stratum'] = step
    # FINAL: leave-one-out modal final match, last word vs a fixed mid position (word index 1)
    def loo(units_, pos):
        hit, n = 0, 0
        for st, lines in units_:
            cl = []
            for l in lines:
                if len(l) >= 3:
                    w = l[-1] if pos == 'last' else l[1]
                    c = E(w)
                    if c is not None:
                        cl.append(c)
            if len(cl) < 3:
                continue
            cnt = Counter(cl)
            for c in cl:
                cnt[c] -= 1
                m = max(cnt.values())
                best = [k for k, v in cnt.items() if v == m]
                hit += (c in best) / len(best); n += 1
                cnt[c] += 1
        return hit / max(n, 1), n
    fl, n1 = loo(units, 'last'); fm, n2 = loo(units, 'mid')
    # null for 'last': classes shuffled between units (keeps marginal)
    lasts = [(st, [l for l in lines if len(l) >= 3]) for st, lines in units]
    pool = [l[-1] for st, ls in lasts for l in ls]
    nulls = []
    for _ in range(100):
        rng.shuffle(pool); it = iter(pool); fake = []
        for st, ls in lasts:
            fake.append((st, [l[:-1] + [next(it)] for l in ls]))
        nulls.append(loo(fake, 'last')[0])
    res['final_last'] = fl; res['final_mid'] = fm; res['final_n'] = n1
    res['final_null_mean'] = float(np.mean(nulls)); res['final_z'] = float((fl - np.mean(nulls)) / (np.std(nulls) + 1e-9))
    # MODE: MI(stratum; modal final) vs MI(stratum; modal mid-class)
    def modal(lines, pos):
        cl = [E(l[-1] if pos == 'last' else l[1]) for l in lines if len(l) >= 3]
        cl = [c for c in cl if c is not None]
        return Counter(cl).most_common(1)[0][0] if cl else None
    def mi(pairs):
        return L53.MI(pairs)
    for pos in ('last', 'mid'):
        pr = [(st, modal(ls, pos)) for st, ls in units if len(ls) >= 3]
        pr = [p for p in pr if p[1] is not None]
        o = mi(pr)
        sts = [p[0] for p in pr]; cls = [p[1] for p in pr]
        nl = []
        for _ in range(300):
            rng.shuffle(sts); nl.append(mi(list(zip(sts, cls))))
        res['mode_%s' % pos] = (round(o, 4), round((o - np.mean(nl)) / (np.std(nl) + 1e-9), 2), len(pr))
        if pos == 'last':
            fin = defaultdict(Counter)
            for st, c in pr:
                fin[st][c] += 1
            res['finals_by_stratum'] = {st: dict(v.most_common(3)) for st, v in fin.items() if sum(v.values()) >= 10}
    return res


if __name__ == '__main__':
    r = main(sys.argv[1], int(sys.argv[2]))
    print(json.dumps(r))
    with open(os.path.join(V.CK, 'c3_results.jsonl'), 'a') as f:
        f.write(json.dumps(r) + '\n')
