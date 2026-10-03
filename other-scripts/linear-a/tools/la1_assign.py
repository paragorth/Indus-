#!/usr/bin/env python3
"""LA-1 cycle 1: assign Linear A fraction letters to Linear B sub-unit roles by commodity profile.

Score(f,u) = log-likelihood of letter f's commodities under P(c|f) ~ P_LA(c) * lift_LB(u,c)
(lift = P_LB(u|c)/P_LB(u), Dirichlet prior alpha=5 centred on P_LB(u)); this removes the
LA-vs-LB commodity-marginal difference. Cycle 1.1 used a raw lift sum and add-0.5 smoothing (biased).
over commodities shared by both tables. Each letter takes its best role (many-to-one: LA has
13+ letters, LB 4 capacity sub-units + 3 weight units). Statistic G = sum_f max_u Score(f,u),
minus the same with no commodity information (0).
Null 1 (label shuffle): permute LA letters across quantities (commodity and letter counts kept).
Null 2 (role shuffle): permute LB unit labels across LB rows.
Planted control: give each LA letter a random true role, redraw its commodities from
P_LA(c) * lift_LB(u,c) with the real LA letter counts, and measure recovery.
"""
import sys, os, math, random, json
from collections import Counter, defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import la1_common as L

random.seed(11)
ALPHA = 5.0
GRA_MERGE = '--merge-hord' in sys.argv
ROLES = ['T', 'S', 'V', 'Z', 'M', 'N', 'P']
CAP_ROLES = ['T', 'S', 'V', 'Z']

lb = [(s, c, u) for s, c, u in L.lb_rows() if u != 'INT']
if GRA_MERGE: lb = [(s, 'GRA' if c == 'HORD' else c, u) for s, c, u in lb]
la = L.la_rows()
SHARED = sorted({c for c, _ in la} & {c for _, c, _ in lb})
la = [(c, f) for c, f in la if c in SHARED]
lb = [(c, u) for _, c, u in lb if c in SHARED]
LETTERS = [f for f, n in Counter(f for _, f in la).most_common() if n >= 3]
la = [(c, f) for c, f in la if f in LETTERS]


def lift(lbrows, roles):
    t = L.table(lbrows); n = len(lbrows)
    pu = Counter(u for _, u in lbrows)
    out = {}
    for c in SHARED:
        nc = sum(t[c].values())
        for u in roles:
            p_u = (pu[u] + 0.5) / (n + 0.5 * len(roles))
            p_uc = (t[c][u] + ALPHA * p_u) / (nc + ALPHA)      # Dirichlet prior centred on the role marginal
            out[(u, c)] = math.log(p_uc / p_u)
    return out


def scores(larows, lf, roles):
    # log-likelihood of each letter's commodities under P(c|f,u) = P_LA(c) e^lift(u,c) / Z_u
    t = L.table(larows, 1, 0); pc = Counter(c for c, _ in larows); n = len(larows)
    out = {}
    for f in LETTERS:
        out[f] = {}
        for u in roles:
            Z = sum(pc[c] / n * math.exp(lf[(u, c)]) for c in SHARED)
            out[f][u] = sum(t[f][c] * (lf[(u, c)] - math.log(Z)) for c in SHARED)
    return out


def best(sc):
    a = {f: max(sc[f], key=sc[f].get) for f in sc}
    return a, sum(sc[f][a[f]] for f in sc)


def run(roles, label):
    lf = lift(lb, roles)
    sc = scores(la, lf, roles)
    a, G = best(sc)
    print(f'\n== roles {roles} ({label}) ==')
    print('LB log-lift by commodity:')
    print('      ' + ' '.join(f'{c:>6s}' for c in SHARED))
    for u in roles: print(f'  {u:3s} ' + ' '.join(f'{lf[(u, c)]:6.2f}' for c in SHARED))
    print('LA letter x commodity counts and best role:')
    t = L.table(la, 1, 0)
    for f in LETTERS:
        srt = sorted(sc[f].items(), key=lambda x: -x[1])
        margin = srt[0][1] - srt[1][1]
        print(f'  {f:3s} ' + ' '.join(f'{t[f][c]:6d}' for c in SHARED) + f'   -> {a[f]} (score {srt[0][1]:.1f}, margin over {srt[1][0]} {margin:.1f})')
    # null 1: label shuffle
    cs = [c for c, _ in la]; fs = [f for _, f in la]; n1 = []
    for _ in range(2000):
        random.shuffle(fs); n1.append(best(scores(list(zip(cs, fs)), lf, roles))[1])
    p1 = sum(v >= G for v in n1) / len(n1)
    # null 2: role shuffle in LB
    lc = [c for c, _ in lb]; lu = [u for _, u in lb]; n2 = []
    for _ in range(500):
        random.shuffle(lu); lf2 = lift([x for x in zip(lc, lu) if x[1] in roles], roles)
        n2.append(best(scores(la, lf2, roles))[1])
    p2 = sum(v >= G for v in n2) / len(n2)
    print(f'G = {G:.1f}; label-shuffle null {sum(n1)/len(n1):.1f} (95% {sorted(n1)[int(.95*len(n1))]:.1f}), P = {p1:.4f}; '
          f'LB role-shuffle null {sum(n2)/len(n2):.1f}, P = {p2:.4f}')
    # per-letter stability: bootstrap LA rows
    stab = defaultdict(Counter)
    for _ in range(500):
        bs = [random.choice(la) for _ in la]
        ab, _ = best(scores(bs, lf, roles))
        for f in LETTERS: stab[f][ab[f]] += 1
    print('bootstrap share of best role: ' + ', '.join(f'{f}:{a[f]} {stab[f][a[f]]/500:.2f}' for f in LETTERS))
    # planted control
    pc = Counter(c for c, _ in la); nf = Counter(f for _, f in la)
    rec = []; Gp = []
    for _ in range(300):
        truth = {f: random.choice(roles) for f in LETTERS}
        rows = []
        for f in LETTERS:
            w = [pc[c] * math.exp(lf[(truth[f], c)]) for c in SHARED]
            rows += [(c, f) for c in random.choices(SHARED, w, k=nf[f])]
        ap, g = best(scores(rows, lf, roles))
        rec.append(sum(ap[f] == truth[f] for f in LETTERS) / len(LETTERS)); Gp.append(g)
    print(f'planted control: recovery {sum(rec)/len(rec):.2f} of letters (chance {1/len(roles):.2f}); planted G mean {sum(Gp)/len(Gp):.1f}')
    return {'assign': a, 'G': G, 'p_label': p1, 'p_role': p2, 'stab': {f: stab[f][a[f]] / 500 for f in LETTERS},
            'planted_recovery': sum(rec) / len(rec), 'planted_G': sum(Gp) / len(Gp)}


print('shared commodities:', SHARED, '| LA rows', len(la), '| LB rows', len(lb), '| letters', LETTERS)
R = {'cap': run(CAP_ROLES, 'capacity sub-units'), 'all': run(ROLES, 'with weight units')}
# dry/liquid family test: does each letter prefer dry (GRA NI OLIV CYP AROM) or liquid (OLE VIN)?
t = L.table(la, 1, 0)
print('\nliquid share by letter (OLE+VIN)/(all):', ', '.join(f"{f} {(t[f]['OLE']+t[f]['VIN'])/sum(t[f].values()):.2f}" for f in LETTERS))
json.dump(R, open(os.path.join(L.D, 'la1_assign' + ('_merged' if GRA_MERGE else '') + '.json'), 'w'), indent=1)
