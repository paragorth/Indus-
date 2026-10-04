"""R2 report: held-out (split C) test of each run's hall of fame, nulls vs real, and a
post-hoc dissection of where a program's gains come from (python re-implementation of
the C interpreter).  usage: python3 r2_report.py TAG [TAG...]"""
import os, sys, json, glob, struct, math, random
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r1_lib as L

R2 = os.path.join(L.VOY, 'results', 'r2')
TN = ['P1', 'P2', 'P3', 'P4', 'P6', 'P8', 'AB', 'AW', 'AW2', 'FW', 'FL', 'PW', 'PL', 'WI', 'LI', 'CW', 'LPW', 'LLW']
AR = {'add': 2, 'sub': 2, 'mul': 2, 'mod': 2, 'min': 2, 'max': 2, 'eq': 2, 'lt': 2, 'if': 3, 'tab1': 1, 'tab2': 2, 'lsys': 1}


LAYOUT = False


def load_bin(name, layout=None):
    layout = LAYOUT if layout is None else layout
    with open(os.path.join(R2, name + '.bin'), 'rb') as f:
        N, NT, V, ND = struct.unpack('iiii', f.read(16))
        X = np.frombuffer(f.read(4 * N), np.int32); SP = np.frombuffer(f.read(4 * N), np.int32)
        DOC = np.frombuffer(f.read(4 * N), np.int32); PB = np.frombuffer(f.read(8 * N), np.float64)
        F = np.frombuffer(f.read(4 * N * NT), np.int32).reshape(NT, N).astype(np.uint8)
    D = dict(N=N, V=V, ND=ND, X=X, SP=SP, DOC=DOC, PB=PB, F=F, BAR=-1)
    if layout:   # line ends given: drop them from scoring, renormalise the baseline (as the C engine, mode 2)
        pbar = np.fromfile(os.path.join(R2, name + '.pbar'), np.float64)
        bar = int(X[-1]); D['BAR'] = bar
        D['SP'] = np.where(X == bar, 3, SP); D['PB'] = np.where(X == bar, PB, PB / (1 - pbar))
    return D


def n_c_docs(name):
    meta = json.load(open(os.path.join(R2, name + '.json')))
    return meta


def boot(perdoc, ncdocs, bits, B=4000, seed=0):
    g = np.zeros(ncdocs); v = [x[1] for x in perdoc]; g[:len(v)] = v
    rng = np.random.default_rng(seed)
    s = g[rng.integers(0, ncdocs, size=(B, ncdocs))].sum(1) - bits
    return float((s <= 0).mean())


def corpus_of(run):
    base = os.path.basename(run)[:-4]
    tag, rest = base.split('_', 1)
    return tag, rest.rsplit('_s', 1)[0]


def summarize(tag):
    rows = {}
    for f in sorted(glob.glob(os.path.join(R2, 'runs', f'{tag}_*.out'))):
        _, name = corpus_of(f)
        d = json.load(open(f))
        D = load_bin(name)
        ncd = len(set(D['DOC'][D['SP'] == 2].tolist()))
        nC = d['nC']
        best = None; nsig = 0
        for h in d['hof']:
            h['netC'] = h['gainC'] - h['bits']
            h['p'] = boot(h['perdoc'], ncd, h['bits'])
            h['sig'] = h['p'] < 0.05 / len(d['hof'])
            nsig += h['sig']
        top = max(d['hof'], key=lambda h: h['fitB'])
        bestC = max(d['hof'], key=lambda h: h['netC'])
        rows[name] = dict(evals=d['evals'], nC=nC, top=top, bestC=bestC, nsig=nsig, nhof=len(d['hof']))
    return rows


def fmt(name, r):
    t = r['top']
    return (f"{name}: evals {r['evals']}; top-by-B: `{t['prog'].strip()}` fitB {t['fitB']:.0f}, "
            f"netC {t['netC']:.0f} bits ({1000 * t['netC'] / r['nC']:.1f} mbit/tok, p {t['p']:.4f}); "
            f"HOF sig on C {r['nsig']}/{r['nhof']}; max netC {r['bestC']['netC']:.0f} "
            f"({1000 * r['bestC']['netC'] / r['nC']:.1f} mbit/tok)")


# ---------------------------------------------------------- python interpreter
def lsys_str(rules):
    a = [0]
    for _ in range(12):
        if len(a) >= 256: break
        a = [x for s in a for x in rules[s]][:590]
    return np.array([a[i % len(a)] for i in range(256)], dtype=np.uint8)


def run_prog(prog, D):
    toks = prog.split(); rules = None
    if '[L:' in prog:
        i = toks.index('[L:'); spec = toks[i + 1:]; toks = toks[:i]
        rules = [[int(c) for c in s.split('>')[1].rstrip(']')] for s in spec]
    N, V, X, SP = D['N'], D['V'], D['X'], D['SP']
    st, purity = [], None
    A = SP == 0
    for t in toks:
        if t in TN: st.append(D['F'][TN.index(t)].copy()); continue
        if t.isdigit(): st.append(np.full(N, int(t), np.uint8)); continue
        ar = AR[t]; args = st[-ar:]; del st[-ar:]
        a = args[0]; b = args[1] if ar > 1 else None; c = args[2] if ar > 2 else None
        with np.errstate(all='ignore'):
            if t == 'add': o = a + b
            elif t == 'sub': o = a - b
            elif t == 'mul': o = a * b
            elif t == 'mod': o = np.where(b > 0, a % np.maximum(b, 1), a).astype(np.uint8)
            elif t == 'min': o = np.minimum(a, b)
            elif t == 'max': o = np.maximum(a, b)
            elif t == 'eq': o = (a == b).astype(np.uint8)
            elif t == 'lt': o = (a < b).astype(np.uint8)
            elif t == 'if': o = np.where(a > 0, b, c).astype(np.uint8)
            elif t == 'lsys': o = lsys_str(rules)[a]
            else:
                key = a.astype(np.int64) if t == 'tab1' else (a.astype(np.int64) << 8 | b)
                cnt, bst, tot = {}, {}, {}
                for k, y in zip(key[A].tolist(), X[A].tolist()):   # same tie rule as the C engine
                    c = cnt.get((k, y), 0) + 1; cnt[(k, y)] = c; tot[k] = tot.get(k, 0) + 1
                    if c > bst.get(k, (0, 0))[0]: bst[k] = (c, y)
                tab = {k: (y, n / tot[k]) for k, (n, y) in bst.items() if n >= 2}
                o = np.array([tab.get(k, (255, 0))[0] for k in key.tolist()], np.uint8)
                pu = np.array([tab.get(k, (255, 0))[1] for k in key.tolist()])
                purity = np.digitize(pu, [0.25, 0.5, 0.75])
        st.append(o.astype(np.uint8))
    pred = st[-1].copy()
    if D.get('BAR', -1) >= 0: pred[pred == D['BAR']] = 255
    if not (toks[-1] in ('tab1', 'tab2')): purity = np.zeros(N, int)
    return pred, purity


def dissect(name, h, D=None):
    """Gain on C split by position type."""
    D = D or load_bin(name)
    pred, pur = run_prog(h['prog'], D)
    lam = np.array(h['lam'])
    X, SP, PB, F = D['X'], D['SP'], D['PB'], D['F']
    meta = json.load(open(os.path.join(R2, name + '.json')))
    sym = meta['symbols']
    l = lam[pur]
    ok = (SP == 2) & (pred < D['V']) & (l > 0)
    g = np.zeros(D['N'])
    hit = pred == X
    g[ok & hit] = np.log2((l + (1 - l) * PB) / PB)[ok & hit]
    g[ok & ~hit] = np.log2(1 - l)[ok & ~hit]
    PL, PW, P1 = F[TN.index('PL')], F[TN.index('PW')], F[TN.index('P1')]
    bar = sym.index('|') if '|' in sym else -1
    us = sym.index('_') if '_' in sym else -1
    types = {'line start': PL == 0, 'word start (not line start)': (PW == 0) & (PL > 0),
             'inside word': PW > 0}
    out = {k: float(g[v].sum()) for k, v in types.items()}
    out['predicting |'] = float(g[X == bar].sum()) if bar >= 0 else 0
    out['predicting _'] = float(g[X == us].sum()) if us >= 0 else 0
    # top hit symbols
    from collections import Counter
    c = Counter()
    for i in np.where(ok & hit)[0]: c[sym[X[i]] if X[i] < len(sym) else 'RARE'] += g[i]
    out['top_symbols'] = [(s, round(v, 1)) for s, v in c.most_common(8)]
    return out


if __name__ == '__main__':
    for tag in sys.argv[1:]:
        LAYOUT = tag.startswith('c3')
        rows = summarize(tag)
        for name, r in rows.items():
            print(fmt(name, r))
        json.dump({n: {k: (v if k not in ('top', 'bestC') else {kk: vv for kk, vv in v.items() if kk != 'perdoc'})
                       for k, v in r.items()} for n, r in rows.items()},
                  open(os.path.join(R2, f'report_{tag}.json'), 'w'), indent=1)


# ---------------------------------------------------------- stronger baseline + refit
def cache_pb(D, order=2, window=400):
    """KN baseline mixed with a document cache (order-0 and order-`order` counts over the
    previous `window` tokens of the same document); weights fitted on split B."""
    X, DOC, PB, V = D['X'], D['DOC'], D['PB'], D['V']
    N = D['N']
    c0 = np.zeros(N); c2 = np.zeros(N)
    from collections import deque, Counter
    last = -1
    for i in range(N):
        if DOC[i] != last:
            hist = deque(); cu = Counter(); cb = Counter(); cc = Counter(); last = DOC[i]
        n = len(hist)
        c0[i] = (cu[X[i]] + 0.1) / (n + 0.1 * V) if n else 1.0 / V
        ctx = tuple(list(hist)[-order:]) if n >= order else None
        if ctx is not None and cc[ctx] > 0:
            c2[i] = cb[ctx + (X[i],)] / cc[ctx]
        else:
            c2[i] = c0[i]
        hist.append(X[i]); cu[X[i]] += 1
        if n + 1 > order:
            h = tuple(list(hist)[-order - 1:]); cb[h] += 1; cc[h[:-1]] += 1
        if len(hist) > window:
            old = hist.popleft(); cu[old] -= 1
    B = D['SP'] == 1
    best = (-1e18, None)
    for a in (0, .02, .05, .1, .2):
        for b in (0, .02, .05, .1, .2, .3):
            if a + b >= 0.6: continue
            p = (1 - a - b) * PB + a * c0 + b * c2
            ll = np.log2(p[B]).sum()
            if ll > best[0]: best = (ll, (a, b))
    a, b = best[1]
    return (1 - a - b) * PB + a * c0 + b * c2, (a, b)


def gain_with(pred, pur, D, PB):
    """refit 4 purity-bin lams on B against baseline PB; return (gainB, gainC, perdocC, lams)."""
    X, SP = D['X'], D['SP']
    lams = np.zeros(4)
    ok = pred < D['V']; hit = pred == X
    for b in range(4):
        m = ok & (pur == b) & (SP == 1)
        h = m & hit; nm = (m & ~hit).sum(); hp = PB[h]
        if len(hp) == 0: continue
        lo, hi = 0.0, 1 - 1e-9
        for _ in range(40):
            l = (lo + hi) / 2; d = -nm / (1 - l) + ((1 - hp) / (l + (1 - l) * hp)).sum()
            lo, hi = (l, hi) if d > 0 else (lo, l)
        lams[b] = (lo + hi) / 2
    l = lams[pur]; g = np.zeros(D['N']); v = ok & (l > 0)
    g[v & hit] = np.log2((l + (1 - l) * PB) / PB)[v & hit]
    g[v & ~hit] = np.log2(1 - l)[v & ~hit]
    per = {}
    for dd, gg in zip(D['DOC'][SP == 2], g[SP == 2]): per[dd] = per.get(dd, 0) + gg
    return float(g[SP == 1].sum()), float(g[SP == 2].sum()), per, lams
