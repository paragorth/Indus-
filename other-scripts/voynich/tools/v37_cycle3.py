"""v37 cycle 3: massive random key -> value hypotheses + catalogue keys.
(A) 2,000 random (key function, value feature) pairs: key = prefix/suffix/identity class of the ANCHOR word
    (first word = KEY pipeline; a random interior word = MID pipeline; last word = END pipeline), value = 'the
    line holds an interior word (distance >= 2 from the anchor) with prefix / suffix / bigram s'.  Score =
    line-specific MI (bits x 1000) = MI(key; value) - mean MI with the key taken from 3 other lines of the
    same paragraph. Ranked on training pages; top 40 re-scored on held-out pages (replication score).
(B) Catalogue keys: (i) do lines sharing an opening word have more similar rest-of-line content than lines
    sharing a mid-line word (Jaccard of 2-unit prefixes of interior words, same-section pairs from different
    pages)? (ii) is a word more section-bound as a line opener than the same word mid-line (equal-count
    subsamples)? (iii) key reuse across pages.
Usage: python3 v37_cycle3.py NAME [NAME...]"""
import sys, time
from v37_lib import *
from v37_cycle1 import get

NH = 2000; TOP = 40


def build(C, rng, body_only):
    U = [(pi, ai, l) for pi, ai, l in line_units(C, body_only) if len(l) >= 6]
    bypara = defaultdict(list); bypage = defaultdict(list)
    for j, (pi, ai, l) in enumerate(U): bypara[(pi, ai)].append(j); bypage[pi].append(j)
    rows = {}
    for kind in ('KEY', 'MID', 'END'):
        anc = []; feats = []
        for pi, ai, l in U:
            n = len(l); a = 0 if kind == 'KEY' else (n - 1 if kind == 'END' else rng.randint(2, n - 3))
            anc.append(l[a]); fs = set()
            for t in targets(n, a):
                w = l[t]
                for r in (1, 2, 3): fs.add('P' + w[:r]); fs.add('S' + w[-r:])
                for i in range(len(w) - 1): fs.add('B' + w[i:i + 2])
            feats.append(fs)
        rows[kind] = (anc, feats)
    swaps = []
    for s in range(3):
        perm = np.arange(len(U))
        for j, (pi, ai, l) in enumerate(U):
            pool = [x for x in bypara[(pi, ai)] if x != j]
            if len(pool) < 3: pool = [x for x in bypage[pi] if x != j]
            perm[j] = rng.choice(pool) if pool else j
        swaps.append(perm)
    pages = np.array([pi for pi, ai, l in U])
    return U, rows, swaps, pages


def keyfun(spec, w):
    t, r = spec
    if t == 'P': return w[:r]
    if t == 'S': return w[-r:]
    if t == 'L': return str(min(len(w), 8))
    return w


def mi(x, y, nx, ny):
    """plug-in MI (bits) of two integer arrays."""
    j = np.bincount(x * ny + y, minlength=nx * ny).reshape(nx, ny).astype(float); n = j.sum()
    px = j.sum(1, keepdims=True) / n; py = j.sum(0, keepdims=True) / n; p = j / n
    with np.errstate(divide='ignore', invalid='ignore'):
        return float(np.nansum(p * np.log2(p / (px * py))))


def score_set(hyps, rows, swaps, idx, kind):
    anc, feats = rows[kind]
    out = []
    for kspec, fv in hyps:
        kv = [keyfun(kspec, anc[i]) for i in range(len(anc))]
        top = [k for k, _ in Counter(kv[i] for i in idx).most_common(7)]
        kid = np.array([top.index(k) if k in top else 7 for k in kv])
        y = np.array([fv in feats[i] for i in range(len(feats))], int)
        yi = y[idx]
        if yi.sum() < 10 or yi.sum() > len(yi) - 10: out.append(np.nan); continue
        real = mi(kid[idx], yi, 8, 2)
        null = np.mean([mi(kid[sw[idx]], yi, 8, 2) for sw in swaps])
        out.append(1000 * (real - null))
    return np.array(out)


def random_hyps(rows, rng, n):
    anc, feats = rows['KEY']
    pool = Counter(f for fs in feats for f in fs)
    vals = [f for f, c in pool.items() if 0.03 * len(feats) <= c <= 0.6 * len(feats)]
    H = []
    for _ in range(n):
        t = rng.choice(['P', 'P', 'S', 'S', 'L', 'W']); r = rng.choice([1, 2, 3])
        H.append(((t, r), rng.choice(vals)))
    return H


def part_A(C, seed, body_only):
    rng = random.Random(seed)
    U, rows, swaps, pages = build(C, rng, body_only)
    up = sorted(set(pages)); rng.shuffle(up); trp = set(up[:len(up) // 2])
    tr = np.array([i for i in range(len(U)) if pages[i] in trp]); te = np.array([i for i in range(len(U)) if pages[i] not in trp])
    H = random_hyps(rows, rng, NH)
    res = {}
    for kind in ('KEY', 'MID', 'END'):
        s_tr = score_set(H, rows, swaps, tr, kind)
        order = np.argsort(-np.nan_to_num(s_tr, nan=-1e9))[:TOP]
        s_te = score_set([H[i] for i in order], rows, swaps, te, kind)
        rnd = rng.sample(range(NH), TOP)
        s_rnd = score_set([H[i] for i in rnd], rows, swaps, te, kind)
        res[kind] = {'train_top_mean': float(np.nanmean(s_tr[order])), 'test_top_mean': float(np.nanmean(s_te)),
                     'test_rand_mean': float(np.nanmean(s_rnd)), 'test_top_pos': float(np.nanmean(s_te > 0)),
                     'train_all_mean': float(np.nanmean(s_tr)),
                     'best': [[str(H[i][0]), H[i][1], float(s_tr[i]), float(v)] for i, v in zip(order[:8], s_te[:8])]}
    return res


def part_B(C, seed, body_only):
    rng = random.Random(seed)
    U = [(pi, ai, l) for pi, ai, l in line_units(C, body_only) if len(l) >= 6]
    sec = [C[pi]['sec'] for pi, ai, l in U]
    def content(l, a):
        return {w[:2] for t, w in enumerate(l) if 0 < t < len(l) - 1 and abs(t - a) >= 2}
    out = {}
    # (i) shared opener vs shared mid word: rest-of-line similarity, pairs from different pages, same section
    for kind in ('KEY', 'MID'):
        anc = []
        for pi, ai, l in U:
            a = 0 if kind == 'KEY' else rng.randint(2, len(l) - 3); anc.append((l[a], a))
        groups = defaultdict(list)
        for j, (w, a) in enumerate(anc): groups[(w, sec[j])].append(j)
        sims = []; base = []
        bysec = defaultdict(list)
        for j in range(len(U)): bysec[sec[j]].append(j)
        for (w, s), js in groups.items():
            if len(js) < 2: continue
            for x in range(len(js)):
                for y in range(x + 1, len(js)):
                    i, k = js[x], js[y]
                    if U[i][0] == U[k][0]: continue
                    A = content(U[i][2], anc[i][1]); B = content(U[k][2], anc[k][1])
                    sims.append(len(A & B) / max(1, len(A | B)))
                    k2 = rng.choice(bysec[s])
                    while U[k2][0] == U[i][0]: k2 = rng.choice(bysec[s])
                    B2 = content(U[k2][2], anc[k2][1])
                    base.append(len(A & B2) / max(1, len(A | B2)))
        out[f'sim_{kind}'] = [float(np.mean(sims)) if sims else None, float(np.mean(base)) if base else None, len(sims)]
    # (ii) section-boundness: same word as opener vs mid-line, equal counts
    secs = sorted(set(sec))
    if len(secs) > 1:
        op = defaultdict(list); md = defaultdict(list)
        for j, (pi, ai, l) in enumerate(U):
            op[l[0]].append(sec[j])
            for w in l[2:-1]: md[w].append(sec[j])
        d = []
        for w, s1 in op.items():
            s2 = md.get(w, [])
            if len(s1) >= 4 and len(s2) >= len(s1):
                def H(x):
                    c = np.array(list(Counter(x).values()), float); p = c / c.sum(); return float(-(p * np.log2(p)).sum())
                hm = np.mean([H(rng.sample(s2, len(s1))) for _ in range(20)])
                d.append(hm - H(s1))
        out['sec_bound'] = [float(np.mean(d)) if d else None, float(np.std(d) / max(1, np.sqrt(len(d)))) if d else None, len(d)]
    # (iii) key reuse: share of body lines whose opener opens lines on >= 3 pages; same for position 3 word
    for kind, pos in (('KEY', 0), ('POS3', 3)):
        pg = defaultdict(set)
        for pi, ai, l in U: pg[l[pos]].add(pi)
        out[f'reuse_{kind}'] = float(np.mean([len(pg[l[pos]]) >= 3 for pi, ai, l in U]))
    return out


if __name__ == '__main__':
  for name in sys.argv[1:]:
    t = time.time(); out = {}
    body = name.startswith('V')
    for seed in (0, 1):
        C = get(name, seed)
        out[f'A_s{seed}'] = part_A(C, seed, body)
        out[f'B_s{seed}'] = part_B(C, seed, body)
    save(f'c3_{name}.json', out)
    A = {k: np.mean([out[f'A_s{s}'][k]['test_top_mean'] for s in (0, 1)]) for k in ('KEY', 'MID', 'END')}
    R = {k: np.mean([out[f'A_s{s}'][k]['test_rand_mean'] for s in (0, 1)]) for k in ('KEY', 'MID', 'END')}
    B = out['B_s0']
    print(name, f'{time.time()-t:.0f}s', 'A held-out top40 LS-MI KEY %.2f MID %.2f END %.2f (random hyps %.2f %.2f %.2f)' % (A['KEY'], A['MID'], A['END'], R['KEY'], R['MID'], R['END']),
          '| B', json.dumps(B), flush=True)
