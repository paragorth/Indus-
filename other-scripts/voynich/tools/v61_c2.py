"""v61 cycle 2: are the coupled alternants one word? (far-side identity test) and does undoing them help page prediction?

Far-side test: for a rule (S,B,C) the tokens stem+S and stem+B (same stems) are labelled 1/0. Sandhi variants of one word
differ only on the junction side; on the far side (R: previous word's last glyph; P: next word's first glyph), in line
position and in page section they should be indistinguishable. Measured as conditional MI I(label; feature | stem),
permutation-corrected (labels shuffled within stem), for the top rules versus the least coupled ending pairs of the same scan.
"""
import sys, json, math, random
from collections import Counter, defaultdict
from multiprocessing import Pool
import numpy as np
import v61_lib as L
from v61_c1 import build


def cond_mi(groups):
    """groups: list of (labels list, feats list) per stem. Pooled conditional MI in bits."""
    N = sum(len(l) for l, f in groups); tot = 0.0
    for lab, fe in groups:
        n = len(lab)
        if n < 2:
            continue
        c = Counter(zip(lab, fe)); a = Counter(lab); b = Counter(fe)
        tot += sum(v * math.log2(v * n / (a[x] * b[y])) for (x, y), v in c.items())
    return tot / max(1, N)


def far_mi(groups, rng, reps=20):
    obs = cond_mi(groups)
    nul = []
    for _ in range(reps):
        g2 = []
        for lab, fe in groups:
            l2 = lab[:]; rng.shuffle(l2); g2.append((l2, fe))
        nul.append(cond_mi(g2))
    return obs - float(np.mean(nul))


def rule_groups(T, lex, S, B, feat):
    """Collect per-stem label/feature lists for medial tokens of stem+S (1) and stem+B (0)."""
    by = defaultdict(lambda: ([], []))
    for t in T:
        w = t['w']
        if t['nxt'] is None:
            continue
        for lab, e, o in ((1, S, B), (0, B, S)):
            if (w.endswith(e) if e else True) and len(w) > len(e):
                st = w[:len(w) - len(e)]
                if st + o in lex:
                    g = by[st]; g[0].append(lab); g[1].append(feat(t))
    return [v for v in by.values() if len(set(v[0])) == 2]


def identity_test(lines, recs, rng, ntop=40, nbase=40):
    T = L.tokens(lines)
    prev = {}
    # previous word's last glyph
    for i, t in enumerate(T):
        t['prv'] = T[i - 1]['w'][-1] if t['pos'] > 0 else '#'
    lex = {t['w'] for t in T}
    feats = {'far': lambda t: t['prv'], 'pos': lambda t: min(t['pos'], 4), 'sec': lambda t: t['sec'],
             'page': lambda t: t['page']}
    good = [r for r in recs if r['z_test'] > 3][:ntop]
    base = sorted(recs, key=lambda r: r['z_train'])[:nbase]
    out = {}
    for nm, rs in (('rules', good), ('uncoupled', base)):
        vals = defaultdict(list)
        for r in rs:
            for fn, f in feats.items():
                gr = rule_groups(T, lex, r['S'], r['B'], f)
                vals[fn].append(far_mi(gr, rng, reps=10))
        out[nm] = {k: float(np.median(v)) for k, v in vals.items()}
        out[nm]['n'] = len(rs)
        out[nm]['z_train_med'] = float(np.median([r['z_train'] for r in rs]))
    return out


def job(arg):
    name, d = arg
    lines = build(name)
    if d == 'P':
        lines = L.reverse_text(lines)
    recs = L.jload('c1_%s_%s.json' % (name.replace(':', '_'), d))['recs']
    rng = random.Random(7)
    res = identity_test(lines, recs, rng)
    # gold check for controls: is the far side indistinguishable for TRUE alternants (surface vs base)?
    res['name'] = name; res['dir'] = d
    L.jsave('c2_id_%s_%s.json' % (name.replace(':', '_'), d), res)
    return res


if __name__ == '__main__':
    jobs = [('Sanskrit', 'R'), ('Italian', 'R'), ('Welsh', 'P'), ('VMS-planted', 'R'), ('VMS-ZL', 'R'), ('VMS-ZL', 'P'),
            ('VMS-IT', 'R'), ('VMS-IT', 'P'), ('null-pairblind:VMS-ZL', 'R'), ('null-pairblind:VMS-ZL', 'P')]
    if len(sys.argv) > 1:
        jobs = [j for j in jobs if j[0] in sys.argv[1:]]
    with Pool(2) as p:
        for r in p.imap_unordered(job, jobs):
            print(json.dumps(r), flush=True)
