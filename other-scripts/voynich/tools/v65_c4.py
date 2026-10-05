"""v65 cycle 4: power. How much adjacency evidence does one page pair carry, and does
whole-page similarity (more power, no direction) agree with the physical claims?

(a) For every binding-adjacent page pair: pct of its score among all directed pairs on
    different leaves of the same quire; mean pct by section (herbal H, bio B, stars S, other).
    Measures: seam nl=3, seam nl=6, page (whole-page, symmetric).
(b) Physical claims E1-E4 (see v65_c3) with the same three measures; pooled mean pct of the
    claims, against the calibration texts poured in the claimed order.
(c) E5 restricted to the herbal quires Q1-Q7 (one section): quire order by signatures vs 10,000
    random orders of the same quires.
"""
import os, sys, json, random
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import v65_lib as L
import v65_c3 as C3

quires, _ = L.structure(); QD = dict(quires)


def mats(pages, meta, feat):
    keys = sorted(k for k in pages if sum(len(x) for x in pages[k]) >= L.MIN_WORDS and k in meta)
    groups = [str(meta[k]['sec']) + str(meta[k]['lang']) for k in keys]
    Js3, Jp = L.carry_matrices(pages, keys, nl=3, feat=feat)
    Js6, _ = L.carry_matrices(pages, keys, nl=6, feat=feat)
    return keys, {'seam3': L.centre(Js3, groups), 'seam6': L.centre(Js6, groups), 'page': L.centre(Jp, groups)}


def pct_pair(W, keys, kidx, a, b, qn):
    ql = set(x for bf in QD[qn] for x in bf[1:] if x)
    cand = [kidx[k] for k in keys if k[0] in ql]
    vals = np.array([W[x, y] for x in cand for y in cand if keys[x][0] != keys[y][0]])
    return float((vals < W[kidx[a], kidx[b]]).mean())


def run(pages, meta, feat, rng):
    keys, M = mats(pages, meta, feat)
    kidx = {k: i for i, k in enumerate(keys)}
    out = {}
    # (a) binding-adjacent pairs within quires
    pairs = []
    for qn, bifs in quires:
        seq = L.seq_for(bifs, tuple(range(len(bifs))), (0,) * len(bifs))
        for x, y in zip(seq[:-1], seq[1:]):
            if x != L.GAP and y != L.GAP and x in kidx and y in kidx and x[0] != y[0]:
                pairs.append((x, y, qn, meta[x]['sec']))
    for m, W in M.items():
        bysec = {}
        for x, y, qn, s in pairs:
            bysec.setdefault(s if s in 'HBS' else 'o', []).append(pct_pair(W, keys, kidx, x, y, qn))
        out['adj_' + m] = {s: (round(float(np.mean(v)), 3), len(v)) for s, v in bysec.items()}
        allv = [p for v in bysec.values() for p in v]
        out['adj_' + m]['all'] = (round(float(np.mean(allv)), 3), len(allv))
        # (b) claims
        cl = {}
        for cn, (a, b, qn) in C3.CLAIMS.items():
            if a in kidx and b in kidx:
                cl[cn] = round(pct_pair(W, keys, kidx, a, b, qn), 3)
        cl['mean_E1_E3'] = round(float(np.mean([cl[c] for c in ('E1', 'E2', 'E3') if c in cl])), 3)
        cl['mean_all'] = round(float(np.mean([v for c, v in cl.items() if c.startswith('E')])), 3)
        out['claims_' + m] = cl
        # (c) herbal quire order
        qf, qlst = [], []
        for qn, bifs in quires:
            if qn not in 'ABCDEFG':
                continue
            seq = [k for k in L.seq_for(bifs, tuple(range(len(bifs))), (0,) * len(bifs)) if k != L.GAP and k in kidx]
            qf.append(kidx[seq[0]]); qlst.append(kidx[seq[-1]])
        n = len(qf)
        sc = lambda p: sum(W[qlst[p[i]], qf[p[i + 1]]] for i in range(n - 1))
        obs = sc(list(range(n))); nul = []
        for _ in range(10000):
            p = list(range(n)); rng.shuffle(p); nul.append(sc(p))
        out['E5herb_' + m] = round(float((np.array(nul) < obs).mean()), 3)
    return out


def main():
    rng = random.Random(654)
    vp, vmeta = L.voynich_pages('ZL3b'); ip, imeta = L.voynich_pages('IT2a')
    isid = L.isidore_words()[5000:]; kon = [w for e in L.konrad_entries() for w in e]
    co = C3.claimed_order()
    runs = {'VOY_ZL|word': (vp, vmeta, 'word'), 'VOY_ZL|skel': (vp, vmeta, 'skel'),
            'VOY_IT|word': (ip, imeta, 'word'), 'VOY_IT|skel': (ip, imeta, 'skel')}
    for s in range(4):
        runs[f'CAL_isid_s{s}|skel'] = (L.pour(vp, isid[s * 9000:], 'flow', 40 + s, quires, order=co), vmeta, 'skel')
        runs[f'CAL_kon_s{s}|skel'] = (L.pour(vp, kon[s * 3000:] + kon[:s * 3000], 'flow', 50 + s, quires, order=co), vmeta, 'skel')
    for s in range(2):
        runs[f'NULL_markov_s{s}|word'] = (L.markov_pages(vp, vmeta, 60 + s), vmeta, 'word')
        runs[f'NULL_entry_kon_s{s}|skel'] = (L.pour(vp, L.konrad_entries()[s * 7:], 'entry', 70 + s, quires, order=co), vmeta, 'skel')
    out = {}
    for name, (p, m, f) in runs.items():
        r = run(p, m, f, rng); out[name] = r
        print(name, json.dumps(r), flush=True)
    json.dump(out, open(os.path.join(L.CK, 'c4.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
