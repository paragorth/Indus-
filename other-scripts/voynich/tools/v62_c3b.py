"""v62 cycle 3b: refrains proper. Cycle 3's skeleton 'recurrences' were runs of the majority label
(clustering of light/heavy lines, paragraph structure), present in prose as well. Here a refrain must
(i) recur at a fixed gap >= 2 lines (A x A x A), (ii) be distinctive: labels covering > 30% of a page's
lines are masked. Score = (observed - shuffle mean) / shuffle sd over 200 within-page shuffles.
Definitions: whole word / last-2 / first-2 glyphs at line start and end, whole-line skeletons
(word-length classes of first three and last two words), and 300 random definitions (family-wise max
on ZL training vs the same on Markov and shuffled copies). Held-out pages and IT2a.
"""
import sys, os, json, random
import numpy as np
import numba
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v62_lib as L
import v62_c3 as C3

A = C3.A
NSH = 200


@numba.njit(cache=True)
def eqgaps2(lab, starts, ends):
    c = 0
    for p in range(len(starts)):
        s, e = starts[p], ends[p]
        for i in range(s, e):
            if lab[i] < 0: continue
            j = -1
            for x in range(i + 1, min(e, i + 9)):
                if lab[x] == lab[i]:
                    j = x; break
            if j < 0: continue
            g = j - i
            if g < 2: continue
            if j + g < e and lab[j + g] == lab[i]:
                ok = True
                for x in range(j + 1, j + g):
                    if lab[x] == lab[i]:
                        ok = False; break
                if ok: c += 1
    return c


@numba.njit(cache=True)
def null2(lab, starts, ends, nsh, seed):
    np.random.seed(seed)
    out = np.zeros(nsh); w = lab.copy()
    for k in range(nsh):
        for p in range(len(starts)):
            s, e = starts[p], ends[p]
            for i in range(e - 1, s, -1):
                j = s + np.random.randint(0, i - s + 1)
                t = w[i]; w[i] = w[j]; w[j] = t
        out[k] = eqgaps2(w, starts, ends)
    return out


def mask(lab, s, e):
    lab = lab.copy()
    for a, b in zip(s, e):
        seg = lab[a:b]; v = seg[seg >= 0]
        if len(v) == 0: continue
        u, c = np.unique(v, return_counts=True)
        for x in u[c > 0.3 * (b - a)]:
            seg[seg == x] = -1
        lab[a:b] = seg
    return lab


def z(pages, f, pos, seed=1):
    lab, s, e = C3.labels(pages, f, pos) if pos != 'skeleton' else C3.skel_labels(pages, f)
    lab = mask(lab, s, e)
    o = eqgaps2(lab, s, e); n = null2(lab, s, e, NSH, seed)
    return float((o - n.mean()) / (n.std() + 0.5)), int(o), float(n.mean())


def main():
    zl_tr, zl_te = C3.split('V-ZL3b'); it_tr, it_te = C3.split('V-IT2a')
    A['Litany-VR(period2)'] = dict(pages=C3.litany_vr(), kind='litany')
    sets = {'ZL-train': zl_tr, 'ZL-test': zl_te, 'IT-test': it_te,
            'markov': L.markov_line_generator(zl_tr, random.Random(8)), 'shuffled': L.shuffle_lines_within_page(zl_tr, random.Random(8))}
    for n in ['Litany(refrain)', 'Litany-VR(period2)', 'Dante(verse,terza)', 'Regimen(verse,rhymed)', 'Macer(verse,hexam)', 'Hildegard(prose herbal)', 'Caesar(prose)']:
        sets[n] = A[n]['pages']
    nat = [('word',), ('end', None, 2), ('beg', None, 2), ('len', np.array([8.5])), ('len', np.array([3.5, 8.5]))]
    out = {'natural': {}, 'family': {}}
    for tag, pages in sets.items():
        row = {}
        for d in nat:
            f = C3.sig_fn(d)
            for pos in ('first', 'last', 'second'):
                if d[0] == 'len' and pos != 'last': continue
                row[f'{C3.dstr(d)}@{pos}'] = round(z(pages, f, pos)[0], 2)
            if d[0] in ('len', 'end', 'beg'):
                row[f'{C3.dstr(d)}@skeleton'] = round(z(pages, f, 'skeleton')[0], 2)
        out['natural'][tag] = row
        print('nat', tag, row, flush=True)
    alpha = L.alphabet(A['V-ZL3b']['pages'])
    defs = C3.make_defs(alpha, 300, random.Random(9))
    best = {}
    for tag in ('ZL-train', 'markov', 'shuffled'):
        sc = []
        for d in defs:
            f = C3.sig_fn(d)
            sc.append(max(z(sets[tag], f, p, 3)[0] for p in ('first', 'last', 'skeleton')))
        sc = np.array(sc); i = int(sc.argmax())
        out['family'][tag] = dict(max=round(float(sc.max()), 2), p95=round(float(np.percentile(sc, 95)), 2), best=C3.dstr(defs[i]))
        best[tag] = (sc, i)
        print('fam', tag, out['family'][tag], flush=True)
    sc, _ = best['ZL-train']
    ho = []
    for i in np.argsort(-sc)[:8]:
        f = C3.sig_fn(defs[i])
        ho.append(dict(df=C3.dstr(defs[i]), train=round(float(sc[i]), 2),
                       zl_test=round(max(z(zl_te, f, p, 4)[0] for p in ('first', 'last', 'skeleton')), 2),
                       it_test=round(max(z(it_te, f, p, 5)[0] for p in ('first', 'last', 'skeleton')), 2),
                       markov_test=round(max(z(sets['markov'], f, p, 6)[0] for p in ('first', 'last', 'skeleton')), 2)))
    out['heldout'] = ho
    print('heldout', ho, flush=True)
    json.dump(out, open(os.path.join(L.CK, 'cycle3b.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
