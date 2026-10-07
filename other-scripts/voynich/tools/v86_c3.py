"""v86 cycle 3: (a) power on the REAL Voynich layout: a deterministic meaning-preserving compression planted
into real line ends at p = 0.05/0.1/0.2/0.35 -> how much recoverable compression could hide in the observed
numbers; (b) the same with a habit plant; (c) section vocabulary (quire x Currier language) instead of the page,
for more recovery power; (d) mark enrichment (a glyph ADDED to a shortened word) per class."""
import os, random
from collections import Counter, defaultdict
from multiprocessing import Pool
import v86_lib as L

NR = int(os.environ.get('NR', 3000))
VP, META = L.voynich_pages('ZL3b')
VF = {g for g in L.alphabet_of(VP) if sum(w.count(g) for ls in VP.values() for ln in ls for w in ln['units'] if w) >= 200}


def relabel_pe(pages):
    return {f: [dict(ln, cls=[('Epe' if (c == 'E' and ln['para_end']) else c) for c in ln['cls']]) for ln in ls]
            for f, ls in pages.items()}


def by_section(pages, meta):
    g = defaultdict(list)
    for f, ls in pages.items():
        q, lang, il, h = meta[f]
        g['%s_%s' % (q, lang)].extend(ls)
    return dict(g)


def mark_enrich(r):
    """share of target unique recoveries that needed an added glyph, and the top glyph's enrichment vs control."""
    mt, mc = r['mark_t'], r['mark_c']
    st, sc = sum(mt.values()), sum(mc.values())
    out = []
    for g, n in mt.most_common(6):
        out.append((g, round(n, 1), round((n / max(st, 1e-9)) / max(mc[g] / max(sc, 1e-9), 1e-3), 2)))
    return out


def job(a):
    kind, p, seed = a
    rng = random.Random(seed)
    pages = relabel_pe(VP)
    if kind.startswith('det'):
        fn, D, k = L.make_det_compress(VF, seed=11 + seed)
        pages = L.plant(pages, fn, classes=('E',), p=p, seed=seed, para_end_exempt=False)
    elif kind.startswith('habit'):
        pages = L.plant(pages, L.make_habit(VF), classes=('E',), p=p, seed=seed, para_end_exempt=False)
    elif kind.startswith('pos'):
        pages = L.shuffle_positions(pages, rng)
    if kind.endswith('_sec'):
        pages = by_section(pages, META)
    rr = random.Random(86)
    R = [L.random_rule(rr, VF) for _ in range(NR)]
    res = {}
    for tg in (('E',), ('B',)):
        s = L.search(pages, R, target=tg, seed=0)
        r = L.subseq_recovery1(pages, list(pages), tg, boot=100) if (kind in ('real', 'real_sec', 'pos_sec') or p in (0.2,)) else None
        res['+'.join(tg)] = dict(held_top=s['held_top'], held_top5=s['held_top5'], top=[(x['rule'], round(x['held_delta'], 4)) for x in s['surv'][:3]],
                                 sub=None if r is None else dict(delta=r['delta'], ci=r['ci'], marks=mark_enrich(r)))
    L.save('c3_%s_%s_%d.json' % (kind, p, seed), res)
    return kind, p, seed, res


if __name__ == '__main__':
    jobs = [('real', 0, 0), ('real_sec', 0, 0), ('pos_sec', 0, 0), ('pos_sec', 0, 1), ('pos_sec', 0, 2)]
    for p in (0.05, 0.1, 0.2, 0.35):
        for s in (0, 1):
            jobs.append(('det', p, s)); jobs.append(('habit', p, s))
    jobs += [('det_sec', 0.1, 0), ('habit_sec', 0.1, 0)]
    with Pool(2) as pl:
        for kind, p, seed, res in pl.imap_unordered(job, jobs):
            for tg, r in res.items():
                line = '%s p%s s%d %s held_top %.4f top5 %.4f top %s' % (kind, p, seed, tg, r['held_top'], r['held_top5'], r['top'][:2])
                if r['sub']:
                    line += ' | sub d %.4f [%.4f,%.4f] marks %s' % (r['sub']['delta'], r['sub']['ci'][0], r['sub']['ci'][1], r['sub']['marks'][:4])
                print(line, flush=True)
