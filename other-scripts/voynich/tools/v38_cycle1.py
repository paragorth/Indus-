"""v38 cycle 1: Voynich herbal pages. Does drawing similarity predict text similarity beyond
language, hand, quire, bifolio, leaf, adjacency, page distance, text length and drawing size?
Controls: planted visual->vocabulary link at several strengths; page-shuffled text.
Writes data/v38_ckpt/c1.json and loops/v38_cycle1.txt.
"""
import os, sys, json, re
import numpy as np
from collections import Counter
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v38_lib import *
from v8_lib import voynich_pages

NPERM = int(os.environ.get('NPERM', 2000))


def voynich_setup(min_tok=30):
    vis = json.load(open(os.path.join(DER, 'v38_vis_voynich.json')))
    pages = [p for p in voynich_pages(min_tokens=min_tok) if p['illus'] == 'H' and p['id'] in vis]
    keys = [p['id'] for p in pages]
    words = [[w for l in p['lines'] for w in l] for p in pages]
    n = len(pages)
    eq = lambda a: np.array([[float(a[i] == a[j] and a[i] is not None) for j in range(n)] for i in range(n)])
    order = np.array([p['order'] for p in pages], float)
    D = np.abs(order[:, None] - order[None, :])
    ln = np.log([len(w) for w in words])
    ar = np.log([vis[k]['area'] + 1e-3 for k in keys])
    conf = dict(lang=eq([p['lang'] for p in pages]), hand=eq([p['hand'] for p in pages]),
                quire=eq([p['quire'] for p in pages]),
                bifolio=eq([(p['quire'], p['bifolio']) if p['bifolio'] else None for p in pages]),
                leaf=eq([p['leafnum'] for p in pages]), adj=(D == 1).astype(float), logdist=np.log1p(D),
                lensum=ln[:, None] + ln[None, :], lendiff=np.abs(ln[:, None] - ln[None, :]),
                areasum=ar[:, None] + ar[None, :], areadiff=np.abs(ar[:, None] - ar[None, :]))
    strata = ['%s%s' % (p['lang'], p['hand']) for p in pages]
    return pages, keys, words, vis, conf, strata


def fmt_table(res):
    s = []
    for f, row in zip(res['fams'], res['obs']):
        s.append('%s: ' % f + ' '.join('%s %+.3f (z %+.1f)' % (m, r, res['cell_z']['%s|%s' % (f, m)])
                                         for m, r in zip(res['mets'], row)))
    return '; '.join(s)


if __name__ == '__main__':
    rng = np.random.default_rng(38)
    pages, keys, words, vis, conf, strata = voynich_setup()
    n = len(pages)
    print('pages', n, Counter(strata))
    V = vis_sims(vis, keys)
    T = text_sims(text_profiles(words))
    part = Partial(list(conf.values()), n)
    part0 = Partial([], n)
    rows, out = [], {'n': n, 'keys': keys}
    # raw (no confounds, free permutation)
    r_raw = mantel_table(V, T, part0, NPERM // 2, rng)
    out['raw'] = r_raw
    rows.append(('V-38.1', 'Raw Mantel, %d herbal pages, 6 visual families x 5 text metrics, free page permutation (%d)' % (n, NPERM // 2),
                 'omnibus mean r %+.4f, z %+.1f, p %.4f; max cell %+.3f p_max %.3f; family z %s' % (
                     r_raw['omni'], r_raw['omni_z'], r_raw['omni_p'], r_raw['max'], r_raw['max_p'],
                     ', '.join('%s %+.1f' % kv for kv in r_raw['fam_z'].items())),
                 'raw link (confounded)' if r_raw['omni_p'] < 0.01 else 'no raw link'))
    # partial, stratified
    r_p = mantel_table(V, T, part, NPERM, rng, strata)
    out['partial'] = r_p
    rows.append(('V-38.2', 'PRIMARY. Partial Mantel: residualise both matrices on same language, hand, quire, bifolio, leaf, adjacency, log page distance, text length (sum, diff), drawing area (sum, diff); permute pages within language x hand (%d)' % NPERM,
                 'omnibus %+.4f, z %+.1f, p %.4f; max cell %+.3f p_max %.3f; family z %s; text z %s' % (
                     r_p['omni'], r_p['omni_z'], r_p['omni_p'], r_p['max'], r_p['max_p'],
                     ', '.join('%s %+.1f' % kv for kv in r_p['fam_z'].items()),
                     ', '.join('%s %+.1f' % kv for kv in r_p['met_z'].items())),
                 'PASS' if r_p['omni_p'] < 0.01 else 'no link beyond confounds'))
    out['partial_table'] = fmt_table(r_p)
    # planted controls
    allw = Counter(w for ws in words for w in ws)
    pool = [w for w, c in allw.items() if 5 <= c <= 40]
    plant_rows = []
    for src_fam in ['dino', 'shape']:
        Zsrc = np.array([vis[k][src_fam] for k in keys], float)
        if src_fam == 'dino':
            U, S, _ = np.linalg.svd(Zsrc - Zsrc.mean(0), full_matrices=False)
            Zsrc = U[:, :10] * S[:10]
        for frac in [0.03, 0.06, 0.1, 0.2]:
            res = []
            for rep in range(3):
                pw = plant_text(words, Zsrc, frac, rng, pool)
                Tp = text_sims(text_profiles(pw))
                rr = mantel_table(V, Tp, part, 400, rng, strata)
                res.append((rr['omni_z'], rr['omni_p'], rr['fam_z']))
            plant_rows.append((src_fam, frac, res))
            print('planted', src_fam, frac, [(round(a, 1), b) for a, b, _ in res], flush=True)
    out['planted'] = [(a, b, [(x, y, z) for x, y, z in c]) for a, b, c in plant_rows]
    rows.append(('V-38.3', 'PLANTED control: %d%%..%d%% of each page\'s tokens replaced by words drawn from a softmax over %d Voynich words with logits linear in the page\'s visual vector (source: DINO top-10 PCs, or silhouette shape); 3 replicates x 400 perms, same partial test' % (3, 20, len(pool)),
                 '; '.join('%s %d%%: omni z %s (p %s)' % (s, round(f * 100), '/'.join('%.1f' % a for a, _, _ in r), '/'.join('%.3f' % b for _, b, _ in r)) for s, f, r in plant_rows),
                 'sensitivity floor = smallest fraction with p<0.01 in 3/3'))
    # page-shuffled text
    sh = []
    for rep in range(10):
        p = rng.permutation(n)
        Ts = text_sims(text_profiles([words[i] for i in p]))
        rr = mantel_table(V, Ts, part, 300, rng, strata)
        sh.append(rr['omni_z'])
    out['shuffled'] = sh
    rows.append(('V-38.4', 'SHUFFLED control: text pages permuted globally (10 shuffles), same partial test', 'omni z: ' + ' '.join('%+.1f' % z for z in sh) + ' (mean %+.2f, sd %.2f)' % (np.mean(sh), np.std(sh)),
                 'null behaves' if abs(np.mean(sh)) < 1 else 'null miscalibrated'))
    json.dump(out, open(os.path.join(CK, 'c1.json'), 'w'), default=str)
    with open(os.path.join(LOOPS, 'v38_cycle1.txt'), 'w') as fh:
        fh.write('# v38 cycle 1 - similar plants, similar text? (Voynich herbal pages)\n')
        fh.write('| id | method and control | result | verdict |\n|---|---|---|---|\n')
        for r in rows:
            fh.write('| %s | %s | %s | %s |\n' % r)
        fh.write('\nPartial-Mantel cells (r, z): ' + out['partial_table'] + '\n')
    print(open(os.path.join(LOOPS, 'v38_cycle1.txt')).read())
