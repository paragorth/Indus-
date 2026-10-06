"""v76 cycle 4: kill test of the C- identifier-stream signal on a third, independent transcription
(Glen Claston's GC2a, converted v101 -> EVA in v74: data/v74_ckpt/GC_lines.json), with the same pipeline:
PARCOPY fit, 7 generators x 2 seeds, 8,000 shared templates (cycle 1), d-stripped alignment (cycle 2b), cycle 3."""
import os, sys, json, re, collections, time
os.environ.setdefault('V76_T', '8000')
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v76_lib as V
L = V.L
import vlib


def gc_pages():
    recs = json.load(open(os.path.join(V.ROOT, 'data', 'v74_ckpt', 'GC_lines.json')))
    zl_sec = {p['id']: p['sec'] for p in L.voynich('ZL3b')}      # same section labels as ZL (page illustration)
    pages = collections.OrderedDict()
    for r in recs:
        if r['ltype'] != 'P': continue
        ws = [''.join(vlib.glyphs(w)) for w in r['words'] if re.fullmatch(r'[a-z]+', w)]
        if not ws: continue
        p = pages.setdefault(r['folio'], dict(id=r['folio'], sec=zl_sec.get(r['folio'], r['illus']), lang=r['lang'] or '-',
                                              hand=r['hand'] or '-', quire=r['quire'], lines=[]))
        p['lines'].append(dict(w=ws, ps=bool(r['para_start'])))
    return [p for p in pages.values() if sum(len(l['w']) for l in p['lines']) >= 20]


def main():
    import v76_c1 as C1
    C1.T = 8000
    surf = gc_pages()
    print('GC pages', len(surf), 'paragraphs', len(V.extracted_paragraphs(surf)), flush=True)
    fits = json.load(open(os.path.join(V.CK, 'c1_fits.json')))
    if 'GC' not in fits:
        fits['GC'] = C1.fit_parcopy(surf, V.extracted_paragraphs(surf)); json.dump(fits, open(os.path.join(V.CK, 'c1_fits.json'), 'w'))
    f = fits['GC']; print('fit', f, flush=True)
    jobs = [('GC', surf)]
    for g in ('WSHUF', 'MK2', 'SELFCIT', 'JUNC', 'SC10'):
        for s in C1.SEEDS: jobs.append(('GC__%s_%d' % (g, s), V.GENS[g](surf, seed=s)))
    for s in C1.SEEDS:
        jobs.append(('GC__PARCOPY_%d' % s, V.gen_parcopy(surf, seed=s, c=f['c'], m=f['m'])))
        jobs.append(('GC__PARCOPYID_%d' % s, V.gen_parcopy(surf, seed=s, c=f['c'], m=f['m'], idslot=True)))
    t0 = time.time()
    with Pool(2) as pool:
        for nm in pool.imap_unordered(C1.score_corpus, jobs): print('c1 done', nm, round(time.time() - t0), flush=True)
    os.environ['V76_C2TAG'] = 'c2b'
    import v76_c2 as C2
    C2.TAG = 'c2b'
    names = [j[0] for j in jobs]
    with Pool(2) as pool:
        for nm in pool.imap_unordered(C2.job, names): print('c2b done', nm, flush=True)
    import v76_c3 as C3
    with Pool(2) as pool:
        for nm in pool.imap_unordered(C3.job, names): print('c3 done', nm, flush=True)


if __name__ == '__main__':
    main()
