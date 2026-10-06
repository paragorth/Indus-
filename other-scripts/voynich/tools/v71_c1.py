"""v71 cycle 1: fingerprints of reference chunks (encoded, laid out), Voynich chunks (ZL3b, IT2a)
and generator nulls (self-citation, table-and-grille, word-bigram Markov) on the Voynich chunks
and on some reference chunks.  Results: data/v71_ckpt/c1/<job>.json.  2 workers."""
import os, sys, json, random, time
from multiprocessing import Pool
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import v71_lib as L

OUTD = os.path.join(L.CK, 'c1'); os.makedirs(OUTD, exist_ok=True)
NMOD = int(os.environ.get('V71_NMOD', '360'))
MODELS = L.model_types(NMOD)

def jobs():
    R = json.load(open(os.path.join(L.CK, 'refs.json')))
    J = []
    for k in sorted(R):
        rng = random.Random('ref' + k)
        for i, ch in enumerate(L.ref_chunks(R, k, rng, nmax=4)):
            J.append(('ref__%s__%d' % (k, i), {'kind': 'ref', 'text': k, 'genre': R[k]['genre'], 'coarse': R[k]['coarse']}, ch))
    gen_src = ['forme_of_cury', 'culpeper', 'caesar', 'apicius_index', 'circa_fr', 'celsus_lat']
    for k in gen_src:
        ch = [j for j in J if j[1].get('text') == k][0][2]
        for g, fn in [('selfcit', L.gen_selfcit), ('grille', L.gen_grille), ('markov', L.gen_markov)]:
            J.append(('gen__%s_%s__0' % (g, k), {'kind': 'gen', 'gen': g, 'src': k}, fn(ch, 7)))
    for tr in ('ZL3b', 'IT2a'):
        V = L.voynich_sections(tr)
        for sec, pages in V.items():
            for i, ch in enumerate(L.split_chunks(pages)):
                J.append(('voy__%s_%s__%d' % (tr, sec, i), {'kind': 'voy', 'tr': tr, 'sec': sec}, ch))
                if tr == 'ZL3b':
                    for g, fn in [('selfcit', L.gen_selfcit), ('grille', L.gen_grille), ('markov', L.gen_markov)]:
                        J.append(('gen__%s_V%s__%d' % (g, sec, i), {'kind': 'gen', 'gen': g, 'src': 'V' + sec}, fn(ch, 11 + i)))
    return J

def run(job):
    name, meta, ch = job
    out = os.path.join(OUTD, name + '.json')
    if os.path.exists(out): return name, 'skip'
    t = time.time()
    res = L.fingerprint(ch, MODELS)
    fp = L.summarise(res, MODELS)
    meta = dict(meta); meta.update({'name': name, 'words': L.nwords(ch), 'pages': len(ch),
                                    'paras': sum(len(p) for p in ch), 'lines': sum(len(a) for p in ch for a in p)})
    json.dump({'meta': meta, 'fp': fp, 'res': [[round(r['G'], 5)] + [round(r[s], 5) for s in L.SCALES] for r in res]}, open(out, 'w'))
    return name, '%.0fs' % (time.time() - t)

if __name__ == '__main__':
    J = jobs()
    print('jobs', len(J), flush=True)
    with Pool(2) as p:
        for name, st in p.imap_unordered(run, J):
            print(name, st, flush=True)
    print('DONE', flush=True)
