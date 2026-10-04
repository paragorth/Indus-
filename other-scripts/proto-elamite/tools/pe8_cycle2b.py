"""pe8 cycle 2b: positive control for test 1.  Run the same form->system predictor on PLANTED corpora (forms fix
the entry system set, so FORM must beat the marginal there), and on NULL corpora (it must not).
Output: data/pe8_cycle2b.json"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pe8_common as C
import pe8_cycle2 as P2

def run(kind, seed, K):
    R0 = C.load_skeletons(min_ent=2, clean=False)
    A = [r['A'] for r in R0]
    G = C.planted_corpus(A, len(A), seed=seed)[0] if kind == 'PLANT' else C.null_corpus(A, len(A), seed=seed)
    rows = [{'id': '%s%d_%d' % (kind, seed, i), 'f': C.features(a), 'ents': a['ents'], 'meta': {}, 'prov': ''}
            for i, a in enumerate(G)]
    P2.load_skeletons = lambda min_ent=2, clean=False: rows
    r = P2.predict(('main', K, False, 0)); r['corpus'] = '%s_%d' % (kind, seed)
    return r

if __name__ == '__main__':
    c1 = {r['name']: r for r in json.load(open(os.path.join(C.DATA, 'pe8_cycle1.json')))}
    out = [run('PLANT', s, c1['PLANT_%d' % s]['K_bic']) for s in (1, 2, 3)] + [run('NULL', s, 5) for s in (1, 2)]
    json.dump(out, open(os.path.join(C.DATA, 'pe8_cycle2b.json'), 'w'), indent=1)
