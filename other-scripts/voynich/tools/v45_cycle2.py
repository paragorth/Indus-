"""v45 cycle 2: does the residual track the drawings better than the surface does?

v38 found a faint Voynich text~drawing link (z ~3) on 119 herbal pages, with no held-out prediction.
If the rules mask a message, the residual (observed minus rule-expected word usage) should track the
drawings MORE strongly than the raw surface. Partial Mantel with the v38 confounds (language, hand, quire,
bifolio, leaf, adjacency, page distance, text length, drawing area), pages permuted within language x hand,
pre-registered visual composite = EfficientNet-B0 + DINOv2 (v38 'fresh').
Positive control: Gerard's Herball text pushed through the planted verbose encodings (GEw, GEl), residual
vs woodcuts. Negative: the generator (GEN0, GEN1) and the richer generator (PL) on the Voynich herbal pages.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v45_lib import *
from v38_lib import Partial, mantel_table, text_sims, text_profiles
import v38_cycle2
v38_cycle2.CACHE = os.path.join(SCR, 'v38img')
from v38_cycle1 import voynich_setup
from v38_cycle2 import gerard_setup

FN = 'v45_cycle2.txt'
NPERM = int(os.environ.get('NPERM', 2000))


def emb(src, keys):
    v2 = json.load(open(os.path.join(vlib.DER if hasattr(vlib, 'DER') else os.path.join(vlib.DATA, 'derived'), 'v38_vis2_%s.json' % src)))
    out = {}
    for f in ('effb0', 'dinov2'):
        F = np.array([v2[k][f] for k in keys], float); out[f] = cos(F - F.mean(0))
    return out


def sims_for(name, ids_wanted):
    C = get_corpus(name); R = residual(name)
    ids, voc, Om, Em, Rm = resid_matrix(C, R, min_tot=2)
    pos = {i: j for j, i in enumerate(ids)}
    ix = [pos[i] for i in ids_wanted]
    O, Rr, E = Om[ix], Rm[ix], Em[ix]
    keep = (O.sum(0) > 0)
    O, Rr, E = O[:, keep], Rr[:, keep], E[:, keep]
    # residual restricted to words that occur on >= 2 of these pages
    df = (O > 0).sum(0)
    return dict(raw=cos(tfidf(O)), resid=cos(Rr), resid_shared=cos(Rr[:, df >= 2]),
                expected=cos(tfidf(E)))


def run(rng, name, V, keys, ids, part, strata):
    T = sims_for(name, ids)
    out = {}
    for k, S in T.items():
        r = mantel_table(V, {k: S}, part, NPERM, rng, strata)
        out[k] = dict(r=r['omni'], z=r['omni_z'], p=r['omni_p'])
    return out


if __name__ == '__main__':
    rng = np.random.default_rng(452)
    res = jload('c2_' + '_'.join(sys.argv[1:]) + '.json') or {}
    pages, keys, words, vis, conf, strata = voynich_setup()
    Vv = emb('voynich', keys)
    part = Partial(list(conf.values()), len(keys))
    have = set(get_corpus('V')[i]['id'] for i in range(len(get_corpus('V'))))
    sel = [i for i, k in enumerate(keys) if k in have]
    keys2 = [keys[i] for i in sel]
    conf2 = {k: v[np.ix_(sel, sel)] for k, v in conf.items()}
    part2 = Partial(list(conf2.values()), len(sel)); strata2 = [strata[i] for i in sel]
    Vv2 = {k: v[np.ix_(sel, sel)] for k, v in Vv.items()}
    print('voynich herbal pages', len(keys2), flush=True)
    names = sys.argv[1:] or ['V', 'GEN0', 'GEN1', 'PL', 'VI', 'GEw', 'GEl']
    for nm in names:
        if nm.startswith('GE') and not nm.startswith('GEN'):
            gk, gw, gvis, gconf = gerard_setup()
            gV = emb('gerard', gk); gpart = Partial(list(gconf.values()), len(gk))
            out = run(rng, nm, gV, gk, ['ge' + k for k in gk], gpart, None)
            if 'GEraw' not in res:
                T = text_sims(text_profiles(gw))
                r = mantel_table(gV, {'tfidf': T['tfidf']}, gpart, NPERM, rng)
                res['GEraw'] = dict(tfidf=dict(r=r['omni'], z=r['omni_z'], p=r['omni_p']))
                print('GEraw', res['GEraw'], flush=True)
        else:
            ids = keys2
            if nm == 'VI':
                hv = set(p['id'] for p in get_corpus('VI'))
                if not all(k in hv for k in ids): ids = [k for k in ids if k in hv]
            if len(ids) != len(keys2):
                s3 = [keys2.index(k) for k in ids]
                pp = Partial([c[np.ix_(s3, s3)] for c in conf2.values()], len(s3))
                out = run(rng, nm, {k: v[np.ix_(s3, s3)] for k, v in Vv2.items()}, ids, ids, pp, [strata2[i] for i in s3])
            else:
                out = run(rng, nm, Vv2, keys2, ids, part2, strata2)
        res[nm] = out; jsave('c2_' + '_'.join(sys.argv[1:]) + '.json', res)
        print(nm, ' '.join('%s r %+.4f z %+.2f p %.4f' % (k, v['r'], v['z'], v['p']) for k, v in out.items()), flush=True)
