"""v45 cycle 6: the off-table words. 30% of Voynich words (42% on paragraph-first lines) lie outside every
choice the rule model allows (pm = 0). If free content lives anywhere, it is there. Partial Mantel (v38
confounds, 119 herbal pages, EfficientNet-B0 + DINOv2) of drawings vs tf-idf of: off-table words, in-table
words, off-table words on paragraph-first lines, off-table words on body lines. Negatives GEN0/GEN1/PL; positive
GEw2000 (Gerard woodcuts)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v45_cycle2 import *

def views(name, ids):
    R = residual(name); out = {}
    sel = dict(off=lambda r: r[9] == 0, on=lambda r: r[9] > 0, off_pf=lambda r: r[9] == 0 and r[2] == 0,
               off_body=lambda r: r[9] == 0 and r[2] > 0)
    for k, f in sel.items():
        W = [[r[0] for r in R[i]['recs'] if f(r)] or ['_'] for i in ids]
        out[k] = cos(tfidf(text_profiles(W)['X']))
    return out

if __name__ == '__main__':
    rng = np.random.default_rng(456); res = {}
    pages, keys, words, vis, conf, strata = voynich_setup()
    have = set(p['id'] for p in get_corpus('V')); sel = [i for i, k in enumerate(keys) if k in have]
    keys = [keys[i] for i in sel]; conf = {k: v[np.ix_(sel, sel)] for k, v in conf.items()}; strata = [strata[i] for i in sel]
    Vv = {k: v[np.ix_(sel, sel)] for k, v in emb('voynich', keys).items()}; part = Partial(list(conf.values()), len(keys))
    for nm in sys.argv[1:] or ['V', 'VI', 'GEN0', 'GEN1', 'PL', 'GEw2000']:
        if nm == 'GEw2000':
            gk, gw, gvis, gconf = gerard_setup(); T = views(nm, ['ge' + k for k in gk])
            V_, P_, S_ = emb('gerard', gk), Partial(list(gconf.values()), len(gk)), None
        else:
            ids = keys
            if nm == 'VI':
                hv = set(p['id'] for p in get_corpus('VI')); ok = [j for j, k in enumerate(keys) if k in hv]
                ids = [keys[j] for j in ok]; V_ = {k: v[np.ix_(ok, ok)] for k, v in Vv.items()}
                P_ = Partial([c[np.ix_(ok, ok)] for c in conf.values()], len(ok)); S_ = [strata[j] for j in ok]
            else: V_, P_, S_ = Vv, part, strata
            T = views(nm, ids)
        res[nm] = {}
        for k, S in T.items():
            r = mantel_table(V_, {k: S}, P_, NPERM, rng, S_); res[nm][k] = dict(r=r['omni'], z=r['omni_z'], p=r['omni_p'])
        jsave('c6.json', res)
        print(nm, ' '.join('%s z %+.2f p %.3f' % (k, v['z'], v['p']) for k, v in res[nm].items()), flush=True)
