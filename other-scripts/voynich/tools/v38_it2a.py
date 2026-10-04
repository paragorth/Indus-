"""v38: replicate the neural-composite partial Mantel with the Takahashi IT2a transliteration."""
import sys, os, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v38_cycle1 as c1
from v38_lib import *
from v8_lib import voynich_pages

P = {p['id']: p for p in voynich_pages(min_tokens=30, name='IT2a')}
pages, keys, words, vis, conf, strata = c1.voynich_setup()
ok = [i for i, k in enumerate(keys) if k in P]
keys = [keys[i] for i in ok]; strata = [strata[i] for i in ok]
words = [[w for l in P[k]['lines'] for w in l] for k in keys]
conf = {c: M[np.ix_(ok, ok)] for c, M in conf.items()}
v2 = json.load(open(os.path.join(DER, 'v38_vis2_voynich.json')))
es = lambda V, f: cos_sim(np.array([V[k][f] for k in keys]) - np.array([V[k][f] for k in keys]).mean(0))
T = text_sims(text_profiles(words)); part = Partial(list(conf.values()), len(keys)); rng = np.random.default_rng(9)
a = mantel_table({f: es(vis, f) for f in ['r18', 'dino']}, T, part, 2000, rng, strata)
b = mantel_table({f: es(v2, f) for f in ['effb0', 'dinov2']}, T, part, 2000, rng, strata)
print('IT2a (%d pages) r18+dino z %.1f p %.4f; effb0+dinov2 z %.1f p %.4f' % (len(keys), a['omni_z'], a['omni_p'], b['omni_z'], b['omni_p']))
