"""pe85 cycle 3b: extended MAKE grid, planted CHOOSE / MAKE worlds through the same discriminator, tag on line art in sd units."""
import sys, os, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pe85_common as C
import pe83_common as P
from pe85_cycle1 import setup, corr
rng = np.random.default_rng(8532)
R, hd, Z, A, vol, strata = setup(True)
Z0 = Z[:, :-1]; top = C.col(R, 'cf_top'); bot = C.col(R, 'cf_bot')
s = P.strata(Z0); H = np.where(hd == 1)[0]; N = np.where(hd == 0)[0]
bystr = {k: N[s[N] == k] for k in np.unique(s)}
def world(mode, q, both=0.0):
    t2, b2 = top.copy(), bot.copy()
    for i in H:
        pool = bystr.get(s[i]); j = rng.choice(pool) if pool is not None and len(pool) else rng.choice(N)
        a, b = top[j], bot[j]
        if mode == 'choose':
            if rng.random() < q: a, b = max(a, b), min(a, b)
            elif rng.random() < .5: a, b = b, a
        else:
            a = min(1.0, a + q); b = min(1.0, b + both)
        t2[i], b2[i] = a, b
    return t2, b2
def rr(t2, b2):
    return P.partial(t2, hd, Z0)['r'], P.partial(b2, hd, Z0)['r']
out = {'observed': rr(top, bot)}
g = {}
for mode, q, both in [('make', .15, 0), ('make', .2, 0), ('make', .25, 0), ('make', .3, 0), ('make', .2, .03), ('make', .25, .04), ('choose', 1.0, 0)]:
    sims = np.array([rr(*world(mode, q, both)) for _ in range(60)])
    g[f'{mode}_q{q}_bottom{both}'] = dict(top=[round(float(np.quantile(sims[:, 0], x)), 3) for x in (.025, .5, .975)],
                                         bot=[round(float(np.quantile(sims[:, 1], x)), 3) for x in (.025, .5, .975)])
out['grid'] = g
# planted check: a CHOOSE world built from real headless pairs is called CHOOSE (bottom r < 0) and a MAKE world is not
pc = [rr(*world('choose', 1.0)) for _ in range(20)]; pm = [rr(*world('make', .25)) for _ in range(20)]
out['planted_choose_bottom_negative'] = int(sum(b < 0 for _, b in pc)); out['planted_make_bottom_negative_below_-0.08'] = int(sum(b < -0.08 for _, b in pm))
print(json.dumps(out, indent=0))
json.dump(out, open(os.path.join(C.CK, 'c3b.json'), 'w'), indent=1)
