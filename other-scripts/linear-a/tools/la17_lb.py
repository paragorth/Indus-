"""LA-17 Linear B control: same inference, KN must be the source / earliest."""
import sys, json, random
from la17_abc import *
mode = sys.argv[1] if len(sys.argv) > 1 else 'si'
tgt = 't' if mode == 'si' else 'tau'
docs = load_lb(); d = lb_data(docs); codes = d['codes']; K = len(codes)
T, S = load_sims('lb', mode)
abc = ABC(T, S, K, target=tgt)
acc, cov, rho = abc.heldout()
s = summaries(d['inc'])
ps, th = abc.rf(s); pr, rk = abc.reject(s)
print('LB', mode, len(S), 'heldout acc/cov/rho', acc, cov, rho)
print(' RF source', fmt_p(ps, codes, 6), '| rej', fmt_p(pr, codes, 6))
print(' RF order', order_str(th, codes), ' KN rank', int(np.argsort(np.argsort(th))[0]) + 1)
print(' rej mean ranks', dict(zip(codes, rk.mean(0).round(2))))
sh = []
for rep in range(30):
    dd = lb_data(doc_shuffle(docs, LB_SITES, random.Random(rep)))
    pp, t2 = abc.rf(summaries(dd['inc'])); sh.append((pp[0], pp.max(), codes[pp.argmax()]))
print(' shuffle: P(KN) %.2f, max %.2f, tops %s' % (np.mean([x[0] for x in sh]), np.mean([x[1] for x in sh]),
      dict(__import__('collections').Counter(x[2] for x in sh))))
json.dump(dict(ps=ps.tolist(), th=th.tolist(), pr=pr.tolist(), heldout=[acc, cov, rho], shuffle=sh),
          open(os.path.join(OUT, f'lb_{mode}.json'), 'w'), default=float)
