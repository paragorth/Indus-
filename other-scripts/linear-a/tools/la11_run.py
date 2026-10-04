#!/usr/bin/env python3
"""LA-11 runner: the full input x target matrix of best-map compression scores, checkpointed (JSONL), 2 workers.

usage: la11_run.py TAG SCHEME K M [--maxtrain N] [--lbscheme ...]
Inputs (all length-matched to the Linear A admin type list, 742 types):
  LA            Linear A administrative word types (sign labels as opaque IDs)
  LB0..LB2      Linear B admin word types, three length-matched subsamples (identities hidden: labels are IDs only)
  MK0..MK3      random first-order Markov strings (structured, related to no language)
  X:<code>      held-out word types of each real language, syllables re-coded to opaque integers
Each input is scored as is (variant 0) and after shuffling its symbol tokens across its types (variants 1..S).
Row: tag, input, variant, target, bits per predicted symbol under the best map, restarts spread.
"""
import os, sys, json, random, time, argparse
from multiprocessing import Pool
import la11_common as c

ap = argparse.ArgumentParser()
ap.add_argument('tag'); ap.add_argument('scheme'); ap.add_argument('K', type=int); ap.add_argument('M', type=int)
ap.add_argument('--maxtrain', type=int, default=None)
ap.add_argument('--nshuf_lang', type=int, default=2)
ap.add_argument('--nshuf_main', type=int, default=8)
ap.add_argument('--nshuf_lb', type=int, default=4)
ap.add_argument('--n_mk', type=int, default=3)
ap.add_argument('--R', type=int, default=6); ap.add_argument('--S', type=int, default=50000)
ap.add_argument('--targets', default=None)
ap.add_argument('--input_scheme', default=None, help='syllabification scheme of the held-out language inputs (default: same as targets)')
ap.add_argument('--extra_nonht', action='store_true')
ap.add_argument('--lb_sizes', default='', help='extra LB inputs, e.g. 1500,full')
ap.add_argument('--inputs', default=None, help='comma list of input names to restrict to')
ap.add_argument('--la_variant', default='admin', help='admin | admin_nonHT | admin_tokens')
args = ap.parse_args()

OUTF = os.path.join(c.D, f'run_{args.tag}.jsonl')

def build_inputs():
    rnd = random.Random(20261004)
    if args.la_variant == 'admin':
        la = c.la_types()
    elif args.la_variant == 'admin_nonHT':
        import la5_common as c5
        la = sorted(set(x[2] for x in c5.words_of(c5.la_docs()) if x[1] != 'Haghia Triada'))
    else:
        import la5_common as c5
        la = [x[2] for x in c5.words_of(c5.la_docs())]
    hist = c.len_hist(la)
    inp = {'LA': la}
    if args.extra_nonht:
        import la5_common as c5
        inp['LAn'] = sorted(set(x[2] for x in c5.words_of(c5.la_docs()) if x[1] != 'Haghia Triada'))
    lb = c.lb_types()
    for i in range(3): inp[f'LB{i}'] = c.length_matched(lb, hist, rnd)
    for i in range(args.n_mk): inp[f'MK{i}'] = c.markov_types(la, args.K, rnd)
    for sz in [x for x in args.lb_sizes.split(',') if x]:
        if sz == 'full': inp['LBfull'] = list(lb)
        else: inp[f'LB{sz}'] = rnd.sample(lb, int(sz))
    isch = args.input_scheme or args.scheme
    for code in c.lang_codes(min_train=0, scheme=isch):
        pool = c.lang_test(code, isch)
        if len(pool) < 400: continue
        inp['X:' + code] = c.length_matched(pool, hist, rnd)
    return inp

def tasks():
    inp = build_inputs()
    targets = args.targets.split(',') if args.targets else c.lang_codes(min_train=1000, scheme=args.scheme)
    names = args.inputs.split(',') if args.inputs else list(inp)
    done = set()
    if os.path.exists(OUTF):
        for l in open(OUTF):
            r = json.loads(l); done.add((r['input'], r['var'], r['target']))
    T = []
    for name in names:
        ns = args.nshuf_main if name in ('LA', 'LAn') else (args.nshuf_lb if name.startswith('LB') else args.nshuf_lang)
        for v in range(ns + 1):
            if v == 0: types = inp[name]
            else: types = c.shuffle_types(inp[name], random.Random(1000 * v + sum(map(ord, name))))
            for t in targets:
                if (name, v, t) in done: continue
                T.append((name, v, t, types))
    return T

def work(task):
    name, v, t, types = task
    try:
        C, top, n = c.count_matrix(types, args.K)
        L, syl = c.lang_model(t, args.scheme, args.M, max_train=args.maxtrain)
        b, m, rest = c.anneal(C, L, args.K, args.M, R=args.R, S=args.S, T0=12.0, T1=0.05, seed=1 + v * 7919 + sum(map(ord, name + t)))
        mp = {str(top[i]): syl[m[i]] for i in range(args.K)} if (v == 0 and name in ('LA', 'LB0')) else None
        return dict(tag=args.tag, input=name, var=v, target=t, bits=-b / n, spread=(max(rest) - min(rest)) / n, n=n, map=mp)
    except Exception as ex:
        return dict(tag=args.tag, input=name, var=v, target=t, error=str(ex))

if __name__ == '__main__':
    T = tasks()
    print(len(T), 'tasks', flush=True)
    t0 = time.time()
    with Pool(2) as p, open(OUTF, 'a') as fo:
        for k, r in enumerate(p.imap_unordered(work, T, chunksize=4)):
            fo.write(json.dumps(r) + '\n')
            if k % 200 == 0: fo.flush(); print(k, round(time.time() - t0), flush=True)
    print('done', round(time.time() - t0), flush=True)
