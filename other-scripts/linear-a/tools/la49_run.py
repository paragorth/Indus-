#!/usr/bin/env python3
"""LA-49 runner: train + dissect a population of tiny models. usage: la49_run.py TAG CORPUS:NTF:NGRU ..."""
import sys, os, json, random, time, multiprocessing as mp
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import la49_common as L


def cfg_for(corpus, arch, k, tag):
    r = random.Random(L.seed('%s-%s-%s-%d' % (tag, corpus, arch, k)))
    d = r.choice([16, 24, 32, 48, 64])
    h = r.choice([h for h in (1, 2, 4) if d % h == 0])
    return dict(corpus=corpus, arch=arch, d=d, h=h, nl=r.choice([1, 2, 3]) if arch == 'tf' else r.choice([1, 2]),
                drop=r.choice([0.0, 0.1, 0.2]), lr=r.choice([1e-3, 2e-3, 4e-3]), epochs=r.choice([30, 40, 60]),
                seed=r.randrange(10 ** 8), k=k, tag=tag)


def job(cfg):
    out = os.path.join(L.CK, cfg['tag'], '%s_%s_%03d.json' % (cfg['corpus'], cfg['arch'], cfg['k']))
    if os.path.exists(out):
        return out
    t = time.time()
    res = L.run_model(cfg)
    res['sec'] = time.time() - t
    json.dump(res, open(out + '.tmp', 'w'))
    os.replace(out + '.tmp', out)
    print('done', out, round(res['sec']), flush=True)
    return out


if __name__ == '__main__':
    tag = sys.argv[1]
    os.makedirs(os.path.join(L.CK, tag), exist_ok=True)
    cfgs = []
    for spec in sys.argv[2:]:
        c, ntf, ngru = spec.split(':')
        cfgs += [cfg_for(c, 'tf', k, tag) for k in range(int(ntf))]
        cfgs += [cfg_for(c, 'gru', k, tag) for k in range(int(ngru))]
    # interleave so partial results cover every corpus
    cfgs.sort(key=lambda c: (c['k'], c['arch'], c['corpus']))
    with mp.get_context('fork').Pool(2) as p:
        for _ in p.imap_unordered(job, cfgs):
            pass
    print('ALL DONE', flush=True)
