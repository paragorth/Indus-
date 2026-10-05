#!/usr/bin/env python3
"""LA-49 cycle 2 runner: train, save weights, function vectors by transplant.
usage: la49_run2.py TAG CORPUS:NTF:NGRU ..."""
import sys, os, json, time, multiprocessing as mp
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import la49_common as L
import torch
from la49_run import cfg_for


def job(cfg):
    out = os.path.join(L.CK, cfg['tag'], '%s_%s_%03d.json' % (cfg['corpus'], cfg['arch'], cfg['k']))
    if os.path.exists(out):
        return out
    t = time.time()
    res, m = L.run_model2(cfg)
    res['sec'] = time.time() - t
    torch.save(m.state_dict(), out.replace('.json', '.pt'))
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
    for c in cfgs:
        c['epochs'] = min(c['epochs'], 30 if c['k'] >= 3 else 40)
        if c['corpus'].endswith('4') and c['k'] > 0:
            c['epochs'] = min(c['epochs'], 15)  # 4x data: same number of steps as a 60-epoch LA run
    cfgs.sort(key=lambda c: (c['k'], c['arch'], c['corpus']))
    with mp.get_context('fork').Pool(2) as p:
        for _ in p.imap_unordered(job, cfgs):
            pass
    print('ALL DONE', flush=True)
