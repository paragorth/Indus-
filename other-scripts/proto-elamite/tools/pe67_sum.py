"""pe67: summarise cycle 2 / cycle 3 checkpoints."""
import sys, json, os
import numpy as np
from collections import Counter, defaultdict
CK = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'pe67_ckpt')

def c2():
    R = json.load(open(os.path.join(CK, 'c2.json')))
    real = [r for r in R if r['kind'] == 'real']
    mp = [r for r in R if r['kind'] == 'mperm' and not r['ident']]
    ss = [r for r in R if r['kind'] == 'sshuf']
    m7 = [r for r in R if r['kind'] == 'mperm7']
    up = [r for r in R if r['kind'] == 'ur3prec']
    for r in real:
        print('real seed', r['i'], 'best %.2f top10 %.2f' % tuple(r['stat']))
    print(' links seed0:', real[0]['links'])
    lk = Counter(tuple(l[:2]) for r in real for l in r['links'])
    print(' links in >=3/5 seeds:', [k for k, v in lk.items() if v >= 3])
    b, t = np.mean([r['stat'][0] for r in real]), np.mean([r['stat'][1] for r in real])
    for name, N in [('mperm(120 rows over 5 training sites)', mp), ('site shuffle', ss)]:
        print(name, 'best p %.3f top10 p %.3f  null mean best %.2f top10 %.2f' % (
            np.mean([n['stat'][0] >= b for n in N]), np.mean([n['stat'][1] >= t for n in N]),
            np.mean([n['stat'][0] for n in N]), np.mean([n['stat'][1] for n in N])))
    T = np.mean([sum(v['T'] for v in r['loo'].values()) for r in real])
    print('LOO T real (mean of 5 seeds) %.2f' % T, {k: round(np.mean([r['loo'][k]['T'] for r in real]), 2) for k in real[0]['loo']})
    for k in real[0]['loo']:
        print('   ', k, real[0]['loo'][k]['links'][:10], 'used', real[0]['loo'][k]['used'])
    for name, N in [('mperm7', m7), ('site shuffle', ss)]:
        print(name, 'LOO p %.3f' % np.mean([n['T'] >= T for n in N]),
              {k: round(float(np.mean([n['perT'][k] >= np.mean([r['loo'][k]['T'] for r in real]) for n in N])), 3) for k in real[0]['loo']})
    tr = [r['prec'] for r in up if r['true']]; sh = [r['prec'] for r in up if not r['true']]
    print('Ur III discovery precision true %.3f vs material-shuffled %.3f (p %.3f)' % (np.mean(tr), np.mean(sh), np.mean([x >= np.mean(tr) for x in sh])))

def c3():
    R = json.load(open(os.path.join(CK, 'c3.json')))
    pe = [r for r in R if r['kind'] == 'pe'][0]
    print('PE training sites', pe['J'])
    for m, v in pe['phantom'].items():
        print('  %-12s col %s best %.2f pct %.2f top3 %.2f pct %.2f' % (m, v['col'], v['best'], v['pct_best'], v['top3'], v['pct_top3']))
    print(' PE dig:', {k: (round(v['rho'], 2), v['rank']) for k, v in pe['dig'].items()})
    ur = [r for r in R if r['kind'] == 'ur3']
    pc = [v['pct_top3'] for r in ur for v in r['phantom'].values()]
    print('Ur III phantom pct_top3 mean %.2f; share >= 0.95: %.2f' % (np.mean(pc), np.mean([x >= 0.95 for x in pc])))
    by = defaultdict(list)
    for r in ur:
        for m, v in r['phantom'].items():
            by[m].append(v['pct_top3'])
    print('   by material', {m: round(np.mean(v), 2) for m, v in by.items()})
    rk = [v['rank'] for r in ur for v in r['dig'].values() if v['rank'] is not None]
    print('Ur III dig: mean rank of true row among other sites %.2f (chance 0.5), n %d; rho mean %.2f' % (np.mean(rk), len(rk), np.mean([v['rho'] for r in ur for v in r['dig'].values()])))
    pt = [r for r in R if r['kind'] == 'perturb']
    c = Counter(tuple(l) for r in pt for l in r['links'])
    print('perturbed tables (n %d): link frequency' % len(pt), [(k, v) for k, v in c.most_common(15)])
    for half in (0, 1):
        cc = Counter(tuple(l) for r in pt if r['i'] % 2 == half for l in r['links'])
        print('  half', half, cc.most_common(8))
    s = [r for r in R if r['kind'] == 'single'][0]
    print('single tablets', json.dumps(s['out']))

{'c2': c2, 'c3': c3}[sys.argv[1]]()
