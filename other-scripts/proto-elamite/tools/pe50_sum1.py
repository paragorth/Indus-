"""Summarise pe50 cycle 1 (data/pe50_ckpt/cycle1.json)."""
import json, os, collections, math
import numpy as np
CK = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'pe50_ckpt')
R = json.load(open(os.path.join(CK, 'cycle1.json')))
EST = ('chao2', 'ichao2', 'jack1', 'jack2', 'chapman', 'll3')

print('== U: Ur III thinned to 607 tablets; truth = full-archive observed count')
cal = {}
g = collections.defaultdict(list)
for arch, d, n, Sfull, r in R['ur3']:
    g[(arch, d)].append((Sfull, r))
for (arch, d), L in g.items():
    Sfull = L[0][0]
    S = np.median([r['S'] for _, r in L])
    line = [f'{arch} {d} Sfull {Sfull} S_seen(med) {S:.0f}']
    for e in EST:
        v = [r[e] for _, r in L if e in r]
        est = np.array([x[0] for x in v])
        cov = np.mean([x[1] <= Sfull <= x[2] for x in v])
        ratio = np.median(Sfull / est)
        cal[(arch, d, e)] = ratio
        line.append(f'{e} med {np.median(est):.0f} cover {cov:.2f} truth/est {ratio:.2f}')
    q = [(r['thin_q'], r['thin_qtrue']) for _, r in L if 'thin_q' in r]
    line.append('thinfit q_hat/q_true ' + ', '.join(f'{a:.4f}/{b:.3f}' for a, b in q))
    print(' | '.join(line))

print('== P: planted archives (truth = persons on >= 1 written tablet)')
cov = collections.defaultdict(list)
rat = collections.defaultdict(list)
th = []
for par, Nseen, r in R['plant']:
    for e in EST:
        if e in r:
            cov[e].append(r[e][1] <= Nseen <= r[e][2])
            rat[e].append(Nseen / r[e][0])
    if 'thin_q' in r:
        th.append((r['thin_q'], r['thin_qtrue']))
for e in EST:
    print(f' {e}: coverage {np.mean(cov[e]):.2f}  truth/est median {np.median(rat[e]):.2f} '
          f'[{np.percentile(rat[e], 10):.2f}-{np.percentile(rat[e], 90):.2f}]')
th = np.array(th)
print(' thinfit: corr(log qhat, log qtrue) %.2f; median qhat/qtrue %.2f' % (
    np.corrcoef(np.log(th[:, 0]), np.log(th[:, 1]))[0, 1], np.median(th[:, 0] / th[:, 1])))
# planted with PE-like singleton fraction
print(' planted S_seen/Nseen median %.2f' % np.median([r['S'] / N for _, N, r in R['plant']]))

print('== S: volume-grouped three-list model, real vs label-shuffled (PE MID2, Susa)')
sh = R['shuffle']
rv = [x['real'][0] for x in sh if x['real']]
sv = [x['shuf'][0] for x in sh if x['shuf']]
print(f' real N median {np.median(rv):.0f} [{np.percentile(rv, 5):.0f}-{np.percentile(rv, 95):.0f}]; '
      f'shuffled {np.median(sv):.0f} [{np.percentile(sv, 5):.0f}-{np.percentile(sv, 95):.0f}]')
print(' models real', collections.Counter(x['real'][1] for x in sh if x['real']).most_common(),
      'shuf', collections.Counter(x['shuf'][1] for x in sh if x['shuf']).most_common())
print(' same-volume share of recaptured entities: real %.2f (n %d), shuffled %.2f' % (
    sh[0]['samevol_real'][0], sh[0]['samevol_real'][1], np.mean([x['samevol_shuf'][0] for x in sh])))

print('== E: PE ensemble')
pe = R['pe']
print(' runs', len(pe))
tab = collections.defaultdict(list)
for x in pe:
    tab[(x['def'], x['est'])].append(x)
for k in sorted(tab):
    L = tab[k]
    N = np.array([x['N'] for x in L])
    print(f' {k[0]:6s} {k[1]:8s} n_runs {len(L):3d} S(all) {max(x["S"] for x in L):5d} N med {np.median(N):9.0f} '
          f'[{np.percentile(N, 5):.0f}-{np.percentile(N, 95):.0f}]')
# calibrated: MID2 all-tablet with Ur III ENT2 calibration
print('== calibrated PE MID2/all estimates (x Ur III truth/est at the same size)')
for e in EST:
    L = [x for x in pe if x['def'] == 'MID2' and x['sub'] == 'all' and x['est'] == e]
    if not L:
        continue
    N = np.median([x['N'] for x in L])
    cs = [cal.get((a, 'ENT2', e)) for a in ('DREHEM', 'UMMA')]
    co = [cal.get((a, 'OFF', e)) for a in ('DREHEM', 'UMMA')]
    print(f' {e}: raw {N:.0f}; x ENT2 cal {cs[0]:.2f}/{cs[1]:.2f} -> {N * cs[0]:.0f}/{N * cs[1]:.0f}; '
          f'x OFF cal {co[0]:.2f}/{co[1]:.2f} -> {N * co[0]:.0f}/{N * co[1]:.0f}')
