import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from v14_lib import *
from v14_cycle1 import summarise_slot

def fmt(r):
    lines = []
    lines.append('%s n=%d truth=%s cuts=%s cov=%.3f' % (r['label'], r['n'], r.get('truth'), r.get('model', {}).get('cuts'), r.get('model', {}).get('coverage', 0)))
    for k, rec in enumerate(r['slots']):
        s = summarise_slot(rec)
        idnull = {sg: Counter(v['ident']).most_common(2) for sg, v in rec['null'].items()}
        top = ' '.join('%s:%.3f' % (f or '_', n / r['n']) for f, n in rec['top'][:6])
        lines.append(' slot%d nfill=%d ident=%s | %s' % (k + 1, rec['nfill'], rec['ident'], top))
        for w in ('z_full', 'z_core'):
            bd, rb, pc = s[w]
            lines.append('   %s best=%s z=%.1f | null median z s0=%.1f s.1=%.1f s.3=%.1f | pct s0=%.2f s.1=%.2f s.3=%.2f' % (
                w, bd, rb, pc[('med', 0.0)], pc[('med', 0.1)], pc[('med', 0.3)], pc[0.0], pc[0.1], pc[0.3]))
        R = lambda d: DEV_ORDER.index(d) if d else len(DEV_ORDER)
        rr = R(rec['ident'])
        pr = {sg: (sum(1 for d in v['ident'] if R(d) <= rr) + 1) / (len(v['ident']) + 1) for sg, v in rec['null'].items()}
        lines.append('   parsimony rank real=%d (%s); P(null rank<=real) s0=%.2f s.1=%.2f s.3=%.2f; null ident: %s' % (
            rr, rec['ident'], pr[0.0], pr[0.1], pr[0.3], idnull))
    if 'indep' in r:
        lines.append(' indep: ' + '  '.join('%d-%d MI=%.3f null=%.4f z=%.0f p=%.3f' % (i + 1, j + 1, v['mi'], v['null_mean'], v['z'], v['p']) for (i, j), v in r['indep'].items()))
    return '\n'.join(lines)

if __name__ == '__main__':
    for n in sys.argv[1:]:
        r = load(n)
        if r: print(fmt(r))
