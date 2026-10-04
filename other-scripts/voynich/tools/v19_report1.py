"""v19 cycle 1 report: per corpus x kind x key, z distribution, BH survivors (Gaussian p from z), pooled tau,
planted detection and order recovery."""
import json, os, glob, math, sys
from v19_lib import RES, order_agreement, bh

D = os.path.join(RES, 'c1')
truth = json.load(open(os.path.join(RES, 'c1_truth.json')))


def gp(z):
    return 0.5 * math.erfc(z / math.sqrt(2))


def load():
    out = {}
    for f in sorted(glob.glob(os.path.join(D, '*.json'))):
        c, k, key = os.path.basename(f)[:-5].split('_')
        d = json.load(open(f))
        if d:
            out[(c, k, key)] = d
    return out


if __name__ == '__main__':
    R = load()
    allp = []
    for (c, k, key), d in R.items():
        for r in d['rows']:
            allp.append(((c, k, key, r['id']), gp(r['z']) if r['T'] > 0 else 1.0, r))
    print('cell                       nstr  z>3  z>4  maxz  (id)              pooled tau/z   pooled order')
    for (c, k, key), d in sorted(R.items()):
        rows = [r for r in d['rows'] if r['T'] > 0]
        if not rows:
            continue
        zs = [r['z'] for r in rows]
        m = max(rows, key=lambda r: r['z'])
        po = d['pooled_ws'] if 'pooled_ws' in d else d['pooled']
        print('%-26s %4d %4d %4d %5.1f  %-17s %6.3f %6.1f   %s' % ('%s/%s/%s' % (c, k, key), len(rows), sum(z > 3 for z in zs),
              sum(z > 4 for z in zs), m['z'], m['id'][:17], po['tau'], po['z'], po['order_s'][:60]))
    # BH within each corpus (all kinds x keys pooled into one family)
    print('\nBH (q=0.05, Gaussian p from z) survivors per corpus:')
    for corp in ['PLANT', 'LatX', 'LatXVI', 'LatXVII', 'ZL', 'IT']:
        items = [a for a in allp if a[0][0] == corp]
        if not items:
            continue
        S = bh([a[1] for a in items])
        surv = sorted([items[i] for i in S], key=lambda a: -a[2]['z'])
        print(' %s: %d tests, %d survive' % (corp, len(items), len(surv)))
        for a in surv[:12]:
            r = a[2]
            print('   %-40s n=%d tau %.3f null %.3f z %.1f  order %s' % ('/'.join(a[0][1:]), r['n'], r['tau'], r['nmean'], r['z'], r['order_s'][:50]))
    print('\nPlanted pages: z and order recovery (Kendall vs true order) by kind/key')
    for pid, (ptype, perm) in truth.items():
        rank = {g: i for i, g in enumerate(perm)}
        line = []
        for k in ['pagewords', 'win5', 'lineinit', 'linefinal']:
            for key in ['F', 'L']:
                d = R.get(('PLANT', k, key))
                if not d:
                    continue
                rs = [r for r in d['rows'] if r['id'].startswith(pid)]
                if not rs:
                    continue
                r = max(rs, key=lambda r: r['z'])
                ag = order_agreement(r['order_s'].split(), rank)
                line.append('%s/%s z%.1f rec%.2f' % (k[:5], key, r['z'], ag))
        print(' %-20s %s' % (pid, ' | '.join(line)))
