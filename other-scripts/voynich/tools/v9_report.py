"""v9: write loop rows from the checkpoints (cycle 1, 2, 3)."""
import sys, os, pickle, math
sys.path.insert(0, os.path.dirname(__file__))
from v9_lib import *
import v9_cycle1 as C1

OUT = C1.OUT
LN2 = math.log(2)
NAMES = {'planted': 'PLANTED volvelle', 'voynich': 'Voynich ZL3b', 'vs_latin': 'verbose Latin', 'vs_italian': 'verbose Italian'}


def bits(ll, n):
    return -ll / n / LN2


def cycle1():
    allp = pickle.load(open(os.path.join(OUT, 'c1_all.pkl'), 'rb'))
    base = allp['base']
    rows, summary = [], {}
    i = 0
    for name in C1.CORPORA:
        b = base[name]; n = b['ntok_te']
        cells, tot = [], dict(M0=0, M1=0, M2=0, soft=0, hard=0, nores=0, oracle=0)
        for k in range(3):
            fits = [pickle.load(open(os.path.join(OUT, 'c1m_%s_%d_n%d.pkl' % (name, k, s)), 'rb')) for s in C1.SIZES]
            f = max(fits, key=lambda r: r['hard_tr'])
            m0, m1, m2 = b['M0_%d' % k][0], b['M1_%d' % k][0], b['M2_%d' % k][0]
            g = lambda ll: (bits(m0, n) - bits(ll, n))
            frac = (f['hard_te'] - m0) / (m1 - m0) if m1 != m0 else float('nan')
            H, top = kernel_summary(np.array(f['q'])[0])
            cells.append('ring %d (n=%d): M0 %.3f b/tok; gain M1 %+.3f, M2 %+.3f, soft volvelle %+.3f, HARD volvelle %+.3f (%.0f%% of M1 gain), no-reset %+.3f; kernel H %.2f bits, top steps %s%s' % (
                k, f['n'], bits(m0, n), g(m1), g(m2), g(f['soft_te']), g(f['hard_te']), 100 * frac, g(f['noreset_te']), H, top,
                ('; oracle %+.3f' % g(b['oracle_%d' % k])) if 'oracle_%d' % k in b else ''))
            tot['M0'] += m0; tot['M1'] += m1; tot['M2'] += m2; tot['soft'] += f['soft_te']; tot['hard'] += f['hard_te']
            tot['nores'] += f['noreset_te']
            if 'oracle_%d' % k in b: tot['oracle'] += b['oracle_%d' % k]
            summary[(name, k)] = dict(n=f['n'], frac=frac, gain_hard=g(f['hard_te']), gain_m1=g(m1),
                                      gain_nores=g(f['noreset_te']), lab=f['lab'], q=f['q'])
        word = 'word level (sum of rings, b/tok): M0 %.3f, M1 %.3f, hard volvelle %.3f, soft %.3f, no-reset %.3f; word trigram W2 %.3f (%d params), pruned W2 %.3f (%d params), tuple unigram W0 %.3f' % (
            bits(tot['M0'], n), bits(tot['M1'], n), bits(tot['hard'], n), bits(tot['soft'], n), bits(tot['nores'], n),
            bits(b['W2'][0], n), b['W2'][2], bits(b['W2p'][0], n), b['W2p'][2], bits(b['W0'][0], n))
        if tot['oracle']:
            word += ', TRUE volvelle %.3f' % bits(tot['oracle'], n)
        summary[name] = dict(tot={k: bits(v, n) for k, v in tot.items() if v}, W2=bits(b['W2'][0], n),
                             W2p=bits(b['W2p'][0], n), W0=bits(b['W0'][0], n))
        i += 1
        rows.append(['V-%d.1' % i, '%s: 3-ring volvelle (slot grammar prefix/core/suffix, 12 fillers + OTHER per ring), ring size 20/30/40/48 chosen on train, Baum-Welch on circulant HMM then hardened; held-out alternate pages; vs within-ring Markov-0/1/2 and word trigram' % NAMES[name],
                     ' ; '.join(cells) + ' || ' + word, ''])
    return rows, summary


if __name__ == '__main__' and len(sys.argv) == 1:
    rows, s = cycle1()
    for r in rows:
        print(r[0], r[2]); print()
    pickle.dump(s, open(os.path.join(OUT, 'c1_summary.pkl'), 'wb'))


def cycle2():
    base = pickle.load(open(os.path.join(OUT, 'c1_all.pkl'), 'rb'))['base']
    c1 = pickle.load(open(os.path.join(OUT, 'c1_summary.pkl'), 'rb'))
    L = lambda kind, name, k: pickle.load(open(os.path.join(OUT, 'c2_%s_%s_%d.pkl' % (kind, name, k)), 'rb'))
    out = []
    for name in C1.CORPORA:
        n = base[name]['ntok_te']
        d = L('disc', name, 0)
        w0, w1, w2 = d['W0'][0], d['W1'][0], d['W2'][0]
        g = lambda ll: bits(w0, n) - bits(ll, n)
        H, top = kernel_summary(np.array(d['q'])[0])
        out.append(('disc', name, 'W0 %.3f b/tok; gain W1 %+.3f, W2 %+.3f, soft disc %+.3f, HARD disc %+.3f; kernel H %.2f (flat %.2f), top %s' % (
            bits(w0, n), g(w1), g(w2), g(d['soft_te']), g(d['hard_te']), H, math.log2(d['n']), top)))
        cells = []
        for k in range(3):
            r = L('R2', name, k)
            m0 = base[name]['M0_%d' % k][0]
            g = lambda ll: bits(m0, n) - bits(ll, n)
            H = np.mean([kernel_summary(q)[0] for q in r['q']])
            cells.append('ring %d: R2 hard %+.3f (R1 hard %+.3f), R2 soft %+.3f; Markov-1 x prev-suffix %+.3f, iid x prev-suffix %+.3f; R2 kernel H %.2f' % (
                k, g(r['hard_te']), c1[(name, k)]['gain_hard'], g(r['soft_te']), g(r['MX1'][0]), g(r['MX0'][0]), H))
        out.append(('R2', name, ' ; '.join(cells)))
    for name in ('voynich', 'planted'):
        cells = []
        for k in range(3):
            r = L('shuf', name, k)
            m0 = r['M0'][0]
            g = lambda ll: bits(m0, n) - bits(ll, n)
            cells.append('ring %d: M1 %+.3f, soft %+.3f, hard %+.3f' % (k, g(r['M1'][0]), g(r['soft_te']), g(r['hard_te'])))
        out.append(('shuf', name, ' ; '.join(cells)))
    return out


if __name__ == '__main__' and len(sys.argv) > 1 and sys.argv[1] == '2':
    for x in cycle2():
        print(x); print()
