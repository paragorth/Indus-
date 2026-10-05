"""pe51 cycle 3 summary: pair interactions. usage: python3 pe51_pairs_sum.py CORPUS [CORPUS ...]
I = D_ab - D_a - D_b per task (all targets pooled, own slots of a and b excluded). Null: a with as many
random other tokens as b has (4 draws): I_q = D_aq - D_a - D_q. Excess = I - mean(I_q); z = excess /
sd(I_q) pooled over models. REDUNDANT pair: excess > 0 with z > 3 (losing both hurts more than the
two losses added: each sign covers for the other)."""
import glob, json, os, sys, collections, math
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pe51_lib as L

TASKS = ['SYS', 'MAG', 'TOT', 'HEAD', 'ENT']


def D(acc, c, k):
    a = acc.get(c, {}).get(k)
    return a[0] / a[1] if a and a[1] else None


def analyse(name):
    F = sorted(glob.glob(os.path.join(L.CK, 'pair_%s_*.json' % name)))
    P = collections.defaultdict(lambda: collections.defaultdict(lambda: [[], []]))
    for f in F:
        d = json.load(open(f))
        for pr, x in d['res'].items():
            acc = x['acc']
            for k in TASKS:
                da, db, dab = D(acc, 'a', k), D(acc, 'b', k), D(acc, 'ab', k)
                if None in (da, db, dab):
                    continue
                I = dab - da - db
                nul = []
                for q in range(4):
                    dq, daq = D(acc, 'q%d' % q, k), D(acc, 'aq%d' % q, k)
                    if dq is not None and daq is not None:
                        nul.append(daq - da - dq)
                if len(nul) < 2:
                    continue
                P[pr][k][0].append(I - np.mean(nul))
                P[pr][k][1].extend([v - np.mean(nul) for v in nul])
    out = []
    for pr, kk in P.items():
        for k, (ex, nl) in kk.items():
            sd = np.std(nl) + 1e-4
            e = float(np.mean(ex))
            z = e / (sd / math.sqrt(len(ex)))
            out.append((pr, k, e, z, len(ex)))
    return out, len(F)


if __name__ == '__main__':
    for name in sys.argv[1:]:
        out, nf = analyse(name)
        print('%s: %d models, %d pair-task cells' % (name, nf, len(out)))
        red = [o for o in out if o[2] >= 0.01 and o[3] > 3]
        sub = [o for o in out if o[2] <= -0.01 and o[3] < -3]
        print('  redundant (z>3, excess>=0.01): %d; overlapping (z<-3): %d' % (len(red), len(sub)))
        by = collections.Counter(o[1] for o in red)
        print('  redundant by task', dict(by))
        for o in sorted(red, key=lambda o: -o[3])[:25]:
            print('   + %-22s %-4s excess %+.3f z %.1f (models %d)' % o)
        for o in sorted(sub, key=lambda o: o[3])[:10]:
            print('   - %-22s %-4s excess %+.3f z %.1f (models %d)' % o)
        if name == 'plant':
            dh = [o for o in out if o[0].startswith('d') and '|h' in o[0]]
            same = [o for o in dh if o[0][1] == o[0].split('|h')[1][0]]
            print('  plant doc-marker x own companion cells:', len(same), 'redundant', sum(1 for o in same if o[2] >= 0.01 and o[3] > 3),
                  '; mean excess by task', {k: round(float(np.mean([o[2] for o in same if o[1] == k])), 3) for k in TASKS if any(o[1] == k for o in same)})
            oth = [o for o in out if not (o in same)]
            print('  plant other pairs:', len(oth), 'redundant', sum(1 for o in oth if o[2] >= 0.01 and o[3] > 3))
