"""pe77 analysis: unforgeability per feature from forger runs."""
import sys, os, json, collections
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pe77_common as pc


def load(tag):
    fn = os.path.join(pc.CKPT, tag + '.jsonl')
    return [json.loads(l) for l in open(fn)]


def unforg(R, q=0.25, min_n=10):
    """U(f) = mean signed z over the strongest q of forgers (lowest full-detector AUC);
    W(f) = mean signed z over the weakest q.  Returns dict f -> (U, W, n_strong)."""
    A = np.array([r['A'] for r in R])
    order = np.argsort(A)
    k = max(1, int(len(R) * q))
    strong = [R[i] for i in order[:k]]
    weak = [R[i] for i in order[-k:]]
    def mz(S):
        acc = collections.defaultdict(list)
        for r in S:
            for f, (d, z) in r['f'].items():
                if z is not None:
                    acc[f].append(z)
        return acc
    s, w = mz(strong), mz(weak)
    out = {}
    for f, v in s.items():
        if len(v) >= min_n:
            out[f] = (float(np.mean(v)), float(np.mean(w[f])) if len(w.get(f, [])) >= min_n else None, len(v),
                      float(np.std(v) / np.sqrt(len(v))))
    return out, float(np.median(A[order[:k]])), float(np.median(A[order[-k:]]))


def sign_of(f):
    return f.split('_', 1)[1] if '_' in f else None


def rank_auc(scores, pos, neg):
    from scipy.stats import mannwhitneyu
    a = [scores[f] for f in pos if f in scores]
    b = [scores[f] for f in neg if f in scores]
    if len(a) < 2 or len(b) < 2:
        return None, len(a), len(b)
    u = mannwhitneyu(a, b, alternative='greater')
    return float(u.statistic / (len(a) * len(b))), len(a), len(b), float(u.pvalue)


if __name__ == '__main__':
    for tag in sys.argv[1:]:
        R = load(tag)
        U, As, Aw = unforg(R)
        print('==', tag, 'n forgers', len(R), 'median AUC strong', round(As, 3), 'weak', round(Aw, 3))
        top = sorted(U.items(), key=lambda x: -x[1][0])[:25]
        for f, (u, w, n, se) in top:
            print('  %-22s U %+6.2f  (se %.2f) weak %s n %d' % (f, u, se, 'NA' if w is None else '%+.2f' % w, n))
        fam = collections.defaultdict(list)
        for f, v in U.items():
            fam[pc.family(f)].append(v[0])
        print('  families (mean U):', {k: round(float(np.mean(v)), 2) for k, v in sorted(fam.items())})
