"""S-DARK-8.2: two-line and boustrophedon reading orders.

Two independent line-break sources:
  (A) Wells/ICIT raw text field in data/raw/inscriptions.csv: '/' separates two
      registers (segment A before '/', segment B after). The canonical corpus reverses
      the whole string, i.e. reads B (reversed) then A (reversed).
  (B) IM77 corpus lines: 'line' 1/2 on one side, each stored in reading order.
Hypotheses for a two-line text (l1, l2 each in stored reading order):
  12, 21 (line order) x flip of each line (boustrophedon variants) = 8 orders;
  column-wise: zip(l1, l2) and zip(l2, l1), with l2 optionally flipped = 4 orders.
Score: mean nll per token under an interpolated trigram model trained on one-line
texts of the same transcription (never on the two-line texts). Paired bootstrap (2,000
resamples over texts) for each hypothesis minus the stored order 12. Null: signs
shuffled inside each line, 200 replicates: the best-hypothesis gain over 12 then
calibrates the search. Held-out split: Mohenjo-daro + Harappa vs other sites.
"""
import json, csv, math, random, collections, sys
sys.path.insert(0, '/home/user/Indus-/tools')
from dark8_permute import Tri
OUT = '/home/user/Indus-/data/derived/dark/'
rng = random.Random(82)

def orders(l1, l2):
    H = {}
    for name, a, b in (('12', l1, l2), ('21', l2, l1)):
        for fa in (0, 1):
            for fb in (0, 1):
                A = tuple(reversed(a)) if fa else tuple(a); B = tuple(reversed(b)) if fb else tuple(b)
                H[f'{name}' + ('r' if fa else '-') + ('r' if fb else '-')] = A + B
    for name, a, b in (('col12', l1, l2), ('col21', l2, l1)):
        for fb in (0, 1):
            B = tuple(reversed(b)) if fb else tuple(b)
            out = []
            for i in range(max(len(a), len(B))):
                if i < len(a): out.append(a[i])
                if i < len(B): out.append(B[i])
            H[name + ('r' if fb else '')] = tuple(out)
    return H

def nll_text(m, s):
    s = ('<s>', '<s>') + s + ('</s>',)
    return sum(-m.lp(s[i-2], s[i-1], s[i]) for i in range(2, len(s))) / (len(s) - 2)

def evaluate(label, one_line, two_line, lines, reps=200):
    V = len({x for s in one_line for x in s}) + 1
    m = Tri(one_line, V)
    names = list(orders((1, 2), (3, 4)).keys())
    per = {n: [] for n in names}
    for l1, l2 in two_line:
        H = orders(l1, l2)
        for n in names: per[n].append(nll_text(m, H[n]))
    N = len(two_line)
    mean = {n: sum(v) / N for n, v in per.items()}
    base = mean['12--']
    # paired bootstrap
    ci = {}
    for n in names:
        d = [a - b for a, b in zip(per[n], per['12--'])]
        bs = []
        for _ in range(2000):
            idx = [rng.randrange(N) for _ in range(N)]
            bs.append(sum(d[i] for i in idx) / N)
        bs.sort(); ci[n] = (bs[50], bs[1950])
    # null: shuffle within line, best gain over 12--
    null = []
    for _ in range(reps):
        best = None
        tot = {n: 0.0 for n in names}
        for l1, l2 in two_line:
            a = list(l1); b = list(l2); rng.shuffle(a); rng.shuffle(b)
            H = orders(a, b)
            for n in names: tot[n] += nll_text(m, H[n])
        null.append(tot['12--'] / N - min(tot.values()) / N)
    null.sort()
    best = min(names, key=lambda n: mean[n])
    gain = base - mean[best]
    p = sum(1 for g in null if g >= gain) / len(null)
    lines.append(f'{label}: {N} two-line texts, model on {len(one_line)} one-line texts. nll/token by order: ' +
                 ', '.join(f'{n} {mean[n]:.3f}' for n in sorted(names, key=lambda n: mean[n])) +
                 f'. Best {best} gain {gain:+.3f} over stored 12 (95% CI of diff {ci[best][0]:+.3f}..{ci[best][1]:+.3f}); '
                 f'null best gain (within-line shuffle, {reps} reps) median {null[len(null)//2]:.3f}, 95th {null[int(0.95*len(null))]:.3f}; P(null >= obs) = {p:.3f}')
    return dict(n=N, mean=mean, ci=ci, best=best, gain=gain, null_med=null[len(null)//2], null95=null[int(0.95 * len(null))], p=p)

def wells():
    rows = list(csv.DictReader(open('/home/user/Indus-/data/raw/inscriptions.csv')))
    one, two = [], []
    for r in rows:
        t = r['text'].strip('+')
        if '[' in t or ']' in t or not t: continue
        segs = t.split('/')
        try:
            segs = [tuple(int(x) for x in s.split('-') if x) for s in segs]
        except ValueError:
            continue
        if any(0 in s for s in segs): continue
        site = r['site']
        if len(segs) == 1:
            if len(segs[0]) >= 2: one.append((site, tuple(reversed(segs[0]))))  # canonical: reverse the string
        elif len(segs) == 2 and all(len(s) >= 1 for s in segs):
            # canonical reading order per segment = reversed segment; segment B (after '/') read first in canonical
            two.append((site, tuple(reversed(segs[1])), tuple(reversed(segs[0]))))
    return one, two

def im77():
    rows = list(csv.DictReader(open('/home/user/Indus-/data/im77/im77_corpus_lines.csv')))
    by = collections.defaultdict(list)
    for r in rows: by[(r['text_no'], r['side'])].append(r)
    one, two = [], []
    for v in by.values():
        if any(x['line'] == '9' for x in v): continue
        seqs = [(x['line'], tuple(int(s) for s in x['signs_clean'].split()), x['site'], x['direction']) for x in v]
        if any(0 in s for _, s, _, _ in seqs): continue
        if len(seqs) == 1 and seqs[0][0] == '0' and len(seqs[0][1]) >= 2:
            one.append((seqs[0][2], seqs[0][1]))
        elif len(seqs) == 2 and sorted(x[0] for x in seqs) == ['1', '2']:
            seqs.sort(key=lambda x: x[0])
            two.append((seqs[0][2], seqs[0][1], seqs[1][1], seqs[0][3], seqs[1][3]))
    return one, two

if __name__ == '__main__':
    lines = []; res = {}
    one, two = wells()
    lines.append(f'Wells: {len(one)} one-line texts (no damage), {len(two)} two-register texts with "/" (segment B-reversed = stored l1, A-reversed = stored l2, i.e. canonical order = 12--)')
    res['wells_all'] = evaluate('WELLS all', [s for _, s in one], [(a, b) for _, a, b in two], lines)
    big = ('Mohenjo-daro', 'Harappa')
    res['wells_MDHP'] = evaluate('WELLS MD+HP', [s for _, s in one], [(a, b) for st, a, b in two if st in big], lines)
    res['wells_other'] = evaluate('WELLS other sites', [s for _, s in one], [(a, b) for st, a, b in two if st not in big], lines)
    two2 = [(a, b) for _, a, b in two if len(a) >= 2 and len(b) >= 2]
    res['wells_both2'] = evaluate('WELLS both lines >= 2 signs', [s for _, s in one], two2, lines)
    one, two = im77()
    lines.append(f'IM77: {len(one)} one-line sides, {len(two)} two-line sides. Direction pairs: ' + str(collections.Counter((d1, d2) for *_, d1, d2 in two).most_common()))
    res['im77_all'] = evaluate('IM77 all', [s for _, s in one], [(a, b) for _, a, b, _, _ in two], lines)
    res['im77_MDHP'] = evaluate('IM77 MD+HP', [s for _, s in one], [(a, b) for st, a, b, _, _ in two if st.startswith(('Mohenjo', 'Harappa'))], lines)
    res['im77_other'] = evaluate('IM77 other sites', [s for _, s in one], [(a, b) for st, a, b, _, _ in two if not st.startswith(('Mohenjo', 'Harappa'))], lines)
    res['im77_both2'] = evaluate('IM77 both lines >= 2 signs', [s for _, s in one], [(a, b) for _, a, b, _, _ in two if len(a) >= 2 and len(b) >= 2], lines)
    res['im77_RL_LR'] = evaluate('IM77 line1 R-L, line2 L-R (recorded boustrophedon)', [s for _, s in one],
                                 [(a, b) for _, a, b, d1, d2 in two if d1 == 'right-to-left' and d2 == 'left-to-right'], lines, reps=200)
    res['im77_RL_RL'] = evaluate('IM77 both lines R-L', [s for _, s in one],
                                 [(a, b) for _, a, b, d1, d2 in two if d1 == 'right-to-left' and d2 == 'right-to-left'], lines)
    json.dump(res, open(OUT + 'loop8_cycle2_lines.json', 'w'), indent=1)
    open(OUT + 'loop8_cycle2_log.txt', 'w').write('\n'.join(lines) + '\n')
    print('\n'.join(lines))
