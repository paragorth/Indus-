"""Loop 4 cycle 4: minimum description length. A hidden lexicon under a wrong segmentation should pay for itself:
DL(lexicon spelled in signs) + DL(corpus coded as units) < DL(corpus coded as signs). Two-part code, unigram and bigram
corpus codes with (V-1)/2 log2 N parameter cost; lexicon = each multi-sign unit spelled with the sign unigram code plus
a length code. BPE greedy k = 1..80 merges on the whole corpus; the DL curve's minimum is the test.
Corpora: Indus seq_raw / seq_strong / seq_all (whole corpus, and MD+Harappa only); Ur III legends (true words known);
planted logo-syllabic corpus (true words known); bigram-generated Indus null (10 replicates): the arrows-fired null for
'best k'. Report: best k, DL gain (%), and in the controls the DL of the TRUE word segmentation."""
import sys, random, collections, json, math
sys.path.insert(0, '/home/user/Indus-/data/derived/dark'); sys.path.insert(0, '/home/user/Indus-')
import loop4_seg as L
CYCLE = sys.argv[1] if len(sys.argv) > 1 else '4'
OUT = open(f'/home/user/Indus-/data/derived/dark/loop4_cycle{CYCLE}.txt', 'a')
def P(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); OUT.write(s + '\n'); OUT.flush()
rng = random.Random(4)

def dl(seg, sign_uni, order):
    """two-part code length in bits for a segmented corpus (list of lists of unit tuples)."""
    N = sum(len(s) + 1 for s in seg)
    if order == 1:
        c = collections.Counter(u for s in seg for u in list(s) + ['E']); V = len(c)
        corp = -sum(n * math.log2(n / N) for n in c.values()); par = (V - 1) / 2 * math.log2(N)
    else:
        big = collections.defaultdict(collections.Counter)
        for s in seg:
            p = 'S'
            for u in list(s) + ['E']: big[p][u] += 1; p = u
        corp = 0.0; npar = 0
        for p, c in big.items():
            t = sum(c.values()); corp -= sum(n * math.log2(n / t) for n in c.values()); npar += len(c) - 1
        par = npar / 2 * math.log2(N)
    units = {u for s in seg for u in s}
    aN = sum(sign_uni.values())
    lex = 0.0
    for u in units:
        if len(u) > 1: lex += math.log2(len(u)) * 2 + sum(-math.log2(sign_uni[a] / aN) for a in u)
    return corp + par + lex, corp, par, lex

def curve(texts, kmax=80, true_seg=None, label=''):
    sign_uni = collections.Counter(a for s in texts for a in s)
    base = {o: dl(L.identity(texts), sign_uni, o) for o in (1, 2)}
    merges = L.bpe_learn(texts, kmax, 1, rng)
    seg = L.identity(texts); best = {1: (base[1][0], 0), 2: (base[2][0], 0)}; rows = []
    for k, pr in enumerate(merges, 1):
        seg = [L.bpe_apply1(s, pr) for s in seg]
        for o in (1, 2):
            d = dl(seg, sign_uni, o)
            if d[0] < best[o][0]: best[o] = (d[0], k)
            if k in (5, 10, 20, 40, 60, 80): rows.append((k, o, d))
    out = {}
    for o in (1, 2):
        gain = (base[o][0] - best[o][0]) / base[o][0] * 100
        out[o] = dict(base=base[o][0], best=best[o][0], k=best[o][1], gain_pct=gain)
        P(f'  {label} order-{o}: sign-level DL {base[o][0]:.0f} bits; best BPE k={best[o][1]} DL {best[o][0]:.0f} (gain {gain:+.2f}%)')
    if true_seg is not None:
        for o in (1, 2):
            d = dl(true_seg, sign_uni, o); g = (base[o][0] - d[0]) / base[o][0] * 100
            out[f'true{o}'] = g; P(f'  {label} order-{o}: TRUE word segmentation DL {d[0]:.0f} (gain {g:+.2f}%; corpus {d[1]:.0f} par {d[2]:.0f} lexicon {d[3]:.0f})')
    P('  ' + label + ' DL by k (order1/order2 gain %): ' + ' '.join(f'k{k}:{(base[o][0]-d[0])/base[o][0]*100:+.1f}' for k, o, d in rows))
    return out

if __name__ == '__main__':
    import datetime
    P(f'# loop4 cycle {CYCLE}: MDL test for a hidden lexicon ({datetime.datetime.now().isoformat(timespec="minutes")})')
    res = {}
    for level in ('seq_raw', 'seq_strong', 'seq_all'):
        fit, test = L.indus(level)
        P(f'\n## Indus {level}: whole corpus {len(fit) + len(test)} texts')
        res[level] = curve(fit + test, label=f'Indus {level} all')
        if level == 'seq_raw': res[level + '_MDH'] = curve(fit, label='Indus seq_raw MD+Harappa')
    # bigram null for Indus seq_raw
    fit, test = L.indus('seq_raw'); m = L.Bigram(L.identity(fit + test)); gains = []
    P('\n## bigram-generated Indus null (10x, same lengths)')
    for r in range(10):
        syn = [[u[0] for u in m.sample(len(s), rng)] for s in fit + test]
        g = curve(syn, label=f'null{r}'); gains.append((g[1]['gain_pct'], g[2]['gain_pct'], g[1]['k'], g[2]['k']))
    res['null'] = gains
    P('  null gains order1: ' + ' '.join(f'{a:+.2f}' for a, _, _, _ in gains) + '  max %.2f' % max(a for a, _, _, _ in gains))
    P('  null gains order2: ' + ' '.join(f'{b:+.2f}' for _, b, _, _ in gains) + '  max %.2f' % max(b for _, b, _, _ in gains))
    # Ur III control
    legs = json.load(open('/home/user/Indus-/data/derived/dark/loop4_ur3_legends.json'))
    distinct = list({tuple(tuple(w) for w in l): l for l in legs}.values())
    P(f'\n## Ur III distinct legends {len(distinct)}')
    res['ur3'] = curve([[s for w in l for s in w] for l in distinct], true_seg=[[tuple(w) for w in l] for l in distinct], label='Ur III')
    # planted control
    import indus_core as IC, synth
    lex = IC.load_lexicon('tamil'); r0 = random.Random(7)
    allv = [s for s in (synth.syllabify(w) for w in sorted(lex)) if s and len(s) <= 3]
    freq = collections.Counter(u for s in allv for u in s); top = set(u for u, _ in freq.most_common(150))
    vocab = [s for s in allv if all(u in top for u in s)]; r0.shuffle(vocab); vocab = vocab[:3000]
    wts = [1 / (r + 1) for r in range(len(vocab))]; logos = {tuple(vocab[i]): f'L{i:02d}' for i in range(60)}
    pl = []
    for s in fit + test:
        t = []; n = 0
        while n < len(s):
            w = tuple(r0.choices(vocab, wts)[0]); unit = [logos[w]] if w in logos else list(w); t.append(tuple(unit)); n += len(unit)
        pl.append(t)
    P(f'\n## planted logo-syllabic corpus {len(pl)} texts')
    res['planted'] = curve([[a for u in t for a in u] for t in pl], true_seg=pl, label='planted')
    json.dump(res, open(f'/home/user/Indus-/data/derived/dark/loop4_cycle{CYCLE}_mdl.json', 'w'))
