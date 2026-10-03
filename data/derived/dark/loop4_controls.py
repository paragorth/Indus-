"""Loop 4 controls for the segmentation search (same statistics and search as loop4_seg.py).
 A. Ur III seal legends (CDLI, distinct legends, sign = grapheme, true word boundaries known): does the BPE search find
    the word level (merges word-internal), and does the true word segmentation beat the sign level on bits/uniq/rep?
 B. Planted logo-syllabic corpus (Tamil words syllabified, 60 commonest words as single word-signs, Zipf weights,
    Indus text lengths and the Indus site split): same questions, truth known.
Usage: python3 loop4_controls.py <cycle>"""
import sys, random, collections, json, math
sys.path.insert(0, '/home/user/Indus-'); sys.path.insert(0, '/home/user/Indus-/data/derived/dark')
import loop4_seg as L
import indus_core as IC, synth
CYCLE = sys.argv[1] if len(sys.argv) > 1 else '1'
OUT = open(f'/home/user/Indus-/data/derived/dark/loop4_cycle{CYCLE}_controls.txt', 'a')
def P(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); OUT.write(s + '\n'); OUT.flush()
rng = random.Random(int(CYCLE) if CYCLE.isdigit() else 1)

def word_internal(merges, legends):
    """share of merge pairs (as adjacent atom sequences) whose occurrences fall inside one word >= 80% of the time"""
    inside = collections.Counter(); total = collections.Counter()
    flat = []
    for leg in legends:
        atoms = []; wid = []
        for wi, w in enumerate(leg):
            for s in w: atoms.append(s); wid.append(wi)
        flat.append((atoms, wid))
    good = 0
    for pr in merges:
        seq = tuple(a for u in pr for a in u); n = len(seq); ins = tot = 0
        for atoms, wid in flat:
            for i in range(len(atoms) - n + 1):
                if tuple(atoms[i:i + n]) == seq:
                    tot += 1; ins += (len(set(wid[i:i + n])) == 1)
        if tot and ins / tot >= 0.8: good += 1
    return good / len(merges) if merges else float('nan')

def run_control(name, fit_legs, test_legs):
    fit = [[s for w in leg for s in w] for leg in fit_legs]; test = [[s for w in leg for s in w] for leg in test_legs]
    nsigns = sum(len(s) for s in test)
    P(f'\n## {name}: fit {len(fit)} texts, test {len(test)} texts, {nsigns} test sign tokens, '
      f'{len({s for t in fit for s in t})} sign types, mean word length {sum(len(w) for leg in fit_legs for w in leg) / sum(len(leg) for leg in fit_legs):.2f}')
    res = L.search(fit, test, rng, name, fams=('bpe',), gram=False)
    # true word segmentation (units = words)
    fw = [[tuple(w) for w in leg] for leg in fit_legs]; tw = [[tuple(w) for w in leg] for leg in test_legs]
    res['TRUE words'] = L.evaluate(fw, tw, nsigns, rng, None, None, False)
    # shuffled-control segmentation: words of the same lengths but boundaries shifted by one atom (wrong units)
    def shift(legs):
        out = []
        for leg in legs:
            atoms = [s for w in leg for s in w]; lens = [len(w) for w in leg]; lens = lens[1:] + lens[:1]
            u = []; i = 0
            for n in lens: u.append(tuple(atoms[i:i + n])); i += n
            out.append([x for x in u if x])
        return out
    res['WRONG words (shifted)'] = L.evaluate(shift(fit_legs), shift(test_legs), nsigns, rng, None, None, False)
    for k, v in sorted(res.items(), key=lambda kv: kv[1]['bits']):
        P(f'    {k:28s} ' + L.fmt(v))
    L.report(res, name)
    bb = L.best({k: v for k, v in res.items() if k.startswith('bpe')}, 'bits')
    for key in [bb[1]] + [k for k in res if k.startswith('bpe k=60 T=1')]:
        if key in res and 'merges' in res[key]:
            P(f'    {key}: word-internal share of merges = {word_internal(res[key]["merges"], fit_legs):.2f}; first merges: ' +
              '; '.join('+'.join(str(a) for u in pr for a in u) for pr in res[key]['merges'][:10]))
    return res

if __name__ == '__main__':
    import datetime
    P(f'# loop4 controls cycle {CYCLE} ({datetime.datetime.now().isoformat(timespec="minutes")})')
    # A. Ur III
    legs = json.load(open('/home/user/Indus-/data/derived/dark/loop4_ur3_legends.json'))
    distinct = list({tuple(tuple(w) for w in l): l for l in legs}.values()); rng.shuffle(distinct)
    cut = int(len(distinct) * 0.7)
    ur3 = run_control('Ur III seal legends', distinct[:cut], distinct[cut:])
    # B. planted logo-syllabic corpus with Indus lengths and site split
    lex = IC.load_lexicon('tamil'); r0 = random.Random(7)
    allv = [s for s in (synth.syllabify(w) for w in sorted(lex)) if s and len(s) <= 3]
    freq = collections.Counter(u for s in allv for u in s); top = set(u for u, _ in freq.most_common(150))
    vocab = [s for s in allv if all(u in top for u in s)]; r0.shuffle(vocab); vocab = vocab[:3000]
    wts = [1 / (r + 1) for r in range(len(vocab))]
    logos = {tuple(vocab[i]): f'L{i:02d}' for i in range(60)}
    fit_raw, test_raw = L.indus('seq_raw')
    def plant(src):
        out = []
        for s in src:
            t = []; n = 0
            while n < len(s):
                w = tuple(r0.choices(vocab, wts)[0]); unit = [logos[w]] if w in logos else list(w); t.append(unit); n += len(unit)
            out.append(t)
        return out
    planted = run_control('planted logo-syllabic (Tamil, 60 word-signs, Indus lengths)', plant(fit_raw), plant(test_raw))
    json.dump({'ur3': {k: {kk: vv for kk, vv in v.items() if kk != 'merges'} for k, v in ur3.items()},
               'planted': {k: {kk: vv for kk, vv in v.items() if kk != 'merges'} for k, v in planted.items()}},
              open(f'/home/user/Indus-/data/derived/dark/loop4_cycle{CYCLE}_controls.json', 'w'))
