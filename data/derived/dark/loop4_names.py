"""Loop 4 cycle 3: a hidden lexicon INSIDE the name slot? Take the NAME residue of every text (S310 parser, learned on the
fit sites) and run the segmentation search on the residues alone: BPE merges fit on MD+Harappa names, tested on other-site
names; bits per sign, uniq ratio, within-name repetition. Positive control: Ur III seal-owner names at the SYLLABLE level
(seal_line1.txt, syllables split on '-'), where whole names recur (a real onomasticon): same search; expect the uniq
ratio to sit well below 1 and BPE merges to lower bits. Negative control: bigram-generated names (same lengths).
Arrows-fired null: the search re-run on 10 bigram-generated name corpora; best delta per statistic recorded."""
import sys, random, collections, json, re
sys.path.insert(0, '/home/user/Indus-/data/derived/dark'); sys.path.insert(0, '/home/user/Indus-')
import loop4_seg as L
CYCLE = sys.argv[1] if len(sys.argv) > 1 else '3'
OUT = open(f'/home/user/Indus-/data/derived/dark/loop4_cycle{CYCLE}.txt', 'a')
def P(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); OUT.write(s + '\n'); OUT.flush()
rng = random.Random(3)
SP = '/tmp/claude-0/-home-user-Indus-/874df4c7-80d6-5f08-b42c-eea96a214079/scratchpad/seal_line1.txt'

def name_runs(texts, QUAL):
    out = []
    for t in texts:
        lab = L.parse(t, QUAL); cur = []
        for a, l in zip(t, lab):
            if l == 'NAME': cur.append(a)
            else:
                if cur: out.append(cur); cur = []
        if cur: out.append(cur)
    return [s for s in out if len(s) >= 1]

def summarize(res, label):
    for k, v in sorted(res.items(), key=lambda kv: kv[1]['bits']): P(f'    {k:28s} ' + L.fmt(v))
    L.report(res, label)
    bb = L.best({k: v for k, v in res.items() if k.startswith('bpe')}, 'bits')
    if bb: P('    first merges of best BPE: ' + '; '.join('+'.join(str(a) for u in pr for a in u) for pr in res[bb[1]]['merges'][:12]))
    return res

if __name__ == '__main__':
    import datetime
    P(f'# loop4 cycle {CYCLE}: segmentation search inside the NAME residue ({datetime.datetime.now().isoformat(timespec="minutes")})')
    summary = {}
    for level in ('seq_raw', 'seq_strong', 'seq_all'):
        fit, test = L.indus(level); QUAL = L.learn_qual(fit)
        F = name_runs(fit, QUAL); T = name_runs(test, QUAL)
        P(f'\n## Indus {level}: name runs fit {len(F)} (mean len {sum(map(len, F))/len(F):.2f}), test {len(T)} (mean len {sum(map(len, T))/len(T):.2f})')
        res = L.search(F, T, rng, f'names {level}', fams=('bpe', 'sub'), gram=False)
        summarize(res, f'names {level}')
        summary[level] = {k: {kk: vv for kk, vv in v.items() if kk != 'merges'} for k, v in res.items()}
        if level == 'seq_raw':
            # arrows-fired null on bigram-generated names (same lengths)
            m = L.Bigram(L.identity(F)); deltas = []
            for r in range(10):
                F2 = [[u[0] for u in m.sample(len(s), rng)] for s in F]; T2 = [[u[0] for u in m.sample(len(s), rng)] for s in T]
                rn = L.search(F2, T2, rng, 'null', fams=('bpe', 'sub'), gram=False, verbose=False); b = rn['sign']
                d = {key: (L.best(rn, key, True)[0] - b[key]) for key in ('bits', 'uniq', 'rep')}
                d['joint'] = sum(1 for k, v in rn.items() if k != 'sign' and v['bits'] < b['bits'] and (v['uniq'] < b['uniq'] or v['rep'] < b['rep']))
                deltas.append(d); P(f'    null names {r}: ' + ' '.join(f'{k}={v:+.3f}' if isinstance(v, float) else f'{k}={v}' for k, v in d.items()))
            summary['null'] = deltas
    # positive control: Ur III seal-owner names at the syllable level
    names = []
    for l in open(SP, errors='ignore'):
        l = re.sub(r'[#\[\]!?<>]', '', l.strip().lower())
        if not l or ' ' in l or 'x' in l.split('-'): continue
        t = [x for x in l.split('-') if x]
        if 2 <= len(t) <= 6: names.append(t)
    rng.shuffle(names); cut = int(0.7 * len(names))
    P(f'\n## Ur III seal-owner names (syllables): fit {cut}, test {len(names) - cut}; distinct {len({tuple(n) for n in names})}')
    res = L.search(names[:cut], names[cut:], rng, 'Ur3 names', fams=('bpe',), gram=False)
    summarize(res, 'Ur3 names'); summary['ur3names'] = {k: {kk: vv for kk, vv in v.items() if kk != 'merges'} for k, v in res.items()}
    # the same with duplicates removed (one copy per distinct name): the lexicon effect without repeated seals
    dn = list({tuple(n): n for n in names}.values()); rng.shuffle(dn); cut = int(0.7 * len(dn))
    P(f'\n## Ur III names, distinct only: fit {cut}, test {len(dn) - cut}')
    res = L.search(dn[:cut], dn[cut:], rng, 'Ur3 distinct names', fams=('bpe',), gram=False)
    summarize(res, 'Ur3 distinct names'); summary['ur3distinct'] = {k: {kk: vv for kk, vv in v.items() if kk != 'merges'} for k, v in res.items()}
    json.dump(summary, open(f'/home/user/Indus-/data/derived/dark/loop4_cycle{CYCLE}_names.json', 'w'))
