"""Audit: are Wells and IM77 'replications' independent samples or the same objects twice?
Bridge merged-corpus texts into M numbers; for IM77 texts (len>=4, no damage, all signs in bridge range)
count exact sequence matches in the merged corpus. Control: same with IM77 texts shuffled internally."""
import json, csv, collections, random, itertools
d = json.load(open('/home/user/Indus-/data/derived/merged-corpus-canonical.json'))
br = {int(k): v for k, v in json.load(open('/home/user/Indus-/data/derived/bridge_extended.json')).items()}
Mcov = set(m for v in br.values() for m in v)
# all bridged merged sequences (expanded alternatives, capped)
wtexts = collections.defaultdict(set)
for x in d:
    s = x['seq_raw']
    if all(c in br for c in s):
        for combo in itertools.islice(itertools.product(*[br[c] for c in s]), 16):
            wtexts[tuple(combo)].add(x['site'])
lines = collections.defaultdict(list)
for r in csv.DictReader(open('/home/user/Indus-/data/im77/im77_corpus_lines.csv')):
    if r['line'] == '9': continue
    lines[(r['text_no'], r['side'])].append((int(r['line']), r['signs_clean'].split()))
im = []
for k, v in lines.items():
    seq = [int(t) for _, l in sorted(v) for t in l]
    if len(seq) >= 4 and 0 not in seq and all(m in Mcov for m in seq): im.append(tuple(seq))
def rate(texts): return sum(t in wtexts for t in texts) / len(texts)
o = rate(im)
random.seed(0); nulls = []
for _ in range(50):
    sh = []
    for t in im:
        l = list(t); random.shuffle(l); sh.append(tuple(l))
    nulls.append(rate(sh))
print('IM77 fully-bridgeable texts len>=4: %d; exact match in merged corpus: %.1f%%; shuffled null mean %.1f%% max %.1f%%'
      % (len(im), 100*o, 100*sum(nulls)/len(nulls), 100*max(nulls)))
