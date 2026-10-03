"""S-DARK-8.4: per-text direction flip. The whole-corpus permutation search (8.1) cannot see
direction (a Markov-2 chain is Markov-2 backwards), but an asymmetric model trained on texts
in one reading order can judge a single text: nll(stored order) - nll(reversed). Model: trigram
on Mohenjo-daro + Harappa texts recorded R/L (one-line, no damage), scored on
  (i) held-out-site R/L texts (baseline rate of 'reverse fits better'),
  (ii) texts recorded L/R, T/B, BUS, NR, '-' (unrecorded) at all sites (MD+HP ones scored with a
       5-fold model that excludes them), by object type, site and length.
Also the length-stratified identity-vs-reverse asymmetry seen in 8.1 at L = 6-7: paired bootstrap
of per-text differences, held-out sites vs a random split inside MD+HP."""
import json, math, random, collections, sys
sys.path.insert(0, '/home/user/Indus-/tools')
from dark8_permute import Tri
OUT = '/home/user/Indus-/data/derived/dark/'
rng = random.Random(84)

def nll_text(m, s):
    s = ('<s>', '<s>') + s + ('</s>',)
    return sum(-m.lp(s[i-2], s[i-1], s[i]) for i in range(2, len(s))) / (len(s) - 2)

def summarize(diffs):
    N = len(diffs)
    if N == 0: return 'n=0'
    mean = sum(diffs) / N; rev = sum(1 for d in diffs if d > 0)
    bs = sorted(sum(diffs[rng.randrange(N)] for _ in range(N)) / N for _ in range(1000))
    return f'n={N}, reverse-better {rev} ({100*rev/N:.0f}%), mean nll(stored)-nll(rev) {mean:+.3f} (95% CI {bs[25]:+.3f}..{bs[975]:+.3f})'

def run(variant, lines):
    d = json.load(open('/home/user/Indus-/data/derived/merged-corpus-canonical.json'))
    tx = []
    for t in d:
        s = t[variant]
        if not s or 0 in s or len(s) < 3: continue
        tx.append(dict(site=t['site'], dir=t['dir.'].strip().upper(), typ=t['type'].split(':')[0], seq=tuple(s), cisi=t['cisi']))
    big = ('Mohenjo-daro', 'Harappa')
    V = len({x for t in tx for x in t['seq']}) + 1
    rl_big = [t for t in tx if t['dir'] == 'R/L' and t['site'] in big]
    # 5-fold models over rl_big so that MD+HP texts can be scored out of fold
    folds = [[] for _ in range(5)]
    for i, t in enumerate(rl_big): folds[i % 5].append(t)
    models = [Tri([t['seq'] for j, f in enumerate(folds) if j != k for t in f], V) for k in range(5)]
    full = Tri([t['seq'] for t in rl_big], V)
    fold_of = {t['cisi']: i % 5 for i, t in enumerate(rl_big)}
    def diff(t):
        m = models[fold_of[t['cisi']]] if t['cisi'] in fold_of else (models[rng.randrange(5)] if t['site'] in big else full)
        return nll_text(m, t['seq']) - nll_text(m, tuple(reversed(t['seq'])))
    for t in tx: t['d'] = diff(t)
    lines.append(f'--- {variant}: {len(tx)} texts of >= 3 signs; model = {len(rl_big)} R/L MD+HP texts (5-fold for scoring MD+HP) ---')
    lines.append('R/L held-out sites (baseline): ' + summarize([t['d'] for t in tx if t['dir'] == 'R/L' and t['site'] not in big]))
    lines.append('R/L MD+HP out-of-fold: ' + summarize([t['d'] for t in tx if t['dir'] == 'R/L' and t['site'] in big]))
    for dr in ('L/R', 'T/B', 'BUS', 'NR', '-', 'SYM'):
        lines.append(f'dir={dr}: ' + summarize([t['d'] for t in tx if t['dir'] == dr]))
    for dr in ('L/R', 'NR', '-'):
        for typ, c in collections.Counter(t['typ'] for t in tx if t['dir'] == dr).most_common(4):
            lines.append(f'  dir={dr} type={typ}: ' + summarize([t['d'] for t in tx if t['dir'] == dr and t['typ'] == typ]))
        for st, c in collections.Counter(t['site'] for t in tx if t['dir'] == dr).most_common(5):
            lines.append(f'  dir={dr} site={st}: ' + summarize([t['d'] for t in tx if t['dir'] == dr and t['site'] == st]))
    # R/L by type and site: is any class systematically reversed?
    for typ, c in collections.Counter(t['typ'] for t in tx if t['dir'] == 'R/L').most_common(6):
        lines.append(f'  R/L type={typ}: ' + summarize([t['d'] for t in tx if t['dir'] == 'R/L' and t['typ'] == typ]))
    for st, c in collections.Counter(t['site'] for t in tx if t['dir'] == 'R/L').most_common(8):
        lines.append(f'  R/L site={st}: ' + summarize([t['d'] for t in tx if t['dir'] == 'R/L' and t['site'] == st]))
    # length-stratified asymmetry, R/L only
    for L in range(3, 9):
        lines.append(f'  R/L length {L}: held-out sites ' + summarize([t['d'] for t in tx if t['dir'] == 'R/L' and t['site'] not in big and len(t['seq']) == L]) +
                     ' | MD+HP out-of-fold ' + summarize([t['d'] for t in tx if t['dir'] == 'R/L' and t['site'] in big and len(t['seq']) == L]))
    # list the most strongly reversed texts (any dir) for a look-up
    worst = sorted(tx, key=lambda t: -t['d'])[:15]
    lines.append('Texts the model would rather read backwards (top 15): ' + '; '.join(f"{t['cisi']} {t['dir']} {t['typ']} {list(t['seq'])} d={t['d']:+.2f}" for t in worst))
    return {k: None for k in ()}

if __name__ == '__main__':
    lines = []
    for v in ('seq_raw', 'seq_strong', 'seq_all'):
        run(v, lines)
    open(OUT + 'loop8_cycle4_log.txt', 'w').write('\n'.join(lines) + '\n')
    print('\n'.join(lines))
