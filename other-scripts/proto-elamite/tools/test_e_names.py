#!/usr/bin/env python3
"""Test e: do multi-sign entry strings repeat like names (a fixed lexicon) or
like free combinations?

Statistic: unique share = (string types seen once) / (string tokens).
Model: bigram (with start/end) trained on the same strings; generate the same
number of strings with length >= 2 (rejection); 200 runs.
Ratio = observed unique share / model unique share. A fixed lexicon of names
(re-used across documents) gives ratio < 1; free composition gives about 1.
Planted control: a lexicon of L strings drawn from the same bigram model, used
with Zipf frequencies, sized so the corpus has the same token count.
Run on full strings and on strings with the final sign removed (the final slot
is a category/commodity marker, test a/b)."""
import collections, json, os, random
from common import load, entries, DATA

random.seed(7)
T = load()
E = entries(T, require_clean=True)


def unique_share(strs):
    c = collections.Counter(strs)
    return sum(1 for v in c.values() if v == 1) / len(strs)


def train(strs):
    m = collections.defaultdict(collections.Counter)
    for s in strs:
        seq = ['<s>'] + list(s) + ['</s>']
        for a, b in zip(seq, seq[1:]):
            m[a][b] += 1
    return {a: (list(c.keys()), list(c.values())) for a, c in m.items()}


def gen(m, minlen=2, maxlen=15):
    while True:
        out, cur = [], '<s>'
        while True:
            ks, ws = m[cur]
            cur = random.choices(ks, ws)[0]
            if cur == '</s>':
                break
            out.append(cur)
            if len(out) > maxlen:
                break
        if minlen <= len(out) <= maxlen:
            return tuple(out)


def run(strs, label, runs=200):
    m = train(strs)
    obs = unique_share(strs)
    mod = [unique_share([gen(m) for _ in strs]) for _ in range(runs)]
    mm = sum(mod) / runs
    sd = (sum((x - mm) ** 2 for x in mod) / runs) ** .5
    # planted lexicon control
    L = max(10, int(len(strs) * 0.35))
    lex = [gen(m) for _ in range(L)]
    w = [1 / (i + 1) for i in range(L)]
    planted = random.choices(lex, w, k=len(strs))
    pm = train(planted)
    p_obs = unique_share(planted)
    p_mod = sum(unique_share([gen(pm) for _ in planted]) for _ in range(50)) / 50
    print('%-28s n=%5d types=%5d unique share obs %.3f model %.3f (sd %.3f) ratio %.2f | planted-lexicon ratio %.2f' % (
        label, len(strs), len(set(strs)), obs, mm, sd, obs / mm, p_obs / p_mod))
    return {'n': len(strs), 'types': len(set(strs)), 'obs': obs, 'model': mm, 'sd': sd,
            'ratio': obs / mm, 'planted_ratio': p_obs / p_mod}


full = [tuple(e['signs']) for e in E if len(e['signs']) >= 2]
core = [tuple(e['signs'][:-1]) for e in E if len(e['signs']) >= 3]
res = {'full': run(full, 'full strings (len>=2)'),
       'core': run(core, 'strings minus final (len>=3)')}
# cross-tablet recurrence of repeated strings
tabs = collections.defaultdict(set)
for e in E:
    if len(e['signs']) >= 2:
        tabs[tuple(e['signs'])].add(e['tablet'])
rep = [s for s, v in collections.Counter(full).items() if v >= 2]
multi_tab = [s for s in rep if len(tabs[s]) >= 2]
print('repeated full strings', len(rep), 'of which on >=2 tablets', len(multi_tab))
print('most repeated', collections.Counter(full).most_common(12))
res['repeated'] = len(rep); res['repeated_multi_tablet'] = len(multi_tab)
# Out-of-Susa check: do strings from other sites occur at Susa?
sus = {tuple(e['signs']) for e in E if 'Susa' in e['prov'] and len(e['signs']) >= 2}
oth = [tuple(e['signs']) for e in E if 'Susa' not in e['prov'] and len(e['signs']) >= 2]
hit = sum(1 for s in oth if s in sus)
m = train([s for s in full])
sim = [sum(1 for _ in range(len(oth)) if gen(m) in sus) for _ in range(50)]
print('non-Susa multi-sign strings %d, found at Susa %d; bigram-model expectation %.1f' % (
    len(oth), hit, sum(sim) / 50))
res['non_susa'] = {'n': len(oth), 'hits': hit, 'model': sum(sim) / 50}
json.dump(res, open(os.path.join(DATA, 'res_e_names.json'), 'w'), indent=1)
