"""v10 cycle 3: is the line-initial chain a PLAINTEXT ACROSTIC under a simple encoding?
Chain per paragraph = first glyph of every line (paragraph-first line included: an acrostic would use it); glyphs
with < 10 occurrences pooled as '?'. For each language model (Latin, Italian, German letter trigrams, add-0.1),
simulated annealing searches an onto map letter -> chain glyph (a lossy, many-letters-per-glyph encoding, the
form an acrostic in a ~10-glyph margin alphabet must take) maximising the likelihood of the chains under the
language's letter-trigram counts pushed through the map (a proper glyph-trigram probability, add-0.5; collapsing
onto frequent letters cannot win). Statistic = best log2-likelihood per chain glyph.
Nulls (each annealed with the same search):
   N1 within-paragraph permutation of lines 2..n (keeps every paragraph's glyph counts);
   N2 first-order Markov surrogates (global transition table of the corpus' own chain, same paragraph lengths and
      first glyphs) -- keeps the KNOWN lag-1 rule, so beating N2 is what a hidden plaintext would need.
Controls: planted Latin acrostic (Caesar letters, lossy 22->10 letter->glyph map, real paragraph lengths, first-line
   the whole margin incl. line 1) scored with the Latin model; planted Italian acrostic (Manzoni) with Italian; both
   must beat N1 and N2. A line-shuffled Voynich (lines 2..n moved at random between paragraphs) must not beat N1."""
import sys, os, json, random, math
from collections import Counter, defaultdict
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v10_lib import *

REPS = int(os.environ.get('REPS', 30)); ITERS = int(os.environ.get('ITERS', 4000))
_LC = {}
def LC(lang):
    if lang not in _LC:
        t = ''.join(letters_of(k) for k in LANGS[lang]); _LC[lang] = LetterCounts(t)
    return _LC[lang]

def chains(name, plant):
    paras = chain_paras(name)
    if plant == 'shuffled':
        rng = random.Random(3); allb = [c for p in paras for c in p['chain'][1:]]; rng.shuffle(allb); k = 0
        for p in paras:
            nb = len(p['chain']) - 1; p['chain'] = [p['chain'][0]] + allb[k:k + nb]; k += nb
    elif plant in ('acro_la', 'acro_it'):
        key = 'Latin-Caesar' if plant == 'acro_la' else 'Italian-Manzoni'
        txt = letters_of(key); txt = txt[len(txt) // 3:]
        glyphs = ['d', 'y', 'o', 'q', 's', 't', 'S', 'C', 'l', 'p']
        order = sorted(set(txt), key=lambda c: -txt.count(c)); rng = random.Random(9); rng.shuffle(order)
        lmap = {c: glyphs[i % len(glyphs)] for i, c in enumerate(order)}; pos = 0
        for p in paras:
            nb = len(p['chain']); p['chain'] = [lmap[c] for c in txt[pos:pos + nb]]; pos += nb   # whole margin incl. line 1
    seqs = [p['chain'] for p in paras]
    cnt = Counter(g for s in seqs for g in s)
    return [[g if cnt[g] >= 10 else '?' for g in s] for s in seqs]

def markov(seqs, rng):
    T = defaultdict(Counter)
    for s in seqs:
        for a, b in zip(s, s[1:]): T[a][b] += 1
    tab = {a: (list(c), list(c.values())) for a, c in T.items()}
    out = []
    for s in seqs:
        q = [s[0]]
        for _ in range(len(s) - 1):
            a = q[-1]
            if a not in tab: a = s[1] if len(s) > 1 else s[0]
            ks, ws = tab.get(a, tab[max(tab, key=lambda x: sum(tab[x][1]))]); q.append(rng.choices(ks, ws)[0])
        out.append(q)
    return out

def perm(seqs, rng): return [[s[0]] + rng.sample(s[1:], len(s) - 1) for s in seqs]

def job(args):
    name, plant, lang, kind, seed = args
    seqs = chains(name, plant); rng = random.Random(1000 + seed)
    if kind == 'N1': seqs = perm(seqs, rng)
    elif kind == 'N2': seqs = markov(seqs, rng)
    sc, mp = anneal_partition(seqs, LC(lang), iters=ITERS, seed=seed, restarts=2)
    return (name, plant, lang, kind, seed, sc, mp)

def decode(seqs, mp, n=6):
    inv = defaultdict(str)
    for l, g in sorted(mp.items()): inv[g] += l
    return [' '.join('[' + inv[g] + ']' for g in s) for s in seqs[:n]]

if __name__ == '__main__':
    runs = [('ZL3b', None, L) for L in ('Latin', 'Italian', 'German')] + [('IT2a', None, 'Latin')] + \
           [('ZL3b', 'acro_la', 'Latin'), ('ZL3b', 'acro_it', 'Italian'), ('ZL3b', 'shuffled', 'Latin')]
    ck = os.path.join(CK, 'c3_runs.jsonl'); done = {}
    if os.path.exists(ck):
        for l in open(ck):
            r = json.loads(l); done[tuple(r[:5])] = r
    todo = []
    for name, plant, lang in runs:
        todo.append((name, plant, lang, 'obs', 0))
        for kind in ('N1', 'N2'):
            for s in range(REPS): todo.append((name, plant, lang, kind, s))
    todo = [t for t in todo if t not in done]
    with Pool(2) as pool, open(ck, 'a') as f:
        for r in pool.imap_unordered(job, todo):
            f.write(json.dumps(list(r)) + '\n'); f.flush(); done[tuple(r[:5])] = list(r)
    S = {}
    for name, plant, lang in runs:
        o = done[(name, plant, lang, 'obs', 0)]
        row = {'obs': round(o[5], 4)}
        for kind in ('N1', 'N2'):
            xs = [done[(name, plant, lang, kind, s)][5] for s in range(REPS)]
            row[kind] = zstat(o[5], xs); row[kind]['max'] = round(max(xs), 4)
        seqs = chains(name, plant); row['map'] = o[6]; row['sample'] = decode(seqs, o[6])
        S['%s %s %s' % (name, plant, lang)] = row
        print(name, plant, lang, json.dumps(row), flush=True)
    json.dump(S, open(os.path.join(CK, 'c3_summary.json'), 'w'), indent=1)
