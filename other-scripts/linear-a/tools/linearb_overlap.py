#!/usr/bin/env python3
"""Test 2e: Linear A words that also occur in Linear B with the same sign values.

LB data: data/damos_items.jsonl (DAMOS, CC BY-NC-SA 4.0). Only spellings and the
site prefix of each document heading (KN, PY, TH, MY, TI, KH ...) are used.
Control: LA word types rebuilt by shuffling signs across all LA types (same
length distribution, same sign frequencies); count LB matches. Repeated 1000x.
Second test: are the LB occurrences of the matched words concentrated at Knossos
(Crete) more than LB words in general?
"""
import json, os, re, random
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, '..', 'data')
random.seed(3)
SUB = str.maketrans('₂₃', '23')

def load_lb():
    occ = defaultdict(Counter)          # word -> Counter(site)
    ndocs = Counter()
    for l in open(os.path.join(D, 'damos_items.jsonl')):
        r = json.loads(l)
        if 'error' in r or not r.get('content'): continue
        site = (r.get('heading') or '?').split()[0]
        ndocs[site] += 1
        txt = r['content']
        txt = re.sub(r'\[|\]|\?|⸤|⸥|⌞|⌟|<|>|\{|\}|\.', '', txt)
        for m in re.finditer(r'(?<![A-Za-z0-9*-])([a-z][a-z0-9]*(?:-[a-z0-9*]+)+)(?![A-Za-z0-9*-])', txt):
            w = m.group(1).upper()
            occ[w][site] += 1
    return occ, ndocs

def main():
    occ, ndocs = load_lb()
    lb = set(occ)
    print('LB documents by site:', ndocs.most_common(8))
    print('LB word types (>=2 signs):', len(lb), ' tokens:', sum(sum(c.values()) for c in occ.values()))
    C = json.load(open(os.path.join(D, 'corpus.json')))
    la_tok = [w.translate(SUB) for i in C for w in i['words'] if '-' in w and '*' not in w]
    la = sorted(set(la_tok))
    S = [w.split('-') for w in la]
    def matches(words):
        return [w for w in words if w in lb]
    obs = matches(la)
    byl = Counter(len(w.split('-')) for w in obs)
    print(f'LA word types {len(la)}; matches in LB: {len(obs)}  by length: {dict(sorted(byl.items()))}')
    allsigns = [x for s in S for x in s]
    null = []; nullL = defaultdict(list)
    for _ in range(1000):
        random.shuffle(allsigns); k = 0; gen = []
        for s in S:
            gen.append('-'.join(allsigns[k:k + len(s)])); k += len(s)
        m = matches(set(gen))
        null.append(len(m)); c = Counter(len(w.split('-')) for w in m)
        for L in range(2, 7): nullL[L].append(c.get(L, 0))
    print(f'  control (signs shuffled across LA types): mean {sum(null)/len(null):.1f}, max {max(null)}, '
          f'P(>=obs) {sum(n >= len(obs) for n in null)/len(null):.4f}')
    for L in range(2, 7):
        o = byl.get(L, 0); n = nullL[L]
        print(f'    length {L}: obs {o}  control {sum(n)/len(n):.2f}  P(>=obs) {sum(x >= o for x in n)/len(n):.4f}')
    print('  matches with >=3 signs (LA tokens / LB sites):')
    long = [w for w in obs if len(w.split('-')) >= 3]
    for w in sorted(long, key=lambda w: -sum(occ[w].values())):
        print(f'    {w:16s} LA x{la_tok.count(w)}   LB {dict(occ[w].most_common(4))}')
    # Knossos concentration
    def kn_share(words):
        kn = sum(occ[w].get('KN', 0) for w in words); tot = sum(sum(occ[w].values()) for w in words)
        return kn, tot
    kn, tot = kn_share(obs)
    base_words = [w for w in lb if len(w.split('-')) >= 2]
    kb, tb = kn_share(base_words)
    print(f'  LB tokens of matched words at Knossos: {kn}/{tot} = {kn/max(1,tot):.2f};'
          f' all LB word tokens at Knossos: {kb}/{tb} = {kb/tb:.2f}')
    # by-type version (each matched word counted once: is its majority site KN?)
    kn_types = sum(1 for w in obs if occ[w].get('KN', 0) > sum(occ[w].values()) / 2)
    base_kn_types = sum(1 for w in base_words if occ[w].get('KN', 0) > sum(occ[w].values()) / 2)
    print(f'  matched types mostly at KN: {kn_types}/{len(obs)} = {kn_types/len(obs):.2f}; '
          f'all LB types mostly at KN: {base_kn_types}/{len(base_words)} = {base_kn_types/len(base_words):.2f}')
    # control: random LB word types of the same length distribution
    lens = Counter(len(w.split('-')) for w in obs)
    bylen = defaultdict(list)
    for w in base_words: bylen[len(w.split('-'))].append(w)
    nn = []
    for _ in range(5000):
        smp = [w for L, n in lens.items() for w in random.sample(bylen[L], n)]
        nn.append(sum(1 for w in smp if occ[w].get('KN', 0) > sum(occ[w].values()) / 2))
    print(f'  control (random LB types, same lengths): mean mostly-KN {sum(nn)/len(nn):.1f}; P(>= {kn_types}) '
          f'{sum(x >= kn_types for x in nn)/len(nn):.4f}')
    # same control at token level, and with the 2 most frequent matches removed
    def kn_tok(ws):
        k, t = kn_share(ws); return k / max(1, t)
    nn = []
    for _ in range(5000):
        smp = [w for L, n in lens.items() for w in random.sample(bylen[L], n)]
        nn.append(kn_tok(smp))
    o = kn_tok(obs)
    print(f'  token-level KN share obs {o:.2f}; control mean {sum(nn)/len(nn):.2f}; P(>=obs) {sum(x >= o for x in nn)/len(nn):.4f}')
    top2 = sorted(obs, key=lambda w: -sum(occ[w].values()))[:2]
    rest = [w for w in obs if w not in top2]
    print(f'  without {top2}: KN token share {kn_tok(rest):.2f}')
    json.dump({'matches': obs, 'lb_sites': {w: dict(occ[w]) for w in obs}},
              open(os.path.join(D, 'la_lb_matches.json'), 'w'), ensure_ascii=False, indent=1)

if __name__ == '__main__':
    main()
