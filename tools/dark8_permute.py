"""S-DARK-8.1: reading-order brute force by position-permutation templates.

For each text length L (3..7) every permutation template pi of the L positions
(sampled to 2,000 for L >= 6, always including identity and full reverse) is applied
to every text of length L. Two scores:
  (a) held-out perplexity: order-2 (trigram, interpolated) model fit on Mohenjo-daro +
      Harappa texts of length L, perplexity on texts of length L from all other sites;
  (b) slot tightness: H(first position) + H(last position) over all length-L texts.
Lower is better for both. Controls: (1) Ur III seal legends from CDLI (known order:
line 1 -> line 2 -> ...), flattened to sign tokens, same search; held-out = random
30% of legends. (2) Indus corpus with signs shuffled inside each text, 20 replicates:
the distribution of the best template's gain over identity gives the max-corrected
null. Run on seq_raw, seq_strong and seq_all.

Note: a forward Markov-2 chain is also a backward Markov-2 chain, so the full
reverse of a text ties with identity up to smoothing on both scores. The search
therefore tests adjacency structure, not direction itself.
"""
import json, math, random, itertools, collections, sys
SP = '/tmp/claude-0/-home-user-Indus-/874df4c7-80d6-5f08-b42c-eea96a214079/scratchpad/'
OUT = '/home/user/Indus-/data/derived/dark/'
rng = random.Random(8)

def load_indus(variant):
    d = json.load(open('/home/user/Indus-/data/derived/merged-corpus-canonical.json'))
    texts = []
    for t in d:
        s = t[variant]
        if not s or 0 in s: continue
        if t.get('complete') not in ('Y', 'y', ''): pass
        texts.append((t['site'], tuple(s)))
    return texts

def load_ur3(min_len=3, max_len=7):
    legs = json.load(open(SP + 'ur3_legends.json'))
    seen = set(); out = []
    for lg in legs:
        flat = tuple(x for ln in lg['lines'] for x in ln)
        if not (min_len <= len(flat) <= max_len) or flat in seen: continue
        seen.add(flat)
        out.append(('train' if rng.random() < 0.7 else 'test', flat))
    return out

class Tri:
    def __init__(self, texts, V):
        self.u = collections.Counter(); self.b = collections.Counter(); self.t = collections.Counter()
        self.bc = collections.Counter(); self.tc = collections.Counter()
        for s in texts:
            s = ('<s>', '<s>') + s + ('</s>',)
            for i in range(2, len(s)):
                self.u[s[i]] += 1
                self.b[(s[i-1], s[i])] += 1; self.bc[s[i-1]] += 1
                self.t[(s[i-2], s[i-1], s[i])] += 1; self.tc[(s[i-2], s[i-1])] += 1
        self.N = sum(self.u.values()); self.V = V
    def lp(self, a, b, c):
        pu = (self.u[c] + 0.5) / (self.N + 0.5 * self.V)
        pb = (self.b[(b, c)] + 1.0 * pu * 4) / (self.bc[b] + 4) if self.bc[b] else pu
        pt = (self.t[(a, b, c)] + 2 * pb) / (self.tc[(a, b)] + 2) if self.tc[(a, b)] else pb
        return math.log(pt)
    def nll(self, texts):
        tot = 0; n = 0
        for s in texts:
            s = ('<s>', '<s>') + s + ('</s>',)
            for i in range(2, len(s)):
                tot -= self.lp(s[i-2], s[i-1], s[i]); n += 1
        return tot / n

def entropy(cnt):
    n = sum(cnt.values())
    return -sum(c / n * math.log2(c / n) for c in cnt.values())

def templates(L, cap=2000):
    ident = tuple(range(L)); rev = tuple(reversed(ident))
    if math.factorial(L) <= cap:
        return list(itertools.permutations(range(L)))
    seen = {ident, rev}
    while len(seen) < cap:
        p = list(range(L)); rng.shuffle(p); seen.add(tuple(p))
    return [ident, rev] + [p for p in seen if p not in (ident, rev)]

def apply(texts, pi):
    return [tuple(s[i] for i in pi) for s in texts]

def score(train, test, allt, pi, V):
    tr = apply(train, pi); te = apply(test, pi); al = apply(allt, pi)
    m = Tri(tr, V)
    ppl = m.nll(te)
    slot = entropy(collections.Counter(s[0] for s in al)) + entropy(collections.Counter(s[-1] for s in al))
    return ppl, slot

def run(texts, is_train, label, lines, shuffle_reps=0):
    V = len({x for _, s in texts for x in s}) + 1
    res = {}
    for L in range(3, 8):
        tl = [s for g, s in texts if len(s) == L]
        train = [s for g, s in texts if len(s) == L and is_train(g)]
        test = [s for g, s in texts if len(s) == L and not is_train(g)]
        if len(train) < 30 or len(test) < 15:
            lines.append(f'{label} L={L}: too few texts (train {len(train)}, test {len(test)})'); continue
        T = templates(L)
        sc = {pi: score(train, test, tl, pi, V) for pi in T}
        ident = tuple(range(L)); rev = tuple(reversed(ident))
        p0, s0 = sc[ident]
        byp = sorted(T, key=lambda p: sc[p][0]); bys = sorted(T, key=lambda p: sc[p][1])
        rank_p = byp.index(ident) + 1; rank_s = bys.index(ident) + 1
        best_p = byp[0]; best_s = bys[0]
        gain_p = p0 - sc[best_p][0]; gain_s = s0 - sc[best_s][1]
        # best non-trivial (neither identity nor reverse)
        nt_p = next(p for p in byp if p not in (ident, rev)); nt_s = next(p for p in bys if p not in (ident, rev))
        # null: shuffle signs inside each text, repeat search, record best gain over identity
        null_p = []; null_s = []
        for r in range(shuffle_reps):
            sh = [tuple(rng.sample(s, L)) for s in tl]
            shtr = [tuple(rng.sample(s, L)) for s in train]; shte = [tuple(rng.sample(s, L)) for s in test]
            sub = T if len(T) <= 720 else rng.sample(T, 400) + [ident, rev]
            ssc = {pi: score(shtr, shte, sh, pi, V) for pi in sub}
            q0, t0 = ssc[ident]
            null_p.append(q0 - min(v[0] for v in ssc.values())); null_s.append(t0 - min(v[1] for v in ssc.values()))
        res[L] = dict(n_train=len(train), n_test=len(test), n_templates=len(T), ppl_identity=p0, slot_identity=s0,
                      ppl_reverse=sc[rev][0], rank_identity_ppl=rank_p, rank_identity_slot=rank_s,
                      best_ppl=list(best_p), gain_ppl=gain_p, best_slot=list(best_s), gain_slot=gain_s,
                      best_nontrivial_ppl=list(nt_p), gain_nontrivial_ppl=p0 - sc[nt_p][0],
                      best_nontrivial_slot=list(nt_s), gain_nontrivial_slot=s0 - sc[nt_s][1],
                      null_best_gain_ppl=null_p, null_best_gain_slot=null_s)
        nb = (f' | null best gain (shuffled, {shuffle_reps} reps): ppl max {max(null_p):.4f} med {sorted(null_p)[len(null_p)//2]:.4f}; '
              f'slot max {max(null_s):.3f}') if shuffle_reps else ''
        lines.append(f'{label} L={L} (train {len(train)}, test {len(test)}, {len(T)} templates): identity nll/token {p0:.4f} '
                     f'(rank {rank_p}/{len(T)}), reverse {sc[rev][0]:.4f}; best {best_p} gain {gain_p:+.4f}; '
                     f'best non-trivial {nt_p} gain {p0 - sc[nt_p][0]:+.4f}. Slot H(first)+H(last) identity {s0:.3f} '
                     f'(rank {rank_s}/{len(T)}), best {best_s} gain {gain_s:+.3f}, best non-trivial {nt_s} gain {s0 - sc[nt_s][1]:+.3f}{nb}')
    return res

if __name__ == '__main__':
    lines = []; allres = {}
    ur3 = load_ur3()
    lines.append(f'Ur III legends (CDLI, Ur III period, @seal lines, unique, lengths 3-7): {len(ur3)} legends')
    allres['ur3'] = run(ur3, lambda g: g == 'train', 'UR3', lines, shuffle_reps=0)
    for variant in ('seq_raw', 'seq_strong', 'seq_all'):
        tx = load_indus(variant)
        md = lambda g: g in ('Mohenjo-daro', 'Harappa')
        lines.append(f'Indus {variant}: {len(tx)} texts without damage marks; train = Mohenjo-daro + Harappa, test = other sites')
        allres[variant] = run(tx, md, f'INDUS-{variant}', lines, shuffle_reps=20 if variant == 'seq_raw' else 5)
    json.dump(allres, open(OUT + 'loop8_cycle1_permute.json', 'w'), indent=1)
    open(OUT + 'loop8_cycle1_log.txt', 'w').write('\n'.join(lines) + '\n')
    print('\n'.join(lines))
