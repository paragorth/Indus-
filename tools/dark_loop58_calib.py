"""S-DARK-58 cycle 3: calibration. Identical position-class code on every corpus.
Role of a token type = its position class in the long texts (>= 4 tokens) of the same corpus:
  FIRST    >= 50% of its long-text tokens stand first
  LAST     >= 50% stand last
  INTERIOR otherwise (>= 3 long-text tokens)
  RARE     fewer than 3 tokens in long texts (incl. never seen there)
Then the class mix of 1-token and 2-token texts against two frequency nulls (all tokens; long-text tokens; 1,000x), plus the
hapax share (type seen once in the whole corpus). Corpora: Ur III seal legends (words; no 1-word legends in the CDLI set,
so 2-word legends are its minimal texts), Linear B tablet lines (DAMOS; sign groups + ideograms + NUM), Latin EDH Italy
(words), Indus Wells seq_raw/strong/all by object type, IM77 (Mahadevan numbers, no bridge needed here).
Usage: python3 tools/dark_loop58_calib.py [nperm]
"""
import sys, os, json, csv, random, collections, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dark_loop58 as D
ROOT = D.ROOT; DARK = D.DARK
NP = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
rnd = random.Random(583)
LOG = []
def P(*a):
    s = ' '.join(str(x) for x in a); print(s); LOG.append(s)
CLASSES = ['FIRST', 'LAST', 'INTERIOR', 'RARE']

def jl(path): return [tuple(json.loads(l)['seq']) for l in open(path)]

def pos_roles(texts, minlen=4, mink=3):
    pos = collections.defaultdict(collections.Counter)
    for s in texts:
        if len(s) < minlen: continue
        for i, a in enumerate(s):
            pos[a]['first' if i == 0 else 'last' if i == len(s) - 1 else 'mid'] += 1
    role = {}
    for a, c in pos.items():
        n = sum(c.values())
        if n < mink: role[a] = 'RARE'
        elif c['first'] / n >= 0.5: role[a] = 'FIRST'
        elif c['last'] / n >= 0.5: role[a] = 'LAST'
        else: role[a] = 'INTERIOR'
    return role

def mixc(signs, role): return collections.Counter(role.get(a, 'RARE') for a in signs)

def run(label, texts, nlen, nperm=NP):
    role = pos_roles(texts)
    freq = collections.Counter(a for s in texts for a in s)
    pool_all = [a for s in texts for a in s]; pool_long = [a for s in texts if len(s) >= 4 for a in s]
    sub = [s for s in texts if len(s) == nlen]
    if len(sub) < 5:
        P(f'  -- {label} {nlen}-token: n = {len(sub)}, too few'); return None
    signs = [a for s in sub for a in s]; n = len(signs); obs = mixc(signs, role)
    hap = sum(1 for a in signs if freq[a] == 1) / n
    nulls = {}
    for pname, pool in (('all', pool_all), ('long', pool_long)):
        draws = collections.defaultdict(list); hs = []
        for _ in range(nperm):
            d = rnd.choices(pool, k=n); c = mixc(d, role)
            for r in CLASSES: draws[r].append(c.get(r, 0))
            hs.append(sum(1 for a in d if freq[a] == 1) / n)
        nulls[pname] = (draws, hs)
    out = {'n_texts': len(sub), 'n_tokens': n, 'obs': {r: obs.get(r, 0) for r in CLASSES}, 'hapax': hap}
    line = f'  -- {label} {nlen}-token texts: {len(sub)} texts; class shares obs (null-all / null-long mean, two-sided P): '
    for r in CLASSES:
        o = obs.get(r, 0); parts = []
        for pname in ('all', 'long'):
            nm = nulls[pname][0][r]; m = sum(nm) / len(nm)
            p2 = min(1, 2 * min(sum(1 for v in nm if v >= o), sum(1 for v in nm if v <= o)) / len(nm))
            parts.append(f'{m/n:.2f} P={p2:.3f}'); out[f'{r}_{pname}'] = dict(mean=m / n, p=p2)
        line += f'{r} {o/n:.2f} ({parts[0]} / {parts[1]}); '
    hs_all = nulls['all'][1]; hs_long = nulls['long'][1]
    line += f'HAPAX {hap:.2f} (null-all {sum(hs_all)/len(hs_all):.2f}, null-long {sum(hs_long)/len(hs_long):.2f})'
    P(line)
    return out

def indus_wells(level):
    rows = D.load_corpus(level)
    out = {}
    for ot in ('seal', 'tablet', 'pot', 'ALL'):
        T = [tuple(x['seq']) for x in rows if x['clean'] and (ot == 'ALL' or x['ot'] == ot)]
        # roles are learned on ALL clean texts of the corpus (as for every calibration corpus), but the minimal set is the type
        out[ot] = T
    return rows, out

def main():
    P(f'== S-DARK-58 cycle 3: position-class calibration, nperm {NP}')
    RES = {}
    ur3 = jl(ROOT + 'data/codelib/ur3_legends.jsonl'); linb = jl(ROOT + 'data/codelib/linear_b.jsonl'); edh = jl(ROOT + 'data/codelib/latin_edh.jsonl')
    # Linear B: drop editorial tokens (deest, mut., inf., sup., lat., v., Greek letters, dotted digits) so that lines are text
    EDIT = {'deest', 'mut.', 'mut.?', 'inf.', 'sup.', 'lat.', 'v.', 'v.↓', 'v.→', 'sigillum', 'vacat', 'vac.', 'α', 'β', 'γ', 'δ', 'ε', 'ζ', 'η'}
    import re
    def ok(t): return t not in EDIT and not re.fullmatch(r'[0-9]̣?|[̣]+', t) and not t.startswith('v.') and not re.fullmatch(r'\d+̣', t)
    linb2 = [tuple(a for a in s if ok(a)) for s in linb]; linb2 = [s for s in linb2 if s]
    P(f'   Ur III legends {len(ur3)} (lengths 1:{sum(len(s)==1 for s in ur3)} 2:{sum(len(s)==2 for s in ur3)}); '
      f'Linear B lines {len(linb2)} after dropping editorial tokens (1:{sum(len(s)==1 for s in linb2)} 2:{sum(len(s)==2 for s in linb2)}); '
      f'Latin EDH {len(edh)} (1:{sum(len(s)==1 for s in edh)} 2:{sum(len(s)==2 for s in edh)})')
    for lab, T in (('UrIII-legends', ur3), ('LinearB-lines', linb2), ('Latin-EDH', edh)):
        role = pos_roles(T); rc = collections.Counter(role.values())
        P(f'   {lab}: types with a position class {len(role)}: {dict(rc)}; FIRST examples {[a for a in role if role[a]=="FIRST"][:6]}; LAST examples {[a for a in role if role[a]=="LAST"][:6]}')
        for n in (1, 2):
            r = run(lab, T, n)
            if r: RES[f'{lab}/{n}'] = r
    # what are the 1-token Linear B entries and 1-word Latin texts
    c1 = collections.Counter(s[0] for s in linb2 if len(s) == 1); P('   Linear B 1-token lines, commonest:', c1.most_common(15))
    c1 = collections.Counter(s[0] for s in edh if len(s) == 1); P('   Latin 1-word texts, commonest:', c1.most_common(12))
    c2 = collections.Counter(s for s in ur3 if len(s) == 2); P('   Ur III 2-word legends, second-word classes:',
        collections.Counter(pos_roles(ur3).get(s[1], 'RARE') for s in ur3 if len(s) == 2).most_common(), 'first-word:', collections.Counter(pos_roles(ur3).get(s[0], 'RARE') for s in ur3 if len(s) == 2).most_common())
    # Indus
    for level in ('seq_raw', 'seq_strong', 'seq_all'):
        rows, T = indus_wells(level)
        allT = T['ALL']; role = pos_roles(allT)
        P(f'   Indus Wells {level}: {len(allT)} clean texts; classes {dict(collections.Counter(role.values()))}; '
          f'LAST signs {sorted(a for a in role if role[a]=="LAST")}; FIRST signs {sorted(a for a in role if role[a]=="FIRST")}')
        for ot in ('seal', 'tablet', 'pot'):
            # minimal texts of this type; roles and pools from the whole corpus (same as the calibration corpora)
            for n in (1, 2):
                sub = [s for s in T[ot] if len(s) == n]
                mixed = [s for s in allT if len(s) != n] + sub     # whole-corpus roles/pools, this type's minimal texts
                r = run(f'Indus-{level}-{ot}', mixed, n)
                if r: RES[f'Indus-{level}-{ot}/{n}'] = r
    # IM77 in its own numbers
    T = D.load_im77()
    for ot in ('seal', 'tablet', 'pot'):
        allT = [tuple(x['seq']) for x in T if x['clean']]
        for n in (1, 2):
            sub = [tuple(x['seq']) for x in T if x['clean'] and x['ot'] == ot and len(x['seq']) == n]
            mixed = [s for s in allT if len(s) != n] + sub
            r = run(f'IM77-{ot}', mixed, n)
            if r: RES[f'IM77-{ot}/{n}'] = r
    json.dump(RES, open(DARK + 'loop58_c3.json', 'w'), indent=0)
    open(DARK + 'loop58_c3_calib.txt', 'w').write('\n'.join(LOG) + '\n')

if __name__ == '__main__':
    main()
