"""Loop 73 cycle 1b: information share of the designation in the whole entry (S-DARK-59 method, simplified to one
identical model for every corpus). Each entry is a token string with slot labels DES (designation), FRAME (prefix /
opener / marker / class sign / closer / suffix / title words), QUAL (Indus title = qualifier before the closer),
COUNT (numeral tokens) and END. Model: Witten-Bell interpolated bigram over a unigram with an UNK mass, fitted on 4/5
of the documents and scored on the held-out fifth (5-fold, folds grouped by tablet where a tablet exists); also the
unigram alone. Reported: bits per entry by label group, share of bits carried by the designation (of all bits; of
non-count bits), and gap closed vs the unigram per group.
Corpora: PE entries (prefix + middle + class sign + numeral codes), Indus seq_raw / seq_strong / seq_all / IM77 whole
texts (parser labels), Ur III seal legends (owner name syllables vs following words), Linear B personnel lines
(name syllables vs ideogram / annotation / NUM).
Usage: python3 tools/dark_loop73_c1b.py
"""
import sys
sys.path.insert(0, '/home/user/Indus-/tools')
from dark_loop73_common import *

class WB:
    def __init__(self, docs):
        self.u = collections.Counter(); self.b = collections.defaultdict(collections.Counter)
        for toks in docs:
            prev = '<s>'
            for t in toks + ['</s>']:
                self.u[t] += 1; self.b[prev][t] += 1; prev = t
        self.N = sum(self.u.values()); self.V = len(self.u)
    def puni(self, t):
        return (self.u[t] + 0.5) / (self.N + 0.5 * (self.V + 1)) if t in self.u else 0.5 / (self.N + 0.5 * (self.V + 1))
    def pbi(self, h, t):
        c = self.b.get(h)
        if not c: return self.puni(t)
        n = sum(c.values()); T = len(c)
        return (c[t] + T * self.puni(t)) / (n + T)

def score(train, test, labtest):
    m = WB(train); out = collections.defaultdict(lambda: [0.0, 0.0, 0])
    for toks, labs in zip(test, labtest):
        prev = '<s>'
        for t, l in zip(toks + ['</s>'], list(labs) + ['END']):
            out[l][0] += -math.log2(m.pbi(prev, t)); out[l][1] += -math.log2(m.puni(t)); out[l][2] += 1; prev = t
    return out

def run(label, docs, groups, k=5, seed=73):
    """docs: list of (tokens, labels); groups: list of group ids (fold by group)."""
    G = sorted(set(groups)); r = random.Random(seed); r.shuffle(G); fold = {g: i % k for i, g in enumerate(G)}
    tot = collections.defaultdict(lambda: [0.0, 0.0, 0])
    for f in range(k):
        tr = [list(d[0]) for d, g in zip(docs, groups) if fold[g] != f]
        te = [(list(d[0]), d[1]) for d, g in zip(docs, groups) if fold[g] == f]
        o = score(tr, [x[0] for x in te], [x[1] for x in te])
        for l, v in o.items():
            for i in range(3): tot[l][i] += v[i]
    n = len(docs); allb = sum(v[0] for v in tot.values()); nonc = sum(v[0] for l, v in tot.items() if l != 'COUNT')
    res = dict(n=n, bits_per_entry=allb / n, groups={})
    line = f'  [{label}] n={n}: bits/entry {allb/n:.2f} (unigram {sum(v[1] for v in tot.values())/n:.2f}); '
    for l in ['DES', 'FRAME', 'QUAL', 'COUNT', 'END']:
        if l not in tot: continue
        b, u, c = tot[l]
        res['groups'][l] = dict(bits_entry=b / n, bits_tok=b / c, uni_tok=u / c, gap=1 - b / u if u else float('nan'), share=b / allb, tokens=c)
        line += f'{l} {b/n:.2f} b/entry ({b/c:.2f} b/tok, unigram {u/c:.2f}, gap closed {1-b/u:.2f}, share {b/allb:.2f}); '
    res['des_share_all'] = tot['DES'][0] / allb; res['des_share_noncount'] = tot['DES'][0] / nonc if nonc else float('nan')
    line += f'DES share of all bits {res["des_share_all"]:.3f}, of non-count bits {res["des_share_noncount"]:.3f}'
    P(line)
    return res

R = {}
P('== Loop 73 cycle 1b: information share of the designation (identical Witten-Bell bigram, 5-fold)')
E = pe_entries()
docs = []; grp = []
for e in E:
    toks = list(e['prefix']) + list(e['mid']) + ([e['cls']] if e['cls'] else [])
    labs = ['FRAME'] * len(e['prefix']) + ['DES'] * len(e['mid']) + (['FRAME'] if e['cls'] else [])
    nums = e['num'][4:].split('+')
    toks += nums; labs += ['COUNT'] * len(nums)
    docs.append((toks, labs)); grp.append(e['tablet'])
R['PE_entries'] = run('PE entries, folds by tablet', docs, grp)
R['PE_entries_withmid'] = run('PE entries with a middle >= 1', [d for d, e in zip(docs, E) if e['mid']], [g for g, e in zip(grp, E) if e['mid']])
R['PE_entries_mid2'] = run('PE entries with a middle >= 2', [d for d, e in zip(docs, E) if len(e['mid']) >= 2], [g for g, e in zip(grp, E) if len(e['mid']) >= 2])
for LV in ['seq_raw', 'seq_strong', 'seq_all', 'im77']:
    O = indus_objs(LV)
    MAP = {'OPENER': 'FRAME', 'MARKER': 'FRAME', 'CLOSER': 'FRAME', 'SUFFIX': 'FRAME', 'TITLE': 'QUAL', 'NAME': 'DES', 'COUNT': 'COUNT'}
    d = [(list(o['seq']), [MAP[l] for l in o['lab']]) for o in O]
    R['Indus_' + LV] = run(f'Indus {LV} whole texts (one per site x type x text), random folds', d, list(range(len(d))))
    s = [(list(o['seq']), [MAP[l] for l in o['lab']]) for o in O if o['ot'] == 'seal']
    R['Indus_' + LV + '_seals'] = run(f'Indus {LV} seals', s, list(range(len(s))))
    if LV != 'im77':
        h = [x for x, o in zip(d, O)]; g = [o['site'] if not o['big'] else ('MD' if o['site'] == 'Mohenjo-daro' else 'H') for o in O]
        R['Indus_' + LV + '_sitefold'] = run(f'Indus {LV} whole texts, folds by site group', h, g, k=3)
U = ur3_legends_labeled()
R['UrIII_legends'] = run('Ur III legends (name syllables = DES, other words = FRAME), random folds', [(list(t), list(l)) for t, l in U], list(range(len(U))))
LB = linb_lines()
d = [(list(x['name']) + list(x['frame']), ['DES'] * len(x['name']) + ['COUNT' if f == 'NUM' else 'FRAME' for f in x['frame']]) for x in LB]
R['LinB_lines'] = run('Linear B personnel lines (name = DES; ideogram / annotation = FRAME; NUM = COUNT), folds by tablet', d, [x['tablet'] for x in LB])
save('loop73_c1b', R)
