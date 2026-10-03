"""Loop 59 cycle 3: the same decomposition for Ur III seal legends (name vs title words; and syllables) and Linear B
tablet lines (sign-groups vs ideograms / numerals), through the same code (unigram, KN1, KN2, KN3, site cache,
global cache, EM-interpolated combination), fitted on one site and scored on the others, so that the Indus split
(frame / middle) can be read against a known name + title legend system and a known administrative list.
Usage: python3 tools/dark_loop59_c3.py   -> data/derived/dark/loop59_c3.txt / .json
"""
import sys, json, math, random, collections, time
sys.path.insert(0, '/home/user/Indus-/tools')
from dark_loop59_common import Unigram, KN, Cache, END, UNK, DARK, bootstrap_ce
OUT = DARK + 'loop59_c3'
logf = open(OUT + '.txt', 'w')
def log(*a):
    s = ' '.join(str(x) for x in a); print(s); logf.write(s + '\n'); logf.flush()

UR3_TITLES = {'dumu', 'dub-sar', 'arad2', 'arad', 'arad2-zu', 'ir11', 'ir11-zu', 'lugal', 'ensi2', 'sagi', 'szagina', 'nu-banda3', 'nu-banda3-gu4',
              'ugula', 'szabra', 'gudu4', 'kuruszda', 'sukkal', 'ra2-gaba', 'muhaldim', 'aga3-us2', 'sanga', 'dam-gar3', 'nar', 'gala', 'iszib',
              'kiszib3', 'sipa', 'szusz3', 'lu2', 'ka-guru7', 'gal5-la2-gal', 'kal-ga', 'umma{ki}', 'uri5{ki}-ma', 'an-ub-da', 'limmu2-ba',
              'sza13-dub-ba-ka', 'sa12-du5-ka', 'dumu-ni', 'ensi2-ka', 'lugal-kal-ga', 'ma-da', 'engar', 'simug', 'aszgab', 'nagar', 'ma2-lah5',
              'gu-za-la2', 'lu2-kas4', 'szu-i', 'lunga', 'ad-kup4', 'nu-{gesz}kiri6', 'tug2-du8', 'azlag2', 'bahar2', 'zadim', 'ku3-dim2', 'unu3'}
def ur3_label(w):
    if w in UR3_TITLES or w.startswith('{d}') or w.startswith('_') or w.endswith('{ki}') or w.endswith('{ki}-ma'): return 'TITLE'
    return 'NAME'
def ur3_syll_labels(words_seq, syll_seq):
    """label syllables by the word they come from; words are hyphen-joined syllables"""
    labs = []
    for w in words_seq:
        n = max(1, len([p for p in w.replace('{', '-{').replace('}', '}-').split('-') if p]))
        labs += [ur3_label(w)] * n
    if len(labs) != len(syll_seq): return None
    return labs
def linb_label(w):
    if w.isupper() or w in ('NUM',) or any(ch.isdigit() for ch in w) or w.startswith('.') or w.endswith('.') or w.startswith('*') or ':' in w or len(w) <= 1: return 'FORMULA'
    return 'WORD'

def load(name):
    return [json.loads(l) for l in open(DARK + f'loop48_corpora/{name}.jsonl')]

def run(corpus_name, texts, labeller, fit_sites, max_fit=4000, max_test=1500, seed=3, slots=('NAME', 'TITLE')):
    rnd = random.Random(seed)
    fit = [t for t in texts if t['site'] in fit_sites]; held = [t for t in texts if t['site'] not in fit_sites and t['site'] != 'uncertain']
    rnd.shuffle(fit); rnd.shuffle(held); fit = fit[:max_fit]; held = held[:max_test]
    for t in fit + held: t['ot'] = 'X'
    V = set(x for t in fit for x in t['seq'])
    uni = Unigram(fit, V); k1 = KN(fit, 1, uni); k2 = KN(fit, 2, uni); k3 = KN(fit, 3, uni)
    cs = Cache(fit, True); cg = Cache(fit, False)
    # EM weights for the combination on 4 folds of the fit set
    idx = list(range(len(fit))); obs = []
    for f in range(4):
        tr = [fit[i] for j, i in enumerate(idx) if j % 4 != f]; te = [fit[i] for j, i in enumerate(idx) if j % 4 == f]
        u = Unigram(tr, V); kk = [KN(tr, 1, u), KN(tr, 2, u), KN(tr, 3, u)]; c1 = Cache(tr, True); c2 = Cache(tr, False)
        for t in te:
            s = list(t['seq']) + [END]
            for i in range(len(s)):
                h = s[:i]; ps = [u.p(s[i], 'X')] + [k.p(s[i], t, h) for k in kk]
                a = c1.p(s[i], t, h); b = c2.p(s[i], t, h); ps += [a if a is not None else -1, b if b is not None else -1]
                obs.append(ps)
            c1.add(t); c2.add(t)
    names = ['unigram', 'KN1', 'KN2', 'KN3', 'cache_site', 'cache_all']
    W = {}
    for pat in set(tuple(p >= 0 for p in ps) for ps in obs):
        L = [ps for ps in obs if tuple(p >= 0 for p in ps) == pat]; act = [k for k in range(6) if pat[k]]
        w = {k: 1 / len(act) for k in act}
        for _ in range(30):
            acc = collections.Counter()
            for ps in L:
                tot = max(sum(w[k] * ps[k] for k in act), 1e-300)
                for k in act: acc[k] += w[k] * ps[k] / tot
            Z = sum(acc.values()); w = {k: (acc[k] + 0.01) / (Z + 0.01 * len(act)) for k in act}
        W[pat] = w
    def combo(x, t, h):
        ps = [uni.p(x, 'X'), k1.p(x, t, h), k2.p(x, t, h), k3.p(x, t, h)]
        a = cs.p(x, t, h); b = cg.p(x, t, h); ps += [a if a is not None else -1, b if b is not None else -1]
        pat = tuple(p >= 0 for p in ps); w = W.get(pat) or {k: 1 / sum(pat) for k in range(6) if pat[k]}
        return sum(w[k] * ps[k] for k in w)
    models = {'uniform': lambda x, t, h: 1 / (len(V) + 2), 'unigram': lambda x, t, h: uni.p(x, 'X'), 'KN1': k1.p, 'KN2': k2.p, 'KN3': k3.p, 'combined': combo}
    recs = []
    for ti, t in enumerate(held):
        s = list(t['seq']); labs = labeller(t)
        if labs is None: continue
        labs = labs + ['END']; toks = s + [END]
        for i in range(len(toks)):
            h = s[:i]; bits = {m: -math.log2(max(f(toks[i], t, h), 1e-12)) for m, f in models.items()}
            recs.append(dict(t=ti, slot=labs[i], x=toks[i], bits=bits))
        cs.add(t); cg.add(t)
    nt = len(set(r['t'] for r in recs)); oov = sum(1 for r in recs if r['x'] != END and r['x'] not in V) / max(1, sum(1 for r in recs if r['x'] != END))
    log(f'\n## {corpus_name}: fit {len(fit)} texts ({sum(len(t["seq"]) for t in fit)} tokens, |V| {len(V)}) on {sorted(fit_sites)}; held-out {nt} texts from other sites; OOV {oov:.3f}')
    log('| model | bits/token (95% CI) | gap closed vs unigram | ' + ' | '.join(f'{s} bits [gap]' for s in list(slots) + ['END']) + ' |')
    log('|---|---|---|' + '---|' * (len(slots) + 1))
    R = {'fit': len(fit), 'held': nt, 'V': len(V), 'oov': oov, 'models': {}}
    def ce(m, sel=lambda r: True):
        L = [r['bits'][m] for r in recs if sel(r)]; return sum(L) / len(L) if L else float('nan')
    u = ce('unigram')
    for m in models:
        c = ce(m); lo, hi = bootstrap_ce(recs, m, len(held), reps=400)
        cells = []
        R['models'][m] = {'bits': c, 'lo': lo, 'hi': hi, 'slots': {}}
        for s in list(slots) + ['END']:
            cu = ce('unigram', lambda r, s=s: r['slot'] == s); cm = ce(m, lambda r, s=s: r['slot'] == s)
            cells.append(f'{cm:.2f} [{100*(1-cm/cu):.0f}%]'); R['models'][m]['slots'][s] = cm
        log(f'| {m} | {c:.3f} [{lo:.3f}, {hi:.3f}] | {100*(1-c/u):.1f}% | ' + ' | '.join(cells) + ' |')
    best = 'combined'; tb = collections.Counter(); ub = 0.0; ns = collections.Counter()
    for r in recs: tb[r['slot']] += r['bits'][best]; ub += r['bits']['unigram']; ns[r['slot']] += 1
    T = sum(tb.values())
    log(f'bits per text under {best}: {T/nt:.1f} (unigram {ub/nt:.1f}; explained {100*(1-T/ub):.0f}%); by slot: ' +
        ', '.join(f'{s} {tb[s]/nt:.1f} bits ({100*tb[s]/T:.0f}% of remaining; {ns[s]/nt:.2f} tokens/text)' for s in list(slots) + ['END']))
    irr = sum(1 for r in recs if min(r['bits'][m] for m in models if m != 'uniform') >= r['bits']['unigram'] - 0.25)
    irr_s = {s: sum(1 for r in recs if r['slot'] == s and min(r['bits'][m] for m in models if m != 'uniform') >= r['bits']['unigram'] - 0.25) / ns[s] for s in ns}
    log(f'irreducible tokens (no model beats the unigram by 0.25 bit): {irr/len(recs):.3f}; by slot ' + ', '.join(f'{s} {v:.2f}' for s, v in irr_s.items()))
    pred = {s: sum(1 for r in recs if r['slot'] == s and r['bits'][best] <= 1) / ns[s] for s in ns}
    log('share predicted with P >= 0.5 under best: ' + ', '.join(f'{s} {v:.2f}' for s, v in pred.items()))
    R['bits_text'] = {s: tb[s] / nt for s in tb}; R['uni_text'] = ub / nt; R['irr'] = irr_s; R['pred'] = pred
    R['name_share'] = tb[slots[0]] / T
    return R

t0 = time.time(); RES = {}
log(f'# loop 59 cycle 3 ({time.strftime("%Y-%m-%d %H:%M")}): Ur III seal legends and Linear B lines through the same predictive ladder')
# Ur III words: fit Umma, test Girsu + Puzrish-Dagan + Nippur + ...
ur3w = load('ur3_words')
# one copy per distinct legend per site (impressions repeat the same seal: a 'die' regime as for Indus)
seen = set(); ur3wd = []
for t in ur3w:
    k = (t['site'], tuple(t['seq']))
    if k in seen: continue
    seen.add(k); ur3wd.append(t)
log(f'Ur III legends: {len(ur3w)} impressions -> {len(ur3wd)} distinct (site, legend)')
RES['ur3_words'] = run('Ur III legends, words (name / title), one per distinct legend', ur3wd, lambda t: [ur3_label(w) for w in t['seq']], {'Umma'})
RES['ur3_words_rows'] = run('Ur III legends, words, all impressions (repeats kept)', ur3w, lambda t: [ur3_label(w) for w in t['seq']], {'Umma'}, max_fit=6000)
# Ur III syllables, labelled through the word corpus (same order of texts in both files; align by index when lengths match)
ur3s = load('ur3_syll')
pairs = []
wi = 0
for s in ur3s:
    # find the matching words line (same site, same syllable count) scanning forward
    while wi < len(ur3w) and (ur3w[wi]['site'] != s['site'] or ur3_syll_labels(ur3w[wi]['seq'], s['seq']) is None):
        wi += 1
        if wi - len(pairs) > 50: break
    if wi < len(ur3w):
        labs = ur3_syll_labels(ur3w[wi]['seq'], s['seq'])
        if labs is not None: s['labs'] = labs; pairs.append(s); wi += 1
seen = set(); ur3sd = []
for t in pairs:
    k = (t['site'], tuple(t['seq']))
    if k in seen: continue
    seen.add(k); ur3sd.append(t)
log(f'Ur III syllables aligned to words: {len(pairs)} of {len(ur3s)}; distinct {len(ur3sd)}')
RES['ur3_syll'] = run('Ur III legends, syllable signs (name / title syllables), one per distinct legend', ur3sd, lambda t: t['labs'], {'Umma'})
# Linear B words: fit Knossos, test Pylos + Thebes + Mycenae
linb = load('linb_words')
RES['linb_words'] = run('Linear B lines, sign-groups vs ideograms/numerals, fit KN test PY/TH/MY', linb, lambda t: [linb_label(w) for w in t['seq']], {'KN'}, slots=('WORD', 'FORMULA'))
json.dump(RES, open(OUT + '.json', 'w'))
log(f'\ndone in {time.time()-t0:.0f}s')
