#!/usr/bin/env python3
"""Attacks A and B on Linear A.

A. Linear B anchors as role probes.
   1. Classify each Linear A / Linear B shared word (3+ signs; 2-sign matches as a
      secondary set) as place-like / person-like / unclear, using only Linear B
      (DAMOS) context and explicit rules (below).  No published interpretations.
   2. In Linear A, measure slot features of the place-like vs person-like anchors.
   3. Controls: random Linear A word types matched for sign length and token
      frequency (1000 draws); label permutation among the anchors.

B. Affix paradigms.
   1. Families: attested Linear A types sharing a core (>=2 signs) after removing
      one prefix (I, SI, A, KI, JA) and/or one suffix (ME, JA, RE, TE).
   2. Productivity control: random sign of matched positional frequency added to
      the same cores (1000 draws).
   3. Feature ties of each affix (document class, position, number, commodity,
      site) vs the sibling tokens of the same families (permutation within the
      family pool, 2000 runs).
   4. Hold-out prediction: families fitted on Hagia Triada only; predict missing
      common affixed forms; check them at all other sites.  Control: random signs.

Inputs: data/corpus.json, data/damos_items.jsonl.  Outputs: data/attack_anchors.json
"""
import json, os, re, random, unicodedata
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, '..', 'data')
SUB = str.maketrans('₂₃', '23')
random.seed(11)
NR = 1000

# ----------------------------------------------------------------- Linear B side
def load_lb():
    docs = []
    for l in open(os.path.join(D, 'damos_items.jsonl')):
        r = json.loads(l)
        if 'error' in r or not r.get('content'): continue
        txt = unicodedata.normalize('NFD', r['content'])
        txt = re.sub(r'\[|\]|\?|⸤|⸥|⌞|⌟|<|>|\{|\}|⟦|⟧', '', txt)
        txt = re.sub(r'[̀-ͯ]', '', txt)          # underdots etc.
        lines = []
        for ln in txt.split('\n'):
            toks = ln.split()
            if toks and re.fullmatch(r'\.?[0-9A-Za-z]{1,3}', toks[0]) and toks[0].startswith('.'):
                toks = toks[1:]                          # line label
            out = []
            for t in toks:
                t = t.strip(',')
                if not t: continue
                if t in ('/',): out.append(('sep', '/')); continue
                if re.fullmatch(r'[a-z][a-z0-9*]*(-[a-z0-9*]+)*', t): out.append(('w', t)); continue
                if re.fullmatch(r'[0-9]+', t): out.append(('n', int(t))); continue
                if re.fullmatch(r'[A-Z][A-Za-z0-9+:*;]*', t): out.append(('i', t)); continue
                out.append(('x', t))
            lines.append(out)
        docs.append({'head': r.get('heading') or '?', 'lines': lines})
    return docs

SYL_I = {}
def to_i(sign):                      # change the vowel of a CV sign to i
    m = re.fullmatch(r'([a-z]*?)([aeiou])([0-9]?)', sign)
    if not m: return None
    return m.group(1) + 'i'

def lb_features(word, docs, vocab, seeds, base_seed_rate):
    on_last = 0
    occ_docs = 0; cooc = 0; partners = set(); q_vir = 0; q_one = 0; after_toso = 0
    big_num = 0; n_tok = 0; series = Counter()
    for d in docs:
        hit = False; dw = set(w for ln in d['lines'] for k, w in ln if k == 'w')
        nonempty = [ln for ln in d['lines'] if ln]
        for ln in d['lines']:
            for j, (k, v) in enumerate(ln):
                if k == 'w' and v == word:
                    hit = True; n_tok += 1
                    if nonempty and ln is nonempty[-1] and len(nonempty) >= 3: on_last += 1
                    prev = ln[j - 1] if j > 0 else None
                    # person rule: entry starts a segment (nothing, or a number, before it)
                    start = prev is None or prev[0] == 'n'
                    nx = ln[j + 1:j + 3]
                    if start and len(nx) >= 2 and nx[0][0] == 'i' and nx[0][1].split(':')[0] in ('VIR', 'MUL') \
                            and nx[1] == ('n', 1):
                        q_vir += 1
                    if start and len(nx) >= 1 and nx[0] == ('n', 1) and (len(nx) == 1 or nx[1][0] != 'n'):
                        q_one += 1
                    if prev and prev[0] == 'w' and prev[1] in ('to-so', 'to-sa', 'to-so-de', 'to-sa-de'):
                        after_toso += 1
                    for kk, vv in ln:
                        if kk == 'w' and vv != word: partners.add(vv)
                    for kk, vv in ln[j + 1:j + 4]:
                        if kk == 'n' and vv >= 10: big_num += 1; break
        if hit:
            occ_docs += 1; series[' '.join(d['head'].split()[:2])] += 1
            if any(s in dw for s in seeds if s != word): cooc += 1
    signs = word.split('-')
    derived = []
    # -de (allative) and -pi (locative/instrumental) count for any length; ethnic -jo/-ja
    # forms count only for bases of 3+ signs (2-sign bases + -jo match by chance too often)
    cands = [word + '-de', word + '-pi', word + '-jo-de']
    if len(signs) >= 3:
        cands += [word + '-jo', word + '-jo-i', word + '-te']
        if signs[-1] == 'ja': cands.append('-'.join(signs[:-1] + ['jo']))
        ti = to_i(signs[-1])
        if ti and ti != signs[-1]: cands += ['-'.join(signs[:-1] + [ti, 'jo']), '-'.join(signs[:-1] + [ti, 'ja'])]
    for c in cands:
        if c in vocab: derived.append(c)
    f = {'lb_docs': occ_docs, 'lb_tokens': n_tok, 'derived_forms': derived,
         'docs_with_seed_toponym': cooc, 'seed_cooc_rate': round(cooc / max(1, occ_docs), 2),
         'base_seed_rate': round(base_seed_rate, 2), 'distinct_line_partners': len(partners),
         'entry_then_VIR_or_MUL_1': q_vir, 'entry_then_bare_1': q_one, 'after_to-so/to-sa': after_toso,
         'followed_by_number_ge10': big_num, 'on_last_line_share': round(on_last / max(1, n_tok), 2), 'series': dict(series.most_common(5))}
    P = int(bool(derived)) + int(occ_docs >= 3 and len(partners) >= 3 and cooc / max(1, occ_docs) > base_seed_rate)
    Q = int(q_vir > 0) + int(q_one > 0)
    total_word = n_tok >= 5 and on_last / n_tok >= 0.5
    if total_word: cls = 'other/unclear'; P = Q = 0
    elif P and not Q: cls = 'place'
    elif Q and not P: cls = 'person'
    elif P and Q: cls = 'mixed'
    else: cls = 'other/unclear'
    f['place_points'] = P; f['person_points'] = Q; f['class'] = cls
    return f

def lb_side(words):
    docs = load_lb()
    vocab = Counter(w for d in docs for ln in d['lines'] for k, w in ln if k == 'w')
    seeds = set(w[:-3] for w in vocab if w.endswith('-de') and w.count('-') >= 2) & set(vocab)
    seeds |= {'ko-no-so', 'pa-i-to', 'ku-do-ni-ja'}
    base = sum(1 for d in docs if any(w in seeds for ln in d['lines'] for k, w in ln if k == 'w')) / len(docs)
    res = {w: lb_features(w.lower(), docs, vocab, seeds, base) for w in words}
    # sanity check of the person rule on LB itself: words with data-derived place forms
    # (have an -de allative) vs all words: how often do they show the person pattern?
    def qrate(ws):
        n = 0; k = 0
        for w in ws:
            f = lb_features(w, docs, vocab, set(), 0)
            n += 1; k += int(f['person_points'] > 0)
        return k, n
    seedlist = sorted(s for s in seeds if vocab[s] >= 2)
    rnd = random.sample(sorted(w for w in vocab if vocab[w] >= 2 and w.count('-') >= 2), 150)
    check = {'place_seeds_with_person_pattern': qrate(seedlist), 'random_words_with_person_pattern': qrate(rnd)}
    return res, check, len(seeds), base

# ----------------------------------------------------------------- Linear A side
DOC_CLASS = {'Tablet': 'tablet', 'Lames (short thin tablet)': 'tablet', '3-sided bar': 'tablet',
             '4-sided bar': 'tablet', 'Label': 'tablet',
             'Nodule': 'sealing', 'Roundel': 'sealing', 'Sealing': 'sealing'}
def dclass(sup):
    if sup in DOC_CLASS: return DOC_CLASS[sup]
    if sup.lower().startswith('stone') or sup in ('Metal object', 'Architecture', 'Inked inscription',
                                                  'ivory object', 'Triton'): return 'object'
    if sup.lower() == 'clay vessel': return 'vessel'
    return 'other'

def la_tokens():
    C = json.load(open(os.path.join(D, 'corpus.json')))
    toks = []
    for r in C:
        T = r['tokens']; widx = [i for i, t in enumerate(T) if t['t'] == 'word']; nw = len(widx)
        for o, i in enumerate(widx):
            w = '-'.join(T[i]['s']).translate(SUB)
            nxt = None
            for t in T[i + 1:]:
                if t['t'] == 'div': continue
                nxt = t; break
            nxt2 = None
            if nxt is not None and nxt['t'] == 'logo':
                k = T.index(nxt, i + 1)
                for t in T[k + 1:]:
                    if t['t'] == 'div': continue
                    nxt2 = t; break
            num = nxt if nxt and nxt['t'] == 'num' else (nxt2 if nxt2 and nxt2['t'] == 'num' else None)
            is1 = num is not None and num['v'] == 1 and not num['frac']
            gt1 = num is not None and not is1
            # heading: first word of a tablet-class record and not directly followed by a number
            head = (o == 0 and (nxt is None or nxt['t'] != 'num'))
            toks.append({'w': w, 'rec': r['id'], 'site': r['site'], 'dc': dclass(r['support']),
                         'first': o == 0, 'head': head, 'num1': is1, 'numgt1': gt1,
                         'nonum': num is None, 'logo': nxt is not None and nxt['t'] == 'logo',
                         'relpos': o / (nw - 1) if nw > 1 else 0.0, 'ht': r['site'] == 'Haghia Triada',
                         'nw': nw})
    return toks

FEATS = ['head', 'first', 'num1', 'numgt1', 'nonum', 'logo', 'relpos', 'ht', 'tablet', 'sealing', 'object_or_vessel']
def featvec(tk):
    v = {k: float(tk[k]) for k in ['head', 'first', 'num1', 'numgt1', 'nonum', 'logo', 'relpos', 'ht']}
    v['tablet'] = float(tk['dc'] == 'tablet'); v['sealing'] = float(tk['dc'] == 'sealing')
    v['object_or_vessel'] = float(tk['dc'] in ('object', 'vessel'))
    return v

def group_mean(tlist):
    if not tlist: return {k: None for k in FEATS}
    vs = [featvec(t) for t in tlist]
    return {k: sum(v[k] for v in vs) / len(vs) for k in FEATS}

def attack_A(toks):
    bytype = defaultdict(list)
    for t in toks: bytype[t['w']].append(t)
    m = json.load(open(os.path.join(D, 'la_lb_matches.json')))['matches']
    long = [w for w in m if w.count('-') >= 2]
    short = [w for w in m if w.count('-') == 1]
    lbres, check, nseeds, base = lb_side(long + short)
    out = {'lb_rule_check': {'n_seed_toponyms': nseeds, 'base_doc_rate_with_seed': round(base, 3),
                             'person_pattern_among_seed_toponyms(k,n)': check['place_seeds_with_person_pattern'],
                             'person_pattern_among_random_LB_words(k,n)': check['random_words_with_person_pattern']},
           'lb_classes': {}}
    for w in long + short:
        f = lbres[w]; f['la_tokens'] = len(bytype.get(w, []))
        f['la_contexts'] = [{'rec': t['rec'], 'site': t['site'], 'doc': t['dc'], 'head': t['head'],
                             'num1': t['num1'], 'numgt1': t['numgt1'], 'logo': t['logo'],
                             'relpos': round(t['relpos'], 2)} for t in bytype.get(w, [])]
        out['lb_classes'][w] = f
    # matching bins for controls
    def lenb(w): return min(len(w.split('-')), 5)
    def freqb(n): return 1 if n == 1 else 2 if n == 2 else 3 if n <= 5 else 4
    pool = defaultdict(list)
    for w, ts in bytype.items():
        if '*' in w or '-' not in w: continue
        pool[(lenb(w), freqb(len(ts)))].append(w)
    def run(set_name, words):
        place = [w for w in words if lbres[w]['class'] == 'place' and bytype.get(w)]
        person = [w for w in words if lbres[w]['class'] == 'person' and bytype.get(w)]
        res = {'place_words': place, 'person_words': person}
        gP = group_mean([t for w in place for t in bytype[w]])
        gQ = group_mean([t for w in person for t in bytype[w]])
        res['place_mean'] = {k: None if v is None else round(v, 3) for k, v in gP.items()}
        res['person_mean'] = {k: None if v is None else round(v, 3) for k, v in gQ.items()}
        if not place or not person: return res
        diff = {k: gP[k] - gQ[k] for k in FEATS}
        # control 1: matched random types, 1000 draws (each group replaced by matched randoms)
        nulld = defaultdict(list); nullP = defaultdict(list); nullQ = defaultdict(list)
        for _ in range(NR):
            rp = [random.choice([x for x in pool[(lenb(w), freqb(len(bytype[w])))] if x != w])
                  for w in place]
            rq = [random.choice([x for x in pool[(lenb(w), freqb(len(bytype[w])))] if x != w])
                  for w in person]
            a = group_mean([t for w in rp for t in bytype[w]]); b = group_mean([t for w in rq for t in bytype[w]])
            for k in FEATS:
                nulld[k].append(a[k] - b[k]); nullP[k].append(a[k]); nullQ[k].append(b[k])
        # control 2: permute place/person labels among the anchors
        both = place + person; perm = defaultdict(list)
        for _ in range(NR):
            random.shuffle(both); a = both[:len(place)]; b = both[len(place):]
            A_ = group_mean([t for w in a for t in bytype[w]]); B_ = group_mean([t for w in b for t in bytype[w]])
            for k in FEATS: perm[k].append(A_[k] - B_[k])
        tab = {}
        for k in FEATS:
            o = diff[k]
            p_two = sum(abs(x) >= abs(o) - 1e-12 for x in nulld[k]) / NR
            p_perm = sum(abs(x) >= abs(o) - 1e-12 for x in perm[k]) / NR
            pP = sum(x >= gP[k] - 1e-12 for x in nullP[k]) / NR
            pQ = sum(x >= gQ[k] - 1e-12 for x in nullQ[k]) / NR
            tab[k] = {'place': round(gP[k], 3), 'person': round(gQ[k], 3), 'diff': round(o, 3),
                      'ctrl_diff_mean': round(sum(nulld[k]) / NR, 3), 'p_two_sided_matched': round(p_two, 3),
                      'p_two_sided_label_perm': round(p_perm, 3),
                      'ctrl_place_mean': round(sum(nullP[k]) / NR, 3), 'p_place_ge': round(pP, 3),
                      'ctrl_person_mean': round(sum(nullQ[k]) / NR, 3), 'p_person_ge': round(pQ, 3)}
        res['features'] = tab
        res['n_tokens'] = {'place': sum(len(bytype[w]) for w in place), 'person': sum(len(bytype[w]) for w in person)}
        return res
    out['test_3plus_signs'] = run('3plus', long)
    out['test_all_matches'] = run('all', long + short)
    # power: how big a difference could 3+ sign anchors detect? (proportion test, rough)
    return out

# ----------------------------------------------------------------- Attack B
PRE = ['I', 'SI', 'A', 'KI', 'JA']
SUF = ['ME', 'JA', 'RE', 'TE']

def decompose(types):
    """core -> {form_label: word}.  form_label like 'I+', '+ME', 'A+ME', '0'."""
    fam = defaultdict(dict)
    T = set(types)
    for w in T:
        s = w.split('-')
        opts = [('0', s)]
        for p in PRE:
            if s[0] == p and len(s) - 1 >= 2: opts.append((p + '+', s[1:]))
        for q in SUF:
            if s[-1] == q and len(s) - 1 >= 2: opts.append(('+' + q, s[:-1]))
        for p in PRE:
            for q in SUF:
                if len(s) >= 4 and s[0] == p and s[-1] == q: opts.append((p + '+' + q, s[1:-1]))
        for lab, core in opts:
            fam['-'.join(core)][lab] = w
    # a family needs >=2 distinct attested types
    return {c: f for c, f in fam.items() if len(set(f.values())) >= 2}

def attack_B(toks):
    types = sorted(set(t['w'] for t in toks if '-' in t['w'] and '*' not in t['w']))
    tokc = Counter(t['w'] for t in toks)
    fams = decompose(types)
    # simple affix pairs: core attested bare AND with affix
    out = {'n_types': len(types)}
    T = set(types)
    first = Counter(w.split('-')[0] for w in types); last = Counter(w.split('-')[-1] for w in types)
    sfirst = sorted(first); slast = sorted(last)
    wf = [first[s] for s in sfirst]; wl = [last[s] for s in slast]
    cores = [w for w in types]           # stems = attested types (>=2 signs)
    prod = {}
    for p in PRE:
        obs = sorted(w for w in cores if p + '-' + w in T)
        null = []
        for _ in range(NR):
            k = 0
            for w in cores:
                x = random.choices(sfirst, wf)[0]
                if x + '-' + w in T: k += 1
            null.append(k)
        # the realistic control: the affix sign itself vs other signs of matched initial frequency
        sims = [s for s in sfirst if s != p and abs(first[s] - first[p]) <= max(3, first[p] * 0.35)]
        alt = [sum(1 for w in cores if s + '-' + w in T) for s in sims]
        prod[p + '-'] = {'obs': len(obs), 'examples': [p + '-' + w + ' / ' + w for w in obs][:12],
                         'ctrl_random_sign_mean': round(sum(null) / NR, 2), 'p': round(sum(n >= len(obs) for n in null) / NR, 4),
                         'freq_matched_signs': dict(zip(sims, alt)),
                         'rank_among_matched': 1 + sum(a > len(obs) for a in alt), 'n_matched': len(sims)}
    for q in SUF:
        obs = sorted(w for w in cores if w + '-' + q in T)
        null = []
        for _ in range(NR):
            k = 0
            for w in cores:
                x = random.choices(slast, wl)[0]
                if w + '-' + x in T: k += 1
            null.append(k)
        sims = [s for s in slast if s != q and abs(last[s] - last[q]) <= max(3, last[q] * 0.35)]
        alt = [sum(1 for w in cores if w + '-' + s in T) for s in sims]
        prod['-' + q] = {'obs': len(obs), 'examples': [w + '-' + q + ' / ' + w for w in obs][:12],
                         'ctrl_random_sign_mean': round(sum(null) / NR, 2), 'p': round(sum(n >= len(obs) for n in null) / NR, 4),
                         'freq_matched_signs': dict(zip(sims, alt)),
                         'rank_among_matched': 1 + sum(a > len(obs) for a in alt), 'n_matched': len(sims)}
    # A-/JA- alternation
    aj = sorted(w[2:] for w in types if w.startswith('A-') and 'JA-' + w[2:] in T)
    out['productivity'] = prod
    out['A_JA_alternation'] = aj
    # paradigm table
    table = []
    for c, f in sorted(fams.items(), key=lambda x: -len(x[1])):
        table.append({'core': c, 'forms': {k: v for k, v in sorted(f.items())},
                      'tokens': {v: tokc[v] for v in set(f.values())}})
    out['n_families'] = len(table)
    out['paradigm_table'] = table[:80]
    # ---- feature ties of each affix (token level) -----------------------------
    bytype = defaultdict(list)
    for t in toks: bytype[t['w']].append(t)
    famtypes = set(v for f in fams.values() for v in f.values())
    def carries(w, aff):
        s = w.split('-')
        if aff.endswith('-'): return s[0] == aff[:-1] and len(s) >= 3
        return s[-1] == aff[1:] and len(s) >= 3
    ties = {}
    pool = [t for w in famtypes for t in bytype[w]]
    baseall = group_mean([t for t in toks if '-' in t['w']])
    for aff in [p + '-' for p in PRE] + ['-' + q for q in SUF]:
        # affixed members: types in families where the affix is actually the alternating element
        aw = set()
        for c, f in fams.items():
            for lab, v in f.items():
                if (aff.endswith('-') and lab.startswith(aff[:-1] + '+')) or \
                   (aff.startswith('-') and lab.endswith('+' + aff[1:])):
                    aw.add(v)
        A_ = [t for w in aw for t in bytype[w]]
        if len(A_) < 3: continue
        rest = [t for t in pool if t['w'] not in aw]
        gA = group_mean(A_); gR = group_mean(rest)
        nd = defaultdict(list); allp = A_ + rest
        for _ in range(2000):
            smp = random.sample(allp, len(A_)); g = group_mean(smp)
            for k in FEATS: nd[k].append(g[k])
        row = {'n_tokens': len(A_), 'types': sorted(aw)[:20]}
        for k in FEATS:
            o = gA[k]; mu = sum(nd[k]) / len(nd[k])
            p = sum(abs(x - mu) >= abs(o - mu) - 1e-12 for x in nd[k]) / len(nd[k])
            row[k] = {'affixed': round(o, 3), 'siblings': round(gR[k], 3), 'p': round(p, 4)}
        ties[aff] = row
    out['feature_ties'] = ties
    # type-level version (each word type counted once: mean of its tokens), with two controls:
    #  (a) permutation among all family types; (b) "random sign on the same stems": attested
    #      forms X+core / core+X for signs X of matched positional frequency.
    def tmean(w): return group_mean(bytype[w])
    famlist = sorted(famtypes)
    tv = {w: tmean(w) for w in famlist}
    tl = {}
    for aff in [p + '-' for p in PRE] + ['-' + q for q in SUF]:
        aw = sorted(set(v for c, f in fams.items() for lab, v in f.items()
                        if (aff.endswith('-') and lab.startswith(aff[:-1] + '+')) or
                           (aff.startswith('-') and lab.endswith('+' + aff[1:]))))
        if len(aw) < 3: continue
        g = {k: sum(tv[w][k] for w in aw) / len(aw) for k in FEATS}
        nd = defaultdict(list)
        for _ in range(2000):
            smp = random.sample(famlist, len(aw))
            for k in FEATS: nd[k].append(sum(tv[w][k] for w in smp) / len(smp))
        # random-sign control types
        if aff.endswith('-'):
            sg = aff[:-1]; sims = [s for s in sfirst if s != sg and abs(first[s] - first[sg]) <= max(3, first[sg] * 0.35)]
            ctrl = sorted(set(s + '-' + w for s in sims for w in types if s + '-' + w in T))
        else:
            sg = aff[1:]; sims = [s for s in slast if s != sg and abs(last[s] - last[sg]) <= max(3, last[sg] * 0.35)]
            ctrl = sorted(set(w + '-' + s for s in sims for w in types if w + '-' + s in T))
        cg = group_mean([t for w in ctrl for t in bytype[w]]) if ctrl else None
        cgt = {k: sum(group_mean(bytype[w])[k] for w in ctrl) / len(ctrl) for k in FEATS} if ctrl else None
        row = {'n_types': len(aw), 'types': aw, 'n_ctrl_types': len(ctrl)}
        for k in FEATS:
            mu = sum(nd[k]) / len(nd[k])
            p = sum(abs(x - mu) >= abs(g[k] - mu) - 1e-12 for x in nd[k]) / len(nd[k])
            row[k] = {'affixed': round(g[k], 3), 'perm_mean': round(mu, 3), 'p_perm': round(p, 4),
                      'random_sign_same_stems': None if cgt is None else round(cgt[k], 3)}
        tl[aff] = row
    out['feature_ties_typelevel'] = tl
    # A- vs JA- directly (types of the form A-x and JA-x, all, not only alternating pairs)
    def dsum(ws):
        tt = [t for w in ws for t in bytype[w]]
        return {'n_types': len(ws), 'n_tokens': len(tt), **{k: round(v, 3) for k, v in group_mean(tt).items()}} if tt else {}
    out['A_vs_JA_alternating_pairs'] = {'A-forms': dsum(['A-' + x for x in aj]), 'JA-forms': dsum(['JA-' + x for x in aj]),
                                        'pairs': aj}
    out['corpus_baseline'] = {k: round(v, 3) for k, v in baseall.items()}
    # ---- hold-out prediction ---------------------------------------------------
    ht_types = sorted(set(t['w'] for t in toks if t['ht'] and '-' in t['w'] and '*' not in t['w']))
    other_types = set(t['w'] for t in toks if not t['ht'] and '-' in t['w'] and '*' not in t['w'])
    fam_ht = decompose(ht_types)
    labs_count = Counter(l for f in fam_ht.values() for l in f)
    common = [l for l, n in labs_count.most_common() if l != '0' and l.count('+') == 1 and len(l) > 1][:6]
    preds = set(); basis = {}
    for c, f in fam_ht.items():
        have = [l for l in f if l != '0']
        if len(set(f.values())) < 2: continue
        for l in common:
            if l in f: continue
            form = (l[:-1] + '-' + c) if l.endswith('+') else (c + '-' + l[1:])
            if form in set(ht_types): continue
            preds.add(form); basis[form] = sorted(f.values())
    hits = sorted(p for p in preds if p in other_types)
    # control: same cores, same number of predictions, random sign of matched positional frequency
    first_o = Counter(w.split('-')[0] for w in ht_types); last_o = Counter(w.split('-')[-1] for w in ht_types)
    fs, fw = zip(*first_o.items()); ls, lw = zip(*last_o.items())
    nullh = []
    plist = sorted(preds)
    for _ in range(NR):
        k = 0
        for form in plist:
            # find the core and side used for this prediction
            c = None
            for cc in fam_ht:
                if form.endswith('-' + cc) and form != cc and form.count('-') == cc.count('-') + 1:
                    c = cc; side = 'pre'; break
                if form.startswith(cc + '-') and form.count('-') == cc.count('-') + 1:
                    c = cc; side = 'suf'; break
            if c is None: continue
            x = random.choices(fs, fw)[0] if side == 'pre' else random.choices(ls, lw)[0]
            g = (x + '-' + c) if side == 'pre' else (c + '-' + x)
            if g in other_types and g not in set(ht_types): k += 1
        nullh.append(k)
    out['holdout'] = {'fit': 'Haghia Triada types only', 'n_ht_types': len(ht_types), 'n_families_ht': len(fam_ht),
                      'common_affix_labels_used': common, 'n_predictions': len(preds),
                      'hits_at_other_sites': [{'form': h, 'fitted_from': basis[h],
                                               'where': sorted(set(t['site'] + ' ' + t['rec'] for t in bytype[h]))}
                                              for h in hits],
                      'n_hits': len(hits), 'ctrl_random_sign_mean': round(sum(nullh) / NR, 2),
                      'ctrl_max': max(nullh), 'p': round(sum(n >= len(hits) for n in nullh) / NR, 4),
                      'other_site_types': len(other_types),
                      'predictions_sample': sorted(preds)[:60]}
    # also: families with 2+ affixes, missing a third (all sites) -> forecasts for new finds
    lab_all = Counter(l for f in fams.values() for l in f)
    commonall = [l for l, n in lab_all.most_common() if l != '0' and l.count('+') == 1][:6]
    fc = []
    for c, f in fams.items():
        affs = [l for l in f if l != '0']
        if len(affs) >= 2:
            miss = [l for l in commonall if l not in f]
            fc.append({'core': c, 'attested': sorted(f.values()),
                       'predicted': [(l[:-1] + '-' + c) if l.endswith('+') else (c + '-' + l[1:]) for l in miss]})
    out['forecasts_for_new_finds'] = fc
    return out

def main():
    toks = la_tokens()
    A = attack_A(toks)
    B = attack_B(toks)
    json.dump({'A_anchor_roles': A, 'B_affix_paradigms': B}, open(os.path.join(D, 'attack_anchors.json'), 'w'),
              ensure_ascii=False, indent=1)
    # console summary
    print('LB rule check', A['lb_rule_check'])
    for w, f in A['lb_classes'].items():
        if w.count('-') >= 2 or f['class'] in ('place', 'person', 'mixed'):
            print(f"{w:13s} {f['class']:14s} P{f['place_points']} Q{f['person_points']} lbdocs {f['lb_docs']:3d} "
                  f"der {f['derived_forms']} seed {f['seed_cooc_rate']} partners {f['distinct_line_partners']} "
                  f"VIR/MUL1 {f['entry_then_VIR_or_MUL_1']} bare1 {f['entry_then_bare_1']} toso {f['after_to-so/to-sa']} LA {f['la_tokens']}")
    for key in ['test_3plus_signs', 'test_all_matches']:
        r = A[key]; print('==', key, 'place', r['place_words'], 'person', r['person_words'], r.get('n_tokens'))
        for k, v in r.get('features', {}).items(): print('  ', k, v)
    print('== B productivity')
    for k, v in B['productivity'].items():
        print('  ', k, v['obs'], 'ctrl', v['ctrl_random_sign_mean'], 'p', v['p'], 'rank', v['rank_among_matched'], '/', v['n_matched'] + 1, v['examples'][:6])
    print('  A/JA', B['A_JA_alternation'], 'families', B['n_families'])
    for k, v in B['feature_ties'].items():
        print('  ', k, v['n_tokens'], {f: (v[f]['affixed'], v[f]['siblings'], v[f]['p']) for f in FEATS if v[f]['p'] < 0.1})
    print('== type level')
    for k, v in B['feature_ties_typelevel'].items():
        print('  ', k, v['n_types'], 'ctrl types', v['n_ctrl_types'],
              {f: (v[f]['affixed'], v[f]['perm_mean'], v[f]['random_sign_same_stems'], v[f]['p_perm']) for f in FEATS if v[f]['p_perm'] < 0.1})
    print('  A vs JA', B['A_vs_JA_alternating_pairs'])
    h = B['holdout']; print('== holdout', {k: h[k] for k in ['n_predictions', 'n_hits', 'ctrl_random_sign_mean', 'ctrl_max', 'p', 'common_affix_labels_used']})
    for x in h['hits_at_other_sites']: print('   ', x)

if __name__ == '__main__':
    main()
