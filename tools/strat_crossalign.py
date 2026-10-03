"""Structural alignment of three accounting-type sign systems (Proto-Elamite, Linear A, Indus).

For every frequent sign in each system a language-independent behaviour vector is computed (slot position,
numeral adjacency, numeral type, numeral variety, prevalence, repetition, context entropy, site spread...).
Roles are known for some Proto-Elamite (PE) and Linear A (LinA) signs:
  OPENER (PE M157; LinA tablet-heading words), COMMODITY-MEASURED (PE M297 M002 M036 M243; LinA GRA VIN OLE CYP
  FIC OLIV), COMMODITY-COUNTED (PE M263 M346 M264 M003; LinA VIR CAP OVIS SUS BOS), TOTAL (LinA KU-RO, KI-RO,
  PO-TO-KU-RO; PE has no dedicated total word), QUALIFIER/PREFIX (PE M387 M370 M124; LinA none),
  NAME-LIKE (PE medial-biased signs of multi-sign entries; LinA frequent list-entry words with quantities),
  NUMERAL (numeral tokens of each system).
A regularised multinomial logistic regression is trained on PE+LinA vectors and cross-validated
PE -> LinA and LinA -> PE (the control that behaviour carries role across systems), then applied to Indus signs
with >= 20 tokens. Controls: Indus signs shuffled within text; label-permuted training.
Two framings: F1 document = tablet (PE/LinA) ; F2 document = entry/line. Indus text = document = line in both.
Output: data/derived/strat_crossalign.txt and .json. Run from the repo root.
"""
import json, math, random, collections, sys
import numpy as np

random.seed(7); np.random.seed(7)
OUT_TXT = 'data/derived/strat_crossalign.txt'; OUT_JSON = 'data/derived/strat_crossalign.json'
log_lines = []
def log(s=''):
    print(s); log_lines.append(s)

# ----------------------------------------------------------------------------- token streams
# Each system -> list of documents; a document = list of lines; a line = list of tokens.
# token = ('S', type) for a sign, ('N', cls, value) for a numeral (cls in small/large/other), ('X',) unreadable.
# item_side: +1 if the counted item stands right BEFORE the numeral in token order (PE, LinA), -1 if right AFTER (Indus).

def pe_stream():
    pe = json.load(open('other-scripts/proto-elamite/data/pe_corpus.json'))
    CAP = {'N39B', 'N30C', 'N24', 'N30D', 'N39C'}
    FRAC = {'N08', 'N08A', 'N8B', 'N02'}
    LARGE = {'N14', 'N34', 'N45', 'N23', 'N51', 'N48', 'N46', 'N51G'}
    docs = []; sites = []
    for t in pe:
        lines = []
        for l in t['lines']:
            toks = []
            for s in l['signs']:
                if s == 'x' or s.startswith('x'):
                    toks.append(('X',))
                else:
                    base = s.split('~')[0] if not s.startswith('|') else s
                    toks.append(('S', base))
            if l['numerals']:
                codes = [c.split('@')[0] for _, c in l['numerals']]
                if any(c in CAP for c in codes): cls = 'other'; val = None
                elif any(c in FRAC for c in codes): cls = 'other'; val = None
                elif any(c in LARGE for c in codes): cls = 'large'; val = None
                else:
                    cls = 'small'; val = sum((n or 0) for n, c in l['numerals'] if c.split('@')[0] == 'N01') or None
                toks.append(('N', cls, val))
            if toks: lines.append(toks)
        if lines:
            docs.append(lines); sites.append(t['provenience'].split(' (')[0])
    return docs, sites, +1

def lina_stream():
    la = json.load(open('other-scripts/linear-a/data/corpus.json'))
    docs = []; sites = []; supports = []
    for r in la:
        if r['support'] not in ('Tablet', 'Nodule', 'Roundel', 'Sealing', 'Lames (short thin tablet)', 'Label'):
            continue  # administrative objects only
        lines = [[]]
        for tk in r['tokens']:
            if tk['t'] == 'nl':
                lines.append([]); continue
            if tk['t'] == 'div': continue
            if tk['t'] == 'word':
                lines[-1].append(('S', 'W:' + '-'.join(tk['s'])))
            elif tk['t'] == 'logo':
                lines[-1].append(('S', 'L:' + tk['v'].split('+')[0]))
            elif tk['t'] == 'num':
                v = tk.get('v'); fr = tk.get('frac') or []
                if fr: cls = 'other'; val = None
                elif v is not None and v >= 10: cls = 'large'; val = None
                else: cls = 'small'; val = v
                lines[-1].append(('N', cls, val))
            else:
                lines[-1].append(('X',))
        lines = [l for l in lines if l]
        if lines:
            docs.append(lines); sites.append(r['site']); supports.append(r['support'])
    lina_stream.supports = supports
    return docs, sites, +1

SHORT = {1: 1, 3: 3, 4: 4, 5: 5, 16: 6, 17: 7, 18: 8}   # W1 is a marker (S234) and is NOT treated as a numeral below
TALL = {31: 1, 32: 2, 33: 3, 34: 4}
BIG = {55: 12, 56: 24}
IND_NUM = set(SHORT) - {1} | set(TALL) | set(BIG)

def indus_stream(key='seq_raw', shuffle=False, rnd=None):
    C = json.load(open('data/derived/merged-corpus-canonical.json'))
    docs = []; sites = []; seen = set()
    for r in C:
        s = r.get(key)
        if not s or len(s) < 2: continue
        k = (r['site'], tuple(s))
        if k in seen: continue
        seen.add(k)
        s = list(s)
        if shuffle: rnd.shuffle(s)
        toks = []
        for x in s:
            if x in IND_NUM:
                if x in TALL: toks.append(('N', 'other', TALL[x]))
                elif x in BIG: toks.append(('N', 'large', BIG[x]))
                else: toks.append(('N', 'small', SHORT[x]))
            else:
                toks.append(('S', x))
        docs.append([toks]); sites.append(r['site'])
    return docs, sites, -1

# ----------------------------------------------------------------------------- behaviour vectors
FEATS = ['doc_first', 'doc_last', 'doc_relpos', 'line_first', 'line_last', 'line_alone', 'adj_num', 'takes_num',
         'num_small', 'num_large', 'num_other', 'num_variety', 'prevalence', 'repeat', 'ctx_entropy', 'offsite',
         'logfreq', 'doc_len', 'is_numeral']

def entropy_norm(counter):
    n = sum(counter.values())
    if n <= 1 or len(counter) <= 1: return 0.0
    h = -sum(c / n * math.log(c / n) for c in counter.values())
    return h / math.log(n)

def vectors(docs, sites, item_side, framing, min_tok):
    """framing 'tablet': document = docs entry ; 'line': every line is its own document."""
    if framing == 'line':
        docs2 = []; sites2 = []
        for d, st in zip(docs, sites):
            for l in d: docs2.append([l]); sites2.append(st)
        docs, sites = docs2, sites2
    site_cnt = collections.Counter(sites); top_site = site_cnt.most_common(1)[0][0]
    ndocs = len(docs)
    tok = collections.Counter(); docs_with = collections.Counter(); docs_rep = collections.Counter()
    dfirst = collections.Counter(); dlast = collections.Counter(); relpos = collections.defaultdict(list)
    lfirst = collections.Counter(); llast = collections.Counter(); lalone = collections.Counter()
    adjn = collections.Counter(); takes = collections.Counter(); ncls = collections.defaultdict(collections.Counter)
    nval = collections.defaultdict(collections.Counter); ctx = collections.defaultdict(collections.Counter)
    offsite = collections.Counter(); dlen = collections.defaultdict(list)
    doclens = []
    def key(t):
        if t[0] == 'S': return ('S', t[1])
        if t[0] == 'N': return ('N', t[1])
        return None
    for d, st in zip(docs, sites):
        flat = [t for l in d for t in l]
        signs_in_doc = [i for i, t in enumerate(flat) if t[0] != 'X']
        nsign = sum(1 for t in flat if t[0] == 'S'); doclens.append(nsign)
        if not signs_in_doc: continue
        cnt = collections.Counter()
        for l in d:
            idx = [i for i, t in enumerate(l) if t[0] != 'X']
            nonnum = [i for i in idx if l[i][0] == 'S']
            for j, i in enumerate(idx):
                t = l[i]; k = key(t)
                if k is None: continue
                cnt[k] += 1
                if j == 0: lfirst[k] += 1
                if j == len(idx) - 1: llast[k] += 1
                if len(nonnum) == 1 and t[0] == 'S': lalone[k] += 1
                nb = []
                if j > 0: nb.append(l[idx[j - 1]])
                if j < len(idx) - 1: nb.append(l[idx[j + 1]])
                for u in nb:
                    if u[0] == 'N' and k[0] == 'S':
                        adjn[k] += 1; ncls[k][u[1]] += 1
                        if u[2] is not None: nval[k][u[2]] += 1
                    if u[0] == 'N' and k[0] == 'N': adjn[k] += 1
                    kk = key(u)
                    if kk: ctx[k][kk] += 1
                # item side: the sign touching the numeral on the item side
                jj = j + item_side
                if 0 <= jj < len(idx) and l[idx[jj]][0] == 'N' and t[0] == 'S': takes[k] += 1
        # document-level
        fk = key(flat[signs_in_doc[0]]); lk = key(flat[signs_in_doc[-1]])
        if fk: dfirst[fk] += 1
        if lk: dlast[lk] += 1
        n = len(signs_in_doc)
        for r, i in enumerate(signs_in_doc):
            k = key(flat[i])
            if k: relpos[k].append(r / (n - 1) if n > 1 else 0.5)
        for k, c in cnt.items():
            tok[k] += c; docs_with[k] += 1; dlen[k].append(nsign)
            if c >= 2: docs_rep[k] += 1
            if st != top_site: offsite[k] += c
    mean_len = np.mean(doclens) if doclens else 1
    maxtok = max(tok.values())
    vec = {}
    for k, n in tok.items():
        if n < min_tok: continue
        a = adjn[k]
        v = [dfirst[k] / n, dlast[k] / n, float(np.mean(relpos[k])) if relpos[k] else 0.5,
             lfirst[k] / n, llast[k] / n, lalone[k] / n, a / n, takes[k] / n,
             ncls[k]['small'] / a if a else 0, ncls[k]['large'] / a if a else 0, ncls[k]['other'] / a if a else 0,
             entropy_norm(nval[k]), docs_with[k] / ndocs, docs_rep[k] / docs_with[k], entropy_norm(ctx[k]),
             offsite[k] / n, math.log(n) / math.log(maxtok), math.log(np.mean(dlen[k]) / mean_len),
             1.0 if k[0] == 'N' else 0.0]
        vec[k] = (np.array(v), n)
    return vec

# ----------------------------------------------------------------------------- role labels
ROLES = ['OPENER', 'COMMODITY-MEASURED', 'COMMODITY-COUNTED', 'TOTAL', 'QUALIFIER', 'NAME-LIKE', 'NUMERAL']

def pe_labels(vec):
    lab = {}
    for m in ['M297', 'M002', 'M036', 'M243']: lab[('S', m)] = 'COMMODITY-MEASURED'
    for m in ['M263', 'M346', 'M264', 'M003']: lab[('S', m)] = 'COMMODITY-COUNTED'
    lab[('S', 'M157')] = 'OPENER'; lab[('S', 'M327')] = 'OPENER'
    for m in ['M387', 'M370', 'M124']: lab[('S', m)] = 'QUALIFIER'
    # NAME-LIKE: medial-biased signs (neither line-first nor line-last in >= 60% of tokens), data-only
    for k, (v, n) in vec.items():
        if k[0] == 'S' and k not in lab and n >= 30:
            fl, ll = v[FEATS.index('line_first')], v[FEATS.index('line_last')]
            if 1 - max(fl, ll) >= 0.55 and fl < 0.3 and ll < 0.3: lab[k] = 'NAME-LIKE'
    for k in vec:
        if k[0] == 'N': lab[k] = 'NUMERAL'
    return {k: r for k, r in lab.items() if k in vec}

def lina_labels(vec, docs, supports):
    lab = {}
    for g in ['GRA', 'VIN', 'OLE', 'CYP', 'FIC', 'OLIV']: lab[('S', 'L:' + g)] = 'COMMODITY-MEASURED'
    for g in ['VIR', 'CAP', 'OVIS', 'SUS', 'BOS']: lab[('S', 'L:' + g)] = 'COMMODITY-COUNTED'
    for w in ['KU-RO', 'KI-RO', 'PO-TO-KU-RO']: lab[('S', 'W:' + w)] = 'TOTAL'
    # headers: words first on the tablet with no number following, >= 70% of their tokens, >= 3 tablets (data-only)
    first = collections.Counter(); tot = collections.Counter()
    for d, sup in zip(docs, supports):
        if sup != 'Tablet' or len(d) < 2: continue
        flat = [t for l in d for t in l]
        for i, t in enumerate(flat):
            if t[0] == 'S' and t[1].startswith('W:'):
                tot[('S', t[1])] += 1
                if i == 0 and not (len(flat) > 1 and flat[1][0] == 'N'): first[('S', t[1])] += 1
    for k in tot:
        if first[k] >= 3 and first[k] / tot[k] >= 0.7 and k not in lab: lab[k] = 'OPENER'
    # name-like: words with >= 4 tokens that take a quantity, not transaction words, not libation words
    for k, (v, n) in vec.items():
        if k[0] == 'S' and k[1].startswith('W:') and '-' in k[1] and k not in lab and n >= 4 and v[FEATS.index('takes_num')] >= 0.5:
            lab[k] = 'NAME-LIKE'
    for k in vec:
        if k[0] == 'N': lab[k] = 'NUMERAL'
    return {k: r for k, r in lab.items() if k in vec}

# ----------------------------------------------------------------------------- classifier
class Clf:
    def __init__(self, C=1.0, iters=3000, lr=0.1):
        self.C = C; self.iters = iters; self.lr = lr
    def fit(self, X, y, roles):
        self.roles = roles; self.mu = X.mean(0); self.sd = X.std(0) + 1e-6
        Z = (X - self.mu) / self.sd; Z = np.hstack([Z, np.ones((len(Z), 1))])
        Y = np.zeros((len(y), len(roles)))
        for i, r in enumerate(y): Y[i, roles.index(r)] = 1
        # class weights balance the roles
        w = 1.0 / (Y.sum(0) + 1e-9); sw = (Y * w).sum(1); sw = sw / sw.mean()
        W = np.zeros((Z.shape[1], len(roles)))
        for _ in range(self.iters):
            P = self.predict_Z(Z, W)
            G = Z.T @ ((P - Y) * sw[:, None]) / len(Z) + W / (self.C * len(Z))
            G[-1] -= W[-1] / (self.C * len(Z))
            W -= self.lr * G
        self.W = W; return self
    @staticmethod
    def predict_Z(Z, W):
        L = Z @ W; L -= L.max(1, keepdims=True); E = np.exp(L); return E / E.sum(1, keepdims=True)
    def proba(self, X):
        Z = (X - self.mu) / self.sd; Z = np.hstack([Z, np.ones((len(Z), 1))])
        return self.predict_Z(Z, self.W)

def evaluate(train, test, roles, C=1.0):
    Xtr = np.array([v for v, _ in train]); ytr = [r for _, r in train]
    Xte = np.array([v for v, _ in test]); yte = [r for _, r in test]
    clf = Clf(C=C).fit(Xtr, ytr, roles)
    P = clf.proba(Xte); pred = [roles[i] for i in P.argmax(1)]
    acc = np.mean([p == t for p, t in zip(pred, yte)])
    # accuracy restricted to roles present in training
    tr_roles = set(ytr); mask = [t in tr_roles for t in yte]
    acc_r = np.mean([p == t for p, t, m in zip(pred, yte, mask) if m]) if any(mask) else float('nan')
    return acc, acc_r, pred, P, clf

# ----------------------------------------------------------------------------- main
def run(framing, key='seq_raw'):
    log(f'\n######## framing {framing}, Indus column {key} ########')
    pe_docs, pe_sites, pe_side = pe_stream(); la_docs, la_sites, la_side = lina_stream()
    pe_vec = vectors(pe_docs, pe_sites, pe_side, framing, 20)
    la_vec = vectors(la_docs, la_sites, la_side, framing, 4)
    pe_lab = pe_labels(pe_vec); la_lab = lina_labels(la_vec, la_docs, lina_stream.supports)
    log(f'PE: {len(pe_docs)} tablets, {len(pe_vec)} sign types >= 20 tokens, {len(pe_lab)} labelled: ' +
        str(collections.Counter(pe_lab.values())))
    log(f'LinA: {len(la_docs)} administrative objects, {len(la_vec)} types >= 4 tokens, {len(la_lab)} labelled: ' +
        str(collections.Counter(la_lab.values())))
    log('LinA labelled: ' + ', '.join(f'{k[1] if k[0]=="S" else "NUM:"+k[1]}={r}' for k, r in sorted(la_lab.items(), key=lambda x: x[1])))
    log('PE NAME-LIKE: ' + ' '.join(k[1] for k, r in pe_lab.items() if r == 'NAME-LIKE'))
    pe_set = [(pe_vec[k][0], r) for k, r in pe_lab.items()]
    la_set = [(la_vec[k][0], r) for k, r in la_lab.items()]
    roles = ROLES
    res = {'framing': framing, 'key': key}
    # --- cross-validation PE <-> LinA
    for name, tr, te, te_lab in [('PE->LinA', pe_set, la_set, la_lab), ('LinA->PE', la_set, pe_set, pe_lab)]:
        acc, acc_r, pred, P, _ = evaluate(tr, te, roles)
        conf = collections.Counter((t, p) for (_, t), p in zip(te, pred))
        log(f'{name}: accuracy {acc:.2f} (all {len(te)}), {acc_r:.2f} on roles present in training')
        cm = collections.defaultdict(collections.Counter)
        for (t, p), c in conf.items(): cm[t][p] += c
        for t in roles:
            if cm[t]: log(f'    true {t:20s} -> ' + ', '.join(f'{p} {c}' for p, c in cm[t].most_common()))
        # label permutation control
        rnd = random.Random(3); accs = []
        for _ in range(100):
            ys = [r for _, r in tr]; rnd.shuffle(ys)
            a, _, _, _, _ = evaluate([(v, y) for (v, _), y in zip(tr, ys)], te, roles); accs.append(a)
        log(f'    label-permuted training: mean {np.mean(accs):.2f}, 95th pct {np.percentile(accs, 95):.2f}, '
            f'p(perm >= obs) = {np.mean([a >= acc for a in accs]):.2f}')
        res[name] = {'acc': acc, 'acc_present_roles': acc_r, 'perm_mean': float(np.mean(accs)),
                     'perm_p': float(np.mean([a >= acc for a in accs])),
                     'pred': {(k[1] if k[0] == 'S' else 'NUM:' + k[1]): [t, p] for (k, t), p in zip(te_lab.items(), pred)}}
    # --- without the NUMERAL class and without position-defined classes (robustness)
    for drop in [('NUMERAL',), ('NUMERAL', 'NAME-LIKE', 'OPENER')]:
        for name, tr, te in [('PE->LinA', pe_set, la_set), ('LinA->PE', la_set, pe_set)]:
            tr2 = [x for x in tr if x[1] not in drop]; te2 = [x for x in te if x[1] not in drop]
            acc, acc_r, _, _, _ = evaluate(tr2, te2, roles)
            rnd = random.Random(5); accs = []
            for _ in range(100):
                ys = [r for _, r in tr2]; rnd.shuffle(ys)
                a, _, _, _, _ = evaluate([(v, y) for (v, _), y in zip(tr2, ys)], te2, roles); accs.append(a)
            log(f'  dropping {drop}: {name} acc {acc:.2f} on {len(te2)} (present-roles {acc_r:.2f}); perm mean {np.mean(accs):.2f}, p {np.mean([a >= acc for a in accs]):.2f}')
            res[f'{name} drop {"+".join(drop)}'] = {'acc': acc, 'acc_present_roles': acc_r, 'n': len(te2), 'perm_mean': float(np.mean(accs))}
    # --- pooled leave-one-out
    pool = pe_set + la_set; hits = 0
    for i in range(len(pool)):
        tr = pool[:i] + pool[i + 1:]
        a, _, _, _, _ = evaluate(tr, [pool[i]], roles); hits += a
    log(f'pooled leave-one-out accuracy {hits / len(pool):.2f} ({len(pool)} items)')
    res['pooled_loo'] = hits / len(pool)
    # --- apply to Indus
    clf = Clf().fit(np.array([v for v, _ in pool]), [r for _, r in pool], roles)
    ind_docs, ind_sites, side = indus_stream(key)
    ind_vec = vectors(ind_docs, ind_sites, side, framing, 20)
    keys = sorted(ind_vec, key=lambda k: -ind_vec[k][1])
    P = clf.proba(np.array([ind_vec[k][0] for k in keys]))
    table = []
    for k, p in zip(keys, P):
        i = int(p.argmax()); table.append({'sign': (f'W{k[1]}' if k[0] == 'S' else 'NUM:' + k[1]), 'n': ind_vec[k][1],
                                           'role': roles[i], 'conf': float(p[i]),
                                           'probs': {r: round(float(x), 3) for r, x in zip(roles, p)}})
    res['indus'] = table
    nconf = sum(1 for t in table if t['conf'] > 0.7)
    log(f'Indus ({len(ind_docs)} texts): {len(table)} signs >= 20 tokens; {nconf} ({nconf / len(table):.0%}) land in a role with conf > 0.7')
    log('role counts: ' + str(collections.Counter(t['role'] for t in table)))
    log('role counts (conf > 0.7): ' + str(collections.Counter(t['role'] for t in table if t['conf'] > 0.7)))
    log(f'{"sign":10s} {"n":>5s} {"role":20s} {"conf":>5s}  probs(OPEN/MEAS/COUNT/TOTAL/QUAL/NAME/NUM)')
    for t in table:
        pr = t['probs']; log(f'{t["sign"]:10s} {t["n"]:5d} {t["role"]:20s} {t["conf"]:5.2f}  ' +
                             ' '.join(f'{pr[r]:.2f}' for r in roles))
    # key questions
    log('\nKey signs:')
    for grp, ss in [('closers', [740, 520, 151, 156, 527, 226, 617, 154, 158, 236, 700]), ('tree', [390, 405, 407]),
                    ('openers', [817, 861, 820, 920, 692]), ('markers', [2, 60, 1]), ('suffix', [400, 90]),
                    ('pre-jar titles', [176, 100, 760, 923, 590]), ('fish', [220, 240, 235, 233, 231]),
                    ('fixed-number terms', [575, 585, 632, 877])]:
        rows = [t for t in table if t['sign'] in {f'W{s}' for s in ss}]
        log(f'  {grp}: ' + '; '.join(f'{t["sign"]} {t["role"]} {t["conf"]:.2f}' for t in rows))
    # --- control (a): Indus signs shuffled within text
    rnd = random.Random(11); shuf_roles = collections.Counter(); shuf_conf = []; agree = []
    for rep in range(20):
        sd, ss, side = indus_stream(key, shuffle=True, rnd=rnd)
        sv = vectors(sd, ss, side, framing, 20)
        ks = [k for k in keys if k in sv]
        Ps = clf.proba(np.array([sv[k][0] for k in ks]))
        real = {t['sign']: t['role'] for t in table}
        for k, p in zip(ks, Ps):
            r = roles[int(p.argmax())]; shuf_roles[r] += 1; shuf_conf.append(float(p.max()))
            agree.append(real[(f'W{k[1]}' if k[0] == 'S' else 'NUM:' + k[1])] == r)
    log(f'\nControl (a) shuffled-within-text Indus (20 reps): roles {dict(shuf_roles)}; '
        f'conf > 0.7 share {np.mean([c > 0.7 for c in shuf_conf]):.0%}; same role as real {np.mean(agree):.0%}')
    res['shuffle_control'] = {'roles': dict(shuf_roles), 'conf_gt_07': float(np.mean([c > 0.7 for c in shuf_conf])),
                              'agree_with_real': float(np.mean(agree))}
    # --- control (b): label-permuted training applied to Indus
    rnd = random.Random(13); perm_tables = []
    for rep in range(50):
        ys = [r for _, r in pool]; rnd.shuffle(ys)
        c2 = Clf().fit(np.array([v for v, _ in pool]), ys, roles)
        P2 = c2.proba(np.array([ind_vec[k][0] for k in keys]))
        perm_tables.append([roles[int(p.argmax())] for p in P2])
    stab = []
    for j, t in enumerate(table):
        stab.append(np.mean([pt[j] == t['role'] for pt in perm_tables]))
    log(f'Control (b) label-permuted training (50 reps): a sign keeps its real role in {np.mean(stab):.0%} of permuted runs '
        f'(chance ~ {1 / len(roles):.0%}); conf > 0.7 share {np.mean([max(collections.Counter(pt).values()) for pt in perm_tables]) / len(table):.0%} of signs in the modal role')
    res['perm_control_keep_role'] = float(np.mean(stab))
    return res

if __name__ == '__main__':
    results = []
    for framing in ['tablet', 'line']:
        results.append(run(framing, 'seq_raw'))
    results.append(run('line', 'seq_all'))
    json.dump(results, open(OUT_JSON, 'w'), indent=1, default=str)
    open(OUT_TXT, 'w').write('\n'.join(log_lines) + '\n')
    print('written', OUT_TXT, OUT_JSON)
