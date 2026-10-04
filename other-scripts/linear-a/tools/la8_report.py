#!/usr/bin/env python3
"""LA-8 report: describe the best evolved machine per corpus and compare them.
Usage: la8_report.py [TAG]   (reads data/la8/gp_<C>_<TAG>_s*.json)"""
import glob, json, math, os, sys
from collections import Counter, defaultdict
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import la8_gp as G
from la8_truth import ari, truth_genome

TAG = sys.argv[1] if len(sys.argv) > 1 else 'main'
CORPORA = ['PFLAT', 'PREC', 'LA', 'LB', 'PE', 'LAS', 'LB2', 'PE2']


def token_classes(D, g):
    return [int(g['tc'][t]) if t >= 0 else int(g['kc'][k]) for t, k in zip(D.ttype, D.tkey)]


def sccs(adj, n):
    idx, low, on, st, res, c = {}, {}, set(), [], [], [0]
    sys.setrecursionlimit(10000)
    def strong(v):
        idx[v] = low[v] = c[0]; c[0] += 1; st.append(v); on.add(v)
        for w in adj[v]:
            if w not in idx: strong(w); low[v] = min(low[v], low[w])
            elif w in on: low[v] = min(low[v], idx[w])
        if low[v] == idx[v]:
            comp = []
            while True:
                w = st.pop(); on.discard(w); comp.append(w)
                if w == v: break
            res.append(comp)
    for v in range(n):
        if v not in idx: strong(v)
    return res


def describe(corpus, D, g, f):
    out, tr, em = D.evaluate(g, mode=1)
    bits, o = D.evaluate(g)
    K = g['K']; S = int(sum(g['nst']))
    tcl = token_classes(D, g)
    toks = [t for d in D.docs for t in d['toks']]
    cls_tok = defaultdict(Counter); cls_kind = defaultdict(Counter); novel = Counter()
    for t, c in zip(toks, tcl):
        cls_tok[c][t[0]] += 1; cls_kind[c][t[1]] += 1
        if t[0] not in D.lexid: novel[c] += 1
    openc = sorted(set(g['kc'].tolist()))
    classes = []
    for c in range(K):
        n = sum(cls_tok[c].values())
        classes.append({'class': c, 'tokens': n, 'open': c in openc, 'members': int((g['tc'] == c).sum()),
                        'novel_tokens': novel[c], 'states': int(g['nst'][c]),
                        'kinds': dict(cls_kind[c].most_common()), 'top': [w for w, _ in cls_tok[c].most_common(12)],
                        'keys': [D.keys[k] for k in range(D.NK) if g['kc'][k] == c][:12]})
    # state graph
    first = np.cumsum([0] + list(g['nst']))[:-1]
    soc = [c for c in range(K) for _ in range(g['nst'][c])]
    P = tr  # rows states+START, cols states+END
    adj = {s: [s2 for s2 in range(S) if P[s, s2] >= 0.05] for s in range(S)}
    comps = sccs(adj, S)
    cyc = [cp for cp in comps if len(cp) > 1 or (cp[0] in adj[cp[0]])]
    edges05 = sum(len(v) for v in adj.values()) + int((P[S, :S] >= 0.05).sum()) + int((P[:S, S] >= 0.05).sum())
    # number attachment
    numc = [c for c in range(K) if cls_tok[c]['NUM'] > 0]
    prev, nxt = Counter(), Counter()
    j = 0
    for d in D.docs:
        n = len(d['toks']); seq = tcl[j:j + n]; ts = [t[0] for t in d['toks']]
        for i in range(n):
            if ts[i] == 'NUM':
                prev[('C%d' % seq[i-1]) if i > 0 else '<s>'] += 1; nxt[('C%d' % seq[i+1]) if i + 1 < n else '</s>'] += 1
        j += n
    # P(next is NUM | class)
    pnum = {}
    j = 0; cnt = Counter(); hit = Counter()
    for d in D.docs:
        n = len(d['toks']); seq = tcl[j:j + n]; ts = [t[0] for t in d['toks']]
        for i in range(n - 1):
            cnt[seq[i]] += 1; hit[seq[i]] += ts[i+1] == 'NUM'
        j += n
    for c in range(K): pnum[c] = round(hit[c] / cnt[c], 2) if cnt[c] else None
    T = D.T
    rep = {'corpus': corpus, 'fitness': round(f, 1), 'heldout_bits': round(bits, 1), 'DL': round(f - bits, 1),
           'bits_per_token': round(bits / T, 3), 'emit_bits_per_token': round(o[0] / T, 3), 'trans_bits_per_token': round(o[1] / T, 3),
           'K': K, 'states': S, 'open_classes': len(openc), 'closed_classes': K - len(openc),
           'edges_p05': edges05, 'cycles': [[('C%d' % soc[s]) for s in cp] for cp in cyc], 'largest_cycle': max([len(cp) for cp in cyc] or [0]),
           'num_prev': prev.most_common(5), 'num_next': nxt.most_common(5), 'p_next_is_num': pnum,
           'novel_token_share': round(sum(novel.values()) / T, 3), 'classes': sorted(classes, key=lambda x: -x['tokens'])}
    return rep


def main():
    reps, best = {}, {}
    truths = json.load(open(os.path.join(G.OUT, 'planted_truth.json')))
    for C in CORPORA:
        fs = sorted(glob.glob(os.path.join(G.OUT, 'gp_%s_%s_s*.json' % (C, TAG))))
        if not fs: continue
        D = G.Data(json.load(open(os.path.join(G.OUT, 'corpus_%s.json' % C))))
        runs = [json.load(open(p)) for p in fs]
        bests = [(r['pop'][0]['f'], G.from_json(r['pop'][0]['g']), r['seed'], r['gen'], r['restarts']) for r in runs]
        bests.sort(key=lambda x: x[0])
        f, g = bests[0][0], bests[0][1]
        rep = describe(C, D, g, f)
        rep['runs'] = [{'seed': s, 'gen': gn, 'fitness': round(ff, 1), 'K': gg['K'], 'S': int(sum(gg['nst'])), 'open': len(set(gg['kc'].tolist())), 'restarts': rs} for ff, gg, s, gn, rs in bests]
        w = D.lexcount
        if len(bests) > 1:
            pairs = [(a, b) for a in range(len(bests)) for b in range(a + 1, len(bests))]
            rep['seed_ARI'] = [round(ari(np.repeat(bests[a][1]['tc'], w).tolist(), np.repeat(bests[b][1]['tc'], w).tolist()), 3) for a, b in pairs]
        if C in truths:
            tg, labs = truth_genome(D, truths[C])
            tf = G.fitness(D, tg)[0]
            rep['truth_fitness'] = round(tf, 1)
            rep['truth_ARI'] = round(ari(np.repeat(g['tc'], w).tolist(), np.repeat(tg['tc'], w).tolist()), 3)
            rep['truth_K'] = tg['K']; rep['truth_open'] = len(set(tg['kc'].tolist()))
        reps[C] = rep
        json.dump({'corpus': C, 'genome': G.to_json(g)}, open(os.path.join(G.OUT, 'best_%s_%s.json' % (C, TAG)), 'w'))
    json.dump(reps, open(os.path.join(G.OUT, 'report_%s.json' % TAG), 'w'), indent=1, ensure_ascii=False)
    for C, r in reps.items():
        print('== %s fit %.0f bits/tok %.3f (emit %.3f trans %.3f) DL %.0f K=%d S=%d open=%d closed=%d edges=%d largest_cycle=%d novel=%.3f' % (
            C, r['fitness'], r['bits_per_token'], r['emit_bits_per_token'], r['trans_bits_per_token'], r['DL'], r['K'], r['states'], r['open_classes'], r['closed_classes'], r['edges_p05'], r['largest_cycle'], r['novel_token_share']))
        print('   runs', r['runs'], 'seedARI', r.get('seed_ARI'), 'truth', r.get('truth_fitness'), r.get('truth_ARI'), r.get('truth_K'), r.get('truth_open'))
        print('   NUM prev', r['num_prev'], 'next', r['num_next'])
        for c in r['classes']:
            print('   C%-2d %s tok=%d mem=%d nov=%d st=%d pNUM=%s kinds=%s top=%s keys=%s' % (c['class'], 'OPEN ' if c['open'] else 'closd', c['tokens'], c['members'], c['novel_tokens'], c['states'], r['p_next_is_num'][c['class']], c['kinds'], ' '.join(c['top'][:8]), ','.join(c['keys'][:6])))


if __name__ == '__main__':
    main()
