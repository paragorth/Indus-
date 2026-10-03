"""Loop 59 cycle 2: the residual map. From the per-token records of cycle 1 (loop59_c1_<level>.json):
(a) contexts where Markov-2 beats the structural model (local statistics not written as rules), with the fit-set
    continuation that carries them; (b) contexts where every model stays at unigram level (irreducible content);
(c) irreducible bits per text and per object type under the best model, split frame / qualifiers / counts / middle;
(d) predictability per token (P >= 0.5) per slot.
Usage: python3 tools/dark_loop59_c2.py <level>   -> data/derived/dark/loop59_c2_<level>.txt / .json
"""
import sys, json, math, collections
sys.path.insert(0, '/home/user/Indus-/tools')
from dark_loop59_common import *
LV = sys.argv[1]
J = json.load(open(DARK + f'loop59_c1_{LV}.json'))
OUT = DARK + f'loop59_c2_{LV}'
logf = open(OUT + '.txt', 'w')
def log(*a):
    s = ' '.join(str(x) for x in a); print(s); logf.write(s + '\n'); logf.flush()
ALL = load_corpus(LV); FIT = [o for o in ALL if o['site'] in BIG]
big = collections.defaultdict(collections.Counter); uni = collections.Counter()
for o in FIT:
    s = list(o['seq']) + [END]
    for i, x in enumerate(s): uni[x] += 1
    for a, b in zip(s, s[1:]): big[a][b] += 1
NTOK = sum(uni.values())
MODELS = ['unigram', 'KN1', 'KN2', 'KN3', 'frame_only', 'frame+rules', 'structural', 'combined', 'S366gen', 'loop33_PFA']
R = {}
log(f'# loop 59 cycle 2, level {LV}: residual map from cycle 1 records (held-out sites + IM77-only pooled, and separately)')
for setname in ('pooled', 'held', 'im'):
    recs = J['held_recs'] + J['im_recs'] if setname == 'pooled' else J[setname + '_recs']
    for r in recs:
        for k in ('x', 'prev'):
            v = r[k]
            r[k] = int(v) if v.lstrip('-').isdigit() else v
    ntexts = len(set((r['site'], r['t'], r['n']) for r in recs))
    log(f'\n## {setname}: {len(recs)} tokens')
    # (a) KN2 vs structural by (zone, prev sign)
    ctx = collections.defaultdict(list)
    for r in recs: ctx[(r['zone'], r['prev'])].append(r)
    rows = []
    for key, L in ctx.items():
        if len(L) < 12: continue
        d = sum(r['bits']['KN2'] for r in L) / len(L) - sum(r['bits']['structural'] for r in L) / len(L)
        rows.append((d, key, len(L), sum(r['bits']['KN2'] for r in L) / len(L), sum(r['bits']['unigram'] for r in L) / len(L)))
    rows.sort()
    log('(a) contexts (zone, previous sign) where Markov-2 beats the structural model, n >= 12; fit-set continuations of the previous sign:')
    log('| zone | prev | n | KN2 bits | structural - KN2 | unigram bits | fit-set top continuations P(next|prev) |')
    log('|---|---|---|---|---|---|---|')
    beats = []
    for d, key, n, k2, u in rows:
        if d > -0.3: break
        z, p = key; cont = big.get(p, collections.Counter()); tot = sum(cont.values())
        top = ', '.join(f'{b}:{c/tot:.2f}' for b, c in cont.most_common(4)) if tot else '-'
        log(f'| {z} | {p} | {n} | {k2:.2f} | {-d:.2f} | {u:.2f} | {top} |'); beats.append(dict(zone=z, prev=str(p), n=n, kn2=k2, gain=-d, uni=u, top=top))
    tot_tokens = len(recs); kn2_better = sum(1 for r in recs if r['bits']['KN2'] < r['bits']['structural'] - 0.5)
    str_better = sum(1 for r in recs if r['bits']['structural'] < r['bits']['KN2'] - 0.5)
    log(f'tokens where KN2 is better by > 0.5 bit: {kn2_better} ({kn2_better/tot_tokens:.3f}); structural better by > 0.5 bit: {str_better} ({str_better/tot_tokens:.3f})')
    # by slot x prevclass: mean gain of KN2 over structural
    sp = collections.defaultdict(list)
    for r in recs: sp[(r['slot'], r['pc'])].append(r['bits']['structural'] - r['bits']['KN2'])
    log('slot x previous-class contexts (n >= 20), structural - KN2 bits (positive = chain knows more):')
    log('  ' + '; '.join(f'{k[0]}<-{k[1]} n={len(v)} {sum(v)/len(v):+.2f}' for k, v in sorted(sp.items(), key=lambda kv: -sum(kv[1]) / len(kv[1])) if len(v) >= 20))
    # (b) irreducible tokens: min over all models >= unigram - 0.25
    def best_bits(r): return min(r['bits'][m] for m in MODELS if m in r['bits'])
    irr = [r for r in recs if best_bits(r) >= r['bits']['unigram'] - 0.25]
    log(f'\n(b) irreducible tokens (no model beats the unigram by 0.25 bit): {len(irr)} of {len(recs)} ({len(irr)/len(recs):.3f})')
    byslot = collections.Counter(r['slot'] for r in irr); nslot = collections.Counter(r['slot'] for r in recs)
    log('  by slot: ' + ', '.join(f'{s} {byslot[s]}/{nslot[s]} ({byslot[s]/nslot[s]:.2f})' for s in ['OPENER', 'MARKER', 'COUNT', 'NAME', 'TITLE', 'CLOSER', 'SUFFIX', 'END'] if nslot[s]))
    byctx = collections.Counter((r['slot'], r['pc']) for r in irr); nctx = collections.Counter((r['slot'], r['pc']) for r in recs)
    hot = [(k, byctx[k], nctx[k]) for k in nctx if nctx[k] >= 20 and byctx[k] / nctx[k] >= 0.7]
    hot.sort(key=lambda x: -x[2])
    log('  contexts (slot <- previous class) with >= 70% irreducible tokens, n >= 20: ' + '; '.join(f'{k[0]}<-{k[1]} {a}/{b}' for k, a, b in hot))
    # which signs are the irreducible ones (next sign identity): unigram frequency of irreducible vs reducible tokens
    fr_irr = sum(uni.get(r['x'], 0) for r in irr) / max(1, len(irr)); fr_red = sum(uni.get(r['x'], 0) for r in recs if r not in irr) / max(1, len(recs) - len(irr))
    log(f'  mean fit-set frequency of the sign at irreducible tokens {fr_irr:.0f} vs reducible {fr_red:.0f} (of {NTOK} tokens)')
    # (c) bits per text by slot group under the best model
    best = 'combined'
    G = {'frame': {'OPENER', 'MARKER', 'CLOSER', 'SUFFIX', 'END'}, 'qualifiers': {'TITLE'}, 'counts': {'COUNT'}, 'middle': {'NAME'}}
    def grp(sl): return next(g for g, S in G.items() if sl in S)
    log(f'\n(c) bits per text under {best} (and under the unigram), by slot group and object type:')
    log('| object type | texts | bits/text best | frame | qualifiers | counts | middle | unigram bits/text | explained share | middle share of remaining |')
    log('|---|---|---|---|---|---|---|---|---|---|')
    R[setname] = {}
    for ot in ('ALL', 'SEAL', 'TAB', 'OTHER'):
        L = [r for r in recs if ot == 'ALL' or r['ot'] == ot]
        nt = len(set((r['site'], r['t'], r['n']) for r in L))
        if nt < 5: continue
        tb = collections.Counter(); ub = 0.0
        for r in L: tb[grp(r['slot'])] += r['bits'][best]; ub += r['bits']['unigram']
        T = sum(tb.values())
        log(f'| {ot} | {nt} | {T/nt:.1f} | {tb["frame"]/nt:.1f} | {tb["qualifiers"]/nt:.1f} | {tb["counts"]/nt:.1f} | {tb["middle"]/nt:.1f} | {ub/nt:.1f} | {100*(1-T/ub):.0f}% | {100*tb["middle"]/T:.0f}% |')
        R[setname][ot] = dict(texts=nt, best=T / nt, uni=ub / nt, by={g: v / nt for g, v in tb.items()}, explained=1 - T / ub, middle_share=tb['middle'] / T)
    # per-slot gap closed relative to unigram, and relative to uniform, for frame vs middle
    log('\nslot-level summary (pooled over object types): bits under unigram -> best; gap closed vs unigram; vs uniform (log2(|V|+2))')
    V2 = J['meta']['V'] + 2; unif = math.log2(V2)
    for g, S in G.items():
        L = [r for r in recs if r['slot'] in S]
        if not L: continue
        u = sum(r['bits']['unigram'] for r in L) / len(L); b = sum(r['bits'][best] for r in L) / len(L)
        log(f'  {g:10s} n={len(L):5d}  unigram {u:.2f} -> best {b:.2f}; gap closed {100*(1-b/u):.0f}% of unigram, {100*(1-b/unif):.0f}% of uniform')
        R[setname][g] = dict(n=len(L), uni=u, best=b)
    # (d) predictability: P >= 0.5 (bits <= 1) per slot under best
    log('\n(d) share of tokens predicted with P >= 0.5 (<= 1 bit) under the best model, by slot:')
    log('  ' + ', '.join(f'{s} {sum(1 for r in recs if r["slot"] == s and r["bits"][best] <= 1)}/{nslot[s]} ({sum(1 for r in recs if r["slot"] == s and r["bits"][best] <= 1)/nslot[s]:.2f})' for s in ['OPENER', 'MARKER', 'COUNT', 'NAME', 'TITLE', 'CLOSER', 'SUFFIX', 'END'] if nslot[s]))
    R[setname]['beats'] = beats; R[setname]['irreducible'] = dict(n=len(irr), of=len(recs), byslot={s: [byslot[s], nslot[s]] for s in nslot})
    R[setname]['kn2_better'] = kn2_better; R[setname]['str_better'] = str_better
json.dump(R, open(OUT + '.json', 'w'))
