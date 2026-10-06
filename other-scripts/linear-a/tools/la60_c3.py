#!/usr/bin/env python3
"""LA-60 cycle 3: outside-corpus predictions, frozen, plus a pseudo-outside test and the gap list.
A. Pseudo-outside test: reading induced on documents published in GORILA (1976-85), used to
   decode the administrative documents published after 1985 (lineara.xyz source field), against
   shuffled and random readings through the identical pipeline.
B. Word -> commodity predictions: scope commodity of each word occurrence (same line, else the last
   commodity above); fitted on a training half, scored on the other half against the site-default
   baseline and a doc-shuffled control; then refitted on everything and frozen.
C. Frozen predictions per site (structure, totals, order, vocabulary novelty), sha256-hashed.
D. Gaps: word types that block a full decode most often under the PRIOR reading."""
import json, os, sys, random, statistics as st, hashlib
from collections import Counter, defaultdict
from la60_common import *
from la60_decode import *
import la60_c2 as C2

C2.NSH = int(os.environ.get('NSH', 200))


def scope_pairs(docs, roles):
    """(word, site, scope commodity base) for multi-sign entry words."""
    out = []
    for d in docs:
        toks = d['toks']; line = 0; lines = defaultdict(list); last = None; seq = []
        for t in toks:
            if t[0] == 'NL':
                line += 1; continue
            seq.append((line, t))
        for ln, t in seq:
            if t[0] == 'L' or (t[0] == 'W' and roles.get(t[1]) == 'COM'):
                lines[ln].append(base_of(t[1]))
        for ln, t in seq:
            if t[0] == 'L' or (t[0] == 'W' and roles.get(t[1]) == 'COM'):
                last = base_of(t[1]); continue
            if t[0] == 'W' and '-' in t[1] and t[1] not in roles:
                c = lines[ln][0] if lines[ln] else last
                if c:
                    out.append((t[1], d['site'], c, d['id']))
    return out


def word_commodity(tr, te, roles, rng, nsh=200):
    P = scope_pairs(tr, roles); Q = scope_pairs(te, roles)
    wc = defaultdict(Counter); sc = defaultdict(Counter)
    for w, s, c, _ in P:
        wc[w][c] += 1; sc[s][c] += 1
    def acc(wc_, Q_):
        hit = base = n = 0
        for w, s, c, _ in Q_:
            if w in wc_ and sum(wc_[w].values()) >= 2:
                n += 1
                hit += wc_[w].most_common(1)[0][0] == c
                base += (sc[s].most_common(1)[0][0] if sc[s] else 'GRA') == c
        return hit, base, n
    real = acc(wc, Q)
    # control: training scope commodities shuffled across occurrences within site
    sh = []
    for _ in range(nsh):
        bys = defaultdict(list)
        for i, (w, s, c, _) in enumerate(P):
            bys[s].append(i)
        cs = [c for _, _, c, _ in P]
        for s, ix in bys.items():
            v = [cs[i] for i in ix]; rng.shuffle(v)
            for i, x in zip(ix, v):
                cs[i] = x
        wc2 = defaultdict(Counter)
        for (w, s, _, _), c in zip(P, cs):
            wc2[w][c] += 1
        sh.append(acc(wc2, Q)[0])
    return dict(hit=real[0], site_default=real[1], n=real[2], shuf_mean=st.mean(sh),
                P=sum(1 for x in sh if x >= real[0]) / len(sh))


def main():
    rng = random.Random(seed('la60-c3'))
    LA = load_la(); A = admin_docs(LA)
    out = {}
    # ---------------- A. pseudo-outside: GORILA -> post-1985
    tr = [d for d in A if d['pub'] != 'post']; te = [d for d in A if d['pub'] == 'post']
    RI = induce_reading(tr, name='INDUCED'); RP = prior_reading()
    out['A_n'] = dict(train=len(tr), test=len(te), test_sites=Counter(d['site'] for d in te))
    out['A_hash'] = {'INDUCED-GORILA': reading_hash(RI), 'PRIOR': reading_hash(RP)}
    for R in (RI, RP):
        out['A_' + R['name']] = C2.evaluate(tr, te, R, False, rng)
        print('A', R['name'], json.dumps(out['A_' + R['name']]), flush=True)
    out['A_EMPTY'] = C2.score(tr, te, dict(name='EMPTY', roles={}, order={}, lib=set(), site_default={}), False)
    with open(os.path.join(CK, 'c3_post_glosses.txt'), 'w') as f:
        for R in (RI, RP):
            acc = acceptor(tr, R)
            for d in te:
                ch, g = decode(d, R, acc)
                f.write('%s\t%s\t%s\n' % (R['name'], 'FULL' if ch['full'] else '-', g))
    # ---------------- B. word -> commodity
    RPr = RP['roles']
    wcs = []
    for s in range(10):
        a, b = split(A, 'la60-c3-wc%d' % s)
        wcs.append(word_commodity(a, b, RPr, rng, nsh=100))
    out['B_word_commodity_splits'] = wcs
    print('B', json.dumps(wcs), flush=True)
    LBd = [d for d in load_lb() if any(t[0] == 'N' for t in d['toks'])]
    lbw = []
    RBP = lb_prior_reading(LBd)
    for s in range(5):
        dr = lb_draw(LBd, A, s); a, b = split(dr, 'la60-c3-lbwc%d' % s)
        lbw.append(word_commodity(a, b, RBP['roles'], rng, nsh=100))
    out['B_LB_word_commodity'] = lbw
    print('B-LB', json.dumps(lbw), flush=True)
    allp = scope_pairs(A, RPr); wc = defaultdict(Counter); wsite = defaultdict(set)
    for w, s, c, i in allp:
        wc[w][c] += 1; wsite[w].add(s)
    word_preds = {w: dict(commodity=c.most_common(1)[0][0], n=sum(c.values()), share=round(c.most_common(1)[0][1] / sum(c.values()), 2),
                          sites=sorted(wsite[w]))
                  for w, c in wc.items() if sum(c.values()) >= 3}
    # ---------------- C. per-site frozen structure predictions
    RF = induce_reading(A, name='INDUCED-ALL')
    site = defaultdict(list)
    for d in A:
        site[d['site'] if d['site'] in ('HT', 'KH', 'ZA', 'PH', 'KN') else 'other'].append(d)
    preds = {}
    for s, ds in site.items():
        tabs = [d for d in ds if d['support'] in ('Tablet', 'Lames (short thin tablet)')]
        n = len(ds); nt = max(len(tabs), 1)
        com = Counter(); kur = 0; sects = 0; closes = 0; hdr = 0; ents = []
        for d in ds:
            for b in com_seq(d, RPr):
                com[b] += 1
        for d in tabs:
            ws = [t[1] for t in d['toks'] if t[0] == 'W']
            kur += any(w in ('KU-RO', 'PO-TO-KU-RO') for w in ws)
            for sc in sections(d, RPr):
                if sc[3] is not None:
                    sects += 1; closes += sc[3]
            rs = roles_doc(d, RP)
            hdr += bool(rs) and rs[0] in ('HDR', 'TRX', 'UNK')
            ents.append(sum(t[0] == 'N' for t in d['toks']))
        tot = sum(com.values())
        preds[s] = dict(n_docs=n, n_tablets=len(tabs),
                        commodity_share={k: round(v / max(sum(1 for d in ds if com_seq(d, RPr)), 1), 2) for k, v in com.most_common(6)},
                        p_tablet_has_KU_RO=round(kur / nt, 2), KU_RO_bound=('< %.3f' % (3 / nt)) if kur == 0 else None,
                        p_total_closes=round(closes / sects, 2) if sects else None, n_sections=sects,
                        p_opens_with_heading=round(hdr / nt, 2), median_numbers_per_tablet=st.median(ents) if ents else None,
                        SA_RA2_or_KI_RO=sum(1 for d in ds for t in d['toks'] if t[0] == 'W' and t[1] in ('SA-RA₂', 'KI-RO')))
    order_all = bt_order(A, RPr)
    pair = Counter()
    for d in A:
        sq = com_seq(d, RPr)
        for i in range(len(sq)):
            for j in range(i + 1, len(sq)):
                pair[(sq[i], sq[j])] += 1
    pair_preds = {}
    for (a, b), n in pair.items():
        m = pair[(b, a)]
        if a < b or (b, a) not in pair:
            if n + m >= 3:
                x, y = (a, b) if n >= m else (b, a)
                pair_preds['%s before %s' % (x, y)] = dict(n=n + m, p=round(max(n, m) / (n + m), 2))
    sara = Counter()
    for d in A:
        toks = d['toks']
        for i, t in enumerate(toks):
            if t[0] == 'W' and t[1] == 'SA-RA₂':
                for t2 in toks[i + 1:]:
                    if t2[0] == 'L' or (t2[0] == 'W' and RPr.get(t2[1]) == 'COM'):
                        sara[base_of(t2[1])] += 1; break
    frozen = dict(
        date='2026-10-06', reading_prior_hash=reading_hash(RP), reading_induced_all_hash=reading_hash(RF),
        reading_induced_all=dict(roles=RF['roles'], order=RF['order'], site_default=RF['site_default']),
        per_site=preds, commodity_order=order_all, commodity_pairs=pair_preds,
        after_SA_RA2_first_commodity=dict(sara),
        word_commodity=word_preds,
        new_tablet_rules=[
            'HT tablet with KU-RO and >= 2 entries: KU-RO equals the running sum (fraction slack) with p ~ observed p_total_closes; a second KU-RO may be a running total (HT 127b).',
            'KU-RO, KI-RO, SA-RA2: on non-HT/ZA tablets each stays rare (< 3/n bound per site given current absence).',
            'Commodity order on any new multi-commodity list: grain-class (GRA, CYP, *307) before OLE/VIR before OLIV/VIN.',
            'After SA-RA2 the first commodity is GRA or CYP; NI follows grain; VIN last.',
            'Single-sign heading signs (*301 KA KU SI RO ZE TE A I) open lines/documents and are mostly not counted.',
            'Totals sit on the bottom line on a line of their own.',
            'About 55 % of the entry words on a new HT tablet will be unseen (la23); ~3 of 4 words on any new tablet are new (la15).',
            'A new document without a commodity sign: Khania -> CYP, Zakros/Arkhalkhori -> GRA (site default, la54).'])
    s = json.dumps(frozen, ensure_ascii=False, sort_keys=True)
    h = hashlib.sha256(s.encode()).hexdigest()
    open(os.path.join(CK, 'c3_frozen_predictions.json'), 'w').write(s)
    out['C_hash'] = h
    print('C hash', h, flush=True)
    # ---------------- D. gaps
    acc = acceptor(A, RP)
    blk = Counter(); blkd = defaultdict(set); fail = Counter(); nonclose = []
    for d in A:
        ch, g = decode(d, RP, acc)
        rs = roles_doc(d, RP)
        for t, r in zip(d['toks'], rs):
            if r == 'UNK':
                blk[t[1]] += 1; blkd[t[1]].add(d['id'])
        for k in ('cover', 'roles', 'totals', 'order', 'substantive'):
            fail[k] += not ch[k]
        for sc in sections(d, RP['roles']):
            if sc[3] is False:
                nonclose.append((d['id'], sc[1], sum(e[1] for e in sc[2])))
    out['D_fail'] = dict(fail); out['D_full'] = sum(1 for d in A if decode(d, RP, acc)[0]['full'])
    out['D_blockers'] = [(w, c, len(blkd[w])) for w, c in blk.most_common(40)]
    out['D_nonclose'] = nonclose
    out['n_admin'] = len(A)
    json.dump(out, open(os.path.join(CK, 'c3.json'), 'w'), ensure_ascii=False, indent=1, default=list)
    print('D', json.dumps(dict(fail=out['D_fail'], full=out['D_full'], n=len(A), blockers=out['D_blockers'][:25]), ensure_ascii=False))


if __name__ == '__main__':
    main()
