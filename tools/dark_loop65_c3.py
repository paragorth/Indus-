"""S-DARK-65 cycle 3: the Lothal and Kalibangan multi-impression sealings. Is there one recurring 'office'
impression (warehouse stamp) with varying partners (consignors)? Co-sealing network from Wells faces AND IM77 sides
(same objects read twice -> transcription-robust, not replicated).
Nulls: (a) the impression slots of the multi-impression sealings refilled by draws from ALL impressions of the site
(frequency null: is the hub more than the site's heaviest-used seal?), 2,000x; (b) partner features (opener-initial,
length, head) vs the other impressions of the site, label permuted 2,000x.
Usage: python3 tools/dark_loop65_c3.py <seq_raw|seq_strong|seq_all> [nperm]"""
import sys, collections, itertools, random
sys.path.insert(0, '/home/user/Indus-/tools')
from dark_loop65 import *
LV = sys.argv[1] if len(sys.argv) > 1 else 'seq_raw'; NP = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
rnd = random.Random(653)
objs, S = load_sealings(LV)
seals = seal_texts(objs)
out = []
def P(*a):
    s = ' '.join(str(x) for x in a); print(s); out.append(s)
P(f'== S-DARK-65 cycle 3 ({LV}, nperm {NP}): office stamp at Lothal / Kalibangan?')

def merge_fragments(sealings):
    """a fragment impression that is a contiguous substring of exactly ONE longer distinct text at the site is read as
    that seal (IM77 marks lost signs 0, Wells 000; both are dropped, so a fragment looks like a shorter text)."""
    texts = collections.Counter(tuple(i) for s in sealings for i in s['imps'])
    longer = sorted(texts, key=len, reverse=True)
    rep = {}
    for t in texts:
        cands = [u for u in longer if len(u) > len(t) and any(u[i:i + len(t)] == t for i in range(len(u) - len(t) + 1))]
        if len(set(cands)) == 1: rep[t] = cands[0]
    for t in list(rep):
        while rep[t] in rep: rep[t] = rep[rep[t]]
    out = []
    for s in sealings:
        imps = []
        for i in s['imps']:
            j = list(rep.get(tuple(i), tuple(i)))
            if j not in imps: imps.append(j)
        out.append(dict(s, imps=imps))
    return out, rep

def network(site, sealings, label, open_set, fmtf=fmt, merge=True):
    """sealings: list of dict(oid, imps=[seq,...]) with >= 2 distinct legible impressions; plus all impressions at site"""
    if merge:
        sealings, rep = merge_fragments(sealings)
        if rep: P(f'   [{label} {site}] fragments read as their unique longer site text: ' + '; '.join(f'{fmtf(a)} -> {fmtf(b)}' for a, b in rep.items()))
    multi = [s for s in sealings if len(s['imps']) >= 2]
    allimp = [tuple(i) for s in sealings for i in s['imps']]
    slots = sum(len(s['imps']) for s in multi)
    freq = collections.Counter(allimp)
    P(f'\n-- {label} {site}: {len(sealings)} sealings with legible text, {len(multi)} with >= 2 distinct impressions ({slots} slots), {len(freq)} distinct texts at the site')
    on = collections.defaultdict(set); partners = collections.defaultdict(set)
    for s in multi:
        for a in s['imps']:
            on[tuple(a)].add(s['oid'])
            for b in s['imps']:
                if tuple(b) != tuple(a): partners[tuple(a)].add(tuple(b))
    ranked = sorted(on, key=lambda t: (-len(on[t]), -len(partners[t])))
    P('   text (head) : on n multi-impression sealings / n distinct partners / n uses at site overall / partner texts')
    for t in ranked[:12]:
        if len(on[t]) < 2 and len(partners[t]) < 2: continue
        P(f'   {fmtf(t):34s} [{head_of(t) if label=="Wells" else head_of(t, M_HEAD, M_SUF, M_OPEN)}] : {len(on[t])} / {len(partners[t])} / {freq[t]} : '
          + '; '.join(fmtf(p) for p in sorted(partners[t], key=len)))
    nd_obs = len(set(allimp[i] for i in range(len(allimp))) & set(on))   # distinct texts among multi slots
    nd_obs = len(on)
    hub_obs = max(len(on[t]) for t in on) if on else 0
    hubp_obs = max(len(partners[t]) for t in on) if on else 0
    # null (a): refill slots from the site's impression frequency, without replacement from the multiset of all impressions
    nd, hub, hubp = [], [], []
    for _ in range(NP):
        bag = list(allimp); rnd.shuffle(bag)
        k = 0; on_n = collections.defaultdict(set); pn = collections.defaultdict(set)
        for s in multi:
            m = len(s['imps']); pick = bag[k:k + m]; k += m
            for a in pick:
                on_n[a].add(s['oid'])
                for b in pick:
                    if b != a: pn[a].add(b)
        nd.append(len(on_n)); hub.append(max((len(v) for v in on_n.values()), default=0)); hubp.append(max((len(v) for v in pn.values()), default=0))
    P(f'   distinct texts among the {slots} multi slots: {nd_obs} vs null {sum(nd)/NP:.1f} (P_lo = {pval(nd_obs, nd, "lo"):.3f}, P_hi = {pval(nd_obs, nd):.3f})')
    P(f'   hub: max sealings for one text {hub_obs} vs null {sum(hub)/NP:.1f} (P = {pval(hub_obs, hub):.3f}); max distinct partners {hubp_obs} vs {sum(hubp)/NP:.1f} (P = {pval(hubp_obs, hubp):.3f})')
    # (b) partner features: for the hub text, its partners vs all other impressions at the site
    if on:
        hubt = ranked[0]
        part = [p for p in partners[hubt]]
        others = [t for t in allimp if t != hubt and t not in partners[hubt]]
        def feats(ts):
            n = len(ts)
            return dict(opener=sum(1 for t in ts if t and t[0] in open_set) / n, meanlen=sum(len(t) for t in ts) / n,
                        jar=sum(1 for t in ts if (head_of(t) if label == 'Wells' else head_of(t, M_HEAD, M_SUF, M_OPEN)) == 'jar') / n,
                        sealmatch=(sum(1 for t in ts if t in seals) / n) if label == 'Wells' else float('nan'))
        fp, fo = feats(part), feats(others)
        P(f'   hub {fmtf(hubt)}: {len(part)} partners vs {len(others)} other impressions: opener-initial {fp["opener"]:.2f} vs {fo["opener"]:.2f}; '
          f'mean length {fp["meanlen"]:.1f} vs {fo["meanlen"]:.1f}; jar-headed {fp["jar"]:.2f} vs {fo["jar"]:.2f}; exact seal match {fp["sealmatch"]:.2f} vs {fo["sealmatch"]:.2f}')
        pool = part + others; nullo = []; nulll = []
        for _ in range(NP):
            rnd.shuffle(pool); f = feats(pool[:len(part)]); nullo.append(f['opener']); nulll.append(f['meanlen'])
        P(f'      opener share P_hi = {pval(fp["opener"], nullo):.3f}; length P_lo = {pval(fp["meanlen"], nulll, "lo"):.3f}, P_hi = {pval(fp["meanlen"], nulll):.3f}')
        # does the hub also occur alone (single-impression sealings)?
        alone = sum(1 for s in sealings if len(s['imps']) == 1 and tuple(s['imps'][0]) == hubt)
        P(f'      hub occurs alone on {alone} single-impression sealings and on {len(on[hubt])} multi-impression ones')
    return on, partners

# ---- Wells ----
for site in ('Lothal', 'Kalibangan', 'Dholavira'):
    sl = [dict(oid=o['oid'], cisi=o['cisi'], imps=[f['seq'] for f in o['imps']]) for o in S.values() if o['site'] == site and o['imps']]
    network(site, sl, 'Wells', OPEN)
# ---- IM77 ----
im = load_im77()
def im_sealings(site):
    res = []
    for o in im.values():
        if o['type'] != 'sealing' or o['site'] != site: continue
        faces = [f['seq'] for f in o['faces'] if f['seq'] and not is_count_face_M(f['seq'])]
        # collapse identical sides and fragments contained in another side
        imps = []
        for f in faces:
            if any(f == g or (len(f) >= 1 and len(f) < len(g) and any(g[i:i + len(f)] == f for i in range(len(g) - len(f) + 1))) for g in imps): continue
            imps = [g for g in imps if not (len(g) < len(f) and any(f[i:i + len(g)] == g for i in range(len(f) - len(g) + 1)))] + [f]
        if imps: res.append(dict(oid=o['oid'], imps=imps))
    return res
for site in ('Lothal', 'Kalibangan', 'Mohenjodaro'):
    network(site, im_sealings(site), 'IM77', M_OPEN)
# ---- Kalibangan detail: are K-87/K-88 one seal stamped twice? ----
P('\n-- Kalibangan detail (Wells faces vs IM77 sides):')
for o in S.values():
    if o['site'] == 'Kalibangan' and len(o['faces']) >= 2:
        P(f'   {o["cisi"]}: faces ' + ' | '.join(f["text"] for f in o['faces']) + f'  -> dies {[fmt(f["seq"]) for f in o["imps"]]} + {o["illegible"]} illegible')
for o in im.values():
    if o['type'] == 'sealing' and o['site'] == 'Kalibangan' and len(o['faces']) >= 2:
        P(f'   IM77 {o["oid"]}: sides ' + ' | '.join(' '.join(map(str, f['seq'])) for f in o['faces']))
P('\n-- Lothal detail: IM77 multi-side sealings (sides, 0 = lost sign dropped):')
for o in im.values():
    if o['type'] == 'sealing' and o['site'] == 'Lothal' and len(o['faces']) >= 2:
        P(f'   IM77 {o["oid"]}: ' + ' | '.join(' '.join(map(str, f['seq'])) for f in o['faces']))
open(OUT + f'loop65_c3_{LV}.txt', 'w').write('\n'.join(out) + '\n')
