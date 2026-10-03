"""Loop 73 cycle 4: is the designation bound to its head?  And what are the PE header designations?
(4a) Among designation tokens (>= 2 elements) whose designation occurs >= 2 times, MI(designation, head) as a share
     of H(head) and the share of recurring designations seen with >= 2 different heads, vs heads permuted among
     those tokens (1,000x). Head = PE final class sign (or none), PE numeral system (second head); Indus closer
     (S310 unit, or none), seq_raw / seq_strong / seq_all, all objects and seals; Linear B = first ideogram on the
     line (DAMOS lines with name + ideogram + number). A person-designation receiving varied goods should be FREE of
     its head (MI near null); a fixed compound term should be BOUND.
(4b) PE header designations (opener removed): where their signs live in the entry grammar (class / prefix / middle
     element / unseen), vs the token mix of entries.
Usage: python3 tools/dark_loop73_c4.py [nperm]
"""
import sys
NPERM = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
sys.path.insert(0, '/home/user/Indus-/tools')
from dark_loop73_common import *

R = {}
P(f'== Loop 73 cycle 4: designation-head binding; PE header designations; nperm {NPERM}')
def binding(label, toks):
    """toks: list of (designation, head)."""
    c = collections.Counter(d for d, _ in toks)
    T = [(d, h) for d, h in toks if c[d] >= 2]
    if len(T) < 20: P(f'  [{label}] too few recurring tokens ({len(T)})'); return None
    def stats(tt):
        j = collections.Counter(tt); a = collections.Counter(d for d, _ in tt); b = collections.Counter(h for _, h in tt); N = len(tt)
        mi = sum(v / N * math.log2(v * N / (a[x] * b[y])) for (x, y), v in j.items())
        hs = collections.defaultdict(set)
        for d, h in tt: hs[d].add(h)
        return mi, sum(1 for s in hs.values() if len(s) >= 2) / len(hs)
    mi, multi = stats(T); hb = H(collections.Counter(h for _, h in T))
    r = random.Random(74); heads = [h for _, h in T]; nm = []; nmu = []
    for _ in range(NPERM):
        r.shuffle(heads); m2, mu2 = stats([(d, h) for (d, _), h in zip(T, heads)]); nm.append(m2); nmu.append(mu2)
    mu = sum(nm) / NPERM; mm = sum(nmu) / NPERM
    p = sum(1 for x in nm if x >= mi) / NPERM
    res = dict(tokens=len(T), designations=len(set(d for d, _ in T)), mi=mi, mi_null=mu, share=(mi - mu) / hb if hb else float('nan'),
               p=p, multi=multi, multi_null=mm, Hhead=hb)
    P(f'  [{label}] recurring tokens {len(T)} ({res["designations"]} designations): MI {mi:.3f} vs {mu:.3f} null, excess {(mi-mu)/hb if hb else float("nan"):.3f} of H(head) {hb:.2f} (P {p:.3f}); designations with >= 2 heads {multi:.3f} vs null {mm:.3f}')
    return res

E = pe_entries()
R['PE_cls'] = binding('PE middle -> class sign', [(e['mid'], e['cls'] or 'none') for e in E if len(e['mid']) >= 2])
R['PE_sys'] = binding('PE middle -> numeral system', [(e['mid'], e['system'] or 'none') for e in E if len(e['mid']) >= 2])
R['PE_cls_1'] = binding('PE middle (>= 1 element) -> class sign', [(e['mid'], e['cls'] or 'none') for e in E if len(e['mid']) >= 1])
for LV in ['seq_raw', 'seq_strong', 'seq_all']:
    O = D56.load_indus(LV, dedup=False)
    R[f'Indus_{LV}'] = binding(f'Indus {LV} middle -> closer, all objects (rows)', [(o['mid'], o['closer'] or 'none') for o in O if len(o['mid']) >= 2])
    R[f'Indus_{LV}_seals'] = binding(f'Indus {LV} middle -> closer, seals', [(o['mid'], o['closer'] or 'none') for o in O if len(o['mid']) >= 2 and o['ot'] == 'seal'])
    Od = D56.load_indus(LV)
    R[f'Indus_{LV}_dedup'] = binding(f'Indus {LV} middle -> closer, one per site x type x text', [(o['mid'], o['closer'] or 'none') for o in Od if len(o['mid']) >= 2])
O = D56.im77_objects()
R['IM77_dedup'] = binding('IM77 middle -> closer, one per site x type x text', [(o['mid'], o['closer'] or 'none') for o in O if len(o['mid']) >= 2])
LBl = linb_lines()
lb = []
for x in LBl:
    ide = next((f for f in x['frame'] if f.isupper() and f != 'NUM'), 'none')
    lb.append((x['name'], ide))
R['LinB_ideogram'] = binding('Linear B name -> ideogram on the line', lb)
lbs = [(x['name'], x['series']) for x in linb_names_tab()]
R['LinB_series'] = binding('Linear B name -> tablet series (all personnel names)', lbs)

P('\n##### (4b) PE header designations: where do their signs live in the entry grammar?')
HD = pe_headers()
mid_el = collections.Counter(a for e in E for a in e['mid']); allsg = collections.Counter(a for e in E for a in e['signs'])
def role(a):
    if a in CLASS: return 'CLASS'
    if a in PREFIX: return 'PREFIX'
    if mid_el[a] >= 1: return 'MIDDLE'
    return 'UNSEEN'
ht = [a for h in HD for a in h['des']]
hc = collections.Counter(role(a) for a in ht); ec = collections.Counter(role(a) for e in E for a in e['signs'])
nh = len(ht); ne = sum(ec.values())
P(f'  header-designation tokens {nh} (from {sum(1 for h in HD if h["des"])} headers): ' + ', '.join(f'{k} {hc[k]/nh:.3f} (entry tokens {ec[k]/ne:.3f})' for k in ['CLASS', 'PREFIX', 'MIDDLE', 'UNSEEN']))
op = collections.Counter(h['opener'] for h in HD)
P(f'  headers {len(HD)}: opener present {sum(v for k,v in op.items() if k)}; designation empty when opener present {sum(1 for h in HD if h["opener"] and not h["des"])}; headers without opener {op[None]}, their single sign commonest {collections.Counter(h["des"] for h in HD if not h["opener"]).most_common(10)}')
hap = sum(1 for a in ht if allsg[a] <= 1) / nh
P(f'  header-designation signs unseen in entries or seen once: {hap:.3f}')
R['headers'] = dict(n=nh, mix={k: hc[k] / nh for k in hc}, entry_mix={k: ec[k] / ne for k in ec}, rare=hap)
save('loop73_c4', R)
