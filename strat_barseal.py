"""S157: bar (long rectangular, script-only) seals vs square seals: does the home frame hold?
Kenoyer & Meadow (CISI 3.1, 2010) date bar seals to Harappa Period 3C (archaeological data, not a reading).
Control: permutation of the seal-type label within site x text-length strata (10,000 draws)."""
import json,collections as C,random
d=json.load(open('data/derived/merged-corpus-reading-order.json'))
OP={817,861,820}
def feats(s):
    return dict(opener=len(s)>1 and s[0] in OP and s[1]==2, jar=s[-1]==740 or (len(s)>1 and s[-2]==740),
                suffix=len(s)>1 and s[-2]==740 and s[-1] in (400,90), person=any(x in (90,91) for x in s))
rows=[]; seen=set()
for x in d:
    if x['type'] not in ('SEAL:S','SEAL:R') or len(x['seq'])<2: continue
    k=(x['site'],tuple(x['seq']))
    if k in seen: continue
    seen.add(k); rows.append((x['site'],min(len(x['seq']),7),x['type']=='SEAL:R',feats(x['seq']),x['time']))
print('texts',len(rows),'bar',sum(r[2] for r in rows))
print('bar by site',C.Counter(r[0] for r in rows if r[2]).most_common(6))
print('Harappa bar by period',C.Counter(r[4] for r in rows if r[2] and r[0]=='Harappa'))
print('mean len sq/bar',sum(r[1] for r in rows if not r[2])/sum(not r[2] for r in rows),sum(r[1] for r in rows if r[2])/sum(r[2] for r in rows))
strata=C.defaultdict(list)
for r in rows: strata[(r[0],r[1])].append(r)
random.seed(1)
for f in ('opener','jar','suffix','person'):
    obs=sum(r[3][f] for r in rows if r[2]); nb=sum(r[2] for r in rows)
    sq=sum(r[3][f] for r in rows if not r[2]); ns=len(rows)-nb
    ge=le=0; tot=0
    for _ in range(10000):
        v=0
        for rs in strata.values():
            k=sum(r[2] for r in rs)
            if k: v+=sum(r[3][f] for r in random.sample(rs,k))
        tot+=v; ge+=v>=obs; le+=v<=obs
    print(f"{f:7s} bar {obs}/{nb} ({obs/nb:.0%}) square {sq}/{ns} ({sq/ns:.0%}) expected-in-strata {tot/10000:.1f}  P(low)={le/10000:.4f} P(high)={ge/10000:.4f}")
