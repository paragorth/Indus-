"""Cycle 2: reading features (text length, frame slots, closer identity, sign families, numeral series and values,
object types) vs geographic gradients. Same engine as cycle 1. Also a seals-only pass (object-type confound) and
an IM77 direction check for the top features on the 5 IM77 sites."""
import sys, collections, csv, numpy as np
sys.path.insert(0,'/home/user/Indus-/data/derived/dark'); from loop6_engine import *

out=open(ROOT+'data/derived/dark/loop6_c2_features.txt','w')
def P(*a):
    print(*a); print(*a,file=out)

VAL={}
for w in range(3,8): VAL[w]=w
for w,v in zip(range(12,21),range(2,11)): VAL[w]=v
for w,v in zip(range(25,30),range(5,10)): VAL[w]=v
for w,v in zip(range(32,40),range(2,10)): VAL[w]=v
VAL[55]=12; VAL[56]=24
SHORT=set(range(3,8))|set(range(12,21))|set(range(25,30)); TALL=set(range(32,40))
FISH={220,240,235,233,231,226}; TREE={390,405,407}; OPEN={817,861,820}; OPEN2={920,692}
PERSON={90,91}; CLOSERS_ALT={520,595,151,156}
dos=json.load(open(ROOT+'data/derived/sign-dossiers-top200.json'))
KW={'fam_fish':['fish'],'fam_jar':['jar'],'fam_tree':['tree'],'fam_leaf':['leaf'],'fam_person':['person'],
    'fam_U':['u with','u '],'fam_triangle':['triangle'],'fam_box':['rectangle','square','box'],'fam_pitchfork':['pitchfork','pitckfork'],
    'fam_stroke':['stroke'],'fam_circle':['circle','oval'],'fam_comb':['comb'],'fam_arrow':['arrow'],'fam_crescent':['crescent']}
FAM={k:{d['glyph'] for d in dos if any(w in d['shape'].lower() for w in ws)} for k,ws in KW.items()}

def feats_for(T):
    f={}
    f['len_mean']=lambda t: len(t['seq'])
    f['len_ge5']=lambda t: len(t['seq'])>=5
    f['len_eq1']=lambda t: len(t['seq'])==1
    f['opener_initial']=lambda t: t['seq'][0] in OPEN|OPEN2
    f['opener_any']=lambda t: bool(set(t['seq'])&OPEN)
    f['marker_W2']=lambda t: 2 in t['seq']
    f['marker_W60']=lambda t: 60 in t['seq']
    f['jar_any']=lambda t: 740 in t['seq']
    f['jar_final']=lambda t: t['seq'][-1]==740
    f['jar_then_suffix']=lambda t: 400 in t['seq'] or (len(t['seq'])>=2 and t['seq'][-2]==740)
    f['suffix_W400']=lambda t: 400 in t['seq']
    f['person_W90']=lambda t: 90 in t['seq']
    f['person_final']=lambda t: t['seq'][-1]==90
    f['twins_W91']=lambda t: 91 in t['seq']
    f['arrow_closer_final']=lambda t: t['seq'][-1]==520
    f['alt_closer_final']=lambda t: t['seq'][-1] in {162,169,15,254,12,595} or t['seq'][-1] in OPEN
    f['numeral_any']=lambda t: bool(set(t['seq'])&set(VAL))
    f['numeral_short']=lambda t: bool(set(t['seq'])&SHORT)
    f['numeral_tall']=lambda t: bool(set(t['seq'])&TALL)
    f['W1_any']=lambda t: 1 in t['seq']
    for v in (2,3,4,5,6,7,8,12):
        f[f'value_{v}']=(lambda t,v=v: any(VAL.get(a)==v for a in t['seq']))
    f['doubled_adjacent']=lambda t: any(t['seq'][i]==t['seq'][i+1] for i in range(len(t['seq'])-1))
    f['repeat_any']=lambda t: len(set(t['seq']))<len(t['seq'])
    f['sign_initial_W817']=lambda t: t['seq'][0]==817
    f['sign_initial_W820']=lambda t: t['seq'][0]==820
    f['sign_initial_W861']=lambda t: t['seq'][0]==861
    for k,S in FAM.items(): f[k]=(lambda t,S=S: bool(set(t['seq'])&S))
    # closer identity: X before final jar, top closers
    pre=collections.Counter(t['seq'][-2] for t in T if len(t['seq'])>=2 and t['seq'][-1]==740)
    for x,n in pre.most_common(10): f[f'closer_{x}-740']=(lambda t,x=x: len(t['seq'])>=2 and t['seq'][-1]==740 and t['seq'][-2]==x)
    f['type_seal']=lambda t: t['type'].startswith('SEAL')
    f['type_tablet']=lambda t: t['type'].startswith('TAB')
    f['type_pot']=lambda t: t['type'].startswith('POT')
    f['type_tag']=lambda t: t['type'].startswith('TAG')
    f['type_bangle']=lambda t: t['type'].startswith('BNGL')
    return f

def site_rates_nan(by,f): return {s:float(np.nanmean([f(t) for t in v])) for s,v in by.items()}

ALL=[]
for label,types in [('all objects',None),('seals only',('SEAL',))]:
    T=load_texts('seq_raw',types=types); by=site_table(T,5)
    feats=feats_for(T)
    P(f'\n=== {label}: {len(T)} texts, {len(by)} sites, {len(feats)} features, {len(feats)*(len(GEOVARS)+1)} arrows')
    # nan-safe for value_mean: replace with nanmean-based site rates by wrapping
    import loop6_engine as E; E.site_rates=site_rates_nan
    res=run_arrows(by,feats); res.sort(key=lambda r:r['p'])
    for r in res: r['pass']=label
    ALL+=res
    P(f'arrows {len(res)}; min p {res[0]["p"]:.4f}; q<0.10: {sum(r["q"]<0.10 for r in res)}; p<0.01: {sum(r["p"]<0.01 for r in res)} (expected {0.01*len(res):.1f})')
    P('top 20:')
    for r in res[:20]: P('  '+fmt(r))
    top=res[:12]
    res2=run_arrows(by,{r['feature']:feats[r['feature']] for r in top},drop=('Mohenjo-daro','Harappa'))
    d2={(r['feature'],r['var']):r for r in res2}
    P('re-test of top 12 without MD + HP:')
    for r in top:
        r2=d2.get((r['feature'],r['var']))
        if r2: P(f"  {r['feature']:>22} ~ {r['var']:<24} full={r['stat']:+.3f} p={r['p']:.4f} | dropMDHP={r2['stat']:+.3f} p={r2['p']:.4f}")
    for r in res[:4]:
        rates=site_rates_nan(by,feats[r['feature']])
        P(f"  per-site {r['feature']} vs {r['var']}:")
        key=(lambda s:COORDS[s][r['var']]) if r['var']!='river' else (lambda s:COORDS[s]['river'])
        for s in sorted(by,key=key): P(f"     {s:<22} n={len(by[s]):>4} rate={rates[s]:.3f} {r['var']}={COORDS[s][r['var']]}")

# joint FDR over both passes
qs=bh([r['p'] for r in ALL])
P(f'\nJOINT FDR over {len(ALL)} arrows (both passes): min q = {qs.min():.3f}; arrows q<0.10: {(qs<0.10).sum()}')

# IM77 direction check on the 5 named sites for frame features (bridge: M342 jar, M267/391 opener, M99 marker, M1 person, M176 suffix)
P('leaf family members: '+str(sorted(FAM['fam_leaf']))+' pitchfork: '+str(sorted(FAM['fam_pitchfork']))+' box: '+str(sorted(FAM['fam_box'])))
P('\nIM77 check (5 named sites): per-site rates of jar-final, opener-initial, marker M99, person M1, length>=5')
rows=list(csv.DictReader(open(ROOT+'data/im77/im77_corpus_lines.csv')))
MAP={'Mohenjodaro':'Mohenjo-daro','Harappa':'Harappa','Lothal':'Lothal','Kalibangan':'Kalibangan','Chanhudaro':'Chanhu-daro'}
txt=collections.defaultdict(list); seen=set(); sides=collections.defaultdict(list)
for r in rows: sides[(r['text_no'],r['side'])].append(r)
for k,ls in sides.items():
    ls=sorted(ls,key=lambda r:int(r['line'])); s=[]
    for l in ls: s+=l['signs_clean'].split()
    s=[int(x) for x in s if x.isdigit() and x not in ('0','000')]
    site=ls[0]['site']
    if not s or site not in MAP or (site,tuple(s)) in seen: continue
    seen.add((site,tuple(s))); txt[MAP[site]].append(s)
P('IM77 texts per site: '+str({k:len(v) for k,v in txt.items()}))
IMF={'jar_final':lambda s:s[-1]==342,'opener_initial':lambda s:s[0] in (267,391),'marker_M99':lambda s:99 in s,'person_M1':lambda s:1 in s,'len_ge5':lambda s:len(s)>=5,'fish_M59':lambda s:59 in s}
for k,f in IMF.items():
    xs=[];ys=[]
    line=[]
    for s in MAP.values():
        v=np.mean([f(t) for t in txt[s]]); xs.append(v); ys.append(COORDS[s]['dist_coast_harappan_km']); line.append(f'{s}={v:.3f}')
    obs,p,k_=exact_or_mc_p(xs,ys,rho)
    P(f'  {k:<16} rho(dist_coast)={obs:+.2f} exact p={p:.3f} (n=5, {k_} perms) | '+' '.join(line))
out.close()
