"""calibrate the v53_check kill: the planted target's best read-back programs on planted half B vs its glyph-trigram
resynthesis and word-type relabelling (if the resynthesis also scores, the check has no power)."""
import sys, os, json, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import importlib.util
here = os.path.dirname(os.path.abspath(__file__))
sys.argv = ['v53_check.py']
src = open(os.path.join(here, 'v53_check.py')).read()
src = src.split("R = json.load(open(os.path.join(CK, 'rb_voynich.json')))")[0]
src = src.replace("sys.argv = ['v53_readback.py', 'voynich', '1', '1']", "sys.argv = ['v53_readback.py', 'planted', '1', '1']")
exec(compile(src, 'v53_check_head', 'exec'))
R = json.load(open(os.path.join(CK, 'rb_planted.json')))
progs = sorted(((t['heldout'].get('score', 0), r['corpus'], t['prog']) for r in R['results'] for t in r['top'][:1]), key=lambda x: -x[0])[:4]
out = []
for hs, c, p in progs:
    for tn, T in {'planted_B': B, 'planted_B_trigram': trigram_resynth(B), 'planted_B_relabel': relabel(B)}.items():
        env = Env(T, CORP, offset=7919)
        sc = rb.rb_score(p, env, T, rng_seed=5)
        out.append({'prog_corpus': c, 'target': tn, **{k: sc.get(k) for k in ('score', 'hit', 'null', 'cov', 'n')}})
        print(c, tn, {k: round(v, 3) if isinstance(v, float) else v for k, v in sc.items() if k in ('score', 'hit', 'null', 'cov', 'n')}, flush=True)
json.dump(out, open(os.path.join(CK, 'check2.json'), 'w'))
