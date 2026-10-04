#!/usr/bin/env python3
"""LA-10 report: summarise run_*.json files.  python3 la10_report.py [glob]
Per run: best synthetic -log-likelihood (lower is better), the rule genes of the best genome with the spread over the
final top-20, and the ablation table (score increase when a rule is set to its null; '_noise' = same genome, other seeds).
A rule counts as REQUIRED when its ablation cost > max(15, 3 * |noise|)."""
import sys, os, json, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import la10_common as L
D = os.path.join(L.C.LAD, 'la10')
RULE_SHOW = ['onset0_init', 'onset0_med', 'cluster', 'coda_med', 'coda_obs', 'coda_fin', 'root_len', 'harm_copy', 'harm_fb',
             'ocp_id', 'ocp_place', 'pre_p', 'pre_n', 'suf_p', 'suf_n', 'aff_2syl', 'reuse', 'fv_str']
AB = ['harmony', 'harm_copy', 'harm_fb', 'ocp', 'ocp_id', 'ocp_place', 'clusters', 'codas', 'hiatus', 'prefixing', 'suffixing',
      'reuse', 'final_vowel', 'swap_pre_suf']

def q(v, p):
    v = sorted(v); return v[min(len(v) - 1, int(p * len(v)))]

def summarise(path):
    d = json.load(open(path)); b = d['best']; out = []
    out.append(f"== {os.path.basename(path)}: n {d['n']} evals {d['nevals']} best {d['best_score']:.1f}" +
               (f"  truth {d['truth_score']:.1f}" if 'truth_score' in d else ''))
    rows = []
    for g in RULE_SHOW:
        vals = [t[g] for t in d['top20']]
        s = f"{g} {b[g]:.2f} [{q(vals, 0.1):.2f},{q(vals, 0.9):.2f}]"
        if 'truth' in d: s += f" (truth {d['truth'][g]:.2f})"
        rows.append(s)
    out.append('  rules: ' + '; '.join(rows))
    inv = {k: round(v, 2) for k, v in b.items() if k.startswith(('cw_', 'vw_', 'fw_'))}
    out.append('  inventory log-weights: ' + ' '.join(f'{k}={v}' for k, v in inv.items()))
    ab = d['ablation']; thr = max(15.0, 3 * abs(ab['_noise']))
    out.append(f"  ablation (noise {ab['_noise']:+.1f}, threshold {thr:.1f}): " +
               ' '.join(f"{k} {ab[k]:+.1f}{'*' if ab[k] > thr else ''}" for k in AB))
    if 'truth_ablation' in d:
        ta = d['truth_ablation']; tthr = max(15.0, 3 * abs(ta['_noise']))
        out.append(f"  truth ablation (noise {ta['_noise']:+.1f}): " + ' '.join(f"{k} {ta[k]:+.1f}{'*' if ta[k] > tthr else ''}" for k in AB))
    fp = d['best_fp']; tf = d['target_fp']; sd = d['target_sd']
    out.append('  fit (target / bred / z): ' + ' '.join(f"{k} {tf[k]:.3f}/{fp[k]:.3f}/{(fp[k] - tf[k]) / sd[k]:+.1f}" for k in sorted(L.CORE, key=L.STAT_NAMES.index)))
    return '\n'.join(out), d

if __name__ == '__main__':
    pat = sys.argv[1] if len(sys.argv) > 1 else 'run_*.json'
    for p in sorted(glob.glob(os.path.join(D, pat))):
        print(summarise(p)[0])
