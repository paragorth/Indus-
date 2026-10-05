"""v51 cycle 3 analysis: sections / hands, and the heard-language shift.

(1) Speech gain of each Voynich section and hand over its own unit shuffle
    (paired by mapping, composite z across all corpora in the mapping).
(2) Heard-language shift: per mapping, log P(language | text in order) minus
    log P(language | same units shuffled), for all 99 languages of the listener.
    Positive control: does a Latin/German/Czech/Italian/Hebrew/Esperanto text in
    real order shift the listener toward its OWN language (rank of own language
    among 99)? Only if that works is the Voynich shift worth reading.
"""
import sys, os, json
from collections import defaultdict
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v51_lib as V
from v51_analyze import load, table, composite, paired

LANGCODE = {'la': 'la', 'de': 'de', 'cs': 'cs', 'it': 'it', 'he': 'he', 'eo': None}

def main(tag):
    rows = load(tag)
    cs, ms, A = table(rows)
    S, Zs = composite(A)
    ci = {c: i for i, c in enumerate(cs)}
    out = [f'# {tag}: {len(ms)} mappings x {len(cs)} corpora', 'SPEECH GAIN over own shuffle (composite diff, paired z, win; per-metric z)']
    for c in cs:
        if '~' in c or c + '~shuf' not in ci:
            continue
        a, b = ci[c], ci[c + '~shuf']
        d, z, w = paired(S[a], S[b])
        mz = [paired(Zs[a, :, j], Zs[b, :, j])[1] for j in range(len(V.METRICS))]
        out.append(f'{c}: {d:+.3f} z {z:+.1f} win {w:.2f} | ' + ' '.join(f'{k}:{v:+.1f}' for k, v in zip(V.METRICS, mz)))
    # heard-language shift
    from transformers import WhisperForConditionalGeneration  # only for language names
    names = None
    try:
        from transformers import GenerationConfig
        gc = GenerationConfig.from_pretrained('openai/whisper-tiny')
        names = [k.strip('<|>') for k, v in sorted(gc.lang_to_id.items(), key=lambda kv: kv[1])]
    except Exception:
        pass
    LP = defaultdict(dict)
    for r in rows:
        if 'lid' in r:
            LP[r['corpus']].setdefault(r['m'], []).append(r['lid'])
    out.append('\nHEARD-LANGUAGE SHIFT (text in order minus its shuffle; z over mappings; own-language rank among 99)')
    for c in sorted(LP):
        if '~' in c or c + '~shuf' not in LP:
            continue
        mm = sorted(set(LP[c]) & set(LP[c + '~shuf']))
        D = np.array([np.mean(LP[c][m], 0) - np.mean(LP[c + '~shuf'][m], 0) for m in mm])
        z = D.mean(0) / (D.std(0, ddof=1) / np.sqrt(len(mm)) + 1e-9)
        o = np.argsort(-z)
        line = f'{c}: up ' + ' '.join(f'{names[i]}{z[i]:+.1f}' for i in o[:5]) + ' | down ' + ' '.join(f'{names[i]}{z[i]:+.1f}' for i in o[-3:])
        code = LANGCODE.get(c)
        if code and names and code in names:
            k = names.index(code)
            line += f' | own {code}: z {z[k]:+.1f}, rank {int(np.where(o == k)[0][0]) + 1}/99'
        out.append(line)
    # similarity of shift profiles: Voynich sections vs each language's shift profile
    prof = {}
    for c in sorted(LP):
        if '~' in c or c + '~shuf' not in LP:
            continue
        mm = sorted(set(LP[c]) & set(LP[c + '~shuf']))
        prof[c] = np.mean([np.mean(LP[c][m], 0) - np.mean(LP[c + '~shuf'][m], 0) for m in mm], 0)
    langs = [c for c in prof if c in LANGCODE]
    out.append('\nSHIFT-PROFILE CORRELATION (Pearson over 99 languages) with each control language')
    for c in prof:
        out.append(f'{c}: ' + ' '.join(f'{l}:{np.corrcoef(prof[c], prof[l])[0, 1]:+.2f}' for l in langs if l != c))
    return '\n'.join(out)

if __name__ == '__main__':
    print(main(sys.argv[1]))
