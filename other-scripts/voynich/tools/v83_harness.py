"""v83 harness: run older loops' own code on an uncertainty-filtered Voynich text.

set_version(v) patches vlib.load_voynich (used by almost every loop loader) and provides derived_lines(name) for the
few loaders that read data/derived/<name>_lines.json directly (v59, v71, v53):
  'legacy'          the old derived files (tools/parse_ivtff.py output), as all loops before v83
  'all'             v83 parser, every word ('<~>' fixed)
  'clean'           only words with no flag (no alt, ?, rare glyph, ligature, apostrophe, uncertain space, questioning or
                    damage comment, damaged page/block)
  'agree'           only words read identically by ZL3b, IT2a and GC2a on the same locus
  'thin-<m>-<seed>' random-thinning control for mode m: from 'all', drop each word at random with the drop rate that m
                    has in the same (transcription, section, line type) stratum
Dropped words are removed from their line (neighbours join), exactly as the legacy loaders already did for '?' words;
the thinning control has the same artefact, so a difference between m and thin-m is due to WHICH words were dropped.
"""
import os, sys, random, collections, functools
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import vlib
import v83_parse as P

_ORIG = vlib.load_voynich
STATE = {'v': 'legacy'}


@functools.lru_cache(maxsize=None)
def _keep_rates(name, mode):
    c = collections.Counter(); k = collections.Counter()
    for r in P.records(name):
        s = (r['illus'], r['ltype'])
        for j in range(len(r['words'])):
            c[s] += 1; k[s] += P.keep_word(r, j, mode)
    return {s: k[s] / c[s] for s in c}


def _records(name, ltypes=None):
    v = STATE['v']
    if v == 'legacy':
        return P.load(name, 'legacy', ltypes=ltypes)
    if v.startswith('thin-'):
        _, mode, seed = v.split('-')
        rates = _keep_rates(name, mode)
        rng = random.Random('%s|%s|%s' % (name, mode, seed))
        out = []
        for r in P.load(name, 'all', ltypes=ltypes):
            p = rates[(r['illus'], r['ltype'])]
            keep = [rng.random() < p for _ in r['words']]
            ws = [w for w, kk in zip(r['words'], keep) if kk]
            if not ws: continue
            rr = dict(r); rr['words'] = ws; rr['uncertain'] = [u for u, kk in zip(r['uncertain'], keep) if kk]
            out.append(rr)
        return out
    return P.load(name, v, ltypes=ltypes)


def patched_load_voynich(name='ZL3b', ltypes=('P',), drop_uncertain=False):
    recs = _records(name, ltypes)
    out = []
    for r in recs:
        ws = r['words']
        if drop_uncertain:
            ws = [w for w, u in zip(ws, r['uncertain']) if not u]
        if not ws: continue
        r = dict(r); r['words'] = ws
        out.append(r)
    return out


def derived_lines(name):
    """replacement for json.load(open(data/derived/<name>_lines.json)) under the current version (all line types)."""
    return _records(name, None)


def set_version(v):
    STATE['v'] = v
    vlib.load_voynich = patched_load_voynich
    # modules that did 'from vlib import load_voynich' are not used by the tests below


VERSIONS_MAIN = ['legacy', 'all', 'clean', 'agree']
THIN = ['thin-clean-1', 'thin-clean-2', 'thin-clean-3', 'thin-agree-1', 'thin-agree-2', 'thin-agree-3']
