"""pe74 harness: run earlier loops' own code on a restoration-filtered corpus.

activate(mode) must be called before the older libraries are imported:
  * sets PE_CORPUS_MODE (read by common.load),
  * redirects every open() of data/pe_corpus.json to the filtered corpus (for libraries that read the
    file directly),
redirect_ck(module, name, copy=[...]) points a library's checkpoint directory (module.CK) to a fresh
per-mode directory under data/pe74_ckpt/<mode>/<name>, copying the listed non-PE cache files (proto-
cuneiform, Ur III) so that only PE-derived caches are rebuilt.
"""
import os, sys, builtins, shutil

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
DATA = os.path.abspath(os.path.join(HERE, '..', 'data'))
ROOT = os.path.join(DATA, 'pe74_ckpt')
_ORIG_OPEN = builtins.open
MODE = None


def activate(mode):
    global MODE
    MODE = mode
    os.environ['PE_CORPUS_MODE'] = mode
    import pe74_parse
    target = pe74_parse.corpus_file(mode)
    src = os.path.join(DATA, 'pe_corpus.json')

    def _open(file, *a, **k):
        try:
            if isinstance(file, str) and os.path.abspath(file) == src:
                file = target
        except Exception:
            pass
        return _ORIG_OPEN(file, *a, **k)
    builtins.open = _open
    d = os.path.join(ROOT, mode)
    os.makedirs(d, exist_ok=True)
    return d


def redirect_ck(module, name, copy=(), attr='CK'):
    old = getattr(module, attr)
    new = os.path.join(ROOT, MODE, name)
    os.makedirs(new, exist_ok=True)
    for f in copy:
        s, t = os.path.join(old, f), os.path.join(new, f)
        if os.path.exists(s) and not os.path.exists(t):
            shutil.copy(s, t)
    setattr(module, attr, new)
    return new
