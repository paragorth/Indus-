"""Loop 75: run an ORIGINAL analysis script unchanged, with the canonical corpus swapped for v2 (or left as v1),
and every file it writes redirected into an output folder so the published outputs are never touched.

Usage: python3 tools/dark_loop75_run.py v1|v2 OUTDIR path/to/script.py [script args...]
  * any read of a path ending 'merged-corpus-canonical.json' is served from merged-corpus-canonical.v2.json (v2);
  * any write ('w','a','x' modes) to a path is redirected to OUTDIR/<basename>;
  * a read of a path whose basename already exists in OUTDIR is served from OUTDIR (so multi-cycle scripts
    that read their own earlier outputs read this run's outputs, not the published ones).
Fields 'id' and 'v1' in the v2 file are extra and ignored by every script tested.
"""
import builtins, io, os, sys, runpy

ROOT = '/home/user/Indus-/'
variant, outdir, script = sys.argv[1], os.path.abspath(sys.argv[2]), sys.argv[3]
os.makedirs(outdir, exist_ok=True)
_open = builtins.open
WRITTEN = set()


def _redir(file, mode='r', *a, **k):
    if isinstance(file, (str, bytes, os.PathLike)):
        p = os.fspath(file)
        if isinstance(p, bytes): p = p.decode()
        base = os.path.basename(p)
        if any(c in mode for c in 'wax+'):
            q = os.path.join(outdir, base); WRITTEN.add(base)
            return _open(q, mode, *a, **k)
        if p.endswith('merged-corpus-canonical.json') and variant == 'v2':
            return _open(ROOT + 'data/derived/merged-corpus-canonical.v2.json', mode, *a, **k)
        q = os.path.join(outdir, base)
        if os.path.exists(q) and os.path.abspath(p) != q and not p.endswith('merged-corpus-canonical.json'):
            return _open(q, mode, *a, **k)
    return _open(file, mode, *a, **k)


builtins.open = _redir
io.open = _redir
try:
    import numpy as np
    _save = np.save
    np.save = lambda f, *a, **k: _save(os.path.join(outdir, os.path.basename(os.fspath(f))) if isinstance(f, (str, os.PathLike)) else f, *a, **k)
except Exception:
    pass
sys.argv = [script] + sys.argv[4:]
sys.path.insert(0, os.path.dirname(os.path.abspath(script)))
sys.path.insert(0, ROOT + 'tools')
os.chdir(ROOT)
runpy.run_path(script, run_name='__main__')
