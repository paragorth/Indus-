#!/usr/bin/env python3
"""la71 version runner: run an existing Linear A script on one corpus version without touching the repo.

usage: python3 la71_run.py VERSION script.py [args...]      VERSION = all | rd | read

- reads of data/corpus.json are served from data/la71_ckpt/versions/corpus_VERSION.json
  (built by la71_parse.version; placeholders {'t':'unk','v':'#R'} keep positions);
- every file the script writes inside the repo goes to data/la71_ckpt/redir/VERSION/<repo path>,
  and later reads of that path are served from there (so caches rebuild per version);
- caches derived from corpus.json (CACHES) are hidden unless rebuilt in the redirect tree;
- LA71_VERSION is set in the environment (la60_common.load_la keeps '#R' placeholders as 'X' tokens).
"""
import os, sys, json, builtins, io, runpy

HERE = os.path.dirname(os.path.abspath(__file__))
LA = os.path.abspath(os.path.join(HERE, '..'))
REPO = os.path.abspath(os.path.join(LA, '..', '..'))
DATA = os.path.join(LA, 'data')
CK = os.path.join(DATA, 'la71_ckpt')
CACHES = {os.path.join(DATA, p) for p in (
    'la60_ckpt/la_docs.json', 'la22_ckpt/docs.json', 'la47_ckpt/layout.json')}


def prepare(ver):
    sys.path.insert(0, HERE)
    import la71_parse
    vd = os.path.join(CK, 'versions'); os.makedirs(vd, exist_ok=True)
    fn = os.path.join(vd, 'corpus_%s.json' % ver)
    src = os.path.join(DATA, 'corpus_ra.json')
    if not os.path.exists(fn) or os.path.getmtime(fn) < os.path.getmtime(src):
        json.dump(la71_parse.load(ver), open(fn, 'w'), ensure_ascii=False)
    return fn


def install(ver, extra_caches=()):
    vfile = prepare(ver)
    redir = os.path.join(CK, 'redir', ver)
    caches = CACHES | {os.path.abspath(c) for c in extra_caches}
    real_open = builtins.open; real_exists = os.path.exists; real_isfile = os.path.isfile
    real_makedirs = os.makedirs
    corpus = os.path.join(DATA, 'corpus.json')

    def mapped(path, write):
        try:
            p = os.path.abspath(os.fspath(path))
        except TypeError:
            return path
        if p == corpus:
            return vfile
        if not p.startswith(REPO + os.sep) or p.startswith(CK + os.sep) or '__pycache__' in p:
            return path
        r = os.path.join(redir, os.path.relpath(p, REPO))
        if write:
            real_makedirs(os.path.dirname(r), exist_ok=True)
            return r
        if real_exists(r):
            return r
        if p in caches:
            return r  # hidden: does not exist until the script rebuilds it
        return path

    def nopen(file, mode='r', *a, **k):
        if isinstance(file, int):
            return real_open(file, mode, *a, **k)
        w = any(c in mode for c in 'wax+')
        return real_open(mapped(file, w), mode, *a, **k)

    def nexists(path):
        return real_exists(mapped(path, False))

    def nisfile(path):
        return real_isfile(mapped(path, False))

    def nmakedirs(path, *a, **k):
        m = mapped(path, True) if os.path.abspath(os.fspath(path)).startswith(REPO + os.sep) else path
        if m != path:
            return real_makedirs(os.path.join(redir, os.path.relpath(os.path.abspath(path), REPO)), *a, **k)
        return real_makedirs(path, *a, **k)

    builtins.open = nopen; io.open = nopen
    os.path.exists = nexists; os.path.isfile = nisfile; os.makedirs = nmakedirs
    os.environ['LA71_VERSION'] = ver
    os.environ['LA71_WRAPPED'] = '1'


def main():
    ver, script = sys.argv[1], sys.argv[2]
    install(ver)
    sys.argv = [script] + sys.argv[3:]
    sys.path.insert(0, os.path.dirname(os.path.abspath(script)))
    runpy.run_path(script, run_name='__main__')


if __name__ == '__main__':
    main()
