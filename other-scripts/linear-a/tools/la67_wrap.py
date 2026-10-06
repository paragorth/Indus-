#!/usr/bin/env python3
"""LA-67: run an earlier loop's script with its checkpoint directory redirected to data/la67_ckpt/<sub>
(fresh tags/seeds are passed on the command line).  usage: la67_wrap.py MODULE SUBDIR SCRIPT [args...]
e.g. la67_wrap.py la21_common la21 la21_run.py LA 1000 100000 15 5 la67
     la67_wrap.py la44_common la44 la44_abc.py BANK 583 TAG robust 1 0"""
import os, sys, runpy, importlib
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
mod, sub, script = sys.argv[1], sys.argv[2], sys.argv[3]
m = importlib.import_module(mod)
m.CK = os.path.join(HERE, '..', 'data', 'la67_ckpt', sub); os.makedirs(m.CK, exist_ok=True)
sys.argv = [os.path.join(HERE, script)] + sys.argv[4:]
runpy.run_path(sys.argv[0], run_name='__main__')
