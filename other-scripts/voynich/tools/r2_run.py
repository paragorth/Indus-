"""R2 runner: run the C program search over a list of corpora with at most 2 workers.
Each job resumes from its checkpoint; finished jobs (.out present) are skipped.
usage: python3 r2_run.py TAG MODE corpus1 corpus2 ...   (budget set per corpus size)"""
import os, sys, subprocess, json, time
from concurrent.futures import ThreadPoolExecutor

R2 = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'results', 'r2')
BIN = os.path.join(R2, 'bin', 'r2_search')
SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'r2_search.c')


def budget(name):
    N = json.load(open(os.path.join(R2, name + '.json')))['N']
    if N > 100000: return 50000, 50000
    if N > 30000: return 150000, 150000
    return 250000, 250000


def job(tag, mode, name, seed):
    pre = os.path.join(R2, 'runs', f'{tag}_{name}_s{seed}')
    if os.path.exists(pre + '.out'):
        return name, 'skip'
    er, ee = budget(name)
    t = time.time()
    with open(pre + '.log', 'a') as lg:
        subprocess.run([BIN, os.path.join(R2, name + '.bin'), pre, str(seed), str(er), str(ee), str(mode)],
                       stderr=lg, check=True)
    return name, f'{time.time() - t:.0f}s'


if __name__ == '__main__':
    os.makedirs(os.path.join(R2, 'runs'), exist_ok=True)
    os.makedirs(os.path.dirname(BIN), exist_ok=True)
    if not os.path.exists(BIN) or os.path.getmtime(BIN) < os.path.getmtime(SRC):
        subprocess.run(['gcc', '-O3', '-march=native', '-o', BIN, SRC, '-lm'], check=True)
    tag, mode = sys.argv[1], int(sys.argv[2])
    names = sys.argv[3:]
    with ThreadPoolExecutor(2) as ex:
        for r in ex.map(lambda n: job(tag, mode, n, 1), names):
            print(*r, flush=True)
