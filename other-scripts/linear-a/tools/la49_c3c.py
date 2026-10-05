#!/usr/bin/env python3
"""LA-49 cycle 3c: causal test of the replicated geometry groups. In every saved cycle-2 transformer the
group direction (mean final state of the group's tokens minus the mean of all word tokens) is projected
out of the final state at masked held-out positions; loss change at the group's own tokens vs other word
tokens. Control groups: random word groups of the same size (20 per model). usage: la49_c3c.py CORPUS"""
import sys, os, json, glob, random
os.environ.setdefault('LA49_MAXLEN', '128')
import numpy as np, torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import la49_common as L
torch.set_num_threads(1)
corp = sys.argv[1]
C3B = json.load(open(os.path.join(L.CK, 'c3', 'c3b.json')))
docs = L.corpus(corp); voc = L.Vocab(docs); ch = L.chunks(docs, voc)
groups = {}
for p, coh, n, ws in C3B[corp]['replicated']:
    full = [g for g in C3B[corp]['groups'].values() if set(ws) <= set(g)][0]
    groups['rep%d' % len(groups)] = full
if corp == 'PLANT':
    groups['planted_total'] = L.planted()[1]['TOTAL']
words = sorted(w for w in voc.count if voc.count[w] >= 3 and w in voc.stoi)
rng = random.Random(5)
res = {g: [] for g in groups}; res['random'] = []
for f in sorted(glob.glob(os.path.join(L.CK, 'c2', corp + '_tf_*.json'))):
    r = json.load(open(f)); cfg = r['cfg']; trd = set(r['train_docs'])
    m = L.TF(len(voc.itos), cfg['d'], cfg['h'], cfg['nl'], 0.0)
    m.load_state_dict(torch.load(f.replace('.json', '.pt'))); m.eval()
    pos = [(ci, j) for ci, c in enumerate(ch) for j, t in enumerate(c[2]) if t[0] == 'T' and t[1] in voc.stoi]
    te = [(ci, j) for ci, j in pos if ch[ci][0] not in trd]
    with torch.no_grad():
        Z, W = [], []
        for s in range(0, len(ch), 128):
            part = ch[s:s + 128]
            _, z = m(L.pad([c[1] for c in part]), hidden=True)
            for k, c in enumerate(part):
                for j, t in enumerate(c[2]):
                    if t[0] == 'T':
                        Z.append(z[k, j].numpy()); W.append(t[1])
        Z = np.array(Z); mu = Z.mean(0)
        # masked held-out states
        Zq, Y, Wq = [], [], []
        for s in range(0, len(te), 256):
            part = te[s:s + 256]
            xs = []
            for ci, j in part:
                ids = list(ch[ci][1]); ids[j] = 1; xs.append(ids)
            _, z = m(L.pad(xs), hidden=True)
            idx = torch.arange(len(part)); cols = torch.tensor([j for _, j in part])
            Zq.append(z[idx, cols]); Y += [ch[ci][1][j] for ci, j in part]; Wq += [ch[ci][2][j][1] for ci, j in part]
        Zq = torch.cat(Zq); Y = torch.tensor(Y)
        base = F.cross_entropy(m.out(Zq), Y, reduction='none').numpy()

        def effect(group):
            gs = set(group)
            sel = np.array([w in gs for w in W])
            if sel.sum() < 3:
                return None
            u = Z[sel].mean(0) - mu; u /= np.linalg.norm(u) + 1e-9
            ut = torch.tensor(u, dtype=torch.float32)
            z2 = Zq - (Zq @ ut)[:, None] * ut[None]
            d = F.cross_entropy(m.out(z2), Y, reduction='none').numpy() - base
            ins = np.array([w in gs for w in Wq])
            if ins.sum() == 0:
                return None
            return float(d[ins].mean()), float(d[~ins].mean())
        for g, ws in groups.items():
            e = effect(ws)
            if e: res[g].append(e)
        for _ in range(20):
            k = len(next(iter(groups.values())))
            e = effect(rng.sample(words, min(k, len(words))))
            if e: res['random'].append(e)
out = {}
for g, v in res.items():
    v = np.array(v)
    spec = v[:, 0] - v[:, 1]
    out[g] = {'n': len(v), 'd_in': round(float(v[:, 0].mean()), 3), 'd_out': round(float(v[:, 1].mean()), 3),
              'spec_mean': round(float(spec.mean()), 3), 'frac_models_spec>0.1': round(float(np.mean(spec > 0.1)), 2)}
rnd = np.array(res['random']); rspec = rnd[:, 0] - rnd[:, 1]
for g in groups:
    v = np.array(res[g]); out[g]['p_vs_random'] = round(float(np.mean(rspec >= (v[:, 0] - v[:, 1]).mean())), 4)
    out[g]['members'] = groups[g][:12]
print(corp, json.dumps(out, ensure_ascii=False))
json.dump(out, open(os.path.join(L.CK, 'c3', 'c3c_%s.json' % corp), 'w'), ensure_ascii=False)
