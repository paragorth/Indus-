"""v51 LET A SPEECH MODEL LISTEN: render texts as sound under random phone
mappings, score each rendering with speech models trained only on human speech.

Pieces:
  corpora()          -> dict name -> list of words (each word = list of unit symbols)
  random_mapping(rng)-> phone inventory indexed by unit frequency rank + prosody rules
  render(words, mapping, rng) -> mono float32 audio at 16 kHz
  Scorer.score(audio) -> dict of model scores

Model (downloaded to the scratchpad HF cache; never committed):
  whisper-tiny (multilingual, 99-language speech): no-speech probability and
      language-ID entropy at the start-of-transcript step.
"""
import os, re, json, math, random, sys, unicodedata
from collections import Counter, defaultdict
import numpy as np
from scipy.signal import lfilter

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, 'data')
CKPT = os.path.join(DATA, 'v51_ckpt')
os.makedirs(CKPT, exist_ok=True)
sys.path.insert(0, HERE)
import vlib as L
import gen as G

SR = 16000
SCRATCH = os.environ.get('V51_SCRATCH', '/tmp/claude-0/-home-user-Indus-/874df4c7-80d6-5f08-b42c-eea96a214079/scratchpad/v51')
os.environ.setdefault('HF_HOME', os.path.join(SCRATCH, 'hf'))

# ------------------------------------------------------------------ corpora
NWORDS = 20000

def _letters(txt, keep):
    txt = unicodedata.normalize('NFC', txt.lower())
    ws = re.split(r'[^' + keep + r']+', txt)
    return [list(w) for w in ws if w]

LAT = 'a-zà-ÿāēīōūăĕĭŏŭœæ'
def _plain(name, keep=LAT, skip=0):
    t = open(os.path.join(DATA, 'plain', name), encoding='utf8').read()
    ws = _letters(t, keep)
    return ws[skip:skip + NWORDS]

def voy_words(name='ZL3b', filt=None):
    out = []
    for r in L.load_voynich(name, drop_uncertain=True):
        if filt and not filt(r):
            continue
        for w in r['words']:
            if '?' in w or '*' in w:
                continue
            g = L.glyphs(w)
            if g:
                out.append(g)
    return out

def as_lines(words, k=8):
    return [{'words': [''.join(w) for w in words[i:i + k]]} for i in range(0, len(words), k)]

def _units_from_lines(lines, inv=None):
    out = []
    for Ln in lines:
        for w in Ln['words']:
            out.append(list(w))
    return out

def null_shuffle(words, seed=1):
    rng = random.Random(seed)
    u = [c for w in words for c in w]; rng.shuffle(u)
    out, i = [], 0
    for w in words:
        out.append(u[i:i + len(w)]); i += len(w)
    return out

def null_markov(words, order=1, seed=1):
    rng = random.Random(seed)
    tr = defaultdict(Counter)
    for w in words:
        s = ['^'] * order + list(w) + ['_']
        for i in range(order, len(s)):
            tr[tuple(s[i - order:i])][s[i]] += 1
    tabs = {k: (list(v.keys()), list(v.values())) for k, v in tr.items()}
    out = []
    for _ in words:
        ctx, w = tuple(['^'] * order), []
        while True:
            ks, vs = tabs[ctx]
            c = rng.choices(ks, vs)[0]
            if c == '_' or len(w) > 20:
                break
            w.append(c); ctx = tuple((list(ctx) + [c])[-order:])
        out.append(w if w else list(rng.choice(words)))
    return out

def _gen_via_lines(words, fn, seed=1):
    # map units to single chars so gen.py's string generators work
    inv = sorted(set(c for w in words for c in w))
    enc = {c: chr(0x4e00 + i) for i, c in enumerate(inv)}
    dec = {v: k for k, v in enc.items()}
    lines = [{'words': [''.join(enc[c] for c in w) for w in words[i:i + 8]]} for i in range(0, len(words), 8)]
    out = fn(lines, seed=seed)
    return [[dec[c] for c in w] for Ln in out for w in Ln['words']]

def gibberish(words, seed=1):
    """Uniform random units (Voynich inventory), real word lengths."""
    rng = random.Random(seed)
    inv = sorted(set(c for w in words for c in w))
    return [[rng.choice(inv) for _ in w] for w in words]

def corpora(include_nulls=True):
    C = {}
    vz = voy_words('ZL3b')
    C['V-ZL'] = vz
    C['V-IT'] = voy_words('IT2a')
    C['V-A'] = voy_words('ZL3b', lambda r: r.get('lang') == 'A')
    C['V-B'] = voy_words('ZL3b', lambda r: r.get('lang') == 'B')
    C['la'] = _plain('la.txt', skip=300)
    C['de'] = [list(w) for Ln in L.load_ref('German-Kafka', max_words=NWORDS + 50, skip_frac=0.02) for w in Ln['words']][:NWORDS]
    C['cs'] = _plain('cs.txt', keep=LAT + 'áčďéěíňóřšťúůýž')
    C['it'] = [list(w) for Ln in L.load_ref('Italian-Manzoni', max_words=NWORDS + 50, skip_frac=0.05) for w in Ln['words']][:NWORDS]
    C['he'] = _plain('he.txt', keep='א-ת')
    C['eo'] = _plain('eo.txt', keep=LAT + 'ĉĝĥĵŝŭ')
    if include_nulls:
        for k in ['V-ZL', 'la', 'de', 'cs', 'it', 'he', 'eo']:
            C[k + '~shuf'] = null_shuffle(C[k], seed=11)
            C[k + '~mk1'] = null_markov(C[k], 1, seed=12)
        C['G-grille'] = _gen_via_lines(vz, G.table_grille, seed=13)
        C['G-selfcit'] = _gen_via_lines(vz, G.self_citation, seed=14)
        C['G-gibber'] = gibberish(vz, seed=15)
        C['G-mk2'] = null_markov(vz, 2, seed=16)
    return C

def rank_index(words):
    cnt = Counter(c for w in words for c in w)
    return {c: i for i, (c, _) in enumerate(cnt.most_common())}

# ------------------------------------------------------------------ mapping
NSLOT = 64

def random_mapping(rng, pv=None):
    """Phone per frequency-rank slot, plus prosody rules. rng: np.random.Generator."""
    if pv is None:
        pv = rng.uniform(0.2, 0.5)
    slots = []
    for r in range(NSLOT):
        if rng.random() < pv:
            cls = 'V'
        else:
            cls = rng.choice(['N', 'L', 'F', 'S', 'S'])
        ph = {'cls': cls}
        if cls == 'V':
            f1 = rng.uniform(280, 820); f2 = rng.uniform(max(800, f1 + 300), 2400)
            ph.update(F=(f1, f2, rng.uniform(2400, 3100)), dur=rng.uniform(85, 140), amp=1.0)
        elif cls == 'N':
            ph.update(F=(rng.uniform(220, 300), rng.uniform(900, 2000), 2700), dur=rng.uniform(55, 85), amp=0.35)
        elif cls == 'L':
            ph.update(F=(rng.uniform(300, 480), rng.uniform(900, 1700), rng.uniform(2200, 2900)), dur=rng.uniform(45, 75), amp=0.55)
        elif cls == 'F':
            ph.update(fc=rng.uniform(1800, 6500), bw=rng.uniform(600, 2500), dur=rng.uniform(70, 115), amp=rng.uniform(0.15, 0.35), voiced=bool(rng.random() < 0.4))
        else:
            ph.update(fc=rng.uniform(800, 5000), bw=rng.uniform(800, 3000), dur=rng.uniform(45, 70), amp=rng.uniform(0.3, 0.6), voiced=bool(rng.random() < 0.5))
        slots.append(ph)
    if not any(s['cls'] == 'V' for s in slots[:4]):
        s = slots[int(rng.integers(0, 4))]
        s.clear(); s.update(cls='V', F=(rng.uniform(300, 800), rng.uniform(1000, 2200), 2700), dur=110.0, amp=1.0)
    pros = dict(f0=rng.uniform(95, 190), decl=rng.uniform(0.1, 0.3), stress=rng.choice(['first', 'penult', 'last', 'none']),
                rate=rng.uniform(0.85, 1.2), wgap=rng.uniform(0, 35), lgap=rng.uniform(220, 380), phrase=8)
    return {'slots': slots, 'pros': pros, 'pv': float(pv)}

# ------------------------------------------------------------------ synthesis
def _reson(x, f, bw):
    r = math.exp(-math.pi * bw / SR); th = 2 * math.pi * f / SR
    b0 = 1 - r
    return lfilter([b0], [1, -2 * r * math.cos(th), r * r], x)

def _voiced(n, f0a, f0b, rng):
    f0 = np.linspace(f0a, f0b, n) * (1 + 0.01 * rng.standard_normal())
    ph = np.cumsum(f0 / SR)
    src = np.zeros(n); idx = np.nonzero(np.diff(np.floor(ph)) > 0)[0]
    src[idx] = 1.0
    src = lfilter([1], [1, -0.97], src)  # glottal tilt
    src = lfilter([1], [1, -0.9], src)
    return src - src.mean()

def _noise(n, fc, bw, rng):
    x = rng.standard_normal(n)
    return _reson(_reson(x, fc, bw), fc, bw)

def _env(n, ramp=0.008):
    k = min(int(ramp * SR), n // 2)
    e = np.ones(n)
    if k > 0:
        e[:k] = np.linspace(0, 1, k); e[-k:] = np.linspace(1, 0, k)
    return e

def _norm(x):
    s = np.sqrt(np.mean(x ** 2)) + 1e-9
    return x / s

def render_phrase(phrase, M, ridx, rng, f0_start, f0_end):
    """phrase: list of words (unit lists). Returns audio array."""
    P = M['pros']; slots = M['slots']
    segs = []
    nseg = sum(len(w) for w in phrase); k = 0
    for w in phrase:
        ph = [slots[min(ridx.get(c, NSLOT - 1), NSLOT - 1)] for c in w]
        vpos = [i for i, p in enumerate(ph) if p['cls'] == 'V']
        st = None
        if vpos and P['stress'] != 'none':
            st = vpos[0] if P['stress'] == 'first' else vpos[-1] if P['stress'] == 'last' else vpos[-2] if len(vpos) > 1 else vpos[0]
        for i, p in enumerate(ph):
            frac = k / max(1, nseg); k += 1
            f0 = f0_start + (f0_end - f0_start) * frac
            dur = p['dur'] / P['rate']; amp = p['amp']
            if i == st:
                dur *= 1.3; amp *= 1.4; f0 *= 1.15
            n = max(80, int(dur * SR / 1000))
            if p['cls'] in 'VNL':
                src = _voiced(n, f0, f0 * 0.98, rng)
                y = src
                out = 0
                for j, F in enumerate(p['F']):
                    out = out + _reson(src, F, 60 + 40 * j) * (1.0 / (1 + j))
                y = _norm(out) * amp
            elif p['cls'] == 'F':
                y = _norm(_noise(n, p['fc'], p['bw'], rng)) * amp
                if p['voiced']:
                    y = y + 0.4 * _norm(_reson(_voiced(n, f0, f0, rng), 250, 80)) * amp
            else:  # stop: closure + burst + short aspiration
                nc = int(n * 0.6); nb = n - nc
                clo = np.zeros(nc)
                if p['voiced']:
                    clo = 0.15 * _norm(_reson(_voiced(nc, f0, f0, rng), 200, 80))
                b = _norm(_noise(nb, p['fc'], p['bw'], rng)) * amp * np.exp(-np.linspace(0, 5, nb))
                y = np.concatenate([clo, b])
            segs.append(y * _env(len(y)))
        if P['wgap'] > 1:
            segs.append(np.zeros(int(P['wgap'] * SR / 1000)))
    return np.concatenate(segs) if segs else np.zeros(1)

def render(words, M, ridx, rng, seconds=8.0):
    P = M['pros']; out = []; tot = 0; need = int(seconds * SR)
    i = 0
    while tot < need and i < len(words):
        phr = words[i:i + P['phrase']]; i += P['phrase']
        f0s = P['f0'] * (1 + 0.05 * rng.standard_normal())
        a = render_phrase(phr, M, ridx, rng, f0s, f0s * (1 - P['decl']))
        out.append(a); out.append(np.zeros(int(P['lgap'] * SR / 1000)))
        tot += len(a) + len(out[-1])
    x = np.concatenate(out)[:need]
    if len(x) < need:
        x = np.pad(x, (0, need - len(x)))
    x = x / (np.max(np.abs(x)) + 1e-9) * 0.8
    return x.astype(np.float32)

def clip_words(words, rng, nwords=60):
    s = int(rng.integers(0, max(1, len(words) - nwords)))
    return words[s:s + nwords]

# ------------------------------------------------------------------ scoring
class Scorer:
    """whisper-tiny run on the clip itself (encoder positional table truncated to
    the clip length, no 30 s padding: 4-5x cheaper on a shared CPU).
    Scores per clip:
      nosp      P(no-speech token) at the start-of-transcript step (lower = more speech-like)
      lid_ent   entropy (nats) of the 99-language ID posterior (lower = more confident language)
      lid_max   top language probability
      lex       mean top-token probability over 5 greedy transcription tokens (how word-like)
      bnd       syllable-boundary confidence: mean height of local maxima of encoder frame change (layer 2, z-scored)
      rhy       rhythm regularity: max autocorrelation of that frame-change curve at 3-8 Hz
    """
    def __init__(self, threads=1, ntok=5):
        import torch
        torch.set_num_threads(threads)
        from transformers import WhisperFeatureExtractor, WhisperForConditionalGeneration
        self.torch = torch
        self.fe = WhisperFeatureExtractor.from_pretrained('openai/whisper-tiny')
        self.wm = WhisperForConditionalGeneration.from_pretrained('openai/whisper-tiny').eval()
        gc = self.wm.generation_config
        self.lang_ids = sorted(gc.lang_to_id.values())
        self.lang_names = [k.strip('<|>') for k, v in sorted(gc.lang_to_id.items(), key=lambda kv: kv[1])]
        self.sot = 50258; self.nosp = 50362; self.transcribe = 50359; self.notime = 50363
        self.ntok = ntok

    def encode(self, xs):
        torch = self.torch; enc = self.wm.model.encoder
        n = max(len(x) for x in xs); T = int(math.ceil(n / 160))
        T += T % 2
        f = self.fe(list(xs), sampling_rate=SR, return_tensors='pt').input_features[:, :, :T]
        with torch.no_grad():
            h = torch.nn.functional.gelu(enc.conv1(f)); h = torch.nn.functional.gelu(enc.conv2(h)).permute(0, 2, 1)
            h = h + enc.embed_positions.weight[:h.shape[1]]
            mid = None
            for j, l in enumerate(enc.layers):
                o = l(h, attention_mask=None)
                h = o[0] if isinstance(o, tuple) else o
                if j == 1:
                    mid = h.clone()
            h = enc.layer_norm(h)
        return h, mid

    def score(self, xs):
        torch = self.torch
        h, mid = self.encode(xs)
        B = len(xs)
        with torch.no_grad():
            lg = self.wm(encoder_outputs=(h,), decoder_input_ids=torch.full((B, 1), self.sot)).logits[:, -1, :].float()
            pr = torch.softmax(lg, -1); nosp = pr[:, self.nosp].numpy()
            ll = torch.log_softmax(lg[:, self.lang_ids], -1); pl = ll.exp()
            ent = (-(pl * ll).sum(-1)).numpy(); top = pl.argmax(-1)
            dec = torch.stack([torch.full((B,), self.sot), torch.tensor(self.lang_ids)[top], torch.full((B,), self.transcribe), torch.full((B,), self.notime)], 1)
            conf = []
            for _ in range(self.ntok):
                l2 = self.wm(encoder_outputs=(h,), decoder_input_ids=dec).logits[:, -1, :].float()
                p2 = torch.softmax(l2, -1); m, a = p2.max(-1)
                conf.append(m); dec = torch.cat([dec, a[:, None]], 1)
            lex = torch.stack(conf, 1).mean(1).numpy()
        res = []
        mid = mid.numpy()
        for i in range(B):
            hm = mid[i]; hn = hm / (np.linalg.norm(hm, axis=1, keepdims=True) + 1e-9)
            d = 1 - (hn[1:] * hn[:-1]).sum(1); d = (d - d.mean()) / (d.std() + 1e-9)
            pk = (d[1:-1] > d[:-2]) & (d[1:-1] > d[2:])
            bnd = float(np.mean(d[1:-1][pk])) if pk.any() else 0.0
            ac = np.correlate(d, d, 'full')[len(d) - 1:]; ac = ac / (ac[0] + 1e-9)
            rhy = float(ac[6:17].max())   # encoder frames at 50 Hz: lags 6-16 = 3.1-8.3 Hz
            res.append(dict(nosp=float(nosp[i]), lid_ent=float(ent[i]), lid_max=float(pl[i].max()),
                            lid_top=self.lang_names[int(top[i])], lex=float(lex[i]), bnd=bnd, rhy=rhy))
        return res, pl.numpy()

METRICS = ['nosp', 'lid_ent', 'lid_max', 'lex', 'bnd', 'rhy']
# sign so that larger = more speech-like
SIGN = {'nosp': -1, 'lid_ent': -1, 'lid_max': 1, 'lex': 1, 'bnd': 1, 'rhy': 1}
