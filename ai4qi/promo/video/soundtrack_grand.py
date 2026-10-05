"""Grand, cinematic soundtrack for the Ai4Qi promo video (original, generated here).
String ensemble swelling through the chords, a driving low-string ostinato that builds,
timpani on each scene change, a cymbal rise into the end card and a full brass + strings chord.
    python3 soundtrack_grand.py out.wav
"""
import sys, wave
import numpy as np

SR, DUR = 48000, 48.0
N = int(SR * DUR)
rng = np.random.default_rng(11)
hz = lambda m: 440.0 * 2 ** ((m - 69) / 12)
BOUNDS = [0, 3.6, 8.4, 13.2, 17.6, 22.4, 27.2, 32.4, 38.4, 43.0, 48.0]
CHORDS = [[50, 57, 62, 66], [47, 54, 59, 62, 66], [43, 50, 55, 59, 62], [45, 52, 57, 61, 64],
          [42, 50, 54, 57, 62], [40, 47, 52, 55, 59], [43, 50, 55, 59, 62, 66], [45, 52, 57, 61, 64, 69],
          [47, 54, 59, 62, 66], [38, 50, 57, 62, 66, 69, 74]]
L = np.zeros(N); R = np.zeros(N)

def add(sig, start, pan=0.0, gain=1.0):
    i = int(start * SR); j = min(N, i + len(sig)); sig = sig[: j - i] * gain
    L[i:j] += sig * np.sqrt(0.5 * (1 - pan)); R[i:j] += sig * np.sqrt(0.5 * (1 + pan))

def env(n, a, r):
    e = np.ones(n); a = int(a * SR); r = int(r * SR)
    if a: e[:a] = (np.linspace(0, 1, a)) ** 1.5
    if r: e[-r:] *= np.linspace(1, 0, r) ** 2
    return e

def saw(f, tt, nh=14, bright=1.0):
    """band-limited, softened sawtooth with gentle vibrato"""
    vib = 1 + 0.0035 * np.sin(2 * np.pi * 5.2 * tt + rng.uniform(0, 6.28)) * np.minimum(1, tt / 0.8)
    ph = 2 * np.pi * f * np.cumsum(vib) / SR
    out = np.zeros_like(tt)
    for h in range(1, nh + 1):
        if f * h > 9000: break
        out += np.sin(h * ph) / h * np.exp(-(h - 1) * 0.22 / bright)
    return out

# overall crescendo: quiet title, building to the end card
t = np.arange(N) / SR
GROW = np.interp(t, [0, 3.6, 13, 32, 38.4, 42.6, 43, 48], [0.75, 0.8, 0.85, 0.95, 1.0, 1.05, 1.0, 1.0])

# string ensemble pads
for k, ch in enumerate(CHORDS):
    a, b = BOUNDS[k], BOUNDS[k + 1]
    s, e = max(0, a - 0.5), min(DUR, b + (2.5 if k == 9 else 0.8))
    n = int((e - s) * SR); tt = np.arange(n) / SR
    pad = np.zeros(n)
    for m in ch[1:] if k < 9 else ch:
        for d in (-0.12, 0.0, 0.12):
            pad += saw(hz(m) * (1 + d / 100), tt, bright=0.8 + 0.1 * (k / 9))
    att = 2.4 if k == 0 else (0.15 if k == 9 else 0.9)
    add(pad * env(n, att, 1.6 if k == 9 else 1.0) / len(ch), s, 0.0, 0.05)

# low-string ostinato (16ths at 100 bpm), from scene 3, accents on beats
step = 60 / 100 / 4
tq, i = 8.4, 0
while tq < 42.9:
    k = max(x for x in range(10) if BOUNDS[x] <= tq + 1e-6)
    root = CHORDS[k][0] if k < 9 else 38
    f = hz(root - 12 if root > 44 else root)
    n = int(0.22 * SR); tt = np.arange(n) / SR
    note = saw(f, tt, nh=10, bright=0.6) * np.exp(-tt * 14) * np.minimum(1, tt / 0.006)
    acc = 1.0 if i % 4 == 0 else 0.6
    add(note, tq, (-0.25, 0.25)[i % 2], 0.11 * acc)
    tq += step; i += 1

# timpani on scene changes (bigger later)
def timpani(f, n_s=2.2):
    n = int(n_s * SR); tt = np.arange(n) / SR
    fr = f * (1 + 0.06 * np.exp(-tt * 18))
    ph = 2 * np.pi * np.cumsum(fr) / SR
    body = (np.sin(ph) + 0.5 * np.sin(1.5 * ph) + 0.25 * np.sin(1.99 * ph)) * np.exp(-tt * 2.2)
    hit = rng.standard_normal(n) * np.exp(-tt * 40) * 0.6
    X = np.fft.rfft(hit); fq = np.fft.rfftfreq(n, 1 / SR); X[fq > 1200] = 0; hit = np.fft.irfft(X, n)
    return (body + hit) * np.minimum(1, tt / 0.002)
for k, b in enumerate(BOUNDS[1:-1], start=1):
    root = CHORDS[k][0]
    g = 0.16 + 0.1 * (k / 9)
    add(timpani(hz(root - 12 if root > 44 else root)), b, 0.0, g)
add(timpani(hz(38)), 43.0, 0.0, 0.26); add(timpani(hz(38)), 43.0 + 0.3, 0.1, 0.12)

# cymbal / noise rise into the end card, and soft rises into scenes
def rise(len_s, lo=3000):
    n = int(len_s * SR); x = rng.standard_normal(n)
    X = np.fft.rfft(x); fq = np.fft.rfftfreq(n, 1 / SR); X[fq < lo] *= 0.05; y = np.fft.irfft(X, n)
    return y / np.abs(y).max() * np.linspace(0, 1, n) ** 3
add(rise(3.2), 43.0 - 3.2, 0.0, 0.05)
for b in BOUNDS[1:-2]:
    add(rise(0.8, 2000), b - 0.8, rng.uniform(-0.3, 0.3), 0.035)
# crash on the end card
n = int(3.5 * SR); tt = np.arange(n) / SR
cr = rng.standard_normal(n); X = np.fft.rfft(cr); fq = np.fft.rfftfreq(n, 1 / SR); X[fq < 2500] *= 0.1; cr = np.fft.irfft(X, n)
add(cr / np.abs(cr).max() * np.exp(-tt * 1.4), 43.0, 0.0, 0.06)

# brass on the end card: bright swelling saw chord
n = int(5.0 * SR); tt = np.arange(n) / SR
br = np.zeros(n)
for m in (50, 57, 62, 66, 69):
    for d in (-0.08, 0.08):
        br += saw(hz(m) * (1 + d / 100), tt, nh=16, bright=1.6)
add(br * env(n, 0.08, 2.2) * (0.8 + 0.2 * np.minimum(1, tt / 0.6)) / 5, 43.0, 0.0, 0.05)
# low brass pedal through the climb (scenes 8–9)
n = int(10.6 * SR); tt = np.arange(n) / SR
add(saw(hz(33), tt, nh=12, bright=1.0) * env(n, 3.0, 0.4), 32.4, 0.0, 0.05)

# apply crescendo, hall reverb, master
L *= GROW; R *= GROW
def reverb(x, rt=3.0, mix=0.32, seed=0):
    r = np.random.default_rng(seed); n = int(rt * SR)
    ir = r.standard_normal(n) * np.exp(-np.arange(n) / SR * 6.9 / rt); ir[: int(0.02 * SR)] = 0
    m = 1 << int(np.ceil(np.log2(len(x) + n)))
    y = np.fft.irfft(np.fft.rfft(x, m) * np.fft.rfft(ir, m), m)[: len(x)]
    return x + mix * y / np.abs(y).max() * np.abs(x).max()
L, R = reverb(L, seed=1), reverb(R, seed=2)
# gentle limiter
def soft(x): return np.tanh(x)
pk = np.percentile(np.abs(np.concatenate([L, R])), 99.7); L, R = soft(L / pk * 0.7), soft(R / pk * 0.7)
fade = np.ones(N); fade[: int(0.2 * SR)] = np.linspace(0, 1, int(0.2 * SR)); fade[-int(1.5 * SR):] *= np.linspace(1, 0, int(1.5 * SR)) ** 1.5
L, R = L * fade * 0.9, R * fade * 0.9
pcm = (np.stack([L, R], 1) * 32767).astype('<i2')
with wave.open(sys.argv[1] if len(sys.argv) > 1 else 'grand.wav', 'wb') as w:
    w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR); w.writeframes(pcm.tobytes())
print('ok')
