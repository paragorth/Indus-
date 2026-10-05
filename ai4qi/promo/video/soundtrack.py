"""Original soundtrack for the Ai4Qi promo video (generated here, so no licensing).
Warm pad chords that change with each scene, a soft piano-like arpeggio, a low bass,
gentle whooshes into each scene change and a chime on the end card.
    python3 soundtrack.py out.wav
"""
import sys, wave
import numpy as np

SR, DUR = 48000, 48.0
N = int(SR * DUR)
t = np.arange(N) / SR
rng = np.random.default_rng(7)
hz = lambda m: 440.0 * 2 ** ((m - 69) / 12)

# scene boundaries (match video.html) and chords as MIDI notes (D major)
BOUNDS = [0, 3.6, 8.4, 13.2, 17.6, 22.4, 27.2, 32.4, 38.4, 43.0, 48.0]
CHORDS = [
    [50, 54, 57, 61],        # Dmaj7
    [47, 50, 54, 57, 61],    # Bm9
    [43, 47, 50, 54],        # Gmaj7
    [45, 50, 52, 57],        # Asus
    [42, 45, 50, 54, 61],    # Dmaj7/F#
    [40, 43, 47, 50],        # Em7
    [43, 47, 50, 54, 57],    # Gmaj9
    [45, 49, 52, 57],        # A
    [47, 50, 54, 57],        # Bm7
    [50, 54, 57, 61, 64],    # Dmaj9
]

def env(n, a, r):
    e = np.ones(n); a = int(a * SR); r = int(r * SR)
    if a: e[:a] = np.linspace(0, 1, a) ** 2
    if r: e[-r:] *= np.linspace(1, 0, r) ** 2
    return e

L = np.zeros(N); R = np.zeros(N)
def add(sig, start, pan=0.0, gain=1.0):
    i = int(start * SR); j = min(N, i + len(sig)); sig = sig[: j - i] * gain
    L[i:j] += sig * np.sqrt(0.5 * (1 - pan)); R[i:j] += sig * np.sqrt(0.5 * (1 + pan))

# pad: detuned soft harmonics, slow attack, overlapping crossfades between chords
for k, ch in enumerate(CHORDS):
    a, b = BOUNDS[k], BOUNDS[k + 1]
    s, e = max(0, a - 0.6), min(DUR, b + 0.9)
    n = int((e - s) * SR); tt = np.arange(n) / SR
    pad = np.zeros(n)
    for m in ch:
        f = hz(m + 12)
        for d in (-0.06, 0.06):
            for h, amp in ((1, 1.0), (2, 0.28), (3, 0.08)):
                pad += amp * np.sin(2 * np.pi * f * h * (1 + d / 100) * tt + rng.uniform(0, 6.28))
    pad *= env(n, 1.2 if k else 2.5, 1.4) * (1 + 0.08 * np.sin(2 * np.pi * 0.2 * tt))
    add(pad / len(ch), s, 0.0, 0.055)
    # bass
    bf = hz(ch[0]); bass = np.sin(2 * np.pi * bf * tt) + 0.3 * np.sin(2 * np.pi * 2 * bf * tt)
    add(bass * env(n, 0.6, 1.2), s, 0.0, 0.025 if k else 0.0)

# arpeggio: 8th notes at 100 bpm, piano-ish pluck, from scene 2 to the end card
step = 60 / 100 / 2
def pluck(f, n_s=1.6):
    n = int(n_s * SR); tt = np.arange(n) / SR
    s = sum(a * np.sin(2 * np.pi * f * h * tt) * np.exp(-tt * (2.6 + h * 1.6)) for h, a in ((1, 1), (2, 0.45), (3, 0.18), (4, 0.08)))
    return s * np.minimum(1, tt / 0.004)
pattern = [0, 1, 2, 3, 2, 1, 3, 2]
tq = 3.6; i = 0
while tq < 43.0 - 0.05:
    k = max(x for x in range(len(CHORDS)) if BOUNDS[x] <= tq + 1e-6)
    ch = CHORDS[k]; m = ch[pattern[i % 8] % len(ch)] + 24
    vel = 0.75 + 0.25 * (i % 2 == 0)
    add(pluck(hz(m)), tq, (-0.35, 0.35)[i % 2], 0.06 * vel)
    tq += step; i += 1

# whoosh into each scene change: band-limited noise swelling over 0.7 s
def whoosh(len_s=0.9):
    n = int(len_s * SR); x = rng.standard_normal(n)
    X = np.fft.rfft(x); f = np.fft.rfftfreq(n, 1 / SR)
    X *= np.exp(-((np.log(f + 1) - np.log(1500)) ** 2) / 0.9)
    y = np.fft.irfft(X, n); y /= np.abs(y).max()
    e = np.concatenate([np.linspace(0, 1, int(n * 0.78)) ** 3, np.linspace(1, 0, n - int(n * 0.78)) ** 2])
    return y * e
for b in BOUNDS[1:-1]:
    add(whoosh(), b - 0.75, rng.uniform(-0.4, 0.4), 0.03)

# end chime: bell partials on D
def bell(f, n_s=4.5):
    n = int(n_s * SR); tt = np.arange(n) / SR
    return sum(a * np.sin(2 * np.pi * f * r * tt) * np.exp(-tt * d) for r, a, d in ((1, 1, 1.1), (2.76, 0.4, 2.2), (5.4, 0.2, 3.5), (8.9, 0.08, 5)))
add(bell(hz(74)), 43.1, -0.2, 0.09); add(bell(hz(81)), 43.35, 0.2, 0.06); add(bell(hz(86)), 43.6, 0.0, 0.05)
# soft tick on the counter (scene 2)
for q in np.arange(3.95, 6.1, 0.09):
    add(pluck(hz(98), 0.12) * 0.5, q, 0.0, 0.012)

# room: FFT convolution with a short decaying-noise impulse
def reverb(x, rt=1.8, mix=0.22):
    n = int(rt * SR); ir = rng.standard_normal(n) * np.exp(-np.arange(n) / SR * 6.9 / rt); ir[0] = 0
    m = 1 << int(np.ceil(np.log2(len(x) + n)))
    y = np.fft.irfft(np.fft.rfft(x, m) * np.fft.rfft(ir, m), m)[: len(x)]
    return x + mix * y / np.abs(y).max() * np.abs(x).max()
L, R = reverb(L), reverb(R)

# fade in/out, normalise
fade = np.ones(N); fade[: int(0.3 * SR)] = np.linspace(0, 1, int(0.3 * SR)); fade[-int(2.0 * SR):] *= np.linspace(1, 0, int(2.0 * SR))
L *= fade; R *= fade
pk = max(np.abs(L).max(), np.abs(R).max()); L, R = L / pk * 0.85, R / pk * 0.85
pcm = (np.stack([L, R], 1) * 32767).astype('<i2')
with wave.open(sys.argv[1] if len(sys.argv) > 1 else 'soundtrack.wav', 'wb') as w:
    w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR); w.writeframes(pcm.tobytes())
print('ok', N / SR, 's')
