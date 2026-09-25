"""Autonomous run: controls, published key, and every search condition.

Appends to RESULTS.md as it goes; per-condition JSON (with all fitted keys) goes
to results/.  If a real-corpus condition passes every criterion it writes
STOP.md and halts.

  python run_all.py                 # everything, 4 worker processes
  python run_all.py --only tamil-full-80 --restarts 4 --fakes 8   # one condition, quick
"""
import argparse
import datetime as dt
import json
import os
import statistics
import sys
import time
from multiprocessing import Pool

import indus_core as C
import experiment as X
import synth
import validator as V

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "RESULTS.md")
OUT = os.path.join(HERE, "results")


def now():
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")


def log(text):
    with open(RES, "a") as f:
        f.write(text.rstrip() + "\n\n")
    print(text, flush=True)


def fmt(x):
    return f"{x:.4f}" if isinstance(x, float) else str(x)


def run_condition(pool, name, lang, kind, N, train, held, top, restarts, fakes, iters,
                  extra=None, control=False):
    t0 = time.time()
    base = dict(lang=lang, kind=kind, N=N, train=train, held=held, top=top, iters=iters)
    if extra:
        base.update(extra)
    tasks = [dict(base, seed=i, fake=None) for i in range(restarts)]
    tasks += [dict(base, seed=1000 + i, fake=20_000 + i) for i in range(fakes)]
    res = pool.map(X.run_task, tasks, chunksize=1)
    s = X.summarise(res, restarts)
    verdict = {}
    if kind in ("full", "skel", "mixed"):
        mode = "skel" if kind == "skel" else "full"
        lex = X._lex(lang)
        # fixed-key validator on the best real key (as specified)
        fk = V.fixed_key_test(s["best_key"], held, lex, mode, n_fake=30)
        # the same validator applied to the best key fitted to a FAKE lexicon,
        # treating that fake as if it were the language: shows what a fitted
        # key achieves on this test with no language present
        bf = max((r for r in res if r["fake"] is not None), key=lambda r: r["train_score"])
        ff = fixed_forms_test(bf["key"], held, C.fake_forms(lex, mode, bf["fake"]),
                              [C.fake_forms(lex, mode, 10_000 + i) for i in range(30)], mode)
        verdict.update(z_fixed=fk["z_fixed"], pass_fixed=fk["pass_fixed"],
                       fakefit_z_fixed=ff["z_fixed"], fakefit_pass_fixed=ff["pass_fixed"])
    passed = (s["gap_sd"] > 3 and s["real_held_mean"] > s["real_scr_mean"]
              and verdict.get("pass_fixed", True))
    verdict["PASS"] = bool(passed)
    stab = X.stability(res, 12)
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, f"{name}.json"), "w") as f:
        json.dump({"name": name, "summary": {k: v for k, v in s.items()}, "verdict": verdict,
                   "stability": stab, "iters": iters,
                   "runs": [{k: v for k, v in r.items()} for r in res]}, f)
    s.pop("best_key")
    line = (f"### {now()}  `{name}`{'  (control)' if control else ''}\n"
            f"method: {kind} key, {lang}, top-{N} signs, {restarts} real restarts + {fakes} "
            f"fake-lexicon searches, {iters} SA iterations each, {time.time() - t0:.0f}s\n\n"
            + "| " + " | ".join(s) + " |\n|" + "---|" * len(s) + "\n| "
            + " | ".join(fmt(v) for v in s.values()) + " |\n\n"
            + "fixed-key / verdict: " + ", ".join(f"{k}={fmt(v)}" for k, v in verdict.items())
            + f"\n\n**gap-over-fake = {s['gap_sd']:+.2f} SD -> {'PASS' if passed else 'fail'}**")
    if not control and kind != "logo":
        line += "\n\nmost stable assignments across real restarts (share of restarts, sign, value, share of fake runs giving that value, also modal under fakes?): " + \
            "; ".join(f"{a:.2f} {sg}->{v!r} ({b:.2f}{', fake-modal' if m else ''})" for a, sg, v, b, m in stab[:8])
    elif not control:
        line += "\n\nmost stable: " + "; ".join(f"{a:.2f} {sg}->{v!r}" for a, sg, v, b, m in stab[:8])
    log(line)
    return s, verdict


def fixed_forms_test(key, held, real_fs, fake_fs_list, mode):
    m = C.MINLEN[mode]
    real = V.rate(key, held, real_fs, m)
    scr = statistics.mean(V.rate(key, V.scramble(held, 100 + i), real_fs, m) for i in range(5))
    fk = [V.rate(key, held, f, m) for f in fake_fs_list]
    mu, sd = statistics.mean(fk), statistics.pstdev(fk)
    z = (real - mu) / sd if sd else 0.0
    return {"z_fixed": z, "pass_fixed": bool(real > scr and z > 3)}


def stop(name, s, verdict, cmd):
    with open(os.path.join(HERE, "STOP.md"), "w") as f:
        f.write(f"# STOP: `{name}` passed\n\n{now()}\n\n```\n{json.dumps(s, indent=1)}\n"
                f"{json.dumps(verdict, indent=1)}\n```\n\nKey and all runs: results/{name}.json\n\n"
                f"Reproduce:\n\n```\n./scripts/fetch_data.sh\n{cmd}\n```\n")
    log(f"**STOP: {name} passed. See STOP.md. Halting.**")
    sys.exit(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--restarts", type=int, default=20)
    ap.add_argument("--fakes", type=int, default=30)
    ap.add_argument("--iter-per-sign", type=int, default=400)
    ap.add_argument("--only", default=None)
    ap.add_argument("--budget-min", type=float, default=235)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--corpus", choices=["yajnadevam", "im77"], default="yajnadevam")
    ap.add_argument("--kinds", default=None, help="comma list of key kinds to run, e.g. full,mixed")
    ap.add_argument("--no-published", action="store_true")
    a = ap.parse_args()
    T0 = time.time()

    texts = C.unique_texts(C.load_corpus() if a.corpus == "yajnadevam" else C.load_im77_corpus())
    train, held = C.split_texts(texts)
    top = [s for s, _ in C.sign_freq(train).most_common()]
    tam, san = C.load_tamil(), C.load_sanskrit()
    log(f"## Run started {now()}\n\ncorpus ({a.corpus}): {len(texts)} unique texts "
        f"({len(train)} train / {len(held)} held-out), {len(top)} signs in train; "
        f"lexicons: Tamil {len(tam)} forms, Sanskrit {len(san)} forms; MINLEN={C.MINLEN}; "
        f"restarts={a.restarts}, fakes={a.fakes}, iterations={a.iter_per_sign}*N")

    conds = []
    # ---- positive controls: planted language, same pipeline
    for lang, lex in (("tamil", tam), ("sanskrit", san)):
        for noise in (0.0, 0.2, 0.4):
            pt, _ = synth.planted_corpus(texts, lex, seed=7, noise=noise)
            ptr, phe = C.split_texts(pt)
            ptop = [s for s, _ in C.sign_freq(ptr).most_common()]
            conds.append((f"control-planted-{lang}-full-80-noise{noise}", lang, "full", 80,
                          ptr, phe, ptop, None, True))
    pt, _ = synth.planted_corpus(texts, tam, seed=7, noise=0.0)
    ptr, phe = C.split_texts(pt)
    ptop = [s for s, _ in C.sign_freq(ptr).most_common()]
    conds.append(("control-planted-tamil-skel-80-noise0.0", "tamil", "skel", 80, ptr, phe, ptop, None, True))
    conds.append(("control-planted-tamil-mixed-150-noise0.2", "tamil", "mixed", 150,
                  *(lambda p: (p[0], p[1]))(C.split_texts(synth.planted_corpus(texts, tam, 7, 0.2)[0])),
                  [s for s, _ in C.sign_freq(C.split_texts(synth.planted_corpus(texts, tam, 7, 0.2)[0])[0]).most_common()],
                  {"n_syl": 80}, True))
    lt, _ = synth.planted_logo_corpus(texts, tam, seed=7, noise=0.2)
    ltr, lhe = C.split_texts(lt)
    ltop = [s for s, _ in C.sign_freq(ltr).most_common()]
    conds.append(("control-planted-tamil-logo-150-noise0.2", "tamil", "logo", 150, ltr, lhe, ltop, None, True))

    # ---- the real searches
    for N in (80, 150, 250):
        for lang in ("tamil", "sanskrit"):
            conds.append((f"{lang}-full-{N}", lang, "full", N, train, held, top, None, False))
    for N in (80, 150, 250):
        conds.append((f"tamil-logo-{N}", "tamil", "logo", N, train, held, top, None, False))
    for lang in ("tamil", "sanskrit"):
        conds.append((f"{lang}-mixed-250", lang, "mixed", 250, train, held, top, {"n_syl": 80}, False))
    for N in (80, 150, 250):
        for lang in ("tamil", "sanskrit"):
            conds.append((f"{lang}-skel-{N}", lang, "skel", N, train, held, top, None, False))

    if a.only:
        conds = [c for c in conds if c[0] == a.only]
    if a.kinds:
        conds = [c for c in conds if c[2] in a.kinds.split(",")]
    if a.corpus != "yajnadevam":
        conds = [(f"{a.corpus}-{c[0]}",) + tuple(c[1:]) for c in conds]

    best = {}
    with Pool(a.workers) as pool:
        # published key first (fixed-key validator only: it was fitted in-sample)
        if not a.only and not a.no_published and a.corpus == "yajnadevam":
            pk = V.published_key()
            for lang, lex in (("sanskrit", san), ("tamil", tam)):
                for mode in ("full", "skel"):
                    k = pk if mode == "full" else {s: C.skeleton(v) for s, v in pk.items()}
                    r = V.fixed_key_test(k, held, lex, mode, n_fake=30)
                    log(f"### {now()}  published key (Yajnadevam xlits.csv) vs {lang}, {mode}\n"
                        + ", ".join(f"{kk}={fmt(v)}" for kk, v in r.items()))
        for name, lang, kind, N, tr, he, tp, extra, control in conds:
            if (time.time() - T0) / 60 > a.budget_min:
                log(f"time budget reached before `{name}`; skipped")
                continue
            iters = a.iter_per_sign * N
            s, v = run_condition(pool, name, lang, kind, N, tr, he, tp, a.restarts, a.fakes,
                                 iters, extra, control)
            if not control:
                k = (lang, kind)
                if k not in best or s["gap_sd"] > best[k][1]:
                    best[k] = (name, s["gap_sd"])
                if v["PASS"]:
                    stop(name, s, v, f"python run_all.py --only {name}")

    if not a.only:
        lines = [f"## Final entry {now()} ({(time.time() - T0) / 60:.0f} min)\n",
                 "Best gap-over-fake per language and search type (real corpus):\n"]
        for (lang, kind), (name, g) in sorted(best.items()):
            lines.append(f"- {lang} / {kind}: {g:+.2f} SD (`{name}`)")
        lines.append("\n**No key survived.**" if best else "")
        log("\n".join(lines))


if __name__ == "__main__":
    main()
