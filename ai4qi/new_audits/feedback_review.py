"""Monthly feedback review: which proposed audits should be improved, from everyone's votes.

    python3 new_audits/feedback_review.py [--min 5]

Reads the public vote totals (the same feedback_summary the site shows; the publishable key in
app/config.json, no secret needed) and writes new_audits/feedback_reviews/YYYY-MM.md:

  - to improve: at least --min votes, and at least twice as many down as up; with the reasons given
  - well liked: at least --min votes, no more than one in five down
  - too few votes yet: everything else is left alone

Nothing in the library changes here. Rewrites of the "to improve" audits are drafted by Claude in the
monthly session and only go live after the owner approves them.
"""
import argparse
import datetime
import json
import sys
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent
APP = HERE.parent / "app"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min", type=int, default=5, help="votes an audit needs before it counts")
    a = ap.parse_args()
    cfg = json.loads((APP / "config.json").read_text())
    url, key = cfg.get("supabase_url"), cfg.get("supabase_anon_key")
    if not url or not key:
        sys.exit("config.json has no supabase_url / supabase_anon_key")
    r = requests.post(f"{url}/rest/v1/rpc/feedback_summary", json={}, timeout=60,
                      headers={"apikey": key, "Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    r.raise_for_status()
    rows = {x["audit_id"]: x for x in r.json()}
    proposed = {p["id"]: p for p in json.loads((APP / "data" / "proposed.json").read_text())}

    improve, liked, few = [], [], 0
    for aid, x in rows.items():
        up, down = int(x.get("up") or 0), int(x.get("down") or 0)
        if up + down < a.min:
            few += 1
            continue
        if down >= 2 * max(1, up):
            improve.append((aid, up, down, x.get("reasons") or {}))
        elif down * 5 <= up + down:
            liked.append((aid, up, down))
    improve.sort(key=lambda t: -(t[2] - t[1]))
    liked.sort(key=lambda t: -t[1])

    month = datetime.date.today().strftime("%Y-%m")
    out = [f"# Feedback review, {month}", "",
           f"Votes counted from {len(rows)} audits. An audit needs at least {a.min} votes to count; "
           f"{few} have fewer and are left alone.", ""]
    out += ["## To improve (at least twice as many down as up)", ""]
    if not improve:
        out.append("None this month.")
    for aid, up, down, reasons in improve:
        p = proposed.get(aid, {})
        why = ", ".join(f"{k} ({v})" for k, v in sorted(reasons.items(), key=lambda kv: -kv[1])) or "no reason given"
        kind = "built audit" if aid.startswith("B-") else "proposed audit"
        out.append(f"- **{aid}** ({kind}) {up} up / {down} down. Reasons: {why}. {p.get('question', '')}")
    out += ["", "## Well liked (no more than one in five down)", ""]
    if not liked:
        out.append("None yet.")
    for aid, up, down in liked:
        out.append(f"- **{aid}** {up} up / {down} down. {proposed.get(aid, {}).get('question', '')}")
    out += ["", "## What happens next", "",
            "Claude drafts a rewrite of each audit under *To improve*, aimed at the reasons given, as a",
            "change for the owner to approve. Nothing goes live until it is approved."]
    dest = HERE / "feedback_reviews" / f"{month}.md"
    dest.parent.mkdir(exist_ok=True)
    dest.write_text("\n".join(out) + "\n")
    print(f"{dest.relative_to(HERE.parent)}: {len(improve)} to improve, {len(liked)} well liked, {few} with too few votes")


if __name__ == "__main__":
    main()
