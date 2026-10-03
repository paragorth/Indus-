"""Rebuild the seed ai4qi-library.json (ids 1-1145) that the user pasted in chat.

The pasted library was transcribed into compact pipe-separated files to keep the
transcription reviewable; this script expands them back to the original schema.
Run once; the pipeline never rewrites seed entries.
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
U1 = "https://www.uhbw.nhs.uk/assets/1/23-639_uhbw_ca_annual_report_2021-22.pdf"
U2 = "https://www.uhbristol.nhs.uk/media/4556025/uhbw_clinical_audit_report_2022.23.pdf"
STOCKPORT = "https://www.stockport.nhs.uk/Documents/AppDocViewer.aspx?e=2580"
OOJ = "https://openorthopaedicsjournal.com/VOLUME/4/PAGE/188/FULLTEXT/"
MTW = "https://www.mtw.nhs.uk/wp-content/uploads/2024/01/Urology-and-General-Surgery.150124.pdf"
SRC = {"U1": U1, "U2": U2, "OOJ": OOJ}


def rows(name, ncols):
    with open(os.path.join(HERE, name), encoding="utf-8") as f:
        out = [line.rstrip("\n").split("|") for line in f if line.strip()]
    for r in out:
        assert len(r) == ncols, (name, r)
    return out


def main():
    audits = []
    nid = 1

    def add(e):
        nonlocal nid
        e["id"] = nid
        audits.append(e)
        nid += 1

    for fn, src in (("a_1_156.txt", U1), ("a_157_342.txt", U2)):
        for sp, ti, st, fi, ch in rows(fn, 5):
            add({"specialty": sp, "title": ti, "standard": st, "finding": fi, "change": ch,
                 "source": src, "status": "completed"})
    for sp, ti, s in rows("b_343_481.txt", 3):
        add({"specialty": sp, "title": ti, "standard": "", "finding": "", "change": "",
             "source": SRC[s], "status": "registered, result not published"})
    for sp, ti, st, fi, s in rows("c_482_530.txt", 5):
        add({"specialty": sp, "title": ti, "standard": st, "finding": fi, "change": "",
             "source": STOCKPORT,
             "status": "completed" if s == "C" else "registered, result not published"})
    extra = json.load(open(os.path.join(HERE, "g_h_json.json"), encoding="utf-8"))
    for sp, ti, st, fi, ch, na, s in rows("d_531_604.txt", 7):
        add({"specialty": sp, "title": ti, "standard": st, "finding": fi, "change": ch,
             "next_audit": na, "source": SRC.get(s, s), "status": "published or reported"})
        audits[-1].update(extra["details"].get(str(audits[-1]["id"]), {}))
    for fn in ("e_605_800.txt", "e_801_1034.txt"):
        for sp, ti, st, fi, ch, na in rows(fn, 6):
            add({"specialty": sp, "title": ti, "standard": st, "finding": fi, "change": ch,
                 "next_audit": na, "source": None,
                 "status": "recurring topic, not individually linked"})
    for sp, ti in rows("f_1035_1113.txt", 2):
        add({"specialty": sp, "title": ti, "standard": "", "finding": "", "change": "",
             "source": MTW, "status": "completed, results not published"})
    for e in extra["tail"]:
        assert e["id"] == nid, (e["id"], nid)
        audits.append(e)
        nid += 1

    ids = [a["id"] for a in audits]
    assert ids == list(range(1, 1146)), "ids must run 1..1145"
    lib = {"about": extra["about"], "topic_knowledge": extra["topic_knowledge"], "audits": audits}
    out = os.path.join(HERE, "ai4qi-library.seed.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(lib, f, ensure_ascii=False, indent=1)
    print(f"wrote {out}: {len(audits)} audits, {len(lib['topic_knowledge'])} topic cards")


if __name__ == "__main__":
    main()
