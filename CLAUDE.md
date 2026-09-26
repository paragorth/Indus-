# Standing instructions for Claude in this repository

- **Keep digging. Do not stop to wait for the user's reply.** When a strategy finishes, log it and start the next one. Report progress in short lines, and only stop when the user explicitly says stop.
- The goal is to decipher the Indus script. Say "cracked" only if a reading passes the calibrated controls (see STRATEGIES.md S107, S113–S137) and predicts something outside the corpus.
- Do the digital legwork yourself: archive.org, CDLI, HathiTrust API, Met API, Crossref, Wikimedia Commons, headless Chromium (the proxy CA is installed in ~/.pki/nssdb; for sites that refuse curl, e.g. harappa.com PDFs, run Chromium non-headless under xvfb-run and screenshot pages with `#page=N&zoom=200`, see scratchpad shot1.py). Ask the user only for things that need a human (logins, CAPTCHAs, physical access), and then give a direct link.
- Rules of evidence: use data and symbols from anyone, but not other people's interpretations. Every strategy gets a control and a verdict row in STRATEGIES.md, inserted before the "## Summary" section. Commit and push to branch `claude/indus-script-dictionary-u3dzf8`.
- Never disable TLS verification, never click through CAPTCHAs, never put model identifiers in commits.
- Current open leads: ANCHORS.md §19–21 (outside anchors), S149 (Kalibangan ↔ Bahrain twins + W56 link), Scheil 1925 Umma tag (possible cuneiform beside an Indus impression), `anchor_test.py` for any proposed sign values.
