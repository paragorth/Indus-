# Ai4Qi worked example: sepsis antibiotics within 1 hour

All data here is fictitious. The hospital (Riverside General), the audit lead and every record were
made up for demonstration.

**Audit:** NNA-074. What proportion of adults at high risk from sepsis (NEWS2 7 or more with
suspected infection) receive IV antibiotics within 1 hour of their first ED NEWS2 score?
**Standard:** NICE NG253 rec 1.8.3. **Target:** ≥90%.

| Step | What happens | Screen |
|---|---|---|
| 1 | Open the protocol (or build one on any topic from the home page) | `screens/02-protocol.jpg` |
| 2 | Press **Choose this audit**; set a passcode so records are stored encrypted on the device | `screens/03-passcode.jpg` |
| 3 | Add the audit details: site, lead, start date, records per cycle | `screens/05-details.jpg` |
| 4 | Download the **Data sheet** (Excel). Fill the Data tab at work | `sepsis-audit-cycle1.xlsx` |
| 5 | Upload the filled sheet. Hospital numbers are replaced with audit codes (P001…) before anything is stored | `screens/06-import-preview.jpg` |
| 6 | Cycle 1 result: 23 of 40 (58%) against a 90% target. Commonest delay: not recognised as sepsis (8) | `screens/08-results-c1.jpg` |
| 7 | Record the change: an e-observations sepsis alert with a 60-minute countdown | `screens/09-change.jpg` |
| 8 | Re-audit, totals only: copy the one-line results code from the Excel Results tab into Ai4Qi | `results-code.txt`, `screens/10-paste-code.jpg` |
| 9 | Re-audit result: 37 of 40 (93%), target met, loop closed | `screens/11-results-both.jpg`, `screens/12-loop-closed.jpg` |
| 10 | Download the **Results presentation** for the governance meeting | `final-results.pptx` |

Step 5 and step 8 show the two data routes. Uploading the sheet keeps de-identified records on the
device. Pasting the results code sends only totals, and the records never leave the Trust.

## Demo mode, for live demonstrations

Open the account menu (top right) and pick **Demo mode**, or go to `#/demo`. A gold bar confirms it
is on. Each step then shows a dashed **Demo:** button:

1. On the passcode page: **Demo: use a demo passcode**. This uses shared-computer storage, which
   the browser clears when it closes.
2. On the audit page, press the **Demo:** button in the Next step card six times: fill the
   details → fill cycle 1 data → go to the change → record the change → fill the re-audit →
   close the loop. It works for any audit, including ones built live on stage.
3. Download the presentation from **Files for you**.

Example audits carry an "Example data" badge. **Clear example audits** in the bar deletes only
those, and **Turn off** ends demo mode. Screens: `screens/d1-passcode.jpg` to `screens/d4-my-audits.jpg`.

Suggested stage run (about 3 minutes): type a topic on the home page → show the built audit and
**Not quite right? Another audit** → **Choose this audit** → demo passcode → six **Demo:**
clicks → open the presentation.
