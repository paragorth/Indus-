# Proposed orthopaedic audits (151)

Designed from current standards and gaps in the corpus. Each has a data template in `templates/`. Not yet run anywhere.

## Hip and fragility fracture

**ONA-001. What proportion of adults having hip fracture surgery receive intravenous tranexamic acid before the start of surgery?** (under-audited)
- Standard: NICE NG24 rec 1.3.1 and 1.3.3 (2015, updated February 2026): "1.3.1 Offer tranexamic acid to adults having surgery in an operating theatre if: there is any risk of bleeding and the procedure will breach the skin or mucous membranes. [2026] ... 1.3.3 When using tranexamic acid for adults having surgery, administer it just before the start of surgery. Typically give 1 g by slow intravenous injection. [2026]" https://www.nice.org.uk/guidance/ng24/chapter/Reducing-requirement-for-blood-transfusion-for-people-having-surgery
- Pass: TXA given IV between induction and knife-to-skin, recorded on the anaesthetic chart; or a specific contraindication documented.. Target: ≥95%. Sample: 40 consecutive operated hip fracture patients over the last 6 weeks.
- Change: Add a default 'Tranexamic acid 1 g IV at induction' line to the hip fracture anaesthetic e-record or pre-printed chart, with an 'omitted because' box.
- Template: `new_audits/templates/ONA-001.csv`

**ONA-002. What proportion of red cell transfusions for non-bleeding hip fracture patients are given as a single unit followed by reassessment?** (under-audited)
- Standard: NICE NG24 rec 1.6.1 and 1.6.2 (2015, updated February 2026): "1.6.1 Consider single‑unit red blood cell transfusions for adults (or equivalent volumes calculated based on body weight for children or adults with low body weight) who do not have active bleeding. [2015] 1.6.2 After each single‑unit red blood cell transfusion (or equivalent volumes calculated based on body weight for children or adults with low body weight), clinically reassess and check haemoglobin levels, and give further transfusions if needed. [2015]" https://www.nice.org.uk/guidance/ng24/chapter/Red-blood-cell-transfusion
- Pass: One unit prescribed per episode, and a documented Hb check or clinical review before any further unit.. Target: ≥90%. Sample: 40 consecutive transfusion episodes in hip fracture patients over the last 3–4 months.
- Change: Set the EPMA red cell order for hip fracture patients to default to 1 unit, with a mandatory reason field for more than 1.
- Template: `new_audits/templates/ONA-002.csv`

**ONA-003. What proportion of older adults admitted with a pelvic fragility fracture receive pharmacological VTE prophylaxis for one month?** (new)
- Standard: NICE NG89 rec 1.11.2 (2018, updated 2019): "Offer VTE prophylaxis for a month to people with fragility fractures of the pelvis, hip or proximal femur if the risk of VTE outweighs the risk of bleeding. Choose either: LMWH, starting 6 to12 hours after surgery or fondaparinux sodium, starting 6 hours after surgery, providing there is low risk of bleeding. [2018]" https://www.nice.org.uk/guidance/ng89/chapter/Recommendations
- Pass: LMWH or fondaparinux prescribed in hospital and on discharge to cover 28 days or more from admission; or a documented reason (bleeding risk, already anticoagulated).. Target: ≥90%. Sample: 30 consecutive admissions over the last 6–9 months.
- Change: Add pelvic fragility fractures to the hip fracture discharge letter template so the 28-day LMWH prompt appears for them too.
- Template: `new_audits/templates/ONA-003.csv`

**ONA-004. What proportion of hip fracture patients whose surgery is delayed beyond the day after admission receive pre-operative pharmacological VTE prophylaxis?** (new)
- Standard: NICE NG89 rec 1.11.3 (2018, updated 2019): "Consider pre‑operative VTE prophylaxis for people with fragility fractures of the pelvis, hip or proximal femur if surgery is delayed beyond the day after admission . Give the last dose no less than 12 hours before surgery for LMWH or 24 hours before" https://www.nice.org.uk/guidance/ng89/chapter/Recommendations
- Pass: At least one pre-operative dose of LMWH (or fondaparinux) given after the day of admission, with the last dose ≥12 h (LMWH) before surgery; or a documented reason not to.. Target: ≥85%. Sample: 40 consecutive delayed patients from the last 3 months of NHFD data.
- Change: Add a rule to the hip fracture admission order set: if not on tomorrow's list, prescribe evening LMWH, with a 12-hour stop note.
- Template: `new_audits/templates/ONA-004.csv`

**ONA-005. What proportion of hip fracture patients who had delirium in hospital have the delirium diagnosis written in the discharge summary to the GP?** (new)
- Standard: NICE CG103 rec 1.6.4 (2010, updated 2023): "Ensure that the diagnosis of delirium is documented both in the person's record or notes, and in their primary care health record. [2010]" https://www.nice.org.uk/guidance/cg103/chapter/Recommendations
- Pass: The discharge summary sent to the GP names delirium (or acute confusional state) as a diagnosis or complication of this admission.. Target: ≥90%. Sample: 40 consecutive eligible discharges over the last 3–4 months.
- Change: Add a mandatory 'Delirium this admission: yes/no' field to the orthogeriatric discharge summary template.
- Template: `new_audits/templates/ONA-005.csv`

**ONA-006. What proportion of hip fracture patients with delirium still present at discharge have a plan for cognitive follow-up?** (new)
- Standard: NICE CG103 rec 1.7.5 (2010, updated 2023): "For people in whom delirium does not resolve: re-evaluate for underlying causes follow up and assess for possible dementia (see the NICE guideline on dementia ). [2010]" https://www.nice.org.uk/guidance/cg103/chapter/Recommendations
- Pass: Discharge summary or notes contain a named follow-up for cognition (GP review with repeat cognitive test, memory service referral, or geriatric clinic) with a timescale.. Target: ≥80%. Sample: 30 consecutive eligible discharges over the last 6 months.
- Change: Add an 'If delirium unresolved: cognitive follow-up plan' box to the orthogeriatric discharge template, triggered when the last 4AT is ≥4.
- Template: `new_audits/templates/ONA-006.csv`

**ONA-007. What proportion of hip fracture inpatients staying longer than 7 days have their malnutrition screen repeated every week?** (under-audited)
- Standard: NICE CG32 rec 1.2.2 (2006, updated 2017): "All hospital inpatients on admission and all outpatients at their first clinic appointment should be screened. Screening should be repeated weekly for inpatients and when there is clinical concern for outpatients." https://www.nice.org.uk/guidance/cg32/chapter/Recommendations
- Pass: A MUST (with a recorded weight) on admission and at least every 7 days until discharge.. Target: ≥90%. Sample: 40 consecutive discharges over the last 2 months.
- Change: Set a recurring 7-day MUST task in the nursing e-record for the hip fracture ward, shown on the ward whiteboard ('MUST Monday').
- Template: `new_audits/templates/ONA-007.csv`

**ONA-008. What proportion of hip fracture patients receive a nerve block within 4 hours of arrival at hospital?** (under-audited)
- Standard: NHFD KPI 0 definition (Royal College of Physicians FFFAP, 2021 onwards): "KPI 0. - Admitted to Orthgeriatric Ward (All NHFD Hospitals) Given a nerve block and admitted to an appropriate orthopaedic or orthogeriatric ward within 4 hours of presentation" https://www.nhfd.co.uk/20/nhfdcharts.nsf/vwInfo/KPI0-Admission
- Pass: Fascia iliaca, PENG or femoral block documented with a time ≤4 h after ED arrival; or a contraindication documented.. Target: ≥80%. Sample: 40 consecutive ED presentations over the last 6 weeks.
- Change: Add 'nerve block' to the ED hip fracture triage order set so it is prescribed at X-ray confirmation, with a time field on the proforma.
- Template: `new_audits/templates/ONA-008.csv`

**ONA-009. What proportion of hip fracture patients have their pain reassessed within 30 minutes of their first analgesia in ED?** (under-audited)
- Standard: NICE CG124 rec 1.3.1 (2011, guideline updated 2023): "Assess the person's pain: immediately upon presentation at hospital and within 30 minutes of administering initial analgesia and hourly until settled on the ward and regularly as part of routine nursing observations throughout admission. [2011]" https://www.nice.org.uk/guidance/cg124/chapter/Recommendations
- Pass: A documented pain score within 30 minutes after the time the first analgesic dose was given.. Target: ≥85%. Sample: 40 consecutive ED presentations over the last 6 weeks.
- Change: Make the ED system create an automatic 'pain score due' task 30 minutes after any analgesic is recorded as given.
- Template: `new_audits/templates/ONA-009.csv`

**ONA-010. What proportion of hip fracture patients are prescribed regular 6-hourly paracetamol before surgery?** (new)
- Standard: NICE CG124 rec 1.3.4 (2011, guideline updated 2023): "Offer paracetamol every 6 hours preoperatively unless contraindicated. [2011]" https://www.nice.org.uk/guidance/cg124/chapter/Recommendations
- Pass: Paracetamol prescribed as a regular (6-hourly) dose on the admission drug chart before surgery; or a contraindication documented.. Target: ≥95%. Sample: 40 consecutive admissions over the last 6 weeks.
- Change: Build a hip fracture analgesia order set in EPMA with regular weight-banded paracetamol ticked by default.
- Template: `new_audits/templates/ONA-010.csv`

**ONA-011. What proportion of stable trochanteric hip fractures are fixed with a sliding hip screw?** (under-audited)
- Standard: NICE CG124 rec 1.6.9 (2011, amended 2023); NICE QS16 statement 4: "Use extramedullary implants such as a sliding hip screw in preference to an intramedullary nail in people with trochanteric fractures above and including the lesser trochanter (except reverse oblique). [2011, amended 2023]" https://www.nice.org.uk/guidance/cg124/chapter/Recommendations
- Pass: Trochanteric fracture above and including the lesser trochanter (not reverse oblique) fixed with a sliding hip screw; or a documented reason for a nail.. Target: ≥90%. Sample: 40 consecutive cases over the last 4–6 months (NHFD extract).
- Change: Add 'fracture pattern and NICE implant (1.6.9)' fields to the trauma meeting list, to be confirmed by the consultant when booking.
- Template: `new_audits/templates/ONA-011.csv`

**ONA-012. What proportion of hemiarthroplasties for displaced intracapsular hip fracture use the hospital's single nominated cemented femoral stem?** (new)
- Standard: NICE CG124 rec 1.6.5 (2023): "Hospitals should aim to use a single type of cemented femoral component for hemiarthroplasties as standard treatment for displaced intracapsular hip fracture management. [2023]" https://www.nice.org.uk/guidance/cg124/chapter/Recommendations
- Pass: The hemiarthroplasty used the hospital's nominated cemented stem; or a documented reason for a different implant.. Target: ≥90%. Sample: 40 consecutive hemiarthroplasties over the last 3–4 months.
- Change: Agree one nominated cemented stem in writing and set theatre stores so only that stem is on the trauma shelf; others by request.
- Template: `new_audits/templates/ONA-012.csv`

**ONA-013. What proportion of patients with suspected hip fracture and normal X-rays have an MRI within 24 hours?** (under-audited)
- Standard: NICE CG124 rec 1.1.1 (2011, amended 2014): "Offer MRI if hip fracture is suspected despite negative X‑rays of the hip of an adequate standard. If MRI is not available within 24 hours or is contraindicated, consider CT. [2011, amended 2014]" https://www.nice.org.uk/guidance/cg124/chapter/Recommendations
- Pass: MRI done within 24 h of the negative X-ray; or CT within 24 h where MRI is contraindicated or not available.. Target: ≥90%. Sample: 30 consecutive cases over the last 6–12 months.
- Change: Agree a protected daily MRI slot for occult hip fracture with radiographer vetting and the MRI safety form in the ED order set.
- Template: `new_audits/templates/ONA-013.csv`

**ONA-014. What proportion of hip fracture patients taking psychotropic medicines have a documented falls-focused review of them before discharge?** (under-audited)
- Standard: NICE NG249 recs 1.1.7 and 1.2.2 (April 2025): "1.1.7 Offer a comprehensive falls assessment and comprehensive falls management to people in hospital inpatient settings and residential care settings . ... 1.2.2 Include the following assessments and examinations (where appropriate) in the comprehensive falls assessment to identify the person's individual fall risk factors: ... Medication review." https://www.nice.org.uk/guidance/ng249/chapter/Recommendations
- Pass: Each psychotropic on admission has a documented decision (stop, reduce, or continue with reason) by a doctor or pharmacist before discharge.. Target: ≥90%. Sample: 40 consecutive eligible discharges over the last 2–3 months.
- Change: Add a 'falls-risk medicines: stop / reduce / continue because' table to the orthogeriatric review template, pre-filled from EPMA.
- Template: `new_audits/templates/ONA-014.csv`

**ONA-015. What proportion of adults aged 50 and over with a low-energy wrist fracture have a fracture risk assessment?** (under-audited)
- Standard: NICE NG259 rec 1.1.1 (July 2026): "Assess fragility fracture risk in all people aged 50 and over and women who have experienced menopause with either of the following risk factors: a previous fragility fracture current or frequent use of systemic glucocorticoids" https://www.nice.org.uk/guidance/ng259/chapter/Fragility-fracture-risk-assessment
- Pass: FRAX or QFracture recorded, or referral to the fracture liaison service documented, within 12 weeks of the fracture.. Target: ≥90%. Sample: 40 consecutive virtual fracture clinic or fracture clinic cases over the last 2 months.
- Change: Add an automatic FLS referral tick box to the virtual fracture clinic outcome form, pre-set to 'yes' for age ≥50 with low-energy mechanism.
- Template: `new_audits/templates/ONA-015.csv`

**ONA-016. What proportion of adults aged 65 and over with a fall-related wrist or shoulder fracture are offered a comprehensive falls assessment?** (new)
- Standard: NICE NG249 rec 1.1.3 (April 2025): "Offer a comprehensive falls assessment and comprehensive falls management to people who have fallen in the last year and meet any of the following criteria (this can be carried out in the same service or involve an appropriate referral): ... Were injured in a fall and needed medical (including surgical) treatment." https://www.nice.org.uk/guidance/ng249/chapter/Recommendations
- Pass: Referral to a falls service, or a documented comprehensive falls assessment, within 16 weeks of the fracture.. Target: ≥80%. Sample: 40 consecutive fracture clinic or virtual clinic cases over the last 2 months.
- Change: Add a falls service referral option to the virtual fracture clinic outcome form, pre-set for age ≥65 with a fall.
- Template: `new_audits/templates/ONA-016.csv`

**ONA-017. What proportion of hip fracture patients recommended intravenous zoledronate receive the first dose before discharge?** (under-audited)
- Standard: NHFD KPI 7 definition (Royal College of Physicians FFFAP, 2021 onwards): "KPI 7. - Bone strengthening medication Given a suitable form of bone strengthening treatment and followed up to ensure that they are still receiving this protection at 120 days after fracture." https://www.nhfd.co.uk/20/nhfdcharts.nsf/vwInfo/KPI7-Medication
- Pass: Zoledronate given before discharge; or a documented reason to defer (low vitamin D, low calcium, eGFR <35, dental issue) with a date for the dose.. Target: ≥80%. Sample: 40 consecutive eligible discharges over the last 2–3 months.
- Change: Add a zoledronate line to the hip fracture order set that is due on post-op day 3–5 once vitamin D and calcium are checked, with a deferral reason box.
- Template: `new_audits/templates/ONA-017.csv`

**ONA-018. What proportion of older adults with a periprosthetic or non-hip femoral fragility fracture are seen by a senior geriatrician within 72 hours?** (under-audited)
- Standard: NHFD KPI 1 definition (RCP FFFAP); BOA Non-Ambulatory Fragility Fracture review principles: "1. Prompt orthogeriatric assessment - Assessed by a senior geriatrician (ST3+) within 72 hours of presentation. BOA: Although originally labelled a “hip fracture review”, we have always recommended that the principles of care should apply equally to all patients with similar clinical needs and the reviews are now formally targeted at the care of all patients with non-ambulatory fragility fractures (NAFFs)." https://www.nhfd.co.uk/20/nhfdcharts.nsf/vwInfo/KPIsOverview
- Pass: Documented review by a geriatrician ST3 or above within 72 hours of arrival.. Target: ≥90%. Sample: 30 consecutive admissions over the last 9–12 months.
- Change: Put all femoral fragility fractures on the orthogeriatric daily referral list automatically from the trauma meeting list.
- Template: `new_audits/templates/ONA-018.csv`

**ONA-019. What proportion of DXA scans done after a fragility fracture in women aged 60 and over or men aged 70 and over include a vertebral fracture assessment?** (new)
- Standard: NICE NG259 rec 1.5.1 (July 2026): "Consider doing a VFA when doing a DXA scan for: men aged 70 and over women aged 60 and over." https://www.nice.org.uk/guidance/ng259/chapter/Identifying-vertebral-fragility-fractures
- Pass: VFA done and reported with the DXA; or a documented reason not to (spinal imaging in past 3 months, scoliosis, body mass limit).. Target: ≥80%. Sample: 40 consecutive FLS DXA scans over the last 1–2 months.
- Change: Make VFA the default protocol on the DXA request form for FLS patients in the NICE age groups.
- Template: `new_audits/templates/ONA-019.csv`

**ONA-020. What proportion of adults aged 50 and over with a newly reported vertebral fracture on CT are identified by the fracture liaison service?** (under-audited)
- Standard: Royal Osteoporosis Society Clinical Standards for FLS, standard 1.1 (September 2025): "1.1. The FLS identifies people aged 50 years or older presenting with a new fragility fracture. This includes: • Newly identified vertebral fracture. (a prevalent vertebral fragility fracture which has not been previously documented)" https://royal-osteoporosis-society.uksouth01.umbraco.io/media/arqkorn2/ros-clinical-standards-for-fracture-liaison-service.pdf
- Pass: Patient appears on the FLS database or has an FLS referral within 12 weeks of the CT report.. Target: ≥80%. Sample: 40 consecutive reports over the last 2–3 months.
- Change: Set a RIS keyword rule that copies reports with a new vertebral fracture in patients ≥50 to the FLS inbox automatically.
- Template: `new_audits/templates/ONA-020.csv`

## Adult trauma

**ONA-021. What proportion of adults admitted with a tibial fracture have a documented compartment syndrome check at least hourly for the first 24 hours?** (new)
- Standard: BOAST Diagnosis and Management of Compartment Syndrome of the Extremities (updated July 2025): "Patients at risk of ACS should be assessed hourly with documentation of findings (whether present or not), an interpretation of these findings and rationale for management." https://www.boa.ac.uk/resource/boast-10-pdf.html
- Pass: In the first 24 hours after admission (or after fixation if sooner), no gap longer than 75 minutes between documented compartment checks, each recording pain and passive stretch findings.. Target: ≥90%. Sample: 40 consecutive eligible admissions over the last 4–6 months
- Change: Add a mandatory hourly compartment chart to the EPR (or a printed RCN-style chart in the admission pack) triggered by any tibial fracture admission.
- Template: `new_audits/templates/ONA-021.csv`

**ONA-022. What proportion of adults discharged within 48 hours of a tibial fracture (or its fixation) have documented advice on self-monitoring for compartment syndrome?** (new)
- Standard: NICE NG37 rec 1.2.7 (2016, guideline updated 2022): "In people with fractures of the tibia, maintain awareness of compartment syndrome for 48 hours after injury or fixation by: regularly assessing and recording clinical symptoms and signs in hospital; considering continuous compartment pressure monitoring in hospital when clinical symptoms and signs cannot be readily identified (for example, because the person is unconscious or has a nerve block); advising people how to self‑monitor for symptoms of compartment syndrome, when they leave hospital." https://www.nice.org.uk/guidance/ng37/chapter/Recommendations
- Pass: Written or documented verbal advice naming compartment syndrome symptoms and where to go, recorded in ED notes, discharge summary or leaflet log.. Target: ≥90%. Sample: 40 consecutive eligible patients over the last 3 months
- Change: Add a compartment syndrome warning paragraph to the EPR discharge template for tibial fractures, plus a one-page leaflet.
- Template: `new_audits/templates/ONA-022.csv`

**ONA-023. What proportion of adults having tibial fracture fixation under a peripheral nerve block have a documented joint decision that mentions compartment syndrome monitoring?** (new)
- Standard: BOAST Diagnosis and Management of Compartment Syndrome of the Extremities (updated July 2025): "The use of regional anaesthesia in extremity trauma: should follow joint decision making involving the patient (where able), anaesthetist and surgeon and include documented consent. should have an agreed policy that includes responsibility for post operative monitoring for compartment syndrome." https://www.boa.ac.uk/resource/boast-10-pdf.html
- Pass: Anaesthetic or surgical record shows the surgeon and anaesthetist agreed the block, the patient consented, and compartment syndrome risk or monitoring plan is written down.. Target: ≥90%. Sample: 30–40 consecutive cases over the last 6 months
- Change: Add a tick box to the anaesthetic chart: 'Block agreed with surgeon; compartment syndrome risk discussed; monitoring plan: ___'.
- Template: `new_audits/templates/ONA-023.csv`

**ONA-024. What proportion of pelvic binders are centred at the level of the greater trochanters on the first trauma image?** (new)
- Standard: BOAST The Management of Patients with Pelvic Fractures (January 2018): "When there is a suspected active bleeding from a pelvic fracture, apply a pelvic binder in the correct position. This should be applied prehospital." https://www.boa.ac.uk/resource/boast-3-pdf.html
- Pass: Binder buckle or band centred over the greater trochanters on the first pelvic radiograph or CT scanogram (operational definition of 'correct position').. Target: ≥90%. Sample: All eligible cases over 12 months (expect 30–50 in a trauma unit; use national trauma registry list)
- Change: Add a 'binder level check on scanogram' line to the trauma CT reporting template and trauma team checklist, and share results with the ambulance service.
- Template: `new_audits/templates/ONA-024.csv`

**ONA-025. What proportion of pelvic binders are removed within 24 hours of application?** (under-audited)
- Standard: NICE NG37 rec 1.2.17 (2016): "Remove all pelvic binders within 24 hours of application." https://www.nice.org.uk/guidance/ng37/chapter/Recommendations
- Pass: Documented binder removal time within 24 hours of the documented application time.. Target: ≥90%. Sample: All eligible cases over 12 months (30–50 expected)
- Change: Add a binder start-time sticker and a 'binder review due' alert to the EPR at application.
- Template: `new_audits/templates/ONA-025.csv`

**ONA-026. What proportion of adults with a high-energy pelvic ring fracture receive IV tranexamic acid within 1 hour of injury?** (new)
- Standard: BOAST The Management of Patients with Pelvic Fractures (January 2018): "All patients require IV Tranexamic Acid as soon as possible and ideally within an hour of injury." https://www.boa.ac.uk/resource/boast-3-pdf.html
- Pass: First TXA dose given (prehospital or hospital) within 60 minutes of the recorded injury time.. Target: ≥90% within 1 hour; 100% within 3 hours. Sample: All eligible cases over 12–18 months (30–40 expected), from trauma registry list
- Change: Add TXA as a default line in the ED trauma booklet or major trauma order set, prescribed on arrival unless ticked 'given prehospital' or contraindicated.
- Template: `new_audits/templates/ONA-026.csv`

**ONA-027. What proportion of adults with a high-energy pelvic ring fracture have a documented perineal, genital and rectal examination?** (under-audited)
- Standard: BOAST The Management of Urological Trauma Associated with Pelvic Fractures (August 2016): "All patients suffering high-energy trauma must have examination of the perineum and genitalia plus a rectal examination and the findings recorded in the medical records." https://www.boa.ac.uk/resource/boast-14-pdf.html
- Pass: Findings of perineum, genitalia and rectal examination recorded (or a documented reason it was deferred and when it was done).. Target: ≥90%. Sample: All eligible cases over 12–18 months (30–40 expected)
- Change: Add a perineum/genitalia/rectal line to the secondary survey section of the trauma booklet.
- Template: `new_audits/templates/ONA-027.csv`

**ONA-028. What proportion of older adults with a displaced fragility pelvic ring fracture who cannot mobilise are discussed with the specialist pelvic centre?** (new)
- Standard: BOAST The Management of Patients with Pelvic Fractures (January 2018): "Patients who suffer displaced low energy fragility fractures of the pelvic ring, who are unable to mobilise due to pain, should be discussed with the specialist centre for consideration of surgical stabilisation." https://www.boa.ac.uk/resource/boast-3-pdf.html
- Pass: Documented discussion (referral system, phone note or letter) with the regional pelvic service during the admission.. Target: ≥90%. Sample: 30–40 consecutive eligible admissions over the last 6–12 months (orthopaedic and medical wards)
- Change: Add a day-5 physiotherapy prompt: 'Not mobilising with pelvic ring fracture — refer to pelvic centre' on the trauma ward round checklist.
- Template: `new_audits/templates/ONA-028.csv`

**ONA-029. What proportion of adults with a cervical spine fracture or subluxation have CT angiography of the neck vessels?** (new)
- Standard: BOAST Assessment of the Spine in the Trauma Patient (April 2025): "Fracture, subluxation or ligamentous injury of the cervical spine requires CT angiography to exclude blunt cerebrovascular injury (Denver criteria)" https://www.boa.ac.uk/resource/boast-assessment-of-the-spine-in-the-trauma-patient.html
- Pass: CT angiography of the neck performed during the index admission (or a documented senior reason not to).. Target: ≥90%. Sample: 40 consecutive cases over the last 12 months
- Change: Add an auto-prompt to the radiology report template: 'Cervical fracture: CTA recommended (BOAST 2025)', and add arterial phase to the trauma CT protocol when cervical injury is seen.
- Template: `new_audits/templates/ONA-029.csv`

**ONA-030. What proportion of adults with a new traumatic spinal column fracture have imaging of the whole spine?** (new)
- Standard: NICE NG41 rec 1.5.11 (2016): "If a new spinal column fracture is confirmed, image the rest of the spinal column." https://www.nice.org.uk/guidance/ng41/chapter/Recommendations
- Pass: Remaining spinal regions covered by CT, MRI or X-ray within the admission (whole-body CT counts if it includes the whole spine with reformats).. Target: ≥90%. Sample: 40 consecutive cases over the last 6 months
- Change: Add a standard line to spine CT and X-ray reports: 'New fracture — recommend imaging of the remaining spine (NICE NG41 1.5.11)'.
- Template: `new_audits/templates/ONA-030.csv`

**ONA-031. What proportion of adults with a spinal fracture treated non-operatively have a documented stability decision and orthosis plan?** (under-audited)
- Standard: BOAST Assessment of the Spine in the Trauma Patient (April 2025): "If a fracture is to be treated non-operatively, the decision-making team should specify the degree of stability of the fracture and details of the planned non-operative management, such as use of collar/ brace, including duration, care and changing procedures." https://www.boa.ac.uk/resource/boast-assessment-of-the-spine-in-the-trauma-patient.html
- Pass: Notes record stability (stable/unstable), orthosis or no orthosis, duration, and care/changing instructions.. Target: ≥90%. Sample: 40 consecutive cases over the last 6 months
- Change: Use a one-page 'non-operative spinal fracture plan' EPR form with the four fields, copied to patient and GP.
- Template: `new_audits/templates/ONA-031.csv`

**ONA-032. What proportion of adults with an ankle fracture of uncertain stability have a weight-bearing radiograph within 2 weeks of injury?** (new)
- Standard: BOAST The Management of Ankle Fractures (August 2016): "In fracture patterns where stability is uncertain, patients should be reviewed within 2 weeks with further radiographs (weight bearing if possible) to confirm the position remains acceptable." https://www.boa.ac.uk/resource/boast-12-pdf.html
- Pass: Radiograph labelled weight-bearing (or documented reason it was not possible) taken within 14 days of injury.. Target: ≥85%. Sample: 40 consecutive cases over the last 3 months
- Change: Add a pre-set radiology request 'Ankle AP/mortise/lateral WEIGHT-BEARING' to the virtual fracture clinic outcome menu for uncertain-stability ankle fractures.
- Template: `new_audits/templates/ONA-032.csv`

**ONA-033. What proportion of adults under 60 with an unstable ankle fracture listed for fixation have surgery on the day of injury or the next day?** (under-audited)
- Standard: BOAST The Management of Ankle Fractures (August 2016); NICE NG38 rec 1.4.2 (2016): "Early fixation (on the day or day after injury) is recommended in the majority of patients under 60 years when the ankle mortise is unstable." https://www.boa.ac.uk/resource/boast-12-pdf.html
- Pass: Knife-to-skin on the calendar day of injury or the following day.. Target: ≥80%. Sample: 40 consecutive cases over the last 4 months
- Change: Create a protected 'first on the list' slot or ambulatory ankle slot on the next-day trauma list for under-60 unstable ankles.
- Template: `new_audits/templates/ONA-033.csv`

**ONA-034. What proportion of adults with a knee dislocation, tibial plateau or distal femoral fracture have timed pulse findings documented for both legs in ED?** (under-audited)
- Standard: NICE NG37 rec 1.3.7 (2016); BOAST Diagnosis and management of arterial injuries associated with musculoskeletal trauma (June 2026): "When assessing neurovascular status in a person with a limb injury, document for both limbs: which nerves and nerve function have been assessed and when; the findings, including: sensibility; motor function using the Medical Research Council (MRC) grading system; which pulses have been assessed and when; how circulation has been assessed when pulses are not accessible. Document and time each repeated assessment." https://www.nice.org.uk/guidance/ng37/chapter/Recommendations
- Pass: Dorsalis pedis and posterior tibial pulses documented for both legs with a time, at first ED assessment.. Target: ≥90%. Sample: 40 consecutive cases over the last 6 months
- Change: Add a two-column (left/right) timed pulse box to the ED and orthopaedic clerking templates for lower limb injuries.
- Template: `new_audits/templates/ONA-034.csv`

**ONA-035. What proportion of adults having humeral shaft or distal humerus fixation have a named-nerve examination documented by the operating team after surgery?** (new)
- Standard: BOAST Peripheral Nerve Injury (December 2021), standard 1.1.4: "post-operatively, by the operating surgeon following any procedure where nerve injury is a recognised risk" https://www.boa.ac.uk/resource/boast-peripheral-nerve-injury.html
- Pass: Radial, median and ulnar motor and sensory findings recorded by the operating team on the day of surgery (or first time the block wears off).. Target: ≥90%. Sample: 30–40 consecutive cases over the last 12 months
- Change: Add a mandatory 'post-op nerve check (radial/median/ulnar)' field to the electronic post-op instructions for upper limb fracture surgery.
- Template: `new_audits/templates/ONA-035.csv`

**ONA-036. What proportion of adults having a native joint aspirated for suspected septic arthritis have a sample sent for crystal microscopy?** (new)
- Standard: BOAST Management of musculoskeletal soft tissue infections (July 2025): "Native joint infections should be treated according to standards outlined in the BOASt for the acute management of periprosthetic joint infection2, recognising the time critical nature of chondral injury. Aspirate should include samples for crystallography." https://www.boa.ac.uk/resource/management-of-musculoskeletal-soft-tissue-infections.html
- Pass: Laboratory record shows crystal microscopy requested on the joint aspirate.. Target: ≥95%. Sample: 40 consecutive aspirates over the last 6 months
- Change: Create a 'synovial fluid' order set in the lab system that requests MC&S and crystals together.
- Template: `new_audits/templates/ONA-036.csv`

**ONA-037. What proportion of systemically well adults with a suspected wound infection after fracture fixation are reviewed by a consultant before antibiotics start?** (new)
- Standard: BOAST Fracture Related Infections (September 2019): "A patient who is not systemically unwell should be reviewed by a consultant in a clinic within 48 hours. Antibiotic treatment should not be commenced before that review." https://www.boa.ac.uk/resource/boast-fracture-related-infections.html
- Pass: Consultant review documented within 48 hours of first presentation, with no antibiotic started before that review.. Target: ≥80%. Sample: 30–40 consecutive cases over the last 12 months
- Change: Set up a 'leaky wound' pathway: patient and GP advice line to a next-day consultant slot, plus a 'do not start antibiotics unless septic' line on the discharge letter.
- Template: `new_audits/templates/ONA-037.csv`

**ONA-038. What proportion of discharge summaries after fracture fixation tell the patient and GP what to do if infection is suspected?** (new)
- Standard: BOAST Fracture Related Infections (September 2019): "There should be readily available guidance for primary carers and patients on how to respond in the event of a suspected fracture related infection. This should be included in discharge documentation." https://www.boa.ac.uk/resource/boast-fracture-related-infections.html
- Pass: Discharge summary names infection warning signs and a direct contact route to the orthopaedic team.. Target: ≥90%. Sample: 50 consecutive discharges over the last 2 months
- Change: Add a fixed FRI advice paragraph and contact number to the orthopaedic EPR discharge template.
- Template: `new_audits/templates/ONA-038.csv`

**ONA-039. What proportion of adults with an open fracture have a wound photograph stored in the record before debridement?** (under-audited)
- Standard: BOAST Open Fractures (December 2017); NICE NG37 recs 1.3.5–1.3.6 (2016): "Photographs of open fracture wounds should be taken when they are first exposed for clinical care, before debridement and at other key stages of management. These should be kept in the patient’s records." https://www.boa.ac.uk/resource/boast-4-pdf.html
- Pass: At least one wound photograph, dated before the debridement start time, retrievable in the EPR or clinical photography system.. Target: ≥90%. Sample: 40 consecutive cases over the last 6–12 months
- Change: Enable the trust secure photo app on ED devices, with 'open fracture photo uploaded' as a required box on the ED trauma or referral form.
- Template: `new_audits/templates/ONA-039.csv`

**ONA-040. What proportion of adult trauma-team patients admitted under orthopaedics have a documented tertiary survey within 24 hours of admission?** (under-audited)
- Standard: South-East Scotland Major Trauma Guidelines, Tertiary Survey (NHS Right Decision Service, v1.0, reviewed 04/01/2025); use your own network guideline if it differs: "Within 24 hours of admission 'after the dust has settled'." https://rightdecisions.scot.nhs.uk/south-east-scotland-major-trauma-guidelines/ongoing-care/tertiary-survey/
- Pass: A structured head-to-toe tertiary survey entry dated within 24 hours of admission (repeated when awake if conscious level was reduced).. Target: ≥90%. Sample: 40 consecutive admissions over the last 4–6 months
- Change: Put a tertiary survey proforma in the EPR and make 'tertiary survey done' a checklist item on the morning trauma meeting list.
- Template: `new_audits/templates/ONA-040.csv`

**ONA-041. What proportion of adults admitted with a complex fracture have a written summary sent to their GP within 24 hours of admission?** (new)
- Standard: NICE NG37 rec 1.3.3 (2016): "Produce a written summary, which gives the diagnosis, management plan and expected outcome, and: is aimed at and sent to the patient's GP within 24 hours of admission; includes a summary written in plain English that is understandable by patients, family members and carers; is readily available in the patient's records." https://www.nice.org.uk/guidance/ng37/chapter/Recommendations
- Pass: A letter or electronic summary with diagnosis, plan and expected outcome sent to the GP within 24 hours of admission.. Target: ≥80%. Sample: 40 consecutive admissions over the last 6 months
- Change: Create an EPR 'admission notification to GP' template auto-populated from the post-take ward round, sent by the trauma coordinator.
- Template: `new_audits/templates/ONA-041.csv`

## Paediatric orthopaedics

**ONA-042. In children who have K-wire fixation of a supracondylar fracture, does the operation note state when the wires will come out?** (new)
- Standard: BOAST Supracondylar Fractures of the Humerus in Children (October 2020), standard 12: "The operating surgeon should determine and document the need for post-operative radiographs and anticipated time of wire removal." https://www.boa.ac.uk/resource/boast-11-pdf.html
- Pass: Operation note records the anticipated time of wire removal (e.g. '3–4 weeks, in clinic'). The need for post-operative X-rays is collected as a secondary item.. Target: ≥90%. Sample: 40 consecutive cases from the theatre system, going back up to 18 months.
- Change: Add two mandatory drop-down fields to the paediatric K-wire operation note template: 'wire removal at __ weeks' and 'post-op X-ray: yes/no'.
- Template: `new_audits/templates/ONA-042.csv`

**ONA-043. When a medial K-wire is used for a child's supracondylar fracture, does the operation note record how the ulnar nerve was protected?** (new)
- Standard: BOAST Supracondylar Fractures of the Humerus in Children (October 2020), standard 6: "When a medial wire is used, techniques to avoid ulnar nerve injury should be employed and recorded on the operation note." https://www.boa.ac.uk/resource/boast-11-pdf.html
- Pass: Operation note names a technique (e.g. mini-open medial incision, elbow extended when the medial wire goes in, nerve palpated and held back).. Target: 100%. Sample: All medial-wire cases over the last 24 months (expect 20–40 in a trauma unit; pool across 2 sites if fewer).
- Change: Add a conditional field to the paediatric K-wire op-note template: if 'medial wire = yes', the note cannot be saved until an ulnar protection technique is chosen.
- Template: `new_audits/templates/ONA-043.csv`

**ONA-044. When a child's supracondylar fracture is fixed at night, is the urgent reason for night surgery documented?** (under-audited)
- Standard: BOAST Supracondylar Fractures of the Humerus in Children (October 2020), standard 2: "Surgical management should be carried out on the day of injury. Night-time operating is not necessary unless there are indications for urgent surgery which should be documented." https://www.boa.ac.uk/resource/boast-11-pdf.html
- Pass: For a case with knife-to-skin between 22:00 and 07:59, the notes or booking record an urgent indication (absent radial pulse, impaired hand perfusion, open injury or threatened skin, or other stated reason).. Target: ≥90% of night cases with a documented indication. Sample: All night-time cases over the last 24 months (expect 20–40); also record total cases to give the night-time rate.
- Change: Add a mandatory 'urgent indication' drop-down to the emergency theatre booking form for any paediatric case booked for 22:00–07:59.
- Template: `new_audits/templates/ONA-044.csv`

**ONA-045. Do children who have a forearm fracture manipulated in the Emergency Department have a documented consultant review within 48 hours of injury?** (under-audited)
- Standard: BOAST Early Management of the Paediatric Forearm Fracture (May 2021), standard 12: "A documented review of the case and images by a consultant orthopaedic surgeon should occur within 48 hours of injury." https://www.boa.ac.uk/resource/boast-early-management-of-the-paediatric-forearm-fracture.html
- Pass: A note naming the consultant who reviewed the case and images, dated within 48 hours of the injury time.. Target: ≥95%. Sample: 40 consecutive cases over the last 3 months.
- Change: ED paediatric manipulation form auto-adds the child to the next trauma meeting list; trauma meeting proforma has a mandatory 'reviewing consultant' field.
- Template: `new_audits/templates/ONA-045.csv`

**ONA-046. Do children discharged after Emergency Department manipulation of a forearm fracture get a leaflet with red-flag symptoms and contact details?** (under-audited)
- Standard: BOAST Early Management of the Paediatric Forearm Fracture (May 2021), standard 11: "Oral analgesia to take home and dedicated information leaflets, that include red flag symptoms and contact details, should be provided. Prior to discharge, a fracture clinic appointment should be made to occur within 7 days of injury." https://www.boa.ac.uk/resource/boast-early-management-of-the-paediatric-forearm-fracture.html
- Pass: Discharge record states a dedicated cast/fracture leaflet with red flags and contact numbers was given. Analgesia advice and clinic ≤7 days are secondary items.. Target: ≥90%. Sample: 40 consecutive cases over the last 3 months.
- Change: Add a mandatory 'paediatric cast leaflet given' tick box to the ED discharge template for paediatric fracture diagnoses, with leaflets stocked in the plaster room.
- Template: `new_audits/templates/ONA-046.csv`

**ONA-047. Do children with a suspected limb fracture and moderate or severe pain get analgesia within 30 minutes of arrival?** (under-audited)
- Standard: RCEM Pain in Children QIP 2021/22, standard 2 (fundamental): "2. Administration of analgesia to patients in severe pain F = within 30 minutes D = within 20 minutes 2. Administration of analgesia to patients in moderate pain F = within 30 minutes D = within 20 minutes" https://rcem.ac.uk/wp-content/uploads/2022/02/Pain_in_Children_QIP_Info_Pack_2021-22_v4.pdf
- Pass: First analgesic given within 30 minutes of arrival (or triage, if earlier) when the pain score is moderate (4–7) or severe (7–10).. Target: ≥75% within 30 minutes (RCEM fundamental standard). Sample: 40 consecutive children over the last 1–2 months.
- Change: Nurse-initiated analgesia at triage: a patient group direction for oral ibuprofen/paracetamol, prompted by the triage template when the pain score is 4 or more.
- Template: `new_audits/templates/ONA-047.csv`

**ONA-048. Do children with a limb fracture who receive analgesia in the Emergency Department have their pain re-scored within 60 minutes?** (under-audited)
- Standard: RCEM Pain in Children QIP 2021/22, standard 3 (fundamental): "3. Patients with severe or moderate pain should have documented evidence of re-evaluation and action within 60 minutes of receiving the first dose of analgesic" https://rcem.ac.uk/wp-content/uploads/2022/02/Pain_in_Children_QIP_Info_Pack_2021-22_v4.pdf
- Pass: A second documented pain score within 60 minutes of the first analgesic dose, with action if pain remains moderate or severe.. Target: ≥75%. Sample: 40 consecutive children over the last 1–2 months.
- Change: EPR task: giving an analgesic to a child with pain score 4 or more automatically creates a 'repeat pain score due' task at 45 minutes on the nursing board.
- Template: `new_audits/templates/ONA-048.csv`

**ONA-049. Do children with a suspected long-bone fracture and moderate or severe pain receive an intranasal or intravenous opioid in the Emergency Department?** (new)
- Standard: NICE NG38 Fractures (non-complex) rec 1.1.8 (2016): "For the initial management of pain in children (under 16s) with suspected long bone fractures of the legs (femur, tibia, fibula) or arms (humerus, radius, ulna), offer: oral ibuprofen, or oral paracetamol, or both for mild to moderate pain intranasal or intravenous opioids for moderate to severe pain (use intravenous opioids if intravenous access has been established)." https://www.nice.org.uk/guidance/ng38/chapter/Recommendations
- Pass: Intranasal or IV opioid given (or offered and declined, documented) when the pain score is 7 or more, or moderate pain persists after oral analgesia.. Target: ≥85%. Sample: 40 consecutive eligible children over the last 2–3 months.
- Change: Add intranasal diamorphine/fentanyl to the ED paediatric fracture order set, pre-filled by weight, triggered when a pain score of 7 or more is entered.
- Template: `new_audits/templates/ONA-049.csv`

**ONA-050. Do children with a suspected displaced femoral shaft fracture receive a femoral nerve or fascia iliaca block in the Emergency Department?** (new)
- Standard: NICE NG38 Fractures (non-complex) rec 1.1.11 (2016): "Consider a femoral nerve block or fascia iliaca block in the emergency department for children (under 16s) with suspected displaced femoral fractures." https://www.nice.org.uk/guidance/ng38/chapter/Recommendations
- Pass: Femoral nerve or fascia iliaca block given in ED, or a documented reason why not.. Target: ≥80% block or documented reason. Sample: All cases over the last 36 months (expect 20–40 in a trauma unit; pool with a neighbouring site or MTC if fewer).
- Change: Add a 'block given / reason not given' sign-off box to the ED paediatric femoral fracture proforma, with a weight-based dosing card.
- Template: `new_audits/templates/ONA-050.csv`

**ONA-051. Do children with a femoral fracture have a documented safeguarding assessment before discharge?** (new)
- Standard: NICE NG38 Fractures (non-complex) rec 1.7.1 (2016): "Address issues of non-accidental injury before discharge in all children with femoral fractures. This is particularly important for children who are not walking or talking. For more information, see the NICE guideline on when to suspect child maltreatment." https://www.nice.org.uk/guidance/ng38/chapter/Recommendations
- Pass: Before discharge, the notes record a safeguarding assessment: mechanism consistent with injury and developmental stage, safeguarding records checked, and the outcome (no concern / referred / paediatric review).. Target: 100%. Sample: All cases over the last 36 months (expect 30–50 in a trauma unit).
- Change: Add a mandatory safeguarding section to the paediatric trauma admission proforma that must be completed before the discharge summary can be finalised.
- Template: `new_audits/templates/ONA-051.csv`

**ONA-052. In children under 2 years with a fracture, does the Emergency Department note record whether the explanation fits the injury?** (new)
- Standard: NICE CG89 Child maltreatment rec 1.1.9 (2009, updated 2025): "Suspect child maltreatment if a child has one or more fractures in the absence of a medical condition that predisposes to fragile bones (for example, osteogenesis imperfecta, osteopenia of prematurity) or if the explanation is absent or unsuitable." https://www.nice.org.uk/guidance/cg89/chapter/Recommendations
- Pass: ED note records the mechanism, the child's mobility or developmental stage, and an explicit judgement that the explanation is suitable or not, with the action taken.. Target: ≥95%. Sample: 40 consecutive cases over the last 6–12 months.
- Change: Mandatory safeguarding prompt in the ED EPR for any fracture diagnosis in a child under 2 (mechanism, mobility, 'explanation suitable? yes/no', action), in line with NICE NG38 rec 1.5.1.
- Template: `new_audits/templates/ONA-052.csv`

**ONA-053. Are skeletal surveys for suspected physical abuse done and reported within 24 hours of the request?** (new)
- Standard: RCR/SCoR The radiological investigation of suspected physical abuse in children (revised first edition, November 2018), recommendation 14: "The skeletal survey should be acquired and reported within 24 hours and certainly no later than 72 hours from the request being made." https://www.rcr.ac.uk/media/nznl1mv4/rcr-publications_the-radiological-investigation-of-suspected-physical-abuse-in-children-revised-first-edition_november-2018.pdf
- Pass: Time from request to final report is 24 hours or less. Over 72 hours is recorded separately as a serious breach.. Target: ≥90% within 24 hours; 100% within 72 hours. Sample: 40 consecutive initial skeletal surveys over the last 12 months.
- Change: Create a RIS priority code for 'suspected physical abuse skeletal survey' that books the next available paediatric slot and alerts the reporting radiologist automatically.
- Template: `new_audits/templates/ONA-053.csv`

**ONA-054. Do children who have a skeletal survey for suspected physical abuse have follow-up imaging within 11 to 14 days?** (new)
- Standard: RCR/SCoR The radiological investigation of suspected physical abuse in children (revised first edition, November 2018), recommendation 33: "Follow-up imaging should be performed ideally within 11 to 14 days, and no later than 28 days after the initial skeletal survey." https://www.rcr.ac.uk/media/nznl1mv4/rcr-publications_the-radiological-investigation-of-suspected-physical-abuse-in-children-revised-first-edition_november-2018.pdf
- Pass: Follow-up imaging done 11–14 days after the first survey. Done by 28 days is recorded as a secondary pass.. Target: ≥90% within 11–14 days; 100% by 28 days. Sample: 40 consecutive children over the last 12 months.
- Change: Book the follow-up survey at the time of the first survey, with a named professional in the safeguarding team to chase missed appointments (RCR rec 34).
- Template: `new_audits/templates/ONA-054.csv`

**ONA-055. Do children admitted with suspected bone or joint infection have FBC, CRP, blood cultures and ESR sent before antibiotics?** (new)
- Standard: BOAST The Management of Children with Acute Musculoskeletal Infection (May 2022), standard 6: "Essential haematological investigations include, in order, FBC, CRP, blood cultures*, ESR. Plain radiographs of the affected bone or joint are required. No single investigation algorithm is completely reliable: diagnosis should be considered in conjunction with history and examination." https://www.boa.ac.uk/resource/boast-the-management-of-children-with-acute-musculoskeletal-infection.html
- Pass: All four tests (FBC, CRP, blood culture, ESR) sent before the first antibiotic dose. Radiograph and culture volume ≥2 ml are secondary items.. Target: ≥90%. Sample: 40 consecutive admissions over the last 12 months.
- Change: Create an EPR 'suspected bone and joint infection – child' order set that requests FBC, CRP, blood culture, ESR and an X-ray together.
- Template: `new_audits/templates/ONA-055.csv`

**ONA-056. Do children with suspected osteomyelitis or complex joint infection who need an MRI get it within 48 hours of admission?** (new)
- Standard: BOAST The Management of Children with Acute Musculoskeletal Infection (May 2022), standard 7: "MRI is the preferred second line modality, ideally within 48 hours unless the clinical condition mandates more urgent imaging. Ultrasound should be performed within the same time frame." https://www.boa.ac.uk/resource/boast-the-management-of-children-with-acute-musculoskeletal-infection.html
- Pass: MRI done within 48 hours of admission.. Target: ≥85%. Sample: 40 consecutive children over the last 12–18 months.
- Change: Agree a protected daily paediatric MRI slot (with GA list access) for suspected bone and joint infection, bookable via a 'BJI – within 48 h' RIS priority code.
- Template: `new_audits/templates/ONA-056.csv`

**ONA-057. Do children with septic arthritis have surgical drainage within 24 hours of diagnosis?** (new)
- Standard: BOAST The Management of Children with Acute Musculoskeletal Infection (May 2022), standard 10: "Septic arthritis requires surgical drainage as soon as possible, ideally within 24 hours of diagnosis." https://www.boa.ac.uk/resource/boast-the-management-of-children-with-acute-musculoskeletal-infection.html
- Pass: Knife-to-skin (or arthroscopy start) within 24 hours of the documented clinical diagnosis or decision to operate.. Target: ≥90%. Sample: All cases over the last 24–36 months (expect 20–40 in a trauma unit; pool across a network if fewer).
- Change: Classify paediatric septic arthritis as NCEPOD category 2 on the emergency theatre booking form, so it is listed with a 24-hour target.
- Template: `new_audits/templates/ONA-057.csv`

**ONA-058. Do children treated for bone or joint infection have clinical and X-ray follow-up for at least 12 months?** (new)
- Standard: BOAST The Management of Children with Acute Musculoskeletal Infection (May 2022), standard 14: "Bone and joint infection should be followed up clinically and radiographically for a minimum of 12 months by a clinician able to identify long-term complications." https://www.boa.ac.uk/resource/boast-the-management-of-children-with-acute-musculoskeletal-infection.html
- Pass: A clinic review with a radiograph at 12 months or later after diagnosis (or documented plan still active if 12 months not yet reached).. Target: ≥90%. Sample: 40 consecutive children diagnosed 12–36 months ago.
- Change: Add a '12-month infection review with X-ray' field to the discharge letter template and an outpatient outcome code that blocks discharge from clinic before 12 months.
- Template: `new_audits/templates/ONA-058.csv`

**ONA-059. Is every child treated for a slipped upper femoral epiphysis discussed at a documented regional peer review meeting?** (new)
- Standard: BSCOS Best Practice in Children's Trauma & Orthopaedics in the UK (November 2025), Section 3 Key Points: Life Changing Injuries: "Specific paediatric injuries that could have potentially life-changing results should be discussed at PRMs. These should include amongst others: open fractures, pelvic fractures, femoral neck fractures, Slipped Upper Femoral Epiphysis (SUFE), lower limb physeal fractures, intra-articular fractures and pathological fractures or fractures occurring in children with underlying musculoskeletal conditions" https://www.bscos.org.uk/Portals/0/Best%20Practice%20in%20Children's%20T%20&%20O%20in%20the%20UK%20Final%20%20%20%20%205%2011%202025.pdf
- Pass: Minutes of a regional or network peer review meeting (or MDT) record discussion of the case.. Target: 100%. Sample: All cases over the last 36 months (expect 15–40 in a DGH; pool with network units if fewer).
- Change: SUFE procedure codes on the theatre system automatically add the case to the next network peer review meeting list, with a structured case template.
- Template: `new_audits/templates/ONA-059.csv`

**ONA-060. Do babies with a screen-positive newborn hip result have their hip ultrasound between 4 and 6 weeks of age?** (under-audited)
- Standard: NHS NIPE screening standards, NIPE-S03 (valid for data collected from 1 April 2024): "The proportion of babies with a screen positive newborn hip result who attend for ultrasound scan of the hips within the designated timescale." https://www.gov.uk/government/publications/newborn-and-infant-physical-examination-screening-standards/newborn-and-infant-physical-examination-screening-standards-valid-for-data-collected-from-1-april-2024
- Pass: Babies born at 34+0 weeks or later scanned at ≥4 weeks 0 days and ≤6 weeks 0 days of age (42 days); babies born before 34+0 weeks scanned at 38+0 to 40+0 weeks corrected age. Thresholds: acceptable ≥90%, achievable ≥95%.. Target: ≥95% (NIPE achievable). Sample: 40 consecutive screen-positive babies over the last 3–6 months.
- Change: At the newborn check, the NIPE practitioner books the hip scan directly into a RIS slot dated for day 28–35, and the date is printed on the parent's discharge letter.
- Template: `new_audits/templates/ONA-060.csv`

**ONA-061. Are infants with a screen-positive hip at the 6 to 8 week check seen by a paediatric orthopaedic surgeon by 10 weeks of age?** (new)
- Standard: NHS NIPE screening programme handbook (updated October 2025), section 14.11: "Infants with screen positive results following NIPE infant 6 to 8-week screening examination should be referred directly to paediatric orthopaedic surgeon for urgent expert opinion and be seen by 10 weeks of age." https://www.gov.uk/government/publications/newborn-and-infant-physical-examination-programme-handbook/newborn-and-infant-physical-examination-screening-programme-handbook
- Pass: First paediatric orthopaedic appointment attended on or before 70 days of age.. Target: ≥90%. Sample: All referrals over the last 12 months (expect 20–40).
- Change: Create an e-Referral triage category 'NIPE 6–8 week hip – see by 10 weeks' that shows the date-of-birth-based deadline to the vetting consultant and books into a reserved slot.
- Template: `new_audits/templates/ONA-061.csv`

## Foot and ankle, hand and wrist, spine

**ONA-062. In adults whose ankle fracture was reduced in ED, was the check X-ray reviewed and the reduction documented as adequate before they left ED?** (under-audited)
- Standard: BOAST The Management of Ankle Fractures (BOA, August 2016), Standards for Practice: "Adequate reduction must be confirmed by review of repeat radiographs and documented before transfer from ED." https://www.boa.ac.uk/resource/boast-12-pdf.html
- Pass: A post-reduction radiograph exists on PACS with a time before the ED departure time, AND the notes contain an entry reviewing it (adequate / not adequate) timed before departure.. Target: ≥90%. Sample: 40 consecutive eligible patients over the last 6 months (fewer in small units: take all).
- Change: Add a mandatory 'post-reduction X-ray reviewed: adequate / not adequate, by whom' line to the ED ankle reduction procedure note (EPR template), which must be completed before the ED discharge/transfer step.
- Template: `new_audits/templates/ONA-062.csv`

**ONA-063. In patients with a suspected Achilles tendon rupture, was a definitive treatment plan (surgery or a named non-operative protocol) documented within 72 hours of first presentation?** (under-audited)
- Standard: BOAST Outpatient and on-call services for people with fractures or musculoskeletal injury (BOA, February 2026): "A definitive treatment plan should be documented for all patients within 72 hours of first presentation and copied to the patient and their GP." https://www.boa.ac.uk/resource/boast-outpatient-and-on-call-services-for-people-with-fractures-or-musculoskeletal-injury.html
- Pass: A documented definitive plan (operative, or non-operative with a named boot/wedge protocol) timed ≤72 hours after first ED/UTC arrival.. Target: ≥80%. Sample: 40 consecutive patients over the last 6–9 months (all cases if fewer).
- Change: A single ED/VFC Achilles pathway sheet: equinus boot at ED, direct VFC referral flagged 'Achilles', and consultant plan at the next-day trauma meeting (no routine wait for ultrasound where clinical tests are clear).
- Template: `new_audits/templates/ONA-063.csv`

**ONA-064. In patients discharged in a boot or cast for an Achilles tendon rupture, was a VTE risk assessment documented at first contact?** (under-audited)
- Standard: BOAST Outpatient and on-call services for people with fractures or musculoskeletal injury (BOA, February 2026); links to NICE NG89 rec 1.11.1 (2018): "A documented risk assessment for venous thromboembolism (VTE) should be conducted for all patients with a lower limb injury. (NICE NG89)" https://www.boa.ac.uk/resource/boast-outpatient-and-on-call-services-for-people-with-fractures-or-musculoskeletal-injury.html
- Pass: A completed VTE risk assessment (tool or free text weighing VTE against bleeding risk) in the ED/UTC or first fracture clinic record, dated the day of first presentation.. Target: ≥95%. Sample: 40 consecutive patients over the last 6–9 months.
- Change: Build the lower-limb immobilisation VTE tool into the ED discharge step so the boot/cast discharge cannot be completed without it (hard stop).
- Template: `new_audits/templates/ONA-064.csv`

**ONA-065. In adults whose dorsally displaced distal radius fracture was manipulated in ED, was the manipulation done under regional anaesthesia (for example Bier's block) rather than a haematoma block?** (new)
- Standard: BOAST The Management of Distal Radial Fractures (BOA, December 2017); also NICE NG38 rec 1.3.1 (2016): "If manipulation is indicated, it should be undertaken using regional anaesthesia, performed by a suitably qualified and trained practitioner (as opposed to local haematoma block)." https://www.boa.ac.uk/resource/boast-16-pdf.html
- Pass: The procedure note records intravenous regional anaesthesia (Bier's block) or another regional block (for example axillary/supraclavicular). Haematoma block, sedation only or gas and air alone = fail.. Target: ≥80%. Sample: 40 consecutive patients over the last 3–6 months.
- Change: Stock a ready-made Bier's block pack in ED resus with a one-page procedure checklist, and put 'Bier's block' as the default option in the ED distal radius reduction template.
- Template: `new_audits/templates/ONA-065.csv`

**ONA-066. In patients aged 50 or over with a low-energy distal radius fracture, was a referral to the fracture liaison service (or a documented bone health and falls assessment) made?** (under-audited)
- Standard: BOAST The Management of Distal Radial Fractures (BOA, December 2017): "Patients should be assessed for falls risks and bone health, and referred to the fracture liaison services and or falls service where appropriate." https://www.boa.ac.uk/resource/boast-16-pdf.html
- Pass: An FLS referral, or a documented bone health and falls assessment with a stated outcome, within 12 weeks of injury.. Target: ≥90%. Sample: 50 consecutive patients over the last 3 months.
- Change: Automatic FLS feed: VFC coding of 'distal radius fracture, age ≥50' triggers an electronic referral to the FLS, with an opt-out box.
- Template: `new_audits/templates/ONA-066.csv`

**ONA-067. In adults with a distal radius fracture treated in a cast, was an X-ray at cast removal avoided when there was no documented clinical concern?** (new)
- Standard: BOAST The Management of Distal Radial Fractures (BOA, December 2017): "A radiograph of the patient’s wrist at the time of removing immobilisation is not required unless there is clinical cause for concern." https://www.boa.ac.uk/resource/boast-16-pdf.html
- Pass: No wrist X-ray taken at the cast-removal visit, OR an X-ray taken with a documented clinical reason (for example tenderness, deformity, new symptoms).. Target: ≥90%. Sample: 50 consecutive patients reaching cast removal over the last 3 months.
- Change: Change the fracture clinic booking template so 'X-ray on arrival' is no longer the default for distal radius cast-removal visits; the clinician must request it with a reason.
- Template: `new_audits/templates/ONA-067.csv`

**ONA-068. In adults with a scaphoid fracture seen on plain X-ray, was a CT scan done to measure displacement?** (new)
- Standard: BSSH Standards of Care in Hand Trauma: Scaphoid fractures, standard 8 (BSSH, 2024; revision due 2027): "If a scaphoid fracture is visible on plain radiographs, CT scan should be performed to measure displacement accurately." https://www.bssh.ac.uk/_userfiles/pages/files/professionals/Trauma%20standards/Scaphoid%20standards.pdf
- Pass: A wrist/scaphoid CT performed after diagnosis (within 14 days of presentation) with displacement reported or measured.. Target: ≥90%. Sample: 30–40 consecutive patients over the last 6–12 months.
- Change: Add 'CT scaphoid (sagittal and coronal along scaphoid axis)' as an automatic request on the VFC scaphoid-fracture outcome, with the reconstruction protocol agreed with radiology.
- Template: `new_audits/templates/ONA-068.csv`

**ONA-069. In adults with a scaphoid fracture treated non-operatively, was union confirmed on CT (or on X-ray at 6 months) before discharge from follow-up?** (new)
- Standard: BSSH Standards of Care in Hand Trauma: Scaphoid fractures, standard 14 (BSSH, 2024; revision due 2027): "Patients should only be discharged once union is confirmed on CT (>50% of the fracture cross-sectional area) or by delayed radiographs at 6 months." https://www.bssh.ac.uk/_userfiles/pages/files/professionals/Trauma%20standards/Scaphoid%20standards.pdf
- Pass: Before the discharge date: a CT reporting union (>50% bridging), OR radiographs at ≥6 months after injury showing union.. Target: ≥90%. Sample: 30–40 consecutive discharged patients over the last 12 months.
- Change: A scaphoid discharge checklist in the clinic letter template: 'Union confirmed on: CT date __ / 6-month X-ray date __; non-union advice given'.
- Template: `new_audits/templates/ONA-069.csv`

**ONA-070. In patients with a closed mallet finger injury, were they seen by a hand surgeon or hand therapist within 72 hours of presentation?** (new)
- Standard: BSSH Standards of Care in Hand Trauma: Mallet injuries (BSSH, published 2020): "for closed injuries; green – next available clinic (real or virtual) within 72 hours" https://www.bssh.ac.uk/_userfiles/pages/files/professionals/Trauma%20standards/7%20Mallet%20Injuries.pdf
- Pass: A hand surgeon or hand therapist review (real or virtual) documented ≤72 hours after first ED/UTC/MIU attendance.. Target: ≥90%. Sample: 40 consecutive patients over the last 3–6 months.
- Change: Direct booking from ED/MIU into hand therapy (a 'mallet' slot in the hand therapy diary) with a pre-made splint and leaflet in the minor injuries cupboard.
- Template: `new_audits/templates/ONA-070.csv`

**ONA-071. In patients with a hand bite that broke the skin, were antibiotics started on the day of first attendance?** (new)
- Standard: BSSH Standards of Care in Hand Trauma: Human and animal bites to the hand (BSSH, 2021): "Antibiotics should be started as soon as possible after injury." https://www.bssh.ac.uk/_userfiles/pages/files/professionals/Trauma%20standards/8%20Bites.pdf
- Pass: An antibiotic dose given or prescribed at the first ED/UTC/MIU attendance (drug chart or discharge prescription).. Target: ≥95%. Sample: 40 consecutive patients over the last 3–6 months.
- Change: Add a 'hand bite' order set to the ED system: first-dose co-amoxiclav (or local alternative) plus referral pathway, triggered by the bite presenting complaint.
- Template: `new_audits/templates/ONA-071.csv`

**ONA-072. In patients with an open hand fracture, were antibiotics stopped by 72 hours or at definitive wound closure, whichever came first?** (under-audited)
- Standard: BSSH Standards of Care in Hand Trauma: Open fractures (BSSH, 2021): "Antibiotics should be stopped at 72 hours or after definitive closure whichever is the sooner, subject to clinical judgement." https://www.bssh.ac.uk/_userfiles/pages/files/professionals/Trauma%20standards/2%20Open%20fractures%20Final.pdf
- Pass: Last antibiotic dose (inpatient chart plus discharge prescription) given no later than 72 hours after the first dose, or no later than definitive closure if that came first; OR a documented clinical reason for continuing.. Target: ≥85%. Sample: 30–40 consecutive patients over the last 6 months.
- Change: Add a 'stop date' field to the open hand fracture antibiotic prescription (EPMA default: 72 hours) and a line in the operation note: 'Antibiotics: stop now / stop at __ (reason)'.
- Template: `new_audits/templates/ONA-072.csv`

**ONA-073. In patients listed for washout of a flexor sheath infection, did surgery start within 24 hours of the decision to operate?** (new)
- Standard: BSSH Standards of Care in Hand Trauma: Flexor sheath infection (BSSH, 2021): "Urgent – as early as possible within safe working hours and within a maximum of 24 hours of decision to operate" https://www.bssh.ac.uk/_userfiles/pages/files/professionals/Trauma%20standards/10%20Flexor%20sheath%20infection.pdf
- Pass: Knife-to-skin time ≤24 hours after the documented decision to operate.. Target: ≥90%. Sample: All cases over the last 12–24 months (expect 30–40 in a unit with a hand service).
- Change: Flag flexor sheath infection as 'priority 2' (within 24 hours) on the trauma list booking form so it is listed first on the next day list.
- Template: `new_audits/templates/ONA-073.csv`

**ONA-074. In adults with suspected cauda equina syndrome who could pass urine, were both pre-void and post-void bladder scan volumes documented?** (new)
- Standard: GIRFT National Suspected Cauda Equina Syndrome Pathway (February 2023, updated March 2026), Bladder scan: "If a patient is able to void, carefully document the following: • pre-void volume; • post-void residual volume (PVR)" https://gettingitrightfirsttime.co.uk/wp-content/uploads/2026/04/National-Suspected-Cauda-Equina-Pathway-March-2026.pdf
- Pass: Both a pre-void and a post-void bladder scan volume (in ml) documented before the MRI.. Target: ≥90%. Sample: 40 consecutive patients over the last 3 months.
- Change: Put 'pre-void __ ml / post-void __ ml' boxes on the ED suspected-CES proforma and give ED nurses a triage standing order to bladder scan.
- Template: `new_audits/templates/ONA-074.csv`

**ONA-075. In adults discharged after an MRI showed no cauda equina compression, was safety-netting with the CES warning card or video documented?** (new)
- Standard: GIRFT National Suspected Cauda Equina Syndrome Pathway (February 2023, updated March 2026), When to refer to the spinal surgical team: "Safety net about progression of CES symptoms via video and card" https://gettingitrightfirsttime.co.uk/wp-content/uploads/2026/04/National-Suspected-Cauda-Equina-Pathway-March-2026.pdf
- Pass: The discharge note or letter records that the CES warning card and/or MACP video link was given (a generic 'return if worse' does not pass).. Target: ≥90%. Sample: 40 consecutive patients over the last 3 months.
- Change: Add a tick box 'CES warning card given / video link sent' to the ED discharge template for the suspected-CES pathway, and stock cards at ED discharge desk.
- Template: `new_audits/templates/ONA-075.csv`

**ONA-076. In adults with neurological symptoms or signs of metastatic spinal cord compression, was a 16 mg dexamethasone dose given within 4 hours of presentation?** (under-audited)
- Standard: NICE NG234 rec 1.8.1 (2023): "Offer 16 mg of oral dexamethasone (or equivalent parenteral dose) as soon as possible." https://www.nice.org.uk/guidance/ng234/chapter/Recommendations
- Pass: A first dose of 16 mg oral dexamethasone (or equivalent IV) given within 4 hours of ED arrival or of first ward suspicion (local operational definition of 'as soon as possible').. Target: ≥90%. Sample: 30–40 consecutive patients over the last 6–12 months (from MSCC coordinator log).
- Change: An 'MSCC' order set on EPMA: dexamethasone 16 mg stat plus PPI and capillary glucose monitoring, linked from the ED MSCC pathway.
- Template: `new_audits/templates/ONA-076.csv`

**ONA-077. In adults with suspected metastatic spinal cord compression, was the spinal MRI done within 24 hours of the MSCC coordinator being contacted?** (under-audited)
- Standard: NICE NG234 rec 1.5.2 (2023): "Offer an MRI scan to people with suspected MSCC (see recommendation 1.3.2), to be done: as soon as possible (and always within 24 hours)" https://www.nice.org.uk/guidance/ng234/chapter/Recommendations
- Pass: MRI scan time ≤24 hours after the first recorded contact with the MSCC coordinator (or after presentation if coordinator time is missing, recorded separately).. Target: ≥95%. Sample: 40 consecutive referrals over the last 3–6 months from the MSCC coordinator log.
- Change: A radiology SOP that treats suspected MSCC as a protected same-day slot with the MSCC referral code auto-vetted.
- Template: `new_audits/templates/ONA-077.csv`

**ONA-078. In adults with diabetes admitted under trauma and orthopaedics, was a documented foot examination (both feet) done within 24 hours of admission?** (new)
- Standard: NICE NG19 rec 1.3.3 (2015, guideline updated 2019): "For adults with diabetes, assess their risk of developing a diabetic foot problem at the following times: ... On any admission to hospital, and if there is any change in their status while they are in hospital." https://www.nice.org.uk/guidance/ng19/chapter/Recommendations
- Pass: A documented foot risk assessment of both feet (at least neuropathy, ischaemia and ulceration recorded, with a risk category) within 24 hours of admission.. Target: ≥90%. Sample: 40 consecutive admissions over the last 1–2 months.
- Change: Add the diabetic foot risk tool to the T&O nursing admission bundle, triggered by a diabetes flag on the EPR.
- Template: `new_audits/templates/ONA-078.csv`

**ONA-079. In adults with a diabetic foot infection started on IV antibiotics, was the IV antibiotic reviewed within 48 hours?** (new)
- Standard: NICE NG19 rec 1.6.11 (2019): "If intravenous antibiotics are given, review by 48 hours and consider switching to oral antibiotics if possible." https://www.nice.org.uk/guidance/ng19/chapter/Recommendations
- Pass: A documented antibiotic review (continue IV with reason, switch to oral, or stop) within 48 hours of the first IV dose.. Target: ≥90%. Sample: 40 consecutive patients over the last 3–6 months.
- Change: EPMA 48-hour 'IV antibiotic review' alert with forced outcome (continue with reason / oral switch / stop) for diabetic foot infection prescriptions.
- Template: `new_audits/templates/ONA-079.csv`

**ONA-080. In adults with a diabetic foot ulcer seen by the orthopaedic or foot and ankle team, was ulcer severity recorded with SINBAD or the University of Texas system?** (new)
- Standard: NICE NG19 rec 1.5.2 (2015): "Use a standardised system to document the severity of the foot ulcer, such as the SINBAD (Site, Ischaemia, Neuropathy, Bacterial Infection, Area and Depth) or the University of Texas classification system." https://www.nice.org.uk/guidance/ng19/chapter/Recommendations
- Pass: A SINBAD score or University of Texas grade recorded at the first orthopaedic/foot and ankle review (Wagner alone = fail).. Target: ≥80%. Sample: 40 consecutive patients over the last 3–6 months.
- Change: A diabetic foot ulcer box with SINBAD scoring in the T&O ward-referral response and clinic templates.
- Template: `new_audits/templates/ONA-080.csv`

## Elective arthroplasty, knee and shoulder

**ONA-081. In adults having primary elective hip or knee replacement without renal impairment, what proportion receive both intravenous and topical tranexamic acid, with a total dose of 3 g or less?** (under-audited)
- Standard: NICE NG157 rec 1.4.1 (2020): "For primary elective hip or knee replacement: Give intravenous tranexamic acid. If there is no renal impairment, also apply 1 g to 2 g of topical (intra-articular) tranexamic acid diluted in saline after the final wash-out and before wound closure. Ensure that the total combined dose of tranexamic acid does not exceed 3 g." https://www.nice.org.uk/guidance/ng157/chapter/Recommendations
- Pass: No renal impairment: IV TXA given and 1–2 g topical TXA recorded, total 3 g or less. Mild or moderate renal impairment: reduced-dose IV TXA only.. Target: ≥90%. Sample: 40 consecutive primary hip and knee replacements over the last 2–3 months
- Change: Add a pre-filled TXA line to the arthroplasty anaesthetic chart and a 'topical TXA given? dose' tick box to the operation note template.
- Template: `new_audits/templates/ONA-081.csv`

**ONA-082. In adults having primary elective shoulder replacement, what proportion receive intravenous tranexamic acid or have a recorded reason for not giving it?** (new)
- Standard: NICE NG157 rec 1.4.1 (2020): "For primary elective shoulder replacement: Consider intravenous tranexamic acid. If there is no renal impairment, consider 1 g to 2 g of topical (intra-articular) tranexamic acid diluted in saline applied after the final wash-out and before wound closure. Ensure that the total combined dose of tranexamic acid does not exceed 3 g." https://www.nice.org.uk/guidance/ng157/chapter/Recommendations
- Pass: IV TXA given at or before incision, or a reason for not giving it written on the anaesthetic chart or op note.. Target: ≥90%. Sample: All primary elective shoulder replacements over the last 6–12 months (usually 30–50 in a DGH)
- Change: Add shoulder replacement to the theatre TXA prompt in the WHO sign-in ("TXA given?") on the electronic checklist.
- Template: `new_audits/templates/ONA-082.csv`

**ONA-083. In primary elective hip, knee and shoulder replacements, what proportion have both implant 'stop moments' recorded: one before implantation and one before wound closure?** (under-audited)
- Standard: NICE NG157 rec 1.6.1 (2020): "Use 2 intraoperative 'stop moments', 1 before implantation and 1 before wound closure, to check all implant details and ensure compatibility of each component." https://www.nice.org.uk/guidance/ng157/chapter/Recommendations
- Pass: Both stop moments recorded (time or tick) on the theatre record, implant chart or op note.. Target: 100%. Sample: 40 consecutive primary joint replacements over the last 1–2 months
- Change: Add two mandatory fields ('implant stop 1: before opening' and 'implant stop 2: before closure') to the electronic theatre record for arthroplasty cases.
- Template: `new_audits/templates/ONA-083.csv`

**ONA-084. In adults having primary elective knee replacement, what proportion have local infiltration analgesia recorded in the operation or anaesthetic record?** (new)
- Standard: NICE NG157 rec 1.3.2 (2020): "Offer people having primary elective knee replacement a choice of: regional anaesthesia in combination with LIA or general anaesthesia in combination with LIA.Consider adding a nerve block that does not impair motor function to either of the options above, provided it does not delay surgery significantly." https://www.nice.org.uk/guidance/ng157/chapter/Recommendations
- Pass: LIA recorded (drug, dose or volume) in the op note or anaesthetic chart, or a reason for not giving it.. Target: ≥95%. Sample: 40 consecutive primary knee replacements over the last 2 months
- Change: Add an LIA line (drug, concentration, volume) to the knee replacement op-note template.
- Template: `new_audits/templates/ONA-084.csv`

**ONA-085. In adults listed for primary total knee replacement, what proportion have patella resurfacing offered and recorded before surgery?** (under-audited)
- Standard: NICE NG157 rec 1.7.2 (2020): "Offer resurfacing of the patella to people having primary elective total knee replacement." https://www.nice.org.uk/guidance/ng157/chapter/Recommendations
- Pass: Clinic letter or consent form records that patella resurfacing was offered, and the decision. Also record whether it was done.. Target: ≥90%. Sample: 40 consecutive primary TKRs over the last 3 months
- Change: Add 'patella resurfacing offered: yes/no, decision' to the TKR consent form (or procedure-specific consent template).
- Template: `new_audits/templates/ONA-085.csv`

**ONA-086. In adults with isolated medial compartment knee osteoarthritis listed for knee replacement, what proportion have a choice of partial or total knee replacement recorded?** (new)
- Standard: NICE NG157 rec 1.7.1 (2020): "Offer a choice of partial or total knee replacement to people with isolated medial compartmental osteoarthritis. Discuss the potential benefits and risks of each option with the person." https://www.nice.org.uk/guidance/ng157/chapter/Recommendations
- Pass: Clinic letter or consent records that both partial and total replacement were discussed, with benefits and risks.. Target: ≥90%. Sample: 30–40 consecutive eligible patients listed over the last 6 months
- Change: Add a 'medial OA: UKR vs TKR discussed' prompt to the knee listing clinic letter template.
- Template: `new_audits/templates/ONA-086.csv`

**ONA-087. In adults having primary elective shoulder replacement, what proportion are seen by a physiotherapist or occupational therapist for rehabilitation within 24 hours of surgery?** (new)
- Standard: NICE NG157 rec 1.10.1 (2020): "A physiotherapist or occupational therapist should offer rehabilitation, on the day of surgery if possible and no more than 24 hours after surgery, to people who have had a primary elective hip, knee or shoulder replacement." https://www.nice.org.uk/guidance/ng157/chapter/Recommendations
- Pass: First physio or OT rehab contact recorded within 24 hours of the end of surgery.. Target: ≥90%. Sample: All primary shoulder replacements over the last 6–12 months (30–50)
- Change: Automatic therapy referral generated when the patient leaves theatre (theatre system to therapy task list), including weekend lists.
- Template: `new_audits/templates/ONA-087.csv`

**ONA-088. In adults discharged after primary elective hip or knee replacement, what proportion have a named point of contact for rehabilitation advice recorded before they leave?** (new)
- Standard: NICE NG157 rec 1.10.4 (2020): "Ensure that people who are undertaking self-directed rehabilitation have: a clear understanding of their rehabilitation goals and the importance of doing the exercises prescribed to achieve these goals a point of contact for advice and support." https://www.nice.org.uk/guidance/ng157/chapter/Recommendations
- Pass: Therapy notes or discharge summary record self-directed rehab advice, goals, and a named contact (phone number or service).. Target: ≥90%. Sample: 40 consecutive discharges over the last 2 months
- Change: Add a 'rehab goals + contact number given' tick box to the arthroplasty discharge summary template.
- Template: `new_audits/templates/ONA-088.csv`

**ONA-089. Of hip and knee osteoarthritis referrals to orthopaedics that were rejected or returned, what proportion were rejected because of BMI or smoking alone?** (new)
- Standard: NICE NG226 rec 1.6.3 (2022): "Do not exclude people with osteoarthritis from referral for joint replacement because of: age sex or gender smoking comorbidities overweight or obesity, based on measurements such as body mass index (BMI)." https://www.nice.org.uk/guidance/ng226/chapter/Recommendations
- Pass: Referral not rejected or returned on BMI, smoking, age or comorbidity alone (fail if one of these is the only stated reason).. Target: 100% (no rejections on these grounds alone). Sample: 40 consecutive rejected or returned OA referrals over the last 6 months
- Change: Remove BMI and smoking as rejection options from the triage proforma. Replace with 'referred to weight/smoking support in parallel'.
- Template: `new_audits/templates/ONA-089.csv`

**ONA-090. In adults with knee osteoarthritis who had a knee arthroscopy, what proportion had a documented history of true mechanical locking?** (new)
- Standard: AoMRC Evidence-Based Interventions: knee arthroscopy for patients with osteoarthritis (2019, reviewed 2024); NICE NG226 rec 1.7.1 (2022): "Referral for arthroscopic lavage and debridement should not be offered as part of treatment for osteoarthritis, unless the person has knee osteoarthritis with a clear history of mechanical locking. (EBI). NICE NG226 1.7.1: Do not offer arthroscopic lavage or debridement to people with osteoarthritis." https://ebi.aomrc.org.uk/interventions/knee-arthroscopy-for-patients-with-osteoarthritis/
- Pass: Pre-operative letter documents a clear history of mechanical locking (or another non-OA indication such as a loose body or bucket-handle tear on MRI).. Target: 100%. Sample: 40 consecutive eligible arthroscopies over the last 12 months
- Change: Add an EBI criteria check box (mechanical locking yes/no) to the knee arthroscopy listing form, with a prior-approval step if not met.
- Template: `new_audits/templates/ONA-090.csv`

**ONA-091. In adults aged 45 or over referred to orthopaedics with knee pain and osteoarthritis features, what proportion had a knee MRI requested without a prior weight-bearing radiograph?** (new)
- Standard: AoMRC Evidence-Based Interventions: knee MRI when symptoms are suggestive of osteoarthritis (2020, reviewed 2024): "An initial diagnosis of OA can be made when clinical assessment is suggestive of this pathology. If imaging is required to confirm the diagnosis, then weight bearing radiographs are the first-line of investigation. Magnetic resonance imaging (MRI) for knees is not usually needed." https://ebi.aomrc.org.uk/interventions/knee-mri-when-symptoms-are-suggestive-of-osteoarthritis/
- Pass: Weight-bearing knee radiograph done before any MRI; MRI only if atypical features or suspected other diagnosis recorded on the request.. Target: ≥90%. Sample: 40 consecutive eligible MRI requests over the last 3 months
- Change: Radiology vetting rule: knee MRI requests for adults 45+ without a weight-bearing radiograph in the last 12 months are returned with a prompt.
- Template: `new_audits/templates/ONA-091.csv`

**ONA-092. In adults listed for arthroscopic subacromial decompression, what proportion had at least 6 months of documented physiotherapy or other non-surgical treatment first?** (under-audited)
- Standard: BESS Patient Care Pathway: Subacromial Pain (Shoulder & Elbow 2025;17(6)); AoMRC EBI arthroscopic shoulder decompression (reviewed 2024): "Physiotherapy or non-surgical treatment should be first line of management. If symptoms fail to resolve with non-surgical treatment by six months, then there is moderate evidence that surgical treatment could be considered and subacromial decompression may offer satisfactory long-term outcomes for patients with subacromial pain. In addition, patients should be made aware that there are studies which show no evidence of the benefit of surgery compared to sham surgery." https://bess.ac.uk/wp-content/uploads/2026/03/Subacromial-Shoulder-Pain-2025.pdf
- Pass: Letters record 6 months or more of physiotherapy or non-surgical treatment before listing. Secondary field: sham-surgery evidence discussed.. Target: ≥90%. Sample: All ASDs over the last 12 months (usually 30–50)
- Change: Add an ASD listing checklist (months of physio, injection, imaging, sham evidence discussed) to the shoulder clinic letter template.
- Template: `new_audits/templates/ONA-092.csv`

**ONA-093. In adults having a subacromial steroid injection, what proportion had it done under image guidance?** (new)
- Standard: AoMRC Evidence-Based Interventions: scans for shoulder pain and guided injections (2020, reviewed 2024): "Image guided subacromial injections are not recommended in primary, intermediate or secondary care. Evidence does not support the use of guided subacromial injections over unguided subacromial injections in the treatment of subacromial shoulder pain." https://ebi.aomrc.org.uk/interventions/scans-for-shoulder-pain-and-guided-injections-for-shoulder-pain/
- Pass: Subacromial injection given without image guidance (landmark), unless a documented reason (e.g. failed landmark injection, body habitus) is recorded.. Target: ≥90%. Sample: 40 consecutive subacromial injections over the last 3 months
- Change: Radiology vetting rule: US-guided subacromial injection requests are returned unless a reason is given. Offer a landmark injection clinic slot instead.
- Template: `new_audits/templates/ONA-093.csv`

**ONA-094. In adults having primary shoulder replacement, what proportion have a pre-operative Oxford Shoulder Score (or other validated shoulder PROM) recorded?** (new)
- Standard: BESS/BOA Patient Care Pathway: Glenohumeral osteoarthritis (Shoulder & Elbow 2016;8(3)): "Acceptable scores include the Disability of Arm, Shoulder and Hand, Constant Score and the Oxford Shoulder Score. Other measures such as EQ5D may be used for economic analysis. Scores should be captured pre-operatively and a minimum of 6 months following intervention, which allows longitudinal analysis to determine magnitude of treatment effect and consequences of any treatment-related adverse events." https://bess.ac.uk/wp-content/uploads/2020/06/Glenohumeral-osteoarthritis.pdf
- Pass: A validated shoulder PROM (OSS, DASH or Constant) recorded in the 3 months before surgery.. Target: ≥90%. Sample: All primary shoulder replacements over the last 6–12 months (30–50)
- Change: Make the Oxford Shoulder Score a required item on the shoulder pre-assessment form (paper or electronic), completed by the patient in the waiting area.
- Template: `new_audits/templates/ONA-094.csv`

**ONA-095. In adults reviewed in clinic for a painful or problematic knee replacement, what proportion had CRP and ESR (or PV) checked at the first review?** (new)
- Standard: BOA Specialty Standard: Investigation and Management of Patients with Problematic Knee Replacements (August 2020), standard 2: "Initial investigation should include: clinical examination, X-Ray imaging and serological screening tests for infection (e.g. CRP, ESR +/- PV)." https://www.boa.ac.uk/asset/7DEFDD00%2D004A%2D45FA%2D870AC5A2DFE332CF/
- Pass: CRP and ESR or PV requested within 2 weeks of the first clinic review for the problem, plus a radiograph.. Target: ≥90%. Sample: 40 consecutive patients over the last 6 months
- Change: Create a 'problematic knee replacement' order set in the EPR (CRP, ESR, AP/lateral, long-leg, skyline) linked to the clinic outcome form.
- Template: `new_audits/templates/ONA-095.csv`

**ONA-096. In adults investigated for a painful or problematic knee replacement, what proportion had the full radiograph set: weight-bearing AP and lateral, long-leg alignment and patella skyline?** (new)
- Standard: BOA Specialty Standard: Investigation and Management of Patients with Problematic Knee Replacements (August 2020), standard 4: "Full weight bearing AP and lateral, long-leg alignment and patella skyline radiographs should be performed in all patients." https://www.boa.ac.uk/asset/7DEFDD00%2D004A%2D45FA%2D870AC5A2DFE332CF/
- Pass: All four views (standing AP, lateral, long-leg, skyline) on PACS within 3 months of the first problem review.. Target: ≥90%. Sample: 40 consecutive patients over the last 6 months
- Change: Same EPR 'problematic knee replacement' order set that requests all four views in one click.
- Template: `new_audits/templates/ONA-096.csv`

**ONA-097. In adults with suspected chronic knee replacement infection who had a joint aspiration or biopsy, what proportion had no antibiotics in the 2 weeks before sampling?** (new)
- Standard: BOA Specialty Standard: Investigation and Management of Prosthetic Joint Infection in Knee Replacement (August 2020), standard 12: "Antibiotics should not be administered for 2 weeks prior to aspiration +/- biopsy except in cases of acute infection in a systemically unwell (septic) patient." https://www.boa.ac.uk/asset/9BA3010B%2D8563%2D4517%2DA2869FD179A1F4B2/
- Pass: No antibiotic in the 14 days before aspiration or biopsy (GP and hospital prescriptions checked), or patient septic.. Target: ≥90%. Sample: All knee aspirations or biopsies for suspected PJI over the last 12–24 months (usually 30–40)
- Change: Add a GP letter template ('suspected joint replacement infection: please do not start antibiotics before aspiration') sent from the first clinic, and an 'antibiotics in last 14 days?' field on the aspiration request.
- Template: `new_audits/templates/ONA-097.csv`

**ONA-098. In adults discharged on VTE prophylaxis after primary hip or knee replacement, what proportion have the drug and stop date stated in the GP discharge summary?** (under-audited)
- Standard: NICE NG89 rec 1.2.8 (2018, updated 2019): "Notify the person's GP if the person has been discharged with pharmacological and/or mechanical VTE prophylaxis to be used at home. [2018]" https://www.nice.org.uk/guidance/ng89/chapter/Recommendations
- Pass: Discharge summary sent to the GP names the VTE drug, dose and stop date (or total duration).. Target: ≥95%. Sample: 40 consecutive discharges over the last 2 months
- Change: Make 'VTE prophylaxis: drug, dose, stop date' a mandatory field in the arthroplasty discharge summary template.
- Template: `new_audits/templates/ONA-098.csv`

**ONA-099. In adults transfused red cells after primary hip, knee or shoulder replacement (not actively bleeding), what proportion received a single unit followed by a haemoglobin check before any further unit?** (under-audited)
- Standard: NICE NG24 recs 1.6.1 and 1.6.2 (2015): "Consider single‑unit red blood cell transfusions for adults (or equivalent volumes calculated based on body weight for children or adults with low body weight) who do not have active bleeding. After each single‑unit red blood cell transfusion (or equivalent volumes calculated based on body weight for children or adults with low body weight), clinically reassess and check haemoglobin levels, and give further transfusions if needed." https://www.nice.org.uk/guidance/ng24/chapter/Red-blood-cell-transfusion
- Pass: Each unit prescribed singly, with Hb rechecked and a clinical review recorded before the next unit. Pre-transfusion Hb 70 g/L or below (80 if acute coronary syndrome) or a reason recorded.. Target: ≥90%. Sample: All transfused primary arthroplasty patients over the last 12 months (often 30–50)
- Change: Change the EPR blood prescription so that red cells default to one unit, with a prompt to recheck Hb before the next.
- Template: `new_audits/templates/ONA-099.csv`

**ONA-100. In adults found to have iron-deficiency anaemia at pre-assessment for primary hip or knee replacement, what proportion were offered iron before surgery?** (under-audited)
- Standard: NICE NG24 recs 1.2.1 and 1.2.2 (2015); NICE NG180 rec 1.3.4 (2020): "Offer oral iron before and after surgery to people with iron‑deficiency anaemia. [2015] Consider intravenous iron before or after surgery for people who: have iron‑deficiency anaemia and cannot tolerate or absorb oral iron, or are unable to adhere to oral iron treatment are diagnosed with functional iron deficiency are diagnosed with iron‑deficiency anaemia, and the interval between the diagnosis of anaemia and surgery is predicted to be too short for oral iron to be effective. [2015]" https://www.nice.org.uk/guidance/ng24/chapter/Reducing-requirement-for-blood-transfusion-for-people-having-surgery
- Pass: Oral or IV iron prescribed or recommended to the GP before the surgery date.. Target: ≥90%. Sample: 30–40 consecutive anaemic pre-assessment patients over the last 6 months
- Change: Pre-assessment reflex rule: Hb below cut-off triggers automatic ferritin/TSAT and a nurse-led iron pathway (oral iron letter to GP or IV iron slot).
- Template: `new_audits/templates/ONA-100.csv`

**ONA-101. In adults with diabetes having primary elective joint replacement, what proportion had an HbA1c result from the 3 months before surgery?** (new)
- Standard: NICE NG45 rec 1.6.2 (2016): "Offer HbA1c testing to people with diabetes having surgery if they have not been tested in the last 3 months." https://www.nice.org.uk/guidance/ng45/chapter/Recommendations
- Pass: HbA1c result dated within 3 months before the operation date.. Target: ≥95%. Sample: 40 consecutive patients with diabetes over the last 6 months
- Change: Add a pre-assessment rule: diabetes flag triggers an HbA1c order if none in the last 3 months.
- Template: `new_audits/templates/ONA-101.csv`

**ONA-102. In adults attending pre-assessment for primary elective joint replacement, what proportion had a routine urine dipstick without symptoms?** (new)
- Standard: NICE NG45 recs 1.7.1 and 1.7.2 (2016): "Do not routinely offer urine dipstick tests before surgery. Consider microscopy and culture of midstream urine sample before surgery if the presence of a urinary tract infection would influence the decision to operate." https://www.nice.org.uk/guidance/ng45/chapter/Recommendations
- Pass: No urine dipstick or MSU done unless urinary symptoms or another reason recorded.. Target: ≥90% with no routine dipstick. Sample: 40 consecutive pre-assessments over the last month
- Change: Remove 'urinalysis' from the default arthroplasty pre-assessment test panel; keep it as a symptom-triggered option.
- Template: `new_audits/templates/ONA-102.csv`

**ONA-103. In adults listed for primary hip or knee replacement, what proportion have prehabilitation advice on exercise, weight and smoking recorded before surgery?** (new)
- Standard: NICE NG157 rec 1.2.1 (2020): "Give people having hip or knee replacement advice on preoperative rehabilitation. Include advice on: exercises to do before and after surgery that will aid recovery lifestyle, including weight management, diet and smoking cessation (see NICE's guidance on lifestyle and wellbeing) maximising functional independence and quality of life before and after surgery." https://www.nice.org.uk/guidance/ng157/chapter/Recommendations
- Pass: Clinic letter, pre-assessment or joint school record shows advice on exercises and lifestyle (weight and, if a smoker, smoking) before surgery.. Target: ≥90%. Sample: 40 consecutive patients over the last 2 months
- Change: Add a 3-item prehab box (exercise sheet given, weight advice, smoking referral) to the pre-assessment form.
- Template: `new_audits/templates/ONA-103.csv`

## Orthopaedic perioperative, theatre and ward care

**ONA-104. In adults who had an orthopaedic implant inserted, was a record of every implant (manufacturer, size and unique identifier or lot number) placed in the patient's notes?** (new)
- Standard: NatSSIPs 2 (CPOC 2023), Sequential Standard: Implant Verification, 'During the procedure': "A record of the implants used must be made in the patient's records. Appropriate details should be shared with the patient after the procedure. When a manufacturer's label is available, this should be placed in the notes. When it is not, for example with electronic patient records, the following should be recorded: Manufacturer; Style; Size; Manufacturer's unique identifier for the implant, or the serial number; Expiry date" https://cpoc.org.uk/sites/cpoc/files/documents/2022-12/CPOC_NatSSIPs2_Implant_2023.pdf
- Pass: Every implant used (including each screw set, plate, nail, prosthetic component and cement) appears in the notes or EPR, either as a label or with manufacturer, size and unique identifier or lot number.. Target: ≥95%. Sample: 40 consecutive implant cases from the theatre system over the last 4 weeks (about 20 trauma, 20 elective).
- Change: Add a mandatory 'implant record complete' tick (with a scan of the labels) to the Sign Out step on the electronic theatre record.
- Template: `new_audits/templates/ONA-104.csv`

**ONA-105. Did each elective orthopaedic theatre session end with a recorded team debrief?** (new)
- Standard: NatSSIPs 2 (CPOC 2023), Sequential Standard: Debrief: "All elective major procedure sessions should end with a Debrief. A Debrief is encouraged if feasible after emergency cases." https://cpoc.org.uk/sites/cpoc/files/documents/2022-12/CPOC_NatSSIPs2_Debrief_2023.pdf
- Pass: A debrief is recorded for the session (theatre system field or debrief log), with at least one entry under problems/actions or 'none'.. Target: ≥90% of elective sessions. Sample: 40 consecutive elective orthopaedic sessions over the last 4–6 weeks.
- Change: Make the debrief a required field that must be completed before the session can be closed on the theatre system, with a short 3-question prompt (went well / problems / actions and owner).
- Template: `new_audits/templates/ONA-105.csv`

**ONA-106. In adults having orthopaedic surgery, was thromboprophylaxis prescribed, or a reason for omission recorded, before the patient left recovery?** (new)
- Standard: NatSSIPs 2 (CPOC 2023), Sequential Standard: Sign Out, safety checks: "Confirmation that VTE risk assessment is completed and actioned" https://cpoc.org.uk/sites/cpoc/files/documents/2022-12/CPOC_NatSSIPs2_Signout_2023.pdf
- Pass: Post-operative pharmacological or mechanical prophylaxis prescribed on the drug chart, or a documented reason for none, timestamped before the recovery discharge time.. Target: ≥95%. Sample: 40 consecutive inpatient orthopaedic operations over the last 3 weeks.
- Change: Add 'VTE prophylaxis prescribed (drug/time) or reason' as a hard stop on the electronic Sign Out and recovery discharge checklist.
- Template: `new_audits/templates/ONA-106.csv`

**ONA-107. In adults on the orthopaedic trauma list, was the gap between their last clear drink and being sent for by theatre 2 hours or less?** (under-audited)
- Standard: NICE NG180 rec 1.4.1 (2020), applied through the CPOC SipTilSend protocol: "Tell people having surgery, including dental surgery, that: they may drink clear fluids until 2 hours before their operation; drinking clear fluids before the operation can help reduce headaches, nausea and vomiting afterwards; clear fluids are water, fruit juice without pulp, coffee or tea without milk and ice lollies. [CPOC SipTilSend: 'From 2 hours before surgery until the operating theatre team 'send for' the patient ... For adults: You may sip slowly on clear fluids ... Up to 170mls per hour (that's a standard NHS cup) until the operating team send for you']" https://www.nice.org.uk/guidance/ng180/chapter/Recommendations
- Pass: Documented last clear fluid time is 2 hours or less before the time theatre sent for the patient.. Target: ≥80%. Sample: 40 consecutive trauma list patients over the last 2–3 weeks.
- Change: Replace 'NBM from midnight' on the trauma ward pre-op checklist with a SipTilSend hourly sip chart that must be signed each hour until the send-for call.
- Template: `new_audits/templates/ONA-107.csv`

**ONA-108. In women aged 12 to 55 having orthopaedic surgery, was the discussion about possible pregnancy documented on the day of surgery?** (new)
- Standard: NICE NG45 recs 1.3.1 and 1.3.3 (2016): "1.3.1 On the day of surgery, sensitively ask all women of childbearing potential whether there is any possibility they could be pregnant. ... 1.3.3 Document all discussions with women about whether or not to carry out a pregnancy test." https://www.nice.org.uk/guidance/ng45/chapter/Recommendations
- Pass: A documented pregnancy status question or test result dated on the day of surgery (before anaesthesia).. Target: 100%. Sample: 40 consecutive eligible women from the theatre system over the last 4–6 weeks.
- Change: Add a required pregnancy status field (asked / test result / not applicable with reason) to the Sign In section of the electronic WHO checklist.
- Template: `new_audits/templates/ONA-108.csv`

**ONA-109. In adults having orthopaedic surgery, was every pre-operative chest X-ray backed by a documented clinical indication?** (new)
- Standard: NICE NG45 rec 1.8.1 (2016): "Do not routinely offer chest X‑rays before surgery." https://www.nice.org.uk/guidance/ng45/chapter/Recommendations
- Pass: No pre-operative chest X-ray, or a chest X-ray with a documented clinical reason (e.g. new respiratory signs, suspected chest injury or metastases).. Target: ≥90%. Sample: 40 consecutive operated adults over the last 4 weeks.
- Change: Remove chest X-ray from the default orthopaedic admission order set; require a free-text clinical indication to request one.
- Template: `new_audits/templates/ONA-109.csv`

**ONA-110. In adults after orthopaedic surgery, was a temperature of 36.0°C or above recorded before transfer from recovery to the ward?** (under-audited)
- Standard: NICE CG65 rec 1.4.1 (2008, updated 2016): "The patient's temperature should be measured and documented on admission to the recovery room and then every 15 minutes. Ward transfer should not be arranged unless the patient's temperature is 36.0°C or above." https://www.nice.org.uk/guidance/cg65/chapter/Recommendations
- Pass: Last recorded recovery temperature before ward transfer is 36.0°C or above.. Target: ≥95%. Sample: 40 consecutive orthopaedic patients over the last 2–3 weeks (mix of hip fracture, trauma and arthroplasty).
- Change: Add 'temperature ≥36.0°C' as a required field in the recovery discharge criteria on the electronic recovery chart.
- Template: `new_audits/templates/ONA-110.csv`

**ONA-111. In adults having upper limb surgery under general anaesthetic for more than 90 minutes, was a VTE prophylaxis decision documented?** (new)
- Standard: NICE NG89 rec 1.11.16 (2018, updated 2019): "Consider VTE prophylaxis for people undergoing upper limb surgery if the person's total time under general anaesthetic is over 90 minutes or where their operation is likely to make it difficult for them to mobilise." https://www.nice.org.uk/guidance/ng89/chapter/Recommendations
- Pass: Documented VTE decision after surgery: prophylaxis prescribed, or a recorded reason for none.. Target: ≥90%. Sample: 30–40 consecutive eligible cases over the last 3–6 months.
- Change: Add an auto-prompt to the operation note template: 'GA >90 min: VTE prophylaxis yes/no and reason'.
- Template: `new_audits/templates/ONA-111.csv`

**ONA-112. In adults having elective primary hip or knee replacement, did antibiotic prophylaxis stop within 24 hours of the operation?** (under-audited)
- Standard: UKHSA Start Smart Then Focus toolkit (updated 12 September 2023): "For surgical prophylaxis: Prescribe single dose antimicrobials where single dose antimicrobials have been shown to be effective – to minimise post-operative surgical site infection at an acceptable risk of harm from antimicrobial exposure. [Figure 2 caveat: 'In some cases, such as surgery involving implant replacement, 24 hours of antimicrobial prophylaxis may be required']" https://www.gov.uk/government/publications/antimicrobial-stewardship-start-smart-then-focus/start-smart-then-focus-antimicrobial-stewardship-toolkit-for-inpatient-care-settings
- Pass: No prophylactic antibiotic dose given more than 24 hours after knife-to-skin (treatment for a documented infection excluded).. Target: ≥95%. Sample: 40 consecutive elective primary hip or knee replacements over the last 6 weeks.
- Change: Build an arthroplasty prophylaxis order set in e-prescribing with a fixed number of doses and an automatic 24-hour stop.
- Template: `new_audits/templates/ONA-112.csv`

**ONA-113. In orthopaedic inpatients on treatment antibiotics, was a documented review decision made 48 to 72 hours after starting them?** (new)
- Standard: UKHSA Start Smart Then Focus toolkit (updated 12 September 2023); NICE NG15 rec 1.1.39: "Review and revise the clinical diagnosis and the continuing need for antimicrobials by 48 to 72 hours and document a clear plan of action – the antimicrobial review outcome." https://www.gov.uk/government/publications/antimicrobial-stewardship-start-smart-then-focus/start-smart-then-focus-antimicrobial-stewardship-toolkit-for-inpatient-care-settings
- Pass: A documented review between 48 and 72 hours after the first dose, with one outcome recorded (stop, switch to oral, change, continue with new review/stop date, or OPAT).. Target: ≥90%. Sample: 40 consecutive courses over the last 4–6 weeks.
- Change: Turn on a mandatory 72-hour antibiotic review task in e-prescribing that must be answered with one outcome before the next dose can be given.
- Template: `new_audits/templates/ONA-113.csv`

**ONA-114. In non-bleeding adults transfused after orthopaedic surgery, was each red cell unit followed by a haemoglobin check before the next unit?** (under-audited)
- Standard: NICE NG24 recs 1.6.1 and 1.6.2 (2015): "1.6.1 Consider single‑unit red blood cell transfusions for adults (or equivalent volumes calculated based on body weight for children or adults with low body weight) who do not have active bleeding. 1.6.2 After each single‑unit red blood cell transfusion (or equivalent volumes calculated based on body weight for children or adults with low body weight), clinically reassess and check haemoglobin levels, and give further transfusions if needed." https://www.nice.org.uk/guidance/ng24/chapter/Red-blood-cell-transfusion
- Pass: Every unit given as a single unit, with a haemoglobin result between units and a pre-transfusion haemoglobin below the local threshold (70 g/L, or 80 g/L with acute coronary syndrome).. Target: ≥90%. Sample: 40 consecutive transfusion episodes over the last 3 months.
- Change: Change the electronic blood request so that one unit is the default for non-bleeding patients, with a required reason for requesting two.
- Template: `new_audits/templates/ONA-114.csv`

**ONA-115. In adults transfused during an orthopaedic admission, did the discharge summary state the transfusion and that they can no longer donate blood?** (new)
- Standard: NICE NG24 rec 1.14.4 (2015): "Provide the person and their GP with copies of the discharge summary or other written communication that explains: the details of any transfusions they had; the reasons for the transfusion; any adverse events; that they are no longer eligible to donate blood." https://www.nice.org.uk/guidance/ng24/chapter/Patient-information
- Pass: Discharge summary includes all four items: transfusion details, reason, adverse events (or 'none') and ineligibility to donate.. Target: ≥90%. Sample: 40 consecutive transfused orthopaedic discharges over the last 3 months.
- Change: Add a transfusion section to the discharge summary template that auto-fills from the transfusion lab system and includes the 'no longer eligible to donate' line.
- Template: `new_audits/templates/ONA-115.csv`

**ONA-116. In adults having orthopaedic trauma surgery other than hip fracture, was tranexamic acid given at the start of surgery?** (new)
- Standard: NICE NG24 recs 1.3.1 and 1.3.3 (updated 2026): "1.3.1 Offer tranexamic acid to adults having surgery in an operating theatre if: there is any risk of bleeding and the procedure will breach the skin or mucous membranes. 1.3.3 When using tranexamic acid for adults having surgery, administer it just before the start of surgery. Typically give 1 g by slow intravenous injection." https://www.nice.org.uk/guidance/ng24/chapter/Reducing-requirement-for-blood-transfusion-for-people-having-surgery
- Pass: IV tranexamic acid given before or at knife-to-skin, or a documented contraindication.. Target: ≥85%. Sample: 40 consecutive eligible trauma operations over the last 4 weeks.
- Change: Add 'Tranexamic acid given / contraindicated' to the Time Out on the electronic WHO checklist for all trauma cases.
- Template: `new_audits/templates/ONA-116.csv`

**ONA-117. In adults aged 65 or over, or with an eGFR below 60, listed for elective hip or knee replacement, did the pre-operative assessment record their risk of acute kidney injury?** (new)
- Standard: NICE NG148 rec 1.1.13 (2019, updated 2024): "Assess the risk of acute kidney injury in adults before surgery. Be aware that increased risk is associated with: emergency surgery, especially when the person has sepsis or hypovolaemia; intraperitoneal surgery; chronic kidney disease (adults with an eGFR less than 60 ml/min/1.73 m2 are at particular risk); diabetes; heart failure; age 65 years or over; liver disease; use of drugs that can cause or exacerbate kidney injury in the perioperative period (in particular, NSAIDs after surgery). Use the risk assessment to inform a clinical management plan." https://www.nice.org.uk/guidance/ng148/chapter/Recommendations
- Pass: Pre-op record documents AKI risk and a plan for ACE inhibitors/ARBs, diuretics and post-op NSAIDs (hold, continue or avoid).. Target: ≥90%. Sample: 40 consecutive eligible patients over the last 2–3 months.
- Change: Add an AKI risk box with a medicines plan (ACE inhibitor/ARB/diuretic/NSAID) to the pre-op assessment template, auto-shown when age ≥65 or eGFR <60.
- Template: `new_audits/templates/ONA-117.csv`

**ONA-118. In adults aged 65 or over admitted with an orthopaedic injury other than hip fracture, was a delirium screen documented within 24 hours of admission?** (under-audited)
- Standard: NICE CG103 recs 1.2.1 and 1.3.1 (2010, updated 2023): "1.2.1 When people first present to hospital or long-term care, assess them for the following risk factors. If any of these risk factors are present, the person is at risk of delirium. Age 65 years or older. ... 1.3.1 At presentation, assess people at risk for recent (within hours or days) changes or fluctuations that may indicate delirium." https://www.nice.org.uk/guidance/cg103/chapter/Recommendations
- Pass: A validated delirium screen (e.g. 4AT) documented within 24 hours of the admission time.. Target: ≥90%. Sample: 40 consecutive eligible admissions over the last 4–6 weeks.
- Change: Add a 4AT field to the orthopaedic admission clerking proforma that is required for all patients aged 65 or over.
- Template: `new_audits/templates/ONA-118.csv`

**ONA-119. In orthopaedic ward patients with suspected infection and a NEWS2 score of 7 or more, were IV antibiotics given within 1 hour of that score?** (new)
- Standard: NICE NG253 rec 1.8.3 (2025, updated 2026): "Give people aged 16 or over who are at high risk of severe illness or death from sepsis broad-spectrum intravenous antibiotic treatment, within 1 hour of calculating the person's NEWS2 score on initial assessment in the emergency department or on ward deterioration." https://www.nice.org.uk/guidance/ng253/chapter/Managing-suspected-sepsis
- Pass: First IV antibiotic dose administered within 60 minutes of the first NEWS2 ≥7 recorded with suspected infection.. Target: ≥90%. Sample: All eligible episodes over the last 6–12 months (expect 30–40), found from e-observations NEWS2 ≥7 reports.
- Change: Set e-observations so a NEWS2 ≥7 on an orthopaedic ward pages the on-call doctor with a sepsis screen, and stock a sepsis grab bag with first-dose antibiotics on each orthopaedic ward.
- Template: `new_audits/templates/ONA-119.csv`

**ONA-120. In adults discharged with opioids after orthopaedic surgery, did the discharge letter state the opioid dose, amount supplied and planned duration?** (new)
- Standard: Faculty of Pain Medicine / RCoA 'Surgery and Opioids: Best Practice Guidelines 2021', Discharge Planning 3 and 4: "The hospital discharge letter should be available in a timely way and provided to all healthcare professionals involved in caring for the patient, including community pharmacists, to avoid an acute prescription of opioids inadvertently becoming a repeat prescription. The hospital discharge letter must explicitly state the recommended opioid dose, amount supplied and planned duration of use. ... Usually 5 days and no more than 7 days of opioids (including Tramadol) should be prescribed." https://www.cpoc.org.uk/sites/cpoc/files/documents/2021-03/surgery-and-opioids-2021.pdf
- Pass: Discharge letter states opioid dose, amount supplied and planned duration or stop date, and supply is 7 days or less.. Target: ≥90%. Sample: 40 consecutive eligible discharges over the last 4 weeks.
- Change: Add a required 'opioid plan' block (dose, quantity, stop date, GP not to add to repeats) to the discharge TTO template when any opioid is prescribed.
- Template: `new_audits/templates/ONA-120.csv`

**ONA-121. In orthopaedic inpatients with Parkinson's disease, was every dose of their Parkinson's medicine given within 30 minutes of the prescribed time in the first 72 hours?** (under-audited)
- Standard: NICE NG71 recs 1.3.2 and 1.3.4 (2017): "1.3.2 Antiparkinsonian medicines should not be withdrawn abruptly or allowed to fail suddenly due to poor absorption (for example, gastroenteritis, abdominal surgery) to avoid the potential for acute akinesia or neuroleptic malignant syndrome. 1.3.4 In view of the risks of sudden changes in antiparkinsonian medicines, people with Parkinson's disease who are admitted to hospital or care homes should have their medicines: given at the appropriate times, which in some cases may mean allowing self-medication; adjusted by, or adjusted only after discussion with, a specialist in the management of Parkinson's disease." https://www.nice.org.uk/guidance/ng71/chapter/Recommendations
- Pass: All Parkinson's medicine doses due in the first 72 hours given within 30 minutes of the prescribed time (local definition of 'appropriate times'), including during nil-by-mouth periods.. Target: ≥90% of patients with all doses on time. Sample: All eligible admissions over the last 12 months (expect 30–40), from pharmacy or e-prescribing search for Parkinson's drugs on orthopaedic wards.
- Change: Flag Parkinson's drugs as time-critical in e-prescribing (exact-time dosing and alert when a dose is 30 minutes late) and add a Parkinson's NBM plan to the pre-op checklist.
- Template: `new_audits/templates/ONA-121.csv`

**ONA-122. In adults having elective primary hip or knee replacement, was a urinary catheter avoided unless a clinical indication was documented?** (new)
- Standard: Intercollegiate Green Theatre Checklist v2.0 (November 2024), item 8: "Avoid clinically unnecessary interventions (e.g. antibiotics, urinary catheterisation, histology examinations)" https://www.rcsed.ac.uk/media/zs2nlvpj/green-theatre-checklist.pdf
- Pass: No urinary catheter inserted in theatre, or a catheter with a documented indication (e.g. retention history, long case, critical fluid balance).. Target: ≥90%. Sample: 40 consecutive elective primary hip or knee replacements over the last 6 weeks.
- Change: Remove catheter packs from the default arthroplasty pick list and add 'catheter: indication' to the Team Brief.
- Template: `new_audits/templates/ONA-122.csv`

**ONA-123. In adults having hemiarthroplasty or total hip replacement for hip fracture, was the femoral head sent for histology only when a clinical indication was documented?** (new)
- Standard: Intercollegiate Green Theatre Checklist v2.0 (November 2024), item 8: "Avoid clinically unnecessary interventions (e.g. antibiotics, urinary catheterisation, histology examinations)" https://www.rcsed.ac.uk/media/zs2nlvpj/green-theatre-checklist.pdf
- Pass: No femoral head histology, or histology with a documented indication (suspected pathological fracture, known malignancy, unusual appearance, or bone bank donation screening).. Target: ≥90%. Sample: 40 consecutive cases over the last 2–3 months.
- Change: Agree a written indication list with pathology and add 'histology: indication or not required' to the Sign Out specimen check.
- Template: `new_audits/templates/ONA-123.csv`

## Fracture clinic, outpatients, imaging, bone health and infection

**ONA-124. What proportion of new fracture clinic referrals have a named consultant review documented within 24 hours of first presentation?** (under-audited)
- Standard: BOASt Outpatient and on-call services for people with fractures or musculoskeletal injury (BOA, February 2026), standard 2: "Patients must have their case reviewed by a named consultant within 24 hours of first presentation. This can be performed in person or virtually in a trauma meeting. The outcome of this review should be documented and a copy sent to the patient and GP." https://www.boa.ac.uk/resource/boast-outpatient-and-on-call-services-for-people-with-fractures-or-musculoskeletal-injury.html
- Pass: Documented review by a named consultant (in person, virtual fracture clinic or trauma meeting) with date and time no more than 24 hours after first presentation (ED or community arrival time).. Target: ≥90%. Sample: 40 consecutive referrals from the VFC or fracture clinic referral list over the last 4 weeks.
- Change: Add a mandatory 'reviewing consultant' and 'review time' field to the VFC outcome template, and add a weekend VFC or trauma-meeting review slot on the rota.
- Template: `new_audits/templates/ONA-124.csv`

**ONA-125. What proportion of patients reviewed in the virtual fracture clinic are sent a copy of the review outcome?** (new)
- Standard: BOASt Outpatient and on-call services for people with fractures or musculoskeletal injury (BOA, February 2026), standard 2: "The outcome of this review should be documented and a copy sent to the patient and GP." https://www.boa.ac.uk/resource/boast-outpatient-and-on-call-services-for-people-with-fractures-or-musculoskeletal-injury.html
- Pass: A letter, text or portal message stating the VFC outcome is addressed or copied to the patient and dated within 7 days of the review.. Target: ≥90%. Sample: 40 consecutive VFC outcomes from the last 4 weeks.
- Change: Set the VFC letter template to add the patient as a recipient by default (opt-out, with a reason box).
- Template: `new_audits/templates/ONA-125.csv`

**ONA-126. What proportion of new fracture clinic patients have a definitive treatment plan documented within 72 hours of first presentation?** (under-audited)
- Standard: BOASt Outpatient and on-call services for people with fractures or musculoskeletal injury (BOA, February 2026), standard 4: "A definitive treatment plan should be documented for all patients within 72 hours of first presentation and copied to the patient and their GP." https://www.boa.ac.uk/resource/boast-outpatient-and-on-call-services-for-people-with-fractures-or-musculoskeletal-injury.html
- Pass: A documented plan stating the definitive treatment (for example non-operative with named immobilisation and duration, or surgery with a date or listing) within 72 hours of first presentation. 'See in clinic' alone fails.. Target: ≥90%. Sample: 40 consecutive new referrals over the last 4 weeks.
- Change: Replace free-text VFC outcomes with a structured field 'Definitive plan' with drop-down options, which cannot be left as 'review'.
- Template: `new_audits/templates/ONA-126.csv`

**ONA-127. What proportion of first fracture clinic letters for lower limb injuries record weightbearing status using the BOASt terms?** (new)
- Standard: BOASt Mobilisation and weightbearing after orthopaedic surgery or musculoskeletal injury (BOA, August 2024), standards 3–5; also BOASt Outpatient and on-call services for people with fractures or musculoskeletal injury (BOA, February 2026): "3. A weightbearing status should be attributed to each affected limb. 4. The following specific terms should be used to define weightbearing status: a. Non Weightbearing b. Limited Weightbearing c. Unrestricted Weightbearing 5. Terms such as touch, partial, proportional, permissive, or progressive weightbearing should no longer be used." https://www.boa.ac.uk/resource/mobilisation-and-weightbearing-after-orthopaedic-surgery-musculoskeletal-injury-boast.html
- Pass: The first fracture clinic or VFC letter states weightbearing for the injured limb using only 'non weightbearing', 'limited weightbearing' or 'unrestricted weightbearing'.. Target: ≥90%. Sample: 40 consecutive adults with non-operative lower limb injuries from the last 4 weeks.
- Change: Add a weightbearing drop-down (NWB / LWB / UWB, with reason and duration boxes) to the fracture clinic letter template.
- Template: `new_audits/templates/ONA-127.csv`

**ONA-128. What proportion of fracture clinic patients aged 50 or over with a fragility fracture are identified by the fracture liaison service?** (under-audited)
- Standard: Royal Osteoporosis Society, Effective Secondary Prevention of Fragility Fractures: Clinical Standards for Fracture Liaison Services, version 3 (September 2025), standard 1.1; and BOASt Outpatient and on-call services for people with fractures or musculoskeletal injury (BOA, February 2026): "1.1. The FLS identifies people aged 50 years or older presenting with a new fragility fracture." https://royal-osteoporosis-society.uksouth01.umbraco.io/media/arqkorn2/ros-clinical-standards-for-fracture-liaison-service.pdf
- Pass: Patient appears on the FLS database or has a documented FLS referral or FLS contact within 12 weeks of fracture diagnosis.. Target: ≥80%. Sample: 40 consecutive eligible patients from fracture clinic lists 3–4 months ago (to allow 12 weeks).
- Change: Add an automatic weekly report of fracture clinic patients aged 50+ sent to the FLS, or a mandatory 'FLS referral: yes/no/why not' field in the VFC template.
- Template: `new_audits/templates/ONA-128.csv`

**ONA-129. What proportion of patients identified by the fracture liaison service complete their FLS assessment within 12 weeks of fracture diagnosis?** (under-audited)
- Standard: Royal Osteoporosis Society, Effective Secondary Prevention of Fragility Fractures: Clinical Standards for Fracture Liaison Services, version 3 (September 2025), criterion 2.2: "2.2. Assessment will be completed within 12 weeks of fracture diagnosis." https://royal-osteoporosis-society.uksouth01.umbraco.io/media/arqkorn2/ros-clinical-standards-for-fracture-liaison-service.pdf
- Pass: Documented FLS assessment (fracture risk assessment with FRAX or QFracture, with DXA where indicated) completed no more than 84 days after fracture diagnosis.. Target: ≥80%. Sample: 40 consecutive patients identified by the FLS 4–6 months ago.
- Change: Book DXA directly from the FLS at identification (protocol-based request) instead of asking the GP to request it.
- Template: `new_audits/templates/ONA-129.csv`

**ONA-130. What proportion of FLS patients aged 65 or over have a documented falls risk assessment?** (under-audited)
- Standard: Royal Osteoporosis Society, Effective Secondary Prevention of Fragility Fractures: Clinical Standards for Fracture Liaison Services, version 3 (September 2025), criterion 2.1: "2.1. The FLS offers people identified as being at increased risk of another fragility fracture, an assessment which will include: ... An assessment of falls risk in people aged 65 or over." https://royal-osteoporosis-society.uksouth01.umbraco.io/media/arqkorn2/ros-clinical-standards-for-fracture-liaison-service.pdf
- Pass: A falls risk assessment (screening questions, gait or balance test, or referral to a falls service) is documented in the FLS record.. Target: ≥90%. Sample: 40 consecutive FLS patients aged 65+ from the last 3 months.
- Change: Add a mandatory three-question falls screen with an automatic falls service referral trigger to the FLS assessment template.
- Template: `new_audits/templates/ONA-130.csv`

**ONA-131. What proportion of FLS patients recommended bone treatment are followed up by the FLS within 8 weeks of the recommendation?** (new)
- Standard: Royal Osteoporosis Society, Effective Secondary Prevention of Fragility Fractures: Clinical Standards for Fracture Liaison Services, version 3 (September 2025), criterion 4.4: "4.4. The FLS reviews people who are recommended interventions to reduce risk of fracture 4-8 weeks after a treatment recommendation is made, and at 52 weeks to ensure that: lifestyle recommendations and treatment decisions are reassessed; where the decision has been taken to treat, it has been started and taken appropriately. referral to falls reduction programmes has been actioned if appropriate." https://royal-osteoporosis-society.uksouth01.umbraco.io/media/arqkorn2/ros-clinical-standards-for-fracture-liaison-service.pdf
- Pass: Documented FLS contact (phone, letter reply or clinic) no more than 8 weeks after the treatment recommendation, recording whether treatment has started.. Target: ≥80%. Sample: 40 consecutive patients recommended treatment 3–5 months ago.
- Change: Create a tracked follow-up task (EPR or FLS database reminder) at the time of each treatment recommendation.
- Template: `new_audits/templates/ONA-131.csv`

**ONA-132. What proportion of radiology reports describing vertebral body height loss call it a 'vertebral fracture'?** (under-audited)
- Standard: Royal Osteoporosis Society, Effective Secondary Prevention of Fragility Fractures: Clinical Standards for Fracture Liaison Services, version 3 (September 2025), Standard 1 (Identify), quoting ROS radiology reporting guidance: "Guidance on identification and reporting of vertebral fractures is given by the ROS. This recommends that radiology: Review the spine in all images of the chest, abdomen and pelvis. Report vertebral fractures clearly using the term ‘vertebral fracture’. Recommend further assessment and management to reduce fracture risk." https://royal-osteoporosis-society.uksouth01.umbraco.io/media/arqkorn2/ros-clinical-standards-for-fracture-liaison-service.pdf
- Pass: Report uses the words 'vertebral fracture' (not only 'wedge', 'collapse', 'height loss' or 'compression'). Whether it also recommends a bone health assessment is recorded as a secondary measure.. Target: ≥80%. Sample: 40 consecutive matching reports from a RIS free-text search over the last 2 months.
- Change: Add a reporting macro in RIS ('Vertebral fracture at Tx. Recommend fracture risk assessment / FLS referral') and send an automatic copy to the FLS.
- Template: `new_audits/templates/ONA-132.csv`

**ONA-133. What proportion of fracture clinic letters are addressed directly to the patient?** (new)
- Standard: Academy of Medical Royal Colleges, Please write to me: guidance for writing directly to patients (April 2026 update), section 'Electronic Patient Records and Artificial Intelligence': "Templates for creating letters within the Electronic Patient Record (EPR) should be consistent with the sections listed above. The default format should be a letter addressed to the patient." https://www.aomrc.org.uk/wp-content/uploads/2026/06/Please_write_to_me_update_130426.pdf
- Pass: The letter opens 'Dear [patient name]' (or parent/carer for a child) and is written in the second person, with the GP copied.. Target: ≥80%. Sample: 40 consecutive fracture clinic letters from the last 2 weeks, across at least 4 clinicians.
- Change: Change the fracture clinic letter template default to 'Dear [patient]' with GP as copy recipient.
- Template: `new_audits/templates/ONA-133.csv`

**ONA-134. What proportion of fracture clinic letters give the patient a hospital phone number and email address for questions?** (new)
- Standard: Academy of Medical Royal Colleges, Please write to me: guidance for writing directly to patients (April 2026 update), 'Structure and content of outpatient letters': "An outpatient clinic letter should include: ... A hospital phone number and email address for follow-up questions and correcting errors." https://www.aomrc.org.uk/wp-content/uploads/2026/06/Please_write_to_me_update_130426.pdf
- Pass: The letter contains both a direct hospital phone number (fracture clinic or team, not switchboard only) and an email address.. Target: ≥95%. Sample: 40 consecutive letters from the last 2 weeks.
- Change: Add a fixed footer with the fracture clinic direct line and team email to the letter template.
- Template: `new_audits/templates/ONA-134.csv`

**ONA-135. What proportion of fracture clinic letters to patients reach a Flesch reading ease score of 70 or more?** (new)
- Standard: Academy of Medical Royal Colleges, Please write to me: guidance for writing directly to patients (April 2026 update), 'Making your writing easier to read and understand': "Avoid complex words and sentences. Measure the Flesch reading ease score and the UK reading age using the NHS Medical Document Readability Tool. Aim for a Flesch reading ease score of 70 or above and a UK reading age of 9–11 years." https://www.aomrc.org.uk/wp-content/uploads/2026/06/Please_write_to_me_update_130426.pdf
- Pass: Patient-facing text of the letter (excluding address block and medication list) scores Flesch reading ease ≥70.. Target: ≥70% of letters. Sample: 40 consecutive letters over 2 weeks, at least 4 authors.
- Change: Replace the template with plain-English section headings and a stock phrase bank (for example 'broken bone' for 'fracture', explained once).
- Template: `new_audits/templates/ONA-135.csv`

**ONA-136. What proportion of fracture clinic patients moved to patient-initiated follow-up are sent a letter confirming PIFU, copied to their GP?** (new)
- Standard: NHS England, Guide to implementing PIFU in adult trauma and orthopaedic secondary care pathways (published 9 Feb 2023, updated 7 Oct 2025), section 'Shared decision-making': "Good practice includes writing to individuals to confirm they will be transferred to PIFU (to include a summary of what the process involves, the decision that has been made and the associated benefits and harms), with a copy of the letter sent to their GP." https://www.england.nhs.uk/long-read/guide-to-implementing-patient-initiated-follow-up-pifu-in-adult-trauma-and-orthopaedic-secondary-care-pathways/
- Pass: Letter to the patient that says they are on PIFU, how to trigger an appointment, and the PIFU end date, with the GP copied.. Target: ≥90%. Sample: 40 consecutive patients put on PIFU over the last 2 months (from the PAS PIFU list).
- Change: Add a PIFU paragraph (contact route, end date, what happens at the end) that auto-inserts when the PIFU outcome is chosen on the clinic outcome form.
- Template: `new_audits/templates/ONA-136.csv`

**ONA-137. What proportion of fracture clinic PIFU pathways past their end date are closed with discharge documented in the notes and on PAS?** (new)
- Standard: NHS England, Guide to implementing PIFU in adult trauma and orthopaedic secondary care pathways (published 9 Feb 2023, updated 7 Oct 2025), section 'Ending the PIFU pathway': "The trauma and orthopaedic team should agree how to end each patient’s pathway safely, to minimise the risk they get lost in the system. Concluded pathways should be documented in patient medical records and on the trust Patient Administration System." https://www.england.nhs.uk/long-read/guide-to-implementing-patient-initiated-follow-up-pifu-in-adult-trauma-and-orthopaedic-secondary-care-pathways/
- Pass: Within 4 weeks after the PIFU end date, PAS shows the pathway closed (discharged) and the notes or a letter record the discharge.. Target: ≥95%. Sample: 40 consecutive PIFU pathways by end date.
- Change: A weekly automated PAS report of expired PIFU pathways sent to the trauma co-ordinator, with a standard discharge letter.
- Template: `new_audits/templates/ONA-137.csv`

**ONA-138. What proportion of X-ray requests from fracture clinic state the clinical question the image should answer?** (under-audited)
- Standard: Ionising Radiation (Medical Exposure) Regulations 2017, regulation 10(5): "The referrer must supply the practitioner with sufficient medical data (such as previous diagnostic information or medical records) relevant to the exposure requested by the referrer to enable the practitioner to decide whether there is a sufficient net benefit as required by regulation 11(1)(b)." https://www.legislation.gov.uk/uksi/2017/1322/regulation/10
- Pass: Request states the injury, time since injury or surgery, and the specific question (for example 'check position after 1 week', 'union?'). 'Check X-ray' or 'fracture clinic' alone fails.. Target: ≥90%. Sample: 40 consecutive fracture clinic X-ray requests over 1 week.
- Change: Add a mandatory 'clinical question' drop-down (position check / union / hardware / new symptom / other) to the fracture clinic order set.
- Template: `new_audits/templates/ONA-138.csv`

**ONA-139. What proportion of trauma operations using fluoroscopy have the radiation dose recorded in the operation note or clinical evaluation?** (under-audited)
- Standard: Ionising Radiation (Medical Exposure) Regulations 2017, Schedule 2 paragraph 1(j): "The employer’s written procedures for exposures must include procedures— ... (j) for the carrying out and recording of a clinical evaluation for each exposure including, where appropriate, factors relevant to patient dose;" https://www.legislation.gov.uk/uksi/2017/1322/schedule/2
- Pass: Operation note (or linked evaluation) records dose-area product and/or screening time, plus a statement of what the images showed.. Target: ≥90%. Sample: 40 consecutive trauma cases with fluoroscopy over 2–3 weeks.
- Change: Add a mandatory 'Imaging: DAP / screening time / findings' section to the trauma operation note template.
- Template: `new_audits/templates/ONA-139.csv`

**ONA-140. What proportion of patients aged 12–55 of childbearing potential have a documented pregnancy enquiry before intraoperative fluoroscopy?** (new)
- Standard: Ionising Radiation (Medical Exposure) Regulations 2017, regulation 11(1)(f): "A person must not carry out an exposure unless— ... (f) in the case of an individual of childbearing potential, the person has enquired whether that individual is pregnant or breastfeeding, if relevant." https://www.legislation.gov.uk/uksi/2017/1322/regulation/11
- Pass: A pregnancy enquiry (status or test result) is recorded on the day of surgery before the first exposure, on the checklist, anaesthetic chart or radiographer record.. Target: 100%. Sample: 40 consecutive eligible patients over 3–4 weeks (trauma and elective).
- Change: Add a hard-stop pregnancy/breastfeeding line to the radiographer exposure record and the theatre sign-in.
- Template: `new_audits/templates/ONA-140.csv`

**ONA-141. What proportion of debridements or revision operations for suspected fracture-related infection send five separate deep microbiology samples?** (under-audited)
- Standard: BOAST Fracture Related Infections (BOA, September 2019; still the current BOAST), 'Assessment of FRI' and 'Secondary fracture surgery': "When debridement is indicated 5 samples should be taken from around the fracture site for microbiological culture using separate sterile instruments and a no touch technique for each. In all chronic infections and when the diagnosis is in doubt in acute infections take 2 samples for histology." https://www.boa.ac.uk/resource/boast-fracture-related-infections.html
- Pass: Five or more separate deep tissue samples received by microbiology from the operation, with separate instruments documented.. Target: ≥90%. Sample: All eligible operations over the last 12–18 months, up to 40.
- Change: A pre-packed 'bone infection sampling kit' (5 labelled pots, 5 sets of instruments, request forms) kept in trauma theatre, added to the WHO sign-in for these cases.
- Template: `new_audits/templates/ONA-141.csv`

**ONA-142. What proportion of non-septic patients with suspected acute prosthetic joint infection receive no antibiotics before deep tissue samples are taken?** (under-audited)
- Standard: BOAST Acute Management of Peri-Prosthetic Joint Infection (BOA, October 2023, updated April 2024), standard 4: "A patient who is not septic should not be given antibiotics until appropriate deep tissue samples have been taken." https://www.boa.ac.uk/resource/boast-acute-management-of-peri-prosthetic-joint-infection.html
- Pass: First antibiotic dose time is after the time of joint aspiration or surgical deep sampling (or no antibiotics before sampling).. Target: ≥90%. Sample: All eligible cases over the last 12 months, up to 40.
- Change: An ED and orthopaedic electronic alert on 'suspected joint replacement infection': 'Not septic? Withhold antibiotics until aspiration – call orthopaedics'.
- Template: `new_audits/templates/ONA-142.csv`

**ONA-143. What proportion of hip and knee replacement discharge summaries tell the patient who to contact if they suspect joint infection?** (new)
- Standard: BOAST Acute Management of Peri-Prosthetic Joint Infection (BOA, October 2023, updated April 2024), standard 1: "Guidance on who to contact and how to respond to a patient with suspected PJI should be readily available to Primary Care providers and Emergency department practitioners, and be included in discharge documentation." https://www.boa.ac.uk/resource/boast-acute-management-of-peri-prosthetic-joint-infection.html
- Pass: Discharge summary (or attached discharge leaflet, documented) names warning signs of infection and gives a specific contact (ward or arthroplasty phone number) plus the advice not to start antibiotics before orthopaedic review.. Target: ≥95%. Sample: 40 consecutive discharges over the last 4–6 weeks.
- Change: Add a fixed 'If you think your joint is infected' box with the arthroplasty hotline and GP advice to the arthroplasty discharge template.
- Template: `new_audits/templates/ONA-143.csv`

**ONA-144. What proportion of fracture fixation patients are given written advice on how to recognise wound infection and who to contact?** (new)
- Standard: NICE NG125 Surgical site infections: prevention and treatment, rec 1.1.3 (2008, NG125 published 2019, updated 2020): "Offer patients and carers information and advice about how to recognise a surgical site infection and who to contact if they are concerned. Use an integrated care pathway for healthcare-associated infections to help communicate this information to both patients and all those involved in their care after discharge." https://www.nice.org.uk/guidance/ng125/chapter/Recommendations
- Pass: Discharge summary or documented leaflet states signs of wound infection and a named contact route (ward, fracture clinic or dressing clinic number).. Target: ≥95%. Sample: 40 consecutive discharges over the last 4 weeks.
- Change: Add a mandatory 'wound care and infection' section (signs, contact number, shower advice) to the trauma discharge summary template.
- Template: `new_audits/templates/ONA-144.csv`

**ONA-145. What proportion of trauma patients are told in their discharge summary that they received antibiotics during their operation?** (new)
- Standard: NICE NG125 rec 1.1.4 (2008, NG125 published 2019, updated 2020): "Always inform patients after their operation if they have been given antibiotics." https://www.nice.org.uk/guidance/ng125/chapter/Recommendations
- Pass: Discharge summary (patient section) states the antibiotic given at surgery, or the notes record that the patient was told.. Target: ≥90%. Sample: 40 consecutive trauma discharges over the last 4 weeks.
- Change: Auto-pull theatre antibiotics into the discharge summary with a plain-English line ('During your operation you were given the antibiotic ...').
- Template: `new_audits/templates/ONA-145.csv`

**ONA-146. What proportion of limb operations under tourniquet have prophylactic antibiotic given before tourniquet inflation?** (under-audited)
- Standard: NICE NG125 rec 1.2.15 (2008, NG125 published 2019, updated 2020): "Consider giving a single dose of antibiotic prophylaxis intravenously on starting anaesthesia. However, give prophylaxis earlier for operations in which a tourniquet is used." https://www.nice.org.uk/guidance/ng125/chapter/Recommendations
- Pass: Antibiotic administration time is before the tourniquet inflation time (local policy may add a minimum gap, for example ≥5 minutes).. Target: ≥95%. Sample: 40 consecutive tourniquet cases over 2–3 weeks.
- Change: Add 'antibiotic given – time' as a required item before the tourniquet inflation entry in the theatre system (WHO time-out prompt).
- Template: `new_audits/templates/ONA-146.csv`

## Orthopaedic perioperative, theatre and ward care

**ONA-147. When a patient is cancelled from the trauma list, are they offered food or drink within 1 hour of the cancellation?** (new)
- Standard: Local standard (no national standard identified): "Patients cancelled from a theatre list are offered food or drink within 60 minutes of the decision to cancel." 
- Pass: Documented offer of food or drink within 60 minutes of the recorded cancellation decision, in patients with no clinical reason to stay nil by mouth.. Target: ≥90%. Sample: Every trauma list cancellation over 4 weeks (typically 20–40); extend until 30 cases.
- Change: Add a 'Cancelled: may eat and drink now?' step to the theatre coordinator's cancellation checklist; the coordinator phones the ward with a one-line script and records the time.
- Template: `new_audits/templates/ONA-147.csv`
- Pitfalls: No time is recorded when a case is cancelled → agree before week 1 that the coordinator's phone call time is the start time, and put it on the checklist; Staff keep patients fasted 'in case they go later' → record 'later list still possible today' and count only definite cancellations; Ward nurses feel blamed → present it as a theatre-to-ward communication gap; show results by day of week, never by named ward or nurse; The fix fades when the audit team rotates → build it into the coordinator checklist, not a poster
- Pearls: Recruit the trauma coordinator and one ward sister in week 1; they collect the data and run the fix; Use a one-line script: '[Patient] is cancelled today and may eat and drink now'; Record next-day nil-by-mouth time as well; it is often the larger harm; Pilot on one ward for 2 weeks before rolling out; name the theatre matron as owner afterwards; Bring one patient story (e.g. 30 hours without food) to governance alongside the percentages

**ONA-148. On elective orthopaedic lists, what proportion of turnarounds take 15 minutes or less?** (under-audited)
- Standard: Local standard; 15-minute turnaround target as used in [2792]: "Turnaround (patient leaves theatre to start of the next patient's anaesthetic) of 15 minutes or less." 
- Pass: Time from one patient leaving theatre to the start of the next patient's anaesthetic is 15 minutes or less.. Target: ≥50% of turnarounds within 15 minutes and median reduced by 10 minutes. Sample: Every turnaround on elective orthopaedic lists over 4 weeks (typically 80–150), plus the previous 4 weeks from the theatre system as a baseline.
- Change: Make 'send for next patient' a fixed step on the theatre checklist, triggered at the start of wound closure and time-logged by the coordinator. If the delay data point elsewhere, fix the leading delay reason instead.
- Template: `new_audits/templates/ONA-148.csv`
- Pitfalls: Theatre-system times are entered late or missing → spot-check 10 cases against a stopwatch before week 1; hand-collect week 1 if they differ; 'Orthopaedic turnaround is longer because of laminar flow and big trays' → report median by procedure type (joints, hands, trauma), not one overall figure; Staff speed up only while watched → use 4 weeks of past theatre-system data as a hidden baseline; Nurses and ODPs hear 'work faster' → involve the theatre manager from day 1 and frame it as removing waits (porters, beds, trays)
- Pearls: Record a delay reason for every turnaround; it tells you which fix to use; Find the list that already turns around fastest and copy its routine; Convert minutes to cases: 20 min saved × 3 turnarounds ≈ one extra case a day; Book the porter for the send-for time, not 'when free'; Take one chart to governance: weekly median turnaround with a line where the change went in

**ONA-149. On elective orthopaedic lists, what percentage of the planned session time is touch time?** (new)
- Standard: GIRFT touchtime utilisation target, as reported in PMC11488670 (CC BY): "The Getting It Right First Time (GIRFT) programme has set targets to achieve 85% touchtime utilisation by 2024/25. Touchtime utilisation is a measure of theatre productivity, defined as the time from the start of anaesthesia to the time a patient leaves the theatre for all cases on a defined theatre list as a percentage of total available theatre time." https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11488670/
- Pass: List touch time (start of anaesthesia to leaving theatre, summed over all cases, within the planned session) is 85% or more of planned session time.. Target: ≥85% touch time. Sample: All elective orthopaedic lists over 4 weeks (one row per list).
- Change: Fix the largest single source of lost time found: usually the first-case start, using a golden patient (first case identified and prepared the day before).
- Template: `new_audits/templates/ONA-149.csv`
- Pitfalls: The percentage rewards overrunning or under-booked lists → report late start, turnaround and early finish minutes alongside it; Session times in the system differ from real contracted times → confirm planned session length with the theatre manager before week 1; Surgeons dispute the data → share the raw list-by-list sheet with each team before presenting
- Pearls: Break lost time into late start, turnarounds and early finish; the fix follows the biggest bar; Pair with the turnaround audit and share one data export; Report cases gained, not percentages, to managers

**ONA-150. In elective hip and knee replacements, what percentage of theatre waste by weight goes into the orange (infectious) waste stream?** (under-audited)
- Standard: Intercollegiate Green Theatre Checklist v2.0 (November 2024), item 14: "RECYCLE/use lowest carbon appropriate waste streams: use recycling waste streams for packaging or, if not available, domestic waste stream (prior to patient entering the room); use non-infectious offensive waste streams (yellow/black tiger) unless clear risk of infection (orange); ensure only appropriate contents in sharps bins (sharps/drugs)" https://www.rcsed.ac.uk/media/zs2nlvpj/green-theatre-checklist.pdf
- Pass: Per case: orange-stream waste as a percentage of total theatre waste weight. Local target: orange share halved from cycle 1.. Target: Orange share halved from cycle 1. Sample: 20 consecutive cases (about 10 hips, 10 knees).
- Change: Place tiger (offensive) and recycling bins in every arthroplasty theatre; open packaging into recycling before the patient enters; keep the orange bin for items with a clear infection risk, as defined in writing with infection prevention.
- Template: `new_audits/templates/ONA-150.csv`
- Pitfalls: Infection prevention objects to moving waste out of orange → agree a written rule for what stays orange before week 1 and quote Green Theatre Checklist item 14; The waste contract has no offensive stream → involve the estates waste manager in week 1; the change date depends on the contract; No scales in theatre → borrow a hanging scale from estates or the waste contractor; weigh each bag before it is tied off; Gains fade after the launch [2576] → make the bins part of the fixed theatre layout, not a poster campaign; Total waste rises while the orange share falls [2692] → report total kg per case alongside the percentage
- Pearls: Recruit the scrub lead and the estates waste manager as co-leads; Open all packaging into recycling before the patient enters, so nothing is contaminated; Photograph a typical orange bag at baseline; it persuades faster than numbers; Convert kg to cost and carbon using the contractor's price per tonne for each stream; Name a green champion for each theatre to keep the bins in place after rotation

**ONA-151. On primary knee replacement instrument trays, what proportion of the instruments are used during the operation?** (new)
- Standard: Intercollegiate Green Theatre Checklist v2.0 (November 2024), item 9: "REVIEW AND RATIONALISE: clarify necessary kit for case and specify what should be available to open only if needed: “Just in time”; take the opportunity to review instrument sets and identify any targets for overage reduction" https://www.rcsed.ac.uk/media/zs2nlvpj/green-theatre-checklist.pdf
- Pass: Per instrument: used in at least 10% of cases. Instruments used in under 10% of cases are candidates for removal to a separate 'open if needed' pack.. Target: ≥80% of tray instruments used per case. Sample: 20 consecutive cases.
- Change: Remove instruments used in under 10% of cases from the main tray into a separate 'open if needed' pack, agreed with all knee surgeons and sterile services.
- Template: `new_audits/templates/ONA-151.csv`
- Pitfalls: Surgeons fear a missing instrument mid-case → keep every removed item in a sealed 'open if needed' pack in the theatre; Surgeons use different instruments → count per surgeon and remove only items unused by all; Sterile services cannot change tray lists quickly → involve the sterile services manager from week 1
- Pearls: Have the scrub practitioner tick the tray list during closure, while it is fresh; Show surgeons the list of never-used instruments with their names on it; agreement follows quickly; Count 'open if needed' pack openings in the re-audit to prove safety
