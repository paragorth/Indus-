# Ai4Qi — regulatory status

> **Draft for the owner's review — not legal advice.** Internal document for the owner, and for
> anyone who asks (for example a Trust governance lead). Version 0.1, 28 September 2026. Facts about
> the app were checked against the code on the same date. Review before any change listed in §8.

## 1. Intended purpose statement

Publish this wording (or a shorter form of it) on the site and in the terms. MHRA guidance says an
intended purpose should describe the product's function, its intended population, its intended
user and its use environment.

> **Ai4Qi helps health professionals plan, run and report clinical audits and quality improvement
> projects, which measure how a service performs against a published standard.**
>
> - **What it does.** It offers a library of published audits and national standards, ready-made
>   audit protocols, and a tool that drafts a new protocol from a topic the user types. It gives a
>   data-collection template, counts how many de-identified audit records meet the standard, and
>   produces a data sheet, slides and reminders.
> - **Who it is for.** Doctors, nurses, allied health professionals, pharmacists, students and QI
>   staff in the UK and Ireland, working under their own organisation's audit governance.
> - **Where.** In a web browser, on work or personal devices. Audit records stay on the user's
>   device.
> - **What it is not for.** It does not diagnose, treat, monitor or predict anything for an
>   individual patient. It gives no advice about any patient's care. It must not be used to make
>   decisions about a patient, and its results describe a service, not a person.

## 2. UK medical device regulation (UK MDR 2002, MHRA guidance)

**The test.** Under regulation 2(1) of the Medical Devices Regulations 2002, software is a medical
device if the manufacturer intends it to be used "for human beings for the purpose of" diagnosis,
prevention, monitoring, treatment or alleviation of disease (or of an injury or handicap),
investigation or modification of anatomy or a physiological process, or control of conception.
The MHRA's stand-alone software guidance (v1.10f) applies this through a decision flowchart: first
ask whether the software has a medical purpose, then look at its functions.

**Ai4Qi's functions against that test:**

| Function | Medical purpose for an individual? | Why |
|---|---|---|
| Library of published audits and standards | No | Reference information. The MHRA lists "software that provides reference information to help a Healthcare Professional to use their knowledge to make a clinical decision" among functions that do not give a medical purpose |
| "Build an audit" (AI drafts a protocol) | No | Output is an audit design for a service: a question, a standard, a sample and a data sheet. It names no patient and recommends no treatment |
| Running an audit: records, compliance counts, charts | No | Counts how many records meet a standard. This is aggregation for a service. The MHRA says "general purpose tools for analysing clinical data e.g. statistical analysis" are not medical devices |
| Reminders, account, feedback | No | Administrative |

**Conclusion.** With its current functions and claims, **Ai4Qi is not a medical device**. It has
no medical purpose for an individual patient. It supports the audit of services against
standards.

**Points that keep this answer true:**

- **Free does not mean exempt.** The rules cover software made available "free of charge" too.
- **Claims decide it.** The MHRA says a manufacturer's own view is "not solely determinative", and
  that a disclaimer such as "this product is not a medical device" is not acceptable if medical
  claims are made elsewhere, including in promotional material. So avoid words like "detects",
  "predicts", "diagnoses", "recommends treatment" or "clinical decision support" anywhere,
  including social media and sponsor material.
- **AI.** The MHRA's 2023 blog on large language models says models "directed toward general
  purposes" with no medical claim are unlikely to be devices, but those "developed for, or adapted,
  modified or directed toward specifically medical purposes are likely to qualify". Ai4Qi directs
  the model only to audit design (the build function refuses non-audit requests).

**Product changes that would change the answer** (each would need a fresh assessment, and could
mean UKCA marking and MHRA registration):

- Any output about an individual patient: flagging a patient as "non-compliant, act now",
  suggesting a dose, a test, a referral or a diagnosis.
- Risk scores, early-warning scores or clinical calculators applied to a patient's data.
- Reading an electronic patient record and prompting action on a patient.
- Letting users ask the AI clinical questions about a patient, or marketing Ai4Qi as improving
  decisions about patients rather than about services.

## 3. NHS clinical risk management: DCB0129 and DCB0160

**What they are.** DCB0129 sets clinical risk management requirements for manufacturers of health
IT systems; DCB0160 does the same for the health and care organisations that deploy them. Both are
information standards under section 250 of the Health and Social Care Act 2012. NHS England reports
that amendments to section 250 (in force 7 July 2025 and 5 February 2026) created a duty to comply
and let IT providers be made subject to that duty. It also says that "as they stand, bodies
exercising a health and care function must continue to have regard to the standards". The current
versions are DCB0129 v4.2 and DCB0160 v3.2, both from 2018. NHS England consulted on revising them
in June 2026. Its focus groups asked for "a risk-based, tiered approach" and flagged AI as a gap.

**Applicability.** NHS England's GP guidance says "not all digital solutions are subject to formal
clinical safety assurance" and points to an NHS applicability tool. The standards exist to manage
the risk that a system causes harm to patients through its use in care. Ai4Qi:

- is not used in the direct care of any patient, and holds no patient record;
- does not connect to any clinical system;
- produces audit designs and service-level results that a clinician, supervisor and audit
  department review before anything changes in practice.

**Scope decision.** Ai4Qi is **not treated as a health IT system in scope of DCB0129**, and no
DCB0160 work is needed for staff to use it. The reasoning should be confirmed by a Clinical Safety
Officer (CSO) when one is available.

**Kept anyway, as good practice (owner's task, not the users'):** a short hazard log for the AI
content. Main hazard: a wrong, out-of-date or invented standard leads a department to change
practice. Controls: standards quoted from a fetched source with a link; "Built for you – not yet
run" and "Proposed – not yet run" labels; evidence lines limited to real library entries; the
correction procedure in `CONTENT_AND_AI_POLICY.md`. An NHS organisation that chooses to adopt
Ai4Qi formally may apply DCB0160 or the Digital Technology Assessment Criteria (DTAC) locally.
That is their choice. Ai4Qi does not ask users to obtain it.

## 4. NHS Data Security and Protection Toolkit (DSPT)

The DSPT site says: "All organisations that have access to NHS patient data and systems must use
this toolkit." **Ai4Qi has no access to NHS patient data or systems.** Audit records stay encrypted
on the user's device, and so does the totals-only "results code" a user can paste in. The server
holds only account, profile, feedback, progress and reminder data (reminder text can include record
counts, such as "32 of 40 entered", but no records). **The DSPT is not needed.** It
would become relevant if Ai4Qi stored or synced audit records, received record-level data (for
example for AI analysis), connected to NHS systems, or provided a patient-data service to an NHS
body under contract.

## 5. NICE Evidence Standards Framework (ESF) for digital health technologies

The ESF (ECD7, last updated 9 August 2022) is for evaluating digital health technologies that are
"likely to be commissioned" in the UK health and care system. It is not a legal requirement.
**It does not apply to Ai4Qi today**, because nobody commissions or buys it. If a commissioner ever
assessed it, the closest class is **Tier A, "System service"**. NICE defines this as technologies
"intended to release costs or staff time, or to improve efficiency" and "unlikely to have direct
health outcomes measurable for individual service users". Tier A needs the lightest evidence.

## 6. Ireland (HPRA)

Ireland applies the EU Medical Devices Regulation (EU) 2017/745. The HPRA's guide to stand-alone
software (SUR-G0040, 2020) asks whether the software performs "an action for the benefit of an
individual patient". The EU guidance (MDCG 2019-11 rev.1, June 2025, decision step 4) says software
is not for the benefit of individual patients if it is intended "only to aggregate population
data, provide generic diagnostic or treatment pathways (not directed to individual patients),
scientific literature, medical atlases, models and templates". **Same conclusion as in the UK:
not a medical device.** 

## 7. Summary

| Framework | Applies? | Reason |
|---|---|---|
| UK MDR 2002 / MHRA | No | No medical purpose for an individual patient; audit of services |
| DCB0129 / DCB0160 | Not in scope (CSO to confirm); voluntary AI hazard log kept | Not used in direct care; no clinical system link |
| DSPT | No | No access to NHS patient data or systems |
| NICE ESF | Not applicable; Tier A if ever assessed | Not commissioned |
| EU MDR (HPRA) | No | MDCG 2019-11 step 4 |

## 8. What would trigger a reassessment

- Any feature that produces output about an individual patient (§2 list).
- Storing, syncing or backing up audit records on a server, or team audits.
- Sending records or text drawn from records to an AI model.
- Connecting to an EPR, NHS login or any NHS system.
- A contract with an NHS body, HSE or ICB, or a request to go on a buying framework.
- New marketing claims, or sponsor material that makes clinical claims about Ai4Qi.
- Changes to the law: revised DCB0129/0160 after the 2026 consultation; new MHRA software or AI
  rules.
- Every 12 months in any case.

## Sources checked (28 Sep 2026)

- Medical Devices Regulations 2002, reg. 2(1) (latest revised version): https://www.legislation.gov.uk/uksi/2002/618/regulation/2
- MHRA, Medical device stand-alone software including apps (v1.10f; page updated 1 July 2023): https://www.gov.uk/government/publications/medical-devices-software-applications-apps (PDF: https://assets.publishing.service.gov.uk/media/64a7d22d7a4c230013bba33c/Medical_device_stand-alone_software_including_apps__including_IVDMDs_.pdf)
- MHRA, Crafting an intended purpose in the context of SaMD (22 March 2023): https://www.gov.uk/government/publications/crafting-an-intended-purpose-in-the-context-of-software-as-a-medical-device-samd
- MHRA, Software and AI as a medical device (updated 3 February 2025): https://www.gov.uk/government/publications/software-and-artificial-intelligence-ai-as-a-medical-device/software-and-artificial-intelligence-ai-as-a-medical-device
- MHRA MedRegs blog, Large Language Models and software as a medical device (3 March 2023): https://medregs.blog.gov.uk/2023/03/03/large-language-models-and-software-as-a-medical-device/
- NHS England, National review of DCB0129 and DCB0160: supporting information (29 June 2026): https://www.england.nhs.uk/long-read/national-review-of-clinical-risk-management-standardsdcb0129-and-dcb0160-supporting-information/
- NHS England, Digital clinical safety assurance (v1.2, updated 4 March 2025): https://www.england.nhs.uk/long-read/digital-clinical-safety-assurance/
- NHS AI and Digital Regulations Service, Complying with NHS Digital clinical risk management standards: https://www.digitalregulations.innovation.nhs.uk/regulations-and-guidance-for-developers/all-developers-guidance/complying-with-nhs-digital-clinical-risk-management-standards/
- Data Security and Protection Toolkit home page: https://www.dsptoolkit.nhs.uk/
- NICE ECD7 Evidence standards framework, sections A and B: https://www.nice.org.uk/corporate/ecd7/chapter/section-a-technologies-suitable-for-evaluation-using-the-evidence-standards-framework and https://www.nice.org.uk/corporate/ecd7/chapter/section-b-classification-of-digital-health-technologies
- HPRA, Guide to placing medical device standalone software on the market (SUR-G0040-2, 21 August 2020): https://assets.hpra.ie/data/docs/default-source/external-guidance-document/sur-g0040-guide-to-placing-medical-device-standalone-software-on-the-market-v2.pdf
- MDCG 2019-11 rev.1 (June 2025): https://health.ec.europa.eu/document/download/b45335c5-1679-4c71-a91c-fc7a4d37f12b_en?filename=md_mdcg_2019_11_guidance_qualification_classification_software_en.pdf
- Not reachable in this session: the DCB0129/DCB0160 specification pages and the applicability tool on digital.nhs.uk (blocked by a bot check). Read them before the CSO review.
