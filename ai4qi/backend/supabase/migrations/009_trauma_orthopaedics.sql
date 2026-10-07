-- Ai4Qi: the specialty is "Trauma and orthopaedics" (the UK name), not "Orthopaedics".
-- The old value stays accepted for existing rows, which are renamed here.
create or replace function public.ai4qi_specialties() returns text[] language sql immutable as $$
  select array[
    'Acute and general medicine', 'Anaesthesia', 'Cardiology', 'Cardiothoracic surgery',
    'Care of the elderly', 'Clinical governance and quality improvement', 'Dentistry and oral surgery',
    'Dermatology', 'Emergency medicine', 'Endocrinology and diabetes', 'ENT', 'Gastroenterology',
    'General practice', 'General surgery', 'Haematology', 'Infectious diseases and microbiology',
    'Intensive care', 'Medical education', 'Neonatology', 'Neurology', 'Neurosurgery',
    'Nursing and midwifery', 'Obstetrics and gynaecology', 'Oncology', 'Ophthalmology', 'Trauma and orthopaedics',
    'Orthopaedics', 'Paediatric surgery', 'Paediatrics', 'Palliative care', 'Pathology', 'Pharmacy', 'Plastic surgery',
    'Psychiatry', 'Radiology', 'Renal medicine', 'Respiratory', 'Rheumatology', 'Sexual health', 'Stroke',
    'Therapies and allied health', 'Urology', 'Vascular surgery', 'Other'
  ]::text[] $$;
update public.profiles set specialty = 'Trauma and orthopaedics' where specialty = 'Orthopaedics';
