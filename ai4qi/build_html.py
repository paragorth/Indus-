"""Write index.html: a single self-contained page (data embedded, opens from file://)
for browsing the library and reviewing the topic judgements."""
import json

import pipeline


def slim(e):
    f = pipeline.flat(e)
    p = e.get("paper") or {}
    f["abstract"] = p.get("abstract", "")
    f["figs"] = [{"file": x.get("file"), "label": x.get("label"), "caption": x.get("caption"),
                  "credit": x.get("credit"), "chart_type": x.get("chart_type")}
                 for x in e.get("figures", []) if x.get("file")]
    f["figure_link"] = e.get("figure_link") or ""
    f["detailed"] = bool(e.get("detail"))
    return f


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Ai4Qi Audit Library</title>
<style>
:root{--bg:#fbfaf7;--panel:#fff;--ink:#1d1f22;--muted:#5f6670;--line:#e3e0d9;--accent:#1f5f8b;--warn:#9a5b00;--warnbg:#fff4e0;--ok:#2d6a3e}
@media (prefers-color-scheme:dark){:root{--bg:#15171a;--panel:#1d2024;--ink:#e8e6e1;--muted:#a3a9b1;--line:#33373d;--accent:#7fb6dd;--warn:#f0b35a;--warnbg:#3a2c14;--ok:#8fcf9f}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 system-ui,-apple-system,Segoe UI,sans-serif}
header{padding:18px 16px 8px;max-width:1200px;margin:auto}h1{font-size:22px;margin:0 0 4px}header p{margin:0;color:var(--muted);font-size:13px}
nav{display:flex;gap:8px;max-width:1200px;margin:10px auto 0;padding:0 16px}nav button{border:1px solid var(--line);background:var(--panel);color:var(--ink);padding:6px 12px;border-radius:6px;cursor:pointer}
nav button.on{background:var(--accent);color:#fff;border-color:var(--accent)}
.filters{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:8px;max-width:1200px;margin:12px auto;padding:0 16px}
.filters select,.filters input{width:100%;padding:6px 8px;border:1px solid var(--line);border-radius:6px;background:var(--panel);color:var(--ink);font:inherit;font-size:13px}
.count{max-width:1200px;margin:0 auto;padding:0 16px;color:var(--muted);font-size:13px}
main{max-width:1200px;margin:8px auto 40px;padding:0 16px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:12px 14px;margin:8px 0}
.card h3{font-size:15px;margin:0 0 4px}.meta{color:var(--muted);font-size:12.5px}.tag{display:inline-block;font-size:11.5px;padding:1px 7px;border-radius:10px;border:1px solid var(--line);margin:2px 4px 2px 0}
.tag.fix{border-color:var(--accent);color:var(--accent)}.draft{background:var(--warnbg);color:var(--warn);border:1px solid var(--warn);font-weight:600}
details{margin-top:6px}summary{cursor:pointer;color:var(--accent);font-size:13px}
dl{display:grid;grid-template-columns:150px 1fr;gap:3px 12px;margin:8px 0;font-size:13.5px}dt{color:var(--muted)}dd{margin:0;overflow-wrap:anywhere}
.figs{display:flex;flex-wrap:wrap;gap:10px}.figs figure{margin:0;width:220px;font-size:11.5px;color:var(--muted)}.figs img{width:100%;border:1px solid var(--line);border-radius:4px;background:#fff}
a{color:var(--accent)}.tk h3{font-size:17px}.tk dl{grid-template-columns:170px 1fr}
@media(max-width:600px){dl,.tk dl{grid-template-columns:1fr}dt{margin-top:6px}}
</style></head><body>
<header><h1>Ai4Qi Audit Library</h1><p id="about"></p></header>
<nav><button id="tabA" class="on">Audits</button><button id="tabT">Topic judgements</button></nav>
<div class="filters">
 <select id="fLib"><option value="">All libraries</option></select>
 <select id="fSpec"><option value="">All specialties</option></select>
 <select id="fTopic"><option value="">All topics</option></select>
 <select id="fFix"><option value="">All fix types</option></select>
 <select id="fStatus"><option value="">All statuses</option></select>
 <select id="fGeo"><option value="">Any country</option><option value="ukie">UK and Ireland only</option></select>
 <select id="fDet"><option value="">Detailed or not</option><option value="1">Detailed only</option><option value="fig">With figures</option></select>
 <input id="fText" placeholder="Search title, standard, finding">
</div>
<div class="count" id="count"></div>
<main id="out"></main>
<script type="application/json" id="data">__DATA__</script>
<script>
const D=JSON.parse(document.getElementById('data').textContent);
document.getElementById('about').textContent=D.about+' Built '+D.built+'.';
const A=D.audits,T=D.topics,$=id=>document.getElementById(id);
const esc=s=>String(s??'').replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
function fill(sel,vals){[...new Set(vals.filter(v=>v!==''&&v!=null))].sort().forEach(v=>{const o=document.createElement('option');o.value=o.textContent=v;sel.appendChild(o)})}
fill($('fLib'),A.map(a=>a.library));fill($('fSpec'),A.map(a=>a.specialty));fill($('fTopic'),A.map(a=>a.topic));
fill($('fFix'),A.map(a=>a.fix_type));fill($('fStatus'),A.map(a=>a.status));
let tab='A';
function match(a){const f=id=>$(id).value;
 if(f('fLib')&&a.library!==f('fLib'))return false;if(f('fSpec')&&a.specialty!==f('fSpec'))return false;
 if(f('fTopic')&&a.topic!==f('fTopic'))return false;if(f('fFix')&&a.fix_type!==f('fFix'))return false;
 if(f('fStatus')&&a.status!==f('fStatus'))return false;
 if(f('fGeo')==='ukie'&&!(a.uk_ireland===true||/^(UK|Ireland)$/.test(a.country)))return false;
 if(f('fDet')==='1'&&!a.detailed)return false;if(f('fDet')==='fig'&&!a.figs.length)return false;
 const q=f('fText').toLowerCase();if(q&&!(a.title+' '+a.standard+' '+a.finding+' '+a.change+' '+a.topic).toLowerCase().includes(q))return false;return true}
function row(k,v){return v===''||v==null?'':`<dt>${k}</dt><dd>${esc(v)}</dd>`}
function audit(a){const src=a.source?`<a href="${esc(a.source)}" target="_blank" rel="noopener">source</a>`:'no source link';
 const figs=a.figs.length?`<div class="figs">${a.figs.map(f=>`<figure><a href="${esc(f.file)}" target="_blank"><img loading="lazy" src="${esc(f.file)}" alt="${esc(f.label)}"></a><figcaption><b>${esc(f.label)}</b> (${esc(f.chart_type)}) ${esc(f.caption).slice(0,200)}<br><i>${esc(f.credit)}</i></figcaption></figure>`).join('')}</div>`:(a.figure_link?`<p class="meta">Figures: link only (licence not CC BY/CC0) — <a href="${esc(a.figure_link)}" target="_blank">view</a></p>`:'');
 return `<div class="card"><h3>#${a.id} ${esc(a.title)}</h3><div class="meta">${esc(a.specialty)} · ${esc(a.status)} · ${src}${a.year?' · '+esc(a.year):''}${a.country?' · '+esc(a.country):''}</div>
 <div>${a.topic?`<span class="tag">${esc(a.topic)}</span>`:''}${a.fix_type?`<span class="tag fix">${esc(a.fix_type)}</span>`:''}${a.loop_closed!==''?`<span class="tag">loop closed: ${esc(a.loop_closed)}</span>`:''}</div>
 <dl>${row('Standard',a.standard)}${row('Finding',a.finding)}${row('Change',a.change)}${row('Next audit',a.next_audit)}</dl>
 ${a.detailed?`<details><summary>Cycles</summary><dl>${row('Setting',a.setting)}${row('Cycle 1',[a.cycle1_period,a.cycle1_n!==''&&a.cycle1_n!=null?'n='+a.cycle1_n:'',a.cycle1_result].filter(Boolean).join(' · '))}${row('Intervention',a.intervention)}${row('Cycle 2',[a.cycle2_period,a.cycle2_n!==''&&a.cycle2_n!=null?'n='+a.cycle2_n:'',a.cycle2_result].filter(Boolean).join(' · '))}${row('Other findings',a.other_findings)}${row('Citation',a.citation)}</dl></details>`:(a.citation?`<p class="meta">${esc(a.citation)}</p>`:'')}
 ${a.abstract?`<details><summary>Abstract</summary><p>${esc(a.abstract)}</p></details>`:''}${figs}</div>`}
function topic(t){const ids=new Set(t.evidence_ids||[]);const ev=A.filter(a=>ids.has(a.id)||(!t.evidence_ids&&a.topic===t.topic&&a.detailed));
 return `<div class="card tk"><h3>${esc(t.topic)}</h3><div>${t.status?`<span class="tag draft">${esc(t.status)}</span>`:'<span class="tag">Seed card (from your library)</span>'}${t.revises_seed_card?'<span class="tag">revision of seed card</span>':''}</div>
 <p class="meta">${esc(t.audits_in_library)} detailed audits · ${esc((t.countries||[]).join(', '))} · ${esc(t.years_seen)}</p>
 <dl>${row('Usual baseline',t.usual_baseline)}${row('Fix that works',t.fix_that_works)}${row('Fix that fails',t.fix_that_fails)}${row('Consultant advice',t.consultant_advice)}</dl>
 <details><summary>Audits behind this card (${ev.length})</summary>${ev.map(audit).join('')}</details></div>`}
function render(){const f=A.filter(match);
 if(tab==='A'){$('count').textContent=`${f.length} of ${A.length} audits`;$('out').innerHTML=f.slice(0,400).map(audit).join('')+(f.length>400?`<p class="meta">Showing first 400; narrow the filters to see the rest.</p>`:'')}
 else{const tp=$('fTopic').value,sp=$('fSpec').value,tt=T.filter(t=>(!tp||t.topic===tp)&&(!sp||A.some(a=>a.topic===t.topic&&a.specialty===sp)));
  $('count').textContent=`${tt.length} topic cards (${T.filter(t=>t.status).length} drafts need consultant sign-off)`;$('out').innerHTML=tt.map(topic).join('')}}
document.querySelectorAll('.filters select,.filters input').forEach(e=>e.addEventListener('input',render));
$('tabA').onclick=()=>{tab='A';$('tabA').className='on';$('tabT').className='';render()};
$('tabT').onclick=()=>{tab='T';$('tabT').className='on';$('tabA').className='';render()};
render();
</script></body></html>"""


def build(lib, path):
    import time
    data = {"about": lib.get("about", ""), "built": time.strftime("%Y-%m-%d"),
            "audits": [slim(e) for e in lib["audits"]], "topics": lib.get("topic_knowledge", [])}
    blob = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    with open(path, "w", encoding="utf-8") as f:
        f.write(PAGE.replace("__DATA__", blob))
