/* Ai4Qi clinical audit library: single-page app (no build step). */
(function () {
  'use strict';

  var S = {
    lib: [], byId: new Map(), shard: 100, shards: new Map(),
    proposed: [], pById: new Map(), cards: [], cardsByTopic: new Map(),
    standards: [], citedBy: new Map(), topicCounts: new Map(),
    pubDocs: [], propDocs: [], vocab: [], groups: [],
    view: { key: '', pubShown: 25, propShown: 6 }
  };
  var main = document.getElementById('main');

  /* ---------- helpers ---------- */
  function esc(v) {
    return String(v == null ? '' : v).replace(/[&<>"']/g, function (c) {
      return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c];
    });
  }
  function attr(v) { return esc(v); }
  function has(v) { return v != null && v !== '' && v !== 'not reported' && !(Array.isArray(v) && !v.length); }
  function show(v) { return v == null || v === '' ? 'Not reported' : (v === 'not reported' ? 'Not reported' : v); }
  function cap(s) { s = String(s || ''); return s.charAt(0).toUpperCase() + s.slice(1); }
  function trunc(s, n) { s = String(s || ''); return s.length > n ? s.slice(0, n).replace(/\s+\S*$/, '') + '…' : s; }
  function fmt(n) { return Number(n).toLocaleString('en-GB'); }
  function topicHref(t) { return '#/topic/' + encodeURIComponent(t); }
  function safeUrl(u) { return /^https?:\/\//i.test(String(u || '')) ? u : ''; }

  // Escape text, then link library ids like [1234] or [1234, 1240] and proposed ids like ONA-012.
  function linkify(text) {
    var out = esc(text);
    out = out.replace(/\[(\d{1,5}(?:\s*[,;]\s*\d{1,5})*)\]/g, function (m, inner) {
      var parts = inner.split(/\s*[,;]\s*/).map(function (n) {
        return S.byId.has(+n) ? '<a href="#/audit/' + n + '">' + n + '</a>' : n;
      });
      return '[' + parts.join(', ') + ']';
    });
    out = out.replace(/\b((?:ONA|NNA)-\d{3})\b/g, function (m) {
      return S.pById.has(m) ? '<a href="#/proposed/' + m + '">' + m + '</a>' : m;
    });
    return out;
  }
  function idsIn(text) {
    var ids = [];
    String(text || '').replace(/\[(\d{1,5}(?:\s*[,;]\s*\d{1,5})*)\]/g, function (m, inner) {
      inner.split(/\s*[,;]\s*/).forEach(function (n) { ids.push(+n); });
    });
    return ids;
  }
  function badge(text, kind) { return '<span class="badge' + (kind ? ' badge-' + kind : '') + '">' + esc(text) + '</span>'; }
  function loopBadge(a) { return a.lc ? badge('Closed loop', 'ok') : ''; }
  function ukBadge(a) { return a.uk ? badge('UK & Ireland', 'info') : ''; }
  var PROPOSED_BADGE = badge('Proposed – not yet run', 'warn');
  var DRAFT_BADGE = badge('Draft – needs consultant sign-off', 'warn');

  function getJSON(url) {
    return fetch(url).then(function (r) {
      if (!r.ok) throw new Error(url + ' ' + r.status);
      return r.json();
    });
  }

  /* ---------- search ---------- */
  var STOP = new Set(('a an and are as at be by can do does for from has have how i in into is it its me my of on or our ' +
    'should that the their them there these this to was we were what when where which who why will with you your want ' +
    'wanting would like audit audits auditing auditable measure measuring look looking proportion').split(' '));

  function stem(w) {
    w = w.replace(/iz(e|ed|es|ing|ation|ations)$/, 'is$1');
    if (w.length > 4 && /ies$/.test(w)) return w.slice(0, -3) + 'y';
    if (w.length > 4 && /(ss|x|ch|sh)es$/.test(w)) return w.slice(0, -2);
    if (w.length > 3 && /s$/.test(w) && !/(ss|us|is)$/.test(w)) return w.slice(0, -1);
    return w;
  }
  function tokens(str) {
    return String(str || '').toLowerCase().normalize('NFKD').replace(/[̀-ͯ]/g, '')
      .split(/[^a-z0-9]+/).filter(function (t) { return t.length > 1 && !STOP.has(t); }).map(stem);
  }
  function indexDoc(fields) {
    var m = new Map();
    fields.forEach(function (f) {
      tokens(f[0]).forEach(function (t) { if ((m.get(t) || 0) < f[1]) m.set(t, f[1]); });
    });
    return m;
  }
  function expand(q) {
    var ex = [[q, 1]];
    if (q.length >= 4) {
      for (var i = 0; i < S.vocab.length; i++) {
        var k = S.vocab[i];
        if (k === q) continue;
        if (k.length > q.length && k.indexOf(q) === 0) ex.push([k, 0.8]);
        else if (k.length >= 5 && q.length > k.length && q.indexOf(k) === 0) ex.push([k, 0.6]);
      }
    }
    return ex;
  }
  function rank(docs, qtoks, phrase) {
    var exps = qtoks.map(expand), res = [], maxCov = 0;
    docs.forEach(function (d) {
      var cov = 0, sc = 0;
      exps.forEach(function (ex) {
        var best = 0;
        for (var i = 0; i < ex.length; i++) {
          var w = d.m.get(ex[i][0]);
          if (w && w * ex[i][1] > best) best = w * ex[i][1];
        }
        if (best) { cov++; sc += best; }
      });
      if (!cov) return;
      if (phrase && d.title.indexOf(phrase) !== -1) sc += 8;
      sc *= d.boost || 1;
      if (cov > maxCov) maxCov = cov;
      res.push({ d: d, cov: cov, sc: sc });
    });
    var need = qtoks.length >= 3 ? Math.max(1, maxCov - 1) : maxCov;
    return res.filter(function (r) { return r.cov >= need; });
  }

  function buildIndexes() {
    var vocab = new Set();
    S.pubDocs = S.lib.map(function (a) {
      var d = {
        a: a, title: (a.t || '').toLowerCase(),
        boost: a.ss === 'recurring topic, not individually linked' ? 0.6 : 1,
        m: indexDoc([[a.t, 6], [a.tp, 5], [a.s, 2], [a.st, 3], [a.f, 2], [a.c, 2], [a.r1, 1.5], [a.iv, 1.5], [a.r2, 1.5]])
      };
      d.m.forEach(function (v, k) { vocab.add(k); });
      return d;
    });
    S.propDocs = S.proposed.map(function (p) {
      var st = p.standard || {};
      var d = {
        p: p, title: (p.question || '').toLowerCase(),
        m: indexDoc([[p.question, 6], [p.area, 3], [st.source, 3], [st.wording, 2], [p.why, 2], [p.change, 2],
          [p.population, 1.5], [p.pass, 1.5]])
      };
      d.m.forEach(function (v, k) { vocab.add(k); });
      return d;
    });
    S.vocab = Array.from(vocab);
  }

  /* ---------- data loading ---------- */
  function load() {
    return Promise.all([
      getJSON('data/library.json'), getJSON('data/proposed.json'),
      getJSON('data/cards.json'), getJSON('data/standards.json')
    ]).then(function (r) {
      S.shard = r[0].shard || 100;
      S.lib = r[0].audits;
      S.lib.forEach(function (a) {
        S.byId.set(a.id, a);
        if (a.tp) S.topicCounts.set(a.tp, (S.topicCounts.get(a.tp) || 0) + 1);
      });
      S.proposed = r[1];
      S.proposed.forEach(function (p) {
        S.pById.set(p.id, p);
        (p.evidence || []).forEach(function (e) {
          idsIn(e).forEach(function (id) {
            if (!S.citedBy.has(id)) S.citedBy.set(id, []);
            if (S.citedBy.get(id).indexOf(p.id) === -1) S.citedBy.get(id).push(p.id);
          });
        });
      });
      S.cards = r[2];
      S.cards.forEach(function (c) {
        if (!S.cardsByTopic.has(c.topic)) S.cardsByTopic.set(c.topic, []);
        S.cardsByTopic.get(c.topic).push(c);
      });
      S.standards = r[3];
      var g = new Map();
      S.lib.forEach(function (a) { g.set(a.g, (g.get(a.g) || 0) + 1); });
      S.groups = Array.from(g.entries()).sort(function (x, y) { return x[0].localeCompare(y[0]); });
      buildIndexes();
    });
  }
  function loadAudit(id) {
    var key = Math.floor(id / S.shard);
    if (!S.shards.has(key)) {
      S.shards.set(key, getJSON('data/audits/' + key + '.json').then(function (rows) {
        var m = new Map(); rows.forEach(function (r) { m.set(r.id, r); }); return m;
      }));
    }
    return S.shards.get(key).then(function (m) { return m.get(id); });
  }

  /* ---------- routing ---------- */
  function parseHash() {
    var h = location.hash.replace(/^#\/?/, '');
    var qi = h.indexOf('?');
    var path = qi === -1 ? h : h.slice(0, qi);
    var params = new URLSearchParams(qi === -1 ? '' : h.slice(qi + 1));
    var parts = path.split('/').map(function (p) { try { return decodeURIComponent(p); } catch (e) { return p; } });
    return { parts: parts, params: params };
  }
  function setNav(name) {
    document.querySelectorAll('.site-nav a').forEach(function (a) {
      if (a.getAttribute('data-nav') === name) a.setAttribute('aria-current', 'page');
      else a.removeAttribute('aria-current');
    });
  }
  function page(html, title, nav, keepScroll) {
    main.innerHTML = html;
    document.title = title ? title + ' · Ai4Qi' : 'Ai4Qi Clinical Audit Library';
    setNav(nav || '');
    if (!keepScroll) window.scrollTo(0, 0);
  }
  function focusMain() {
    var h = main.querySelector('h1');
    if (h) { h.setAttribute('tabindex', '-1'); h.focus({ preventScroll: true }); }
  }

  function route() {
    var r = parseHash(), p = r.parts, name = p[0] || '';
    var sameView = false;
    try {
      if (name === '') renderHome();
      else if (name === 'search') renderSearch(r.params, sameView);
      else if (name === 'proposed' && p[1]) renderProposed(p[1]);
      else if (name === 'proposed') renderProposedList(r.params);
      else if (name === 'audit' && p[1]) renderAudit(+p[1]);
      else if (name === 'topic' && p[1]) renderTopic(p[1]);
      else if (name === 'topics') renderTopics();
      else if (name === 'standards') renderStandards(r.params);
      else renderNotFound();
    } catch (err) {
      renderNotFound();
      if (window.console) console.warn(err);
    }
    if (!sameView) focusMain();
  }

  function searchForm(q, big) {
    return '<form class="search-form" role="search" data-search>' +
      '<label for="q"' + (big ? '' : ' class="visually-hidden"') + '>What do you want to audit?</label>' +
      '<input id="q" name="q" type="search" autocomplete="off" value="' + attr(q || '') + '" placeholder="e.g. urinary catheter documentation, VTE assessment, consent">' +
      '<button class="btn" type="submit"><svg width="18" height="18" viewBox="0 0 24 24" aria-hidden="true"><circle cx="11" cy="11" r="7" fill="none" stroke="currentColor" stroke-width="2.4"/><path d="M16.5 16.5L21 21" stroke="currentColor" stroke-width="2.4" stroke-linecap="round"/></svg>Search</button>' +
      '</form>';
  }

  /* ---------- home ---------- */
  function renderHome() {
    var closed = S.lib.filter(function (a) { return a.lc; }).length;
    var uk = S.lib.filter(function (a) { return a.uk; }).length;
    var topGroups = S.groups.slice().sort(function (a, b) { return b[1] - a[1]; }).slice(0, 18)
      .sort(function (a, b) { return a[0].localeCompare(b[0]); });
    var areas = new Map();
    S.proposed.forEach(function (p) { areas.set(p.area, (areas.get(p.area) || 0) + 1); });
    var areaList = Array.from(areas.entries()).sort(function (a, b) { return a[0].localeCompare(b[0]); });
    var examples = ['urinary catheter', 'VTE risk assessment', 'operation note', 'consent', 'delirium screening', 'antibiotic prophylaxis'];

    page(
      '<div class="hero"><h1>Find, plan and run a clinical audit</h1>' +
      '<p class="lede">Search published audits and closed-loop quality improvement projects, or pick a ready-to-run protocol with the standard, template and timeline already worked out.</p>' +
      searchForm('', true) +
      '<p class="examples"><span>Try:</span>' + examples.map(function (e) { return '<a href="#/search?q=' + encodeURIComponent(e) + '">' + esc(e) + '</a>'; }).join('') + '</p></div>' +
      '<ul class="stats" aria-label="Library at a glance">' +
      stat(S.lib.length, 'audits in the library') + stat(closed, 'closed the loop') + stat(uk, 'from the UK and Ireland') +
      stat(S.proposed.length, 'proposed audits ready to run') + stat(S.standards.length, 'standards quoted') + stat(S.cards.length, 'topic cards') +
      '</ul>' +
      '<div class="section-head"><h2>Browse by specialty</h2><a href="#/search">All published audits</a></div>' +
      '<ul class="chip-grid">' + topGroups.map(function (g) {
        return '<li><a class="chip-link" href="#/search?sp=' + encodeURIComponent(g[0]) + '"><span>' + esc(g[0]) + '</span><span class="count">' + fmt(g[1]) + '</span></a></li>';
      }).join('') + '</ul>' +
      '<div class="section-head"><h2>Proposed audits by area</h2><a href="#/proposed">All proposed audits</a></div>' +
      '<ul class="chip-grid">' + areaList.map(function (g) {
        return '<li><a class="chip-link" href="#/proposed?area=' + encodeURIComponent(g[0]) + '"><span>' + esc(g[0]) + '</span><span class="count">' + fmt(g[1]) + '</span></a></li>';
      }).join('') + '</ul>',
      '', '');
  }
  function stat(n, label) { return '<li class="stat"><b>' + fmt(n) + '</b><span>' + esc(label) + '</span></li>'; }

  /* ---------- search results ---------- */
  function searchState(params) {
    return {
      q: (params.get('q') || '').trim(), sp: params.get('sp') || '',
      closed: params.get('closed') === '1', detailed: params.get('detailed') === '1'
    };
  }
  function searchHash(st) {
    var p = new URLSearchParams();
    if (st.q) p.set('q', st.q);
    if (st.sp) p.set('sp', st.sp);
    if (st.closed) p.set('closed', '1');
    if (st.detailed) p.set('detailed', '1');
    var s = p.toString();
    return '#/search' + (s ? '?' + s : '');
  }
  function runSearch(st) {
    var qt = tokens(st.q), phrase = qt.length > 1 ? st.q.toLowerCase() : '';
    var pubs, props;
    var pubPool = S.pubDocs.filter(function (d) {
      var a = d.a;
      return (!st.sp || a.g === st.sp) && (!st.closed || a.lc) && (!st.detailed || a.dt);
    });
    if (qt.length) {
      pubs = rank(pubPool, qt, phrase).sort(function (x, y) {
        return (y.d.a.uk - x.d.a.uk) || (y.cov - x.cov) || (y.sc - x.sc) || ((y.d.a.lc || 0) - (x.d.a.lc || 0)) || ((+y.d.a.y || 0) - (+x.d.a.y || 0));
      }).map(function (r) { return r.d.a; });
      props = rank(S.propDocs, qt, phrase).sort(function (x, y) { return (y.cov - x.cov) || (y.sc - x.sc); })
        .map(function (r) { return r.d.p; });
    } else {
      pubs = pubPool.map(function (d) { return d.a; }).sort(function (x, y) {
        return ((y.uk || 0) - (x.uk || 0)) || ((y.lc || 0) - (x.lc || 0)) || ((y.dt || 0) - (x.dt || 0)) || ((+y.y || 0) - (+x.y || 0)) || (x.id - y.id);
      });
      if (st.sp) {
        var spt = tokens(st.sp);
        props = st.sp === 'Orthopaedics' ? S.proposed.filter(function (p) { return p.group === 'Orthopaedics'; })
          : rank(S.propDocs, spt, '').sort(function (x, y) { return y.sc - x.sc; }).map(function (r) { return r.d.p; });
      } else props = [];
    }
    return { pubs: pubs, props: props };
  }
  function renderSearch(params, sameView) {
    var st = searchState(params);
    var key = searchHash(st);
    if (S.view.key !== key) { S.view = { key: key, pubShown: 25, propShown: 6 }; }
    var res = runSearch(st);
    var heading = st.q ? 'Results for “' + st.q + '”' : (st.sp ? st.sp + ' audits' : 'Search the library');

    var filters = '<aside class="filters" aria-label="Filters"><h2>Filter published audits</h2>' +
      '<label for="f-sp">Specialty</label><select id="f-sp" data-filter="sp"><option value="">All specialties</option>' +
      S.groups.map(function (g) { return '<option value="' + attr(g[0]) + '"' + (g[0] === st.sp ? ' selected' : '') + '>' + esc(g[0]) + ' (' + fmt(g[1]) + ')</option>'; }).join('') +
      '</select>' +
      '<label class="check"><input type="checkbox" data-filter="closed"' + (st.closed ? ' checked' : '') + '><span>Closed loop only</span></label>' +
      '<label class="check"><input type="checkbox" data-filter="detailed"' + (st.detailed ? ' checked' : '') + '><span>Detailed records only</span></label>' +
      '<p class="hint">UK and Ireland audits are listed first.</p></aside>';

    var propHtml = '';
    if (st.q || st.sp) {
      var pv = res.props.slice(0, S.view.propShown);
      propHtml = '<div class="results-title"><h2 id="prop-h">Proposed audits (ready to run)</h2><span class="count">' + fmt(res.props.length) + ' found</span></div>' +
        (pv.length ? '<ul class="result-list" aria-labelledby="prop-h">' + pv.map(propResult).join('') + '</ul>' :
          '<p class="empty">No proposed audit matches this search. Try a broader term, or browse <a href="#/proposed">all proposed audits</a>.</p>') +
        (res.props.length > pv.length ? '<div class="more"><button class="btn btn-secondary" type="button" data-more="prop">Show more proposed audits (' + fmt(res.props.length - pv.length) + ' more)</button></div>' : '');
    }
    var pv2 = res.pubs.slice(0, S.view.pubShown);
    var pubHtml = '<div class="results-title"><h2 id="pub-h">Published audits</h2><span class="count">' + fmt(res.pubs.length) + ' found</span></div>' +
      (pv2.length ? '<ul class="result-list" aria-labelledby="pub-h">' + pv2.map(pubResult).join('') + '</ul>' :
        '<p class="empty">No published audit matches this search and these filters. Try fewer words or clear a filter.</p>') +
      (res.pubs.length > pv2.length ? '<div class="more"><button class="btn btn-secondary" type="button" data-more="pub">Show more published audits (' + fmt(res.pubs.length - pv2.length) + ' more)</button></div>' : '');

    var html = '<h1>' + esc(heading) + '</h1>' + searchForm(st.q, false) +
      '<div class="results-layout" style="margin-top:18px">' + filters + '<div class="results" aria-live="polite">' + propHtml + pubHtml + '</div></div>';
    var y = window.scrollY;
    page(html, heading, 'search', sameView);
    if (sameView) window.scrollTo(0, y);
  }
  function propResult(p) {
    var st = p.standard || {};
    return '<li class="result result-proposed"><div class="badges">' + PROPOSED_BADGE + badge(p.area, 'primary') +
      (p.novelty ? badge(cap(p.novelty)) : '') + '</div>' +
      '<h3><a href="#/proposed/' + attr(p.id) + '">' + esc(p.question) + '</a></h3>' +
      '<p class="meta"><span class="id-tag">' + esc(p.id) + '</span> · Standard: ' + esc(trunc(st.source, 110)) + '</p>' +
      '<p class="snippet">' + esc(trunc(p.why, 230)) + '</p></li>';
  }
  function pubResult(a) {
    var meta = [a.s, a.co, a.y].filter(Boolean).map(esc).join(' · ');
    var snip = a.r2 ? (trunc(a.f, 120) + ' → ' + trunc(a.r2, 120)) : trunc(a.f, 230);
    return '<li class="result"><h3><a href="#/audit/' + a.id + '">' + esc(a.t) + '</a></h3>' +
      '<p class="meta">' + meta + ' <span class="id-tag">[' + a.id + ']</span></p>' +
      '<p class="snippet">' + esc(snip) + '</p>' +
      '<div class="badges">' + ukBadge(a) + loopBadge(a) + (a.fx ? badge('Fix: ' + a.fx) : '') + (a.dt ? '' : badge(cap(a.ss))) + '</div></li>';
  }

  /* ---------- proposed audit ---------- */
  function parseTimeline(t) {
    var segs = String(t || '').split(/;\s*/).map(function (s) {
      var m = s.trim().match(/^(?:wk|wks|week|weeks)\s*(\d+)(?:\s*[–\-]\s*(\d+))?\s*[:,]?\s*(.+)$/i);
      return m ? { a: +m[1], b: +(m[2] || m[1]), label: cap(m[3].trim().toLowerCase()).replace(/\s*\/\s*/g, ' / ') } : null;
    });
    if (!segs.length || segs.some(function (s) { return !s; })) return null;
    return segs;
  }
  function timelineHtml(t) {
    var segs = parseTimeline(t);
    if (!segs) return '<p>' + esc(t) + '</p>';
    var weeks = Math.max.apply(null, segs.map(function (s) { return s.b; }));
    var cols = 'grid-template-columns:repeat(' + weeks + ',minmax(0,1fr))';
    var scale = '';
    for (var i = 1; i <= weeks; i++) scale += '<span>' + i + '</span>';
    return '<div class="timeline" role="img" aria-label="' + attr('Timeline: ' + t) + '">' +
      '<div class="tl-scale" aria-hidden="true"><span class="tl-label muted" style="text-transform:none;font-weight:400">Week</span><div class="tl-track" style="' + cols + '">' + scale + '</div></div>' +
      segs.map(function (s, i) {
        var weeksTxt = s.a === s.b ? 'Wk ' + s.a : 'Wk ' + s.a + '–' + s.b;
        return '<div class="tl-row" aria-hidden="true"><span class="tl-label">' + esc(s.label) + '</span>' +
          '<div class="tl-track" style="' + cols + ';--weeks:' + weeks + '"><div class="tl-bar' + (/embed/i.test(s.label) ? ' alt' : '') +
          '" style="grid-column:' + s.a + ' / ' + (s.b + 1) + '">' + esc(weeksTxt) + '</div></div></div>';
      }).join('') + '</div>';
  }
  function templateTable(p) {
    var rows = (p.template || []).map(function (f) {
      var opts = (f.options || []).join(' / ');
      return '<tr><td data-label="Field"><code>' + esc(f.field).replace(/_/g, '_<wbr>') + '</code></td><td data-label="Type">' + esc(f.type) + '</td>' +
        '<td class="opts' + (opts ? '' : ' empty-cell') + '" data-label="Options">' + esc(opts) + '</td>' +
        '<td data-label="Note"' + (f.note ? '' : ' class="empty-cell"') + '>' + esc(f.note || '') + '</td></tr>';
    }).join('');
    return '<div class="table-wrap"><table class="tpl"><caption class="visually-hidden">Data collection template for ' + esc(p.id) + '</caption>' +
      '<thead><tr><th scope="col">Field</th><th scope="col">Type</th><th scope="col">Options</th><th scope="col">Note</th></tr></thead><tbody>' + rows + '</tbody></table></div>';
  }
  function targetHtml(t) {
    var m = String(t || '').match(/^\s*([≥≤<>]?\s*\d+(?:\.\d+)?\s*%)\s*(.*)$/);
    if (!m) return '<p class="prose"><strong>Target:</strong> ' + linkify(t) + '</p>';
    return '<p class="target-line"><span class="target">' + esc(m[1]) + '</span>' + (m[2] ? '<span>' + linkify(m[2]) + '</span>' : '') + '</p>';
  }
  function pitfallItem(t) {
    var i = t.indexOf('→');
    if (i === -1) return '<li>' + linkify(t) + '</li>';
    return '<li><span class="risk">' + linkify(t.slice(0, i).trim()) + '</span><span class="arrow" aria-label="prevent by">→</span>' + linkify(t.slice(i + 1).trim()) + '</li>';
  }
  function sec(n, title, body) {
    return '<section><h2><span class="num" aria-hidden="true">' + n + '</span>' + esc(title) + '</h2>' + body + '</section>';
  }
  function sub(title, body) { return '<div class="sub"><h3>' + esc(title) + '</h3>' + body + '</div>'; }

  function renderProposed(id) {
    var p = S.pById.get(id);
    if (!p) return renderNotFound();
    var st = p.standard || {}, url = safeUrl(st.url), n = 0;
    var dl = '<a class="btn" href="' + attr(p.template_file) + '" download="' + attr(p.id + '.csv') + '">' +
      '<svg width="18" height="18" viewBox="0 0 24 24" aria-hidden="true"><path d="M12 4v11m0 0l-4.5-4.5M12 15l4.5-4.5M5 19h14" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/></svg>Download template (CSV)</a>';

    var how =
      sub('Standard', '<blockquote class="standard"><p>“' + esc(st.wording) + '”</p></blockquote>' +
        '<p class="standard-source">' + esc(st.source) + (url ? ' · <a class="print-url" href="' + attr(url) + '" target="_blank" rel="noopener">Read the standard</a>' : '') + '</p>') +
      sub('Pass', '<p class="prose">' + linkify(p.pass) + '</p>') +
      sub('Population', '<p class="prose">' + linkify(p.population) + '</p>') +
      sub('Sample', '<p class="prose">' + linkify(p.sample) + '</p>') +
      sub('Data source', '<p class="prose">' + linkify(p.data_source) + '</p>') +
      sub('Template', templateTable(p) + '<div class="no-print">' + dl + '</div>') +
      sub('Timeline', timelineHtml(p.timeline));

    var body =
      sec(++n, 'Why', '<p class="prose">' + linkify(p.why) + '</p>') +
      sec(++n, 'How', how) +
      sec(++n, 'Change', '<p class="prose">' + linkify(p.change) + '</p>') +
      sec(++n, 'Re-audit and target', targetHtml(p.target) + '<p class="prose">' + linkify(p.reaudit) + '</p>') +
      sec(++n, 'Close the loop', '<p class="prose">' + linkify(p.close_loop) + '</p>') +
      (p.evidence && p.evidence.length ? sec(++n, 'Evidence', '<ul class="ev-list">' + p.evidence.map(function (e) { return '<li>' + linkify(e) + '</li>'; }).join('') + '</ul>') : '') +
      (p.pitfalls && p.pitfalls.length ? sec(++n, 'Pitfalls', '<ul class="pit-list">' + p.pitfalls.map(pitfallItem).join('') + '</ul>') : '') +
      (p.pearls && p.pearls.length ? sec(++n, 'Pearls', '<ul class="pearl-list">' + p.pearls.map(function (e) { return '<li>' + linkify(e) + '</li>'; }).join('') + '</ul>') : '') +
      sec(++n, 'Status and effort', '<div class="status-box">' + PROPOSED_BADGE +
        '<span><strong>Data collection effort:</strong> ' + esc(p.effort) + '</span>' +
        (p.novelty ? '<span><strong>Gap:</strong> ' + esc(cap(p.novelty)) + '</span>' : '') + '</div>');

    page('<nav class="breadcrumb" aria-label="Breadcrumb"><a href="#/">Home</a> › <a href="#/proposed">Proposed audits</a> › <a href="#/proposed?area=' + encodeURIComponent(p.area) + '">' + esc(p.area) + '</a></nav>' +
      '<article class="doc"><header class="doc-head"><div class="eyebrow"><span class="id-tag">' + esc(p.id) + '</span>' + PROPOSED_BADGE + badge(p.area, 'primary') + '</div>' +
      '<h1>' + esc(p.question) + '</h1>' +
      '<div class="doc-actions">' + dl + '<button class="btn btn-secondary" type="button" data-print>Print protocol</button></div></header>' +
      body + '</article>', p.id + ' ' + trunc(p.question, 60), 'proposed');
  }

  function renderProposedList(params) {
    var area = params.get('area') || '';
    var areas = new Map();
    S.proposed.forEach(function (p) {
      if (area && p.area !== area) return;
      if (!areas.has(p.area)) areas.set(p.area, []);
      areas.get(p.area).push(p);
    });
    var keys = Array.from(areas.keys()).sort();
    var html = '<nav class="breadcrumb" aria-label="Breadcrumb"><a href="#/">Home</a> › ' + (area ? '<a href="#/proposed">Proposed audits</a> › ' + esc(area) : 'Proposed audits') + '</nav>' +
      '<h1>' + esc(area || 'Proposed audits') + '</h1>' +
      '<p class="page-intro">Ready-to-run audit protocols built on current standards. Each has one measurable question, the exact standard wording, a pass definition, a data collection template and a timeline. ' + PROPOSED_BADGE + '</p>' +
      '<div class="toolbar"><label class="visually-hidden" for="pf">Filter proposed audits</label><input id="pf" type="search" placeholder="Filter by words in the question" data-listfilter=".result"></div>' +
      keys.map(function (k) {
        return '<div class="area-group"><h2>' + esc(k) + ' <span class="muted" style="font-weight:400;font-size:.9rem">(' + areas.get(k).length + ')</span></h2><ul class="result-list">' +
          areas.get(k).map(function (p) {
            return '<li class="result result-proposed" data-text="' + attr((p.id + ' ' + p.question + ' ' + (p.standard || {}).source).toLowerCase()) + '">' +
              '<h3><a href="#/proposed/' + attr(p.id) + '">' + esc(p.question) + '</a></h3><p class="meta"><span class="id-tag">' + esc(p.id) + '</span> · ' +
              esc(trunc((p.standard || {}).source, 120)) + (p.novelty ? ' · ' + esc(cap(p.novelty)) : '') + '</p></li>';
          }).join('') + '</ul></div>';
      }).join('');
    page(html, area || 'Proposed audits', 'proposed');
  }

  /* ---------- published audit ---------- */
  function renderAudit(id) {
    var a = S.byId.get(id);
    if (!a) return renderNotFound();
    page('<div class="loading" role="status"><span class="spinner" aria-hidden="true"></span>Loading audit…</div>', a.t, 'search');
    loadAudit(id).then(function (f) {
      if (!f) return renderNotFound();
      if (parseHash().parts[1] !== String(id)) return;
      renderAuditFull(a, f);
      focusMain();
    }).catch(function () {
      page('<h1>This audit could not be loaded</h1><p>Please check your connection and <a href="' + location.hash + '">try again</a>.</p>', 'Error', 'search');
    });
  }
  function abstractHtml(t) {
    var re = /\s(?=(?:Background|Introduction|Aims?|Objectives?|Methods?|Results?|Conclusions?|Discussion)\s+[A-Z0-9])/;
    return String(t).split(re).map(function (para) {
      var m = para.match(/^(Background|Introduction|Aims?|Objectives?|Methods?|Results?|Conclusions?|Discussion)\s+(.*)$/);
      return m ? '<p><strong>' + esc(m[1]) + '</strong> ' + esc(m[2]) + '</p>' : '<p>' + esc(para) + '</p>';
    }).join('');
  }
  function cycleStep(title, c) {
    c = c || {};
    var n = c.sample_size != null ? c.sample_size : c.n;
    var meta = [has(c.period) ? esc(c.period) : '', has(n) ? 'n = ' + esc(n) : ''].filter(Boolean).join(' · ');
    return '<div class="flow-step"><h3>' + esc(title) + '</h3>' + (meta ? '<p class="meta">' + meta + '</p>' : '') +
      '<p>' + esc(show(c.result)) + '</p></div>';
  }
  function renderAuditFull(a, f) {
    var d = f.detail || {}, p = f.paper || {};
    var loop = d.loop_closed === true ? 'yes' : d.loop_closed;
    var loopTxt = !has(loop) ? 'Not reported' : cap(String(loop));
    var country = has(d.country) ? d.country : a.co;
    var meta = [
      ['Specialty', esc(f.specialty)],
      f.topic ? ['Topic', '<a href="' + topicHref(f.topic) + '">' + esc(f.topic) + '</a>'] : null,
      has(d.setting) ? ['Setting', esc(d.setting)] : null,
      country ? ['Country', esc(country)] : null,
      p.year ? ['Year', esc(p.year)] : null,
      p.journal ? ['Journal', esc(p.journal)] : null,
      ['Status', esc(cap(f.status))]
    ].filter(Boolean).map(function (r) { return '<dt>' + r[0] + '</dt><dd>' + r[1] + '</dd>'; }).join('');

    var n = 0, body = '';
    body += sec(++n, 'Standard', '<p class="prose">' + esc(show(f.standard)) + '</p>');
    if (f.detail) {
      body += sec(++n, 'What was done', '<div class="flow">' + cycleStep('Cycle 1', d.cycle1) + '<div class="flow-arrow" aria-hidden="true">→</div>' +
        '<div class="flow-step"><h3>Intervention</h3><p>' + esc(show(d.intervention)) + '</p></div><div class="flow-arrow" aria-hidden="true">→</div>' +
        cycleStep('Cycle 2', d.cycle2) + '</div>' +
        '<dl class="kv" style="margin-top:14px"><dt>Loop closed</dt><dd>' + esc(loopTxt) + '</dd><dt>Fix type</dt><dd>' + esc(cap(show(d.fix_type))) + '</dd></dl>' +
        (has(d.other_findings) ? sub('Other findings', '<p class="prose">' + esc(d.other_findings) + '</p>') : ''));
    } else {
      body += sec(++n, 'Findings and change', '<dl class="kv"><dt>Finding</dt><dd>' + esc(show(f.finding)) + '</dd><dt>Change</dt><dd>' + esc(show(f.change)) + '</dd></dl>');
    }
    if (has(f.next_audit)) body += sec(++n, 'Suggested next audit', '<p class="prose">' + esc(f.next_audit) + '</p>');
    if (p.abstract) {
      body += sec(++n, 'Abstract', '<div class="abstract">' + abstractHtml(p.abstract) + '</div>' +
        '<p class="licence-note">Abstract reproduced verbatim under the article’s ' + esc(String(p.licence).toUpperCase()) + ' licence.</p>');
    }
    if (f.figures && f.figures.length) {
      body += sec(++n, 'Figures', '<div class="figures">' + f.figures.map(function (g) {
        return '<figure><a href="' + attr(g.file) + '" target="_blank" rel="noopener"><img src="' + attr(g.file) + '" alt="' + attr((g.label || 'Figure') + ': ' + (g.caption || '')) + '" loading="lazy"></a>' +
          '<figcaption><strong>' + esc(g.label || 'Figure') + '.</strong> ' + esc(g.caption || '') + '<span class="credit">' + esc(g.credit || '') + '</span></figcaption></figure>';
      }).join('') + '</div>');
    }
    var cite = p.citation || (has(d.citation) ? d.citation : '');
    var links = [];
    if (p.doi) links.push('<a href="https://doi.org/' + attr(p.doi) + '" target="_blank" rel="noopener">DOI: ' + esc(p.doi) + '</a>');
    if (p.pmid) links.push('<a href="https://pubmed.ncbi.nlm.nih.gov/' + attr(p.pmid) + '/" target="_blank" rel="noopener">PubMed ' + esc(p.pmid) + '</a>');
    if (p.pmcid) links.push('<a href="https://pmc.ncbi.nlm.nih.gov/articles/' + attr(p.pmcid) + '/" target="_blank" rel="noopener">Full text (' + esc(p.pmcid) + ')</a>');
    var src = safeUrl(f.source);
    if (src && !links.length) links.push('<a href="' + attr(src) + '" target="_blank" rel="noopener">Original source</a>');
    body += sec(++n, 'Citation', (cite ? '<p class="citation">' + esc(cite) + '</p>' : (links.length ? '' : '<p class="muted">No individual source is recorded for this entry.</p>')) +
      (links.length ? '<p class="links">' + links.join('') + '</p>' : ''));

    var related = [];
    var cited = S.citedBy.get(a.id) || [];
    if (cited.length) related.push(sub('Proposed audits that build on this', '<ul>' + cited.map(function (pid) {
      var q = S.pById.get(pid); return '<li><a href="#/proposed/' + pid + '">' + esc(q.question) + '</a> <span class="id-tag">' + pid + '</span></li>';
    }).join('') + '</ul>'));
    if (f.topic) {
      var same = S.lib.filter(function (x) { return x.tp === f.topic && x.id !== a.id; })
        .sort(function (x, y) { return (y.uk - x.uk) || ((y.lc || 0) - (x.lc || 0)); });
      if (same.length) related.push(sub('More on this topic', '<ul>' + same.slice(0, 5).map(function (x) {
        return '<li><a href="#/audit/' + x.id + '">' + esc(x.t) + '</a>' + (x.lc ? ' ' + badge('Closed loop', 'ok') : '') + '</li>';
      }).join('') + '</ul>' + (same.length > 5 ? '<p><a href="' + topicHref(f.topic) + '">All ' + (same.length + 1) + ' audits on this topic</a></p>' : '')));
    }
    if (related.length) body += '<section class="no-print"><h2>Related</h2>' + related.join('') + '</section>';

    page('<nav class="breadcrumb" aria-label="Breadcrumb"><a href="#/">Home</a> › <a href="#/search?sp=' + encodeURIComponent(a.g) + '">' + esc(a.g) + '</a>' +
      (f.topic ? ' › <a href="' + topicHref(f.topic) + '">' + esc(f.topic) + '</a>' : '') + '</nav>' +
      '<article class="doc"><header class="doc-head"><div class="eyebrow"><span class="id-tag">[' + a.id + ']</span>' + ukBadge(a) + loopBadge(a) + '</div>' +
      '<h1>' + esc(f.title) + '</h1><dl class="kv">' + meta + '</dl></header>' + body + '</article>', trunc(f.title, 70), 'search', true);
  }

  /* ---------- topics ---------- */
  function cardHtml(c) {
    var draft = !!c.draft;
    var facts = [c.audits_in_library != null ? c.audits_in_library + ' audits' : '', (c.countries || []).join(', '), c.years_seen].filter(Boolean).map(esc).join(' · ');
    return '<div class="card-panel ' + (draft ? 'draft' : 'seed') + '">' +
      '<div class="badges">' + (draft ? DRAFT_BADGE : badge('Clinical team card', 'ok')) + '</div>' +
      (draft ? '<p class="draft-banner"><span aria-hidden="true">⚠</span><span>This card summarises the audits below and has not yet been reviewed by a consultant.</span></p>' : '') +
      (facts ? '<p class="muted">' + facts + '</p>' : '') +
      '<div class="card-grid">' +
      sub('Usual baseline', '<p>' + linkify(show(c.usual_baseline)) + '</p>') +
      sub('Fix that works', '<p>' + linkify(show(c.fix_that_works)) + '</p>') +
      sub('Fix that fails', '<p>' + linkify(show(c.fix_that_fails)) + '</p>') +
      sub('Consultant advice', '<p>' + linkify(show(c.consultant_advice)) + '</p>') +
      '</div></div>';
  }
  function renderTopic(topic) {
    var cards = (S.cardsByTopic.get(topic) || []).slice().sort(function (x, y) { return (x.draft || 0) - (y.draft || 0); });
    var ids = new Set();
    S.lib.forEach(function (a) { if (a.tp === topic) ids.add(a.id); });
    cards.forEach(function (c) { (c.evidence_ids || []).forEach(function (i) { if (S.byId.has(i)) ids.add(i); }); });
    if (!cards.length && !ids.size) return renderNotFound();
    var audits = Array.from(ids).map(function (i) { return S.byId.get(i); }).sort(function (x, y) {
      return (y.uk - x.uk) || ((y.lc || 0) - (x.lc || 0)) || ((+y.y || 0) - (+x.y || 0));
    });
    var props = [];
    S.proposed.forEach(function (p) {
      if ((p.evidence || []).some(function (e) { return idsIn(e).some(function (i) { return ids.has(i); }); })) props.push(p);
    });
    var closed = audits.filter(function (a) { return a.lc; }).length;
    page('<nav class="breadcrumb" aria-label="Breadcrumb"><a href="#/">Home</a> › <a href="#/topics">Topics</a> › ' + esc(topic) + '</nav>' +
      '<h1>' + esc(topic) + '</h1>' +
      '<p class="page-intro">' + fmt(audits.length) + ' audit' + (audits.length === 1 ? '' : 's') + ' in the library, ' + fmt(closed) + ' closed the loop.</p>' +
      cards.map(cardHtml).join('') +
      (props.length ? '<div class="results-title"><h2>Proposed audits (ready to run)</h2><span class="count">' + props.length + '</span></div><ul class="result-list">' + props.map(propResult).join('') + '</ul>' : '') +
      '<div class="results-title"><h2>Published audits</h2><span class="count">' + audits.length + '</span></div>' +
      '<ul class="result-list">' + audits.map(pubResult).join('') + '</ul>', topic, 'topics');
  }
  function renderTopics() {
    var list = [];
    S.cardsByTopic.forEach(function (cs, t) { list.push({ t: t, draft: cs.every(function (c) { return c.draft; }), n: S.topicCounts.get(t) || 0 }); });
    list.sort(function (a, b) { return a.t.localeCompare(b.t); });
    page('<nav class="breadcrumb" aria-label="Breadcrumb"><a href="#/">Home</a> › Topics</nav><h1>Topic cards</h1>' +
      '<p class="page-intro">Each card sums up what the audits on a topic usually find, which fix works, which fails, and advice on what to audit next.</p>' +
      '<div class="toolbar"><label class="visually-hidden" for="tf">Filter topics</label><input id="tf" type="search" placeholder="Filter topics" data-listfilter=".result"></div>' +
      '<ul class="result-list">' + list.map(function (x) {
        return '<li class="result" data-text="' + attr(x.t.toLowerCase()) + '"><h3><a href="' + topicHref(x.t) + '">' + esc(x.t) + '</a></h3>' +
          '<div class="badges">' + badge(x.n + ' audits') + (x.draft ? DRAFT_BADGE : badge('Clinical team card', 'ok')) + '</div></li>';
      }).join('') + '</ul>', 'Topics', 'topics');
  }

  /* ---------- standards ---------- */
  function renderStandards(params) {
    var q = params.get('q') || '';
    page('<nav class="breadcrumb" aria-label="Breadcrumb"><a href="#/">Home</a> › Standards</nav><h1>Standards</h1>' +
      '<p class="page-intro">The exact wording of each standard used by the proposed audits, with a link to the source.</p>' +
      '<div class="toolbar"><label class="visually-hidden" for="sf">Filter standards</label><input id="sf" type="search" value="' + attr(q) + '" placeholder="Filter by source or wording, e.g. NICE NG24, BOAST" data-listfilter=".std"></div>' +
      '<p class="muted" data-count></p>' +
      '<ul class="std-list">' + S.standards.map(function (s) {
        var url = safeUrl(s.url);
        return '<li class="std" data-text="' + attr((s.source + ' ' + s.wording + ' ' + (s.used_by || []).join(' ')).toLowerCase()) + '">' +
          '<h3>' + esc(s.source) + '</h3><blockquote class="standard"><p>“' + esc(s.wording) + '”</p></blockquote>' +
          (url ? '<p class="standard-source"><a href="' + attr(url) + '" target="_blank" rel="noopener">Read the standard</a></p>' : '') +
          ((s.used_by || []).length ? '<div class="used"><span class="muted">Used by:</span>' + s.used_by.map(function (id) {
            var p = S.pById.get(id);
            return p ? '<a href="#/proposed/' + id + '" title="' + attr(p.question) + '">' + id + '</a>' : '';
          }).join('') + '</div>' : '') + '</li>';
      }).join('') + '</ul>', 'Standards', 'standards');
    var inp = document.getElementById('sf');
    if (q) applyListFilter(inp);
    else updateCount(inp);
  }

  function renderNotFound() {
    page('<h1>Page not found</h1><p>The page you asked for does not exist. <a href="#/">Go to the home page</a> or search the library.</p>' + searchForm('', false), 'Page not found', '');
  }

  /* ---------- list filters ---------- */
  function applyListFilter(inp) {
    var sel = inp.getAttribute('data-listfilter'), words = tokensLoose(inp.value);
    main.querySelectorAll(sel).forEach(function (el) {
      var t = el.getAttribute('data-text') || '';
      el.hidden = !words.every(function (w) { return t.indexOf(w) !== -1; });
    });
    main.querySelectorAll('.area-group').forEach(function (g) {
      g.hidden = !g.querySelector('.result:not([hidden])');
    });
    updateCount(inp);
  }
  function updateCount(inp) {
    var c = main.querySelector('[data-count]');
    if (!c || !inp) return;
    var sel = inp.getAttribute('data-listfilter');
    var all = main.querySelectorAll(sel).length, vis = main.querySelectorAll(sel + ':not([hidden])').length;
    c.textContent = vis === all ? fmt(all) + ' standards' : fmt(vis) + ' of ' + fmt(all) + ' standards';
  }
  function tokensLoose(s) { return String(s || '').toLowerCase().split(/\s+/).filter(Boolean); }

  /* ---------- events ---------- */
  document.addEventListener('submit', function (e) {
    var f = e.target.closest('[data-search]');
    if (!f) return;
    e.preventDefault();
    var q = f.querySelector('input[name="q"]').value.trim();
    var cur = parseHash();
    var st = cur.parts[0] === 'search' ? searchState(cur.params) : { sp: '', closed: false, detailed: false };
    st.q = q;
    location.hash = searchHash(st);
  });
  document.addEventListener('change', function (e) {
    var el = e.target.closest('[data-filter]');
    if (!el) return;
    var st = searchState(parseHash().params);
    var k = el.getAttribute('data-filter');
    if (k === 'sp') st.sp = el.value; else st[k] = el.checked;
    history.replaceState(null, '', searchHash(st));
    renderSearch(parseHash().params, true);
    var again = main.querySelector('[data-filter="' + k + '"]');
    if (again) again.focus();
  });
  document.addEventListener('input', function (e) {
    var el = e.target.closest('[data-listfilter]');
    if (el) applyListFilter(el);
  });
  document.addEventListener('click', function (e) {
    var m = e.target.closest('[data-more]');
    if (m) {
      var which = m.getAttribute('data-more');
      var before = which === 'pub' ? S.view.pubShown : S.view.propShown;
      if (which === 'pub') S.view.pubShown += 25; else S.view.propShown += 10;
      renderSearch(parseHash().params, true);
      var list = main.querySelector(which === 'pub' ? '[aria-labelledby="pub-h"]' : '[aria-labelledby="prop-h"]');
      var next = list && list.children[before];
      if (next) { var a = next.querySelector('a'); if (a) a.focus(); }
      return;
    }
    if (e.target.closest('[data-print]')) window.print();
  });
  window.addEventListener('hashchange', route);

  load().then(route).catch(function (err) {
    main.innerHTML = '<h1>The library could not be loaded</h1><p>Please check your connection and refresh the page.</p>';
    if (window.console) console.warn(err);
  });

  /* ---------- installable app and offline use ---------- */
  var installBtn = document.querySelector('[data-install]');
  var installPrompt = null;
  window.addEventListener('beforeinstallprompt', function (e) {
    e.preventDefault();
    installPrompt = e;
    if (installBtn) installBtn.hidden = false;
  });
  window.addEventListener('appinstalled', function () {
    installPrompt = null;
    if (installBtn) installBtn.hidden = true;
  });
  if (installBtn) installBtn.addEventListener('click', function () {
    if (!installPrompt) return;
    var ev = installPrompt;
    installPrompt = null;
    installBtn.hidden = true;
    ev.prompt();
  });

  if ('serviceWorker' in navigator && window.isSecureContext) {
    window.addEventListener('load', function () {
      navigator.serviceWorker.register('sw.js', { scope: './', updateViaCache: 'none' }).catch(function () {});
      navigator.serviceWorker.ready.then(function () {
        return window.caches ? caches.match('data/library.json') : null;
      }).then(function (hit) {
        var note = document.querySelector('[data-offline]');
        if (hit && note) note.hidden = false;
      }).catch(function () {});
    });
  }
})();
