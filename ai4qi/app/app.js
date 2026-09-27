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
      else if (name === 'account' && BE.url) renderAccount();
      else if (name === 'admin' && p[1] === 'feedback' && BE.url) renderAdmin();
      else if (name === 'admin' && p[1] === 'stats' && BE.url) renderStats();
      else if (name === 'my-audits' && BE.url) renderMyAudits();
      else renderNotFound();
    } catch (err) {
      renderNotFound();
      if (window.console) console.warn(err);
    }
    if (!sameView) focusMain();
    trackPageview();
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
        (p.novelty ? '<span><strong>Gap:</strong> ' + esc(cap(p.novelty)) + '</span>' : '') + '</div>') +
      feedbackBox(p.id);

    page('<nav class="breadcrumb" aria-label="Breadcrumb"><a href="#/">Home</a> › <a href="#/proposed">Proposed audits</a> › <a href="#/proposed?area=' + encodeURIComponent(p.area) + '">' + esc(p.area) + '</a></nav>' +
      '<article class="doc"><header class="doc-head"><div class="eyebrow"><span class="id-tag">' + esc(p.id) + '</span>' + PROPOSED_BADGE + badge(p.area, 'primary') + '</div>' +
      '<h1>' + esc(p.question) + '</h1>' +
      '<div class="doc-actions">' + dl + '<button class="btn btn-secondary" type="button" data-print>Print protocol</button></div>' +
      '<div class="track no-print" data-track="' + attr(p.id) + '" hidden></div></header>' +
      body + '</article>', p.id + ' ' + trunc(p.question, 60), 'proposed');
    showUsefulCount(p.id);
    showTracker(p.id);
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

  /* ---------- feedback (thumbs up / down) ---------- */
  var FB_KEY = 'ai4qi_feedback_v1';
  var FB_REASONS = ['Too generic', 'Too specific', 'Poor framing', 'Too complex',
    'Not relevant to my specialty', 'Not an important topic to audit'];
  var FB_URL = '';                       // set from config.json ("feedback_url") when a collector exists
  function fbAll() { try { return JSON.parse(localStorage.getItem(FB_KEY) || '[]'); } catch (e) { return []; } }
  function fbSave(list) { try { localStorage.setItem(FB_KEY, JSON.stringify(list)); } catch (e) {} }
  function fbFor(id) { return fbAll().filter(function (f) { return f.id === id; }).pop(); }
  function flushFeedback() {
    if (!navigator.onLine) return;
    var list = fbAll(), pending = list.filter(function (f) { return !f.sent; });
    if (!pending.length) return;
    if (BE.url) { flushToSupabase(); return; }
    if (!FB_URL) return;
    pending.forEach(function (f) {
      fetch(FB_URL, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(f) })
        .then(function (r) { if (r.ok) { f.sent = true; fbSave(list); } }).catch(function () {});
    });
  }
  window.addEventListener('online', flushFeedback);
  function recordFeedback(id, rating, reasons, comment) {
    var list = fbAll().filter(function (f) { return f.id !== id; });
    list.push({ id: id, rating: rating, reasons: reasons || [], comment: comment || '', at: new Date().toISOString(), sent: false, uid: uuid() });
    fbSave(list); flushFeedback();
  }
  function thumb(dir) {
    var d = dir === 'up'
      ? 'M7 10v10H4V10h3zm2 10h8.2a2 2 0 0 0 2-1.6l1.3-6.5A2 2 0 0 0 18.5 9.5H14l.8-3.9a1.6 1.6 0 0 0-2.9-1.2L9 9.3V20z'
      : 'M7 14V4H4v10h3zm2-10h8.2a2 2 0 0 1 2 1.6l1.3 6.5a2 2 0 0 1-2 2.4H14l.8 3.9a1.6 1.6 0 0 1-2.9 1.2L9 14.7V4z';
    return '<svg width="20" height="20" viewBox="0 0 24 24" aria-hidden="true"><path d="' + d + '" fill="currentColor"/></svg>';
  }
  function feedbackBox(id) {
    var prev = fbFor(id);
    var done = prev ? '<p class="fb-thanks" role="status">Thank you – your feedback has been recorded.</p>' : '';
    return '<section class="fb no-print" data-fb="' + attr(id) + '" aria-label="Feedback on this audit">' +
      '<div class="fb-row"><span class="fb-q">Was this audit idea useful?</span>' +
      '<button type="button" class="fb-btn' + (prev && prev.rating === 'up' ? ' is-on' : '') + '" data-fb-rate="up" aria-pressed="' + !!(prev && prev.rating === 'up') + '">' + thumb('up') + '<span class="sr-only">Yes, useful</span></button>' +
      '<button type="button" class="fb-btn' + (prev && prev.rating === 'down' ? ' is-on' : '') + '" data-fb-rate="down" aria-pressed="' + !!(prev && prev.rating === 'down') + '">' + thumb('down') + '<span class="sr-only">No, not useful</span></button>' +
      '<span class="fb-count" data-fb-count hidden></span></div>' +
      '<div class="fb-more" hidden><p class="fb-sub">What was wrong? Choose any that apply.</p><div class="fb-chips">' +
      FB_REASONS.map(function (r) { return '<button type="button" class="fb-chip" data-fb-reason="' + attr(r) + '" aria-pressed="false">' + esc(r) + '</button>'; }).join('') +
      '</div><label class="fb-sub" for="fb-c-' + attr(id) + '">Anything else? (optional)</label>' +
      '<textarea id="fb-c-' + attr(id) + '" class="fb-comment" rows="2" maxlength="500"></textarea>' +
      '<button type="button" class="btn" data-fb-send>Send feedback</button></div>' +
      '<div class="fb-done">' + done + '</div></section>';
  }
  document.addEventListener('click', function (e) {
    var box = e.target.closest('[data-fb]');
    if (!box) return;
    var id = box.getAttribute('data-fb');
    var rate = e.target.closest('[data-fb-rate]'), chip = e.target.closest('[data-fb-reason]'), send = e.target.closest('[data-fb-send]');
    var more = box.querySelector('.fb-more'), doneEl = box.querySelector('.fb-done');
    if (rate) {
      var r = rate.getAttribute('data-fb-rate');
      box.querySelectorAll('[data-fb-rate]').forEach(function (b) { var on = b === rate; b.classList.toggle('is-on', on); b.setAttribute('aria-pressed', on); });
      if (r === 'up') { more.hidden = true; recordFeedback(id, 'up'); doneEl.innerHTML = '<p class="fb-thanks" role="status">Thank you – your feedback has been recorded.</p>'; }
      else { more.hidden = false; doneEl.innerHTML = ''; var first = more.querySelector('.fb-chip'); if (first) first.focus(); }
    } else if (chip) {
      var on = chip.getAttribute('aria-pressed') !== 'true';
      chip.setAttribute('aria-pressed', on); chip.classList.toggle('is-on', on);
    } else if (send) {
      var reasons = Array.prototype.map.call(box.querySelectorAll('.fb-chip.is-on'), function (c) { return c.getAttribute('data-fb-reason'); });
      recordFeedback(id, 'down', reasons, box.querySelector('.fb-comment').value.trim());
      more.hidden = true;
      doneEl.innerHTML = '<p class="fb-thanks" role="status">Thank you – this helps us improve the audit library.</p>';
    }
  });
  /* ---------- optional Supabase backend: shared feedback, sign-in, admin view ----------
     Switched on only when config.json has supabase_url and supabase_anon_key. Without them the
     app behaves exactly as before (feedback stays on the device or goes to feedback_url). */
  var SUPABASE_JS = 'https://cdn.jsdelivr.net/npm/@supabase/supabase-js@2';
  var DEVICE_KEY = 'ai4qi_device_v1';
  var GRADES = ['Medical student', 'Foundation doctor (FY1–FY2)', 'Core or specialty trainee (CT/ST1–2)',
    'Specialty registrar (ST3+)', 'Specialty doctor or specialist (SAS)', 'Locally employed doctor', 'Consultant',
    'General practitioner', 'Nurse or midwife', 'Allied health professional', 'Pharmacist',
    'Quality improvement or clinical governance lead', 'Other'];
  // Fixed lists; they must match ai4qi_specialties() and ai4qi_regions() in backend migration 002.
  var SPECIALTIES = ['Acute and general medicine', 'Anaesthesia', 'Cardiology', 'Cardiothoracic surgery',
    'Care of the elderly', 'Clinical governance and quality improvement', 'Dentistry and oral surgery',
    'Dermatology', 'Emergency medicine', 'Endocrinology and diabetes', 'ENT', 'Gastroenterology',
    'General practice', 'General surgery', 'Haematology', 'Infectious diseases and microbiology',
    'Intensive care', 'Medical education', 'Neonatology', 'Neurology', 'Neurosurgery',
    'Nursing and midwifery', 'Obstetrics and gynaecology', 'Oncology', 'Ophthalmology', 'Orthopaedics',
    'Paediatric surgery', 'Paediatrics', 'Palliative care', 'Pathology', 'Pharmacy', 'Plastic surgery',
    'Psychiatry', 'Radiology', 'Renal medicine', 'Respiratory', 'Rheumatology', 'Sexual health', 'Stroke',
    'Therapies and allied health', 'Urology', 'Vascular surgery', 'Other'];
  var REGIONS = ['North East and Yorkshire', 'North West', 'Midlands', 'East of England', 'London', 'South East',
    'South West', 'Scotland', 'Wales', 'Northern Ireland', 'Ireland', 'Outside the UK and Ireland'];
  // "My audits" steps, in order; the last one counts as completed.
  var STEPS = [['started', 'Started'], ['cycle1', 'Cycle 1 collected'], ['change', 'Change made'],
    ['reaudit', 'Re-audit done'], ['closed', 'Loop closed']];
  var BE = {
    url: '', key: '', client: null, loading: null, user: null, admin: null, profile: undefined,
    version: null, summary: null, summaryAt: 0, authError: '', flushing: false
  };
  var accountLink = document.querySelector('[data-account]');

  function uuid() {
    var c = window.crypto;
    if (c && c.randomUUID) return c.randomUUID();
    var b = c.getRandomValues(new Uint8Array(16)), h = [];
    b[6] = (b[6] & 15) | 64; b[8] = (b[8] & 63) | 128;
    for (var i = 0; i < 16; i++) h.push((b[i] + 256).toString(16).slice(1));
    return h.slice(0, 4).join('') + '-' + h.slice(4, 6).join('') + '-' + h.slice(6, 8).join('') + '-' + h.slice(8, 10).join('') + '-' + h.slice(10).join('');
  }
  function deviceId() {
    try {
      var d = localStorage.getItem(DEVICE_KEY);
      if (!d || !/^[A-Za-z0-9_-]{8,64}$/.test(d)) { d = uuid(); localStorage.setItem(DEVICE_KEY, d); }
      return d;
    } catch (e) { return BE.tmpDevice || (BE.tmpDevice = uuid()); }
  }
  function hasStoredSession() {
    try {
      for (var i = 0; i < localStorage.length; i++) if (/^sb-.+-auth-token$/.test(localStorage.key(i))) return true;
    } catch (e) {}
    return false;
  }

  // A magic link brings the reader back with ?code=… (or ?error=…). Note it before routing so the
  // account page opens, and let supabase-js exchange the code for a session.
  var AUTH_RETURN = (function () {
    var q = new URLSearchParams(location.search), h = new URLSearchParams(location.hash.replace(/^#/, ''));
    var err = q.get('error_description') || h.get('error_description') || q.get('error') || h.get('error');
    if (!err && !q.get('code') && !h.get('access_token')) return null;
    return { error: err || '' };
  })();
  var ORIGINAL_HASH = location.hash;
  if (AUTH_RETURN) history.replaceState(null, '', location.pathname + location.search + '#/account');
  function cleanAuthUrl() {
    if (location.search) history.replaceState(null, '', location.pathname + location.hash);
  }

  var configReady = fetch('config.json', { cache: 'no-store' })
    .then(function (r) { return r.ok ? r.json() : {}; })
    .catch(function () { return {}; })
    .then(function (c) {
      c = c || {};
      FB_URL = c.feedback_url || '';
      BE.cfg = c;
      try { initAnalytics(c); } catch (e) {}
      var u = String(c.supabase_url || '').trim().replace(/\/+$/, ''), k = String(c.supabase_anon_key || '').trim();
      var okUrl = /^https:\/\/[a-z0-9.-]+(:\d+)?$/i.test(u) || /^http:\/\/(localhost|127\.0\.0\.1)(:\d+)?$/i.test(u);
      if (okUrl && k) { BE.url = u; BE.key = k; }
      if (BE.url) {
        if (accountLink) accountLink.hidden = false;
        updateAccountLink();
        if (AUTH_RETURN || hasStoredSession()) sbClient().catch(function () {});
      } else if (AUTH_RETURN) {
        // No backend: put the address back exactly as it was and show that page.
        history.replaceState(null, '', location.pathname + location.search + ORIGINAL_HASH);
        if (S.lib.length) route();
      }
      flushFeedback();
      // If the library finished loading first, an account or admin link was shown as "not found".
      var cur = parseHash().parts[0];
      if (BE.url && S.lib.length && (cur === 'account' || cur === 'admin' || cur === 'my-audits')) route();
    });

  function sbClient() {
    if (!BE.url) return Promise.reject(new Error('Supabase is not configured'));
    if (BE.client) return Promise.resolve(BE.client);
    if (!BE.loading) {
      BE.loading = new Promise(function (resolve, reject) {
        if (window.supabase && window.supabase.createClient) { resolve(); return; }
        var sc = document.createElement('script');
        sc.src = SUPABASE_JS; sc.async = true; sc.crossOrigin = 'anonymous';
        sc.onload = function () { resolve(); };
        sc.onerror = function () { sc.remove(); reject(new Error('supabase-js could not be loaded')); };
        document.head.appendChild(sc);
      }).then(function () {
        var c = window.supabase.createClient(BE.url, BE.key, {
          auth: { flowType: 'pkce', persistSession: true, autoRefreshToken: true, detectSessionInUrl: true }
        });
        c.auth.onAuthStateChange(function (event, session) {
          // Keep this callback free of other Supabase calls (they would wait on the auth lock).
          setTimeout(function () { setUser(session ? session.user : null); }, 0);
        });
        return c.auth.getSession().then(function (res) {
          if (AUTH_RETURN && AUTH_RETURN.error && !(res.data && res.data.session)) BE.authError = AUTH_RETURN.error;
          if (AUTH_RETURN) cleanAuthUrl();
          BE.client = c;
          BE.user = res.data && res.data.session ? res.data.session.user : null;
          updateAccountLink();
          if (BE.user) { setTimeout(flushFeedback, 0); recordActivity(); }
          return c;
        });
      });
      BE.loading.catch(function () { BE.loading = null; });   // allow a retry, e.g. once back online
    }
    return BE.loading;
  }

  function setUser(u) {
    var before = BE.user ? BE.user.id : null, after = u ? u.id : null;
    BE.user = u || null;
    updateAccountLink();
    if (before === after) return;
    BE.admin = null; BE.profile = undefined;
    BE.tracker = null;
    if (u) { flushFeedback(); recordActivity(); }
    var name = parseHash().parts[0];
    if (S.lib.length && (name === 'account' || name === 'admin' || name === 'my-audits')) route();
    else if (name === 'proposed' && parseHash().parts[1]) showTracker(parseHash().parts[1]);
  }
  function updateAccountLink() {
    if (!accountLink) return;
    var signedIn = BE.user || (!BE.client && hasStoredSession());
    accountLink.textContent = signedIn ? 'Account' : 'Sign in';
  }

  function appVersion() {
    if (!BE.version) {
      BE.version = fetch('data/version.json', { cache: 'no-store' })
        .then(function (r) { return r.ok ? r.json() : {}; })
        .then(function (v) { return String((v && v.version) || '').slice(0, 32); })
        .catch(function () { return ''; });
    }
    return BE.version;
  }

  // Send queued feedback rows one at a time. Each row carries a client-made id, so a retry after a
  // lost response is recognised as a duplicate (23505) and simply marked as sent.
  function flushToSupabase() {
    if (BE.flushing) return;
    var list = fbAll(), pending = list.filter(function (f) { return !f.sent; });
    if (!pending.length) return;
    BE.flushing = true;
    pending.forEach(function (f) { if (!f.uid) f.uid = uuid(); });
    fbSave(list);
    var done = {};
    Promise.all([sbClient(), appVersion()]).then(function (r) {
      var c = r[0], ver = r[1], dev = deviceId();
      return pending.reduce(function (chain, f) {
        return chain.then(function () {
          var row = {
            id: f.uid, audit_id: f.id, rating: f.rating, reasons: f.reasons || [],
            comment: String(f.comment || '').slice(0, 500), device_id: dev,
            user_id: BE.user ? BE.user.id : null, app_version: ver || null
          };
          return c.from('feedback').insert(row).then(function (res) {
            var code = res.error ? String(res.error.code || '') : '';
            if (!res.error || code === '23505') done[f.uid] = 'sent';
            else if (/^(23502|23503|23514|22P02|42501)$/.test(code)) done[f.uid] = 'rejected';  // would never be accepted
            else if (code === 'PT429') throw new Error('rate limited');                          // try again later
          });
        });
      }, Promise.resolve());
    }).catch(function () {}).then(function () {
      var cur = fbAll();
      cur.forEach(function (f) {
        if (done[f.uid]) { f.sent = true; if (done[f.uid] === 'rejected') f.rejected = true; }
      });
      fbSave(cur);
      BE.flushing = false;
      if (Object.keys(done).length) BE.summaryAt = 0;
    });
  }

  function restHeaders() {
    var h = { apikey: BE.key, 'Content-Type': 'application/json' };
    if (/^eyJ/.test(BE.key)) h.Authorization = 'Bearer ' + BE.key;   // legacy JWT anon key
    return h;
  }
  function loadSummary() {
    if (!BE.url || !navigator.onLine) return Promise.resolve(null);
    if (BE.summary && Date.now() - BE.summaryAt < 5 * 60 * 1000) return BE.summary;
    BE.summaryAt = Date.now();
    BE.summary = fetch(BE.url + '/rest/v1/rpc/feedback_summary', {
      method: 'POST', headers: restHeaders(), body: '{}', cache: 'no-store'
    }).then(function (r) {
      if (!r.ok) throw new Error('summary ' + r.status);
      return r.json();
    }).then(function (rows) {
      var m = new Map();
      (rows || []).forEach(function (row) { m.set(row.audit_id, row); });
      return m;
    }).catch(function () { BE.summaryAt = 0; return null; });
    return BE.summary;
  }
  function showUsefulCount(id) {
    configReady.then(loadSummary).then(function (m) {
      var el = main.querySelector('[data-fb="' + id + '"] [data-fb-count]');
      var row = m && m.get(id), n = row ? Number(row.up) : 0;
      if (!el || !n) return;
      el.textContent = n === 1 ? '1 person found this useful' : fmt(n) + ' people found this useful';
      el.hidden = false;
    });
  }

  /* account page */
  function stillOn(name) { return parseHash().parts[0] === name; }
  function loadingHtml(text) {
    return '<div class="loading" role="status"><span class="spinner" aria-hidden="true"></span>' + esc(text) + '</div>';
  }
  function unavailable(title, nav) {
    page('<div class="doc narrow"><h1>' + esc(title) + '</h1><p>Sign-in is not available at the moment. Please check your connection and try again.</p></div>', title, nav);
    focusMain();
  }
  function renderAccount() {
    page(loadingHtml('Loading your account…'), 'Account', 'account');
    sbClient().then(function (c) {
      if (!BE.user) return null;
      return Promise.all([loadProfile(c), checkAdmin(c)]);
    }).then(function () {
      if (!stillOn('account')) return;
      if (BE.user) renderSignedIn(); else renderSignIn();
      focusMain();
    }, function () { if (stillOn('account')) unavailable('Account', 'account'); });
  }
  function loadProfile(c) {
    if (BE.profile !== undefined) return Promise.resolve(BE.profile);
    return c.from('profiles').select('specialty, grade, region').eq('user_id', BE.user.id).maybeSingle()
      .then(function (res) { BE.profile = res.error ? null : (res.data || null); return BE.profile; });
  }
  function checkAdmin(c) {
    if (BE.admin !== null) return Promise.resolve(BE.admin);
    // Row level security returns the admins row only to the admin it belongs to.
    return c.from('admins').select('email').limit(1)
      .then(function (res) { BE.admin = !res.error && !!(res.data && res.data.length); return BE.admin; });
  }
  function renderSignIn(title) {
    var err = BE.authError
      ? '<div class="notice notice-warn" role="alert">That sign-in link has expired or has already been used. Please request a new one.</div>'
      : '';
    page('<nav class="breadcrumb" aria-label="Breadcrumb"><a href="#/">Home</a> › Sign in</nav>' +
      '<div class="doc narrow"><h1>' + esc(title || 'Sign in') + '</h1>' +
      '<p class="page-intro">Signing in is optional. It lets you keep track of the audits you run and add your grade, specialty and region, so that your feedback on proposed audits can be read in context. There is no password: we email you a secure sign-in link.</p>' +
      err +
      '<form class="stack-form" data-signin novalidate>' +
      '<label for="acc-email">Email address</label>' +
      '<input id="acc-email" name="email" type="email" inputmode="email" autocomplete="email" spellcheck="false" required maxlength="254">' +
      '<button class="btn" type="submit">Email me a sign-in link</button>' +
      '<p class="form-status" data-form-status role="status" aria-live="polite"></p>' +
      '</form>' +
      '<p class="muted small-print">Please do not enter patient information anywhere in this app.</p></div>',
      'Sign in', 'account', true);
  }
  function optionList(list, current) {
    return list.map(function (g) { return '<option' + (g === current ? ' selected' : '') + '>' + esc(g) + '</option>'; }).join('');
  }
  function renderSignedIn() {
    var pf = BE.profile || {};
    page('<nav class="breadcrumb" aria-label="Breadcrumb"><a href="#/">Home</a> › Account</nav>' +
      '<div class="doc narrow"><h1>Your account</h1>' +
      '<p>Signed in as <strong>' + esc(BE.user.email || '') + '</strong>. Your feedback on proposed audits is now linked to your profile.</p>' +
      '<p><a href="#/my-audits">My audits</a> <span class="muted">– the proposed audits you are running, and how far each has got.</span></p>' +
      '<section class="account-section" aria-labelledby="pf-h"><h2 id="pf-h">Your profile</h2>' +
      '<p class="muted">Optional. Used only in anonymous totals, to understand who uses the library and how feedback differs between groups.</p>' +
      '<form class="stack-form" data-profile>' +
      '<label for="pf-grade">Grade or role</label>' +
      '<select id="pf-grade" name="grade"><option value="">Prefer not to say</option>' + optionList(GRADES, pf.grade) + '</select>' +
      '<label for="pf-specialty">Specialty</label>' +
      '<select id="pf-specialty" name="specialty"><option value="">Prefer not to say</option>' + optionList(SPECIALTIES, pf.specialty) + '</select>' +
      '<label for="pf-region">Region</label>' +
      '<select id="pf-region" name="region"><option value="">Prefer not to say</option>' +
      '<optgroup label="England (NHS region)">' + optionList(REGIONS.slice(0, 7), pf.region) + '</optgroup>' +
      '<optgroup label="Elsewhere">' + optionList(REGIONS.slice(7), pf.region) + '</optgroup></select>' +
      '<button class="btn" type="submit">Save profile</button>' +
      '<p class="form-status" data-form-status role="status" aria-live="polite"></p>' +
      '</form></section>' +
      (BE.admin ? '<section class="account-section" aria-labelledby="adm-h"><h2 id="adm-h">Administration</h2><ul><li><a href="#/admin/stats">Usage statistics</a></li><li><a href="#/admin/feedback">Feedback on proposed audits</a></li></ul></section>' : '') +
      '<div class="account-actions"><button class="btn btn-secondary" type="button" data-signout>Sign out</button></div>' +
      '</div>', 'Account', 'account', true);
  }

  document.addEventListener('submit', function (e) {
    var signin = e.target.closest('[data-signin]'), prof = e.target.closest('[data-profile]');
    if (!signin && !prof) return;
    e.preventDefault();
    var form = signin || prof, btn = form.querySelector('button[type="submit"]'), status = form.querySelector('[data-form-status]');
    function say(text, bad) { status.textContent = text; status.classList.toggle('is-error', !!bad); }
    if (signin) {
      var input = form.querySelector('input[name="email"]'), email = input.value.trim();
      if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email)) {
        input.setAttribute('aria-invalid', 'true'); say('Please enter a valid email address.', true); input.focus(); return;
      }
      input.removeAttribute('aria-invalid');
      btn.disabled = true; say('Sending…');
      sbClient().then(function (c) {
        return c.auth.signInWithOtp({ email: email, options: { emailRedirectTo: location.origin + location.pathname } });
      }).then(function (res) {
        if (res.error) throw res.error;
        BE.authError = '';
        var box = form.parentNode;
        form.outerHTML = '<div class="notice notice-ok" data-sent tabindex="-1"><p><strong>Check your email.</strong> We have sent a sign-in link to ' + esc(email) + '.</p>' +
          '<p>Open the link on this device, in this browser, to finish signing in. If it does not arrive within a few minutes, check your junk folder.</p>' +
          '<button class="btn btn-secondary" type="button" data-signin-again>Use a different email address</button></div>';
        var sent = box.querySelector('[data-sent]');
        if (sent) sent.focus();
      }).catch(function (err) {
        btn.disabled = false;
        var st = err && (err.status || err.code);
        say(st === 429 || st === 'over_email_send_rate_limit'
          ? 'Too many sign-in requests. Please wait a few minutes and try again.'
          : 'The sign-in link could not be sent. Please check the address and your connection, then try again.', true);
      });
    } else {
      var pick = function (name, list) { var v = form.querySelector('[name="' + name + '"]').value; return list.indexOf(v) === -1 ? null : v; };
      var spec = pick('specialty', SPECIALTIES), grade = pick('grade', GRADES), region = pick('region', REGIONS);
      btn.disabled = true; say('Saving…');
      sbClient().then(function (c) {
        if (!BE.user) throw new Error('signed out');
        return c.from('profiles').upsert({ user_id: BE.user.id, specialty: spec, grade: grade, region: region }, { onConflict: 'user_id' });
      }).then(function (res) {
        if (res.error) throw res.error;
        BE.profile = { specialty: spec, grade: grade, region: region };
        btn.disabled = false; say('Your profile has been saved.');
      }).catch(function () {
        btn.disabled = false; say('Your profile could not be saved. Please check your connection and try again.', true);
      });
    }
  });
  document.addEventListener('click', function (e) {
    if (e.target.closest('[data-signin-again]')) { renderSignIn(); var i = main.querySelector('#acc-email'); if (i) i.focus(); return; }
    if (e.target.closest('[data-signout]')) {
      sbClient().then(function (c) { return c.auth.signOut({ scope: 'local' }); })
        .catch(function () {}).then(function () { setUser(null); });
      return;
    }
    if (e.target.closest('[data-fb-csv]')) downloadFeedbackCsv();
  });

  /* admin: feedback on proposed audits (#/admin/feedback) */
  function fetchAllFeedback(c) {
    var rows = [], size = 1000;
    function next(from) {
      return c.from('feedback').select('id, audit_id, rating, reasons, comment, user_id, device_id, created_at, app_version')
        .order('created_at', { ascending: false }).range(from, from + size - 1)
        .then(function (res) {
          if (res.error) throw res.error;
          rows = rows.concat(res.data || []);
          return (res.data || []).length === size && rows.length < 50000 ? next(from + size) : rows;
        });
    }
    return next(0);
  }
  function renderAdmin() {
    page(loadingHtml('Loading feedback…'), 'Feedback', '');
    sbClient().then(function (c) {
      if (!BE.user) return null;
      return checkAdmin(c).then(function (isAdmin) { return isAdmin ? fetchAllFeedback(c) : null; });
    }).then(function (rows) {
      if (!stillOn('admin')) return;
      if (!BE.user) { renderSignIn(); focusMain(); return; }
      if (!rows) { renderNotFound(); focusMain(); return; }
      BE.adminRows = rows;
      renderAdminPage(rows);
      focusMain();
    }, function () { if (stillOn('admin')) unavailable('Feedback', ''); });
  }
  function summarise(rows) {
    // Latest rating from each person (signed-in user, otherwise device) per audit, as the public count does.
    var seen = new Set(), per = new Map();
    rows.forEach(function (r) {   // rows are newest first
      var who = r.audit_id + '|' + (r.user_id || r.device_id);
      if (seen.has(who)) return;
      seen.add(who);
      if (!per.has(r.audit_id)) per.set(r.audit_id, { id: r.audit_id, up: 0, down: 0, reasons: new Map(), comments: 0 });
      var a = per.get(r.audit_id);
      a[r.rating === 'up' ? 'up' : 'down']++;
      if (r.rating === 'down') (r.reasons || []).forEach(function (x) { a.reasons.set(x, (a.reasons.get(x) || 0) + 1); });
    });
    rows.forEach(function (r) { if (r.comment && per.has(r.audit_id)) per.get(r.audit_id).comments++; });
    return Array.from(per.values()).sort(function (a, b) { return (b.down - a.down) || (b.up - a.up) || a.id.localeCompare(b.id); });
  }
  function auditLink(id) {
    var p = S.pById.get(id);
    return '<a href="#/proposed/' + encodeURIComponent(id) + '" class="id-tag">' + esc(id) + '</a>' +
      (p ? '<span class="adm-q">' + esc(trunc(p.question, 90)) + '</span>' : '');
  }
  function dateGB(iso) {
    var d = new Date(iso);
    return isNaN(d) ? '' : d.toLocaleDateString('en-GB', { day: 'numeric', month: 'short', year: 'numeric' });
  }
  function renderAdminPage(rows) {
    var per = summarise(rows), up = 0, down = 0;
    per.forEach(function (a) { up += a.up; down += a.down; });
    var withComments = rows.filter(function (r) { return r.comment; });
    var table = per.length
      ? '<div class="table-wrap"><table class="adm-table"><caption class="visually-hidden">Ratings per proposed audit</caption><thead><tr><th scope="col">Audit</th><th scope="col" class="num-col">Useful</th><th scope="col" class="num-col">Not useful</th><th scope="col">Reasons given</th><th scope="col" class="num-col">Comments</th></tr></thead><tbody>' +
        per.map(function (a) {
          var rs = Array.from(a.reasons.entries()).sort(function (x, y) { return y[1] - x[1]; })
            .map(function (x) { return esc(x[0]) + ' (' + x[1] + ')'; }).join('<br>');
          return '<tr><td>' + auditLink(a.id) + '</td><td class="num-col">' + fmt(a.up) + '</td><td class="num-col">' + fmt(a.down) + '</td><td>' + (rs || '<span class="muted">–</span>') + '</td><td class="num-col">' + fmt(a.comments) + '</td></tr>';
        }).join('') + '</tbody></table></div>'
      : '<p class="empty">No feedback has been received yet.</p>';
    var comments = withComments.length
      ? '<ul class="adm-comments">' + withComments.map(function (r) {
          return '<li><p class="meta">' + auditLink(r.audit_id) + '</p><p class="meta">' + esc(dateGB(r.created_at)) + ' · ' + (r.rating === 'up' ? 'Useful' : 'Not useful') +
            (r.reasons && r.reasons.length ? ' · ' + esc(r.reasons.join(', ')) : '') + '</p><p class="adm-comment">' + esc(r.comment) + '</p></li>';
        }).join('') + '</ul>'
      : '<p class="empty">No comments yet.</p>';
    page('<nav class="breadcrumb" aria-label="Breadcrumb"><a href="#/">Home</a> › <a href="#/account">Account</a> › Feedback</nav>' +
      '<h1>Feedback on proposed audits</h1>' +
      '<p class="page-intro">Counts include each person’s latest rating of an audit. The CSV contains every response received.</p>' +
      '<ul class="stats"><li class="stat"><b>' + fmt(rows.length) + '</b><span>responses</span></li>' +
      '<li class="stat"><b>' + fmt(per.length) + '</b><span>audits rated</span></li>' +
      '<li class="stat"><b>' + fmt(up) + '</b><span>useful</span></li>' +
      '<li class="stat"><b>' + fmt(down) + '</b><span>not useful</span></li>' +
      '<li class="stat"><b>' + fmt(withComments.length) + '</b><span>comments</span></li></ul>' +
      '<div class="toolbar"><button class="btn" type="button" data-fb-csv' + (rows.length ? '' : ' disabled') + '>' +
      '<svg width="18" height="18" viewBox="0 0 24 24" aria-hidden="true"><path d="M12 4v11m0 0l-4.5-4.5M12 15l4.5-4.5M5 19h14" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/></svg>Download CSV</button></div>' +
      '<h2>By audit</h2>' + table +
      '<h2 class="adm-h2">Comments</h2>' + comments,
      'Feedback on proposed audits', '');
  }
  function csvCell(v) {
    var s = String(v == null ? '' : v);
    if (/^[=+\-@\t\r]/.test(s)) s = "'" + s;           // stop spreadsheets treating text as a formula
    return /[",\r\n]/.test(s) ? '"' + s.replace(/"/g, '""') + '"' : s;
  }
  function downloadFeedbackCsv() {
    var rows = BE.adminRows || [];
    var head = ['id', 'audit_id', 'question', 'rating', 'reasons', 'comment', 'created_at', 'signed_in', 'app_version'];
    var lines = [head.join(',')].concat(rows.map(function (r) {
      var p = S.pById.get(r.audit_id);
      return [r.id, r.audit_id, p ? p.question : '', r.rating, (r.reasons || []).join('; '), r.comment, r.created_at,
        r.user_id ? 'yes' : 'no', r.app_version || ''].map(csvCell).join(',');
    }));
    saveCsv(lines, 'ai4qi-feedback-' + new Date().toISOString().slice(0, 10) + '.csv');
  }

  /* ---------- usage counting (optional backend): activity, "My audits" tracker, admin statistics ---------- */
  function recordActivity() {
    var uid = BE.user && BE.user.id;
    if (!uid || BE.activityFor === uid) return;
    BE.activityFor = uid;   // one call per visit; the server keeps at most one active day per user per day
    sbClient().then(function (c) { return c.rpc('record_activity'); })
      .then(function (res) { if (res && res.error) BE.activityFor = null; }, function () { BE.activityFor = null; });
  }
  function stepIndex(status) {
    for (var i = 0; i < STEPS.length; i++) if (STEPS[i][0] === status) return i;
    return 0;
  }
  function loadTracker(c) {
    if (BE.tracker) return Promise.resolve(BE.tracker);
    return c.from('my_audits').select('audit_id, status, started_at, updated_at, completed_at')
      .order('started_at', { ascending: false })
      .then(function (res) {
        if (res.error) throw res.error;
        BE.tracker = new Map();
        (res.data || []).forEach(function (r) { BE.tracker.set(r.audit_id, r); });
        return BE.tracker;
      });
  }
  function stepsHtml(id, status) {
    var cur = stepIndex(status);
    return '<ol class="steps" aria-label="Progress of ' + attr(id) + '">' + STEPS.map(function (s, i) {
      return '<li><button type="button" class="step' + (i <= cur ? ' is-done' : '') + '" data-track-step="' + s[0] + '" data-track-id="' + attr(id) + '" aria-pressed="' + (i === cur) + '">' +
        '<span class="step-n" aria-hidden="true">' + (i < cur || (i === cur && status === 'closed') ? '✓' : i + 1) + '</span>' + esc(s[1]) + '</button></li>';
    }).join('') + '</ol>';
  }
  function trackerHtml(id, row) {
    if (!BE.user) {
      return '<p class="track-note"><a href="#/account">Sign in</a> to keep track of this audit in My audits.</p>';
    }
    if (!row) {
      return '<div class="track-start"><button class="btn btn-secondary" type="button" data-track-start="' + attr(id) + '">Start this audit</button>' +
        '<span class="muted">Adds it to <a href="#/my-audits">My audits</a>, where you can record each step.</span></div>';
    }
    return '<p class="track-head">You are running this audit. Choose the step you have reached. <a href="#/my-audits">My audits</a></p>' + stepsHtml(id, row.status) +
      '<p class="track-status" role="status" aria-live="polite" data-track-status></p>';
  }
  function showTracker(id) {
    configReady.then(function () {
      if (!BE.url) return null;
      if (!BE.user && !hasStoredSession()) return 'signed-out';
      return sbClient().then(function (c) { return BE.user ? loadTracker(c) : 'signed-out'; });
    }).then(function (m) {
      var el = main.querySelector('[data-track="' + id + '"]');
      if (!el || m === null || m === undefined) return;
      el.innerHTML = trackerHtml(id, m === 'signed-out' ? null : m.get(id));
      el.hidden = false;
    }).catch(function () {});
  }
  function refreshTrackers(id, focusSel) {
    var row = BE.tracker && BE.tracker.get(id);
    main.querySelectorAll('[data-track="' + id + '"]').forEach(function (el) { el.innerHTML = trackerHtml(id, row); });
    main.querySelectorAll('[data-my-steps="' + id + '"]').forEach(function (el) { el.innerHTML = row ? stepsHtml(id, row.status) : ''; });
    if (focusSel) { var f = main.querySelector(focusSel); if (f) f.focus(); }
  }
  function trackSay(id, text) {
    var st = main.querySelector('[data-track="' + id + '"] [data-track-status], [data-my-item="' + id + '"] [data-track-status]');
    if (st) st.textContent = text;
  }
  document.addEventListener('click', function (e) {
    var start = e.target.closest('[data-track-start]'), step = e.target.closest('[data-track-step]'), rm = e.target.closest('[data-track-remove]');
    if (!start && !step && !rm) return;
    var id = (start && start.getAttribute('data-track-start')) || (step && step.getAttribute('data-track-id')) || rm.getAttribute('data-track-remove');
    var btn = start || step || rm;
    if (rm && !window.confirm('Remove ' + id + ' from My audits? Your recorded progress will be deleted.')) return;
    btn.disabled = true;
    sbClient().then(function (c) {
      if (!BE.user) throw new Error('signed out');
      return loadTracker(c).then(function () {
        if (start) return c.from('my_audits').insert({ user_id: BE.user.id, audit_id: id }).select('audit_id, status, started_at, updated_at, completed_at').single();
        if (rm) return c.from('my_audits').delete().eq('user_id', BE.user.id).eq('audit_id', id);
        return c.from('my_audits').update({ status: step.getAttribute('data-track-step') }).eq('user_id', BE.user.id).eq('audit_id', id)
          .select('audit_id, status, started_at, updated_at, completed_at').single();
      });
    }).then(function (res) {
      if (res.error && !(start && String(res.error.code) === '23505')) throw res.error;
      if (rm) {
        BE.tracker.delete(id);
        var item = main.querySelector('[data-my-item="' + id + '"]');
        if (item) { item.remove(); var h = main.querySelector('h1'); if (h) h.focus(); }
        if (!main.querySelector('[data-my-item]') && stillOn('my-audits')) renderMyAudits();
        return;
      }
      if (res.data) BE.tracker.set(id, res.data);
      else BE.tracker = null;   // already started elsewhere: reload next time
      var st = res.data ? res.data.status : 'started';
      refreshTrackers(id, start ? '[data-track="' + id + '"] [data-track-step="started"]' : '[data-track-id="' + id + '"][data-track-step="' + st + '"]');
      trackSay(id, st === 'closed' ? 'Loop closed – well done. This audit now counts as completed.' : 'Progress saved: ' + STEPS[stepIndex(st)][1] + '.');
    }).catch(function () {
      btn.disabled = false;
      trackSay(id, 'Your progress could not be saved. Please check your connection and try again.');
      if (start) btn.textContent = 'Start this audit – try again';
    });
  });

  function renderMyAudits() {
    page(loadingHtml('Loading your audits…'), 'My audits', 'account');
    sbClient().then(function (c) { return BE.user ? loadTracker(c) : null; }).then(function (m) {
      if (!stillOn('my-audits')) return;
      if (!BE.user) { renderSignIn('Sign in to see your audits'); focusMain(); return; }
      var rows = Array.from(m.values());
      var done = rows.filter(function (r) { return r.status === 'closed'; }).length;
      var list = rows.length
        ? '<ul class="my-list">' + rows.map(function (r) {
            var p = S.pById.get(r.audit_id);
            return '<li class="my-item" data-my-item="' + attr(r.audit_id) + '">' +
              '<p class="my-title"><a href="#/proposed/' + encodeURIComponent(r.audit_id) + '">' + esc(p ? p.question : r.audit_id) + '</a> <span class="id-tag">' + esc(r.audit_id) + '</span></p>' +
              '<p class="meta muted">Started ' + esc(dateGB(r.started_at)) + (r.completed_at ? ' · loop closed ' + esc(dateGB(r.completed_at)) : '') + '</p>' +
              '<div data-my-steps="' + attr(r.audit_id) + '">' + stepsHtml(r.audit_id, r.status) + '</div>' +
              '<p class="track-status" role="status" aria-live="polite" data-track-status></p>' +
              '<button class="link-btn" type="button" data-track-remove="' + attr(r.audit_id) + '">Remove from My audits</button></li>';
          }).join('') + '</ul>'
        : '<p class="empty">You have not started any audits yet. Open a <a href="#/proposed">proposed audit</a> and choose <strong>Start this audit</strong>.</p>';
      page('<nav class="breadcrumb" aria-label="Breadcrumb"><a href="#/">Home</a> › <a href="#/account">Account</a> › My audits</nav>' +
        '<div class="doc"><h1>My audits</h1>' +
        '<p class="page-intro">Record how far each audit has got. Only the step and dates are stored – never notes or patient information.</p>' +
        (rows.length ? '<p>' + fmt(rows.length) + (rows.length === 1 ? ' audit' : ' audits') + ' · ' + fmt(done) + ' with the loop closed</p>' : '') +
        list + '</div>', 'My audits', 'account');
      focusMain();
    }, function () { if (stillOn('my-audits')) unavailable('My audits', 'account'); });
  }

  /* admin: usage statistics (#/admin/stats) */
  var STAT_COLS = [['signups', 'Sign-ups'], ['cumulative_users', 'Registered users'], ['audits_started', 'Audits started'],
    ['audits_completed', 'Audits completed'], ['active_users', 'Active users'], ['returning_users', 'Returning users']];
  function isoDate(d) { return d.toISOString().slice(0, 10); }
  function renderStats() {
    var params = parseHash().params, today = new Date();
    var okDate = function (v) { return /^\d{4}-\d{2}-\d{2}$/.test(v || '') && !isNaN(new Date(v)) ? v : ''; };
    var end = okDate(params.get('end')) || isoDate(today);
    var start = okDate(params.get('start')) || isoDate(new Date(today.getTime() - 83 * 864e5));   // 12 weeks
    if (start > end) { var t = start; start = end; end = t; }
    page(loadingHtml('Loading statistics…'), 'Usage statistics', '');
    sbClient().then(function (c) {
      if (!BE.user) return null;
      return checkAdmin(c).then(function (isAdmin) {
        if (!isAdmin) return null;
        return Promise.all([c.rpc('weekly_stats', { p_from: start, p_to: end }), c.rpc('signup_breakdown', { p_from: start, p_to: end })]);
      });
    }).then(function (r) {
      if (!stillOn('admin')) return;
      if (!BE.user) { renderSignIn(); focusMain(); return; }
      if (!r) { renderNotFound(); focusMain(); return; }
      if (r[0].error || r[1].error) throw r[0].error || r[1].error;
      BE.stats = { start: start, end: end, weeks: r[0].data || [], breakdown: r[1].data || [] };
      renderStatsPage(BE.stats);
      focusMain();
    }).catch(function () { if (stillOn('admin')) unavailable('Usage statistics', ''); });
  }
  function miniChart(title, weeks, key) {
    var W = 320, H = 132, top = 16, base = 112, n = Math.max(weeks.length, 1);
    var vals = weeks.map(function (w) { return Number(w[key]) || 0; });
    var max = Math.max.apply(null, vals.concat([1])), total = vals.reduce(function (a, b) { return a + b; }, 0);
    var slot = W / n, bw = Math.max(2, Math.min(28, slot - 2));
    var bars = weeks.map(function (w, i) {
      var v = vals[i], h = v ? Math.max(2, (v / max) * (base - top)) : 0, x = i * slot + (slot - bw) / 2;
      return '<g><title>' + esc(w.iso_week + ': ' + fmt(v)) + '</title>' +
        '<rect class="hit" x="' + (i * slot).toFixed(1) + '" y="0" width="' + slot.toFixed(1) + '" height="' + base + '"></rect>' +
        (h ? '<rect class="bar" x="' + x.toFixed(1) + '" y="' + (base - h).toFixed(1) + '" width="' + bw.toFixed(1) + '" height="' + h.toFixed(1) + '" rx="2"></rect>' : '') + '</g>';
    }).join('');
    var lab = weeks.length ? '<text x="0" y="128">' + esc(weeks[0].iso_week) + '</text>' +
      (weeks.length > 1 ? '<text x="' + W + '" y="128" text-anchor="end">' + esc(weeks[weeks.length - 1].iso_week) + '</text>' : '') : '';
    return '<figure class="mini-chart"><figcaption><strong>' + esc(title) + '</strong> <span class="muted">' + fmt(total) + ' in period · peak ' + fmt(max === 1 && !total ? 0 : max) + ' a week</span></figcaption>' +
      '<svg viewBox="0 0 ' + W + ' ' + H + '" role="img" aria-label="' + attr(title + ' per week: ' + weeks.map(function (w, i) { return w.iso_week + ' ' + vals[i]; }).join(', ')) + '">' +
      '<line class="axis" x1="0" x2="' + W + '" y1="' + base + '" y2="' + base + '"></line>' + bars + lab + '</svg></figure>';
  }
  function breakdownTable(rows, dim, title) {
    var r = rows.filter(function (x) { return x.dimension === dim; });
    return '<div class="bd"><h3>' + esc(title) + '</h3>' + (r.length
      ? '<div class="table-wrap"><table class="adm-table"><thead><tr><th scope="col">' + esc(title) + '</th><th scope="col" class="num-col">Sign-ups</th></tr></thead><tbody>' +
        r.map(function (x) { return '<tr><td>' + esc(x.value) + '</td><td class="num-col">' + (x.users == null ? '<span class="muted">fewer than 5</span>' : fmt(x.users)) + '</td></tr>'; }).join('') +
        '</tbody></table></div>'
      : '<p class="empty">No sign-ups in this period.</p>') + '</div>';
  }
  function analyticsLinkHtml() {
    var cfg = BE.cfg || {};
    if (cfg.analytics === 'plausible' && cfg.plausible_domain) {
      var host = /^https:\/\/[a-z0-9.-]+$/i.test(cfg.plausible_host || '') ? cfg.plausible_host : 'https://plausible.io';
      return '<p class="muted">Visitor numbers and traffic sources (including links tagged <code>?from=</code>) are in the <a href="' + attr(host + '/' + encodeURIComponent(cfg.plausible_domain)) + '" target="_blank" rel="noopener">Plausible dashboard</a>.</p>';
    }
    if (cfg.analytics === 'cloudflare') return '<p class="muted">Visitor numbers are in <a href="https://dash.cloudflare.com/?to=/:account/web-analytics" target="_blank" rel="noopener">Cloudflare Web Analytics</a>.</p>';
    return '<p class="muted">Visitor numbers will appear in Plausible or Cloudflare Web Analytics once one is set up in config.json.</p>';
  }
  function renderStatsPage(st) {
    var w = st.weeks, last = w[w.length - 1] || {};
    var sum = function (k) { return w.reduce(function (a, r) { return a + (Number(r[k]) || 0); }, 0); };
    var table = w.length
      ? '<div class="table-wrap"><table class="adm-table"><caption class="visually-hidden">Weekly usage</caption><thead><tr><th scope="col">Week</th>' +
        STAT_COLS.map(function (c) { return '<th scope="col" class="num-col">' + esc(c[1]) + '</th>'; }).join('') + '</tr></thead><tbody>' +
        w.slice().reverse().map(function (r) {
          return '<tr><td class="wk">' + esc(r.iso_week) + '<span class="muted">from ' + esc(dateGB(r.week_start)) + '</span></td>' +
            STAT_COLS.map(function (c) { return '<td class="num-col">' + fmt(r[c[0]] || 0) + '</td>'; }).join('') + '</tr>';
        }).join('') + '</tbody></table></div>'
      : '<p class="empty">No activity in this period.</p>';
    page('<nav class="breadcrumb" aria-label="Breadcrumb"><a href="#/">Home</a> › <a href="#/account">Account</a> › Usage statistics</nav>' +
      '<h1>Usage statistics</h1>' +
      '<p class="page-intro">Signed-in use of the library, per ISO week (Monday to Sunday, in UTC). A returning user is someone active in the week who was also active in an earlier week. An audit counts as completed when its loop is closed.</p>' +
      '<form class="range-form" data-stats-range><div><label for="st-start">From</label><input id="st-start" name="start" type="date" value="' + attr(st.start) + '" required></div>' +
      '<div><label for="st-end">To</label><input id="st-end" name="end" type="date" value="' + attr(st.end) + '" required></div>' +
      '<button class="btn btn-secondary" type="submit">Show</button>' +
      '<button class="btn" type="button" data-stats-csv><svg width="18" height="18" viewBox="0 0 24 24" aria-hidden="true"><path d="M12 4v11m0 0l-4.5-4.5M12 15l4.5-4.5M5 19h14" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/></svg>Download CSV</button></form>' +
      '<ul class="stats"><li class="stat"><b>' + fmt(sum('signups')) + '</b><span>sign-ups in period</span></li>' +
      '<li class="stat"><b>' + fmt(last.cumulative_users || 0) + '</b><span>registered users</span></li>' +
      '<li class="stat"><b>' + fmt(sum('audits_started')) + '</b><span>audits started</span></li>' +
      '<li class="stat"><b>' + fmt(sum('audits_completed')) + '</b><span>audits completed</span></li></ul>' +
      '<div class="chart-grid">' + miniChart('Sign-ups', w, 'signups') + miniChart('Audits started', w, 'audits_started') +
      miniChart('Audits completed', w, 'audits_completed') + miniChart('Returning users', w, 'returning_users') + '</div>' +
      '<h2 class="adm-h2">By week</h2>' + table +
      '<h2 class="adm-h2">Who signed up in this period</h2>' +
      '<p class="muted">From the optional profile. Groups with fewer than five people are not shown, so that nobody can be identified.</p>' +
      '<div class="bd-grid">' + breakdownTable(st.breakdown, 'grade', 'Grade or role') + breakdownTable(st.breakdown, 'specialty', 'Specialty') +
      breakdownTable(st.breakdown, 'region', 'Region') + '</div>' +
      '<h2 class="adm-h2">Website visitors</h2>' + analyticsLinkHtml(),
      'Usage statistics', '');
  }
  document.addEventListener('submit', function (e) {
    var f = e.target.closest('[data-stats-range]');
    if (!f) return;
    e.preventDefault();
    location.hash = '#/admin/stats?start=' + encodeURIComponent(f.start.value) + '&end=' + encodeURIComponent(f.end.value);
  });
  document.addEventListener('click', function (e) {
    if (!e.target.closest('[data-stats-csv]') || !BE.stats) return;
    var st = BE.stats, lines = ['Ai4Qi usage statistics,' + st.start + ' to ' + st.end, '',
      ['iso_week', 'week_start'].concat(STAT_COLS.map(function (c) { return c[0]; })).join(',')];
    st.weeks.forEach(function (r) { lines.push([r.iso_week, r.week_start].concat(STAT_COLS.map(function (c) { return r[c[0]] || 0; })).map(csvCell).join(',')); });
    lines.push('', 'breakdown,value,signups (blank = fewer than 5)');
    st.breakdown.forEach(function (r) { lines.push([r.dimension, r.value, r.users == null ? '' : r.users].map(csvCell).join(',')); });
    saveCsv(lines, 'ai4qi-usage-' + st.start + '-to-' + st.end + '.csv');
  });
  function saveCsv(lines, name) {
    var blob = new Blob(['﻿' + lines.join('\r\n') + '\r\n'], { type: 'text/csv;charset=utf-8' });
    var a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = name;
    document.body.appendChild(a); a.click(); a.remove();
    setTimeout(function () { URL.revokeObjectURL(a.href); }, 1000);
  }

  /* ---------- privacy-friendly visitor analytics (optional) ----------
     config.json "analytics": "plausible" (with "plausible_script", the per-site script URL from the
     Plausible snippet) or "cloudflare" (with "cloudflare_token"). No cookies, nothing stored on the
     device. Page addresses are cleaned before they are sent: search words and filters are removed,
     only the route (e.g. #/proposed/ONA-012) and utm_*, ref and from parameters are kept. */
  var FROM = (function () {
    try {
      var v = new URLSearchParams(location.search).get('from') || '';
      return /^[A-Za-z0-9_.-]{1,40}$/.test(v) ? v.toLowerCase() : '';
    } catch (e) { return ''; }
  })();
  var AN = { kind: '', last: '' };
  var KNOWN_ROUTES = ['search', 'proposed', 'audit', 'topic', 'topics', 'standards', 'account', 'my-audits', 'admin'];
  function cleanPageUrl() {
    var keep = new URLSearchParams();
    new URLSearchParams(location.search).forEach(function (v, k) {
      if (/^(utm_(source|medium|campaign|term|content)|ref|from)$/.test(k)) keep.set(k, v.slice(0, 100));
    });
    var parts = parseHash().parts, name = parts[0] || '', path = [];
    if (name && KNOWN_ROUTES.indexOf(name) === -1) path = ['not-found'];
    else if (name === 'proposed' && parts[1]) path = [name, S.pById.has(parts[1]) ? parts[1] : 'not-found'];
    else if ((name === 'audit' || name === 'topic' || name === 'admin') && parts[1]) path = [name, parts[1].slice(0, 80)];
    else if (name) path = [name];   // search words and filters are never sent
    var qs = keep.toString();
    return location.origin + location.pathname + (qs ? '?' + qs : '') + '#/' + path.map(encodeURIComponent).join('/');
  }
  function trackPageview() {
    if (AN.kind !== 'plausible' || typeof window.plausible !== 'function') return;
    var u = cleanPageUrl();
    if (u === AN.last) return;
    AN.last = u;
    window.plausible('pageview', { url: u });
  }
  function initAnalytics(c) {
    var kind = String(c.analytics || '').toLowerCase(), sc;
    if (kind === 'plausible') {
      var src = String(c.plausible_script || '').trim();
      if (!/^https:\/\/[a-z0-9.-]+(:\d+)?\/[^\s"'<>]+\.js$/i.test(src)) return;
      // Queue stub from Plausible's snippet: calls made before the script loads are kept.
      window.plausible = window.plausible || function () { (window.plausible.q = window.plausible.q || []).push(arguments); };
      window.plausible.init = window.plausible.init || function (i) { window.plausible.o = i || {}; };
      window.plausible.init({
        hashBasedRouting: true,
        autoCapturePageviews: false,       // sent from the router with a cleaned address
        customProperties: function () { return FROM ? { from: FROM } : {}; }
      });
      sc = document.createElement('script');
      sc.async = true; sc.src = src; sc.setAttribute('data-analytics', 'plausible');
      document.head.appendChild(sc);
      AN.kind = 'plausible';
      if (S.lib.length) trackPageview();
    } else if (kind === 'cloudflare') {
      var token = String(c.cloudflare_token || '').trim();
      if (!/^[A-Za-z0-9]{16,64}$/.test(token)) return;
      sc = document.createElement('script');
      sc.defer = true; sc.src = 'https://static.cloudflareinsights.com/beacon.min.js';
      // spa: false – one visit per page load; in-app routes (which can contain search words) are not reported.
      sc.setAttribute('data-cf-beacon', JSON.stringify({ token: token, spa: false }));
      sc.setAttribute('data-analytics', 'cloudflare');
      document.head.appendChild(sc);
      AN.kind = 'cloudflare';
    }
  }

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
