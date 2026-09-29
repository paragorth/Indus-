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
    var ab = document.querySelector('[data-acct-btn]');
    if (ab) ab.classList.toggle('is-current', name === 'my-audits' || name === 'account');
  }
  function page(html, title, nav, keepScroll) {
    main.innerHTML = (DEMO ? demoBar() : '') + html;
    var dt = document.querySelector('[data-demo-toggle]');
    if (dt) { dt.textContent = DEMO ? 'Turn off demo mode' : 'Demo mode'; dt.setAttribute('href', DEMO ? '#/demo/off' : '#/demo'); }
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
      if (name === 'demo') { demoSet(p[1] !== 'off'); location.replace('#/' + (p[1] === 'off' ? '' : 'suggest')); return; }
      if (name === '') renderHome();
      else if (name === 'search') renderSearch(r.params, sameView);
      else if (name === 'build') renderBuild(r.params);
      else if (name === 'suggest') renderSuggest(r.params);
      else if (name === 'proposed' && p[1]) renderProposed(p[1]);
      else if (name === 'proposed') renderProposedList(r.params);
      else if (name === 'audit' && p[1]) renderAudit(+p[1]);
      else if (name === 'topic' && p[1]) renderTopic(p[1]);
      else if (name === 'topics') renderTopics();
      else if (name === 'standards') renderStandards(r.params);
      else if (name === 'account' && BE.url) renderAccount();
      else if (name === 'admin' && p[1] === 'feedback' && BE.url) renderAdmin();
      else if (name === 'admin' && p[1] === 'stats' && BE.url) renderStats();
      else if (name === 'my-audits') renderRuns();
      else if (name === 'run' && p[1]) renderRun(p[1]);
      else if (name === 'privacy') renderPrivacy();
      else if (LEGAL_PAGES.indexOf(name) !== -1) renderLegal(name);
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
    var topGroups = S.groups.slice().sort(function (a, b) { return b[1] - a[1]; }).slice(0, 24)
      .sort(function (a, b) { return a[0].localeCompare(b[0]); });
    var areas = new Map();
    S.proposed.forEach(function (p) { areas.set(p.area, (areas.get(p.area) || 0) + 1); });
    var areaList = Array.from(areas.entries()).sort(function (a, b) { return a[0].localeCompare(b[0]); });
    var examples = ['sepsis antibiotics within one hour', 'VTE risk assessment', 'delirium screening', 'reusable PPE in theatre'];
    function index(list, href) {
      return '<ul class="index-list">' + list.map(function (g) {
        return '<li><a href="' + href(g[0]) + '"><span class="il-name">' + esc(g[0]) + '</span><span class="il-dots" aria-hidden="true"></span><span class="il-n">' + fmt(g[1]) + '</span></a></li>';
      }).join('') + '</ul>';
    }
    page(
      '<section class="hero" aria-labelledby="hero-h">' +
      '<p class="kicker">Clinical audit for UK and Irish clinicians</p>' +
      '<h1 id="hero-h">Build a clinical audit on any topic.</h1>' +
      '<p class="lede">A complete protocol in a minute: one measurable question, the standard quoted word for word, a data sheet, timeline, change and re-audit, backed by audits that closed the loop.</p>' +
      buildForm('', true) +
      '<p class="examples"><span>For example</span>' + examples.map(function (e) { return '<a href="#/build?q=' + encodeURIComponent(e) + '">' + esc(e) + '</a>'; }).join('<span class="sep" aria-hidden="true">·</span>') + '</p>' +
      '</section>' +
      '<dl class="facts">' +
      '<div><dt>Published audits</dt><dd>' + fmt(S.lib.length) + '</dd></div>' +
      '<div><dt>Closed the loop</dt><dd>' + fmt(closed) + '</dd></div>' +
      '<div><dt>Ready-made protocols</dt><dd>' + fmt(S.proposed.length) + '</dd></div>' +
      '<div><dt>Standards quoted</dt><dd>' + fmt(S.standards.length) + '</dd></div></dl>' +
      '<div class="home-cols"><section><div class="section-head"><h2>Ready-made protocols</h2><a href="#/proposed">All ' + fmt(S.proposed.length) + '</a></div>' +
      index(areaList, function (n) { return '#/proposed?area=' + encodeURIComponent(n); }) + '</section>' +
      '<section><div class="section-head"><h2>Published audits by specialty</h2><a href="#/search">Search all</a></div>' +
      index(topGroups, function (n) { return '#/search?sp=' + encodeURIComponent(n); }) + '</section></div>',
      '', 'build');
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
    if (st.q && isThemed(st.q)) {
      propHtml = '<div class="build-cta"><div><h2>Build a complete audit on “' + esc(st.q) + '”</h2>' +
        '<p>Question, exact standard, template, timeline, change, re-audit, pitfalls and pearls, with the published audits below as evidence.</p></div>' +
        '<a class="btn" href="#/build?q=' + encodeURIComponent(st.q) + '">Build this audit</a></div>';
    } else if (st.q || st.sp) {
      var pv = res.props.slice(0, S.view.propShown);
      propHtml = '<div class="results-title"><h2 id="prop-h">Suggested audits (ready to run)</h2><span class="count">' + fmt(res.props.length) + ' found</span></div>' +
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
    var nr = function (v) { return !v || /^not reported/i.test(v); };
    var snip = !nr(a.r2) && !nr(a.f) ? (trunc(a.f, 120) + ' → ' + trunc(a.r2, 120)) : trunc(nr(a.f) ? a.r2 : a.f, 230);
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
  function cellHint(f) {
    var t = String(f.type || '').toLowerCase();
    if (f.options && f.options.length) return '▾ ' + f.options.slice(0, 3).join(' / ') + (f.options.length > 3 ? ' …' : '');
    if (/yes/.test(t)) return '▾ yes / no';
    if (/cycle/.test(t)) return '▾ Cycle 1 / Re-audit';
    if (/datetime/.test(t)) return 'dd/mm/yyyy hh:mm';
    if (/date/.test(t)) return 'dd/mm/yyyy';
    if (/time/.test(t)) return 'hh:mm';
    if (/num|min|hour|score|count/.test(t)) return '123';
    if (/code|pseud|hospital_number/i.test(f.field)) return 'P001';
    return 'text';
  }
  function colName(i) { var n = i + 1, out = ''; while (n > 0) { var m = (n - 1) % 26; out = String.fromCharCode(65 + m) + out; n = Math.floor((n - 1) / 26); } return out; }
  /* The template as it looks in the Excel data sheet: column letters, row numbers, headings, an example row. */
  function templateTable(p) {
    var t = (p.template || []).filter(function (f) { return f && f.field; });
    var sheet = '<div class="sheet-wrap" role="region" tabindex="0" aria-label="Data collection sheet, ' + t.length + ' columns"><table class="sheet">' +
      '<caption class="visually-hidden">Data collection sheet for ' + esc(p.id) + ': one row per patient</caption>' +
      '<thead><tr><th class="sh-corner" aria-hidden="true"></th>' + t.map(function (f, i) { return '<th class="sh-col" aria-hidden="true">' + colName(i) + '</th>'; }).join('') + '</tr></thead><tbody>' +
      '<tr><th class="sh-row" aria-hidden="true">1</th>' + t.map(function (f) { return '<th scope="col" class="sh-head" title="' + attr(f.note || '') + '">' + esc(fieldLabel(f.field)) + '</th>'; }).join('') + '</tr>' +
      '<tr><th class="sh-row" aria-hidden="true">2</th>' + t.map(function (f) { return '<td class="sh-hint">' + esc(cellHint(f)) + '</td>'; }).join('') + '</tr>' +
      [3, 4, 5].map(function (n) { return '<tr aria-hidden="true"><th class="sh-row">' + n + '</th>' + t.map(function () { return '<td></td>'; }).join('') + '</tr>'; }).join('') +
      '</tbody></table></div><p class="sheet-cap">One row per patient. ▾ marks a drop-down list in the Excel sheet.</p>';
    return sheet + '<details class="field-notes"><summary>Column details (' + t.length + ')</summary>' + fieldTable(p) + '</details>';
  }
  function fieldTable(p) {
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

  function copyBtn(id) {
    return '<button class="btn" type="button" data-copy-csv="' + attr(id) + '">' +
      '<svg width="18" height="18" viewBox="0 0 24 24" aria-hidden="true"><rect x="8" y="8" width="12" height="12" rx="2" fill="none" stroke="currentColor" stroke-width="2"/><path d="M16 8V5a1 1 0 0 0-1-1H5a1 1 0 0 0-1 1v10a1 1 0 0 0 1 1h3" fill="none" stroke="currentColor" stroke-width="2"/></svg>Copy template (CSV)</button><span class="copy-status" role="status" data-copy-status></span>';
  }
  /* The protocol layout shared by proposed audits and audits built on request. */
  function protocolBody(p, dl, extra) {
    var st = p.standard || {}, url = safeUrl(st.url), n = 0;
    var how =
      sub('Standard', (st.wording ? '<blockquote class="standard"><p>“' + esc(st.wording) + '”</p></blockquote>' : '<p class="prose">Read the exact recommendation at the source below.</p>') +
        '<p class="standard-source">' + esc(st.source) + (url ? ' · <a class="print-url" href="' + attr(url) + '" target="_blank" rel="noopener">Read the standard</a>' : '') + '</p>' + niceAttribution(st)) +
      sub('Pass', '<p class="prose">' + linkify(p.pass) + '</p>') +
      sub('Population', '<p class="prose">' + linkify(p.population) + '</p>') +
      sub('Sample', '<p class="prose">' + linkify(p.sample) + '</p>') +
      sub('Data source', '<p class="prose">' + linkify(p.data_source) + '</p>') +
      sub('Template', templateTable(p) + '<div class="no-print">' + dl + '</div>') +
      sub('Timeline', timelineHtml(p.timeline));
    extra = extra || {};
    return (extra.before ? sec(++n, extra.before[0], extra.before[1]) : '') +
      sec(++n, 'Why', '<p class="prose">' + linkify(p.why) + '</p>') +
      sec(++n, 'How', how) +
      sec(++n, 'Change', '<p class="prose">' + linkify(p.change) + '</p>') +
      sec(++n, 'Re-audit and target', targetHtml(p.target) + '<p class="prose">' + linkify(p.reaudit) + '</p>') +
      sec(++n, 'Close the loop', '<p class="prose">' + linkify(p.close_loop) + '</p>') +
      (p.evidence && p.evidence.length ? sec(++n, 'Evidence', '<ul class="ev-list">' + p.evidence.map(function (e) { return '<li>' + linkify(e) + '</li>'; }).join('') + '</ul>') : '') +
      (p.pitfalls && p.pitfalls.length ? sec(++n, 'Pitfalls', '<ul class="pit-list">' + p.pitfalls.map(pitfallItem).join('') + '</ul>') : '') +
      (p.pearls && p.pearls.length ? sec(++n, 'Pearls', '<ul class="pearl-list">' + p.pearls.map(function (e) { return '<li>' + linkify(e) + '</li>'; }).join('') + '</ul>') : '') +
      (extra.after ? extra.after.map(function (x) { return sec(++n, x[0], x[1]); }).join('') : '');
  }

  /* ---------- send a proposal to a supervisor: email text + Word and PowerPoint files ---------- */
  var ME_KEY = 'ai4qi_me_v1';
  function meGet() { try { return JSON.parse(localStorage.getItem(ME_KEY) || '{}') || {}; } catch (e) { return {}; } }
  function meSet(o) { try { localStorage.setItem(ME_KEY, JSON.stringify(o)); } catch (e) {} }
  var SUP_FIELDS = [['lead', 'Your name', 'text'], ['role', 'Your role or grade', 'text'], ['email', 'Your email', 'email'],
    ['supervisor', 'Supervisor\'s name', 'text'], ['supEmail', 'Supervisor\'s email', 'email'], ['site', 'Hospital or practice', 'text'], ['department', 'Department or ward', 'text']];
  function supWho(box) {
    var o = {}, f = box.querySelector('[data-sup-form]');
    SUP_FIELDS.forEach(function (x) { o[x[0]] = (f.elements[x[0]].value || '').trim(); });
    var r = box.getAttribute('data-sup-run') && S.runs.get(box.getAttribute('data-sup-run'));
    if (r) { o.title = r.details.title; o.startDate = r.details.startDate; o.sampleSize = r.details.sampleSize; o.team = r.details.team; }
    return o;
  }
  function supProtocol(box) {
    var r = box.getAttribute('data-sup-run') && S.runs.get(box.getAttribute('data-sup-run'));
    return r ? r.protocol : anyAudit(box.getAttribute('data-sup'));
  }
  function firstSent(t) { var m = String(t || '').match(/^[\s\S]*?[.!?](\s|$)/); return (m ? m[0] : String(t || '')).trim(); }
  function supEmail(p, d) {
    var st = p.standard || {}, sup = d.supervisor ? d.supervisor.trim() : '', place = [d.department, d.site].filter(Boolean).join(', ');
    var wording = String(st.wording || '').replace(/\s+/g, ' ').trim(), ww = wording.split(' ');
    if (ww.length > 45) wording = ww.slice(0, 42).join(' ').replace(/[.,;:]+$/, '') + '…';
    var subject = 'Audit proposal for your approval: ' + trunc(d.title || p.question, 80);
    var lines = [
      'Dear ' + (sup || 'Dr [name]') + ',', '',
      'I would like to run a clinical audit' + (place ? ' in ' + place : '') + ' and would be grateful if you would supervise it.', '',
      'Question: ' + p.question,
      'Standard: ' + (st.source || 'local standard') + (wording ? ' – "' + wording + '"' : ''),
      p.target ? 'Target: ' + p.target : '',
      'Sample: ' + (d.sampleSize ? d.sampleSize + ' records per cycle. ' : '') + (p.sample || ''),
      p.timeline ? 'Timeline: ' + p.timeline : '',
      p.change ? 'If we fall short: ' + firstSent(p.change) : '', '',
      isNice(st) && wording ? 'NICE wording © NICE, reproduced under the NICE UK Open Content Licence; check the current version at ' + (st.url || 'www.nice.org.uk') + '.' : '',
      '', 'I have attached a short proposal (Word) and slides (PowerPoint) with the full protocol, the data collection sheet and the timeline. They contain no patient data.', '',
      'Could you let me know whether you are happy to supervise it, and whether you would change the standard, target or sample? Once it is agreed I will register it with the clinical audit department.', '',
      'Many thanks,', d.lead || '[Your name]', d.role || '', d.email || ''];
    return { subject: subject, body: lines.filter(function (l, i, a) { return l !== '' || a[i - 1] !== ''; }).join('\n').replace(/\n+$/, '') };
  }
  function supervisorBox(p, run) {
    var me = meGet(), d = run ? run.details : {};
    var val = { lead: d.lead || me.lead, role: me.role, email: me.email, supervisor: d.supervisor || me.supervisor, supEmail: me.supEmail, site: d.site || me.site, department: d.department || me.department };
    return '<details class="sup-box no-print" id="send-supervisor" data-sup="' + attr(p.id) + '"' + (run ? ' data-sup-run="' + attr(run.id) + '"' : '') + '>' +
      '<summary><span class="sup-ic" aria-hidden="true"><svg width="20" height="20" viewBox="0 0 24 24"><path d="M3 6.5h18v11H3z M3 7l9 6.5L21 7" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linejoin="round"/></svg></span>' +
      '<span><b>Send to your supervisor</b><small>An email ready to send, with the proposal as a Word document and as slides</small></span></summary>' +
      '<div class="sup-body"><form class="det-form" data-sup-form>' + SUP_FIELDS.map(function (x) {
        return '<div class="rec-f"><label for="sp-' + x[0] + '">' + x[1] + '</label><input id="sp-' + x[0] + '" name="' + x[0] + '" type="' + x[2] + '" maxlength="120" autocomplete="' +
          (x[0] === 'lead' ? 'name' : x[0] === 'email' ? 'email' : 'off') + '" value="' + attr(val[x[0]] || '') + '"></div>';
      }).join('') + '</form>' +
      '<div class="sup-mail"><div class="rec-f"><label for="sp-subj">Subject</label><input id="sp-subj" data-sup-subject readonly></div>' +
      '<div class="rec-f"><label for="sp-body">Email <span class="muted">You can edit it before copying</span></label><textarea id="sp-body" rows="14" data-sup-body></textarea></div></div>' +
      '<div class="out-grid sup-actions">' +
      '<button class="out-btn" type="button" data-sup-docx><b>Proposal</b><span>Word, with a sign-off box</span></button>' +
      '<button class="out-btn" type="button" data-sup-pptx><b>Proposal slides</b><span>PowerPoint, 8 slides</span></button>' +
      '<button class="out-btn" type="button" data-sup-copy><b>Copy email</b><span>Paste into Outlook or NHSmail</span></button>' +
      '<a class="out-btn" data-sup-mailto href="mailto:"><b>Open in email app</b><span>Then attach the files</span></a></div>' +
      '<p class="muted sup-note">Download the files first and attach them to the email; an email link cannot attach files. The proposal holds the protocol only, never patient data. Names and emails you type are remembered on this device only.</p>' +
      '<p class="form-status" role="status" data-sup-status></p></div></details>';
  }
  function supRefresh(box, keepBody) {
    var p = supProtocol(box); if (!p) return;
    var d = supWho(box), m = supEmail(p, d);
    box.querySelector('[data-sup-subject]').value = m.subject;
    var ta = box.querySelector('[data-sup-body]');
    if (!keepBody || !ta.value) ta.value = m.body;
    box.querySelector('[data-sup-mailto]').href = 'mailto:' + encodeURIComponent(d.supEmail || '').replace(/%40/g, '@') +
      '?subject=' + encodeURIComponent(m.subject) + '&body=' + encodeURIComponent(ta.value);
  }
  document.addEventListener('toggle', function (e) {
    var box = e.target;
    if (box.matches && box.matches('.sup-box') && box.open) supRefresh(box, true);
  }, true);
  document.addEventListener('input', function (e) {
    var box = e.target.closest && e.target.closest('.sup-box');
    if (!box) return;
    if (e.target.closest('[data-sup-form]')) {
      var d = supWho(box), me = meGet();
      SUP_FIELDS.forEach(function (x) { me[x[0]] = d[x[0]]; }); meSet(me);
      supRefresh(box, false);
    } else if (e.target.matches('[data-sup-body]')) supRefresh(box, true);
  });
  document.addEventListener('click', function (e) {
    var open = e.target.closest('[data-sup-open]');
    if (open) { var bx = document.getElementById('send-supervisor'); if (bx) { bx.open = true; bx.scrollIntoView({ behavior: 'smooth', block: 'start' }); bx.querySelector('input').focus({ preventScroll: true }); } return; }
    var box = e.target.closest('.sup-box'); if (!box) return;
    var st = box.querySelector('[data-sup-status]'), p = supProtocol(box); if (!p) return;
    var d = supWho(box), name = fileSlug(d.title || p.question) + '-proposal';
    if (e.target.closest('[data-sup-docx]')) {
      sayIn(st, 'Making the Word proposal…');
      exporter().then(function (x) { return x.proposalDocx(p, d); }).then(function (b) { return saveFile(name + '.docx', b); })
        .then(function () { sayIn(st, 'Word proposal ready. Attach it to the email.'); }, function (err) { sayIn(st, downloadError(err)); });
    } else if (e.target.closest('[data-sup-pptx]')) {
      sayIn(st, 'Making the slides…');
      exporter().then(function (x) { return x.proposalPptx(p, d); }).then(function (b) { return saveFile(name + '.pptx', b); })
        .then(function () { sayIn(st, 'Slides ready. Attach them to the email.'); }, function (err) { sayIn(st, downloadError(err)); });
    } else if (e.target.closest('[data-sup-copy]')) {
      var txt = 'Subject: ' + box.querySelector('[data-sup-subject]').value + '\n\n' + box.querySelector('[data-sup-body]').value;
      var ok = function () { sayIn(st, 'Email copied. Paste it into a new message.'); };
      try {
        navigator.clipboard.writeText(txt).then(ok, function () { box.querySelector('[data-sup-body]').select(); sayIn(st, 'Select all and copy the email text.'); });
      } catch (err) { box.querySelector('[data-sup-body]').select(); sayIn(st, 'Select all and copy the email text.'); }
    }
  });

  function chooseActions(id) {
    var run = Array.from(S.runs.values()).filter(function (r) { return r.auditId === id && !r.closed; })[0];
    return (run ? '<a class="btn" href="#/run/' + attr(run.id) + '">Open in My audits</a>' :
      '<button class="btn" type="button" data-choose="' + attr(id) + '">Choose this audit</button>') +
      '<button class="btn btn-secondary" type="button" data-proto-xlsx="' + attr(id) + '">Data sheet (Excel)</button>' +
      '<button class="btn btn-secondary" type="button" data-sup-open>Send to supervisor</button>' +
      '<span class="copy-status" role="status" data-proto-status></span>';
  }
  document.addEventListener('click', function (e) {
    var b = e.target.closest('[data-proto-xlsx]');
    if (!b) return;
    var p = anyAudit(b.getAttribute('data-proto-xlsx')), st = b.parentNode.querySelector('[data-proto-status]');
    if (!p) return;
    sayIn(st, 'Making the data sheet…');
    exporter().then(function (x) { return x.templateXlsx(p, {}); }).then(function (blob) { return saveFile(p.id + '-data-sheet.xlsx', blob); })
      .then(function () { sayIn(st, 'Data sheet ready.'); }, function (err) { sayIn(st, downloadError(err)); });
  });
  function renderProposed(id) {
    var p = S.pById.get(id);
    if (!p) return renderNotFound();
    var dl = window.AI4QI_EMBED ? copyBtn(p.id) :
      '<a class="btn" href="' + attr(p.template_file) + '" download="' + attr(p.id + '.csv') + '">' +
      '<svg width="18" height="18" viewBox="0 0 24 24" aria-hidden="true"><path d="M12 4v11m0 0l-4.5-4.5M12 15l4.5-4.5M5 19h14" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/></svg>Download template (CSV)</a>';

    var body = protocolBody(p, dl, { after: [['Status and effort', '<div class="status-box">' + PROPOSED_BADGE +
        '<span><strong>Data collection effort:</strong> ' + esc(p.effort) + '</span>' +
        (p.novelty ? '<span><strong>Gap:</strong> ' + esc(cap(p.novelty)) + '</span>' : '') + '</div>']] }) +
      feedbackBox(p.id);

    page('<nav class="breadcrumb" aria-label="Breadcrumb"><a href="#/">Home</a> › <a href="#/proposed">Proposed audits</a> › <a href="#/proposed?area=' + encodeURIComponent(p.area) + '">' + esc(p.area) + '</a></nav>' +
      '<article class="doc"><header class="doc-head"><div class="eyebrow"><span class="id-tag">' + esc(p.id) + '</span>' + PROPOSED_BADGE + badge(p.area, 'primary') + '</div>' +
      '<h1>' + esc(p.question) + '</h1>' +
      '<div class="doc-actions">' + chooseActions(p.id) + (window.AI4QI_EMBED ? '' : '<button class="btn btn-secondary" type="button" data-print>Print protocol</button>') + '</div></header>' +
      supervisorBox(p) + body + '</article>', p.id + ' ' + trunc(p.question, 60), 'proposed');
    showUsefulCount(p.id);
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
    if (!window.AI4QI_EMBED && f.figures && f.figures.length) {
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
          '<h3>' + esc(s.source) + '</h3><blockquote class="standard"><p>“' + esc(s.wording) + '”</p></blockquote>' + niceAttribution(s) +
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

  /* ---------- build an audit on request ---------- */
  /* A themed request ("reusable PPE") gets a complete new protocol in the standard layout, written from
     the closest published audits and quoted standards in the library. A request with no theme
     ("a quick closed-loop audit") gets ready-made proposed audits instead. */
  var GENERIC = new Set(('audit audits auditing qi quality improvement project projects idea ideas suggest suggestion suggestions ' +
    'give me my an a the for of to on in do want need would like some any good new quick quickly easy simple fast short ' +
    'closed loop cycle cycles two 2 one uk nhs application portfolio please can you i help with that is are small ' +
    'doable achievable junior doctor doctors trainee foundation fy1 fy2 ct1 st1 imt cst gp').split(' '));
  function themeWords(q) {
    return String(q || '').toLowerCase().split(/[^a-z0-9]+/).filter(function (w) { return w && !GENERIC.has(w); });
  }
  function isThemed(q) { return themeWords(q).length > 0; }

  var BUILT_KEY = 'ai4qi_built_v1';
  S.built = new Map();
  try { (JSON.parse(localStorage.getItem(BUILT_KEY) || '[]') || []).forEach(function (b) { if (b && b.id) S.built.set(b.id, b); }); } catch (e) {}
  function builtGet(id) { return S.built.get(id); }
  function builtSave(p) {
    S.built.delete(p.id); S.built.set(p.id, p);
    var list = Array.from(S.built.values()).slice(-30);
    try { localStorage.setItem(BUILT_KEY, JSON.stringify(list)); } catch (e) {}
  }
  function builtId(q, n) {
    n = +n || 1;
    var slug = themeWords(q).join('-').slice(0, n > 1 ? 26 : 29).replace(/-+$/, '') || 'audit';
    return 'B-' + slug + (n > 1 ? '-' + n : '');
  }
  function variantsOf(q) {
    var t = String(q).trim().toLowerCase(), out = [];
    S.built.forEach(function (b) { if (String(b.topic).trim().toLowerCase() === t) out.push(b); });
    return out.sort(function (a, b) { return (a.variant || 1) - (b.variant || 1); });
  }

  /* Library material for a topic: published audits (closed-loop, UK and detailed first), standards, proposed. */
  function resourcesFor(q) {
    var qt = tokens(themeWords(q).join(' ') || q), phrase = qt.length > 1 ? q.toLowerCase() : '';
    var hits = qt.length ? rank(S.pubDocs, qt, phrase) : [];
    if (hits.length < 8 && qt.length > 1) {
      var seen = new Set(hits.map(function (r) { return r.d.a.id; }));
      qt.forEach(function (t) {
        rank(S.pubDocs, [t], '').forEach(function (r) { if (!seen.has(r.d.a.id)) { seen.add(r.d.a.id); r.sc *= 0.5; hits.push(r); } });
      });
    }
    var pubs = hits.sort(function (x, y) {
      return (y.cov - x.cov) || ((y.d.a.dt || 0) - (x.d.a.dt || 0)) || ((y.d.a.lc || 0) - (x.d.a.lc || 0)) || (y.d.a.uk - x.d.a.uk) || (y.sc - x.sc);
    }).map(function (r) { return r.d.a; });
    var stds = S.standards.map(function (s) {
      var m = indexDoc([[s.source, 3], [s.wording, 2]]), sc = 0, cov = 0;
      qt.forEach(function (t) { var v = m.get(t); if (v) { sc += v; cov++; } });
      return { s: s, sc: sc, cov: cov };
    }).filter(function (x) { return x.cov; }).sort(function (x, y) { return (y.cov - x.cov) || (y.sc - x.sc); })
      .slice(0, 8).map(function (x) { return x.s; });
    var props = qt.length ? rank(S.propDocs, qt, phrase).sort(function (x, y) { return (y.cov - x.cov) || (y.sc - x.sc); })
      .slice(0, 4).map(function (r) { return r.d.p; }) : [];
    return { pubs: pubs, stds: stds, props: props };
  }

  function auditLine(a) {
    var parts = ['[' + a.id + '] ' + a.t];
    var where = [a.s, a.co, a.y].filter(Boolean).join(', ');
    if (where) parts.push(where);
    if (a.st) parts.push('standard: ' + a.st);
    if (a.f) parts.push('before: ' + a.f);
    if (a.c || a.iv) parts.push('fix: ' + (a.c || a.iv) + (a.fx ? ' (' + a.fx + ')' : ''));
    if (a.r2) parts.push('after: ' + a.r2);
    parts.push(a.lc ? 'closed loop' : 'single cycle');
    if (a.uk) parts.push('UK/Ireland');
    return parts.join(' | ').slice(0, 700);
  }

  /* NICE's UK Open Content Licence does not cover AI use: unless the owner records NICE's written
     permission (config "nice_ai_permission": true), NICE wording is never put in a build prompt;
     the app inserts the published wording itself, with NICE's attribution. */
  var NICE_LICENCE_URL = 'https://www.nice.org.uk/reusing-our-content/nice-uk-open-content-licence';
  function isNice(s) { s = s || {}; return /^\s*NICE\b/.test(s.source || '') || /(^|\.)nice\.org\.uk\//i.test(s.url || ''); }
  function niceAI() { return !!(BE.cfg && BE.cfg.nice_ai_permission === true); }
  function niceAttribution(s) {
    if (!isNice(s)) return '';
    var y = (String(s.source || '').match(/(19|20)\d\d(?!.*(19|20)\d\d)/) || [''])[0], url = safeUrl(s.url);
    return '<p class="nice-attr">© NICE ' + esc(y) + ' ' + esc(String(s.source || '').replace(/^NICE\s*/, '')) + '. Available from ' +
      (url ? '<a href="' + attr(url) + '" target="_blank" rel="noopener">' + esc(url.replace(/^https?:\/\//, '')) + '</a>' : 'www.nice.org.uk') +
      '. All rights reserved. Subject to <a href="' + NICE_LICENCE_URL + '" target="_blank" rel="noopener">Notice of rights</a>. NICE guidance is prepared for the National Health Service in England. ' +
      'All NICE guidance is subject to regular review and may be updated or withdrawn. NICE accepts no responsibility for the use of its content in this product.</p>';
  }
  var BUILD_RULES = [
    'You write clinical audit protocols for Ai4Qi, a professional clinical audit library used by UK and Irish doctors, nurses and allied health professionals.',
    'Write ONE complete, ready-to-run audit protocol on the requested theme. It is shown directly to clinicians as a finished document.',
    '',
    'Rules:',
    '- The audit question is ONE plain, measurable question: who, against what standard, what counts as a pass. One audit = one question (never "X and Y"). No jargon; define any term.',
    '- Pick the highest-volume, highest-harm aspect of the theme that a small team can measure in a few weeks from routine records or direct observation.',
    '- Standard: prefer a standard from the STANDARDS list below and copy its wording and URL exactly. Otherwise use a current UK national standard (NICE, Royal Colleges, NHS England, HSE Ireland, GIRFT, CQC, national audits, statutory guidance) and quote only wording you are certain is verbatim; never overstate it (e.g. "regularly" is not "daily"). If no national standard exists, set source to "Local standard" plus a short description of it, and never present it as national.',
    '- Evidence: 2-4 lines, each exactly "[id] where: before → after (fix)", citing ONLY ids from the PUBLISHED AUDITS list below, with numbers exactly as given there. Prefer closed-loop and UK/Ireland audits. If only related-topic audits exist, cite the closest and end the line with "(related topic)". If nothing fits, return an empty list.',
    '- Change: the one fix to put in. Prefer a form, template, checklist, default or other system change; teaching alone rarely works.',
    '- Pitfalls: 3-4 lines, each "what can go wrong or who will object → how to prevent it".',
    '- Pearls: 3-4 short practical tips that make it succeed.',
    '- Template: 8-14 data-collection fields, snake_case, each with type yes/no, date, datetime, number, choice (with options) or text, and an optional short note. Start with a pseudonymised local audit code, never names or NHS numbers. Include a pass field.',
    '- Timeline in weeks, exactly in this form: "Wk 1–2 collect; Wk 3 analyse/present; Wk 4 change; Wk 5–10 embed; Wk 11–12 re-audit".',
    '- Similar: if one of the PROPOSED AUDITS below would clearly serve this user better (more important, easier to measure, or better standard), give its id and one sentence saying why; otherwise null.',
    '- UK English. No preamble, no notes about your process, sources you lack, or uncertainty. Short, professional sentences.',
    '',
    'Reply with only one JSON object with exactly these keys:',
    '{"question": string, "area": string (clinical area), "alternative": {"question": string, "why": string}, "why": string (1-2 sentences: the gap or harm), ' +
    '"standard": {"source": string, "wording": string, "url": string or ""}, "pass": string, "population": string (include exclusions), ' +
    '"sample": string, "data_source": string, "template": [{"field": string, "type": string, "options": [string], "note": string}], "timeline": string, ' +
    '"change": string, "target": string (starts with e.g. "≥90%"), "reaudit": string, "close_loop": string, "evidence": [string], ' +
    '"pitfalls": [string], "pearls": [string], "effort": string (e.g. "~15 min per 10 patients"), "similar": {"id": string, "better_because": string} or null}'
  ].join('\n');

  function buildPrompt(q, res, avoid) {
    var pubs = res.pubs.slice(0, 30).map(auditLine).join('\n') || '(none on this theme)';
    var nice = !niceAI();
    var stds = res.stds.map(function (s) {
      return '- ' + s.source + ' | ' + (nice && isNice(s) ? '(NICE: wording not supplied)' : '"' + s.wording + '"') + ' | ' + (s.url || '');
    }).join('\n') || '(none listed)';
    var props = res.props.map(function (p) { return '- ' + p.id + ': ' + p.question + ' (standard: ' + ((p.standard || {}).source || '') + ')'; }).join('\n') || '(none)';
    return BUILD_RULES + '\n\nTHEME REQUESTED: ' + q.slice(0, 300) +
      '\n\nPUBLISHED AUDITS (library ids; closest first):\n' + pubs +
      '\n\nSTANDARDS (exact wording):\n' + stds +
      (nice ? '\nFor any NICE standard, do not write NICE wording: set "wording" to "" and give the exact NICE source (e.g. "NICE NG253 rec 1.8.3") and URL; the app inserts the published wording.' : '') +
      '\n\nPROPOSED AUDITS already in the library:\n' + props +
      (avoid && avoid.length ? '\n\nALREADY OFFERED ON THIS THEME. Write a DIFFERENT audit: a different aspect of the theme, a different question and a different standard where possible. Do not repeat these:\n' +
        avoid.map(function (x) { return '- ' + x; }).join('\n') : '');
  }

  /* Tidy what came back: keep only known citations and sensible shapes. */
  function normaliseBuilt(o, q, res, n) {
    if (!o || typeof o !== 'object' || !o.question) throw { code: 'invalid_json' };
    function str(v) { return typeof v === 'string' ? v.trim() : (v == null ? '' : String(v)); }
    function list(v) { return Array.isArray(v) ? v.map(str).filter(Boolean) : []; }
    var st = o.standard || {};
    var p = {
      id: builtId(q, n), variant: +n || 1, topic: q, built: new Date().toISOString().slice(0, 10),
      question: str(o.question), area: str(o.area),
      alternative: o.alternative && o.alternative.question ? { question: str(o.alternative.question), why: str(o.alternative.why) } : null,
      why: str(o.why), standard: { source: str(st.source), wording: str(st.wording), url: safeUrl(str(st.url)) },
      pass: str(o.pass), population: str(o.population), sample: str(o.sample), data_source: str(o.data_source),
      template: (Array.isArray(o.template) ? o.template : []).filter(function (f) { return f && f.field; }).map(function (f) {
        return { field: str(f.field).replace(/[^A-Za-z0-9_]+/g, '_').toLowerCase(), type: str(f.type) || 'text',
          options: list(f.options), note: str(f.note) };
      }),
      timeline: str(o.timeline), change: str(o.change), target: str(o.target), reaudit: str(o.reaudit),
      close_loop: str(o.close_loop), pitfalls: list(o.pitfalls), pearls: list(o.pearls), effort: str(o.effort),
      evidence: list(o.evidence).filter(function (e) {
        var ids = idsIn(e);
        return ids.length && ids.every(function (i) { return S.byId.has(i); });
      }),
      resources: res.pubs.slice(0, 8).map(function (a) { return a.id; }),
      similar: null
    };
    if (!niceAI() && isNice(p.standard)) {                  // NICE wording comes from the library copy, never from the AI
      var key = function (src) {                           // e.g. "ng253|1.8.3": the guideline and the recommendation number
        var s = String(src || '').toLowerCase(), g = (s.match(/\b(ng|cg|qs|dg|ta|htg|ph|sc)\s?\d+/) || [''])[0].replace(/\s/g, ''),
          r = (s.match(/\b(?:rec(?:ommendation)?s?|statement|qs\d+\s+statement)\s*(\d+(?:\.\d+)*)/) || [])[1] || '';
        return g && r ? g + '|' + r : '';
      };
      var want = key(p.standard.source), hit = want ? res.stds.concat(S.standards || []).filter(function (s) { return isNice(s) && key(s.source) === want; })[0] : null;
      p.standard.wording = hit ? hit.wording : '';          // no exact match: show the source and link only
      if (hit && !p.standard.url) p.standard.url = safeUrl(hit.url);
    }
    var sim = o.similar;
    if (sim && sim.id && S.pById.has(str(sim.id)) && str(sim.better_because)) p.similar = { id: str(sim.id), better_because: str(sim.better_because) };
    return p;
  }

  var BUILD_STEPS = [['question', 'Audit question'], ['standard', 'Standard'], ['template', 'Data template'], ['timeline', 'Timeline'],
    ['change', 'Change'], ['reaudit', 'Re-audit'], ['evidence', 'Evidence'], ['pitfalls', 'Pitfalls'], ['pearls', 'Pearls']];
  function progressHtml(text) {
    return '<ol class="build-steps">' + BUILD_STEPS.map(function (s) {
      var done = text && text.indexOf('"' + s[0] + '"') !== -1;
      return '<li class="' + (done ? 'is-done' : '') + '"><span class="dot" aria-hidden="true"></span>' + esc(s[1]) + (done ? '<span class="sr-only"> written</span>' : '') + '</li>';
    }).join('') + '</ol>';
  }

  var GEN = { ctl: null, sample: undefined };
  function samplerReady() {
    if (GEN.sample !== undefined) return Promise.resolve(GEN.sample);
    if (!window.claude || typeof window.claude.use !== 'function') { GEN.sample = null; return Promise.resolve(null); }
    return window.claude.use('sample').then(function (s) { GEN.sample = s || null; return GEN.sample; }, function () { GEN.sample = null; return null; });
  }
  /* Generation: inside Claude (artifact) via the sample capability; on the hosted site via the
     build_url function in config.json (server-side, key never in the page). */
  function generate(q, res, fresh, onText, signal, n, avoid) {
    var prompt = buildPrompt(q, res, avoid);
    return samplerReady().then(function (sample) {
      if (sample) {
        return sample.json(prompt, { onText: onText, signal: signal, cache: fresh ? false : { gcTime: 86400000 } });
      }
      var url = BE.cfg && safeUrl(BE.cfg.build_url);
      if (!url || !BE.url) throw { code: 'no_generator' };
      return sbClient().then(function (c) { return c.auth.getSession(); }).then(function (r) {
        var sess = r && r.data && r.data.session;
        if (!sess) throw { code: 'sign_in' };
        return post(url, { 'Content-Type': 'application/json', apikey: BE.key, Authorization: 'Bearer ' + sess.access_token });
      });
    });
    function post(url, headers) {
      return fetch(url, { method: 'POST', headers: headers, signal: signal, body: JSON.stringify({ topic: q.slice(0, 300), key: builtId(q, n), fresh: !!fresh, prompt: prompt }) })
        .then(function (r) {
          if (r.status === 429) return r.json().catch(function () { return {}; }).then(function (b) {
            throw { code: b && b.error === 'monthly limit' ? 'site_limit' : 'rate_limited' };
          });
          if (r.status === 401) throw { code: 'sign_in' };
          if (r.status === 422) throw { code: 'refused' };
          if (!r.ok) throw { code: 'upstream_error' };
          return r.json();
        }, function (e) { throw { code: e && e.name === 'AbortError' ? 'cancelled' : 'upstream_error' }; });
    }
  }
  var BUILD_ERRORS = {
    not_granted: 'Building audits needs permission to use Claude on your account. Reload the page to be asked again.',
    sampling_disabled: 'Building audits is not available for this account.',
    rate_limited: 'You have reached the limit for now. Please try again later.',
    sign_in: 'Sign in (free, by emailed link) to build audits.',
    site_limit: 'Building on this site is paused until next month. You can still build any audit free in Claude, using your own Claude account.',
    session_expired: 'Please sign in to Claude again, then try again.',
    refused: 'This theme could not be turned into an audit. Try wording it as a clinical process, e.g. “reusable gowns in theatre”.',
    invalid_json: 'The protocol came back incomplete. Please try again.',
    empty_completion: 'The protocol came back empty. Please try again.',
    no_generator: 'Building new audits is switched on in the hosted Ai4Qi app. Meanwhile, the published audits and standards below are the evidence base for this theme.'
  };

  function resourceSection(res) {
    var pubs = res.pubs.slice(0, 8);
    if (!pubs.length) return '';
    var more = res.pubs.length > pubs.length ? '<p class="more-link"><a href="#/search?q=">All ' + fmt(res.pubs.length) + ' published audits on this theme</a></p>' : '';
    return '<ul class="result-list">' + pubs.map(pubResult).join('') + '</ul>' + more;
  }

  /* One build per audit id, shared by the page and the background prefetch, so opening an audit that
     is already being written attaches to it instead of starting again. */
  var BUILDS = new Map();          // id -> { promise, text, listeners, done }
  function startBuild(q, n, fresh) {
    var id = builtId(q, n), cur = BUILDS.get(id);
    if (cur && !cur.failed) return cur;
    var res = resourcesFor(q);
    var avoid = n > 1 ? variantsOf(q).filter(function (b) { return b.id !== id; }).map(function (b) { return b.question; }) : [];
    var entry = { text: '', listeners: [], res: res, ctl: new AbortController() };
    entry.promise = generate(q, res, fresh, function (u) {
      entry.text = u.text; entry.listeners.forEach(function (fn) { try { fn(u.text); } catch (e) {} });
    }, entry.ctl.signal, n, avoid).then(function (o) {
      var p = normaliseBuilt(o, q, res, n);
      builtSave(p); entry.done = true;
      if (AN.kind === 'plausible' && typeof window.plausible === 'function') window.plausible('Audit built');
      return p;
    }, function (e) { entry.failed = true; throw e; });
    BUILDS.set(id, entry);
    return entry;
  }
  /* While someone reads an audit, quietly write the next one on the same theme. */
  function prefetchNext(p) {
    var next = (p.variant || 1) + 1;
    while (next <= 9 && S.built.get(builtId(p.topic, next))) next++;     // the next one not written yet
    if (next > 9 || BUILDS.get(builtId(p.topic, next))) return;
    var e = startBuild(p.topic, next, false);
    e.promise.then(function () {
      var nav = main.querySelector('[data-variants]');
      if (nav && parseHash().parts[0] === 'build') { var cur = S.built.get(nav.getAttribute('data-variants')); if (cur) nav.outerHTML = variantsNav(cur); }
    }, function () { BUILDS.delete(builtId(p.topic, next)); });
  }

  function renderBuild(params) {
    var q = scrub((params.get('q') || '').trim(), { redacted: 0 }).slice(0, 300);
    if (!q) { location.hash = '#/proposed'; return; }
    if (!isThemed(q)) { location.hash = '#/suggest?q=' + encodeURIComponent(q); return; }
    var fresh = params.get('fresh') === '1', n = Math.max(1, Math.min(9, +params.get('n') || 1));
    var id = builtId(q, n), have = !fresh && S.built.get(id);
    var crumbs = '<nav class="breadcrumb" aria-label="Breadcrumb"><a href="#/">Home</a> › Build an audit</nav>';
    if (have && have.topic.toLowerCase() === q.toLowerCase()) { showBuilt(have, resourcesFor(q), crumbs); prefetchNext(have); return; }
    if (fresh) BUILDS.delete(id);
    var entry = startBuild(q, n, fresh), res = entry.res;
    page(crumbs + '<article class="doc"><header class="doc-head"><div class="eyebrow">' + BUILT_BADGE + '</div>' +
      '<h1>' + (n > 1 ? 'Another audit on ' : 'Building your audit: ') + esc(q) + '</h1>' +
      '<p class="lede">A complete protocol: question, exact standard, template, timeline, change, re-audit and evidence from published audits.</p>' +
      '<div class="build-status" role="status" aria-live="polite" data-build-status><span class="spinner" aria-hidden="true"></span><span data-build-msg>' +
      (entry.text ? 'Nearly ready…' : 'Writing the protocol… this usually takes under a minute.') + '</span></div>' +
      '<div data-build-progress>' + progressHtml(entry.text) + '</div>' +
      '<div class="doc-actions"><button class="btn btn-secondary" type="button" data-build-stop>Stop</button></div></header>' +
      (res.pubs.length ? sec('•', 'Published audits on this theme', resourceSection(res).replace('#/search?q=', '#/search?q=' + encodeURIComponent(q))) : '') +
      '</article>', 'Build: ' + q, 'build');
    GEN.ctl = entry.ctl;
    var prog = main.querySelector('[data-build-progress]'), last = 0;
    var listen = function (text) {
      var now = Date.now();
      if (now - last > 400 && prog && prog.isConnected) { last = now; prog.innerHTML = progressHtml(text); }
    };
    entry.listeners.push(listen);
    var here = function () { var r = parseHash(); return r.parts[0] === 'build' && builtId(scrub((r.params.get('q') || '').trim(), { redacted: 0 }).slice(0, 300), Math.max(1, +r.params.get('n') || 1)) === id; };
    entry.promise.then(function (p) {
      entry.listeners = entry.listeners.filter(function (f) { return f !== listen; });
      if (here()) { showBuilt(p, res, crumbs); prefetchNext(p); }
    }).catch(function (e) {
      BUILDS.delete(id);
      if (!here()) return;
      var code = (e && e.code) || 'upstream_error';
      var msg = code === 'cancelled' ? 'Stopped.' : (BUILD_ERRORS[code] || 'The connection was interrupted. Please try again.');
      var st = main.querySelector('[data-build-status]');
      if (st) st.innerHTML = '<span data-build-msg>' + esc(msg) + '</span>';
      if (prog) prog.innerHTML = '';
      var act = main.querySelector('.doc-actions');
      var cl = BE.cfg && safeUrl(BE.cfg.claude_link);
      if (act && (code === 'site_limit' || code === 'no_generator') && cl) {
        act.innerHTML = '<a class="btn" href="' + attr(cl) + '" target="_blank" rel="noopener">Build it in Claude</a>' +
          '<span class="muted">Type “' + esc(q) + '” in the box there.</span>';
      } else if (act && code === 'sign_in') act.innerHTML = '<a class="btn" href="#/account">Sign in</a>';
      else if (act) act.innerHTML = code === 'no_generator' || code === 'not_granted' || code === 'sampling_disabled' ? '' :
        '<a class="btn" href="#/build?q=' + encodeURIComponent(q) + (n > 1 ? '&n=' + n : '') + '&fresh=1">Try again</a>';
      if (code === 'no_generator' && res.props.length && act) {
        act.insertAdjacentHTML('afterend', '<p class="prose">Closest ready-made protocol: <a href="#/proposed/' + attr(res.props[0].id) + '">' + esc(res.props[0].id + ' – ' + res.props[0].question) + '</a></p>');
      }
    });
  }

  var BUILT_BADGE = badge('Built for you – not yet run', 'warn');
  function showBuilt(p, res, crumbs) {
    var dl = copyBtn(p.id);
    if (p.similar) {
      var sp = S.pById.get(p.similar.id);
      if (sp) p._sim = '<aside class="similar-box no-print"><h2>A ready-made audit may suit you better</h2>' +
        '<p><a href="#/proposed/' + attr(sp.id) + '"><span class="id-tag">' + esc(sp.id) + '</span> ' + esc(sp.question) + '</a></p>' +
        '<p class="muted">' + esc(p.similar.better_because) + '</p></aside>';
    }
    var resHtml = resourceSection(res).replace('#/search?q=', '#/search?q=' + encodeURIComponent(p.topic));
    var after = [];
    if (resHtml) after.push(['Published audits on this theme', resHtml]);
    after.push(['Status and effort', '<div class="status-box">' + BUILT_BADGE +
      (p.effort ? '<span><strong>Data collection effort:</strong> ' + esc(p.effort) + '</span>' : '') +
      '<span><strong>Built:</strong> ' + esc(p.built) + '</span></div>']);
    var body = '<aside class="ai-note"><p><strong>How this was made.</strong> This protocol was drafted by an AI model (Claude, made by Anthropic) from the topic you typed, ' +
      'using published audits and standards from the Ai4Qi library. It is a draft. Check the standard against its linked source, and ask your supervisor to review the protocol before you collect data. ' +
      'Your audit records are never sent to the AI or to Ai4Qi.</p></aside>' + protocolBody(p, dl, { after: after }) + feedbackBox(p.id);
    page(crumbs + '<article class="doc"><header class="doc-head"><div class="eyebrow"><span class="id-tag">' + esc(p.topic) + '</span>' + BUILT_BADGE +
      (p.area ? badge(p.area, 'primary') : '') + '</div>' +
      '<h1>' + esc(p.question) + '</h1>' +
      '<div class="doc-actions">' + chooseActions(p.id) + '</div>' + variantsNav(p) + '</header>' +
      supervisorBox(p) + (p._sim || '') + body + '</article>', trunc(p.question, 70), 'build');
    delete p._sim;
    showUsefulCount(p.id);
  }
  function variantsNav(p) {
    var vs = variantsOf(p.topic), next = (p.variant || 1) + 1;          // the one after this: usually already written
    var links = vs.length > 1 ? vs.map(function (b) {
      var here = b.id === p.id;
      return '<li><a href="#/build?q=' + encodeURIComponent(p.topic) + ((b.variant || 1) > 1 ? '&n=' + b.variant : '') + '"' + (here ? ' aria-current="page"' : '') + '>' +
        '<span class="v-n">' + (b.variant || 1) + '</span>' + esc(trunc(b.question, 90)) + (b.chosen ? ' ' + badge('Chosen', 'ok') : '') + '</a></li>';
    }).join('') : '';
    return '<div class="variants no-print" data-variants="' + attr(p.id) + '">' +
      (next <= 9 ? '<a class="btn btn-secondary" href="#/build?q=' + encodeURIComponent(p.topic) + '&n=' + next + '">Not quite right? Another audit on this theme</a>' : '') +
      (links ? '<details class="v-more"><summary>Audits on this theme (' + vs.length + ')</summary><ol class="v-list">' + links + '</ol></details>' : '') + '</div>';
  }


  /* No theme given: suggest ready-made audits, quick closed-loop ones first. */
  function renderSuggest(params) {
    var q = (params.get('q') || '').trim();
    var fb = new Map();
    fbAll().forEach(function (f) { fb.set(f.id, f.rating); });
    var list = S.proposed.filter(function (p) {
      var f = p.feedback || {};
      return fb.get(p.id) !== 'down' && !((f.down || 0) > (f.up || 0));
    });
    function effortMin(p) { var m = String(p.effort || '').match(/(\d+)\s*min/); return m ? +m[1] : 60; }
    function weeks(p) { var s = parseTimeline(p.timeline); return s ? Math.max.apply(null, s.map(function (x) { return x.b; })) : 52; }
    list = list.slice().sort(function (a, b) { return (weeks(a) - weeks(b)) || (effortMin(a) - effortMin(b)); }).slice(0, 12);
    page('<nav class="breadcrumb" aria-label="Breadcrumb"><a href="#/">Home</a> › Suggested audits</nav>' +
      '<h1>Suggested audits</h1><p class="lede">Ready-to-run protocols with the shortest route to a closed loop. Type a theme above to build one on any topic instead.</p>' +
      buildForm('', false) +
      '<ul class="result-list" style="margin-top:18px">' + list.map(propResult).join('') + '</ul>' +
      '<p class="more-link"><a href="#/proposed">Browse all ' + fmt(S.proposed.length) + ' proposed audits</a></p>', 'Suggested audits', 'proposed');
  }

  function buildForm(q, big) {
    return '<form class="search-form" data-build>' +
      '<label for="bq"' + (big ? '' : ' class="visually-hidden"') + '>What do you want to audit?</label>' +
      '<input id="bq" name="q" type="search" autocomplete="off" value="' + attr(q || '') + '" placeholder="e.g. reusable PPE in theatre, sepsis antibiotics within an hour, falls after admission">' +
      '<button class="btn" type="submit"><svg width="18" height="18" viewBox="0 0 24 24" aria-hidden="true"><path d="M5 12h14M12 5v14" stroke="currentColor" stroke-width="2.4" stroke-linecap="round"/></svg>Build audit</button>' +
      '</form>';
  }
  document.addEventListener('submit', function (e) {
    var f = e.target.closest('[data-build]');
    if (!f) return;
    e.preventDefault();
    var q = scrub(f.querySelector('input[name="q"]').value.trim(), { redacted: 0 });   // never let identifiers reach the builder
    location.hash = q && isThemed(q) ? '#/build?q=' + encodeURIComponent(q) : '#/suggest' + (q ? '?q=' + encodeURIComponent(q) : '');
  });
  document.addEventListener('click', function (e) {
    if (e.target.closest('[data-build-stop]') && GEN.ctl) GEN.ctl.abort();
  });


  /* ---------- running an audit: dashboard, data, de-identification, outputs ---------- */
  /* Audit data stays on the device (browser storage). Nothing entered here is sent anywhere.
     Signed-in users may opt in to reminders; then only the audit question, the next step, a due
     date and counts are sent, never data. */
  var RUNS_KEY = 'ai4qi_runs_v1';
  var RUN_STAGES = [
    ['setup', 'Set up'], ['cycle1', 'Collect cycle 1'], ['present', 'Analyse and present'],
    ['change', 'Make the change'], ['reaudit', 'Re-audit'], ['close', 'Close the loop']
  ];
  var STAGE_STATUS = ['started', 'started', 'cycle1', 'cycle1', 'change', 'reaudit', 'closed'];
  S.runs = new Map();
  /* Records are encrypted at rest with a key derived from the user's passcode (PBKDF2 → AES-GCM).
     The key lives only in memory; after 15 idle minutes it is dropped and the audits lock.
     "Shared computer" keeps everything in session storage, which the browser clears on close. */
  var VAULT_META = 'ai4qi_vault_v1', VAULT_DATA = 'ai4qi_runs_enc_v1', PBKDF2_ITER = 310000, IDLE_MS = 15 * 60000;
  var V = { key: null, store: null, meta: null, last: Date.now(), legacy: null };
  function b64(buf) { var s2 = '', a = new Uint8Array(buf); for (var i = 0; i < a.length; i++) s2 += String.fromCharCode(a[i]); return btoa(s2); }
  function unb64(t) { var s2 = atob(t), a = new Uint8Array(s2.length); for (var i = 0; i < s2.length; i++) a[i] = s2.charCodeAt(i); return a; }
  function stGet(st, k) { try { return st.getItem(k); } catch (e) { return null; } }
  var DEMO = (function () { try { return localStorage.getItem('ai4qi_demo') === '1'; } catch (e) { return false; } })();
  (function initVault() {
    var m = null;
    try { m = JSON.parse(stGet(sessionStorage, VAULT_META) || 'null'); if (m) V.store = sessionStorage; } catch (e) {}
    if (!m) { try { m = JSON.parse(stGet(localStorage, VAULT_META) || 'null'); if (m) V.store = localStorage; } catch (e) {} }
    V.meta = m;
    try { var old = JSON.parse(stGet(localStorage, RUNS_KEY) || 'null'); if (old && old.length) V.legacy = old; } catch (e) {}
  })();
  function vaultSet() { return !!V.meta; }
  function vaultOpen() { return !!V.key; }
  function deriveKey(pass, salt, iter) {
    var enc = new TextEncoder();
    return crypto.subtle.importKey('raw', enc.encode(pass), 'PBKDF2', false, ['deriveKey']).then(function (base) {
      return crypto.subtle.deriveKey({ name: 'PBKDF2', salt: salt, iterations: iter, hash: 'SHA-256' }, base, { name: 'AES-GCM', length: 256 }, false, ['encrypt', 'decrypt']);
    });
  }
  function seal(key, text) {
    var iv = crypto.getRandomValues(new Uint8Array(12));
    return crypto.subtle.encrypt({ name: 'AES-GCM', iv: iv }, key, new TextEncoder().encode(text)).then(function (ct) { return { iv: b64(iv), ct: b64(ct) }; });
  }
  function unseal(key, box) {
    return crypto.subtle.decrypt({ name: 'AES-GCM', iv: unb64(box.iv) }, key, unb64(box.ct)).then(function (pt) { return new TextDecoder().decode(pt); });
  }
  function vaultCreate(pass, shared) {
    var salt = crypto.getRandomValues(new Uint8Array(16)), store = shared ? sessionStorage : localStorage;
    return deriveKey(pass, salt, PBKDF2_ITER).then(function (key) {
      return seal(key, 'ai4qi-ok').then(function (check) {
        V.meta = { v: 1, salt: b64(salt), iter: PBKDF2_ITER, check: check, shared: !!shared };
        V.store = store; V.key = key; V.last = Date.now();
        store.setItem(VAULT_META, JSON.stringify(V.meta));
        (V.legacy || []).forEach(function (r) { if (r && r.id) S.runs.set(r.id, r); });
        return runsSave().then(function () { try { localStorage.removeItem(RUNS_KEY); } catch (e) {} V.legacy = null; });
      });
    });
  }
  function vaultUnlock(pass) {
    var m = V.meta;
    return deriveKey(pass, unb64(m.salt), m.iter || PBKDF2_ITER).then(function (key) {
      return unseal(key, m.check).then(function (t) {
        if (t !== 'ai4qi-ok') throw new Error('wrong');
        var box = null; try { box = JSON.parse(stGet(V.store, VAULT_DATA) || 'null'); } catch (e) {}
        return (box ? unseal(key, box) : Promise.resolve('[]')).then(function (json) {
          S.runs = new Map(); (JSON.parse(json) || []).forEach(function (r) { if (r && r.id) S.runs.set(r.id, r); });
          V.key = key; V.last = Date.now();
        });
      }, function () { throw new Error('wrong'); });
    });
  }
  function vaultLock() { V.key = null; S.runs = new Map(); S.pendingImport = null; }
  function vaultErase() {
    [localStorage, sessionStorage].forEach(function (st) { try { st.removeItem(VAULT_META); st.removeItem(VAULT_DATA); st.removeItem(RUNS_KEY); } catch (e) {} });
    V.meta = null; V.store = null; V.legacy = null; vaultLock();
  }
  var saveChain = Promise.resolve();
  function runsSave() {
    if (!V.key) return Promise.resolve(false);
    var key = V.key, json = JSON.stringify(Array.from(S.runs.values())), store = V.store;
    saveChain = saveChain.then(function () {
      return seal(key, json).then(function (box) { store.setItem(VAULT_DATA, JSON.stringify(box)); return true; });
    }).catch(function () { return false; });
    return saveChain;
  }
  ['pointerdown', 'keydown', 'scroll', 'touchstart'].forEach(function (ev) { window.addEventListener(ev, function () { V.last = Date.now(); }, { passive: true }); });
  setInterval(function () {
    if (V.key && Date.now() - V.last > IDLE_MS) {
      vaultLock();
      var n = parseHash().parts[0];
      if (n === 'run' || n === 'my-audits') route();
    }
  }, 20000);
  function runPut(r) {
    if (!V.key) return false;
    r.updated = new Date().toISOString(); S.runs.set(r.id, r);
    runsSave().then(function (ok) { if (!ok) { var o = main.querySelector('[data-out-status]') || main.querySelector('[data-det-status]'); if (o) o.textContent = 'Could not save: this browser has no space left. Download a backup.'; } });
    syncRun(r); return true;
  }
  function newRunId() {
    var a = new Uint8Array(8); (window.crypto || window.msCrypto).getRandomValues(a);
    return 'r-' + Array.prototype.map.call(a, function (b) { return ('0' + (b % 36).toString(36)).slice(-1); }).join('') + Date.now().toString(36).slice(-4);
  }
  function todayIso() { var d = new Date(); return new Date(d.getTime() - d.getTimezoneOffset() * 60000).toISOString().slice(0, 10); }
  function addDays(iso, n) { var d = new Date(iso + 'T12:00:00'); d.setDate(d.getDate() + n); return d.toISOString().slice(0, 10); }
  function dateGBs(iso) { if (!iso) return ''; var d = new Date(iso + 'T12:00:00'); return isNaN(d) ? '' : d.toLocaleDateString('en-GB', { day: 'numeric', month: 'short', year: 'numeric' }); }
  function stageIdx(r) { if (r.closed) return RUN_STAGES.length; for (var i = 0; i < RUN_STAGES.length; i++) if (RUN_STAGES[i][0] === r.stage) return i; return 0; }
  function sampleGuess(p) {
    var nums = [], re = /(\d{1,4})(?:\s*[–-]\s*\d{1,4})?\s*(months?|weeks?|days?|years?|hours?|mins?|minutes?|%)?/gi, m;
    while ((m = re.exec(String(p.sample || '')))) if (!m[2] && +m[1] >= 10) nums.push(+m[1]);
    return nums.length ? Math.min(nums[0], 500) : 30;
  }

  /* --- demo mode (#/demo on, #/demo/off): one-click example data for live demonstrations.
         Everything it makes is fictitious, marked "Example data", and kept in shared-computer storage. --- */
  var DEMO_PASS = 'ai4qi-demo-passcode';
  function demoSet(on) { DEMO = on; try { if (on) localStorage.setItem('ai4qi_demo', '1'); else localStorage.removeItem('ai4qi_demo'); } catch (e) {} }
  function demoBar() {
    return '<div class="demo-bar" role="note"><span><strong>Demo mode.</strong> Example-data buttons are on; everything they fill is fictitious.</span>' +
      '<span><button type="button" class="link-btn" data-demo-reset>Clear example audits</button> · <a href="#/demo/off">Turn off</a></span></div>';
  }
  function demoRand(seed) { var x = seed % 2147483647 || 7; return function () { x = x * 16807 % 2147483647; return (x - 1) / 2147483646; }; }
  function demoStepLabel(r) {
    var k = RUN_STAGES[stageIdx(r)][0];
    return { setup: 'Demo: fill the details', cycle1: 'Demo: fill cycle 1 data', present: 'Demo: go to the change', change: 'Demo: record the change',
      reaudit: 'Demo: fill the re-audit', close: 'Demo: close the loop' }[k] || 'Demo: next step';
  }
  var NO_CAUSE_RE = /^(no delay|no delays|none|nil|no reason|no problem|no issue|not applicable|n\/?a|not delayed|on time)$/i;
  function demoRows(r, ck, n) {
    var p = r.protocol, t = p.template || [], pf = passField(p), tg = parseTarget(p.target) || { op: '≥', value: 90 };
    var low = tg.op === '≤' || tg.op === '<';
    var rate = ck === 'c1' ? (low ? Math.min(95, tg.value + 25) : Math.max(25, tg.value - 32)) : (low ? Math.max(0, tg.value - 4) : Math.min(98, tg.value + 3));
    var yes = Math.round(n * rate / 100), R = demoRand(Date.now() + (ck === 'c1' ? 1 : 2));
    var o = r.options || {}, start = r.details.startDate || todayIso();
    var from = ck === 'c1' ? addDays(start, -56) : addDays((r.changeMade && r.changeMade.date) || addDays(start, 28), 7);
    var flags = []; for (var i = 0; i < n; i++) flags.push(i < yes); flags.sort(function () { return R() - 0.5; });
    var seq = r.codeSeq || 1;
    var rows = flags.map(function (pass, i) {
      var row = {}, day = addDays(from, Math.floor(i * 56 / n)), mins = 8 * 60 + Math.floor(R() * 14 * 60);
      t.forEach(function (f) {
        var nm = f.field, v = '';
        if (isCodeField(f)) v = 'P' + ('00' + (seq++)).slice(-3);
        else if (nm === pf) v = pass ? 'Yes' : 'No';
        else if (/yes/i.test(f.type)) v = R() < (ck === 'c1' ? 0.7 : 0.9) ? 'Yes' : 'No';
        else if (f.type === 'choice' && (f.options || []).length) {
          var ops = f.options, cause = /delay|reason|barrier|cause|why|fail/i.test(nm), none = ops.filter(function (x) { return NO_CAUSE_RE.test(x); })[0];
          if (cause && none && pass) v = none;
          else {
            var pool = ops.filter(function (x) { return x !== none && !/^other$/i.test(x); }); if (!pool.length) pool = ops;
            v = pool[Math.min(pool.length - 1, Math.floor(Math.pow(R(), 1.8) * pool.length))];
          }
        }
        else if (f.type === 'date') v = o.monthOnly ? day.slice(0, 7) : day;
        else if (f.type === 'datetime') { mins += 5 + Math.floor(R() * (pass ? 25 : 70)); var h = Math.min(23, Math.floor(mins / 60)), m = mins % 60; v = o.monthOnly ? day.slice(0, 7) : day + ' ' + ('0' + h).slice(-2) + ':' + ('0' + m).slice(-2); }
        else if (f.type === 'number') {
          v = /minute|mins/i.test(nm) ? (pass ? 15 + Math.floor(R() * 40) : 65 + Math.floor(R() * 110)) :
            /hour/i.test(nm) ? (pass ? 1 + Math.floor(R() * 20) : 26 + Math.floor(R() * 48)) :
            /news|score/i.test(nm) ? 5 + Math.floor(R() * 7) : /age.*month/i.test(nm) ? 1 + Math.floor(R() * 23) :
            /age/i.test(nm) ? 45 + Math.floor(R() * 45) : /day/i.test(nm) ? 1 + Math.floor(R() * 9) :
            /percent|pct|%/i.test(nm) ? 40 + Math.floor(R() * 60) : /ml|volume/i.test(nm) ? 250 * (1 + Math.floor(R() * 4)) : 1 + Math.floor(R() * 20);
        }
        if (v !== '') row[nm] = v;
      });
      return row;
    });
    r.codeSeq = seq;
    return rows;
  }
  function demoDetails(r) {
    var d = r.details;
    if (!d.site) d.site = 'Riverside General Hospital (fictional)';
    if (!d.department) d.department = { 'Emergency medicine': 'Emergency Department', 'Surgery': 'General Surgery' }[r.protocol.area] || (r.protocol.area || 'Medicine');
    if (!d.lead) d.lead = 'Dr Alex Morgan (example)';
    if (!d.team) d.team = 'Two foundation doctors; ward pharmacist';
    if (!d.supervisor) d.supervisor = 'Supervising consultant (example)';
    d.startDate = addDays(todayIso(), -16 * 7);
  }
  function demoStep(r) {
    var k = RUN_STAGES[stageIdx(r)][0], want = +r.details.sampleSize || 30;
    r.demo = true;
    if (k === 'setup') { demoDetails(r); r.stage = 'cycle1'; }
    else if (k === 'cycle1') { if (!r.cycles.c1.rows.length) r.cycles.c1.rows = demoRows(r, 'c1', want); r.stage = 'present'; S.view.runTab = 'c1'; }
    else if (k === 'present') r.stage = 'change';
    else if (k === 'change') {
      if (!r.changeMade.description) r.changeMade = { description: String(r.protocol.change || 'The agreed change').slice(0, 600), date: addDays(r.details.startDate || todayIso(), 28) };
      r.stage = 'reaudit'; S.view.runTab = 'c2';
    }
    else if (k === 'reaudit') { if (!r.cycles.c2.rows.length) r.cycles.c2.rows = demoRows(r, 'c2', want); r.stage = 'close'; S.view.runTab = 'c2'; }
    else r.closed = true;
  }
  document.addEventListener('click', function (e) {
    if (!DEMO) return;
    if (e.target.closest('[data-demo-vault]')) { vaultCreate(DEMO_PASS, true).then(afterUnlock); return; }
    if (e.target.closest('[data-demo-unlock]')) {
      vaultUnlock(DEMO_PASS).then(afterUnlock, function () { var st = main.querySelector('[data-vault-status]'); sayIn(st, 'These audits were not made with the demo passcode.'); });
      return;
    }
    if (e.target.closest('[data-demo-reset]')) {
      if (!V.key) return;
      Array.from(S.runs.values()).forEach(function (r) { if (r.demo) S.runs.delete(r.id); });
      runsSave(); if (parseHash().parts[0] === 'run') location.hash = '#/my-audits'; else route();
      return;
    }
    var r = curRun(); if (!r) return;
    var fill = e.target.closest('[data-demo-fill]');
    if (fill) {
      var ck = fill.getAttribute('data-demo-fill');
      r.demo = true; r.cycles[ck].rows = r.cycles[ck].rows.concat(demoRows(r, ck, +r.details.sampleSize || 30));
      if (ck === 'c1' && stageIdx(r) < 1) r.stage = 'cycle1';
      runPut(r); S.view.runTab = ck; renderRunKeep(r, '[data-run-tab="' + ck + '"]');
      return;
    }
    if (e.target.closest('[data-demo-step]')) { demoStep(r); runPut(r); renderRunKeep(r, '[data-demo-step]'); }
  });

  function chooseAudit(p) {
    if (!V.key) { S.pendingChoose = p.id; location.hash = '#/my-audits'; return; }
    var existing = Array.from(S.runs.values()).filter(function (r) { return r.auditId === p.id && !r.closed; })[0];
    if (existing) { location.hash = '#/run/' + existing.id; return; }
    var proto = JSON.parse(JSON.stringify(p));
    delete proto.feedback;
    var r = { id: newRunId(), auditId: p.id, protocol: proto, created: new Date().toISOString(), stage: 'setup', closed: false,
      details: { title: p.question, site: '', department: '', lead: '', team: '', supervisor: '', startDate: todayIso(), sampleSize: sampleGuess(p) },
      cycles: { c1: { rows: [] }, c2: { rows: [] } }, changeMade: { description: '', date: '' }, reminders: false,
      options: { monthOnly: false, noFreeText: false } };
    if (DEMO) r.demo = true;                             // chosen in demo mode: an example audit from the start
    if (!runPut(r)) { window.alert && 0; }
    if (/^B-/.test(p.id)) { var b = builtGet(p.id); if (b) { b.chosen = true; builtSave(b); } }
    if (AN.kind === 'plausible' && typeof window.plausible === 'function') window.plausible('Audit chosen', { props: { kind: /^B-/.test(p.id) ? 'built' : 'proposed' } });
    location.hash = '#/run/' + r.id;
  }

  /* --- the pass field and results --- */
  function passField(p) {
    var t = p.template || [];
    var f = t.filter(function (x) { return /^(pass|met_standard|meets_standard|compliant|standard_met)$/i.test(x.field); })[0] ||
      t.filter(function (x) { return /pass|compliant|met/i.test(x.field) && /yes/i.test(x.type); })[0] ||
      t.filter(function (x) { return /yes/i.test(x.type); }).slice(-1)[0];
    return f ? f.field : null;
  }
  function yn(v) {
    var s = String(v == null ? '' : v).trim().toLowerCase();
    if (/^(y|yes|true|1|✓|pass|met)$/.test(s)) return 'Yes';
    if (/^(n|no|false|0|✗|fail|not met)$/.test(s)) return 'No';
    if (/^(n\/?a|not applicable)$/.test(s)) return 'N/A';
    return '';
  }
  function parseTarget(t) {
    var m = String(t || '').match(/([≥≤<>]?)\s*(\d+(?:\.\d+)?)\s*%/);
    return m ? { op: m[1] || '≥', value: +m[2], text: String(t) } : null;
  }
  function cycleStats(p, rows) {
    var pf = passField(p), passN = 0, failN = 0, dates = [];
    rows.forEach(function (row) {
      var v = pf ? yn(row[pf]) : '';
      if (v === 'Yes') passN++; else if (v === 'No') failN++;
      (p.template || []).forEach(function (f) { if (/date/.test(f.type) && row[f.field]) dates.push(String(row[f.field]).slice(0, 10)); });
    });
    dates.sort();
    var denom = passN + failN;
    return { n: rows.length, passN: passN, failN: failN, pct: denom ? Math.round(passN / denom * 1000) / 10 : null,
      from: dates[0] || null, to: dates[dates.length - 1] || null };
  }
  function runStats(r) {
    var p = r.protocol, out = { target: parseTarget(p.target), cycles: [], breakdowns: [] };
    [['c1', 'Cycle 1'], ['c2', 'Re-audit']].forEach(function (c) {
      var s = cycleStats(p, r.cycles[c[0]].rows), t = r.totals && r.totals[c[0]];
      if (!s.n && t && (t.n || t.den)) {
        s = { n: t.n || t.den, passN: t.met, failN: t.den - t.met, pct: t.den ? Math.round(t.met / t.den * 1000) / 10 : null, from: t.from || null, to: t.to || null, fromTotals: true };
      }
      s.key = c[0]; s.label = c[1]; out.cycles.push(s);
    });
    (p.template || []).forEach(function (f) {
      if (f.type !== 'choice' || !(f.options || []).length) return;
      var b = { field: f.field, label: fieldLabel(f.field), cycles: {} };
      ['c1', 'c2'].forEach(function (k) {
        var counts = new Map();
        r.cycles[k].rows.forEach(function (row) { var v = row[f.field]; if (v) counts.set(v, (counts.get(v) || 0) + 1); });
        b.cycles[k] = Array.from(counts.entries()).map(function (e) { return { option: e[0], n: e[1] }; }).sort(function (a, c) { return c.n - a.n; });
      });
      ['c1', 'c2'].forEach(function (k) {
        var tb = !b.cycles[k].length && r.totals && r.totals.breakdown && r.totals.breakdown[k] && r.totals.breakdown[k][f.field];
        if (tb) b.cycles[k] = tb.slice().sort(function (a, c) { return c.n - a.n; });
      });
      if (b.cycles.c1.length || b.cycles.c2.length) out.breakdowns.push(b);
    });
    return out;
  }
  function fieldLabel(f) { return cap(String(f).replace(/_/g, ' ')); }

  /* --- totals: the one-line results code from the Excel Results tab, or totals typed by hand --- */
  function parseResultsCode(text) {
    var line = String(text || '').split(/\r?\n/).filter(function (l) { return /AI4QI\s+v1/i.test(l); })[0];
    if (!line) return null;
    var parts = line.split('|').map(function (x) { return x.trim(); }), out = { id: parts[1] || '', c1: null, c2: null, breakdown: { c1: {}, c2: {} } };
    parts.slice(2).forEach(function (seg) {
      var m = seg.match(/^(C1|RE)\s+(\d+)\s*\/\s*(\d+)(?:\s+n\s*=\s*(\d+))?$/i);
      if (m) { out[m[1].toUpperCase() === 'C1' ? 'c1' : 'c2'] = { met: +m[2], den: +m[3], n: m[4] != null ? +m[4] : +m[3] }; return; }
      seg.split(';').forEach(function (b) {
        var mm = b.trim().match(/^(C1|RE)\s+([^:]+):\s*(.*)$/i);
        if (!mm) return;
        var ck = mm[1].toUpperCase() === 'C1' ? 'c1' : 'c2', key = mm[2].trim().toLowerCase().replace(/[\s_]+/g, '_');
        out.breakdown[ck][key] = mm[3].split(',').map(function (pair) {
          var q = pair.split('='); return q.length === 2 && q[0].trim() ? { option: q[0].trim(), n: +q[1] || 0 } : null;
        }).filter(function (x) { return x && x.n > 0; });
      });
    });
    if (!out.c1 && !out.c2) return null;
    ['c1', 'c2'].forEach(function (k) { var t = out[k]; if (t && (t.met > t.den || t.den > 100000)) out[k] = null; });
    return out;
  }
  function applyTotals(r, code) {
    r.totals = r.totals || {};
    ['c1', 'c2'].forEach(function (k) { if (code[k] && (code[k].den || code[k].n)) r.totals[k] = code[k]; });
    var fields = {}; (r.protocol.template || []).forEach(function (f) { fields[f.field.toLowerCase()] = f.field; });
    r.totals.breakdown = r.totals.breakdown || { c1: {}, c2: {} };
    ['c1', 'c2'].forEach(function (k) {
      Object.keys(code.breakdown[k] || {}).forEach(function (key) { var f = fields[key]; if (f) r.totals.breakdown[k][f] = code.breakdown[k][key]; });
    });
    var i = stageIdx(r);
    if (r.totals.c2 && r.totals.c2.den && i < 5) r.stage = 'close';
    else if (r.totals.c1 && r.totals.c1.den && i < 2) r.stage = 'present';
  }
  function totalsBox(r) {
    var t = r.totals || {}, c1 = t.c1 || {}, c2 = t.c2 || {};
    function row(k, lab, v) {
      return '<div class="rec-f"><label for="tt-' + k + '">' + lab + '</label><input id="tt-' + k + '" name="' + k + '" type="number" min="0" max="100000" value="' + attr(v == null ? '' : v) + '"></div>';
    }
    return '<details class="rec-box" open><summary>Paste your results code (totals only)</summary>' +
      '<p class="muted">Fill in the Ai4Qi data sheet at work. Its Results tab shows one line starting <code>AI4QI v1</code>: copy it here. It holds totals only, never patient data.</p>' +
      '<form class="det-form" data-totals-code><div class="rec-f rec-wide"><label for="tc-code">Results code</label><textarea id="tc-code" name="code" rows="2" placeholder="AI4QI v1 | ' + attr(r.auditId) + ' | C1 30/42 n=45 | RE 37/40 n=41 | …"></textarea></div>' +
      '<div class="rec-actions"><button class="btn" type="submit">Use these results</button><span class="form-status" role="status" data-totals-status></span></div></form>' +
      '<details class="rec-box nested"><summary>Or type the totals</summary><form class="det-form" data-totals-hand>' +
      row('c1n', 'Cycle 1: records audited', c1.n) + row('c1met', 'Cycle 1: met the standard', c1.met) + row('c1den', 'Cycle 1: met + not met', c1.den) +
      row('c2n', 'Re-audit: records audited', c2.n) + row('c2met', 'Re-audit: met the standard', c2.met) + row('c2den', 'Re-audit: met + not met', c2.den) +
      '<div class="rec-actions"><button class="btn btn-secondary" type="submit">Save totals</button></div></form></details></details>';
  }

  /* --- schedule from the protocol timeline and the start date --- */
  function schedule(r) {
    var segs = parseTimeline(r.protocol.timeline) || [], start = r.details.startDate || todayIso();
    function find(re) { return segs.filter(function (s) { return re.test(s.label); })[0]; }
    function due(seg, fallbackWeeks) { return addDays(start, ((seg ? seg.b : fallbackWeeks) * 7) - 1); }
    return {
      setup: start,
      cycle1: due(find(/collect/i), 3),
      present: due(find(/analy|present/i), 4),
      change: due(find(/change/i), 5),
      reaudit: due(find(/re-?audit/i), 12),
      close: addDays(due(find(/re-?audit/i), 12), 14)
    };
  }
  function nextStep(r) {
    var i = stageIdx(r), st = runStats(r), n1 = st.cycles[0].n, n2 = st.cycles[1].n, want = +r.details.sampleSize || 30, sch = schedule(r);
    if (r.closed) return { text: 'Loop closed. Keep the change going and report it at governance.', due: null };
    var key = RUN_STAGES[i][0];
    var map = {
      setup: 'Add the audit lead, site and start date',
      cycle1: 'Collect cycle 1 data (' + n1 + ' of ' + want + ' entered)',
      present: 'Present the cycle 1 results and agree the change',
      change: 'Put the change in place and record it',
      reaudit: 'Collect re-audit data (' + n2 + ' of ' + want + ' entered)',
      close: 'Present the re-audit and close the loop'
    };
    return { text: map[key], due: sch[key], key: key };
  }

  /* --- de-identification: applied to every record before it is stored --- */
  var ID_HEADER = /(^|_)(name|names|surname|forename|first_?name|last_?name|nhs|nhs_?(no|num|number)|chi|mrn|hospital_?(no|num|number)|patient_?(id|no|number)|unit_?(no|number)|dob|date_?of_?birth|birth|address|street|postcode|post_?code|zip|phone|mobile|telephone|email|e_?mail|next_?of_?kin|nok|gp_?name)(_|$)/i;
  var RX = [
    [/\b\d{3}[\s-]?\d{3}[\s-]?\d{4}\b/g, '[number removed]'],                       // NHS / 10-digit numbers
    [/\b[A-Z]{1,2}\d[A-Z\d]?\s*\d[A-Z]{2}\b/gi, '[postcode removed]'],               // UK postcodes
    [/[\w.+-]+@[\w-]+\.[\w.-]+/g, '[email removed]'],                                  // email
    [/(\+44\s?|\b0)(\d[\s-]?){9,10}\b/g, '[phone removed]'],                           // UK phone
    [/\b(Mr|Mrs|Ms|Miss|Mx|Dr|Prof|Sister|Nurse)\.?\s+[A-Z][a-zA-Z'-]+(\s+[A-Z][a-zA-Z'-]+)?/g, '[name removed]'],
    [/\b(DOB|d\.o\.b\.?|born)\s*[:\-]?\s*\d{1,2}[\/.-]\d{1,2}[\/.-]\d{2,4}/gi, '[date of birth removed]'],
    [/\b[A-Z]\d{6,8}\b/g, '[number removed]'],                                         // hospital numbers like K1234567
    [/\b\d{7,12}\b/g, '[number removed]']
  ];
  function scrub(text, rep) {
    var s = String(text);
    RX.forEach(function (x) { s = s.replace(x[0], function () { rep.redacted++; return x[1]; }); });
    return s;
  }
  function isCodeField(f) { return /pseudonym|audit_?code|local_?code|patient_?code|hospital_number|study_?id|case_?(id|no|number)/i.test(f.field); }
  function normDate(v, withTime) {
    if (v == null || v === '') return '';
    if (typeof v === 'number' && v > 20000 && v < 80000) {       // Excel serial date
      var d = new Date(Math.round((v - 25569) * 86400000));
      return withTime ? d.toISOString().slice(0, 16).replace('T', ' ') : d.toISOString().slice(0, 10);
    }
    var s = String(v).trim(), m;
    if ((m = s.match(/^(\d{4})-(\d{2})$/))) return m[1] + '-' + m[2] + '-01';
    if ((m = s.match(/^(\d{4})-(\d{2})-(\d{2})(?:[T ](\d{2}):(\d{2}))?/))) return m[1] + '-' + m[2] + '-' + m[3] + (withTime && m[4] ? ' ' + m[4] + ':' + m[5] : '');
    if ((m = s.match(/^(\d{1,2})[\/.-](\d{1,2})[\/.-](\d{2,4})(?:\s+(\d{1,2}):(\d{2}))?/))) {
      var y = m[3].length === 2 ? '20' + m[3] : m[3];
      return y + '-' + ('0' + m[2]).slice(-2) + '-' + ('0' + m[1]).slice(-2) + (withTime && m[4] ? ' ' + ('0' + m[4]).slice(-2) + ':' + m[5] : '');
    }
    return '';
  }
  /* Clean one record against the template. codes: Map of original identifier → audit code. */
  function isFreeText(f) { return !isCodeField(f) && !/yes|date|number/i.test(f.type) && !(f.type === 'choice' && (f.options || []).length); }
  function cleanRecord(p, raw, codes, rep, opts) {
    var out = {}; opts = opts || {};
    (p.template || []).forEach(function (f) {
      var v = raw[f.field];
      if (v == null || v === '') return;
      if (isCodeField(f)) {
        var key = String(v).trim();
        if (/^[A-Z]{0,3}\d{1,4}$/i.test(key) && key.length <= 6) { out[f.field] = key.toUpperCase(); return; }   // already a short audit code
        if (!codes.map.has(key)) { codes.map.set(key, 'P' + ('00' + codes.next).slice(-3)); codes.next++; }
        out[f.field] = codes.map.get(key); rep.coded++;
        return;
      }
      if (/yes/i.test(f.type)) { var y = yn(v); if (y) out[f.field] = y; return; }
      if (f.type === 'date' || f.type === 'datetime') {
        var d = normDate(v, f.type === 'datetime' && !opts.monthOnly);
        if (d) out[f.field] = opts.monthOnly ? d.slice(0, 7) : d; else rep.badDates++;
        return;
      }
      if (opts.noFreeText && isFreeText(f)) { rep.freeDropped = (rep.freeDropped || 0) + 1; return; }
      if (f.type === 'number') { var n = parseFloat(String(v).replace(/[^\d.\-]/g, '')); if (!isNaN(n)) out[f.field] = n; return; }
      if (f.type === 'choice' && (f.options || []).length) {
        var hit = f.options.filter(function (o) { return o.toLowerCase() === String(v).trim().toLowerCase(); })[0];
        out[f.field] = hit || scrub(String(v).slice(0, 120), rep);
        return;
      }
      out[f.field] = scrub(String(v).slice(0, 300), rep);
    });
    return out;
  }
  function normHead(h) { return String(h || '').toLowerCase().replace(/[^a-z0-9]+/g, '_').replace(/^_|_$/g, ''); }
  function mapHeaders(p, heads) {
    var fields = (p.template || []).map(function (f) { return f.field; }), used = new Set();
    return heads.map(function (h) {
      var n = normHead(h);
      if (!n) return null;
      var hit = fields.filter(function (f) { return f === n && !used.has(f); })[0] ||
        fields.filter(function (f) { return !used.has(f) && n.length > 3 && (f.indexOf(n) === 0 || n.indexOf(f) === 0); })[0] || null;
      if (hit) used.add(hit);
      return hit;
    });
  }
  function parseCsv(text) {
    var rows = [], row = [], cell = '', q = false, i, c;
    text = String(text).replace(/^﻿/, '');
    var delim = (text.split('\n')[0].split('\t').length > text.split('\n')[0].split(',').length) ? '\t' : ',';
    for (i = 0; i < text.length; i++) {
      c = text[i];
      if (q) { if (c === '"') { if (text[i + 1] === '"') { cell += '"'; i++; } else q = false; } else cell += c; }
      else if (c === '"') q = true;
      else if (c === delim) { row.push(cell); cell = ''; }
      else if (c === '\n' || c === '\r') { if (c === '\r' && text[i + 1] === '\n') i++; row.push(cell); rows.push(row); row = []; cell = ''; }
      else cell += c;
    }
    if (cell !== '' || row.length) { row.push(cell); rows.push(row); }
    return rows.filter(function (r) { return r.some(function (x) { return String(x).trim() !== ''; }); });
  }
  var SHEETJS = 'vendor/xlsx.mini.min.js', sheetjsP = null;   // SheetJS 0.18.5, served from this site
  function loadScript(src, globalName) {
    return new Promise(function (res, rej) {
      if (window[globalName]) return res(window[globalName]);
      var s = document.createElement('script'); s.src = src; s.async = true;
      s.onload = function () { window[globalName] ? res(window[globalName]) : rej(new Error('load')); };
      s.onerror = function () { rej(new Error('load')); };
      document.head.appendChild(s);
    });
  }
  function readTable(file) {
    if (/\.(csv|tsv|txt)$/i.test(file.name)) return file.text().then(parseCsv);
    sheetjsP = sheetjsP || loadScript(SHEETJS, 'XLSX');
    return Promise.all([sheetjsP, file.arrayBuffer()]).then(function (r) {
      var wb = r[0].read(r[1], { type: 'array' });
      var name = wb.SheetNames.filter(function (n) { return /data/i.test(n); })[0] || wb.SheetNames[0];
      return r[0].utils.sheet_to_json(wb.Sheets[name], { header: 1, raw: true, defval: '' });
    });
  }
  /* Find the header row, map columns, clean every record. Returns a preview for the user to confirm. */
  function prepareImport(p, table, startCode, opts) {
    var best = 0, bestHits = -1;
    for (var i = 0; i < Math.min(table.length, 12); i++) {
      var hits = mapHeaders(p, table[i]).filter(Boolean).length;
      if (hits > bestHits) { bestHits = hits; best = i; }
    }
    var heads = table[best] || [], map = mapHeaders(p, heads), rep = { redacted: 0, coded: 0, badDates: 0 }, codes = { map: new Map(), next: startCode || 1 };
    var dropped = [], kept = [];
    heads.forEach(function (h, j) {
      if (!String(h).trim()) return;
      if (map[j] && !(ID_HEADER.test(normHead(h)) && !isCodeField({ field: map[j] }))) kept.push([h, map[j]]);
      else { dropped.push(String(h)); map[j] = null; }
    });
    var cycCol = heads.map(normHead).indexOf('cycle'), re = [];
    if (cycCol >= 0) dropped = dropped.filter(function (h) { return normHead(h) !== 'cycle'; });
    var rows = [];
    table.slice(best + 1).forEach(function (r) {
      var raw = {}; map.forEach(function (f, j) { if (f) raw[f] = r[j]; });
      var o = cleanRecord(p, raw, codes, rep, opts);
      if (!Object.keys(o).length) return;
      if (cycCol >= 0 && /re-?audit|cycle\s*2/i.test(String(r[cycCol] || ''))) re.push(o); else rows.push(o);
    });
    return { rows: rows, reRows: re, kept: kept, dropped: dropped, rep: rep, nextCode: codes.next };
  }

  /* --- files: download on the hosted site, the downloads capability inside Claude --- */
  function saveFile(filename, blob) {
    if (window.AI4QI_EMBED && window.claude && typeof window.claude.use === 'function') {
      return window.claude.use('downloads').then(function (d) {
        if (!d) throw { code: 'unavailable' };
        return d.save({ filename: filename, data: blob });
      });
    }
    var url = URL.createObjectURL(blob), a = document.createElement('a');
    a.href = url; a.download = filename; document.body.appendChild(a); a.click(); a.remove();
    setTimeout(function () { URL.revokeObjectURL(url); }, 4000);
    return Promise.resolve();
  }
  function fileSlug(s) { return String(s || 'audit').toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '').slice(0, 40) || 'audit'; }
  function exporter() {
    if (window.AI4QI_EXPORT) return Promise.resolve(window.AI4QI_EXPORT);
    return loadScript('export.js', 'AI4QI_EXPORT');
  }
  function sayIn(el, text) { if (el) el.textContent = text; }
  function downloadError(e) {
    var c = e && e.code;
    if (c === 'declined') return 'Download cancelled.';
    if (c === 'unavailable' || c === 'not_granted') return 'Downloads are not available in this view. Open Ai4Qi in a browser to download.';
    return (e && e.message && !c) ? e.message : 'The file could not be created. Please try again.';
  }

  /* --- reminders: opt-in, signed-in users, metadata only --- */
  var syncTimers = {};
  function syncRun(r) {
    if (!BE.url || !BE.user || r.demo) return;           // example audits never leave the device
    clearTimeout(syncTimers[r.id]);
    syncTimers[r.id] = setTimeout(function () {
      sbClient().then(function (c) {
        var i = stageIdx(r), status = STAGE_STATUS[i], jobs = [];
        jobs.push(c.from('my_audits').upsert({ user_id: BE.user.id, audit_id: r.auditId, status: status }, { onConflict: 'user_id,audit_id' }));
        if (r.reminders && !r.closed) {
          var ns = nextStep(r);
          jobs.push(c.from('run_reminders').upsert({ user_id: BE.user.id, run_id: r.id, audit_title: trunc(r.protocol.question, 190),
            next_step: trunc(ns.text, 190), due_date: ns.due || addDays(todayIso(), 7), email_opt_in: true }, { onConflict: 'user_id,run_id' }));
        } else jobs.push(c.from('run_reminders').delete().eq('user_id', BE.user.id).eq('run_id', r.id));
        return Promise.all(jobs);
      }).catch(function () {});
    }, 800);
  }

  /* --- pages --- */
  function stageStepper(r) {
    var cur = stageIdx(r), sch = schedule(r);
    return '<ol class="run-steps" aria-label="Audit stages">' + RUN_STAGES.map(function (s, i) {
      var state = i < cur ? 'is-done' : (i === cur ? 'is-now' : '');
      return '<li class="' + state + '"><span class="rs-dot" aria-hidden="true">' + (i < cur ? '✓' : i + 1) + '</span><span class="rs-t">' + esc(s[1]) +
        '</span><span class="rs-d">' + (i === 0 ? '' : 'by ' + esc(dateGBs(sch[s[0]]))) + '</span>' + (i === cur ? '<span class="sr-only"> (current stage)</span>' : '') + '</li>';
    }).join('') + '</ol>';
  }
  function pctBar(label, s, target, cls) {
    var v = s.pct == null ? 0 : s.pct;
    return '<div class="sp-row ' + cls + '"><span>' + esc(label) + '</span><div class="sp-track">' +
      (target ? '<i class="sp-target" style="left:' + Math.min(target.value, 100) + '%" title="Target ' + attr(target.text) + '"></i>' : '') +
      '<div class="sp-fill" style="width:' + v + '%"></div></div><b>' + (s.pct == null ? '–' : Math.round(s.pct) + '%') + '</b></div>';
  }
  function vaultGate() {
    if (V.key) return false;
    var pending = S.pendingChoose && anyAudit(S.pendingChoose);
    var intro = pending ? '<p class="notice notice-ok">You chose <strong>' + esc(trunc(pending.question, 120)) + '</strong>. ' + (vaultSet() ? 'Unlock' : 'Set a passcode') + ' to add it to My audits.</p>' : '';
    var html;
    if (!vaultSet()) {
      html = '<article class="doc narrow vault"><h1>Protect your audit data</h1>' + intro +
        '<p class="prose">Your audit records stay on this device. Set a passcode so they are stored encrypted and lock after 15 minutes without use. ' +
        'Ai4Qi never sees the passcode, so it cannot be reset: if you forget it, you can only erase the audits on this device.</p>' +
        (V.legacy ? '<p class="notice notice-warn">You have ' + V.legacy.length + ' audit(s) saved before passcodes were added. They will be encrypted with your new passcode.</p>' : '') +
        '<form class="stack-form" data-vault-new><label for="vp1">New passcode (at least 8 characters)</label><input id="vp1" name="p1" type="password" minlength="8" autocomplete="new-password" required>' +
        '<label for="vp2">Type it again</label><input id="vp2" name="p2" type="password" minlength="8" autocomplete="new-password" required>' +
        '<label class="check"><input type="checkbox" name="shared"><span><strong>This is a shared computer.</strong> Keep audits only until the browser is closed, then delete them. Download a backup to keep your work.</span></label>' +
        '<button class="btn" type="submit">Set passcode</button><p class="form-status" role="status" data-vault-status></p></form>' +
        (DEMO ? '<p><button type="button" class="btn demo-btn" data-demo-vault>Demo: use a demo passcode</button></p>' : '') + privacyLink() + '</article>';
    } else {
      html = '<article class="doc narrow vault"><h1>My audits are locked</h1>' + intro +
        '<p class="prose">Enter your passcode to open your audits on this device' + (V.meta.shared ? ' (shared-computer mode: they are deleted when the browser closes)' : '') + '.</p>' +
        '<form class="stack-form" data-vault-open><label for="vpo">Passcode</label><input id="vpo" name="p" type="password" autocomplete="current-password" required>' +
        '<button class="btn" type="submit">Unlock</button><p class="form-status" role="status" data-vault-status></p></form>' +
        (DEMO ? '<p><button type="button" class="btn demo-btn" data-demo-unlock>Demo: unlock with the demo passcode</button></p>' : '') +
        '<details class="rec-box"><summary>Forgotten your passcode?</summary><p class="prose">It cannot be recovered. You can erase all audits on this device and start again (restore from a backup file if you have one).</p>' +
        '<button type="button" class="btn btn-secondary" data-vault-erase>Erase all audits on this device</button> <span data-vault-erase-confirm></span></details></article>';
    }
    page(html, 'My audits', 'my-audits');
    return true;
  }
  function afterUnlock() {
    var id = S.pendingChoose; S.pendingChoose = null;
    var p = id && anyAudit(id);
    if (p) chooseAudit(p); else route();
  }
  document.addEventListener('submit', function (e) {
    var f = e.target, st = f.querySelector && f.querySelector('[data-vault-status]');
    if (f.matches && f.matches('[data-vault-new]')) {
      e.preventDefault();
      var p1 = f.elements.p1.value, p2 = f.elements.p2.value;
      if (p1.length < 8) { sayIn(st, 'Use at least 8 characters.'); return; }
      if (p1 !== p2) { sayIn(st, 'The two passcodes do not match.'); return; }
      sayIn(st, 'Setting up encryption…');
      vaultCreate(p1, f.elements.shared.checked).then(afterUnlock, function () { sayIn(st, 'This browser cannot store encrypted data here. Try a different browser.'); });
    } else if (f.matches && f.matches('[data-vault-open]')) {
      e.preventDefault();
      sayIn(st, 'Unlocking…');
      vaultUnlock(f.elements.p.value).then(afterUnlock, function () { sayIn(st, 'That passcode is not right.'); f.elements.p.select(); });
    }
  });
  document.addEventListener('click', function (e) {
    if (e.target.closest('[data-vault-erase]')) {
      main.querySelector('[data-vault-erase-confirm]').innerHTML = '<strong>Erase everything?</strong> <button type="button" class="btn btn-secondary" data-vault-erase-yes>Erase</button>';
    } else if (e.target.closest('[data-vault-erase-yes]')) { vaultErase(); route(); }
    else if (e.target.closest('[data-vault-lock]')) { vaultLock(); route(); }
  });

  function renderRuns() {
    if (vaultGate()) return;
    var list = Array.from(S.runs.values()).sort(function (a, b) { return (a.closed - b.closed) || String(b.updated).localeCompare(String(a.updated)); });
    var body = list.length ? '<ul class="run-list">' + list.map(function (r) {
      var st = runStats(r), ns = nextStep(r), i = stageIdx(r), t = st.target;
      if (r.demo) ns.due = null;
      var overdue = ns.due && ns.due < todayIso();
      return '<li class="run-card"><div class="rc-top"><span class="id-tag">' + esc(r.auditId) + '</span>' +
        (r.closed ? badge('Loop closed', 'ok') : badge(RUN_STAGES[i][1], 'primary')) + (r.demo ? badge('Example data', 'demo') : '') + '</div>' +
        '<h2><a href="#/run/' + attr(r.id) + '">' + esc(r.details.title || r.protocol.question) + '</a></h2>' +
        '<p class="meta">' + esc([r.details.site, r.details.lead].filter(Boolean).join(' · ') || 'Details not added yet') + '</p>' +
        '<div class="sp-bars">' + pctBar('Cycle 1', st.cycles[0], t, 'before') + (st.cycles[1].n ? pctBar('Re-audit', st.cycles[1], t, 'after') : '') + '</div>' +
        '<p class="rc-next' + (overdue ? ' is-late' : '') + '"><strong>Next:</strong> ' + esc(ns.text) + (ns.due ? ' · ' + (overdue ? 'was due ' : 'due ') + esc(dateGBs(ns.due)) : '') + '</p></li>';
    }).join('') + '</ul>' :
      '<div class="empty"><p><strong>No audits yet.</strong> Build one or pick a ready-made protocol, then press <em>Choose this audit</em>. It will appear here with its data sheet, deadlines and results.</p>' +
      '<p><a class="btn" href="#/">Build an audit</a> <a class="btn btn-secondary" href="#/suggest">See suggested audits</a></p></div>';
    var pasteBox = '<details class="rec-box paste-any"><summary>Paste a results code</summary><form class="det-form" data-paste-any>' +
      '<div class="rec-f rec-wide"><label for="pa-code">Results code from the Ai4Qi data sheet</label><textarea id="pa-code" name="code" rows="2" placeholder="AI4QI v1 | NNA-245 | C1 30/42 n=45 | …"></textarea></div>' +
      '<div class="rec-actions"><button class="btn" type="submit">Use these results</button><span class="form-status" role="status" data-paste-status></span></div></form></details>';
    page('<div class="page-head"><h1>My audits</h1><p class="page-intro">Everything you record here stays on this device. ' + privacyLink() + '</p></div>' + pasteBox + body +
      '<div class="restore"><label class="file-pick"><input type="file" accept=".json,application/json" data-restore><span class="btn btn-secondary">Restore a backup</span></label>' +
      '<span class="form-status" role="status" data-restore-status></span><button type="button" class="link-btn" data-vault-lock>Lock now</button></div>', 'My audits', 'my-audits');
  }
  function privacyLink() { return '<a href="#/privacy">How your data is protected</a>'; }

  function recordForm(r, ck) {
    var p = r.protocol, o = r.options || {};
    return '<form class="rec-form" data-rec-form="' + attr(ck) + '"><div class="rec-grid">' + (p.template || []).filter(function (f) { return !(o.noFreeText && isFreeText(f)); }).map(function (f, i) {
      var id = 'rf-' + ck + '-' + i, lab = '<label for="' + id + '">' + esc(fieldLabel(f.field)) + (f.note ? ' <span class="muted">' + esc(f.note) + '</span>' : '') + '</label>', ctl;
      if (/yes/i.test(f.type)) ctl = '<select id="' + id + '" name="' + attr(f.field) + '"><option value=""></option><option>Yes</option><option>No</option><option>N/A</option></select>';
      else if (f.type === 'choice') ctl = '<select id="' + id + '" name="' + attr(f.field) + '"><option value=""></option>' + (f.options || []).map(function (o) { return '<option>' + esc(o) + '</option>'; }).join('') + '</select>';
      else if ((f.type === 'date' || f.type === 'datetime') && o.monthOnly) ctl = '<input id="' + id + '" name="' + attr(f.field) + '" type="month">';
      else if (f.type === 'date') ctl = '<input id="' + id + '" name="' + attr(f.field) + '" type="date">';
      else if (f.type === 'datetime') ctl = '<input id="' + id + '" name="' + attr(f.field) + '" type="datetime-local">';
      else if (f.type === 'number') ctl = '<input id="' + id + '" name="' + attr(f.field) + '" type="number" step="any" inputmode="decimal">';
      else ctl = '<input id="' + id + '" name="' + attr(f.field) + '" type="text" maxlength="300" autocomplete="off"' + (isCodeField(f) ? ' placeholder="e.g. P001"' : ' placeholder="No names or identifiers"') + '>' +
        (isCodeField(f) ? '' : '<span class="ft-warn">Free text: never write names, numbers or anything that could identify a patient.</span>');
      return '<div class="rec-f">' + lab + ctl + '</div>';
    }).join('') + '</div><div class="rec-actions"><button class="btn" type="submit">Add record</button><span class="form-status" role="status" data-rec-status></span></div></form>';
  }
  function rowsTable(r, ck) {
    var p = r.protocol, rows = r.cycles[ck].rows, t = p.template || [];
    if (!rows.length) return '<p class="muted">No records yet.</p>';
    var shown = rows.slice(-200);
    return '<div class="table-wrap"><table class="rows-t"><thead><tr><th scope="col">#</th>' + t.map(function (f) { return '<th scope="col">' + esc(fieldLabel(f.field)) + '</th>'; }).join('') +
      '<th scope="col"><span class="sr-only">Remove</span></th></tr></thead><tbody>' + shown.map(function (row, j) {
        var idx = rows.length - shown.length + j;
        return '<tr><td class="num-col">' + (idx + 1) + '</td>' + t.map(function (f) { return '<td>' + esc(row[f.field] == null ? '' : row[f.field]) + '</td>'; }).join('') +
          '<td><button type="button" class="link-btn" data-rec-del="' + ck + ':' + idx + '" aria-label="Remove record ' + (idx + 1) + '">Remove</button></td></tr>';
      }).join('') + '</tbody></table></div>' + (rows.length > shown.length ? '<p class="muted">Showing the last 200 of ' + fmt(rows.length) + ' records.</p>' : '');
  }
  function cyclePanel(r, ck, st) {
    var s = st.cycles[ck === 'c1' ? 0 : 1], want = +r.details.sampleSize || 30, t = st.target;
    var met = s.pct != null && t ? (t.op === '≤' || t.op === '<' ? s.pct <= t.value : s.pct >= t.value) : null;
    return '<div class="cycle-panel">' +
      '<div class="kpis"><div class="kpi"><span>Records</span><b>' + s.n + '<small> / ' + want + '</small></b><div class="mini-track"><i style="width:' + Math.min(100, s.n / want * 100) + '%"></i></div></div>' +
      '<div class="kpi"><span>Met the standard</span><b>' + (s.pct == null ? '–' : Math.round(s.pct) + '%') + '</b><small>' + s.passN + ' of ' + (s.passN + s.failN) + '</small></div>' +
      '<div class="kpi"><span>Target</span><b>' + (t ? esc(t.op + t.value + '%') : '–') + '</b><small>' + (met == null ? 'No data yet' : met ? 'Met' : 'Not met') + '</small></div></div>' +
      (ck === 'c1' ? totalsBox(r) : '') +
      (DEMO ? '<p><button type="button" class="btn demo-btn" data-demo-fill="' + ck + '">Demo: fill ' + (ck === 'c1' ? 'cycle 1' : 'the re-audit') + ' with ' + want + ' example records</button></p>' : '') +
      '<details class="rec-box"><summary>Add a record by hand</summary>' + recordForm(r, ck) + '</details>' +
      '<details class="rec-box"><summary>Upload a spreadsheet (Excel or CSV)</summary>' +
      '<p class="muted">Use the Ai4Qi data sheet, or any sheet whose column headings match the template. Columns that are not part of the audit are left out, and patient identifiers are removed before anything is stored.</p>' +
      '<label class="file-pick"><input type="file" accept=".xlsx,.xls,.csv,.tsv" data-import="' + ck + '"><span class="btn btn-secondary">Choose a file</span></label>' +
      '<div data-import-preview="' + ck + '"></div></details>' +
      rowsTable(r, ck) + '</div>';
  }
  function renderRun(id) {
    if (vaultGate()) return;
    var r = S.runs.get(id);
    if (!r) return renderNotFound();
    var p = r.protocol, d = r.details, st = runStats(r), ns = nextStep(r), i = stageIdx(r), ck = S.view.runTab || (i >= 4 ? 'c2' : 'c1');
    if (r.demo) ns.due = null;                                   // example audits are backdated: no "was due" warnings on stage
    var overdue = ns.due && ns.due < todayIso() && !r.closed;
    var detailsForm = '<form class="det-form" data-run-details>' +
      [['title', 'Audit title', 'text'], ['site', 'Hospital or practice', 'text'], ['department', 'Department or ward', 'text'], ['lead', 'Audit lead', 'text'],
        ['team', 'Team members', 'text'], ['supervisor', 'Supervising consultant', 'text'], ['startDate', 'Start date', 'date'], ['sampleSize', 'Records per cycle', 'number']]
        .map(function (x) {
          return '<div class="rec-f"><label for="rd-' + x[0] + '">' + x[1] + '</label><input id="rd-' + x[0] + '" name="' + x[0] + '" type="' + x[2] + '"' +
            (x[2] === 'number' ? ' min="1" max="2000"' : ' maxlength="200"') + ' value="' + attr(d[x[0]] == null ? '' : d[x[0]]) + '"></div>';
        }).join('') +
      '<fieldset class="rec-wide opt-set"><legend>Extra data protection</legend>' +
      '<label class="check"><input type="checkbox" name="monthOnly"' + ((r.options || {}).monthOnly ? ' checked' : '') + '><span>Store dates as month and year only (existing dates are shortened too)</span></label>' +
      '<label class="check"><input type="checkbox" name="noFreeText"' + ((r.options || {}).noFreeText ? ' checked' : '') + '><span>Switch off free-text fields (existing free text is deleted)</span></label></fieldset>' +
      '<div class="rec-actions"><button class="btn" type="submit">Save details</button><span class="form-status" role="status" data-det-status></span></div></form>';
    var remind = BE.url ? (BE.user ?
      '<label class="check"><input type="checkbox" data-run-remind' + (r.reminders ? ' checked' : '') + '><span>Email me when a step is due. Only the audit question, the next step and its date are sent; never your data.</span></label>' :
      '<p class="muted"><a href="#/account">Sign in</a> to get email reminders when a step is due.</p>') : '';
    var changeForm = '<form class="det-form" data-run-change><div class="rec-f rec-wide"><label for="rc-desc">What did you change?</label><textarea id="rc-desc" name="description" rows="3" maxlength="600">' + esc(r.changeMade.description || '') + '</textarea></div>' +
      '<div class="rec-f"><label for="rc-date">Date it started</label><input id="rc-date" name="date" type="date" value="' + attr(r.changeMade.date || '') + '"></div>' +
      '<div class="rec-actions"><button class="btn" type="submit">Save change</button><span class="form-status" role="status" data-chg-status></span></div>' +
      '<p class="muted">Planned in the protocol: ' + esc(p.change) + '</p></form>';
    var results = '<div class="sp-bars">' + pctBar('Cycle 1', st.cycles[0], st.target, 'before') + pctBar('Re-audit', st.cycles[1], st.target, 'after') + '</div>' +
      (st.target ? '<p class="muted">Dashed line: target ' + esc(st.target.text) + '</p>' : '') +
      st.breakdowns.map(function (b) {
        var all = b.cycles.c1.concat(b.cycles.c2), max = Math.max.apply(null, all.map(function (x) { return x.n; }).concat([1]));
        var two = b.cycles.c1.length && b.cycles.c2.length, opts = [];
        b.cycles.c1.concat(b.cycles.c2).forEach(function (x) { if (opts.indexOf(x.option) < 0) opts.push(x.option); });
        function n(k, o) { var h = b.cycles[k].filter(function (x) { return x.option === o; })[0]; return h ? h.n : 0; }
        function bar(k, o, cls) { return '<i class="' + cls + '" style="width:' + (n(k, o) / max * 100) + '%"></i>'; }
        return '<div class="sub"><h3>' + esc(b.label) + '</h3>' + (two ? '<p class="bd-key"><span class="k-before">Cycle 1</span><span class="k-after">Re-audit</span></p>' : '') +
          '<ul class="bd-bars' + (two ? ' is-two' : '') + '">' + opts.map(function (o) {
            return two ? '<li><span>' + esc(o) + '</span><div class="bd-pair">' + bar('c1', o, 'before') + bar('c2', o, 'after') + '</div><b>' + n('c1', o) + ' → ' + n('c2', o) + '</b></li>' :
              '<li><span>' + esc(o) + '</span><i style="width:' + (n(b.cycles.c1.length ? 'c1' : 'c2', o) / max * 100) + '%"></i><b>' + n(b.cycles.c1.length ? 'c1' : 'c2', o) + '</b></li>';
          }).join('') + '</ul></div>';
      }).join('');
    var dl = '<div class="out-grid">' +
      '<button class="out-btn" type="button" data-run-xlsx><b>Data sheet</b><span>Excel, with drop-downs</span></button>' +
      '<button class="out-btn" type="button" data-run-pptx><b>Results presentation</b><span>PowerPoint, ready for your meeting</span></button>' +
      '<button class="out-btn" type="button" data-run-csv><b>Your records</b><span>CSV, de-identified</span></button>' +
      '<button class="out-btn" type="button" data-run-backup><b>Backup</b><span>To move this audit to another device</span></button>' +
      (window.AI4QI_EMBED ? '' : '<button class="out-btn" type="button" data-run-ics><b>Calendar</b><span>Add the deadlines</span></button>') +
      '</div><p class="export-warn">These files hold de-identified patient records. Keep them on your organisation\'s systems and share them only inside it.</p><p class="form-status" role="status" data-out-status></p>';
    var stageBtn = r.closed ? '<button class="btn btn-secondary" type="button" data-run-stage="reopen">Reopen</button>' :
      (DEMO ? '<button class="btn demo-btn" type="button" data-demo-step>' + esc(demoStepLabel(r)) + '</button>' : '') +
      '<button class="btn" type="button" data-run-stage="next">' + (i === RUN_STAGES.length - 1 ? 'Mark the loop closed' : 'Done – go to ' + esc(RUN_STAGES[i + 1][1].toLowerCase())) + '</button>' +
      (i > 0 ? '<button class="btn btn-secondary" type="button" data-run-stage="back">Back a stage</button>' : '');

    page('<nav class="breadcrumb" aria-label="Breadcrumb"><a href="#/">Home</a> › <a href="#/my-audits">My audits</a></nav>' +
      '<article class="doc run" data-run="' + attr(r.id) + '"><header class="doc-head"><div class="eyebrow"><span class="id-tag">' + esc(r.auditId) + '</span>' +
      (r.closed ? badge('Loop closed', 'ok') : badge(RUN_STAGES[i][1], 'primary')) + (r.demo ? badge('Example data', 'demo') : '') + '<a href="' + auditHref(r.auditId) + '">View protocol</a></div>' +
      '<h1>' + esc(d.title || p.question) + '</h1>' + stageStepper(r) +
      '<div class="next-card' + (overdue ? ' is-late' : '') + '"><div><p class="nc-label">' + (r.closed ? 'Done' : 'Next step') + (ns.due ? ' · ' + (overdue ? 'was due ' : 'due ') + esc(dateGBs(ns.due)) : '') + '</p>' +
      '<p class="nc-text">' + esc(ns.text) + '</p></div><div class="nc-actions">' + stageBtn + '</div></div>' +
      '<p class="privacy-note"><svg width="16" height="16" viewBox="0 0 24 24" aria-hidden="true"><path d="M12 3l7 3v5c0 4.5-3 8.3-7 10-4-1.7-7-5.5-7-10V6l7-3z" fill="none" stroke="currentColor" stroke-width="2" stroke-linejoin="round"/></svg>' +
      'Stored encrypted on this device only; identifiers are removed as records are entered. ' + privacyLink() +
      ' <button type="button" class="link-btn" data-vault-lock>Lock</button></p></header>' +
      sec(1, 'Audit details', detailsForm + remind + supervisorBox(p, r)) +
      sec(2, 'Data', '<div class="tabs" role="tablist"><button type="button" role="tab" data-run-tab="c1" aria-selected="' + (ck === 'c1') + '">Cycle 1 <span class="count">' + st.cycles[0].n + '</span></button>' +
        '<button type="button" role="tab" data-run-tab="c2" aria-selected="' + (ck === 'c2') + '">Re-audit <span class="count">' + st.cycles[1].n + '</span></button></div>' +
        '<div role="tabpanel">' + cyclePanel(r, ck, st) + '</div>') +
      sec(3, 'Results', results) +
      sec(4, 'The change', changeForm) +
      sec(5, 'Files for you', dl) +
      '<section class="danger-zone"><button type="button" class="link-btn" data-run-delete>Delete this audit and its data from this device</button><span data-del-confirm></span></section>' +
      '</article>', 'My audit', 'my-audits');
  }

  var LEGAL_PAGES = ['terms', 'privacy-notice', 'cookies', 'accessibility', 'security'], LEGAL = null;
  function renderLegal(slug) {
    function show(all) {
      var d = all && all[slug];
      if (!d) return renderNotFound();
      page('<article class="doc narrow legal">' + d.html + '</article>', d.title, '');
      var h = main.querySelector('h1'); if (h) h.setAttribute('tabindex', '-1');
      main.querySelectorAll('.legal a[href^="http"]').forEach(function (x) { x.setAttribute('target', '_blank'); x.setAttribute('rel', 'noopener'); });
    }
    if (LEGAL) return show(LEGAL);
    page('<p class="muted">Loading…</p>', '', '');
    getJSON('data/legal.json').then(function (d) { LEGAL = d; if (parseHash().parts[0] === slug) show(d); }, function () { renderNotFound(); });
  }

  function renderPrivacy() {
    page('<article class="doc narrow"><h1>How your audit data is protected</h1>' +
      '<p class="prose"><strong>Your audit records stay on your device.</strong> Records you type or upload in My audits are kept in this browser only. They are never sent to Ai4Qi, and we cannot see them.</p>' +
      '<p class="prose"><strong>Encrypted with your passcode.</strong> Records are stored encrypted (AES-256) with a key made from a passcode only you know. The audits lock after 15 minutes without use. On a shared computer, choose shared-computer mode and everything is deleted when the browser closes. We cannot reset a forgotten passcode.</p>' +
      '<p class="prose"><strong>De-identified as they are entered.</strong> Before a record is stored, Ai4Qi:</p><ul class="prose">' +
      '<li>keeps only the columns that are part of the audit template and leaves out everything else (for example name, NHS number, date of birth or address columns);</li>' +
      '<li>replaces patient or hospital numbers with audit codes (P001, P002…);</li>' +
      '<li>removes NHS numbers and other long numbers, postcodes, phone numbers, email addresses, dates of birth and names written with a title (Mr, Mrs, Dr…) from any text;</li>' +
      '<li>can store dates as month and year only, and can switch free-text fields off, for each audit.</li></ul>' +
      '<p class="prose"><strong>This is de-identified, not anonymous, data.</strong> Dates and details together can sometimes identify a person, so treat your records as patient data under your organisation\'s rules. Automatic checks cannot catch every identifier written in free text: never type names or numbers. Register the audit with your audit department before you start.</p>' +
      '<p class="prose"><strong>Files you download</strong> (data sheet, presentation, records, backup) are made on your device. Records and backups contain de-identified patient data: keep them on your organisation\'s systems.</p>' +
      '<p class="prose"><strong>Reminders.</strong> If you sign in and turn on email reminders, only the audit question, the next step, its due date and record counts are sent to our server, never records.</p>' +
      '<p class="prose"><strong>No outside code.</strong> Every script this page runs is served by Ai4Qi itself' +
      (window.AI4QI_EMBED ? '.' : ', fonts included, and the page blocks scripts and connections to anywhere else.') + '</p></article>', 'Privacy', '');
  }

  /* --- events --- */
  function curRun() { var el = main.querySelector('[data-run]'); return el ? S.runs.get(el.getAttribute('data-run')) : null; }
  document.addEventListener('click', function (e) {
    var ch = e.target.closest('[data-choose]');
    if (ch) { var p = anyAudit(ch.getAttribute('data-choose')); if (p) chooseAudit(p); return; }
    var r = curRun(); if (!r) return;
    var tab = e.target.closest('[data-run-tab]');
    if (tab) { S.view.runTab = tab.getAttribute('data-run-tab'); renderRunKeep(r, '[data-run-tab="' + S.view.runTab + '"]'); return; }
    var stg = e.target.closest('[data-run-stage]');
    if (stg) {
      var a = stg.getAttribute('data-run-stage'), i = stageIdx(r);
      if (a === 'reopen') { r.closed = false; r.stage = RUN_STAGES[RUN_STAGES.length - 1][0]; }
      else if (a === 'back') { if (r.closed) r.closed = false; else r.stage = RUN_STAGES[Math.max(0, i - 1)][0]; }
      else if (i >= RUN_STAGES.length - 1) r.closed = true;
      else r.stage = RUN_STAGES[i + 1][0];
      if (r.stage === 'reaudit' || r.stage === 'close') S.view.runTab = 'c2';
      runPut(r); renderRunKeep(r, '[data-run-stage]');
      return;
    }
    var del = e.target.closest('[data-rec-del]');
    if (del) { var k = del.getAttribute('data-rec-del').split(':'); r.cycles[k[0]].rows.splice(+k[1], 1); runPut(r); renderRunKeep(r, '[data-run-tab="' + k[0] + '"]'); return; }
    if (e.target.closest('[data-run-delete]')) {
      var box = main.querySelector('[data-del-confirm]');
      box.innerHTML = ' <strong>Delete permanently?</strong> <button type="button" class="btn btn-secondary" data-run-delete-yes>Delete</button> <button type="button" class="link-btn" data-run-delete-no>Keep it</button>';
      return;
    }
    if (e.target.closest('[data-run-delete-no]')) { main.querySelector('[data-del-confirm]').innerHTML = ''; return; }
    if (e.target.closest('[data-run-delete-yes]')) {
      S.runs.delete(r.id); runsSave(); r.reminders = false; syncRun(r);
      location.hash = '#/my-audits'; return;
    }
    var out = main.querySelector('[data-out-status]'), name = fileSlug(r.details.title || r.auditId);
    if (e.target.closest('[data-run-xlsx]')) {
      sayIn(out, 'Making the data sheet…');
      exporter().then(function (x) { return x.templateXlsx(r.protocol, {}); }).then(function (b) { return saveFile(name + '-data-sheet.xlsx', b); })
        .then(function () { sayIn(out, 'Data sheet ready.'); }, function (err) { sayIn(out, downloadError(err)); });
      return;
    }
    if (e.target.closest('[data-run-pptx]')) {
      sayIn(out, 'Making the presentation…');
      exporter().then(function (x) { return x.deckPptx(r, runStats(r)); }).then(function (b) { return saveFile(name + '-results.pptx', b); })
        .then(function () { sayIn(out, 'Presentation ready.'); }, function (err) { sayIn(out, downloadError(err)); });
      return;
    }
    if (e.target.closest('[data-run-csv]')) {
      var t = r.protocol.template || [], lines = [['cycle'].concat(t.map(function (f) { return f.field; }))];
      ['c1', 'c2'].forEach(function (c) { r.cycles[c].rows.forEach(function (row) { lines.push([c === 'c1' ? 'Cycle 1' : 'Re-audit'].concat(t.map(function (f) { return row[f.field] == null ? '' : row[f.field]; }))); }); });
      var csv = lines.map(function (l) { return l.map(function (v) { v = String(v); return /[",\n]/.test(v) ? '"' + v.replace(/"/g, '""') + '"' : v; }).join(','); }).join('\n') + '\n';
      saveFile(name + '-records.csv', new Blob([csv], { type: 'text/csv' })).then(function () { sayIn(out, 'Records downloaded.'); }, function (err) { sayIn(out, downloadError(err)); });
      return;
    }
    if (e.target.closest('[data-run-backup]')) {
      seal(V.key, JSON.stringify(r)).then(function (box) {
        var file = { ai4qi_backup: 2, note: 'Encrypted Ai4Qi audit backup. Open it in Ai4Qi > My audits > Restore a backup, with the passcode used when it was made.', salt: V.meta.salt, iter: V.meta.iter || PBKDF2_ITER, box: box };
        return saveFile(name + '-ai4qi-backup.json', new Blob([JSON.stringify(file)], { type: 'application/json' }));
      })
        .then(function () { sayIn(out, 'Backup downloaded. Open it from My audits on another device to continue there.'); }, function (err) { sayIn(out, downloadError(err)); });
      return;
    }
    if (e.target.closest('[data-run-ics]')) {
      var sch = schedule(r), evs = RUN_STAGES.slice(1).map(function (s) { return { date: sch[s[0]], title: 'Audit: ' + s[1], description: r.protocol.question }; });
      exporter().then(function (x) { return saveFile(name + '-deadlines.ics', new Blob([x.ics(r, evs)], { type: 'text/calendar' })); })
        .then(function () { sayIn(out, 'Calendar file downloaded.'); }, function (err) { sayIn(out, downloadError(err)); });
    }
  });
  function renderRunKeep(r, focusSel) {
    var y = window.scrollY; renderRun(r.id); window.scrollTo(0, y);
    var f = focusSel && main.querySelector(focusSel); if (f) f.focus({ preventScroll: true });
  }
  document.addEventListener('submit', function (e) {
    var r = curRun(); if (!r) return;
    var f = e.target, fd = new FormData(f);
    if (f.matches('[data-run-details]')) {
      e.preventDefault();
      fd.forEach(function (v, k) { if (k === 'monthOnly' || k === 'noFreeText') return; r.details[k] = k === 'sampleSize' ? Math.max(1, Math.min(2000, +v || 30)) : String(v).slice(0, 200); });
      r.options = { monthOnly: fd.has('monthOnly'), noFreeText: fd.has('noFreeText') };
      ['c1', 'c2'].forEach(function (c) {
        r.cycles[c].rows.forEach(function (row) {
          (r.protocol.template || []).forEach(function (tf) {
            if (row[tf.field] == null) return;
            if (r.options.monthOnly && /date/.test(tf.type)) row[tf.field] = String(row[tf.field]).slice(0, 7);
            if (r.options.noFreeText && isFreeText(tf)) delete row[tf.field];
          });
        });
      });
      if (r.stage === 'setup' && r.details.lead && r.details.site) r.stage = 'cycle1';
      var ok = runPut(r); renderRunKeep(r); sayIn(main.querySelector('[data-det-status]'), ok ? 'Saved.' : 'Could not save: this browser has no space left.');
      return;
    }
    if (f.matches('[data-totals-code]')) {
      e.preventDefault();
      var code = parseResultsCode(fd.get('code'));
      var tst = f.querySelector('[data-totals-status]');
      if (!code) { sayIn(tst, 'That is not a results code. Copy the whole line that starts AI4QI v1.'); return; }
      if (code.id && code.id !== r.auditId) { sayIn(tst, 'This code is for ' + code.id + ', not ' + r.auditId + '. Paste it into that audit instead.'); return; }
      applyTotals(r, code); runPut(r); renderRunKeep(r);
      sayIn(main.querySelector('[data-totals-status]'), 'Results updated from your code.');
      return;
    }
    if (f.matches('[data-totals-hand]')) {
      e.preventDefault();
      function num(k) { var v = fd.get(k); return v === '' || v == null ? null : Math.max(0, Math.min(100000, Math.round(+v) || 0)); }
      var hand = { breakdown: { c1: {}, c2: {} } };
      [['c1', 'c1'], ['c2', 'c2']].forEach(function (k) {
        var met = num(k[0] + 'met'), den = num(k[0] + 'den'), n = num(k[0] + 'n');
        if (den != null && met != null && met <= den) hand[k[1]] = { met: met, den: den, n: n == null ? den : n };
      });
      if (!hand.c1 && !hand.c2) { window.setTimeout(function () {}, 0); return; }
      applyTotals(r, hand); runPut(r); renderRunKeep(r);
      return;
    }
    if (f.matches('[data-run-change]')) {
      e.preventDefault();
      var rep0 = { redacted: 0 };
      r.changeMade = { description: scrub(String(fd.get('description') || '').slice(0, 600), rep0), date: String(fd.get('date') || '') };
      runPut(r); renderRunKeep(r); sayIn(main.querySelector('[data-chg-status]'), rep0.redacted ? 'Saved. Identifiers were removed from the text.' : 'Saved.');
      return;
    }
    if (f.matches('[data-rec-form]')) {
      e.preventDefault();
      var ck = f.getAttribute('data-rec-form'), raw = {};
      fd.forEach(function (v, k) { raw[k] = String(v).replace('T', ' '); });
      var rep = { redacted: 0, coded: 0, badDates: 0 }, codes = { map: new Map(), next: r.codeSeq || 1 };
      var rec = cleanRecord(r.protocol, raw, codes, rep, r.options);
      r.codeSeq = codes.next;
      if (!Object.keys(rec).length) { sayIn(f.querySelector('[data-rec-status]'), 'Fill in at least one field.'); return; }
      r.cycles[ck].rows.push(rec);
      if (r.stage === 'setup' && ck === 'c1') r.stage = 'cycle1';
      runPut(r); S.view.runTab = ck; renderRunKeep(r);
      var box = main.querySelector('[data-rec-form="' + ck + '"]');
      if (box) { box.closest('details').open = true; sayIn(box.querySelector('[data-rec-status]'), 'Record ' + r.cycles[ck].rows.length + ' added.' + (rep.redacted || rep.coded ? ' Identifiers were replaced or removed.' : '')); var first = box.querySelector('input,select'); if (first) first.focus(); }
    }
  });
  document.addEventListener('change', function (e) {
    var r = curRun(); if (!r) return;
    if (e.target.matches('[data-run-remind]')) { r.reminders = e.target.checked; runPut(r); return; }
    var inp = e.target.closest('[data-import]');
    if (!inp || !inp.files || !inp.files[0]) return;
    var ck = inp.getAttribute('data-import'), box = main.querySelector('[data-import-preview="' + ck + '"]'), file = inp.files[0];
    box.innerHTML = '<p class="muted">Reading ' + esc(file.name) + '…</p>';
    readTable(file).then(function (table) {
      var prep = prepareImport(r.protocol, table, r.codeSeq || 1, r.options);
      if (prep.reRows.length) { S.pendingImport = { runId: r.id, ck: 'c1', rows: prep.rows, reRows: prep.reRows, nextCode: prep.nextCode }; }
      else S.pendingImport = { runId: r.id, ck: ck, rows: prep.rows, nextCode: prep.nextCode };
      var total = prep.rows.length + prep.reRows.length;
      box.innerHTML = '<div class="import-preview"><p><strong>' + total + ' records ready to add' + (prep.reRows.length ? ' (' + prep.rows.length + ' cycle 1, ' + prep.reRows.length + ' re-audit)' : '') + '.</strong> Nothing has been stored yet.</p>' +
        '<ul><li>Columns used: ' + (prep.kept.length ? prep.kept.map(function (k) { return esc(k[0]); }).join(', ') : 'none matched the template') + '</li>' +
        '<li>Columns left out: ' + (prep.dropped.length ? esc(prep.dropped.join(', ')) : 'none') + '</li>' +
        '<li>Patient numbers replaced with audit codes: ' + prep.rep.coded + '</li>' +
        '<li>Identifiers removed from text: ' + prep.rep.redacted + '</li>' +
        (prep.rep.badDates ? '<li>Dates that could not be read (left blank): ' + prep.rep.badDates + '</li>' : '') + '</ul>' +
        (total ? '<button class="btn" type="button" data-import-ok>Add ' + total + ' records</button> ' : '') +
        '<button class="link-btn" type="button" data-import-cancel>Cancel</button></div>';
    }).catch(function () { box.innerHTML = '<p class="notice notice-warn">This file could not be read. Save it as .xlsx or .csv and try again.</p>'; });
    inp.value = '';
  });
  document.addEventListener('click', function (e) {
    if (e.target.closest('[data-import-cancel]')) { S.pendingImport = null; var b = e.target.closest('[data-import-preview]'); if (b) b.innerHTML = ''; return; }
    if (!e.target.closest('[data-import-ok]')) return;
    var pi = S.pendingImport, r = pi && S.runs.get(pi.runId);
    if (!r) return;
    r.cycles[pi.ck].rows = r.cycles[pi.ck].rows.concat(pi.rows);
    if (pi.reRows && pi.reRows.length) r.cycles.c2.rows = r.cycles.c2.rows.concat(pi.reRows);
    r.codeSeq = Math.max(r.codeSeq || 1, pi.nextCode || 1);
    if (r.stage === 'setup' && pi.ck === 'c1') r.stage = 'cycle1';
    S.pendingImport = null; S.view.runTab = pi.ck;
    var ok = runPut(r); renderRunKeep(r, '[data-run-tab="' + pi.ck + '"]');
    if (!ok) window.setTimeout(function () { var o = main.querySelector('[data-out-status]'); sayIn(o, 'This browser is out of space. Download a backup.'); }, 0);
  });
  document.addEventListener('submit', function (e) {
    var f = e.target;
    if (!f.matches || !f.matches('[data-paste-any]')) return;
    e.preventDefault();
    var st = f.querySelector('[data-paste-status]'), code = parseResultsCode(new FormData(f).get('code'));
    if (!code) { sayIn(st, 'That is not a results code. Copy the whole line that starts AI4QI v1.'); return; }
    var r = Array.from(S.runs.values()).filter(function (x) { return x.auditId === code.id && !x.closed; })[0];
    if (!r) {
      var p = anyAudit(code.id);
      if (!p) { sayIn(st, 'Audit ' + code.id + ' is not on this device. Open it (Build an audit or Proposed audits), press Choose this audit, then paste the code there.'); return; }
      chooseAudit(p);
      r = Array.from(S.runs.values()).filter(function (x) { return x.auditId === code.id && !x.closed; })[0];
      if (!r) return;
    }
    applyTotals(r, code); runPut(r); location.hash = '#/run/' + r.id;
  });
  /* Restore a backup from My audits. Backups are encrypted with the passcode in use when they were made. */
  function restoreRun(r) {
    if (!r || !/^r-[a-z0-9]{6,24}$/.test(r.id) || !r.protocol || !r.cycles) throw new Error('bad');
    if (!runPut(r)) throw new Error('locked');
    location.hash = '#/run/' + r.id;
  }
  document.addEventListener('change', function (e) {
    var inp = e.target.closest('[data-restore]');
    if (!inp || !inp.files || !inp.files[0]) return;
    var st = main.querySelector('[data-restore-status]');
    inp.files[0].text().then(function (t) {
      var o = JSON.parse(t);
      if (o && o.ai4qi_run && o.run) return restoreRun(o.run);           // older, unencrypted backups
      if (!o || o.ai4qi_backup !== 2 || !o.box || !o.salt) throw new Error('bad');
      S.pendingRestore = o;
      st.innerHTML = '<form class="inline-form" data-restore-pass><label for="rsp">Passcode used when this backup was made</label>' +
        '<input id="rsp" name="p" type="password" autocomplete="off" required><button class="btn" type="submit">Restore</button></form>';
      st.querySelector('input').focus();
    }).catch(function () { sayIn(st, 'That file is not an Ai4Qi backup.'); });
    inp.value = '';
  });
  document.addEventListener('submit', function (e) {
    var f = e.target;
    if (!f.matches || !f.matches('[data-restore-pass]')) return;
    e.preventDefault();
    var o = S.pendingRestore, st = main.querySelector('[data-restore-status]');
    if (!o) return;
    deriveKey(f.elements.p.value, unb64(o.salt), o.iter || PBKDF2_ITER)
      .then(function (k) { return unseal(k, o.box); })
      .then(function (json) { S.pendingRestore = null; restoreRun(JSON.parse(json)); })
      .catch(function () { var m = f.querySelector('.form-status') || document.createElement('p'); m.className = 'form-status is-error'; m.textContent = 'That passcode does not open this backup.'; f.appendChild(m); });
  });

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
      recordFeedback(id, 'down', reasons, scrub(box.querySelector('.fb-comment').value.trim(), { redacted: 0 }));
      more.hidden = true;
      doneEl.innerHTML = '<p class="fb-thanks" role="status">Thank you – this helps us improve the audit library.</p>';
    }
  });
  /* ---------- optional Supabase backend: shared feedback, sign-in, admin view ----------
     Switched on only when config.json has supabase_url and supabase_anon_key. Without them the
     app behaves exactly as before (feedback stays on the device or goes to feedback_url). */
  var SUPABASE_JS = 'vendor/supabase.js';   // supabase-js 2.117.2, served from this site
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
  // Fixed lists; they must match the checks in backend migration 006.
  var WORK_SETTINGS = ['NHS or HSE hospital', 'General practice', 'Community or mental health service', 'Private or independent sector',
    'University or medical school', 'Working outside the UK and Ireland', 'Not currently working in healthcare'];
  var AUDIT_PURPOSES = ['Portfolio or ARCP', 'Job or training application', 'Portfolio pathway (CESR) or specialist registration',
    'Departmental or service improvement', 'Research or a qualification', 'Other'];
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
      if (BE.url && S.lib.length && (cur === 'account' || cur === 'admin')) route();
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
  function initialsOf(email) {
    var local = String(email || '').split('@')[0].replace(/[0-9]+/g, '');
    var parts = local.split(/[._\-+]+/).filter(Boolean);
    var s = parts.length >= 2 ? parts[0][0] + parts[parts.length - 1][0] : local.slice(0, 2);
    return s.toUpperCase();
  }
  function updateAccountLink() {
    var signedIn = BE.user || (!BE.client && hasStoredSession());
    if (accountLink) accountLink.textContent = signedIn ? 'Account and sign out' : 'Sign in';
    var av = document.querySelector('[data-acct-initials]'), who = document.querySelector('[data-acct-who]');
    var lab = document.querySelector('[data-acct-label]');
    if (av && !av.dataset.icon) av.dataset.icon = av.innerHTML;
    if (av && !(BE.user && BE.user.email) && av.classList.contains('has-initials')) {
      av.innerHTML = av.dataset.icon; av.classList.remove('has-initials'); if (who) who.hidden = true;
    }
    if (av && BE.user && BE.user.email) {
      av.textContent = initialsOf(BE.user.email); av.classList.add('has-initials');
      if (who) { who.textContent = BE.user.email; who.hidden = false; }
      if (lab) lab.textContent = 'Your menu, signed in as ' + BE.user.email;
    }
  }
  (function acctMenu() {
    var btn = document.querySelector('[data-acct-btn]'), menu = document.querySelector('[data-acct-menu]');
    if (!btn || !menu) return;
    function set(open) { menu.hidden = !open; btn.setAttribute('aria-expanded', String(open)); }
    btn.addEventListener('click', function (e) { e.stopPropagation(); set(menu.hidden); if (!menu.hidden) { var f = menu.querySelector('a:not([hidden])'); if (f) f.focus(); } });
    document.addEventListener('click', function (e) { if (!menu.hidden && !e.target.closest('[data-acct]')) set(false); });
    document.addEventListener('keydown', function (e) { if (e.key === 'Escape' && !menu.hidden) { set(false); btn.focus(); } });
    menu.addEventListener('click', function (e) { if (e.target.closest('a')) set(false); });
  })();

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
  document.addEventListener('click', function (e) {
    if (e.target.closest('[data-delete-account]')) {
      main.querySelector('[data-delete-account-box]').innerHTML = ' <strong>Delete your account, profile, feedback links and reminders for good?</strong> ' +
        '<button class="btn btn-secondary" type="button" data-delete-account-yes>Delete permanently</button>' +
        '<span class="form-status" role="status" data-delete-account-status></span>';
      return;
    }
    if (!e.target.closest('[data-delete-account-yes]')) return;
    var st = main.querySelector('[data-delete-account-status]');
    sayIn(st, ' Deleting…');
    sbClient().then(function (c) {
      return c.rpc('delete_my_account').then(function (res) { if (res.error) throw res.error; return c.auth.signOut(); });
    }).then(function () {
      BE.user = null; BE.tracker = null;
      page('<article class="doc narrow"><h1>Your account has been deleted</h1><p class="prose">Your sign-in, profile, audit progress and reminders have been removed from our server. Audits stored on this device are not affected; delete them from My audits if you wish.</p></article>', 'Account deleted', 'account');
    }).catch(function () { sayIn(st, ' Your account could not be deleted. Please try again, or email us.'); });
  });
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
    return c.from('profiles').select('specialty, grade, region, work_setting, audit_purpose, consent_news, consent_sponsors').eq('user_id', BE.user.id).maybeSingle()
      .then(function (res) {
        BE.profile = res.error ? null : (res.data || null);
        var pend = null; try { pend = JSON.parse(localStorage.getItem('ai4qi_consent_pending') || 'null'); } catch (e) {}
        if (pend && !res.error) {
          try { localStorage.removeItem('ai4qi_consent_pending'); } catch (e) {}
          if (pend.news || pend.sponsors) {
            var row = { user_id: BE.user.id, consent_news: !!pend.news, consent_sponsors: !!pend.sponsors, consent_updated_at: new Date().toISOString() };
            BE.profile = Object.assign({}, BE.profile || {}, row);
            return c.from('profiles').upsert(row, { onConflict: 'user_id' }).then(function () { return BE.profile; });
          }
        }
        return BE.profile;
      });
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
      '<p class="page-intro">Signing in is free and takes a minute. You need it to build new audits and to get reminders; everything else works without it. There is no password: we email you a secure sign-in link.</p>' +
      err +
      '<form class="stack-form" data-signin novalidate>' +
      '<label for="acc-email">Email address</label>' +
      '<input id="acc-email" name="email" type="email" inputmode="email" autocomplete="email" spellcheck="false" required maxlength="254">' +
      '<fieldset class="consent"><legend>Optional</legend>' +
      '<label class="check"><input type="checkbox" name="consent_news"><span>Email me Ai4Qi news and new features (about once a month).</span></label>' +
      '<label class="check"><input type="checkbox" name="consent_sponsors"><span>Email me occasional offers from Ai4Qi\'s sponsors (courses, events, jobs). We never share your email address with them.</span></label>' +
      '</fieldset>' +
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
      '<label for="pf-work">Where you work</label>' +
      '<select id="pf-work" name="work_setting"><option value="">Prefer not to say</option>' + optionList(WORK_SETTINGS, pf.work_setting) + '</select>' +
      '<label for="pf-purpose">Main reason for your audits</label>' +
      '<select id="pf-purpose" name="audit_purpose"><option value="">Prefer not to say</option>' + optionList(AUDIT_PURPOSES, pf.audit_purpose) + '</select>' +
      '<fieldset class="consent"><legend>Optional</legend>' +
      '<label class="check"><input type="checkbox" name="consent_news"' + (pf.consent_news ? ' checked' : '') + '><span>Email me Ai4Qi news and new features (about once a month).</span></label>' +
      '<label class="check"><input type="checkbox" name="consent_sponsors"' + (pf.consent_sponsors ? ' checked' : '') + '><span>Email me occasional offers from Ai4Qi\'s sponsors (courses, events, jobs). We never share your email address with them.</span></label>' +
      '</fieldset>' +
      '<button class="btn" type="submit">Save profile</button>' +
      '<p class="form-status" data-form-status role="status" aria-live="polite"></p>' +
      '</form></section>' +
      (BE.admin ? '<section class="account-section" aria-labelledby="adm-h"><h2 id="adm-h">Administration</h2><ul><li><a href="#/admin/stats">Usage statistics</a></li><li><a href="#/admin/feedback">Feedback on proposed audits</a></li></ul></section>' : '') +
      '<div class="account-actions"><button class="btn btn-secondary" type="button" data-signout>Sign out</button> ' +
      '<button class="link-btn" type="button" data-delete-account>Delete my account</button><span data-delete-account-box></span></div>' +
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
      try { localStorage.setItem('ai4qi_consent_pending', JSON.stringify({ news: form.elements.consent_news.checked, sponsors: form.elements.consent_sponsors.checked })); } catch (e2) {}
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
      var row = { user_id: BE.user && BE.user.id, specialty: spec, grade: grade, region: region,
        work_setting: pick('work_setting', WORK_SETTINGS), audit_purpose: pick('audit_purpose', AUDIT_PURPOSES),
        consent_news: form.elements.consent_news.checked, consent_sponsors: form.elements.consent_sponsors.checked };
      var prev = BE.profile || {};
      if (row.consent_news !== !!prev.consent_news || row.consent_sponsors !== !!prev.consent_sponsors) row.consent_updated_at = new Date().toISOString();
      btn.disabled = true; say('Saving…');
      sbClient().then(function (c) {
        if (!BE.user) throw new Error('signed out');
        return c.from('profiles').upsert(row, { onConflict: 'user_id' });
      }).then(function (res) {
        if (res.error) throw res.error;
        BE.profile = row;
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
  function anyAudit(id) { return S.pById.get(id) || builtGet(id); }
  function auditHref(id) {
    if (/^B-/.test(id)) { var b = builtGet(id); return '#/build?q=' + encodeURIComponent(b ? b.topic : id.slice(2).replace(/-/g, ' ')); }
    return '#/proposed/' + encodeURIComponent(id);
  }
  function auditLink(id) {
    var p = anyAudit(id);
    return '<a href="' + auditHref(id) + '" class="id-tag">' + esc(id) + '</a>' +
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
      var p = anyAudit(r.audit_id);
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
            var p = anyAudit(r.audit_id);
            return '<li class="my-item" data-my-item="' + attr(r.audit_id) + '">' +
              '<p class="my-title"><a href="' + auditHref(r.audit_id) + '">' + esc(p ? p.question : r.audit_id) + '</a> <span class="id-tag">' + esc(r.audit_id) + '</span></p>' +
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
      breakdownTable(st.breakdown, 'region', 'Region') + breakdownTable(st.breakdown, 'verified', 'NHS or HSE email') +
      breakdownTable(st.breakdown, 'work_setting', 'Where they work') + breakdownTable(st.breakdown, 'audit_purpose', 'Reason for audits') +
      breakdownTable(st.breakdown, 'consent', 'Email consent') + '</div>' +
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
  var KNOWN_ROUTES = ['terms', 'privacy-notice', 'cookies', 'accessibility', 'security', 'build', 'suggest', 'run', 'privacy', 'search', 'proposed', 'audit', 'topic', 'topics', 'standards', 'account', 'my-audits', 'admin'];
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

  function templateCsv(p) {
    function cell(v) { v = String(v == null ? '' : v); return /[",\n]/.test(v) ? '"' + v.replace(/"/g, '""') + '"' : v; }
    var t = p.template || [];
    return [t.map(function (f) { return cell(f.field); }).join(','),
      t.map(function (f) { return cell(f.type + (f.options && f.options.length ? ': ' + f.options.join(' / ') : '') + (f.note ? ' (' + f.note + ')' : '')); }).join(',')].join('\n') + '\n';
  }
  document.addEventListener('click', function (e) {
    var b = e.target.closest('[data-copy-csv]');
    if (!b) return;
    var cid = b.getAttribute('data-copy-csv'), p = S.pById.get(cid) || builtGet(cid), st = b.parentNode.querySelector('[data-copy-status]');
    if (!p) return;
    var csv = templateCsv(p);
    function fallback() {
      var ta = document.createElement('textarea'); ta.value = csv; ta.rows = 4; ta.className = 'copy-fallback'; ta.readOnly = true;
      b.parentNode.appendChild(ta); ta.focus(); ta.select();
      if (st) st.textContent = 'Select all and copy the text above into a spreadsheet.';
    }
    try {
      navigator.clipboard.writeText(csv).then(function () { if (st) st.textContent = 'Copied – paste into Excel or Google Sheets.'; }, fallback);
    } catch (err) { fallback(); }
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

  if (!window.AI4QI_EMBED && 'serviceWorker' in navigator && window.isSecureContext) {
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
