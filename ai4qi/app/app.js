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
  function ukBadge(a) { return ''; }   // no country badge on audits
  var PROPOSED_BADGE = badge('Ready-made audit', 'primary');
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

  /* ---------- standards in plain words ----------
     "NICE NG253 rec 1.8.3 (November 2025, updated September 2026)" reads as
     "NICE guideline on suspected sepsis in people aged 16 or over (NG253)"; other sources lose their
     rec numbers, dates and "accessed" notes. The exact reference stays in the link and the NICE notice.
     Titles come from data/nice_titles.json (standards/fetch_nice_titles.py). Used everywhere a source
     is shown, including the downloads (export.js reads window.AI4QI_humanSource). */
  var NICE_TITLES = {};
  var NICE_KIND = { ng: 'guideline', cg: 'guideline', ph: 'guideline', nm: 'guideline', sc: 'guideline', qs: 'quality standard',
    ta: 'technology appraisal', dg: 'diagnostics guidance', ipg: 'interventional procedures guidance', mtg: 'medical technologies guidance' };
  function topicOf(t) {
    t = String(t || '').split(':')[0].trim();
    return /^[A-Z][a-z]/.test(t) ? t.charAt(0).toLowerCase() + t.slice(1) : t;
  }
  function humanSource(src) {
    src = String(src || '').replace(/\s+/g, ' ').trim();
    if (!src) return '';
    var codes = [], m, re = /\b(NG|CG|QS|PH|NM|TA|DG|IPG|MTG|SC)\s?(\d+)/gi;
    if (/^\s*NICE\b/.test(src)) {
      while ((m = re.exec(src))) { var c = (m[1] + m[2]).toLowerCase(); if (codes.indexOf(c) < 0) codes.push(c); }
      if (codes.length) {
        return codes.map(function (c, i) {
          var k = c.replace(/\d+/, ''), code = c.toUpperCase(), t = NICE_TITLES[c], kind = NICE_KIND[k] || 'guidance';
          if (!t) return (i ? '' : 'NICE ') + kind + ' ' + code;
          if (/^guidance on /i.test(t)) return (i ? '' : 'NICE ') + topicOf(t) + ' (' + code + ')';
          return (i ? '' : 'NICE ') + kind + ' on ' + topicOf(t) + ' (' + code + ')';
        }).join(' and ');
      }
    }
    var out = src
      .replace(/\s*\((?:[^()]*\b(?:19|20)\d\d\b[^()]*|accessed[^()]*|updated[^()]*)\)/gi, '')   // (2015, updated 2023), (accessed 2026)
      .replace(/,?\s*summarised on .*$/i, '')
      .replace(/\bchapter\s+\d+[a-z]?,\s*/i, 'chapter on ')
      .replace(/,?\s*\b(?:recs?|recommendations?|statements?|standards?|sections?|elements?|interventions?|para(?:graph)?s?|practice points?|criteri(?:on|a))\s+[\d.]+(?:\s*(?:and|,|&|–|-)\s*[\d.]+)*/gi, '')
      .replace(/,\s*(?:19|20)\d\d\b(?:\s*guidance)?/g, '')
      .replace(/\s*,\s*,/g, ',').replace(/[\s,;:–-]+$/, '').trim();
    return out || src;
  }
  window.AI4QI_humanSource = humanSource;

  /* ---------- data loading ---------- */
  function load() {
    return Promise.all([
      getJSON('data/library.json'), getJSON('data/proposed.json'),
      getJSON('data/cards.json'), getJSON('data/standards.json'),
      getJSON('data/nice_titles.json').catch(function () { return {}; })
    ]).then(function (r) {
      NICE_TITLES = r[4] || {};
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
    document.querySelectorAll('[data-demo-toggle]').forEach(function (dt) {
      dt.textContent = DEMO ? 'Turn off demo mode' : (dt.closest('.footer-links') ? 'Try demo mode' : 'Demo mode');
      dt.setAttribute('href', DEMO ? '#/demo/off' : '#/demo');
    });
    document.title = title ? title + ' · Ai4Qi' : 'Ai4Qi Clinical Audit Library';
    setNav(nav || '');
    if (!keepScroll) window.scrollTo(0, 0);
  }
  function focusMain() {
    var h = main.querySelector('h1');
    if (h) { h.setAttribute('tabindex', '-1'); h.focus({ preventScroll: true }); }
  }

  var AFTER_SIGNIN = 'ai4qi_after_signin';
  function route() {
    var r = parseHash(), p = r.parts, name = p[0] || '';
    if (name !== 'account' && name !== 'demo') S.lastRoute = location.hash || '#/';   // where to go back to after signing in
    var sameView = false;
    try {
      // Demo mode on or off keeps you on the page you were on (sign-in and your own audits are untouched).
      if (name === 'demo') { demoSet(p[1] !== 'off'); location.replace(S.lastRoute && S.lastRoute !== '#/' ? S.lastRoute : (p[1] === 'off' ? '#/' : '#/suggest')); return; }
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
      else if (name === 'signed-out') renderSignedOut();
      else if (name === 'ideas') renderIdeas();
      else if (name === 'access') renderAccess();
      else if (name === 'admin' && p[1] === 'access' && BE.url) renderAdminAccess();
      else if (name === 'my-audits') renderRuns();
      else if (name === 'run' && p[1]) renderRun(p[1]);
      else if (name === 'privacy') renderPrivacy();
      else if (name === 'how-it-works') renderHow();
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
      '<p class="kicker">Clinical audit, start to closed loop</p>' +
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
        props = st.sp === 'Trauma and orthopaedics' ? S.proposed.filter(function (p) { return p.group === 'Trauma and orthopaedics'; })
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
      '</aside>';

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
      '<h3><a href="#/proposed/' + attr(p.id) + '">' + esc(auditName(p)) + '</a></h3><p class="audit-q small">' + esc(p.question) + '</p>' +
      '<p class="meta"><span class="id-tag">' + esc(p.id) + '</span> · Standard: ' + esc(trunc(humanSource(st.source), 110)) + '</p>' +
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
  /* Pitfalls and pearls: a short heading, then the full explanation. Headings come with the
     ready-made audits; for built audits the risk (before the arrow) becomes the heading. */
  function ppItem(t, head) {
    var i = t.indexOf('→'), risk = i === -1 ? '' : t.slice(0, i).trim(), fix = i === -1 ? t : t.slice(i + 1).trim();
    var h = head || (risk && risk.split(/\s+/).length <= 8 ? risk : '');
    if (!h) return '<li><p class="pp-txt">' + linkify(t) + '</p></li>';
    var body = h === risk ? '<span class="pp-fix">' + linkify(cap(fix)) + '</span>' :
      risk ? linkify(risk) + ' <span class="pp-arrow" aria-label="prevent by">→</span> <span class="pp-fix">' + linkify(fix) + '</span>' : linkify(t);
    return '<li><b class="pp-t">' + esc(h) + '</b><p class="pp-txt">' + body + '</p></li>';
  }

  function sec(n, title, body) {
    return '<section><h2><span class="num" aria-hidden="true">' + n + '</span>' + esc(title) + '</h2>' + body + '</section>';
  }
  function sub(title, body) { return '<div class="sub"><h3>' + esc(title) + '</h3>' + body + '</div>'; }

  function copyBtn(id) {
    return '<button class="btn" type="button" data-copy-csv="' + attr(id) + '">' +
      '<svg width="18" height="18" viewBox="0 0 24 24" aria-hidden="true"><rect x="8" y="8" width="12" height="12" rx="2" fill="none" stroke="currentColor" stroke-width="2"/><path d="M16 8V5a1 1 0 0 0-1-1H5a1 1 0 0 0-1 1v10a1 1 0 0 0 1 1h3" fill="none" stroke="currentColor" stroke-width="2"/></svg>Copy template (CSV)</button><span class="copy-status" role="status" data-copy-status></span>';
  }
  /* The protocol layout shared by proposed audits and audits built on request:
     why and how; status; a rule; closing the loop (change, re-audit, embed); a rule; evidence;
     pitfalls and pearls together in one box. */
  function whyPoints(t) {
    var parts = String(t || '').replace(/\s+/g, ' ').trim().replace(/([.!?])\s+(?=[A-Z\[])/g, '$1\u0001').split('\u0001').filter(Boolean);
    if (parts.length < 2) return '<p class="prose why-lead">' + linkify(t) + '</p>';
    return '<ul class="why-list">' + parts.map(function (x) { return '<li>' + linkify(x) + '</li>'; }).join('') + '</ul>';
  }
  /* Data collection effort as a scale: "~15 min per 10 patients" -> Medium, marker on a green-to-red bar. */
  var ICON = {
    clock: '<svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/></svg>',
    day: '<svg viewBox="0 0 24 24" aria-hidden="true"><rect x="3.5" y="5" width="17" height="15" rx="2"/><path d="M3.5 10h17M8 3v4M16 3v4"/><rect x="7" y="13" width="3" height="3" rx=".5"/></svg>',
    chart: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M4 20h16"/><rect x="6" y="11" width="3" height="7"/><rect x="11" y="7" width="3" height="11"/><rect x="16" y="13" width="3" height="5"/></svg>',
    tool: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M14.7 6.3a4 4 0 0 0-5.4 5.2L4 16.8V20h3.2l5.3-5.3a4 4 0 0 0 5.2-5.4l-2.5 2.5-2.6-.6-.6-2.6z"/></svg>'
  };
  function minsText(m) {
    if (m < 60) return Math.max(5, Math.round(m / 5) * 5) + ' min';
    var h = Math.round(m / 30) / 2; return h + (h === 1 ? ' hour' : ' hours');
  }
  // Time with and without the site, estimated from the protocol. Data collection is the same either way
  // (records still have to be read); the site saves the design, the sums and charts, the slides and the paperwork.
  function timeBudget(p, per10) {
    if (per10 == null) return '';
    var n = sampleGuess(p), segs = parseTimeline(p.timeline) || [];
    var weeks = segs.length ? Math.max.apply(null, segs.map(function (x) { return x.b; })) : 12;
    var collect = per10 * n / 10 * 2;                                  // cycle 1 and re-audit
    var rows = [
      ['Design the audit', 'find the standard, write the protocol, build the data sheet', 240, 5],
      ['Proposal for your supervisor', 'email and proposal document', 60, 5],
      ['Collect the data', n + ' records, twice', collect, collect],
      ['Work out the results', 'totals, percentages and charts, twice', 120, 5],
      ['Make the presentation', 'slides for each cycle', 180, 5]
    ];
    var tot0 = 0, tot1 = 0, mx = 0;
    rows.forEach(function (r) { tot0 += r[2]; tot1 += r[3]; mx = Math.max(mx, r[2]); });
    function bar(v, cls) { return '<span class="tc-bar ' + cls + '" style="width:' + Math.max(2, v / mx * 70) + '%"></span>'; }
    return '<div class="time-compare">' +
      '<div class="tc-head"><span></span><span class="tc-k tc-k0">By hand</span><span class="tc-k tc-k1">With Ai4Qi</span></div>' +
      rows.map(function (r) {
        return '<div class="tc-row"><div class="tc-lab"><b>' + r[0] + '</b><span>' + r[1] + '</span></div>' +
          '<div class="tc-cell">' + bar(r[2], 'is-hand') + '<em>' + minsText(r[2]) + '</em></div>' +
          '<div class="tc-cell">' + bar(r[3], 'is-app') + '<em>' + minsText(r[3]) + '</em></div></div>';
      }).join('') +
      '<div class="tc-row tc-total"><div class="tc-lab"><b>Total, over ' + weeks + ' weeks</b></div><div class="tc-cell"><em>' + minsText(tot0) + '</em></div><div class="tc-cell"><em>' + minsText(tot1) + '</em></div></div>' +
      '<div class="tc-saved">' + ICON.clock + '<span>Time saved: <b>about ' + minsText(tot0 - tot1) + '</b></span></div>' +
      '<p class="tb-note">Estimates, to help you plan. Collecting the data takes the same time either way.</p></div>';
  }
  function effortScale(p) {
    var t = String((p && p.effort) || ''); if (!t) return '';
    var m = t.match(/(\d+)\s*(min|minutes|h|hours?)\b[^\d]*?(\d+)?\s*(patients|records|cases|notes)?/i);
    var per10 = null;
    if (m) { var v = +m[1] * (/^h/i.test(m[2]) ? 60 : 1), n = m[3] ? +m[3] : 10; per10 = v * 10 / (n || 10); }
    // Five fixed positions, by minutes per 10 patients.
    var lvl = per10 == null ? null : per10 <= 5 ? 0 : per10 <= 10 ? 1 : per10 <= 20 ? 2 : per10 <= 30 ? 3 : 4;
    var names = ['Very easy', 'Easy', 'Medium', 'Hard', 'Very hard'], pos = lvl == null ? 50 : 10 + lvl * 20;
    return '<div class="effort"><span class="effort-h">Data collection effort' + (lvl == null ? '' : ': <b>' + names[lvl] + '</b>') + '</span>' +
      '<div class="effort-bar" role="img" aria-label="' + attr((lvl == null ? '' : names[lvl] + ': ') + t) + '"><i style="left:' + pos + '%"></i></div>' +
      '<div class="effort-scale" aria-hidden="true">' + names.map(function (n, i) { return '<span' + (i === lvl ? ' class="on"' : '') + '>' + n + '</span>'; }).join('') + '</div>' +
      '<p class="effort-t">' + esc(t.replace(/^~/, 'About ')) + '</p>' + timeBudget(p, per10) + '</div>';
  }
  // Closing the loop as a flowchart: each step with its weeks (from the timeline) and what to do.
  var FLOW_ICON = {
    collect: '<svg viewBox="0 0 24 24"><rect x="4" y="3" width="16" height="18" rx="2"/><path d="M8 8h8M8 12h8M8 16h5"/></svg>',
    present: '<svg viewBox="0 0 24 24"><path d="M4 20h16"/><rect x="6" y="11" width="3" height="7"/><rect x="11" y="7" width="3" height="11"/><rect x="16" y="13" width="3" height="5"/></svg>',
    change: '<svg viewBox="0 0 24 24"><path d="M14.7 6.3a4 4 0 0 0-5.4 5.2L4 16.8V20h3.2l5.3-5.3a4 4 0 0 0 5.2-5.4l-2.5 2.5-2.6-.6-.6-2.6z"/></svg>',
    embed: '<svg viewBox="0 0 24 24"><path d="M12 21c-4-2-7-5-7-10V5l7-2 7 2v6c0 5-3 8-7 10z"/><path d="M9 12l2 2 4-4"/></svg>',
    reaudit: '<svg viewBox="0 0 24 24"><path d="M20 12a8 8 0 1 1-2.3-5.7"/><path d="M20 4v5h-5"/></svg>',
    share: '<svg viewBox="0 0 24 24"><circle cx="6" cy="12" r="2.5"/><circle cx="18" cy="6" r="2.5"/><circle cx="18" cy="18" r="2.5"/><path d="M8.2 11l7.6-4M8.2 13l7.6 4"/></svg>'
  };
  function loopFlow(p) {
    var segs = parseTimeline(p.timeline) || [];
    function wk(re) { var sg = segs.filter(function (x) { return re.test(x.label); })[0]; return sg ? 'Week' + (sg.a === sg.b ? ' ' + sg.a : 's ' + sg.a + '–' + sg.b) : ''; }
    var last = segs.length ? Math.max.apply(null, segs.map(function (x) { return x.b; })) : 0;
    var steps = [
      ['collect', 'Collect cycle 1', wk(/collect/i), 'Fill in the cycle 1 sheet for ' + lowerFirst(noDot(p.sample) || 'your sample') + '.'],
      ['present', 'Analyse and present', wk(/analy|present/i), 'Upload the sheet in My audits: the results, charts and slides are made for you. Present them and agree the change.'],
      ['change', 'Make the change', wk(/change/i), noDot(p.change) + '.'],
      ['embed', 'Let it bed in', wk(/embed/i), 'Keep the change in daily use: remind the team, check it is being used, and fix what gets in the way.'],
      ['reaudit', 'Re-audit', wk(/re-?audit/i), noDot(p.reaudit) + '. Use the re-audit sheet from My audits: its code ties it to this audit.' + (p.target ? ' Target: ' + p.target + '.' : '')],
      ['share', 'Share and keep it going', last ? 'Week ' + (last + 1) + ' on' : '', noDot(p.close_loop) + '.']
    ];
    var strip = '<ol class="flow-strip" aria-hidden="true">' + steps.map(function (st, i) {
      return '<li class="fs-' + st[0] + '"><span class="fs-i">' + FLOW_ICON[st[0]] + '</span><span class="fs-t">' + st[1] + '</span><span class="fs-w">' + esc(st[2]) + '</span></li>';
    }).join('') + '</ol>';
    var detail = '<ol class="flow-steps">' + steps.map(function (st, i) {
      return '<li class="fl-' + st[0] + '"><span class="fl-n">' + (i + 1) + '</span><div class="fl-card"><div class="fl-head"><span class="fl-i">' + FLOW_ICON[st[0]] + '</span><h3>' + st[1] + '</h3>' +
        (st[2] ? '<span class="fl-w">' + esc(st[2]) + '</span>' : '') + '</div><p>' + linkify(st[3]) + '</p></div></li>';
    }).join('') + '</ol>';
    var change = '<div class="loop-change"><span class="lc-i" aria-hidden="true">' + FLOW_ICON.change + '</span><div><b>The change</b><p>' + linkify(noDot(p.change)) + '.</p>' +
      (p.change_alt ? '<p class="lc-alt"><b class="lc-k">Or change 2</b>' + linkify(noDot(p.change_alt)) + '.</p>' : '') +
      '<small>These are suggestions you can change. Pick one, or your own, that will work on your ward, agree it with your team, and record it in My audits.</small></div></div>';
    return { strip: strip, detail: detail, change: change };
  }
  window.addEventListener('beforeprint', function () { document.querySelectorAll('details.loop-part').forEach(function (d) { d.open = true; }); });
  function protocolBody(p, dl, extra) {
    var st = p.standard || {}, url = safeUrl(st.url), n = 0;
    extra = extra || {};
    // Method: the question across the top, then a 2 x 2 grid: standard (with the pass criterion), selection, data collection, sample size.
    function cell(t, body) { return '<div class="m-cell"><h3 class="m-lab">' + t + '</h3>' + body + '</div>'; }
    var how =
      '<div class="m-q m-card"><h3 class="m-lab">Audit question</h3><p>' + esc(p.question) + '</p></div>' +
      '<div class="m-grid">' +
      cell('Standard', (st.wording ? '<blockquote class="standard"><p>“' + esc(st.wording) + '”</p></blockquote>' : '<p class="prose">Read the exact recommendation at the source below.</p>') +
        '<p class="standard-source" title="' + attr(st.source) + '">' + esc(humanSource(st.source)) + (url ? ' · <a class="print-url" href="' + attr(url) + '" target="_blank" rel="noopener">Read the standard</a>' : '') + '</p>' + niceAttribution(st) +
        '<div class="m-pass"><h4 class="m-sublab">Pass criterion</h4>' + passHtml(p) + '</div>') +
      cell('Selection criteria', selectionHtml(p.population)) +
      cell('Data collection', bullets(splitTop(p.data_source, /;/), 'bul')) +
      cell('Sample size', '<p class="m-n">' + linkify(p.sample) + '</p>' +
        '<small class="prec-note">' + precisionText(sampleGuess(p)) + powerText(p) + ' Need to do fewer? After choosing the audit, set your own number in My audits (at least 20).</small>') +
      '</div>' +
      sub('Timeline', timelineHtml(p.timeline));
    var pits = (p.pitfalls || []).length, pearls = (p.pearls || []).length;
    return (extra.before ? sec(++n, extra.before[0], extra.before[1]) : '') +
      sec(++n, 'Why this audit matters', whyPoints(p.why)) +
      sec(++n, 'Method', how) +
      sec(++n, 'Data sheet', templateTable(p) + '<div class="no-print sheet-dl">' + dl + '</div>' +
        '<p class="muted sheet-note">Choosing the audit adds it to <strong>My audits</strong>: you upload your sheet there and get the results and slides. The re-audit gets its own sheet, with a code that ties it to this audit. Only the sheet\'s own columns are ever read, so names, NHS numbers and dates of birth are never imported.</p>') +

      (extra.status ? '<div class="status-row">' + extra.status + '</div>' : '') +
      (extra.feedback || '') +
      '<div class="part-rule" role="separator"><span>Closing the loop</span></div>' +
      (function () { var lf = loopFlow(p); return '<details class="loop-part"><summary><span class="loop-cta"><b>Make it a closed loop</b><span>Six steps from the first data to a change that lasts. Tap to see what to do at each one.</span></span>' + lf.strip + lf.change + '</summary>' + lf.detail + '</details>'; })() +
      '<div class="part-rule" role="separator"><span>Evidence and advice</span></div>' +
      (pits || pearls ? '<section class="pp-box"><div class="pp-grid">' +
        (pits ? '<div><h2 class="pp-h">Pitfalls</h2><ul class="pit-list">' + p.pitfalls.map(function (e, i) { return ppItem(e, (p.pitfall_heads || [])[i]); }).join('') + '</ul></div>' : '') +
        (pearls ? '<div><h2 class="pp-h">Pearls</h2><ul class="pearl-list">' + p.pearls.map(function (e, i) { return ppItem(e, (p.pearl_heads || [])[i]); }).join('') + '</ul></div>' : '') +
        '</div></section>' : '') +
      (p.evidence && p.evidence.length ? sec(++n, 'Evidence', '<ul class="ev-list">' + p.evidence.map(function (e) { return '<li>' + linkify(e) + '</li>'; }).join('') + '</ul>') : '') +
      (extra.after ? extra.after.map(function (x) {
        var cnt = (x[1].match(/<li/g) || []).length;
        return '<details class="more-box"><summary><span>' + esc(x[0]) + (cnt ? ' <span class="count">' + cnt + '</span>' : '') + '</span><span class="more-hint">Show</span></summary>' + x[1] + '</details>';
      }).join('') : '');
  }

  /* ---------- send a proposal to a supervisor: a plain, friendly email + a Word proposal ---------- */
  var ME_KEY = 'ai4qi_me_v1';
  function meGet() { try { return JSON.parse(localStorage.getItem(ME_KEY) || '{}') || {}; } catch (e) { return {}; } }
  function meSet(o) { try { localStorage.setItem(ME_KEY, JSON.stringify(o)); } catch (e) {} }
  // [key, label, type, needed before sending, remembered on this device]
  var SUP_FIELDS = [['lead', 'Your name', 'text', 1, 1], ['role', 'Your role or grade', 'text', 0, 1], ['email', 'Your email', 'email', 0, 1],
    ['supervisor', 'Supervisor\'s name', 'text', 1, 1], ['supEmail', 'Supervisor\'s email', 'email', 1, 1], ['site', 'Hospital, Trust or practice', 'text', 1, 1],
    ['department', 'Department or ward', 'text', 0, 1], ['startDate', 'Proposed start date', 'date', 1, 0], ['sampleSize', 'Records per cycle', 'number', 0, 0]];
  // NHS trusts (England), health boards (Wales, Scotland), HSC trusts (NI) and HSE regions, for a typeahead list.
  // Anyone can still type an organisation that is not on it.
  var ORGS = null;
  function orgList() {
    if (!ORGS) {
      getJSON('data/organisations.json').then(function (o) {
        ORGS = o || {}; var dl = document.getElementById('org-list'); if (dl) dl.innerHTML = orgOptions();
      }, function () { ORGS = {}; });
      return '<datalist id="org-list"></datalist>';
    }
    return '<datalist id="org-list">' + orgOptions() + '</datalist>';
  }
  function orgOptions() {
    var out = []; Object.keys(ORGS || {}).forEach(function (k) { (ORGS[k] || []).forEach(function (n) { out.push('<option value="' + attr(n) + '">' + esc(k) + '</option>'); }); });
    return out.join('');
  }
  function supRun(box) { var id = box.getAttribute('data-sup-run'); return id ? S.runs.get(id) : null; }
  function supWho(box) {
    var o = {}, f = box.querySelector('[data-sup-form]'), r = supRun(box);
    SUP_FIELDS.forEach(function (x) { o[x[0]] = (f.elements[x[0]].value || '').trim(); });
    if (r) { o.title = r.details.title; o.team = r.details.team; }
    var pr = supProtocol(box); o.links = pr ? supLinks(pr) : [];
    return o;
  }
  function supProtocol(box) { var r = supRun(box); return r ? r.protocol : anyAudit(box.getAttribute('data-sup')); }
  function firstSent(t) { var m = String(t || '').match(/^[\s\S]*?[.!?](\s|$)/); return (m ? m[0] : String(t || '')).trim(); }
  function lowerFirst(t) { t = String(t || ''); return /^[A-Z][a-z]/.test(t) ? t.charAt(0).toLowerCase() + t.slice(1) : t; }
  function noDot(t) { return String(t || '').trim().replace(/[.\s]+$/, ''); }
  function standardQuote(st) {
    var w = String(st.wording || '').replace(/\s+/g, ' ').replace(/\s*\.\.\.\s*/g, ' ').trim();
    w = w.replace(/^\d+(\.\d+)+\s+/, '').replace(/\s*\[\d{4}[^\]]*\]/g, '');
    w = firstSent(w);
    var ww = w.split(' ');
    if (ww.length > 38) w = ww.slice(0, 34).join(' ').replace(/[.,;:]+$/, '') + '…';
    return noDot(w);
  }
  function targetPhrase(t) {
    t = String(t || '').trim();
    var m = t.match(/^([≥>≤<]?)\s*(\d+(?:\.\d+)?)\s*%\s*(.*)$/);
    if (!m) return t ? 'Our target is ' + lowerFirst(noDot(t)) + '.' : '';
    var more = m[1] === '≤' || m[1] === '<' ? ' or less' : m[1] ? ' or more' : '';
    return 'We are aiming for ' + m[2] + '%' + more + (m[3] ? ' (' + noDot(m[3]).replace(/^[,;:\s-]+/, '') + ')' : '') + '.';
  }
  function supTimeline(p, start) {
    var segs = parseTimeline(p.timeline);
    if (!segs) return p.timeline ? [p.timeline] : [];
    return segs.map(function (s) {
      var wk = s.a === s.b ? 'Week ' + s.a : 'Weeks ' + s.a + '–' + s.b;
      var when = start ? ' (' + (s.a === s.b ? 'w/c ' + dateGBs(addDays(start, (s.a - 1) * 7)).replace(/ \d{4}$/, '') :
        dateGBs(addDays(start, (s.a - 1) * 7)).replace(/ \d{4}$/, '') + ' – ' + dateGBs(addDays(start, s.b * 7 - 1)).replace(/ \d{4}$/, '')) + ')' : '';
      return wk + when + ': ' + lowerFirst(s.label).replace(/\//g, ' and ');
    });
  }
  /* The email: {subject, html, text}. Written the way a trainee would write it. */
  function supEmail(p, d) {
    var st = p.standard || {}, place = d.department && d.site ? 'the ' + d.department.replace(/^the\s+/i, '') + ' at ' + d.site : (d.department || d.site || '');
    var name = d.supervisor ? d.supervisor.trim() : '[supervisor\'s name]';
    var quote = standardQuote(st), sName = humanSource(st.source) || 'our local standard';
    if (/^NICE /.test(sName)) sName = 'the ' + sName;
    var stdLine = 'We\'ll measure this against ' + sName + (quote ? ', which says: “' + quote + '”.' : '.');
    var sample = d.sampleSize ? 'We\'ll look at ' + d.sampleSize + ' patients in each cycle, and do the same again at re-audit.' :
      (p.sample ? 'Sample: ' + noDot(firstSent(p.sample)) + '.' : '');
    var change = p.change ? 'If we fall short, the plan is to ' + lowerFirst(noDot(firstSent(p.change))).replace(/^(build|put|add|introduce|create|use|make|run|start|set|change|move|give|train|display|embed)\b/i, function (m) { return m.toLowerCase(); }) + '.' : '';
    var blocks = [
      { p: 'Dear ' + name + ',' },
      { p: 'I\'d like to run a clinical audit' + (place ? ' in ' + place : '') + ', and I was hoping you might supervise it.' },
      { h: 'The audit' },
      { b: noDot(p.question).replace(/\?+$/, '') + '?' },
      { p: stdLine + (targetPhrase(p.target) ? ' ' + targetPhrase(p.target) : '') },
      { h: 'How we\'ll do it' },
      { p: [sample, change].filter(Boolean).join(' ') },
      { h: 'Timeline' + (d.startDate ? ', starting ' + dateGBs(d.startDate) : '') },
      { list: supTimeline(p, d.startDate) },
      { p: 'If you are happy with the audit and the standard, would you agree to be my supervisor for this audit? Please find the proposal attached, with a form to fill in.' },
      { p: 'Best wishes,', sig: [d.lead || '[your name]', d.role, d.email].filter(Boolean) },
      { links: supLinks(p) }
    ];
    var html = '', text = '';
    blocks.forEach(function (b) {
      if (b.h) { html += '<p style="margin:16px 0 4px" data-h><strong>' + esc(b.h) + '</strong></p>'; text += '\n' + b.h.toUpperCase() + '\n'; }
      else if (b.b) { html += '<p style="margin:0 0 8px"><strong>' + esc(b.b) + '</strong></p>'; text += b.b + '\n\n'; }
      else if (b.links) {
        if (!b.links.length) return;
        html += '<p style="margin:18px 0 4px;font-size:10pt;color:#555"><strong>Links</strong></p>' + b.links.map(function (l) {
          return '<p style="margin:0 0 2px;font-size:10pt"><a href="' + attr(l[1]) + '">' + esc(l[0]) + '</a></p>'; }).join('');
        text += '\nLINKS\n' + b.links.map(function (l) { return l[0] + ': ' + l[1]; }).join('\n') + '\n';
      }
      else if (b.list) { if (!b.list.length) return; html += '<ul style="margin:0 0 8px;padding-left:20px">' + b.list.map(function (x) { return '<li>' + esc(x) + '</li>'; }).join('') + '</ul>'; text += b.list.map(function (x) { return '- ' + x; }).join('\n') + '\n'; }
      else if (b.p) {
        html += '<p style="margin:0 0 8px">' + esc(b.p) + (b.sig ? '<br>' + b.sig.map(esc).join('<br>') : '') + '</p>';
        text += b.p + (b.sig ? '\n' + b.sig.join('\n') : '') + '\n' + (b.sig ? '' : '\n');
      }
    });
    text = text.replace(/\n{3,}/g, '\n\n').replace(/\n+(?=[A-Z][A-Z' ]+\n)/g, '\n\n').trim();
    return { subject: 'New audit proposal: ' + (p.short || (p.topic ? cap(p.topic) : '') || shortTitle(d.title || p.question)), html: html, text: text };
  }
  // Links sit at the bottom of the email and the proposal, never in the middle of a sentence.
  function supLinks(p) {
    var st = p.standard || {}, out = [], u = safeUrl(st.url), site = BE.cfg && BE.cfg.legal && safeUrl(BE.cfg.legal.site_url);
    if (u) out.push([cap(humanSource(st.source) || 'The standard'), u]);
    if (site && S.pById.has(p.id)) out.push(['The full protocol on Ai4Qi', site.replace(/\/+$/, '') + '/#/proposed/' + encodeURIComponent(p.id)]);
    return out;
  }
  // The signed-in trainee's own details, from their account; supervisor details are always typed.
  function supProfile() {
    var p = BE.user ? (BE.profile || {}) : {};
    return { lead: p.full_name || '', email: BE.user ? BE.user.email || '' : '', site: p.organisation || '', department: p.department || '',
      role: [p.grade, p.specialty].filter(function (x) { return x && !/prefer not/i.test(x); }).join(', ') };
  }
  function supFillFromProfile(box) {
    if (!BE.user) return;
    var go = function () {
      var pf = supProfile(), f = box.querySelector('[data-sup-form]'), r = supRun(box), changed = false;
      ['lead', 'email', 'role', 'site', 'department'].forEach(function (k) {
        if (pf[k] && f.elements[k] && !f.elements[k].value && !(r && r.details[k])) { f.elements[k].value = pf[k]; changed = true; }
      });
      if (changed) supRefresh(box, false);
    };
    if (BE.profile !== undefined) return go();
    sbClient().then(function (c) { return loadProfile(c); }).then(go, function () {});
  }
  // "What proportion of adults at high risk from sepsis (NEWS2…) receive IV antibiotics within 1 hour of…?"
  // -> "Adults at high risk from sepsis receive IV antibiotics within 1 hour of…" (for subject lines)
  // Method cells as short, scannable lists.
  function splitTop(t, sepRe) {                 // split on a separator, but not inside brackets
    var out = [], cur = '', depth = 0;
    String(t || '').split('').forEach(function (ch) {
      if (ch === '(') depth++; else if (ch === ')') depth = Math.max(0, depth - 1);
      if (depth === 0 && sepRe.test(ch)) { out.push(cur); cur = ''; } else cur += ch;
    });
    out.push(cur);
    return out.map(function (x) { return x.trim().replace(/[.\s]+$/, ''); }).filter(Boolean);
  }
  function bullets(items, cls) { return items.length ? '<ul class="' + cls + '">' + items.map(function (x) { return '<li>' + linkify(cap(x)) + '</li>'; }).join('') + '</ul>' : ''; }
  function selectionHtml(t) {
    var inc = [], exc = [];
    String(t || '').replace(/\.\s+(?=[A-Z])/g, '.\u0001').split('\u0001').forEach(function (sent) {
      var x = sent.trim(); if (!x) return;
      if (/^(exclude[sd]?|exclusions?|excluding)\b\s*:?\s*/i.test(x)) exc = exc.concat(splitTop(x.replace(/^(exclude[sd]?|exclusions?|excluding)\b\s*:?\s*/i, ''), /[,;]/));
      else inc.push(x.replace(/^(include[sd]?|inclusions?( criteria)?)\b\s*:?\s*/i, '').replace(/[.\s]+$/, ''));
    });
    return (inc.length ? '<p class="sel-h is-inc">Include</p>' + bullets(inc, 'bul inc') : '') + (exc.length ? '<p class="sel-h is-exc">Exclude</p>' + bullets(exc, 'bul exc') : '') ||
      '<p class="prose">' + linkify(t) + '</p>';
  }
  function passHtml(p) {
    if (!p.pass_short) return '<p class="prose">' + linkify(p.pass) + '</p>';
    return '<p class="pass-short">' + esc(cap(p.pass_short)) + '</p><details class="pass-full"><summary>Exact definition</summary><p>' + linkify(p.pass) + '</p></details>';
  }
  // Heading and subheading: the audit's short name, then the full question underneath.
  function auditName(p) { return cap(String((p && p.short) || shortTitle((p && p.question) || ''))); }
  function titleBlock(p, tag) {
    tag = tag || 'h1';
    return '<' + tag + ' class="audit-name">' + esc(auditName(p)) + '</' + tag + '><p class="audit-q">' + esc(p.question) + '</p>';
  }
  function shortTitle(q) {
    var t = String(q || '').replace(/\s*\([^)]*\)/g, '').replace(/\?+\s*$/, '').trim()
      .replace(/^(what (proportion|percentage|share|fraction) of|what %( of)?|how (many|often|much)( of)?|in what proportion of|do|does|are|is|were|was)\s+/i, '');
    t = t.charAt(0).toUpperCase() + t.slice(1);
    return t.split(/\s+/).slice(0, 5).join(' ');
  }
  function supMissing(box) {
    var d = supWho(box);
    return SUP_FIELDS.filter(function (x) { return x[3] && !d[x[0]]; });
  }
  function supervisorBox(p, run) {
    var me = meGet(), d = run ? run.details : {};
    var pf = supProfile();
    var val = { lead: d.lead || pf.lead || me.lead, role: pf.role || me.role, email: pf.email || me.email, supervisor: d.supervisor || me.supervisor, supEmail: me.supEmail,
      site: d.site || pf.site || me.site, department: d.department || pf.department || me.department, startDate: d.startDate || '', sampleSize: d.sampleSize || '' };
    return '<details class="sup-box no-print" id="send-supervisor" data-sup="' + attr(p.id) + '"' + (run ? ' data-sup-run="' + attr(run.id) + '"' : '') + '>' +
      '<summary><span class="sup-ic" aria-hidden="true"><svg width="20" height="20" viewBox="0 0 24 24"><path d="M3 6.5h18v11H3z M3 7l9 6.5L21 7" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linejoin="round"/></svg></span>' +
      '<span><b>Send to your supervisor</b><small>A ready-to-send email and a Word proposal with a sign-off box</small></span></summary>' +
      '<div class="sup-body"><form class="det-form" data-sup-form novalidate>' + SUP_FIELDS.map(function (x) {
        return '<div class="rec-f"><label for="sp-' + x[0] + '">' + x[1] + (x[3] ? '' : ' <span class="muted">optional</span>') + '</label><input id="sp-' + x[0] + '" name="' + x[0] + '" type="' + x[2] + '"' +
          (x[2] === 'number' ? ' min="1" max="2000"' : ' maxlength="120"') + (x[0] === 'site' ? ' list="org-list"' : '') + ' autocomplete="' + (x[0] === 'lead' ? 'name' : x[0] === 'email' ? 'email' : 'off') + '" value="' + attr(val[x[0]] || '') + '"></div>';
      }).join('') + '</form>' + orgList() +
      '<p class="sup-missing" data-sup-missing role="status"></p>' +
      '<div class="sup-mail"><div class="rec-f"><label for="sp-subj">Subject</label><input id="sp-subj" data-sup-subject></div>' +
      '<div class="rec-f"><span class="sup-lab" id="sp-body-l">Email <span class="muted">Click to edit before copying</span></span>' +
      '<div class="sup-preview" id="sp-body" data-sup-body contenteditable="true" role="textbox" aria-multiline="true" aria-labelledby="sp-body-l"></div></div></div>' +
      '<div class="out-grid sup-actions">' +
      '<button class="out-btn" type="button" data-sup-docx><b>Proposal</b><span>Word, with a sign-off box</span></button>' +
      '<button class="out-btn" type="button" data-sup-copy><b>Copy email</b><span>Pastes with bold headings into Outlook or NHSmail</span></button>' +
      (window.AI4QI_EMBED ? '' : '<a class="out-btn" data-sup-mailto href="mailto:"><b>Open in email app</b><span>Copies the email, opens a new message: paste it in</span></a>') +
      '</div>' +
      '<p class="muted sup-note">Download the proposal first and attach it to the email; an email link cannot attach files. The proposal holds the protocol only, never patient data. Your supervisor\'s details are remembered on this device only. If you are signed in, your own name and hospital come from your account profile.</p>' +
      '<p class="form-status" role="status" data-sup-status></p></div></details>';
  }
  function supRefresh(box, keepBody) {
    var p = supProtocol(box); if (!p) return;
    var d = supWho(box), m = supEmail(p, d), pv = box.querySelector('[data-sup-body]');
    if (!keepBody || !pv.innerHTML) { box.querySelector('[data-sup-subject]').value = m.subject; pv.innerHTML = m.html; }
    var miss = supMissing(box);
    box.querySelector('[data-sup-missing]').textContent = miss.length ? 'Still to fill in: ' + miss.map(function (x) { return x[1].toLowerCase(); }).join(', ') + '.' : '';
    SUP_FIELDS.forEach(function (x) { var el = box.querySelector('#sp-' + x[0]); if (x[3]) el.setAttribute('aria-invalid', box.hasAttribute('data-touched') && !d[x[0]] ? 'true' : 'false'); });
    var ml = box.querySelector('[data-sup-mailto]');          // a long body in a mailto link breaks many mail apps:
    if (ml) ml.href = 'mailto:' + encodeURIComponent(d.supEmail || '').replace(/%40/g, '@') +   // address and subject only
      '?subject=' + encodeURIComponent(box.querySelector('[data-sup-subject]').value);
  }
  function supText(pv) {       // the edited email as plain text: headings in capitals, lists with dashes
    var out = [];
    pv.childNodes.forEach(function (n) {
      if (n.nodeName === 'UL') out.push(Array.prototype.map.call(n.querySelectorAll('li'), function (li) { return '- ' + li.textContent.trim(); }).join('\n'));
      else if (n.hasAttribute && n.hasAttribute('data-h')) out.push('\n' + n.textContent.trim().toUpperCase());
      else if (n.nodeType === 1 && n.querySelector && n.querySelector('a')) out.push(Array.prototype.map.call(n.querySelectorAll('a'), function (a) { return a.textContent.trim() + ': ' + a.getAttribute('href'); }).join('\n'));
      else if (n.nodeType === 1) out.push((n.innerText || n.textContent).trim());
      else if (n.textContent.trim()) out.push(n.textContent.trim());
    });
    return out.join('\n\n').replace(/\n{3,}/g, '\n\n').trim();
  }
  function supSaveRun(box) {   // the start date and sample typed here also update the running audit
    var r = supRun(box); if (!r) return;
    var d = supWho(box);
    ['lead', 'supervisor', 'site', 'department', 'startDate', 'sampleSize'].forEach(function (k) { if (d[k]) r.details[k] = k === 'sampleSize' ? +d[k] : d[k]; });
    runPut(r);
  }
  document.addEventListener('toggle', function (e) {
    var box = e.target;
    if (box.matches && box.matches('.sup-box') && box.open) { supRefresh(box, true); supFillFromProfile(box); }
  }, true);
  document.addEventListener('input', function (e) {
    var box = e.target.closest && e.target.closest('.sup-box');
    if (!box) return;
    if (e.target.closest('[data-sup-form]')) {
      var d = supWho(box), me = meGet();
      SUP_FIELDS.forEach(function (x) { if (x[4]) me[x[0]] = d[x[0]]; }); meSet(me);
      supRefresh(box, false);
    } else supRefresh(box, true);
  });
  document.addEventListener('change', function (e) {
    var box = e.target.closest && e.target.closest('.sup-box');
    if (box && e.target.closest('[data-sup-form]')) supSaveRun(box);
  });
  document.addEventListener('click', function (e) {
    var open = e.target.closest('[data-sup-open]');
    if (open) { var bx = document.getElementById('send-supervisor'); if (bx) { bx.open = true; bx.scrollIntoView({ behavior: 'smooth', block: 'start' }); bx.querySelector('input').focus({ preventScroll: true }); } return; }
    var box = e.target.closest('.sup-box'); if (!box) return;
    var act = e.target.closest('[data-sup-docx],[data-sup-copy],[data-sup-mailto]'); if (!act) return;
    var st = box.querySelector('[data-sup-status]'), p = supProtocol(box); if (!p) return;
    box.setAttribute('data-touched', ''); supRefresh(box, true);
    var miss = supMissing(box), warn = miss.length ? ' Fill in ' + miss.map(function (x) { return x[1].toLowerCase(); }).join(', ') + ' before you send it.' : '';
    var d = supWho(box), name = fileSlug(d.title || p.question) + '-proposal';
    if (act.matches('[data-sup-docx]')) {
      sayIn(st, 'Making the Word proposal…');
      exporter().then(function (x) { return x.proposalDocx(p, d); }).then(function (b) { return saveFile(name + '.docx', b); })
        .then(function () { sayIn(st, 'Proposal ready. Attach it to the email.' + warn); }, function (err) { sayIn(st, downloadError(err)); });
    } else if (act.matches('[data-sup-copy],[data-sup-mailto]')) {
      var viaMail = act.matches('[data-sup-mailto]');
      var pv = box.querySelector('[data-sup-body]'), subj = box.querySelector('[data-sup-subject]').value;
      var plain = supText(pv), ok = function () {
        sayIn(st, viaMail ? 'Email copied. Your email app is opening a new message: click in the message and paste (Ctrl+V or Cmd+V).' + warn :
          'Email copied. Paste it into a new message; the subject is: ' + subj + warn);
      };
      var fallback = function () {
        var r = document.createRange(); r.selectNodeContents(pv); var sel = window.getSelection(); sel.removeAllRanges(); sel.addRange(r);
        sayIn(st, 'The email is selected: press Ctrl+C (or Cmd+C) to copy it.' + warn);
      };
      try {
        if (window.ClipboardItem && navigator.clipboard.write) {
          navigator.clipboard.write([new ClipboardItem({ 'text/html': new Blob(['<div style="font-family:Calibri,Arial,sans-serif;font-size:11pt">' + pv.innerHTML + '</div>'], { type: 'text/html' }),
            'text/plain': new Blob([plain], { type: 'text/plain' }) })]).then(ok, function () { navigator.clipboard.writeText(plain).then(ok, fallback); });
        } else navigator.clipboard.writeText(plain).then(ok, fallback);
      } catch (err) { fallback(); }
    }
  });

  function chooseActions(id) {
    var run = Array.from(S.runs.values()).filter(function (r) { return r.auditId === id && !r.closed; })[0];
    return (run ? '<a class="btn" href="#/run/' + attr(run.id) + '">Open in My audits</a>' :
      '<button class="btn" type="button" data-choose="' + attr(id) + '">Choose this audit</button>') +
      '';
  }
  document.addEventListener('click', function (e) {
    var b = e.target.closest('[data-proto-xlsx]');
    if (!b) return;
    var p = anyAudit(b.getAttribute('data-proto-xlsx')), st = b.parentNode.querySelector('[data-proto-status]') || b.parentNode.querySelector('[data-copy-status]');
    if (!p) return;
    sayIn(st, 'Making the data sheet…');
    exporter().then(function (x) { return x.templateXlsx(p, { code: p.id + '//C1' }); }).then(function (blob) { return saveFile(p.id + '-data-sheet.xlsx', blob); })
      .then(function () { sayIn(st, 'Data sheet ready.'); }, function (err) { sayIn(st, downloadError(err)); });
  });
  // Template downloads, under the data sheet: the Excel sheet (drop-downs, results) and the plain CSV.
  function sheetButton(id) {
    return '<button class="btn" type="button" data-proto-xlsx="' + attr(id) + '">' +
      '<svg width="18" height="18" viewBox="0 0 24 24" aria-hidden="true"><path d="M12 4v11m0 0l-4.5-4.5M12 15l4.5-4.5M5 19h14" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/></svg>Download data sheet (Excel)</button>';
  }
  function renderProposed(id) {
    var p = S.pById.get(id);
    if (!p) return renderNotFound();
    var dl = sheetButton(p.id) + (window.AI4QI_EMBED ? copyBtn(p.id) : chooseActions(p.id)) + '<span class="copy-status" role="status" data-proto-status></span>';

    var simHtml = resourceSection(resourcesFor(p.question)).replace('#/search?q=', '#/search?q=' + encodeURIComponent(p.question));
    var body = protocolBody(p, dl, { status: effortScale(p) + (p.novelty ? '<p class="gap-note"><strong>Gap:</strong> ' + esc(cap(p.novelty)) + '</p>' : ''),
      feedback: feedbackBox(p.id), after: simHtml ? [['Similar published audits', simHtml]] : [] });

    page('<nav class="breadcrumb" aria-label="Breadcrumb"><a href="#/">Home</a> › <a href="#/proposed">Proposed audits</a> › <a href="#/proposed?area=' + encodeURIComponent(p.area) + '">' + esc(p.area) + '</a></nav>' +
      '<article class="doc"><header class="doc-head"><div class="eyebrow"><span class="id-tag">' + esc(p.id) + '</span>' + PROPOSED_BADGE + badge(p.area, 'primary') + '</div>' +
      titleBlock(p) +
      (window.AI4QI_EMBED ? '<div class="doc-actions">' + chooseActions(p.id) + '</div>' : '') + '</header>' +
      supervisorBox(p) + body + '</article>', p.id + ' ' + trunc(p.question, 60), 'proposed');
    showUsefulCount(p.id);
  }

  /* Everyone's votes, live from the server (feedback_summary). They only count once an audit has
     MIN_VOTES votes, so a handful of early responses cannot move anything. */
  var MIN_VOTES = 5;
  function votesOf(p) {
    var live = S.votes && S.votes.get(p.id), f = live || p.feedback || {};
    return { up: Number(f.up) || 0, down: Number(f.down) || 0 };
  }
  function voteScore(p) { var v = votesOf(p), n = v.up + v.down; return n >= MIN_VOTES ? (v.up - v.down) / n : 0; }
  function mixedFeedback(p) { var v = votesOf(p); return v.up + v.down >= MIN_VOTES && v.down >= 2 * Math.max(1, v.up); }
  function renderProposedList(params) {
    var area = params.get('area') || '';
    var areas = new Map();
    S.proposed.forEach(function (p) {
      if (area && p.area !== area) return;
      if (!areas.has(p.area)) areas.set(p.area, []);
      areas.get(p.area).push(p);
    });
    areas.forEach(function (list) { list.sort(function (a, b) { return voteScore(b) - voteScore(a); }); });   // stable: unrated keep their order
    var keys = Array.from(areas.keys()).sort();
    var html = '<nav class="breadcrumb" aria-label="Breadcrumb"><a href="#/">Home</a> › ' + (area ? '<a href="#/proposed">Proposed audits</a> › ' + esc(area) : 'Proposed audits') + '</nav>' +
      '<h1>' + esc(area || 'Proposed audits') + '</h1>' +
      '<p class="page-intro">Ready-to-run audit protocols built on current standards. Each has one measurable question, the exact standard wording, a pass definition, a data collection template and a timeline. ' + PROPOSED_BADGE + '</p>' +
      '<div class="toolbar"><label class="visually-hidden" for="pf">Filter proposed audits</label><input id="pf" type="search" placeholder="Filter by words in the question" data-listfilter=".result"></div>' +
      keys.map(function (k) {
        return '<div class="area-group"><h2>' + esc(k) + ' <span class="muted" style="font-weight:400;font-size:.9rem">(' + areas.get(k).length + ')</span></h2><ul class="result-list">' +
          areas.get(k).map(function (p) {
            return '<li class="result result-proposed" data-text="' + attr((p.id + ' ' + p.question + ' ' + (p.standard || {}).source).toLowerCase()) + '">' +
              '<h3><a href="#/proposed/' + attr(p.id) + '">' + esc(auditName(p)) + '</a></h3><p class="audit-q small">' + esc(p.question) + '</p><p class="meta"><span class="id-tag">' + esc(p.id) + '</span> · ' +
              esc(trunc(humanSource((p.standard || {}).source), 120)) + (p.novelty ? ' · ' + esc(cap(p.novelty)) : '') + (mixedFeedback(p) ? ' · ' + badge('Mixed feedback', 'warn') : '') + '</p></li>';
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
      ['Specialty', esc(String(f.specialty || '').replace(/^Orthopaedics\b/, 'Trauma and orthopaedics'))],
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
          '<h3 title="' + attr(s.source) + '">' + esc(humanSource(s.source)) + '</h3><blockquote class="standard"><p>“' + esc(s.wording) + '”</p></blockquote>' + niceAttribution(s) +
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
    '{"question": string, "short": string (the audit\'s name in 3-4 words, e.g. "Sepsis antibiotics within 1 hour"), "area": string (clinical area), "alternative": {"question": string, "why": string}, "why": string (2-3 short, forceful sentences, each a separate point: the harm to patients, the gap in practice, why audit it now; lead with a number from the evidence where there is one; no hedging), ' +
    '"standard": {"source": string, "wording": string, "url": string or ""}, "pass": string, "pass_short": string (the pass criterion in 14 words or fewer, keeping the key threshold), "population": string (who to include, then a sentence starting "Exclude" listing exclusions separated by commas), ' +
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
      (avoid && avoid.fix && avoid.fix.length ? '\n\nTHE USER SAID THE EARLIER VERSION WAS NOT RIGHT, BECAUSE: ' + avoid.fix.join(' | ') +
        '. Fix exactly that. Too generic: narrow the population, name one ward or pathway and a time window. Too specific: widen it to a common, high-volume group. ' +
        'Poor framing: one plain, measurable question with an obvious pass. Too complex: fewer data fields and a shorter timeline. ' +
        'Not relevant to my specialty: stay inside the specialty the theme implies. Not an important topic: pick the higher-harm, higher-volume aspect.' : '') +
      (avoid && avoid.length ? '\n\nALREADY OFFERED ON THIS THEME. Write a DIFFERENT audit: a different aspect of the theme, a different question and a different standard where possible. Do not repeat these:\n' +
        avoid.map(function (x) { return '- ' + x; }).join('\n') : '');
  }

  /* Tidy what came back: keep only known citations and sensible shapes. */
  /* Every audit must quote its standard word for word. When a protocol has a source but no wording
     (an AI build on a NICE topic whose recommendation number is not in our checked list, or an older
     saved audit), use the checked standard that fits it best: the same recommendation, else the most
     relevant recommendation of the same guideline, else the closest standard overall. */
  function stdKey(src) {
    var t = String(src || '').toLowerCase(), g = (t.match(/\b(ng|cg|qs|dg|ta|htg|ph|sc)\s?\d+/) || [''])[0].replace(/\s/g, ''),
      r = (t.match(/\b(?:rec(?:ommendation)?s?|statement|qs\d+\s+statement)\s*(\d+(?:\.\d+)*)/) || [])[1] || '';
    return { g: g, k: g && r ? g + '|' + r : '' };
  }
  function bestByText(list, text) {
    var want = tokens(text), best = null, top = 0;
    (list || []).forEach(function (s) {
      var have = new Set(tokens(s.source + ' ' + s.wording)), n = 0;
      want.forEach(function (t) { if (have.has(t)) n++; });
      if (n > top) { top = n; best = s; }
    });
    return top >= 2 ? best : null;
  }
  function fillStandard(p, ranked) {
    if (!p) return false;
    var st = p.standard = p.standard || {};
    if (String(st.wording || '').trim()) return false;
    var pool = (ranked || []).concat(S.standards || []).filter(function (s) { return s && String(s.wording || '').trim(); });
    var want = stdKey(st.source), text = [p.question, p.pass, p.area].join(' '), hit = null;
    if (want.k) hit = pool.filter(function (s) { return stdKey(s.source).k === want.k; })[0] || null;
    if (!hit && want.g) hit = bestByText(pool.filter(function (s) { return stdKey(s.source).g === want.g; }), text);
    if (!hit) hit = bestByText(pool, text);
    if (!hit) return false;
    st.source = hit.source; st.wording = hit.wording; if (hit.url) st.url = safeUrl(hit.url);
    return true;
  }

  function normaliseBuilt(o, q, res, n) {
    if (!o || typeof o !== 'object' || !o.question) throw { code: 'invalid_json' };
    function str(v) { return typeof v === 'string' ? v.trim() : (v == null ? '' : String(v)); }
    function list(v) { return Array.isArray(v) ? v.map(str).filter(Boolean) : []; }
    var st = o.standard || {};
    var p = {
      id: builtId(q, n), variant: +n || 1, topic: q, built: new Date().toISOString().slice(0, 10),
      question: str(o.question), area: str(o.area),
      alternative: o.alternative && o.alternative.question ? { question: str(o.alternative.question), why: str(o.alternative.why) } : null,
      short: str(o.short).split(/\s+/).slice(0, 6).join(' '), why: str(o.why), standard: { source: str(st.source), wording: str(st.wording), url: safeUrl(str(st.url)) },
      pass: str(o.pass), pass_short: str(o.pass_short).split(/\s+/).slice(0, 16).join(" "), population: str(o.population), sample: str(o.sample), data_source: str(o.data_source),
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
    if (!niceAI() && isNice(p.standard)) p.standard.wording = '';   // NICE wording comes from the library copy, never from the AI
    fillStandard(p, res.stds);
    var sim = o.similar;
    if (sim && sim.id && S.pById.has(str(sim.id)) && str(sim.better_because)) p.similar = { id: str(sim.id), better_because: str(sim.better_because) };
    return p;
  }

  var BUILD_STEPS = [['question', 'Audit question'], ['standard', 'Standard'], ['template', 'Data template'], ['timeline', 'Timeline'],
    ['change', 'Change'], ['reaudit', 'Re-audit'], ['evidence', 'Evidence'], ['pitfalls', 'Pitfalls'], ['pearls', 'Pearls']];
  function progressHtml(text) {
    var qm = /"question"\s*:\s*"((?:[^"\\]|\\.)*)"/.exec(text || '');
    var qText = qm ? qm[1].replace(/\\"/g, '"').replace(/\\n/g, ' ') : '';
    return (qText ? '<p class="build-q"><span class="muted">Audit question</span><br><strong>' + esc(qText) + '</strong></p>' : '') + '<ol class="build-steps">' + BUILD_STEPS.map(function (s) {
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
          if (/ndjson/.test(r.headers.get('content-type') || '') && r.body && r.body.getReader) return readStream(r.body.getReader());
          return r.json();                                  // a saved protocol comes back whole
        }, function (e) { throw { code: e && e.name === 'AbortError' ? 'cancelled' : 'upstream_error' }; });
    }
    // The function streams one JSON object per line: {"hb": 1} heartbeats, {"t": more text}, then
    // {"done": protocol} or {"error": code}. Give up after 60 s of silence or 3 minutes in all.
    function readStream(reader) {
      var dec = new TextDecoder(), buf = '', text = '', started = Date.now();
      var ERR = { 'not an audit topic': 'refused', 'invalid json': 'invalid_json', busy: 'rate_limited', 'daily limit': 'rate_limited',
        'monthly limit': 'site_limit', 'sign in': 'sign_in', 'no access': 'no_access' };
      function next() {
        var timer;
        var quiet = new Promise(function (res, rej) { timer = setTimeout(function () { rej({ code: 'upstream_error' }); }, 60000); });
        return Promise.race([reader.read(), quiet]).then(function (c) { clearTimeout(timer); return c; }, function (e) { clearTimeout(timer); reader.cancel().catch(function () {}); throw e; });
      }
      return (function pump() {
        if (Date.now() - started > 180000) { reader.cancel().catch(function () {}); return Promise.reject({ code: 'upstream_error' }); }
        return next().then(function (chunk) {
          if (chunk.done) throw { code: 'upstream_error' };  // ended without a result
          buf += dec.decode(chunk.value, { stream: true });
          var lines = buf.split('\n'); buf = lines.pop();
          for (var i = 0; i < lines.length; i++) {
            if (!lines[i]) continue;
            var m; try { m = JSON.parse(lines[i]); } catch (e) { continue; }
            if (m.done) { reader.cancel().catch(function () {}); return m.done; }
            if (m.error) throw { code: ERR[m.error] || 'upstream_error' };
            if (typeof m.t === 'string') { text += m.t; if (onText) onText({ text: text }); }
            else if (m.hb && onText && !text) onText({ text: '', waiting: Date.now() - started });
          }
          return pump();
        });
      })().catch(function (e) { throw e && e.code ? e : { code: e && e.name === 'AbortError' ? 'cancelled' : 'upstream_error' }; });
    }
  }
  var BUILD_ERRORS = {
    not_granted: 'Building audits needs permission to use Claude on your account. Reload the page to be asked again.',
    sampling_disabled: 'Building audits is not available for this account.',
    rate_limited: 'You have reached the limit for now. Please try again later.',
    sign_in: 'Sign in (free, with an emailed code) to build audits. Institutional emails get instant access; others need a quick approval.',
    no_access: 'Institutional emails (NHS, HSE or university) get instant access. Other emails need a quick approval: ask for access.',
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
    var others = n > 1 ? variantsOf(q).filter(function (b) { return b.id !== id; }) : [];
    var avoid = others.map(function (b) { return b.question; });
    // 'Not quite right?' learns why: the reasons (and comment) the user gave on earlier versions.
    avoid.fix = others.map(function (b) { var f = fbFor(b.id); return f && f.rating === 'down' ? f : null; }).filter(Boolean)
      .map(function (f) { return (f.reasons || []).join(', ') + (f.comment ? (f.reasons && f.reasons.length ? '; ' : '') + '"' + scrub(String(f.comment).slice(0, 200), { redacted: 0 }) + '"' : ''); })
      .filter(Boolean);
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
    if (!accessGate(function () { renderBuild(params); })) return;
    if (fresh) BUILDS.delete(id);
    var entry = startBuild(q, n, fresh), res = entry.res;
    if (n > 1) prefetchNext({ topic: q, variant: n });      // 'Not quite right?': line up the one after straight away
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
      if (text && now - last > 400 && prog && prog.isConnected) { last = now; prog.innerHTML = progressHtml(text); }
    };
    var slow = setTimeout(function () {
      var msg = main.querySelector('[data-build-msg]');
      if (msg && here() && !entry.text) msg.textContent = 'Still writing… a full protocol can take up to two minutes.';
    }, 20000);
    entry.promise.then(function () { clearTimeout(slow); }, function () { clearTimeout(slow); });
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
      else if (act && code === 'no_access') act.innerHTML = '<a class="btn" href="#/access">Get access</a>';
      else if (act) act.innerHTML = code === 'no_generator' || code === 'not_granted' || code === 'sampling_disabled' ? '' :
        '<a class="btn" href="#/build?q=' + encodeURIComponent(q) + (n > 1 ? '&n=' + n : '') + '&fresh=1">Try again</a>';
      if (code === 'no_generator' && res.props.length && act) {
        act.insertAdjacentHTML('afterend', '<p class="prose">Closest ready-made protocol: <a href="#/proposed/' + attr(res.props[0].id) + '">' + esc(res.props[0].id + ' – ' + res.props[0].question) + '</a></p>');
      }
    });
  }

  var BUILT_BADGE = badge('Your bespoke audit', 'primary');
  function showBuilt(p, res, crumbs) {
    fillStandard(p);
    var dl = sheetButton(p.id) + (window.AI4QI_EMBED ? copyBtn(p.id) : chooseActions(p.id)) + '<span class="copy-status" role="status" data-proto-status></span>';
    if (p.similar) {
      var sp = S.pById.get(p.similar.id);
      if (sp) p._sim = '<aside class="similar-box no-print"><h2>A ready-made audit may suit you better</h2>' +
        '<p><a href="#/proposed/' + attr(sp.id) + '"><span class="id-tag">' + esc(sp.id) + '</span> ' + esc(sp.question) + '</a></p>' +
        '<p class="muted">' + esc(p.similar.better_because) + '</p></aside>';
    }
    var resHtml = resourceSection(res).replace('#/search?q=', '#/search?q=' + encodeURIComponent(p.topic));
    var after = [];
    if (resHtml) after.push(['Similar published audits', resHtml]);
    var status = effortScale(p);
    var body = protocolBody(p, dl, { status: status, after: after, feedback: feedbackBox(p.id) }) +
      '<aside class="ai-note"><p><strong>How this was made.</strong> This protocol was drafted by an AI model (Claude, made by Anthropic) from the topic you typed, ' +
      'using published audits and standards from the Ai4Qi library. It is a draft. Check the standard against its linked source, and ask your supervisor to review the protocol before you collect data. ' +
      'Your audit records are never sent to the AI or to Ai4Qi. <a href="#/how-it-works">How Ai4Qi works</a></p></aside>';
    page(crumbs + '<article class="doc"><header class="doc-head"><div class="eyebrow"><span class="id-tag">' + esc(p.topic) + '</span>' + BUILT_BADGE +
      (p.area ? badge(p.area, 'primary') : '') + '</div>' +
      titleBlock(p) +
      (window.AI4QI_EMBED ? '<div class="doc-actions">' + chooseActions(p.id) + '</div>' : '') + variantsNav(p) + '</header>' +
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
    var list = S.proposed.filter(function (p) { return fb.get(p.id) !== 'down' && !mixedFeedback(p); });
    function effortMin(p) { var m = String(p.effort || '').match(/(\d+)\s*min/); return m ? +m[1] : 60; }
    function weeks(p) { var s = parseTimeline(p.timeline); return s ? Math.max.apply(null, s.map(function (x) { return x.b; })) : 52; }
    list = list.slice().sort(function (a, b) { return (voteScore(b) - voteScore(a)) || (weeks(a) - weeks(b)) || (effortMin(a) - effortMin(b)); }).slice(0, 12);
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
  /* Records are encrypted at rest (AES-GCM). By default the key is a random device key kept in the
     browser, so My audits opens without a passcode (Trust computers lock themselves; personal devices
     are the owner's). Optionally the key comes from a passcode (PBKDF2); then it lives only in memory
     and the audits lock after 15 idle minutes. "Shared computer" keeps everything in session storage,
     which the browser clears on close. */
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
  function vaultDevice() { return !!(V.meta && V.meta.device); }
  function loadRuns(key) {
    var box = null; try { box = JSON.parse(stGet(V.store, VAULT_DATA) || 'null'); } catch (e) {}
    return (box ? unseal(key, box) : Promise.resolve('[]')).then(function (json) {
      S.runs = new Map(); (JSON.parse(json) || []).forEach(function (r) { if (r && r.id) S.runs.set(r.id, r); });
      V.key = key; V.last = Date.now();
    });
  }
  // No passcode: a random key kept in this browser.
  function vaultCreateDevice(shared) {
    var store = shared ? sessionStorage : localStorage;
    return crypto.subtle.generateKey({ name: 'AES-GCM', length: 256 }, true, ['encrypt', 'decrypt']).then(function (key) {
      return crypto.subtle.exportKey('raw', key).then(function (raw) {
        V.meta = { v: 1, device: b64(raw), shared: !!shared };
        V.store = store; V.key = key; V.last = Date.now();
        store.setItem(VAULT_META, JSON.stringify(V.meta));
        (V.legacy || []).forEach(function (r) { if (r && r.id) S.runs.set(r.id, r); });
        return runsSave().then(function () { try { localStorage.removeItem(RUNS_KEY); } catch (e) {} V.legacy = null; });
      });
    });
  }
  function vaultOpenDevice() {
    return crypto.subtle.importKey('raw', unb64(V.meta.device), { name: 'AES-GCM' }, true, ['encrypt', 'decrypt']).then(loadRuns);
  }
  // Switch protection without losing records: re-encrypt everything under the new key and settings.
  function vaultRekey(pass, shared) {
    var runs = Array.from(S.runs.values()), old = V.store;
    var target = shared ? sessionStorage : localStorage;
    var make = pass ? (function () {
      var salt = crypto.getRandomValues(new Uint8Array(16));
      return deriveKey(pass, salt, PBKDF2_ITER).then(function (key) {
        return seal(key, 'ai4qi-ok').then(function (check) { return { key: key, meta: { v: 1, salt: b64(salt), iter: PBKDF2_ITER, check: check, shared: !!shared } }; });
      });
    })() : crypto.subtle.generateKey({ name: 'AES-GCM', length: 256 }, true, ['encrypt', 'decrypt']).then(function (key) {
      return crypto.subtle.exportKey('raw', key).then(function (raw) { return { key: key, meta: { v: 1, device: b64(raw), shared: !!shared } }; });
    });
    return make.then(function (k) {
      return seal(k.key, JSON.stringify(runs)).then(function (box) {
        target.setItem(VAULT_DATA, JSON.stringify(box)); target.setItem(VAULT_META, JSON.stringify(k.meta));
        if (old && old !== target) { try { old.removeItem(VAULT_DATA); old.removeItem(VAULT_META); } catch (e) {} }
        V.meta = k.meta; V.store = target; V.key = k.key; V.last = Date.now();
      });
    });
  }
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
    if (V.key && !vaultDevice() && Date.now() - V.last > IDLE_MS) {
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
  // How precise a percentage is with n records: 95% margin of error at the worst case (50%).
  function marginPts(n) { return n > 0 ? Math.round(1.96 * Math.sqrt(0.25 / n) * 100) : null; }
  function precisionText(n) {
    n = Math.round(+n || 0); if (n < 1) return '';
    var m = marginPts(n);
    return (n < 20 ? '<b class="prec-warn">Too few to judge.</b> ' : '') + 'With ' + n + ' records, your result is accurate to about ±' + m + ' percentage points.' +
      (n >= 20 && n < 30 ? ' Fine for a first look.' : n >= 30 ? '' : ' Aim for at least 20.');
  }
  document.addEventListener('input', function (e) {
    var i = e.target.closest && e.target.closest('[data-sample-input]'); if (!i) return;
    var n = i.parentNode.querySelector('[data-prec]'); if (n) n.innerHTML = precisionText(+i.value);
  });
  /* How many records each cycle needs to show a real rise to the target (two-sided 5%, 80% power). */
  function nNeeded(p1, p2) {
    if (!(p2 > p1) || p1 < 0 || p2 > 1) return null;
    var pb = (p1 + p2) / 2, a = 1.959964 * Math.sqrt(2 * pb * (1 - pb)), b = 0.841621 * Math.sqrt(p1 * (1 - p1) + p2 * (1 - p2));
    return Math.ceil(Math.pow(a + b, 2) / Math.pow(p2 - p1, 2));
  }
  function powerText(p, baseline) {
    var t = parseTarget(p && p.target);
    if (!t || !/[≥>]/.test(t.op) || t.value <= 5) return '';
    var known = isFinite(baseline) && baseline !== null, b = known ? baseline : Math.max(10, Math.round((t.value - 30) / 5) * 5);
    if (b >= t.value) return known ? ' Cycle 1 already meets the ' + t.op + t.value + '% target.' : '';
    var n = nNeeded(b / 100, t.value / 100); if (!n) return '';
    return ' To show a real rise ' + (known ? 'from your ' + Math.round(b) + '%' : 'from about ' + b + '% now') + ' to the ' + t.op + t.value + '% target, collect about <b>' + n + ' per cycle</b>.';
  }
  /* Is the difference between the cycles real? 95% CI (Newcombe) and a two-sided p value
     (Fisher's exact test when an expected count is under 5, otherwise chi-squared). */
  function wilsonCI(x, n) { var z = 1.959964, p = x / n, d = 1 + z * z / n, c = (p + z * z / (2 * n)) / d, h = z * Math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d; return [Math.max(0, c - h), Math.min(1, c + h)]; }
  function lgamI(n) { var s = 0; for (var i = 2; i <= n; i++) s += Math.log(i); return s; }
  function cycleTest(c1, c2) {
    var a = c2.passN || 0, b = c2.failN || 0, c = c1.passN || 0, d = c1.failN || 0, n2 = a + b, n1 = c + d, n = n1 + n2;
    if (!n1 || !n2) return null;
    var p1 = c / n1, p2 = a / n2, w1 = wilsonCI(c, n1), w2 = wilsonCI(a, n2), df = p2 - p1;
    var lo = df - Math.sqrt(Math.pow(p2 - w2[0], 2) + Math.pow(w1[1] - p1, 2)), hi = df + Math.sqrt(Math.pow(w2[1] - p2, 2) + Math.pow(p1 - w1[0], 2));
    var small = [[a + c, n2], [a + c, n1], [b + d, n2], [b + d, n1]].some(function (q) { return q[0] * q[1] / n < 5; }), pv;
    if (small) {
      var r1 = a + b, k1 = a + c, lf = function (x) { return lgamI(r1) + lgamI(n - r1) + lgamI(k1) + lgamI(n - k1) - lgamI(n) - lgamI(x) - lgamI(r1 - x) - lgamI(k1 - x) - lgamI(n - r1 - k1 + x); };
      var p0 = lf(a); pv = 0;
      for (var x = Math.max(0, r1 + k1 - n); x <= Math.min(r1, k1); x++) { var q = lf(x); if (q <= p0 + 1e-7) pv += Math.exp(q); }
    } else {
      var den = (a + b) * (c + d) * (a + c) * (b + d), z = den ? Math.sqrt(n * Math.pow(a * d - b * c, 2) / den) : 0;
      var t = 1 / (1 + 0.2316419 * z); pv = 2 * 0.3989423 * Math.exp(-z * z / 2) * t * (0.3193815 + t * (-0.3565638 + t * (1.781478 + t * (-1.821256 + t * 1.330274))));
    }
    pv = Math.min(1, pv);
    return { diff: df * 100, lo: lo * 100, hi: hi * 100, p: pv, real: pv < 0.05 };
  }
  function pFmt(p) { return p < 0.001 ? 'p < 0.001' : 'p = ' + (p < 0.01 ? p.toFixed(3) : p.toFixed(2)); }
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
      '<span><button type="button" class="link-btn" data-demo-reset>Clear example audits (yours are kept)</button> · <a href="#/demo/off">Turn off</a></span></div>';
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
    if (r.demo) d.startDate = addDays(todayIso(), -16 * 7);   // example audits are backdated; your own keep their dates
  }
  function demoStep(r) {
    var k = RUN_STAGES[stageIdx(r)][0], want = +r.details.sampleSize || 30;
    // An audit chosen in demo mode is an example (r.demo). On your own audit the demo buttons only fill the
    // empty parts with example data and mark it; nothing you entered is changed or deleted.
    if (!r.demo) { r.exampleData = true; r.committed = true; }
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
    if (!accessGate(function () { chooseAudit(p); })) return;
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
    var t = (p.template || []).filter(function (f) { return !f.added; });   // a column the trainee adds is never the pass column
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
    var pf = passField(p), passN = 0, failN = 0, dates = [], byMonth = {};
    var df = (p.template || []).filter(function (f) { return /date/.test(f.type); })[0];
    rows.forEach(function (row) {
      var v = pf ? yn(row[pf]) : '';
      if (v === 'Yes') passN++; else if (v === 'No') failN++;
      (p.template || []).forEach(function (f) { if (/date/.test(f.type) && row[f.field]) dates.push(String(row[f.field]).slice(0, 10)); });
      var m = df && String(row[df.field] || '').slice(0, 7);          // month of the first date field, for the run chart
      if (m && /^\d{4}-\d{2}$/.test(m) && (v === 'Yes' || v === 'No')) {
        byMonth[m] = byMonth[m] || { m: m, pass: 0, fail: 0 };
        byMonth[m][v === 'Yes' ? 'pass' : 'fail']++;
      }
    });
    dates.sort();
    var denom = passN + failN;
    return { n: rows.length, passN: passN, failN: failN, pct: denom ? Math.round(passN / denom * 1000) / 10 : null,
      from: dates[0] || null, to: dates[dates.length - 1] || null,
      months: Object.keys(byMonth).sort().map(function (k) { return byMonth[k]; }) };
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
    var pfield = passField(p);
    (p.template || []).forEach(function (f) {
      var isYN = /^yes\/?no$/i.test(String(f.type)) && f.field !== pfield;       // yes/no columns other than the pass one
      if (!isYN && (f.type !== 'choice' || !(f.options || []).length)) return;
      var b = { field: f.field, label: fieldLabel(f.field), cycles: {} };
      ['c1', 'c2'].forEach(function (k) {
        var counts = new Map();
        r.cycles[k].rows.forEach(function (row) { var v = isYN ? yn(row[f.field]) : row[f.field]; if (v) counts.set(v, (counts.get(v) || 0) + 1); });
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
      setup: addDays(start, 7),
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
  var EXPORT_V = '4d7ee7f4df';   // stamped by build_app_data.py: a new export.js gets a new address, so no browser keeps an old copy
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
    return loadScript('export.js?v=' + EXPORT_V, 'AI4QI_EXPORT');
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
  // Reminders are on for every signed-in user unless they switch them off once, in Account
  // (profiles.reminders_off). One email lists every step that is due, across all their audits.
  function remindersOn() { return !!BE.user && !(BE.profile && BE.profile.reminders_off); }
  // An audit counts as started once the person presses "Start this audit" (or has moved past set-up).
  // Until then they are only looking around: no reminders, nothing on the server.
  function isStarted(r) { return !!(r.committed || r.closed || stageIdx(r) > 0); }
  function syncAllRuns() { S.runs.forEach(function (r) { syncRun(r); }); }
  function syncRun(r) {
    if (!BE.url || !BE.user || r.demo) return;           // example audits never leave the device
    clearTimeout(syncTimers[r.id]);
    syncTimers[r.id] = setTimeout(function () {
      sbClient().then(function (c) {
        var i = stageIdx(r), status = STAGE_STATUS[i], jobs = [];
        if (isStarted(r)) jobs.push(c.from('my_audits').upsert({ user_id: BE.user.id, audit_id: r.auditId, status: status }, { onConflict: 'user_id,audit_id' }));
        if (remindersOn() && isStarted(r) && !r.closed && S.runs.has(r.id)) {
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
    if (!vaultSet() || vaultDevice()) {                      // no passcode: open (or set up) straight away
      page(loadingHtml('Opening My audits…'), 'My audits', 'my-audits');
      (vaultSet() ? vaultOpenDevice() : vaultCreateDevice(false)).then(afterUnlock, function () {
        page('<article class="doc narrow"><h1>My audits could not open</h1><p class="prose">This browser is not letting Ai4Qi store data (private browsing, or storage switched off). Try a normal window or another browser.</p></article>', 'My audits', 'my-audits');
      });
      return true;
    }
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
      html = '<article class="doc narrow vault"><h1>One last passcode</h1>' + intro +
        '<p class="prose">Ai4Qi no longer uses passcodes. Enter the passcode you set on this device once more; after that My audits opens straight away.</p>' +
        '<form class="stack-form" data-vault-open><label for="vpo">Passcode</label><input id="vpo" name="p" type="password" autocomplete="current-password" required>' +
        '<button class="btn" type="submit">Open My audits</button><p class="form-status" role="status" data-vault-status></p></form>' +
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
      vaultUnlock(f.elements.p.value).then(function () {
        // Passcodes are retired: re-encrypt with this browser's own key so it never asks again.
        return vaultRekey(null, !!(V.meta && V.meta.shared)).catch(function () {});
      }).then(afterUnlock, function () { sayIn(st, 'That passcode is not right.'); f.elements.p.select(); });
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
    var open = list.filter(function (r) { return !r.closed; }), done = list.filter(function (r) { return r.closed; });
    var overdueN = list.filter(function (r) { var ns = nextStep(r); return !r.closed && !r.demo && isStarted(r) && ns.due && ns.due < todayIso(); }).length;
    var startNew = '<div class="dash-new-row" role="group" aria-label="Start a new audit">' +
      '<a class="btn" href="#/">+ Build an audit</a><a class="btn btn-secondary" href="#/suggest">Suggested audits</a>' +
      '<a class="btn btn-secondary" href="#/proposed">Ready-made audits</a></div>';
    var stats = list.length ? '<p class="dash-stats"><span><b>' + open.length + '</b> in progress</span><span><b>' + done.length + '</b> completed</span>' +
      '<span' + (overdueN ? ' class="is-late"' : '') + '><b>' + overdueN + '</b> overdue</span></p>' : '';
    // One tidy row per audit: name and place, where it is in the loop, the result, what is next.
    function row(r) {
      var st = runStats(r), ns = nextStep(r), i = stageIdx(r), c1 = st.cycles[0], c2 = st.cycles[1];
      var started = isStarted(r) || r.demo;
      if (r.demo || !started) ns.due = null;
      var overdue = ns.due && ns.due < todayIso() && !r.closed;
      var name = r.details.title && r.details.title !== r.protocol.question ? r.details.title : auditName(r.protocol);
      var dots = '<span class="ar-dots" aria-hidden="true">' + RUN_STAGES.map(function (sg, k) {
        return '<i class="' + (r.closed || k < i ? 'is-done' : k === i && started ? 'is-now' : '') + '"></i>';
      }).join('') + '</span>';
      var res = c1.pct == null ? '<span class="muted">No data yet</span>' :
        '<span class="ar-pct">' + Math.round(c1.pct) + '%' + (c2.pct != null ? ' <span class="ar-arrow">→</span> <b>' + Math.round(c2.pct) + '%</b>' : '') + '</span>';
      return '<li><a class="ar" href="#/run/' + attr(r.id) + '">' +
        '<span class="ar-main"><b class="ar-name">' + esc(name) + '</b><span class="ar-meta">' + esc([r.details.site, r.details.department].filter(Boolean).join(' · ') || r.auditId) + '</span>' +
          (r.demo ? '<span class="ar-demo">Example data</span>' : '') + '</span>' +
        '<span class="ar-stage">' + dots + '<span>' + (r.closed ? 'Loop closed' : started ? RUN_STAGES[i][1] : 'Not started') + '</span></span>' +
        '<span class="ar-res">' + res + '</span>' +
        '<span class="ar-next' + (overdue ? ' is-late' : '') + '">' + (r.closed ? 'Keep the change going' : started ? esc(ns.text) : 'Press Start this audit when ready') +
          (ns.due ? '<small>' + (overdue ? 'Was due ' : 'Due ') + esc(dateGBs(ns.due)) + '</small>' : '') + '</span>' +
        '<span class="ar-go" aria-hidden="true">›</span></a></li>';
    }
    var head = '<div class="ar-head" aria-hidden="true"><span>Audit</span><span>Stage</span><span>Result</span><span>Next step</span><span></span></div>';
    var body = !list.length ?
      '<div class="empty"><p><strong>No audits yet.</strong> Build one or pick a ready-made protocol, then press <em>Choose this audit</em>. It will appear here with its data sheet, deadlines and results.</p></div>' :
      (open.length ? '<section class="dash-list"><h2>In progress</h2>' + head + '<ul class="ar-list">' + open.map(row).join('') + '</ul></section>' : '') +
      (done.length ? '<details class="dash-list dash-done"' + (open.length ? '' : ' open') + '><summary><h2>Completed <span class="count">' + done.length + '</span></h2></summary>' + head + '<ul class="ar-list">' + done.map(row).join('') + '</ul></details>' : '');
    var hello = BE.user && BE.profile && BE.profile.full_name ? 'Welcome back, ' + esc(String(BE.profile.full_name).replace(/^(dr|mr|mrs|ms|miss|prof)\.?\s+/i, '').split(/\s+/)[0]) + '.' : 'Your audits, where each one is up to, and what to do next.';
    var pf0 = BE.profile || {}, noProfile = BE.user && BE.profile !== undefined && !pf0.full_name && !pf0.grade && !pf0.specialty;
    var signedNote = S.justSignedIn || noProfile ? '<div class="welcome-card dash-signed"><p><strong>' + (S.justSignedIn ? 'You\'re signed in.' : 'Your profile is empty.') + '</strong> ' +
      (noProfile ? 'Add your name, hospital and grade once, and your supervisor emails fill themselves in.' : 'Your audits are below.') + '</p>' +
      (noProfile ? '<a class="btn btn-secondary" href="#/account">Add my details</a>' : '') + '</div>' : '';
    S.justSignedIn = false;
    var care = '<details class="rec-box dash-care"><summary>Backup and protection</summary>' +
      '<p class="muted">Everything here is stored encrypted on this device only. ' + privacyLink() + '</p>' +
      '<div class="restore"><label class="file-pick"><input type="file" accept=".json,application/json" data-restore><span class="btn btn-secondary">Restore a backup</span></label>' +
      '<span class="form-status" role="status" data-restore-status></span>' + (vaultDevice() ? '' : '<button type="button" class="link-btn" data-vault-lock>Lock now</button>') + '</div>' +
      protectionBox() + '</details>';
    page('<div class="dash-top"><div><h1>My audits</h1><p class="page-intro">' + hello + '</p></div>' + startNew + '</div>' + signedNote + stats + body + care, 'My audits', 'my-audits');
  }
  function privacyLink() { return '<a href="#/privacy">How your data is protected</a>'; }
  function protectionBox() {
    var dev = vaultDevice(), shared = !!(V.meta && V.meta.shared);
    return '<details class="rec-box protect"><summary>Protection on this device' + (shared ? ': shared computer' : '') + '</summary>' +
      '<p class="prose">Your records are stored encrypted in this browser and open like your other work on this device.</p>' +
      (dev ? '' : '<p><button type="button" class="btn btn-secondary" data-protect-nopass>Remove the old passcode</button></p>') +
      '<label class="check"><input type="checkbox" data-protect-shared' + (shared ? ' checked' : '') + '><span><strong>This is a shared computer.</strong> Delete my audits when the browser closes. Download a backup to keep your work.</span></label>' +
      '<p class="form-status" role="status" data-protect-status></p></details>';
  }
  document.addEventListener('submit', function (e) {
    var f = e.target.closest && e.target.closest('[data-protect-pass]'); if (!f) return;
    e.preventDefault();
    var st = main.querySelector('[data-protect-status]'), p1 = f.elements.p1.value;
    if (p1.length < 6) return sayIn(st, 'Use at least 6 characters.');
    if (p1 !== f.elements.p2.value) return sayIn(st, 'The two passcodes do not match.');
    sayIn(st, 'Adding the passcode…');
    vaultRekey(p1, !!(V.meta && V.meta.shared)).then(function () { route(); }, function () { sayIn(st, 'The passcode could not be added. Please try again.'); });
  });
  document.addEventListener('click', function (e) {
    if (!e.target.closest('[data-protect-nopass]')) return;
    vaultRekey(null, !!(V.meta && V.meta.shared)).then(function () { route(); }, function () { sayIn(main.querySelector('[data-protect-status]'), 'Could not change the protection. Please try again.'); });
  });
  document.addEventListener('change', function (e) {
    var c = e.target.closest && e.target.closest('[data-protect-shared]'); if (!c) return;
    var st = main.querySelector('[data-protect-status]');
    // Keep the current key type; only move where the records are kept.
    var move = function () {
      var target = c.checked ? sessionStorage : localStorage, old = V.store;
      if (old === target) return Promise.resolve();
      return seal(V.key, JSON.stringify(Array.from(S.runs.values()))).then(function (box) {
        V.meta.shared = c.checked;
        target.setItem(VAULT_DATA, JSON.stringify(box)); target.setItem(VAULT_META, JSON.stringify(V.meta));
        try { old.removeItem(VAULT_DATA); old.removeItem(VAULT_META); } catch (e2) {}
        V.store = target;
      });
    };
    move().then(function () { sayIn(st, c.checked ? 'Shared computer: your audits will be deleted when the browser closes.' : 'Your audits are kept on this device.'); },
      function () { sayIn(st, 'Could not change this. Please try again.'); c.checked = !c.checked; });
  });


  /* Results dashboard: a doughnut per cycle, the change between them, a before/after column chart
     against the target and, when dates allow, a month-by-month run chart. Plain inline SVG. */
  function donutSvg(c, t, cls) {
    var R = 44, C = 2 * Math.PI * R, pct = c.pct == null ? 0 : c.pct, len = C * Math.min(100, pct) / 100;
    var tick = '';
    if (t) { var a = t.value / 100 * 2 * Math.PI; tick = '<line class="dn-tick" x1="' + (60 + 34 * Math.sin(a)).toFixed(1) + '" y1="' + (60 - 34 * Math.cos(a)).toFixed(1) + '" x2="' + (60 + 56 * Math.sin(a)).toFixed(1) + '" y2="' + (60 - 56 * Math.cos(a)).toFixed(1) + '"/>'; }
    return '<svg class="donut ' + cls + '" viewBox="0 0 120 120" role="img" aria-label="' + (c.pct == null ? 'No data yet' : Math.round(pct) + '% met the standard') + '">' +
      '<circle class="dn-track" cx="60" cy="60" r="' + R + '"/>' +
      (c.pct == null ? '' : '<circle class="dn-val" cx="60" cy="60" r="' + R + '" stroke-dasharray="' + len.toFixed(1) + ' ' + C.toFixed(1) + '" transform="rotate(-90 60 60)"/>') + tick +
      '<text class="dn-pct" x="60" y="62" text-anchor="middle">' + (c.pct == null ? '–' : Math.round(pct) + '%') + '</text>' +
      '<text class="dn-sub" x="60" y="80" text-anchor="middle">' + (c.pct == null ? 'no data yet' : c.passN + ' of ' + (c.passN + c.failN)) + '</text></svg>';
  }
  function resultsDash(r, st) {
    var c1 = st.cycles[0], c2 = st.cycles[1], t = st.target;
    if (!c1.n && !c2.n) return '<div class="rd-empty">' + ICON.chart + '<p><strong>Your results appear here</strong> as soon as you upload your cycle 1 sheet.</p></div>';
    function met(c) { return c.pct == null || !t ? null : (t.op === '≤' || t.op === '<' ? c.pct <= t.value : c.pct >= t.value); }
    var two = c1.pct != null && c2.pct != null, key = two ? c2 : (c1.pct != null ? c1 : c2), m = met(key);
    var diff = two ? Math.round((c2.pct - c1.pct) * 10) / 10 : null;
    // Headline: what the audit showed or achieved, in words and numbers
    var head, sub;
    if (two) {
      head = diff > 0 ? 'Up from ' + Math.round(c1.pct) + '% to ' + Math.round(c2.pct) + '%' : diff < 0 ? 'Down from ' + Math.round(c1.pct) + '% to ' + Math.round(c2.pct) + '%' : 'Unchanged at ' + Math.round(c2.pct) + '%';
      var more = c2.passN - Math.round(c1.pct / 100 * (c2.passN + c2.failN));
      sub = (diff !== 0 ? (diff > 0 ? '+' : '−') + Math.abs(diff) + ' percentage points after the change' : 'No change after the intervention') +
        (more > 0 ? '. About ' + more + ' more patients out of ' + (c2.passN + c2.failN) + ' now get it right.' : '.');
    } else {
      head = Math.round(key.pct) + '% met the standard';
      sub = key.passN + ' of ' + (key.passN + key.failN) + ' patients' + (t ? (m ? ': the target is met.' : ': ' + Math.round(Math.abs(t.value - key.pct)) + ' points short of the ' + esc(t.text) + ' target.') : '.');
    }
    var badge2 = m === null ? '' : '<span class="rd-state ' + (m ? 'is-ok' : 'is-bad') + '">' + (m ? '✓ Target met' : 'Below target') + '</span>';
    // Progress bar to the target (one bar per cycle)
    function track(c, label, cls) {
      if (c.pct == null) return '';
      return '<div class="rd-track"><span class="rd-tl">' + label + '</span><div class="rd-tb"><i class="' + cls + '" style="width:' + Math.min(100, c.pct) + '%"></i>' +
        (t ? '<b class="rd-tt" style="left:' + t.value + '%" title="Target"></b>' : '') + '</div><span class="rd-tv">' + Math.round(c.pct) + '%</span></div>';
    }
    var tiles = '<div class="rd-tiles">' +
      '<div class="rd-tile"><span class="rd-ti">' + ICON.day + '</span><b>' + (key.passN + key.failN) + '</b><span>patients with a result</span></div>' +
      '<div class="rd-tile is-ok"><span class="rd-ti">✓</span><b>' + key.passN + '</b><span>met the standard</span></div>' +
      '<div class="rd-tile is-bad"><span class="rd-ti">✕</span><b>' + key.failN + '</b><span>did not</span></div>' +
      (two ? '<div class="rd-tile is-gold"><span class="rd-ti">↗</span><b>' + (diff > 0 ? '+' : diff < 0 ? '−' : '±') + Math.abs(diff) + '</b><span>points after the change</span></div>' :
        (t ? '<div class="rd-tile is-gold"><span class="rd-ti">◎</span><b>' + esc(t.op + t.value) + '%</b><span>target</span></div>' : '')) + '</div>';
    var hero = '<div class="rd-hero' + (m ? ' is-met' : '') + '">' + donutSvg(key, t, two ? 'is-c2' : 'is-c1') +
      '<div class="rd-hero-t"><span class="rd-kicker">' + (two ? 'Before and after the change' : 'Cycle 1 result') + '</span><h3>' + head + '</h3><p>' + sub + '</p>' + badge2 +
      (function () {
        var ts = two ? cycleTest(c1, c2) : null; if (!ts) return '';
        var ci = ' 95% confidence interval of the change: ' + (ts.lo >= 0 ? '+' : '') + Math.round(ts.lo) + ' to ' + (ts.hi >= 0 ? '+' : '') + Math.round(ts.hi) + ' points.';
        return '<p class="rd-sig ' + (ts.real ? 'is-real' : 'is-maybe') + '"><b>' + (ts.real ? (diff > 0 ? '✓ A real improvement' : 'A real fall') + '</b> — unlikely to be chance (' + pFmt(ts.p) + ').' :
          'Could be chance</b> (' + pFmt(ts.p) + '): with these numbers the difference is not certain. More records in each cycle would settle it.') + ci + '</p>';
      })() +
      '<div class="rd-tracks">' + track(c1, 'Cycle 1', 'is-c1') + (two ? track(c2, 'Re-audit', 'is-c2') : '') + '</div>' +
      (t ? '<p class="rd-key"><span class="dn-tick-key"></span> Target ' + esc(t.text) + '</p>' : '') + '</div></div>' + tiles;
    // Charts that add something: before/after columns once there are two cycles; a run chart when months allow
    var W = 320, H = 180, x0 = 36, y0 = 12, ch = 140;
    function y(p) { return y0 + ch - ch * Math.max(0, Math.min(100, p)) / 100; }
    var cols = two ? '<svg class="rd-cols" viewBox="0 0 ' + W + ' ' + H + '" role="img" aria-label="Cycle 1 and re-audit compared with the target">' +
      [0, 50, 100].map(function (g) { return '<line class="rd-grid" x1="' + x0 + '" y1="' + y(g) + '" x2="' + (W - 8) + '" y2="' + y(g) + '"/><text class="rd-ax" x="' + (x0 - 6) + '" y="' + (y(g) + 4) + '" text-anchor="end">' + g + '%</text>'; }).join('') +
      [[c1, 'Cycle 1', 'is-c1', 90], [c2, 'Re-audit', 'is-c2', 210]].map(function (cc) {
        var v = cc[0].pct;
        return '<rect class="rd-bar ' + cc[2] + '" x="' + cc[3] + '" y="' + y(v) + '" width="60" height="' + (y0 + ch - y(v)) + '" rx="3"/>' +
          '<text class="rd-val" x="' + (cc[3] + 30) + '" y="' + (y(v) - 5) + '" text-anchor="middle">' + Math.round(v) + '%</text><text class="rd-cat" x="' + (cc[3] + 30) + '" y="' + (H - 6) + '" text-anchor="middle">' + cc[1] + '</text>';
      }).join('') +
      '<path class="rd-rise" d="M150 ' + y(c1.pct) + ' L210 ' + y(c2.pct) + '" marker-end="url(#rd-ar)"/><defs><marker id="rd-ar" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0 0L10 5L0 10z" fill="currentColor"/></marker></defs>' +
      (t ? '<line class="rd-target" x1="' + x0 + '" y1="' + y(t.value) + '" x2="' + (W - 8) + '" y2="' + y(t.value) + '"/>' : '') + '</svg>' : '';
    var pts = [];
    [[c1, 1], [c2, 2]].forEach(function (cc) { (cc[0].months || []).forEach(function (mm) { var d = mm.pass + mm.fail; if (d) pts.push({ m: mm.m, p: mm.pass / d * 100, c: cc[1] }); }); });
    pts.sort(function (a, b) { return a.m < b.m ? -1 : a.m > b.m ? 1 : a.c - b.c; });
    var run = '';
    if (pts.length >= 3) {
      var RW = 320, step = (RW - x0 - 12) / pts.length, MON = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
      var xs = pts.map(function (q, i) { return x0 + step * (i + 0.5); });
      var srt = pts.map(function (q) { return q.p; }).sort(function (a, b) { return a - b; }), med = srt.length % 2 ? srt[(srt.length - 1) / 2] : (srt[srt.length / 2 - 1] + srt[srt.length / 2]) / 2;
      var chg = String((r.changeMade || {}).date || '').slice(0, 7), ci = /^\d{4}-\d{2}$/.test(chg) ? pts.findIndex(function (q) { return q.m >= chg; }) : -1;
      run = '<svg class="rd-run" viewBox="0 0 ' + RW + ' ' + H + '" role="img" aria-label="Month by month run chart">' +
        [0, 50, 100].map(function (g) { return '<line class="rd-grid" x1="' + x0 + '" y1="' + y(g) + '" x2="' + (RW - 8) + '" y2="' + y(g) + '"/><text class="rd-ax" x="' + (x0 - 6) + '" y="' + (y(g) + 4) + '" text-anchor="end">' + g + '%</text>'; }).join('') +
        '<line class="rd-median" x1="' + x0 + '" y1="' + y(med) + '" x2="' + (RW - 8) + '" y2="' + y(med) + '"/>' +
        (t ? '<line class="rd-target" x1="' + x0 + '" y1="' + y(t.value) + '" x2="' + (RW - 8) + '" y2="' + y(t.value) + '"/>' : '') +
        (ci > 0 ? '<line class="rd-chg" x1="' + ((xs[ci - 1] + xs[ci]) / 2).toFixed(1) + '" y1="' + y0 + '" x2="' + ((xs[ci - 1] + xs[ci]) / 2).toFixed(1) + '" y2="' + (y0 + ch) + '"/><text class="rd-chg-t" x="' + ((xs[ci - 1] + xs[ci]) / 2 + 4).toFixed(1) + '" y="' + (y0 + 10) + '">Change</text>' : '') +
        '<defs><linearGradient id="rd-fill" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="var(--viz-c2)" stop-opacity=".28"/><stop offset="1" stop-color="var(--viz-c1)" stop-opacity="0"/></linearGradient></defs>' +
        '<path class="rd-area" d="M' + xs[0].toFixed(1) + ' ' + (y0 + ch) + ' ' + pts.map(function (q, i) { return 'L' + xs[i].toFixed(1) + ' ' + y(q.p).toFixed(1); }).join(' ') + ' L' + xs[xs.length - 1].toFixed(1) + ' ' + (y0 + ch) + 'Z"/>' +
        '<polyline class="rd-line" points="' + pts.map(function (q, i) { return xs[i].toFixed(1) + ',' + y(q.p).toFixed(1); }).join(' ') + '"/>' +
        pts.map(function (q, i) { return '<circle class="rd-pt ' + (q.c === 2 ? 'is-c2' : 'is-c1') + '" cx="' + xs[i].toFixed(1) + '" cy="' + y(q.p).toFixed(1) + '" r="4.5"/>' +
          (pts.length <= 12 || i % 2 === 0 ? '<text class="rd-ax" x="' + xs[i].toFixed(1) + '" y="' + (H - 6) + '" text-anchor="middle">' + MON[+q.m.slice(5) - 1] + '</text>' : ''); }).join('') + '</svg>';
    }
    var charts = (cols ? '<figure><figcaption>Before and after the change</figcaption>' + cols + '</figure>' : '') +
      (run ? '<figure><figcaption>Month by month <span class="muted">(dotted: median; dashed: target)</span></figcaption>' + run + '</figure>' : '');
    return hero + (charts ? '<div class="rd-charts">' + charts + '</div>' : '');
  }
  // Trainees can add or remove columns for their own audit. The pass column stays (results need it);
  // column names that look like patient identifiers are refused.
  var COL_TYPES = [['yes/no', 'Yes / No'], ['choice', 'Choice from a list'], ['number', 'Number'], ['date', 'Date'], ['datetime', 'Date and time'], ['text', 'Short text']];
  var ICON_RE = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M20 12a8 8 0 1 1-2.3-5.7"/><path d="M20 4v5h-5"/></svg>';
  function colEditor(r) {
    var t = r.protocol.template || [], pf = passField(r.protocol);
    return '<details class="rec-box col-edit"' + (S.view.colEditOpen ? ' open' : '') + '><summary>Edit columns (' + t.length + ')</summary>' +
      '<p class="muted">Add a column you want to record, or remove one you do not need. Then download the data sheet again: it will have your columns, and uploads will read them.</p>' +
      '<ul class="col-list">' + t.map(function (f) {
        var ty = (COL_TYPES.filter(function (c) { return c[0] === String(f.type).toLowerCase(); })[0] || [0, f.type])[1];
        return '<li><span><b>' + esc(fieldLabel(f.field)) + '</b><small>' + esc(ty) + (f.added ? ' · added by you' : '') + '</small></span>' +
          (f.field === pf ? '<small class="muted">Needed for the results</small>' : '<button type="button" class="link-btn" data-col-del="' + attr(f.field) + '">Remove</button>') + '</li>';
      }).join('') + '</ul>' +
      '<form class="det-form col-add" data-col-add><div class="rec-f"><label for="ca-name">New column</label><input id="ca-name" name="name" maxlength="60" required placeholder="e.g. Seen by senior"></div>' +
      '<div class="rec-f"><label for="ca-type">Type</label><select id="ca-type" name="type">' + COL_TYPES.map(function (c) { return '<option value="' + c[0] + '">' + c[1] + '</option>'; }).join('') + '</select></div>' +
      '<div class="rec-f rec-wide"><label for="ca-opts">Choices, separated by commas <span class="muted">(for "Choice from a list")</span></label><input id="ca-opts" name="options" maxlength="300"></div>' +
      '<div class="rec-actions"><button class="btn btn-secondary" type="submit">Add column</button><span class="form-status" role="status" data-col-status></span></div></form></details>';
  }
  document.addEventListener('click', function (e) {
    var b = e.target.closest && e.target.closest('[data-col-del]'); if (!b) return;
    var r = curRun(); if (!r) return;
    var f = b.getAttribute('data-col-del'), used = ['c1', 'c2'].some(function (k) { return r.cycles[k].rows.some(function (row) { return row[f] != null && row[f] !== ''; }); });
    if (used && b.getAttribute('data-sure') !== '1') { b.setAttribute('data-sure', '1'); b.textContent = 'Remove it and its data?'; return; }
    r.protocol.template = (r.protocol.template || []).filter(function (x) { return x.field !== f; });
    ['c1', 'c2'].forEach(function (k) { r.cycles[k].rows.forEach(function (row) { delete row[f]; }); });
    S.view.colEditOpen = true; runPut(r); renderRunKeep(r, '.col-edit summary');
  });
  document.addEventListener('submit', function (e) {
    var fm = e.target.closest && e.target.closest('[data-col-add]'); if (!fm) return;
    e.preventDefault();
    var r = curRun(); if (!r) return;
    var st = fm.querySelector('[data-col-status]'), name = fm.elements.name.value.trim(), type = fm.elements.type.value;
    var field = name.toLowerCase().replace(/[^a-z0-9]+/g, '_').replace(/^_+|_+$/g, '').slice(0, 40);
    var opts = fm.elements.options.value.split(',').map(function (x) { return x.trim(); }).filter(Boolean).slice(0, 20);
    if (!field) return sayIn(st, 'Give the column a name.');
    if (ID_HEADER.test(field)) return sayIn(st, 'That looks like a patient identifier (name, NHS or hospital number, date of birth, address). Those are never collected.');
    if ((r.protocol.template || []).some(function (x) { return x.field === field; })) return sayIn(st, 'There is already a column with that name.');
    if (type === 'choice' && opts.length < 2) return sayIn(st, 'Add at least two choices, separated by commas.');
    r.protocol.template = (r.protocol.template || []).concat([{ field: field, type: type, options: type === 'choice' ? opts : [], note: '', added: true }]);
    S.view.colEditOpen = true; runPut(r); renderRunKeep(r, '.col-edit summary');
  });
  function rowsTable(r, ck) {
    var p = r.protocol, rows = r.cycles[ck].rows, t = p.template || [];
    if (!rows.length) return '<p class="muted">No records yet.</p>';
    var shown = rows.slice(-200);
    // Shown like the Excel sheet it came from: column letters, row numbers and the green header row.
    return '<div class="sheet-wrap" role="region" tabindex="0" aria-label="Your records"><table class="sheet rows-sheet">' +
      '<thead><tr><th class="sh-corner" aria-hidden="true"></th>' + t.map(function (f, i) { return '<th class="sh-col" aria-hidden="true">' + colName(i) + '</th>'; }).join('') + '<th class="sh-col" aria-hidden="true"></th></tr></thead><tbody>' +
      '<tr><th class="sh-row" aria-hidden="true">1</th>' + t.map(function (f) { return '<th scope="col" class="sh-head">' + esc(fieldLabel(f.field)) + '</th>'; }).join('') + '<th class="sh-head"><span class="sr-only">Remove</span></th></tr>' +
      shown.map(function (row, j) {
        var idx = rows.length - shown.length + j;
        return '<tr><th class="sh-row">' + (idx + 2) + '</th>' + t.map(function (f) { return '<td>' + esc(row[f.field] == null ? '' : row[f.field]) + '</td>'; }).join('') +
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

      (DEMO ? '<p><button type="button" class="btn demo-btn" data-demo-fill="' + ck + '">Demo: fill ' + (ck === 'c1' ? 'cycle 1' : 'the re-audit') + ' with ' + want + ' example records</button></p>' : '') +
      '<div class="rec-box upload-box"><h3>Upload your data sheet</h3>' +
      '<p class="muted">The Excel sheet you downloaded for this audit. Rows marked Re-audit in the Cycle column go to the re-audit. Only the sheet\'s own columns are read: names, NHS numbers, dates of birth and addresses are never imported, hospital numbers become audit codes, and numbers or names typed in free text are removed.</p>' +
      '<label class="file-pick"><input type="file" accept=".xlsx,.xls,.csv,.tsv" data-import="' + ck + '"><span class="btn">Choose your Excel sheet</span></label> ' +
      '<button class="btn btn-secondary" type="button" data-run-xlsx="' + ck + '">Download the ' + (ck === 'c2' ? 're-audit' : 'cycle 1') + ' sheet</button>' +
      '<div data-import-preview="' + ck + '"></div></div>' +
      rowsTable(r, ck) + '</div>';
  }
  // Keep in step with DECK_THEMES in export.js.
  var DECK_THEME_NAMES = ['Classic navy', 'NHS blue', 'Forest', 'Plum and coral', 'Slate and coral', 'Teal', 'Burgundy', 'Charcoal and lime', 'Ocean',
    'Graphite and amber', 'Emerald', 'Indigo', 'Terracotta', 'Midnight and rose', 'Pine and sand', 'Royal', 'Steel and cyan', 'Aubergine and mint', 'Oxford', 'Sage'];
  function deckThemeOptions(r) {
    var cur = r.deckTheme === undefined || r.deckTheme === null || r.deckTheme === '' ? '' : String(r.deckTheme);
    return '<option value=""' + (cur === '' ? ' selected' : '') + '>Design: automatic</option>' + DECK_THEME_NAMES.map(function (n, i) {
      return '<option value="' + i + '"' + (cur === String(i) ? ' selected' : '') + '>Design ' + (i + 1) + ': ' + n + '</option>';
    }).join('');
  }
  document.addEventListener('change', function (e) {
    var sel = e.target.closest && e.target.closest('[data-deck-theme]'); if (!sel) return;
    var r = curRun(); if (!r) return;
    r.deckTheme = sel.value === '' ? null : +sel.value; runPut(r);
  });
  function renderRun(id) {
    if (vaultGate()) return;
    var r = S.runs.get(id);
    if (!r) return renderNotFound();
    if (fillStandard(r.protocol)) runPut(r);                   // older saves with a source but no wording
    var p = r.protocol, d = r.details, st = runStats(r), ns = nextStep(r), i = stageIdx(r), ck = S.view.runTab || (i >= 4 ? 'c2' : 'c1');
    if (r.demo) ns.due = null;                                   // example audits are backdated: no "was due" warnings on stage
    var overdue = ns.due && ns.due < todayIso() && !r.closed;
    var detailsForm = '<form class="det-form" data-run-details>' +
      '<div class="rec-f rec-wide"><label for="rd-question">Audit question <span class="muted">(reword it for your service if you need to; this changes only your copy)</span></label>' +
      '<textarea id="rd-question" name="question" rows="2" maxlength="400">' + esc(p.question) + '</textarea></div>' +
      [['title', 'Audit title', 'text'], ['site', 'Hospital or practice', 'text'], ['department', 'Department or ward', 'text'], ['lead', 'Audit lead', 'text'],
        ['team', 'Team members', 'text'], ['supervisor', 'Supervising consultant', 'text'], ['startDate', 'Start date', 'date'], ['sampleSize', 'Records per cycle', 'number']]
        .map(function (x) {
          return '<div class="rec-f"><label for="rd-' + x[0] + '">' + x[1] + '</label><input id="rd-' + x[0] + '" name="' + x[0] + '" type="' + x[2] + '"' +
            (x[2] === 'number' ? ' min="1" max="2000" data-sample-input' : ' maxlength="200"') + ' value="' + attr(d[x[0]] == null ? '' : d[x[0]]) + '">' +
            (x[0] === 'sampleSize' ? '<small class="prec-note" data-prec>' + precisionText(+d.sampleSize || 30) + '</small><small class="prec-note">' + powerText(p, st.cycles[0].pct) + '</small>' : '') + '</div>';
        }).join('') +
      '<fieldset class="rec-wide opt-set"><legend>Extra data protection</legend>' +
      '<label class="check"><input type="checkbox" name="monthOnly"' + ((r.options || {}).monthOnly ? ' checked' : '') + '><span>Store dates as month and year only (existing dates are shortened too)</span></label>' +
      '<label class="check"><input type="checkbox" name="noFreeText"' + ((r.options || {}).noFreeText ? ' checked' : '') + '><span>Switch off free-text fields (existing free text is deleted)</span></label></fieldset>' +
      '<div class="rec-actions"><button class="btn" type="submit">Save details</button><span class="form-status" role="status" data-det-status></span></div></form>';
    var remind = BE.url ? '<p class="muted remind-note">' + (BE.user ? (remindersOn() ?
      'Email reminders are on: once you start an audit, we email you when a step is due (only the audit question, the step and its date; never your data). <a href="#/account">Turn them off</a>' :
      'Email reminders are off. <a href="#/account">Turn them on</a>') :
      '<a href="#/account">Sign in</a> to get an email when a step is due.') + '</p>' : '';
    var chg = r.changeMade.description || '', alt = p.change_alt || '';
    var pick = !chg ? '' : chg === noDot(p.change) + '.' || chg === p.change ? '1' : alt && (chg === alt || chg === noDot(alt) + '.') ? '2' : 'own';
    function opt(v, title, text) {
      return '<label class="chg-opt"><input type="radio" name="pick" value="' + v + '"' + (pick === v ? ' checked' : '') + (text ? ' data-text="' + attr(noDot(text) + '.') + '"' : '') + '>' +
        '<span><b>' + title + '</b>' + (text ? '<small>' + esc(noDot(text)) + '.</small>' : '<small>Write the change that fits your ward.</small>') + '</span></label>';
    }
    var changeForm = '<form class="det-form" data-run-change><fieldset class="chg-opts"><legend>Choose the change</legend>' +
      opt('1', 'Suggested change 1', p.change) + (alt ? opt('2', 'Suggested change 2', alt) : '') + opt('own', 'My own change', '') + '</fieldset>' +
      '<div class="rec-f rec-wide"><label for="rc-desc">What did you change? <span class="muted">You can edit this</span></label><textarea id="rc-desc" name="description" rows="3" maxlength="600">' + esc(chg) + '</textarea></div>' +
      '<div class="rec-f"><label for="rc-date">Date it started</label><input id="rc-date" name="date" type="date" value="' + attr(r.changeMade.date || '') + '"></div>' +
      '<div class="rec-actions"><button class="btn" type="submit">Save change</button><span class="form-status" role="status" data-chg-status></span></div></form>';
    var results = resultsDash(r, st) +
      st.breakdowns.map(function (b) {
        var all = b.cycles.c1.concat(b.cycles.c2), max = Math.max.apply(null, all.map(function (x) { return x.n; }).concat([1]));
        var two = b.cycles.c1.length && b.cycles.c2.length, opts = [];
        b.cycles.c1.concat(b.cycles.c2).forEach(function (x) { if (opts.indexOf(x.option) < 0) opts.push(x.option); });
        function n(k, o) { var h = b.cycles[k].filter(function (x) { return x.option === o; })[0]; return h ? h.n : 0; }
        function bar(k, o, cls) { return '<i class="' + cls + '" style="width:' + (n(k, o) / max * 100) + '%"></i>'; }
        return '<div class="sub"><h3>' + esc(b.label) + '</h3>' + (two ? '<p class="bd-key"><span class="k-before">Cycle 1</span><span class="k-after">Re-audit</span></p>' : '') +
          '<ul class="bd-bars' + (two ? ' is-two' : '') + '">' + opts.map(function (o, oi) {
            return two ? '<li><span>' + esc(o) + '</span><div class="bd-pair">' + bar('c1', o, 'before') + bar('c2', o, 'after') + '</div><b>' + n('c1', o) + ' → ' + n('c2', o) + '</b></li>' :
              '<li><span><em class="sw s' + Math.min(oi + 1, 8) + '" aria-hidden="true"></em>' + esc(o) + '</span><i class="s' + Math.min(oi + 1, 8) + '" style="width:' + (n(b.cycles.c1.length ? 'c1' : 'c2', o) / max * 100) + '%"></i><b>' + n(b.cycles.c1.length ? 'c1' : 'c2', o) + '</b></li>';
          }).join('') + '</ul></div>';
      }).join('');
    var dl = '<div class="out-grid">' +
      '<button class="out-btn" type="button" data-run-xlsx><b>Data sheet</b><span>Excel, with drop-downs</span></button>' +
      '<div class="out-btn deck-pick"><button class="link-btn" type="button" data-run-pptx><b>Results presentation</b><span>PowerPoint, with charts</span></button>' +
        '<label><span class="sr-only">Design</span><select data-deck-theme>' + deckThemeOptions(r) + '</select></label></div>' +

      '<button class="out-btn" type="button" data-run-backup><b>Backup</b><span>To move this audit to another device</span></button>' +
      (window.AI4QI_EMBED ? '' : '<button class="out-btn" type="button" data-run-ics><b>Calendar</b><span>Add the deadlines</span></button>') +
      '</div><p class="export-warn">These files hold de-identified patient records. Keep them on your organisation\'s systems and share them only inside it.</p><p class="form-status" role="status" data-out-status></p>';
    var started = isStarted(r) || r.demo;
    var stageBtn = !started ? (DEMO ? '<button class="btn demo-btn" type="button" data-demo-step>' + esc(demoStepLabel(r)) + '</button>' : '') + '<button class="btn" type="button" data-run-commit>Start this audit</button>' :
      r.closed ? '<button class="btn btn-secondary" type="button" data-run-stage="reopen">Reopen</button>' :
      (DEMO ? '<button class="btn demo-btn" type="button" data-demo-step>' + esc(demoStepLabel(r)) + '</button>' : '') +
      '<button class="btn" type="button" data-run-stage="next">' + (i === RUN_STAGES.length - 1 ? 'Mark the loop closed' : 'Done – go to ' + esc(RUN_STAGES[i + 1][1].toLowerCase())) + '</button>' +
      (i > 0 ? '<button class="btn btn-secondary" type="button" data-run-stage="back">Back a stage</button>' : '');

    page('<nav class="breadcrumb" aria-label="Breadcrumb"><a href="#/">Home</a> › <a href="#/my-audits">My audits</a></nav>' +
      '<article class="doc run" data-run="' + attr(r.id) + '"><header class="doc-head"><div class="eyebrow"><span class="id-tag">' + esc(r.auditId) + '</span>' +
      (r.closed ? badge('Loop closed', 'ok') : badge(RUN_STAGES[i][1], 'primary')) + (r.demo ? badge('Example data', 'demo') : r.exampleData ? badge('Includes example data', 'demo') : '') + '<a href="' + auditHref(r.auditId) + '">View protocol</a></div>' +
      '<h1 class="audit-name">' + esc(d.title && d.title !== p.question ? d.title : auditName(p)) + '</h1><p class="audit-q">' + esc(p.question) + '</p>' + stageStepper(r) +
      (started ? '<div class="next-card' + (overdue ? ' is-late' : '') + '"><div><p class="nc-label">' + (r.closed ? 'Done' : 'Next step') + (ns.due ? ' · ' + (overdue ? 'was due ' : 'due ') + esc(dateGBs(ns.due)) : '') + '</p>' +
      '<p class="nc-text">' + esc(ns.text) + '</p></div>' :
      '<div class="next-card is-trial"><div><p class="nc-label">Not started yet</p>' +
      '<p class="nc-text">Look around, download the data sheet or send it to your supervisor. When you are ready to run it, press <strong>Start this audit</strong>: the start date becomes today' +
      (BE.url ? ' and we email you when each step is due' : '') + '. Nothing is sent before then.</p></div>') + '<div class="nc-actions">' + stageBtn + '</div></div>' +
      '<p class="privacy-note"><svg width="16" height="16" viewBox="0 0 24 24" aria-hidden="true"><path d="M12 3l7 3v5c0 4.5-3 8.3-7 10-4-1.7-7-5.5-7-10V6l7-3z" fill="none" stroke="currentColor" stroke-width="2" stroke-linejoin="round"/></svg>' +
      'Stored encrypted on this device only; identifiers are removed as records are entered. ' + privacyLink() +
      ' <button type="button" class="link-btn" data-vault-lock>Lock</button></p></header>' +
      sec(1, 'Audit details', detailsForm + remind + supervisorBox(p, r)) +
      // In the order of the loop: cycle 1, the change, the re-audit, then what it all achieved.
      // Cycle 1, then the re-audit (closed until clicked: the change and the re-audit sheet), then the results.
      sec(2, 'Cycle 1', colEditor(r) + cyclePanel(r, 'c1', st)) +
      '<details class="reaudit-part" data-reaudit-part' + (S.view.reauditOpen === r.id ? ' open' : '') + '><summary>' +
        '<span class="ra-i">' + ICON_RE + '</span><span class="ra-t"><b>Re-audit: close the loop</b><span>Record the change you made, then collect the same data again with the re-audit sheet.</span></span>' +
        '<span class="ra-go">Open</span></summary>' +
        sec(3, 'The change', '<p class="step-why">After cycle 1, you and your team change one thing to fix what the results showed: a form, a checklist, an alert or a new pathway. ' +
          'Write down what you changed and the day it started. It goes on your slides, marks the date on the month-by-month chart, and shows the loop was closed.</p>' + changeForm) +
        sec(4, 'Re-audit data', '<p class="step-why">Once the change has bedded in, collect the same data again with the re-audit sheet. It has its own code, so it can only go into this audit, as the re-audit. The results below then show before and after.</p>' + cyclePanel(r, 'c2', st)) +
      '</details>' +
      sec(5, 'Results', results + (r.demo ? '' : feedbackBox(r.auditId))) +
      sec(6, 'Files for you', dl) +
      '<section class="danger-zone"><button type="button" class="link-btn" data-run-delete>Delete this audit and its data from this device</button><span data-del-confirm></span></section>' +
      '</article>', 'My audit', 'my-audits');
  }

  document.addEventListener('toggle', function (e) {
    var d = e.target; if (!d.matches || !d.matches('[data-reaudit-part]')) return;
    var run = d.closest('[data-run]'); S.view.reauditOpen = d.open && run ? run.getAttribute('data-run') : null;
  }, true);

  document.addEventListener('change', function (e) {
    var o = e.target.closest && e.target.closest('.chg-opt input[name="pick"]'); if (!o) return;
    var ta = o.form && o.form.elements.description; if (!ta) return;
    if (o.value === 'own') { if (o.form.querySelector('[data-text]') && Array.prototype.some.call(o.form.querySelectorAll('[data-text]'), function (x) { return x.getAttribute('data-text') === ta.value; })) ta.value = ''; ta.focus(); }
    else ta.value = o.getAttribute('data-text') || '';
  });

  document.addEventListener('click', function (e) {
    if (!e.target.closest || !e.target.closest('[data-ideas-copy]')) return;
    var st = main.querySelector('[data-ideas-status]'), t = S.ideasText || '';
    function fallback() { var ta = document.createElement('textarea'); ta.value = t; ta.rows = 8; ta.className = 'ideas-copy'; e.target.closest('p').appendChild(ta); ta.select(); if (st) st.textContent = 'Select all and copy the text below.'; }
    try { navigator.clipboard.writeText(t).then(function () { if (st) st.textContent = 'Copied. Paste it into your Claude conversation.'; }, fallback); } catch (err) { fallback(); }
  });

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

  function renderHow() {
    var nLib = fmt(S.lib.length), nProp = fmt(S.proposed.length), nStd = fmt(S.standards.length);
    function step(n, t, body) { return '<li class="how-step"><span class="how-n" aria-hidden="true">' + n + '</span><div><h2>' + t + '</h2>' + body + '</div></li>'; }
    page('<article class="doc narrow how"><h1>How Ai4Qi works</h1>' +
      '<p class="page-intro">Ai4Qi takes you from an audit idea to a closed loop: a clear question, the right standard, a data sheet, results and the re-audit. Every audit is built on real evidence, not on guesswork.</p>' +
      '<ol class="how-steps">' +
      step(1, 'A library of real audits', '<p class="prose">' + nLib + ' published clinical audits and quality improvement projects, each read and summarised: the setting, the standard, the result before, the change that was made and the result after. Only real, published projects are in it.</p>') +
      step(2, 'Checked standards', '<p class="prose">' + nStd + ' national standards (NICE, the Royal Colleges, NHS England, national audits and others), with their exact wording and a link to the source.</p>') +
      step(3, 'Ready-made audits', '<p class="prose">' + nProp + ' audits designed by us from current standards and from the gaps in the library: topics that matter but few people have audited. Each has one question, a pass definition, a sample, a timeline and a data sheet.</p>') +
      step(4, 'Build your own audit', '<p class="prose">Type any topic and Ai4Qi writes a new audit for it in under a minute. The AI (Claude, made by Anthropic) is given material from the library and must follow these rules:</p>' +
        '<ul class="prose how-rules">' +
        '<li><strong>New, not copied.</strong> Every audit is written fresh for your topic: its own question, standard, data sheet, pitfalls and pearls. The library supplies the facts; it is not a template.</li>' +
        '<li><strong>From the library first.</strong> It reads the closest published audits on your topic (up to 30, closed-loop and detailed ones first), the best-matching standards and our closest ready-made audits.</li>' +
        '<li><strong>One plain question.</strong> Who, against what standard, and what counts as a pass. One audit answers one question.</li>' +
        '<li><strong>The real standard.</strong> A standard from our checked list, quoted exactly and linked. Where no national standard exists, it says "Local standard" and never pretends otherwise. NICE wording is added by the site itself, straight from NICE, not written by the AI.</li>' +
        '<li><strong>Evidence you can check.</strong> It may only cite audits from the library, with their numbers exactly as published. Any citation that is not in the library is removed before you see it.</li>' +
        '<li><strong>A change that lasts.</strong> The change it suggests is a form, checklist, default or system change, because teaching alone rarely works.</li>' +
        '<li><strong>Honest suggestions.</strong> If one of our ready-made audits would suit you better, it tells you.</li>' +
        '<li><strong>It learns from you.</strong> If you press <em>Not quite right</em>, your reasons go into the next version.</li></ul>' +
        '<p class="prose">A topic nobody has published on still gets a full audit, built from the national standard; it simply has no evidence section.</p>') +
      step(5, 'Run it and close the loop', '<p class="prose">Sign in first: an institutional email (NHS, HSE or university) gets instant access; other emails go through a quick approval. Then press <em>Choose this audit</em>, then <em>Start this audit</em> when you are ready. Download the data sheet, send the proposal to your supervisor, record your data, see the results, make the change and re-audit. Once you start, we email you when each step is due.</p>') +
      '</ol>' +
      '<h2>What stays with you</h2><p class="prose">Your audit records stay encrypted on your own device. They are never sent to Ai4Qi or to the AI. <a href="#/privacy">How your data is protected</a></p>' +
      '<h2>Always check</h2><p class="prose">An audit built by AI is a well-founded draft. Check the standard against its linked source and ask your supervisor to review the protocol before you collect data. Register the audit with your audit department.</p>' +
      '</article>', 'How Ai4Qi works', '');
  }
  function renderPrivacy() {
    page('<article class="doc narrow"><h1>How your audit data is protected</h1>' +
      '<p class="prose"><strong>Your audit records stay on your device.</strong> Records you type or upload in My audits are kept in this browser only. They are never sent to Ai4Qi, and we cannot see them.</p>' +
      '<p class="prose"><strong>Stored encrypted.</strong> Records are stored encrypted (AES-256) in this browser and open without a passcode, like your other work on this device: Trust computers lock themselves, and a personal device is yours to keep locked. On a shared computer, choose shared-computer mode under <em>Backup and protection</em> on My audits, so everything is deleted when the browser closes.</p>' +
      '<p class="prose"><strong>De-identified as they are entered.</strong> Before a record is stored, Ai4Qi:</p><ul class="prose">' +
      '<li>keeps only the columns that are part of the audit template and leaves out everything else (for example name, NHS number, date of birth or address columns);</li>' +
      '<li>replaces patient or hospital numbers with audit codes (P001, P002…);</li>' +
      '<li>removes NHS numbers and other long numbers, postcodes, phone numbers, email addresses, dates of birth and names written with a title (Mr, Mrs, Dr…) from any text;</li>' +
      '<li>can store dates as month and year only, and can switch free-text fields off, for each audit.</li></ul>' +
      '<p class="prose"><strong>This is de-identified, not anonymous, data.</strong> Dates and details together can sometimes identify a person, so treat your records as patient data under your organisation\'s rules. Automatic checks cannot catch every identifier written in free text: never type names or numbers. Register the audit with your audit department before you start.</p>' +
      '<p class="prose"><strong>Files you download</strong> (data sheet, presentation, records, backup) are made on your device. Records and backups contain de-identified patient data: keep them on your organisation\'s systems.</p>' +
      '<p class="prose"><strong>Reminders.</strong> If you sign in, email reminders are on for audits you have started (pressed <em>Start this audit</em>) unless you turn them off in Account. Only the audit question, the next step, its due date and record counts are sent to our server, never records.</p>' +
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
    if (e.target.closest('[data-run-commit]')) {
      r.committed = true; r.committedAt = new Date().toISOString();
      if (!r.details.startDate || r.details.startDate < todayIso()) r.details.startDate = todayIso();
      runPut(r); renderRunKeep(r, '[data-run-stage="next"]');
      return;
    }
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
      var xb = e.target.closest('[data-run-xlsx]'), cyc = xb.getAttribute('data-run-xlsx') || (stageIdx(r) >= 4 ? 'c2' : 'c1');
      exporter().then(function (x) { return x.templateXlsx(r.protocol, { code: r.auditId + '/' + r.id + '/' + (cyc === 'c2' ? 'RE' : 'C1') }); })
        .then(function (b) { return saveFile(name + (cyc === 'c2' ? '-re-audit' : '-cycle-1') + '-data-sheet.xlsx', b); })
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
        var file = vaultDevice() ?
          { ai4qi_backup: 3, note: 'Ai4Qi audit backup (no passcode). Open it in Ai4Qi > My audits > Restore a backup. It holds de-identified audit records: keep it on your organisation\'s systems.', key: V.meta.device, box: box } :
          { ai4qi_backup: 2, note: 'Encrypted Ai4Qi audit backup. Open it in Ai4Qi > My audits > Restore a backup, with the passcode used when it was made.', salt: V.meta.salt, iter: V.meta.iter || PBKDF2_ITER, box: box };
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
      var oldQ = r.protocol.question;
      fd.forEach(function (v, k) {
        if (k === 'monthOnly' || k === 'noFreeText') return;
        if (k === 'question') { var qv = String(v).trim().slice(0, 400); if (qv) r.protocol.question = qv; return; }
        r.details[k] = k === 'sampleSize' ? Math.max(1, Math.min(2000, +v || 30)) : String(v).slice(0, 200);
      });
      if (r.details.title === oldQ) r.details.title = r.protocol.question;     // an untouched title follows the reworded question
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
  });
  // Columns added in Excel: offer to add them to this audit in one click (never identifier-like ones),
  // with the type worked out from what was typed in them.
  function guessColumn(table, head) {
    var j = -1, i0 = -1;
    for (var i = 0; i < Math.min(table.length, 12) && j < 0; i++) { j = (table[i] || []).indexOf(head); i0 = i; }
    var vals = table.slice(i0 + 1).map(function (r) { return r[j]; }).filter(function (v) { return v !== null && v !== undefined && String(v).trim() !== ''; });
    var sv = vals.map(function (v) { return v instanceof Date ? v.toISOString() : String(v).trim(); });
    var col = { field: normHead(head), type: 'text', options: [], note: '', added: true };
    if (!sv.length) return col;
    if (sv.every(function (v) { return /^(y|yes|n|no|true|false|n\/?a)$/i.test(v); })) col.type = 'yes/no';
    else if (sv.every(function (v) { return /^-?\d+(\.\d+)?$/.test(v); })) col.type = 'number';
    else if (sv.every(function (v) { return /^\d{4}-\d{2}-\d{2}|^\d{1,2}\/\d{1,2}\/\d{2,4}/.test(v); })) col.type = sv.some(function (v) { return /\d{1,2}:\d{2}/.test(v) && !/T00:00:00/.test(v); }) ? 'datetime' : 'date';
    else {
      var uniq = sv.filter(function (v, k) { return sv.indexOf(v) === k; });
      if (uniq.length <= 8 && sv.length >= uniq.length * 2) { col.type = 'choice'; col.options = uniq; }
    }
    return col;
  }
  function sheetCode(table) {
    for (var i = 0; i < Math.min(table.length, 6); i++) {
      var m = (table[i] || []).join(' ').match(/Sheet code:\s*([A-Za-z0-9-]+)\/(r-[a-z0-9]+)?\/(C1|RE)\b/);
      if (m) return { audit: m[1], run: m[2] || '', round: m[3] };
    }
    return null;
  }
  function showImportPreview(r, ck, table, box) {
    var code = sheetCode(table);
    if (code && (code.audit !== r.auditId || (code.run && code.run !== r.id))) {
      box.innerHTML = '<div class="import-preview notice notice-warn"><p><strong>This sheet belongs to ' + (code.audit !== r.auditId ? 'a different audit (' + esc(code.audit) + ')' : 'another copy of this audit') + '.</strong> Nothing was added. Upload it in that audit, or download this audit\'s sheet.</p>' +
        '<button class="link-btn" type="button" data-import-cancel>Close</button></div>';
      S.pendingImport = null; return;
    }
    if (code && code.round === 'RE') ck = 'c2';
    var prep = prepareImport(r.protocol, table, r.codeSeq || 1, r.options);
    if (prep.reRows.length) { S.pendingImport = { runId: r.id, ck: 'c1', rows: prep.rows, reRows: prep.reRows, nextCode: prep.nextCode, table: table, tab: ck }; }
    else S.pendingImport = { runId: r.id, ck: ck, rows: prep.rows, nextCode: prep.nextCode, table: table, tab: ck };
    var fresh = prep.dropped.filter(function (h) { var n = normHead(h); return n && n !== 'cycle' && !ID_HEADER.test(n); });
    var ids = prep.dropped.filter(function (h) { return ID_HEADER.test(normHead(h)); });
    var total = prep.rows.length + prep.reRows.length;
    box.innerHTML = '<div class="import-preview"><p><strong>' + total + ' records ready to add' + (prep.reRows.length ? ' (' + prep.rows.length + ' cycle 1, ' + prep.reRows.length + ' re-audit)' : code ? ' to the ' + (ck === 'c2' ? 're-audit' : 'cycle 1') : '') + '.</strong> Nothing has been stored yet.</p>' +
      (code ? '<p class="muted">Sheet code ' + esc(code.audit + (code.run ? '/' + code.run : '') + '/' + code.round) + ': ' + (code.round === 'RE' ? 'the re-audit' : 'cycle 1') + ' of this audit.</p>' : '') +
      (fresh.length ? '<div class="new-cols"><p><strong>New columns in your sheet:</strong> ' + esc(fresh.join(', ')) + '</p>' +
        '<button class="btn btn-secondary" type="button" data-import-addcols>Add ' + (fresh.length === 1 ? 'it' : 'them') + ' to this audit</button> <span class="muted">They will then count in your results.</span></div>' : '') +
      '<ul><li>Columns used: ' + (prep.kept.length ? prep.kept.map(function (k) { return esc(k[0]); }).join(', ') : 'none matched the template') + '</li>' +
      (ids.length ? '<li>Identifier columns ignored: ' + esc(ids.join(', ')) + ' <span class="muted">(never imported)</span></li>' : '') +
      '<li>Patient numbers replaced with audit codes: ' + prep.rep.coded + '</li>' +
      '<li>Identifiers removed from text: ' + prep.rep.redacted + '</li>' +
      (prep.rep.badDates ? '<li>Dates that could not be read (left blank): ' + prep.rep.badDates + '</li>' : '') + '</ul>' +
      (total ? '<button class="btn" type="button" data-import-ok>Add ' + total + ' records</button> ' : '') +
      '<button class="link-btn" type="button" data-import-cancel>Cancel</button></div>';
  }
  document.addEventListener('click', function (e) {
    if (!e.target.closest('[data-import-addcols]')) return;
    var pi = S.pendingImport, r = pi && S.runs.get(pi.runId); if (!r || !pi.table) return;
    var box = e.target.closest('[data-import-preview]');
    var prep = prepareImport(r.protocol, pi.table, r.codeSeq || 1, r.options);
    var have = (r.protocol.template || []).map(function (f) { return f.field; });
    prep.dropped.forEach(function (h) {
      var n = normHead(h);
      if (!n || n === 'cycle' || ID_HEADER.test(n) || have.indexOf(n) >= 0) return;
      r.protocol.template = (r.protocol.template || []).concat([guessColumn(pi.table, h)]); have.push(n);
    });
    runPut(r);
    showImportPreview(r, pi.tab || pi.ck, pi.table, box);
  });
  document.addEventListener('change', function (e) {
    var r = curRun(); if (!r) return;
    var inp = e.target.closest('[data-import]');
    if (!inp || !inp.files || !inp.files[0]) return;
    var ck = inp.getAttribute('data-import'), box = main.querySelector('[data-import-preview="' + ck + '"]'), file = inp.files[0];
    box.innerHTML = '<p class="muted">Reading ' + esc(file.name) + '…</p>';
    readTable(file).then(function (table) { showImportPreview(r, ck, table, box);
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
      if (o && o.ai4qi_backup === 3 && o.key && o.box) {                 // made without a passcode
        return crypto.subtle.importKey('raw', unb64(o.key), { name: 'AES-GCM' }, false, ['decrypt'])
          .then(function (k) { return unseal(k, o.box); }).then(function (json) { restoreRun(JSON.parse(json)); });
      }
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
      '<div class="fb-row"><div class="fb-q"><b>Is this a good audit to run?</b><span>Your vote improves it for everyone: audits with more thumbs down get rewritten.</span></div>' +
      '<div class="fb-btns"><button type="button" class="fb-btn' + (prev && prev.rating === 'up' ? ' is-on' : '') + '" data-fb-rate="up" aria-pressed="' + !!(prev && prev.rating === 'up') + '">' + thumb('up') + '<span>Yes, useful</span></button>' +
      '<button type="button" class="fb-btn' + (prev && prev.rating === 'down' ? ' is-on' : '') + '" data-fb-rate="down" aria-pressed="' + !!(prev && prev.rating === 'down') + '">' + thumb('down') + '<span>Not quite right</span></button></div>' +
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
    'Nursing and midwifery', 'Obstetrics and gynaecology', 'Oncology', 'Ophthalmology', 'Trauma and orthopaedics',
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
    // Our own emails link to ai4qi.com with ?token_hash=… (checked here in the page, so an email
    // scanner that opens the link cannot use it up, and the link points to the site it came from).
    var th = q.get('token_hash'), ty = q.get('type');
    if (!err && !q.get('code') && !h.get('access_token') && !th) return null;
    return { error: err || '', code: q.get('code') || '', tokenHash: th || '', type: /^(email|magiclink|signup|invite|email_change)$/.test(ty || '') ? ty : 'email' };
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
          // An emailed link is only used when the person presses "Sign me in": email security scanners
          // (Microsoft Defender on NHSmail) open links but never press buttons, so they cannot use it up.
          if (AUTH_RETURN && AUTH_RETURN.tokenHash && !(res.data && res.data.session)) BE.pendingLink = { h: AUTH_RETURN.tokenHash, t: AUTH_RETURN.type };
          return res;
        }).then(function (res) {
          if (AUTH_RETURN && AUTH_RETURN.error && !(res.data && res.data.session)) BE.authError = AUTH_RETURN.error;
          // A ?code= link only works in the browser it was asked from (e.g. requested in Chrome, opened in
          // Safari, the Mac's default): offer the code from the same email instead of a dead end.
          if (AUTH_RETURN && !AUTH_RETURN.tokenHash && !(res.data && res.data.session)) {
            BE.otherBrowser = true;
            BE.linkWhy = AUTH_RETURN.error ? /expired|invalid/i.test(AUTH_RETURN.error) ? 'expired' : 'error' : AUTH_RETURN.code ? 'browser' : 'unknown';
          }
          if (AUTH_RETURN) cleanAuthUrl();
          BE.client = c;
          BE.user = res.data && res.data.session ? res.data.session.user : null;
          updateAccountLink();
          if (BE.user) { setTimeout(flushFeedback, 0); recordActivity(); loadProfile(c).then(updateAccountLink, function () {}); }
          if (BE.user && AUTH_RETURN && !AUTH_RETURN.error && parseHash().parts[0] === 'account') { S.justSignedIn = true; location.hash = signedInHome(); }
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
    BE.admin = null; BE.profile = undefined; S.accessOk = false;
    BE.tracker = null;
    if (u) { if (u.email) setLastEmail(u.email); flushFeedback(); recordActivity(); if (BE.client) loadProfile(BE.client).then(updateAccountLink, function () {}); }
    // A non-institutional email that has not been approved goes straight to the approval page.
    if (u && !isInstitutional(u.email) && !DEMO) setTimeout(function () {
      accessStatus().then(function (st) {
        if (st === 'none' && BE.client) return BE.client.from('access_requests').insert({ user_id: u.id }).then(function () { return 'pending'; }, function () { return 'pending'; });
        return st;
      }).then(function (st) { if (st !== 'ok' && BE.user && BE.user.id === u.id && parseHash().parts[0] !== 'access') { S.accessNext = signedInHome(); location.hash = '#/access'; } });
    }, 0);
    var name = parseHash().parts[0];
    if (S.lib.length && (name === 'account' || name === 'admin' || name === 'my-audits')) route();
    else if (name === 'proposed' && parseHash().parts[1]) showTracker(parseHash().parts[1]);
  }
  function initialsOf(email) {
    var name = String((BE.profile && BE.profile.full_name) || '').replace(/\b(dr|mr|mrs|ms|miss|mx|prof|professor|sir|dame)\.?\s+/gi, '').trim().split(/\s+/).filter(Boolean);
    if (name.length > 1) return (name[0][0] + name[name.length - 1][0]).toUpperCase();   // "Parag Garg" -> PG
    if (name.length === 1) {                                 // "Parag" + paraggarg@... -> PG
      var loc = String(email || '').split('@')[0].toLowerCase().replace(/[^a-z]/g, ''), first = name[0].toLowerCase();
      var rest = loc.indexOf(first) === 0 ? loc.slice(first.length) : '';
      return (name[0][0] + (rest ? rest[0] : '')).toUpperCase();
    }
    var local = String(email || '').split('@')[0].replace(/[0-9]+/g, '');
    var parts = local.split(/[._\-+]+/).filter(Boolean);
    var s = parts.length >= 2 ? parts[0][0] + parts[parts.length - 1][0] : local.slice(0, 2);
    return s.toUpperCase();
  }
  function updateAccountLink() {
    var signedIn = BE.user || (!BE.client && hasStoredSession());
    if (accountLink) accountLink.textContent = signedIn ? 'Account' : 'Sign in';
    var so = document.querySelector('[data-acct-signout]'); if (so) so.hidden = !signedIn;
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
    menu.addEventListener('click', function (e) { if (e.target.closest('a, button')) set(false); });
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
  function loadVotes() {
    configReady.then(loadSummary).then(function (m) {
      if (!m) return;
      S.votes = m;
      var n = parseHash().parts[0];
      if ((n === 'proposed' && !parseHash().parts[1]) || n === 'suggest') route();
    });
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
      return c.rpc('delete_my_account').then(function (res) { if (res.error) throw res.error; setLastEmail(''); return c.auth.signOut(); });
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
      if (BE.user) renderSignedIn();
      else if (BE.pendingLink && askedHere()) { renderLinkConfirm(); var lb = main.querySelector('[data-link-confirm]'); if (lb) lb.click(); }   // asked on this device: straight in
      else if (BE.pendingLink) renderLinkConfirm();
      else renderSignIn();
      focusMain();
    }, function () { if (stillOn('account')) unavailable('Account', 'account'); });
  }
  function loadProfile(c) {
    if (BE.profile !== undefined) return Promise.resolve(BE.profile);
    return c.from('profiles').select('specialty, grade, region, work_setting, audit_purpose, consent_news, consent_sponsors, full_name, organisation, department, reminders_off').eq('user_id', BE.user.id).maybeSingle()
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
  /* --- sign in: one card, one field, one button; "Welcome back" for someone who has signed in here before --- */
  var LAST_EMAIL = 'ai4qi_last_email';
  function lastEmail() { try { return localStorage.getItem(LAST_EMAIL) || ''; } catch (e) { return ''; } }
  function setLastEmail(v) { try { if (v) localStorage.setItem(LAST_EMAIL, v); else localStorage.removeItem(LAST_EMAIL); } catch (e) {} }
  var ICON_MAIL = '<svg viewBox="0 0 48 48" width="56" height="56" aria-hidden="true"><rect x="5" y="11" width="38" height="27" rx="4" fill="none" stroke="currentColor" stroke-width="2.5"/><path d="M6 13l18 14 18-14" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linejoin="round"/></svg>';
  var ICON_SENT = '<svg viewBox="0 0 48 48" width="64" height="64" aria-hidden="true"><rect x="5" y="11" width="38" height="27" rx="4" fill="none" stroke="currentColor" stroke-width="2.5"/><path d="M6 13l18 14 18-14" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linejoin="round"/><circle cx="38" cy="36" r="8" fill="var(--ok-text)"/><path d="M34.5 36l2.5 2.5 4.5-5" fill="none" stroke="#fff" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/></svg>';
  function emailInitials(e) {
    var loc = String(e || '').split('@')[0].replace(/[0-9]+/g, ''), parts = loc.split(/[._\-+]+/).filter(Boolean);
    return (parts.length >= 2 ? parts[0][0] + parts[parts.length - 1][0] : loc.slice(0, 1)).toUpperCase() || '?';
  }
  // A button straight to the person's inbox, when we can tell which webmail they use.
  function inboxLink(email) {
    var d = String(email || '').split('@')[1] || '', m = null;
    if (/^(gmail|googlemail)\.com$/i.test(d)) m = ['https://mail.google.com/mail/u/0/#inbox', 'Open Gmail'];
    else if (/(^|\.)(nhs\.net|nhs\.uk|hse\.ie|outlook\.com|hotmail\.[a-z.]+|live\.[a-z.]+|msn\.com)$/i.test(d)) m = ['https://outlook.office.com/mail/', 'Open Outlook'];
    else if (/^(icloud|me|mac)\.com$/i.test(d)) m = ['https://www.icloud.com/mail', 'Open iCloud Mail'];
    else if (/^yahoo\./i.test(d)) m = ['https://mail.yahoo.com', 'Open Yahoo Mail'];
    return m ? '<a class="btn auth-wide" href="' + m[0] + '" target="_blank" rel="noopener">' + esc(m[1]) + ' &nbsp;↗</a>' : '';
  }
  function renderSignIn(title, fresh) {
    var known = !fresh && lastEmail();
    // A work email's link scanner (Microsoft Defender on NHSmail) can open the link first and use it up,
    // along with the code in that email. A fresh code, typed rather than clicked, always works.
    var err = BE.authError ? '<div class="notice notice-warn" role="alert">That sign-in link has expired or was already used. Send a new one below.</div>' : '';
    if (BE.otherBrowser && known) err = '<div class="notice notice-warn" role="alert">' + ({
        browser: 'That link opened in a different browser from the one you asked in (on a Mac, links often open in Safari).',
        expired: 'That link was already used: email security (e.g. NHSmail) often opens links before you do.',
        error: 'That link did not sign you in.', unknown: 'That link did not sign you in.' })[BE.linkWhy || 'unknown'] +
      ' Type the code from the same email instead:</div>' +
      '<form class="auth-form auth-code" data-otp novalidate><input type="hidden" name="email" value="' + attr(known) + '">' +
      '<input name="code" inputmode="numeric" autocomplete="one-time-code" pattern="[0-9]*" maxlength="10" placeholder="Code from the email" aria-label="Code from the email">' +
      '<button class="btn auth-wide" type="submit">Sign in</button><p class="form-status" data-form-status role="status" aria-live="polite"></p></form>' +
      '<p class="auth-note">No code in the email? Send a new link below and open it in this browser.</p>';
    var body = known ?
      '<div class="auth-avatar" aria-hidden="true">' + esc(emailInitials(known)) + '</div>' +
      '<h1>Welcome back</h1><p class="auth-sub">' + esc(known) + '</p>' + err + (isInstitutional(known) ? '' : '<p class="auth-warn">' + NONINST_NOTE + '</p>') +
      '<form class="auth-form" data-signin novalidate><input type="hidden" name="email" value="' + attr(known) + '">' +
      '<button class="btn auth-wide" type="submit">Email me a sign-in link &nbsp;→</button>' +
      '<p class="form-status" data-form-status role="status" aria-live="polite"></p></form>' +
      '<button type="button" class="link-btn auth-alt" data-signin-other>Not you? Use another email</button>' :
      '<div class="auth-icon">' + ICON_MAIL + '</div>' +
      '<h1>' + esc(title || 'Sign in') + '</h1><p class="auth-sub">No password. We email you a sign-in link.</p>' + err +
      '<form class="auth-form" data-signin novalidate>' +
      '<label for="acc-email" class="sr-only">Email address</label>' +
      '<div class="em-wrap">' +
      '<input id="acc-email" name="email" type="email" inputmode="email" autocomplete="email" spellcheck="false" required maxlength="254" placeholder="e.g. firstname.lastname@nhs.net">' + '<span class="em-ghost" aria-hidden="true" hidden><span data-em-typed></span><span class="em-suf">@nhs.net</span></span></div>' +
      '<p class="auth-warn" data-email-warn hidden></p>' +
      '<button class="btn auth-wide" type="submit">Continue &nbsp;→</button>' +
      '<p class="form-status" data-form-status role="status" aria-live="polite"></p></form>' +
      '<p class="auth-note">New here? The same step creates your free account.</p>';
    page('<div class="auth-wrap"><div class="auth-card">' + body + '</div>' +
      '<p class="auth-foot">Never enter patient information. <a href="#/privacy-notice">Privacy</a></p></div>',
      'Sign in', 'account', true);
  }
  var NONINST_NOTE = 'Personal email: access to build and run audits follows in about 5 minutes. Institutional emails (NHS, HSE or university) get instant access.';
  var NONINST_WARN = '<strong>Use your institutional email (NHS, HSE or university) for instant access.</strong> Personal emails need moderation and have delayed access (about 5 minutes).';
  document.addEventListener('input', function (e) {
    var i = e.target.closest && e.target.closest('#acc-email'); if (!i) return;
    var g = i.parentNode.querySelector('.em-ghost');
    if (g) { g.firstChild.textContent = i.value; g.hidden = !i.value || i.value.indexOf('@') !== -1; }
    var w = main.querySelector('[data-email-warn]'), v = i.value.trim(), done = /^[^\s@]+@[^\s@]+\.[a-z]{2,}$/i.test(v);
    if (!w) return;
    var f = i.form; if (f && f.getAttribute('data-warned') && f.getAttribute('data-warned') !== v.toLowerCase()) {
      f.removeAttribute('data-warned');
      var bt = f.querySelector('button[type="submit"]'); if (bt) bt.innerHTML = 'Continue &nbsp;→';
    }
    // As soon as a full personal address is typed (e.g. ...@gmail.com), say what that means.
    w.hidden = !(done && !isInstitutional(v));
    if (!w.hidden && f) {                      // warned while typing: one press of Continue then goes ahead
      w.innerHTML = NONINST_WARN; f.setAttribute('data-warned', v.toLowerCase());
      var bt2 = f.querySelector('button[type="submit"]'); if (bt2) bt2.innerHTML = 'Continue with this email &nbsp;→';
    }
  });
  // The link was asked for in this browser in the last 2 hours: an email scanner or another
  // device never has this mark, so they still get the button and cannot use the link up.
  function askedHere() {
    try { var t = +localStorage.getItem('ai4qi_link_asked') || 0; return t && Date.now() - t < 7200000; } catch (e) { return false; }
  }
  function renderLinkConfirm() {
    page('<div class="auth-wrap"><div class="auth-card"><div class="auth-icon is-ok">' + ICON_SENT + '</div>' +
      '<h1>Finish signing in</h1><p class="auth-sub">' + (lastEmail() ? esc(lastEmail()) : 'Your sign-in link is ready.') + '</p>' +
      '<button class="btn auth-wide" type="button" data-link-confirm>Sign me in &nbsp;→</button>' +
      '<p class="form-status" role="status" data-form-status></p></div></div>', 'Sign in', 'account', true);
  }
  document.addEventListener('click', function (e) {
    var b = e.target.closest && e.target.closest('[data-link-confirm]'); if (!b || !BE.pendingLink) return;
    var st = main.querySelector('[data-form-status]'), pl = BE.pendingLink;
    b.disabled = true; if (st) st.textContent = 'Signing you in…';
    sbClient().then(function (c) { return c.auth.verifyOtp({ token_hash: pl.h, type: pl.t }); }).then(function (v) {
      BE.pendingLink = null; try { localStorage.removeItem('ai4qi_link_asked'); } catch (e2) {}
      if (v.error) throw v.error;
      S.justSignedIn = true;
      if (v.data && v.data.user) setUser(v.data.user);
      location.hash = signedInHome();
    }).catch(function () {
      BE.pendingLink = null;
      BE.authError = 'This sign-in link has expired or was already used.';
      renderSignIn();
    });
  });
  function renderSent(email) {
    var card = main.querySelector('.auth-card'); if (!card) return;
    card.innerHTML = '<div class="auth-icon is-ok">' + ICON_SENT + '</div>' +
      '<h1 tabindex="-1">Check your email</h1><p class="auth-sub">We sent a sign-in link to<br><strong>' + esc(email) + '</strong>. Open it on this device and press <em>Sign me in</em>.</p>' +
      inboxLink(email) +
      '<form class="auth-form auth-code" data-otp novalidate><input type="hidden" name="email" value="' + attr(email) + '">' +
      '<label for="otp-code">Reading the email on another device? Type the code from it</label>' +
      '<input id="otp-code" name="code" inputmode="numeric" autocomplete="one-time-code" pattern="[0-9]*" maxlength="10" placeholder="••••••">' +
      '<button class="btn btn-secondary auth-wide" type="submit">Sign in with the code</button>' +
      '<p class="form-status" data-form-status role="status" aria-live="polite"></p></form>' +
      (isInstitutional(email) ? '' : '<p class="auth-warn">' + NONINST_NOTE + '</p>') +
      (WORK_EMAIL.test(email) ? '<p class="auth-tip"><strong>Using a work email?</strong> Not in your inbox? Check <em>Junk</em> and the <em>Other</em> tab, and mark it <em>Not junk</em>. Still nothing after 3 minutes? Use a personal email for now.</p>' : '') +
      '<p class="auth-note">Nothing yet? Check junk. <button type="button" class="link-btn" data-signin-resend="' + attr(email) + '" disabled>Send again</button> · ' +
      '<button type="button" class="link-btn" data-signin-other>Use another email</button></p>';
    var h = card.querySelector('h1'); if (h) h.focus();
    var rb = card.querySelector('[data-signin-resend]'), left = 60;
    var t = setInterval(function () {
      left--; if (!rb.isConnected) return clearInterval(t);
      rb.textContent = left > 0 ? 'Send again (' + left + ')' : 'Send again';
      if (left <= 0) { rb.disabled = false; clearInterval(t); }
    }, 1000);
  }
  function sendLink(email) {
    try { localStorage.setItem(AFTER_SIGNIN, S.lastRoute || '#/'); localStorage.setItem('ai4qi_link_asked', String(Date.now())); } catch (e2) {}
    return sbClient().then(function (c) {
      return c.auth.signInWithOtp({ email: email, options: { emailRedirectTo: location.origin + location.pathname } });
    }).then(function (res) { if (res.error) throw res.error; BE.authError = ''; setLastEmail(email); });
  }
  function sendError(err) {
    var st = err && (err.status || err.code);
    return st === 429 || st === 'over_email_send_rate_limit' ? 'Too many requests. Wait a minute and try again.' : 'Could not send. Check the address and your connection.';
  }
  function optionList(list, current) {
    return list.map(function (g) { return '<option' + (g === current ? ' selected' : '') + '>' + esc(g) + '</option>'; }).join('');
  }
  // Straight after signing in: back to the audit they were on, otherwise to My audits (the dashboard).
  function signedInHome() {
    var h = ''; try { h = localStorage.getItem(AFTER_SIGNIN) || ''; localStorage.removeItem(AFTER_SIGNIN); } catch (e) {}
    return /^#\/(build|proposed\/|run\/|suggest|ideas)/.test(h) ? h : '#/my-audits';
  }
  // After the emailed link: offer to carry on straight away; the profile is optional.
  function afterSigninHref() {
    var h = ''; try { h = localStorage.getItem(AFTER_SIGNIN) || ''; } catch (e) {}
    return /^#\/(?!account)/.test(h) ? h : (S.lastRoute && !/^#\/account/.test(S.lastRoute) ? S.lastRoute : '#/');
  }
  function welcomeBack() {
    var pf = BE.profile || {}, empty = !pf.grade && !pf.specialty && !pf.full_name && !pf.organisation;
    if (!AUTH_RETURN && !empty) return '';
    var to = afterSigninHref();
    return '<div class="welcome-card"><p><strong>You\'re signed in.</strong> ' + (empty ? 'The profile below is optional: fill it in now or later, or skip it.' : 'Welcome back.') + '</p>' +
      '<a class="btn" href="' + attr(to) + '" data-skip-profile>' + (to === '#/' ? 'Continue to Ai4Qi' : 'Continue where you were') + '</a></div>';
  }
  var WORK_EMAIL = /@(nhs\.net|([a-z0-9-]+\.)*nhs\.uk|([a-z0-9-]+\.)*nhs\.scot|([a-z0-9-]+\.)*nhs\.wales|([a-z0-9-]+\.)*hscni\.net|([a-z0-9-]+\.)*hse\.ie)$/i;
  function renderSignedIn() {
    var pf = BE.profile || {}, email = BE.user.email || '', verified = WORK_EMAIL.test(email);
    function card(id, title, sub, body) {
      return '<section class="ac-card" aria-labelledby="' + id + '"><header><h2 id="' + id + '">' + title + '</h2>' + (sub ? '<p>' + sub + '</p>' : '') + '</header>' + body + '</section>';
    }
    function field(id, label, input) { return '<div class="ac-f"><label for="' + id + '">' + label + '</label>' + input + '</div>'; }
    function sel(id, name, opts) { return '<select id="' + id + '" name="' + name + '"><option value="">Prefer not to say</option>' + opts + '</select>'; }
    function sw(name, on, title, sub) {
      return '<label class="ac-sw"><span><b>' + title + '</b><small>' + sub + '</small></span><input type="checkbox" role="switch" name="' + name + '"' + (on ? ' checked' : '') + '><i aria-hidden="true"></i></label>';
    }
    page('<div class="ac-page">' +
      '<div class="ac-hero"><div class="auth-avatar" aria-hidden="true">' + esc(initialsOf(email)) + '</div><div class="ac-id">' +
        '<h1>' + esc(pf.full_name || 'Your account') + '</h1><p>' + esc(email) + '</p>' +
        (verified ? '<span class="ac-badge is-ok">✓ Verified NHS / HSE email</span>' : '<span class="ac-badge">Personal email</span>') +
      '</div><a class="btn" href="#/my-audits">My audits →</a></div>' +
      '<form class="ac-form" data-profile novalidate>' +
      card('ac-you', 'About you', 'Filled into the email and Word proposal you send to your supervisor.',
        '<div class="ac-grid">' +
        field('pf-name', 'Full name', '<input id="pf-name" name="full_name" maxlength="120" autocomplete="name" placeholder="e.g. Priya Shah" value="' + attr(pf.full_name || '') + '">') +
        field('pf-org', 'Hospital, Trust or practice', '<input id="pf-org" name="organisation" maxlength="160" autocomplete="organization" list="org-list" placeholder="Start typing to pick" value="' + attr(pf.organisation || '') + '">' + orgList()) +
        '</div>') +
      card('ac-role', 'Your role', 'Optional. Only ever counted in anonymous totals.',
        '<div class="ac-grid">' +
        field('pf-grade', 'Grade or role', sel('pf-grade', 'grade', optionList(GRADES, pf.grade))) +
        field('pf-specialty', 'Specialty', sel('pf-specialty', 'specialty', optionList(SPECIALTIES, pf.specialty))) +
        field('pf-region', 'Region', '<select id="pf-region" name="region"><option value="">Prefer not to say</option>' +
          '<optgroup label="England (NHS region)">' + optionList(REGIONS.slice(0, 7), pf.region) + '</optgroup>' +
          '<optgroup label="Elsewhere">' + optionList(REGIONS.slice(7), pf.region) + '</optgroup></select>') +
        '</div>') +
      card('ac-mail', 'Emails', 'We never share your email address.',
        sw('reminders', !pf.reminders_off, 'Audit reminders', 'When a step in one of your audits is due') +
        sw('consent_news', pf.consent_news, 'Ai4Qi news', 'New features, about once a month') +
        sw('consent_sponsors', pf.consent_sponsors, 'Offers from our sponsors', 'Courses, events and jobs, sent by us')) +
      '<div class="ac-save"><button class="btn" type="submit">Save changes</button>' +
        '<a class="btn btn-secondary" href="' + attr(afterSigninHref()) + '" data-skip-profile>Skip for now</a>' +
        '<p class="form-status" data-form-status role="status" aria-live="polite"></p></div>' +
      '</form>' +
      (BE.admin ? card('ac-adm', 'Administration', '', '<p class="ac-links"><a href="#/admin/stats">Usage statistics →</a><a href="#/admin/feedback">Feedback on audits →</a><a href="#/admin/access">Access requests →</a></p>') : '') +
      '<div class="ac-danger"><button class="link-btn" type="button" data-delete-account>Delete my account</button><span data-delete-account-box></span></div>' +
      '</div>', 'Account', 'account', true);
  }
  /* ---------- who can build and run audits: NHS, HSE and university staff ----------
     Work and university emails get in straight away. Anyone else asks for access; a request is
     approved automatically after 5 minutes unless an admin rejects it (migrations 010, 013). Demo mode
     and a site without a backend are never gated. The build function checks the same rule. */
  var INST_EMAIL = /@(([a-z0-9-]+\.)*ac\.uk|([a-z0-9-]+\.)*(tcd|ucd|ucc|ul|dcu|mu|rcsi|universityofgalway|nuigalway|atu|tus|setu|mtu|tudublin)\.ie|([a-z0-9-]+\.)*rcsi\.com)$/i;
  function isInstitutional(email) { return WORK_EMAIL.test(email || '') || INST_EMAIL.test(email || ''); }
  function accessStatus() {
    if (!BE.url || DEMO) return Promise.resolve('ok');
    return sbClient().then(function (c) {
      if (!BE.user) return 'signed_out';
      if (isInstitutional(BE.user.email)) return 'ok';
      return c.rpc('my_access').then(function (r) { return r.error ? 'ok' : String(r.data || 'none'); }, function () { return 'ok'; });
    }, function () { return 'ok'; });
  }
  // Returns true when the action can go ahead now; otherwise checks, then runs `then` or shows the gate.
  function accessGate(then) {
    if (S.accessOk || !BE.url || DEMO) return true;
    accessStatus().then(function (st) {
      if (st === 'ok') { S.accessOk = true; then(); return; }
      S.accessNext = location.hash; renderAccess(st);
    });
    return false;
  }
  function renderAccess(known) {
    if (!known) { page(loadingHtml('Checking…'), 'Access', ''); accessStatus().then(function (st) { if (parseHash().parts[0] === 'access') renderAccess(st); }); return; }
    var st = known, email = BE.user ? BE.user.email || '' : '', body;
    if (st === 'ok') {
      S.accessOk = true;
      body = '<div class="auth-icon is-ok" aria-hidden="true">' + ICON_SENT + '</div><h1>You have access</h1><p class="auth-sub">You can build and run audits.</p>' +
        '<a class="btn auth-wide" href="' + attr(S.accessNext && S.accessNext !== '#/access' ? S.accessNext : '#/my-audits') + '">Continue</a>';
    } else if (st === 'signed_out') {
      body = '<div class="auth-icon" aria-hidden="true">' + ICON_MAIL + '</div><h1>Sign in with your work email</h1>' +
        '<p class="auth-sub">Use your institutional email (nhs.net, Trust, HSE or university) for instant access. Other emails need a quick approval first.</p>' +
        '<a class="btn auth-wide" href="#/account">Sign in</a><p class="auth-note">Browsing the library, standards and ready-made audits stays open to everyone.</p>';
    } else if (st === 'pending') {
      body = '<div class="auth-icon" aria-hidden="true">' + ICON_MAIL + '</div><h1>Request received</h1>' +
        '<p class="auth-sub">Personal emails are approved automatically within about 5 minutes. This page lets you in as soon as it is done.</p>' +
        '<button class="btn auth-wide" type="button" data-access-recheck>Check now</button><p class="form-status" role="status" data-access-status></p>' +
        '<p class="auth-note">Signed in as ' + esc(email) + '. Have a work email? <button type="button" class="link-btn" data-signout-to-signin>Sign in with it instead</button></p>';
    } else if (st === 'rejected') {
      body = '<h1>Access not approved</h1><p class="auth-sub">Ai4Qi is for NHS, HSE and university staff. Sign in with your work email to build and run audits.</p>' +
        '<button class="btn auth-wide" type="button" data-signout-to-signin>Sign in with a work email</button>';
    } else {
      body = '<h1>Use your work email</h1><p class="auth-sub">Institutional emails (NHS, HSE or university) get instant access. You are signed in as <strong>' + esc(email) + '</strong>, which needs a quick approval first.</p>' +
        '<button class="btn auth-wide" type="button" data-signout-to-signin>Sign in with my work email</button>' +
        '<details class="acc-req"><summary>No work email? Ask for access</summary>' +
        '<form class="auth-form acc-form" data-access-request novalidate>' +
        '<label for="ar-role">Your role</label><select id="ar-role" name="role" required><option value="">Choose…</option>' + optionList(GRADES, '') + '</select>' +
        '<label for="ar-work">Where you work or study</label><input id="ar-work" name="workplace" maxlength="160" list="org-list" required>' + orgList() +
        '<label for="ar-why">Why you need access <span class="muted">optional</span></label><textarea id="ar-why" name="reason" maxlength="500" rows="3"></textarea>' +
        '<button class="btn auth-wide" type="submit">Ask for access</button><p class="form-status" role="status" data-access-status></p></form></details>';
    }
    page('<div class="auth-wrap"><div class="auth-card">' + body + '</div></div>', 'Access', '');
    clearInterval(S.accessTimer);
    if (st === 'pending') S.accessTimer = setInterval(function () {      // let them in as soon as it is approved
      if (!document.querySelector('[data-access-recheck]')) { clearInterval(S.accessTimer); return; }
      accessStatus().then(function (v) { if (v !== 'pending') { clearInterval(S.accessTimer); renderAccess(v); } });
    }, 30000);
  }
  document.addEventListener('submit', function (e) {
    var f = e.target.closest && e.target.closest('[data-access-request]'); if (!f) return;
    e.preventDefault();
    var stEl = f.querySelector('[data-access-status]'), role = f.elements.role.value, work = f.elements.workplace.value.trim();
    if (!role || !work) { stEl.textContent = 'Please choose your role and say where you work or study.'; stEl.classList.add('is-error'); return; }
    stEl.classList.remove('is-error'); stEl.textContent = 'Sending…';
    sbClient().then(function (c) {
      return c.from('access_requests').insert({ user_id: BE.user.id, role: role, workplace: work.slice(0, 160), reason: f.elements.reason.value.trim().slice(0, 500) || null });
    }).then(function (r) {
      if (r.error && !/duplicate|unique/i.test(r.error.message || '')) throw r.error;
      renderAccess('pending');
    }).catch(function () { stEl.textContent = 'Your request could not be sent. Please try again.'; stEl.classList.add('is-error'); });
  });
  document.addEventListener('click', function (e) {
    if (e.target.closest('[data-access-recheck]')) {
      var st = main.querySelector('[data-access-status]'); if (st) st.textContent = 'Checking…';
      accessStatus().then(function (v) { if (v === 'pending' && st) st.textContent = 'Not yet. Please check again in a few minutes.'; else renderAccess(v); });
    } else if (e.target.closest('[data-signout-to-signin]')) {
      sbClient().then(function (c) { return c.auth.signOut({ scope: 'local' }); }).catch(function () {}).then(function () {
        setUser(null); setLastEmail(''); S.accessOk = false; location.hash = '#/account'; renderSignIn(null, true);
      });
    }
  });
  /* admin: access requests (#/admin/access) */
  function renderAdminAccess() {
    page(loadingHtml('Loading requests…'), 'Access requests', '');
    sbClient().then(function (c) {
      if (!BE.user) return null;
      return checkAdmin(c).then(function (ok) {
        if (!ok) return null;
        return Promise.all([c.from('access_requests').select('*').order('created_at', { ascending: false }).limit(200), c.from('allowed_domains').select('*').order('domain')]);
      });
    }).then(function (res) {
      if (!res) { page('<article class="doc narrow"><h1>Not available</h1><p class="prose">Sign in with an admin account.</p></article>', 'Access requests', ''); return; }
      var rows = (res[0].data || []), doms = (res[1].data || []);
      function stateOf(r) { return r.decision || (Date.now() - new Date(r.created_at).getTime() > 300000 ? 'approved (auto)' : 'waiting'); }
      page('<nav class="breadcrumb" aria-label="Breadcrumb"><a href="#/">Home</a> › <a href="#/account">Account</a> › Access requests</nav><h1>Access requests</h1>' +
        '<p class="prose">Requests are approved automatically 5 minutes after they arrive unless you reject them. Allowing a domain lets everyone with that email in straight away.</p>' +
        '<div class="table-wrap"><table class="sheet"><thead><tr><th>When</th><th>Email</th><th>Role</th><th>Workplace</th><th>Why</th><th>Status</th><th></th></tr></thead><tbody>' +
        (rows.length ? rows.map(function (r) {
          var dom = String(r.email || '').split('@')[1] || '';
          return '<tr><td>' + esc(new Date(r.created_at).toLocaleString('en-GB')) + '</td><td>' + esc(r.email) + '</td><td>' + esc(r.role || '') + '</td><td>' + esc(r.workplace || '') + '</td><td>' + esc(r.reason || '') + '</td><td>' + esc(stateOf(r)) + '</td>' +
            '<td><button class="link-btn" data-acc-decide="approved" data-uid="' + attr(r.user_id) + '">Approve</button> · <button class="link-btn" data-acc-decide="rejected" data-uid="' + attr(r.user_id) + '">Reject</button>' +
            (dom ? ' · <button class="link-btn" data-acc-domain="' + attr(dom) + '">Allow @' + esc(dom) + '</button>' : '') + '</td></tr>';
        }).join('') : '<tr><td colspan="7">No requests yet.</td></tr>') + '</tbody></table></div>' +
        '<h2>Allowed domains</h2><p class="prose">' + (doms.length ? doms.map(function (d) { return esc(d.domain); }).join(', ') : 'None yet (NHS, HSE and university emails are always allowed).') + '</p>' +
        '<p class="form-status" role="status" data-acc-admin-status></p>', 'Access requests', '');
    }).catch(function () { page('<article class="doc narrow"><h1>Not available yet</h1><p class="prose">Run migration 010 in Supabase first.</p></article>', 'Access requests', ''); });
  }
  document.addEventListener('click', function (e) {
    var d = e.target.closest('[data-acc-decide]'), dm = e.target.closest('[data-acc-domain]');
    if (!d && !dm) return;
    var st = main.querySelector('[data-acc-admin-status]');
    sbClient().then(function (c) {
      return d ? c.from('access_requests').update({ decision: d.getAttribute('data-acc-decide'), decided_at: new Date().toISOString() }).eq('user_id', d.getAttribute('data-uid'))
        : c.from('allowed_domains').upsert({ domain: dm.getAttribute('data-acc-domain').toLowerCase() });
    }).then(function (r) { if (r.error) throw r.error; renderAdminAccess(); }, function () {}).catch(function () { if (st) st.textContent = 'That did not save. Please try again.'; });
  });

  /* ---------- "Make Ai4Qi better": ideas for the site, from anyone, kept with the audit feedback
     (audit_id "SITE-<area>-<random>", so each idea is its own row). ---------- */
  var IDEA_AREAS = [['build', 'Building an audit'], ['audit', 'Audit page'], ['sheet', 'Data sheet'], ['results', 'Results'],
    ['slides', 'Presentation'], ['signin', 'Signing in'], ['other', 'Something else']];
  function renderIdeas(sent) {

    page('<article class="doc narrow ideas">' +
      '<p class="eyebrow-k">Make Ai4Qi better</p><h1>Hit a block? Found a problem? Want something changed?</h1>' +
      '<p class="page-intro">Post it here and we will make the change for you. We read every post and build the good ones, often within days.</p>' +
      (sent ? '<div class="idea-done" role="status"><b>' + esc(IDEA_THANKS()) + '</b>' +
        '<button type="button" class="btn btn-secondary" data-idea-again>Post another</button></div>' :
      '<form class="idea-form" data-idea novalidate>' +
        '<fieldset><legend>Which part?</legend><div class="idea-areas">' + IDEA_AREAS.map(function (a, i) {
          return '<label class="idea-chip"><input type="radio" name="area" value="' + a[0] + '"' + (i === IDEA_AREAS.length - 1 ? ' checked' : '') + '><span>' + a[1] + '</span></label>';
        }).join('') + '</div></fieldset>' +
        '<label for="idea-t">What happened, or what would you change?</label>' +
        '<textarea id="idea-t" name="text" rows="5" maxlength="480" placeholder="e.g. The results page should also show the trend by ward, or: I could not find where to change the sample size."></textarea>' +
        '<p class="idea-count" data-idea-count>0 / 480</p>' +
        (BE.user ? '<p class="idea-mail">We will email you our answer: done, where it already is, or why we cannot change it.</p>' : '') +
        '<p class="muted idea-note">Read by the Ai4Qi team. Please do not include names or patient details.</p>' +
        '<button class="btn" type="submit">Post it</button><p class="form-status" role="status" data-idea-status></p>' +
      '</form>') + '</article>', 'Suggest a change', '');
  }
  document.addEventListener('input', function (e) {
    var t = e.target.closest && e.target.closest('#idea-t'); if (!t) return;
    var c = main.querySelector('[data-idea-count]'); if (c) c.textContent = t.value.length + ' / 480';
  });
  document.addEventListener('submit', function (e) {
    var f = e.target.closest && e.target.closest('[data-idea]'); if (!f) return;
    e.preventDefault();
    var txt = scrub(f.elements.text.value.trim(), { redacted: 0 }), st = f.querySelector('[data-idea-status]');
    if (txt.length < 5) { st.textContent = 'Please write a few words about your idea.'; st.classList.add('is-error'); return; }
    var area = (f.querySelector('input[name="area"]:checked') || {}).value || 'other';
    var rnd = Math.random().toString(36).slice(2, 8);
    recordFeedback('SITE-' + area + '-' + rnd, 'up', [], txt.slice(0, 500));
    whoosh(); f.elements.text.classList.add('is-sent');
    setTimeout(function () { renderIdeas(true); }, 650);
  });
  document.addEventListener('click', function (e) { if (e.target.closest && e.target.closest('[data-idea-again]')) renderIdeas(false); });

  function renderSignedOut() {
    page('<div class="auth-wrap"><div class="auth-card"><div class="auth-icon is-ok" aria-hidden="true"><svg width="44" height="44" viewBox="0 0 24 24"><path d="M5 12.5l4.5 4.5L19 7.5" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"/></svg></div><h1>You are signed out</h1>' +
      '<p class="auth-sub">Your audits stay on this device. Sign in again to get reminders.</p>' +
      '<a class="btn auth-wide" href="#/account">Sign in again</a><a class="link-btn" href="#/">Go to the home page</a></div></div>', 'Signed out', '');
  }

  document.addEventListener('click', function (e) {
    if (e.target.closest('[data-skip-profile]')) { try { localStorage.removeItem(AFTER_SIGNIN); } catch (e2) {} }
  });
  document.addEventListener('submit', function (e) {
    var otp = e.target.closest && e.target.closest('[data-otp]');
    if (otp) {
      e.preventDefault();
      var code = otp.elements.code.value.replace(/\D/g, ''), ost = otp.querySelector('[data-form-status]');
      if (code.length < 6) { ost.textContent = 'Type the code from the email.'; ost.classList.add('is-error'); return; }
      ost.classList.remove('is-error'); ost.textContent = 'Checking…';
      sbClient().then(function (c) { return c.auth.verifyOtp({ email: otp.elements.email.value, token: code, type: 'email' }); })
        .then(function (res) {
          if (res.error) throw res.error;
          S.justSignedIn = true;
          if (res.data && res.data.user) setUser(res.data.user);
          location.hash = signedInHome();
        }).catch(function () { ost.textContent = 'That code did not work. Check it, or use the newest email.'; ost.classList.add('is-error'); });
      return;
    }
    var signin = e.target.closest('[data-signin]'), prof = e.target.closest('[data-profile]');
    if (!signin && !prof) return;
    e.preventDefault();
    var form = signin || prof, btn = form.querySelector('button[type="submit"]'), status = form.querySelector('[data-form-status]');
    function say(text, bad) { status.textContent = text; status.classList.toggle('is-error', !!bad); }
    if (signin) {
      var input = form.querySelector('input[name="email"]'), email = input.value.trim();
      if (email && email.indexOf('@') === -1 && input.id === 'acc-email') { email += '@nhs.net'; input.value = email; }   // the grey @nhs.net ending
      if (/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email) && !isInstitutional(email) && form.getAttribute('data-warned') !== email.toLowerCase()) {
        form.setAttribute('data-warned', email.toLowerCase());
        var w = form.querySelector('[data-email-warn]') || (function () { var p = document.createElement('p'); p.className = 'auth-warn'; form.insertBefore(p, btn); return p; })();
        w.innerHTML = NONINST_WARN; w.hidden = false;
        btn.innerHTML = 'Continue with this email &nbsp;→';
        input.focus(); return;
      }
      if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email)) {
        input.setAttribute('aria-invalid', 'true'); say('Please enter a valid email address.', true); input.focus(); return;
      }
      input.removeAttribute('aria-invalid');
      btn.disabled = true; say('Sending…');
      sendLink(email).then(function () { renderSent(email); }, function (err) { btn.disabled = false; say(sendError(err), true); });
    } else {
      var pick = function (name, list) { var v = form.querySelector('[name="' + name + '"]').value; return list.indexOf(v) === -1 ? null : v; };
      var spec = pick('specialty', SPECIALTIES), grade = pick('grade', GRADES), region = pick('region', REGIONS);
      var row = { user_id: BE.user && BE.user.id, specialty: spec, grade: grade, region: region,
        work_setting: (BE.profile || {}).work_setting || null, audit_purpose: (BE.profile || {}).audit_purpose || null,
        consent_news: form.elements.consent_news.checked, consent_sponsors: form.elements.consent_sponsors.checked,
        full_name: form.elements.full_name.value.trim().slice(0, 120) || null, organisation: form.elements.organisation.value.trim().slice(0, 160) || null,
        department: (BE.profile || {}).department || null, reminders_off: !form.elements.reminders.checked };
      var prev = BE.profile || {};
      if (row.consent_news !== !!prev.consent_news || row.consent_sponsors !== !!prev.consent_sponsors) row.consent_updated_at = new Date().toISOString();
      btn.disabled = true; say('Saving…');
      sbClient().then(function (c) {
        if (!BE.user) throw new Error('signed out');
        return c.from('profiles').upsert(row, { onConflict: 'user_id' });
      }).then(function (res) {
        if (res.error) throw res.error;
        BE.profile = row; updateAccountLink();
        if (row.reminders_off) sbClient().then(function (c2) { return c2.from('run_reminders').delete().eq('user_id', BE.user.id); }).catch(function () {});
        else syncAllRuns();                                  // audits on this device; others sync when opened
        btn.disabled = false; say('Your profile has been saved.');
        setTimeout(function () { location.hash = '#/my-audits'; }, 600);
      }).catch(function () {
        btn.disabled = false; say('Your profile could not be saved. Please check your connection and try again.', true);
      });
    }
  });
  document.addEventListener('click', function (e) {
    if (e.target.closest('[data-signin-again]') || e.target.closest('[data-signin-other]')) { renderSignIn(null, true); var i = main.querySelector('#acc-email'); if (i) i.focus(); return; }
    var again = e.target.closest('[data-signin-resend]');
    if (again) {
      var em = again.getAttribute('data-signin-resend'); again.disabled = true; again.textContent = 'Sending…';
      sendLink(em).then(function () { renderSent(em); }, function (err) { again.disabled = false; again.textContent = sendError(err); });
      return;
    }
    if (e.target.closest('[data-signout]')) {
      sbClient().then(function (c) { return c.auth.signOut({ scope: 'local' }); })
        .catch(function () {}).then(function () { setUser(null); location.hash = '#/signed-out'; });
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
  function renderAdminPage(allRows) {
    var ideas = allRows.filter(function (r) { return /^SITE-/.test(r.audit_id || ''); }), rows = allRows.filter(function (r) { return !/^SITE-/.test(r.audit_id || ''); });
    var ideasHtml = '<h2>Ideas for the site (' + ideas.length + ')</h2>' + (ideas.length ? '<ul class="adm-comments">' + ideas.map(function (r) {
      var area = (String(r.audit_id).split('-')[1] || 'other'), lab = (IDEA_AREAS.filter(function (a) { return a[0] === area; })[0] || [0, area])[1];
      return '<li><p class="meta">' + esc(dateGB(r.created_at)) + ' · ' + esc(lab) + (r.user_id ? ' · signed in' : '') + '</p><p class="adm-comment">' + esc(r.comment) + '</p></li>';
    }).join('') + '</ul>' : '<p class="empty">No ideas yet.</p>');
    var per = summarise(rows), up = 0, down = 0;
    S.ideasText = 'Ai4Qi site ideas (' + ideas.length + '), newest first. Format: date | part | signed in | idea\n' + ideas.map(function (r) {
      var area = (String(r.audit_id).split('-')[1] || 'other');
      return String(r.created_at || '').slice(0, 10) + ' | ' + area + ' | ' + (r.user_id ? 'yes' : 'no') + ' | ' + String(r.comment || '').replace(/\s+/g, ' ');
    }).join('\n');
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
      '<h1>Feedback</h1>' + '<section class="adm-ideas">' + ideasHtml +
      (ideas.length ? '<p><button class="btn" type="button" data-ideas-copy>Copy ideas for Claude</button> <span class="muted" data-ideas-status>Then paste them into your Claude conversation and ask for a review.</span></p>' : '') + '</section>' +
      '<h2>Feedback on proposed audits</h2>' +
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
      // The hosted pages carry the beacon in their head (build_app_data.py writes it); never add a second one.
      if (document.querySelector('script[data-cf-beacon]')) { AN.kind = 'cloudflare'; return; }
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

  load().then(function () { route(); loadVotes(); }).catch(function (err) {
    main.innerHTML = '<h1>The library could not be loaded</h1><p>Please check your connection and refresh the page.</p>';
    if (window.console) console.warn(err);
  });

  /* ---------- help box: instant answers, no AI, nothing leaves the device ---------- */
  // Each answer: q (the question), k (extra words people use), a (short answer), go ([label, link]).
  var HELP = [
    { q: 'How do I start an audit?', k: 'begin new first create make choose pick', a: 'Type a topic on the home page and press Build, or pick one of the ready-made audits. Open it, press Choose this audit, then Start this audit when you are ready.', go: [['Build an audit', '#/'], ['Ready-made audits', '#/proposed']] },
    { q: 'Is Ai4Qi free?', k: 'cost price pay money subscription charge', a: 'Yes. Ai4Qi is free to use.' },
    { q: 'Do I need an account?', k: 'sign register login log email password', a: 'Browsing is open to everyone. To build or run an audit, sign in: an institutional email (NHS, HSE or university) gets instant access; other emails go through a quick approval first. We send a code; there is no password.', go: [['Sign in', '#/account']] },
    { q: 'I did not get the sign-in email', k: 'email code link spam junk arrive missing login not received', a: 'Wait a minute or two, then check Junk and the Other tab. NHSmail is slower with new senders. Still nothing after 3 minutes? Sign in with a personal email instead.', go: [['Sign in again', '#/account']] },
    { q: 'Where are my audits?', k: 'dashboard saved find lost my audits list progress', a: 'In My audits. They are kept on this device and in this browser. If you switched device or browser, restore a backup there.', go: [['My audits', '#/my-audits']] },
    { q: 'How do I enter my data?', k: 'data upload excel spreadsheet sheet enter record collect import input', a: 'Open your audit in My audits, download the Excel data sheet, fill in one row per patient, then upload it under Upload your data sheet. The results appear straight away.' },
    { q: 'Can I add or remove columns in the data sheet?', k: 'column add delete extra change excel field question', a: 'Yes. Open the audit in My audits and add or remove columns before you download the sheet. Or add them in Excel: when you upload, Ai4Qi offers to add the new columns and includes them in the results.' },
    { q: 'What happens to patient identifiers?', k: 'identifier nhs number hospital number name mrn dob patient confidential gdpr id', a: 'Do not put them in the sheet. If a column looks like an identifier (name, NHS or hospital number, date of birth), Ai4Qi leaves it out on upload. The Code column refuses long numbers like NHS numbers.' },
    { q: 'Is my data safe? Who can see it?', k: 'privacy secure safe data protection gdpr encrypted server see', a: 'Your records stay encrypted in your own browser. They are never sent to Ai4Qi or to the AI, and we cannot see them.', go: [['How your data is protected', '#/privacy']] },
    { q: 'How many patients do I need?', k: 'sample size number patients how many cases enough power strength', a: 'Each audit suggests a sample, and shows how precise it is (50 patients gives about ±14 points) and how many you need in each cycle to show a real rise to the target. You can do fewer; after the re-audit the results say whether the change could be chance.' },
    { q: 'Can I change the audit question or title?', k: 'edit change reword title question tweak adapt modify', a: 'Yes. Open the audit in My audits and edit the Audit question under Audit details. The title follows it unless you changed the title yourself.', go: [['My audits', '#/my-audits']] },
    { q: 'What is the re-audit?', k: 'reaudit re audit second cycle close loop closing again repeat', a: 'After cycle 1 you make one change, then collect the same data again with a new re-audit sheet. The results then show before and after. Open Re-audit: close the loop on the audit page.' },
    { q: 'What goes in The change?', k: 'change intervention action improvement what did you change', a: 'The one thing you changed after cycle 1, for example a new checklist, a form or a default in the system, and the date you started it. It appears on the results and in the presentation.' },
    { q: 'How do I make the presentation?', k: 'powerpoint pptx slides presentation deck present meeting teaching', a: 'Under Files on your audit, press Results presentation. Choose a design from the list first if you like; there are 20. The slides contain your charts and tables and no Ai4Qi branding.' },
    { q: 'How do I send the proposal to my supervisor?', k: 'proposal supervisor consultant approve approval word document send registration', a: 'Open Send to your supervisor on your audit. It writes the email and a Word proposal with a sign-off box. Download the proposal and attach it; it holds the protocol only, never patient data. Register the audit with your audit department too.' },
    { q: 'How do I move my audits to another device?', k: 'backup restore transfer device phone computer laptop move export', a: 'Under Files on your audit, press Backup. On the other device, open My audits, then Backup and protection, then Restore a backup.', go: [['My audits', '#/my-audits']] },
    { q: 'What is demo mode?', k: 'demo example try test practice sample data', a: 'Demo mode fills an audit with example data so you can see every step, the results and the presentation. Your own audits are kept separately.', go: [['Try demo mode', '#/demo']] },
    { q: 'Where do the standards come from?', k: 'standard nice guideline royal college source reference evidence', a: 'From NICE, the Royal Colleges, NHS England, national audits and others, quoted exactly with a link to the source. Where none exists the audit says Local standard.', go: [['Browse standards', '#/standards']] },
    { q: 'Is the audit written by AI? Can I trust it?', k: 'ai artificial intelligence claude trust accurate correct wrong made up', a: 'Audits you build are written by AI from the library and checked standards, following strict rules. Treat it as a well-founded draft: check the standard against its source and ask your supervisor to review it.', go: [['How Ai4Qi works', '#/how-it-works']] },
    { q: 'The audit is not quite right', k: 'wrong bad not right different another version improve feedback', a: 'On a built audit press Not quite right? to get another version on the same theme, and tell us why. On any audit, use the feedback box to vote and comment.' },
    { q: 'What is the difference between an audit and a QI project?', k: 'qi quality improvement pdsa difference project service evaluation', a: 'An audit measures care against a standard, makes a change and measures again. QI tests changes in small cycles. Ai4Qi audits close the loop with a re-audit.' },
    { q: 'Will I get reminders?', k: 'reminder email notification due date deadline alert', a: 'Yes, if you sign in and press Start this audit. We email you when a step is due. You can turn them off in your account.', go: [['Account', '#/account']] },
    { q: 'Can I install it as an app?', k: 'install app phone android iphone home screen offline', a: 'Yes. In Chrome or Edge press Install app at the top. On an iPhone, tap Share, then Add to Home Screen. It then works offline too.' },
    { q: 'Can I search published audits?', k: 'library published search find papers similar examples', a: 'Yes. Search the library of published audits and QI projects by topic, with the result before and after each change.', go: [['Search the library', '#/search']] },
    { q: 'How long does an audit take?', k: 'time long duration weeks hours effort how long quick', a: 'Setting up, the proposal, the results and the presentation take about 5 minutes each with Ai4Qi. Collecting the data is the part that takes time; each audit shows an estimate and a timeline.' },
    { q: 'How do I report a problem?', k: 'bug problem error broken contact help issue report', a: 'Use the feedback box on any audit page to tell us, or see the Security page for anything sensitive.', go: [['Security', '#/security']] }
  ];
  // Opening the suggestion form gives it the whole panel (the questions fold away), so the box and
  // the Post button are always in view; Enter posts it (Shift+Enter for a new line).
  document.addEventListener('toggle', function (e) {
    var d = e.target; if (!d.matches || !d.matches('details.hb-idea')) return;
    var box = d.closest('.help-box'); if (box) box.classList.toggle('idea-open', d.open);
    if (d.open) { var t = d.querySelector('textarea'); if (t) setTimeout(function () { t.focus(); }, 0); }
  }, true);
  document.addEventListener('keydown', function (e) {
    var t = e.target; if (e.key !== 'Enter' || e.shiftKey || !t.closest || !t.closest('[data-hb-idea] textarea')) return;
    e.preventDefault(); var f = t.form; if (f.requestSubmit) f.requestSubmit(); else f.dispatchEvent(new Event('submit', { cancelable: true, bubbles: true }));
  });
  document.addEventListener('submit', function (e) {
    var f = e.target.closest && e.target.closest('[data-hb-idea]'); if (!f) return;
    e.preventDefault();
    var txt = scrub(f.elements.text.value.trim(), { redacted: 0 }), st = f.querySelector('.hb-idea-st');
    if (txt.length < 5) { st.textContent = 'Please write a few words.'; return; }
    recordFeedback('SITE-' + (f.elements.area.value || 'other') + '-' + Math.random().toString(36).slice(2, 8), 'up', [], txt.slice(0, 500));
    whoosh();
    var ta = f.elements.text;
    ta.classList.add('is-sent');                       // the note flies away
    setTimeout(function () {
      ta.value = ''; ta.classList.remove('is-sent');
      st.textContent = IDEA_THANKS();
    }, 650);
  });
  function IDEA_THANKS() { return BE.user ? 'Thanks! We are looking into it and will reply.' : 'Thanks! We are looking into it.'; }
  // A short "swoosh": filtered noise sweeping down, made in the browser (no sound file). Only after a tap.
  function whoosh() {
    try {
      var AC = window.AudioContext || window.webkitAudioContext; if (!AC) return;
      var ac = whoosh.ac || (whoosh.ac = new AC()), t = ac.currentTime, len = Math.floor(ac.sampleRate * 0.5);
      var buf = ac.createBuffer(1, len, ac.sampleRate), d = buf.getChannelData(0);
      for (var i = 0; i < len; i++) d[i] = (Math.random() * 2 - 1) * Math.pow(1 - i / len, 2);
      var src = ac.createBufferSource(); src.buffer = buf;
      var bp = ac.createBiquadFilter(); bp.type = 'bandpass'; bp.Q.value = 1.2;
      bp.frequency.setValueAtTime(2400, t); bp.frequency.exponentialRampToValueAtTime(300, t + 0.45);
      var g = ac.createGain(); g.gain.setValueAtTime(0.0001, t); g.gain.exponentialRampToValueAtTime(0.35, t + 0.06); g.gain.exponentialRampToValueAtTime(0.0001, t + 0.48);
      src.connect(bp); bp.connect(g); g.connect(ac.destination); src.start(t); src.stop(t + 0.5);
    } catch (e) {}
  }
  var HELP_IDX = null;
  function helpIndex() {
    if (HELP_IDX) return HELP_IDX;
    HELP_IDX = HELP.map(function (h) { return { h: h, m: indexDoc([[h.q, 3], [h.k, 2], [h.a, 1]]) }; });
    return HELP_IDX;
  }
  function helpScore(m, qt) {
    var sc = 0, hit = 0;
    qt.forEach(function (t) {
      var best = m.get(t) || 0;
      if (!best && t.length >= 3) m.forEach(function (w, k) { if (k.indexOf(t) === 0 && w * 0.8 > best) best = w * 0.8; });
      if (best) { hit++; sc += best; }
    });
    return hit ? sc * hit / qt.length : 0;
  }
  function helpFind(q) {
    var qt = tokens(q);
    if (!qt.length) return { faq: [], audits: [] };
    var faq = helpIndex().map(function (d) { return { h: d.h, s: helpScore(d.m, qt) }; })
      .filter(function (x) { return x.s >= 1.5; }).sort(function (a, b) { return b.s - a.s; }).slice(0, 3).map(function (x) { return x.h; });
    var audits = (S.proposed || []).map(function (p) {
      p._hm = p._hm || indexDoc([[auditName(p), 3], [p.question, 2], [p.area, 1]]);
      return { p: p, s: helpScore(p._hm, qt) };
    }).filter(function (x) { return x.s >= 2; }).sort(function (a, b) { return b.s - a.s; }).slice(0, 3).map(function (x) { return x.p; });
    return { faq: faq, audits: audits };
  }
  function helpAnswer(h, open) {
    return '<details class="hb-a"' + (open ? ' open' : '') + '><summary>' + esc(h.q) + '</summary><p>' + esc(h.a) + '</p>' +
      (h.go ? '<p class="hb-go">' + h.go.map(function (g) { return '<a href="' + attr(g[1]) + '" data-help-close>' + esc(g[0]) + ' →</a>'; }).join('') + '</p>' : '') + '</details>';
  }
  var HELP_START = [0, 5, 7, 9, 13, 11];
  function helpResults(q) {
    q = String(q || '').trim();
    if (!q) return '<p class="hb-k">Common questions</p>' + HELP_START.map(function (i) { return helpAnswer(HELP[i]); }).join('');
    var r = helpFind(q), out = '';
    if (r.faq.length) out += r.faq.map(function (h, i) { return helpAnswer(h, i === 0); }).join('');
    if (r.audits.length) out += '<p class="hb-k">Ready-made audits</p><ul class="hb-list">' + r.audits.map(function (p) {
      return '<li><a href="#/proposed/' + encodeURIComponent(p.id) + '" data-help-close>' + esc(auditName(p)) + '</a></li>'; }).join('') + '</ul>';
    if (!r.faq.length) out += '<p class="hb-none">' + (r.audits.length ? '' : 'No answer found for that. ') + '</p>';
    if (r.faq.length && !r.audits.length) return out;
    out += '<p class="hb-k">' + (r.faq.length || r.audits.length ? 'Or' : 'Try') + '</p><ul class="hb-list">' +
      '<li><a href="#/build?q=' + encodeURIComponent(q) + '" data-help-close>Build an audit on “' + esc(q) + '”</a></li>' +
      '<li><a href="#/search?q=' + encodeURIComponent(q) + '" data-help-close>Search published audits for “' + esc(q) + '”</a></li></ul>';
    return out;
  }
  if (!window.AI4QI_EMBED) (function () {
    var btn = document.createElement('button');
    btn.type = 'button'; btn.className = 'help-fab'; btn.setAttribute('aria-expanded', 'false'); btn.setAttribute('aria-controls', 'help-box');
    btn.innerHTML = '<span aria-hidden="true">?</span><span class="help-fab-t">Help</span>';
    var box = document.createElement('section');
    box.id = 'help-box'; box.className = 'help-box'; box.hidden = true; box.setAttribute('aria-label', 'Help');
    box.innerHTML = '<div class="hb-head"><h2>How can we help?</h2><button type="button" class="hb-x" data-help-close aria-label="Close help">×</button></div>' +
      '<form class="hb-form" role="search" data-help-form><label class="visually-hidden" for="help-q">Your question</label>' +
      '<input id="help-q" type="search" autocomplete="off" placeholder="Ask a question, e.g. how do I upload data?"></form>' +
      '<div class="hb-res" data-help-res aria-live="polite"></div>' +
      '<details class="hb-idea"><summary><b>💡 Hit a block or want a change?</b><span>Post it here and we will make the change for you.</span></summary>' +
        '<form class="hb-idea-form" data-hb-idea novalidate><select name="area" aria-label="Which part">' + IDEA_AREAS.map(function (a) { return '<option value="' + a[0] + '"' + (a[0] === 'other' ? ' selected' : '') + '>' + a[1] + '</option>'; }).join('') + '</select>' +
        '<textarea name="text" rows="3" maxlength="480" aria-label="What happened, or what would you change?" placeholder="What happened, or what would you change? (no patient details)"></textarea>' +
        '<button class="btn" type="submit">Post it</button><p class="hb-idea-st" role="status"></p></form></details>' +
      '<p class="hb-foot">Answers come from this site. Nothing you type in the search box is sent anywhere. <a href="#/how-it-works" data-help-close>How Ai4Qi works</a></p>';
    document.body.appendChild(box); document.body.appendChild(btn);
    var inp = box.querySelector('#help-q'), res = box.querySelector('[data-help-res]'), t = null;
    function show(open) {
      box.hidden = !open; btn.setAttribute('aria-expanded', open ? 'true' : 'false'); btn.classList.toggle('is-open', open);
      if (open) { res.innerHTML = helpResults(inp.value); inp.focus(); } else btn.focus({ preventScroll: true });
    }
    btn.addEventListener('click', function () { show(box.hidden); });
    inp.addEventListener('input', function () { clearTimeout(t); t = setTimeout(function () { res.innerHTML = helpResults(inp.value); }, 120); });
    box.querySelector('[data-help-form]').addEventListener('submit', function (e) { e.preventDefault(); res.innerHTML = helpResults(inp.value); });
    box.addEventListener('click', function (e) { if (e.target.closest('[data-help-close]')) { box.hidden = true; btn.setAttribute('aria-expanded', 'false'); btn.classList.remove('is-open'); } });
    document.addEventListener('keydown', function (e) { if (e.key === 'Escape' && !box.hidden) show(false); });
    document.addEventListener('click', function (e) { if (!box.hidden && !box.contains(e.target) && !btn.contains(e.target)) { box.hidden = true; btn.setAttribute('aria-expanded', 'false'); btn.classList.remove('is-open'); } });
  })();

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
      // A new version installs in the background; when it takes over within the first minute of a visit,
      // reload once so people see today's site, not yesterday's. Later on, it waits for the next visit.
      var hadCtrl = !!navigator.serviceWorker.controller, loadedAt = Date.now(), reloaded = false;
      navigator.serviceWorker.addEventListener('controllerchange', function () {
        if (!hadCtrl || reloaded) return;
        if (Date.now() - loadedAt <= 60000) { reloaded = true; location.reload(); return; }
        // Later in a visit: offer the new version rather than reloading under someone's feet.
        if (document.querySelector('.update-bar')) return;
        var bar = document.createElement('div');
        bar.className = 'update-bar'; bar.setAttribute('role', 'status');
        bar.innerHTML = '<span>A new version of Ai4Qi is ready.</span><button type="button" class="btn">Refresh</button>';
        bar.querySelector('button').addEventListener('click', function () { reloaded = true; location.reload(); });
        document.body.appendChild(bar);
      });
      navigator.serviceWorker.register('sw.js', { scope: './', updateViaCache: 'none' }).then(function (reg) { reg.update().catch(function () {}); }).catch(function () {});
      navigator.serviceWorker.ready.then(function () {
        return window.caches ? caches.match('data/library.json') : null;
      }).then(function (hit) {
        var note = document.querySelector('[data-offline]');
        if (hit && note) note.hidden = false;
      }).catch(function () {});
    });
  }
})();
