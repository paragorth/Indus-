/* Ai4Qi service worker: offline app shell and data. All paths are relative to this file. */
'use strict';

// Stamped by build_app_data.py on every rebuild; a new value makes browsers install a fresh cache.
var VERSION = '217b0d148a69';
var PREFIX = 'ai4qi-';
// GEN 2 (Sep 2026): caches from before the redirect fix are dropped on activate, whatever VERSION says.
var GEN = 'g2-';
var SHELL = PREFIX + 'shell-' + GEN + VERSION;
var RUNTIME = PREFIX + 'runtime-' + GEN + VERSION;

var SHELL_FILES = [
  './', 'app.js', 'export.js', 'styles.css', 'manifest.webmanifest',
  'icons/icon-192.png', 'icons/icon-512.png', 'icons/icon-maskable-512.png', 'icons/apple-touch-icon-180.png',
  'data/library.json', 'data/proposed.json', 'data/cards.json', 'data/standards.json'
];
var BASE = new URL('./', self.location).href;
var SHELL_URLS = new Set(SHELL_FILES.map(function (p) { return new URL(p, BASE).href; }));
var RUNTIME_PATHS = /^(data\/audits\/[^/]+\.json|templates\/[^/]+\.csv|figures\/.+)$/;

// Cloudflare Pages answers /index.html with a redirect to /. Chrome refuses a redirected response for a
// navigation ("This site can't be reached"), so every response is rebuilt without the redirect flag
// before it is cached or returned. index.html itself is never precached: './' is the app shell.
function unredirect(res) {
  if (!res || !res.redirected) return Promise.resolve(res);
  return res.blob().then(function (body) {
    return new Response(body, { status: res.status, statusText: res.statusText, headers: res.headers });
  });
}
function fetchClean(req, init) { return fetch(req, init).then(unredirect); }

self.addEventListener('install', function (event) {
  event.waitUntil(
    caches.open(SHELL).then(function (cache) {
      return Promise.all(SHELL_FILES.map(function (p) {
        return fetchClean(new Request(p, { cache: 'reload' })).then(function (res) {
          if (!res.ok) throw new Error('precache ' + p + ' ' + res.status);
          return cache.put(new URL(p, BASE).href, res);
        });
      }));
    }).then(function () { return self.skipWaiting(); })
  );
});

self.addEventListener('activate', function (event) {
  event.waitUntil(
    caches.keys().then(function (keys) {
      return Promise.all(keys.filter(function (k) {
        return k.indexOf(PREFIX) === 0 && k !== SHELL && k !== RUNTIME;
      }).map(function (k) { return caches.delete(k); }));
    }).then(function () { return self.clients.claim(); })
  );
});

function staleWhileRevalidate(request) {
  return caches.open(RUNTIME).then(function (cache) {
    return cache.match(request).then(function (cached) {
      var network = fetchClean(request).then(function (res) {
        if (res && res.ok) cache.put(request, res.clone());
        return res;
      });
      if (cached) { network.catch(function () {}); return cached; }
      return network;
    });
  });
}

// Supabase API traffic (sign-in, feedback, counts) is never cached: it always goes to the network.
var API_PATHS = /\/(auth|rest|storage|functions|realtime|graphql)\/v1(\/|$)/;
var ANALYTICS = /(^|\.)(plausible\.io|cloudflareinsights\.com)$/i;
var ANALYTICS_PATHS = /\/(js\/(pa-[^/]+|script[^/]*)\.js|api\/event|cdn-cgi\/rum)$/;
var NETWORK_ONLY = new Set(['config.json', 'data/version.json']);

self.addEventListener('fetch', function (event) {
  var req = event.request;
  if (req.method !== 'GET') return;
  var url = new URL(req.url);
  if (API_PATHS.test(url.pathname) || /\.supabase\.(co|in)$/i.test(url.hostname)) return;
  // Visitor analytics (Plausible, Cloudflare Web Analytics) always go straight to the network.
  if (ANALYTICS.test(url.hostname) || ANALYTICS_PATHS.test(url.pathname)) return;
  if (url.origin !== self.location.origin || url.href.indexOf(BASE) !== 0) return;

  var rel = url.href.slice(BASE.length).split(/[?#]/)[0];

  // Runtime settings are read fresh every time and never stored.
  if (NETWORK_ONLY.has(rel)) {
    event.respondWith(fetchClean(req, { cache: 'no-store' }));
    return;
  }

  // Opening the app (any hash or query, or the old /index.html address) gets the cached app shell './'.
  if (req.mode === 'navigate' && (rel === '' || rel === 'index.html')) {
    event.respondWith(
      caches.match(BASE, { cacheName: SHELL }).then(unredirect).then(function (cached) {
        return cached || fetchClean(BASE);
      }).catch(function () { return fetchClean(BASE); })
    );
    return;
  }

  var clean = url.origin + url.pathname;
  if (SHELL_URLS.has(clean)) {
    event.respondWith(
      caches.match(clean, { cacheName: SHELL }).then(unredirect).then(function (cached) { return cached || fetchClean(req); })
    );
    return;
  }

  if (RUNTIME_PATHS.test(rel)) {
    event.respondWith(staleWhileRevalidate(req));
  }
});
