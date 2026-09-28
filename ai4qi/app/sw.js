/* Ai4Qi service worker: offline app shell and data. All paths are relative to this file. */
'use strict';

// Stamped by build_app_data.py on every rebuild; a new value makes browsers install a fresh cache.
var VERSION = '832d091f1403';
var PREFIX = 'ai4qi-';
var SHELL = PREFIX + 'shell-' + VERSION;
var RUNTIME = PREFIX + 'runtime-' + VERSION;

var SHELL_FILES = [
  './', 'index.html', 'app.js', 'export.js', 'styles.css', 'manifest.webmanifest',
  'icons/icon-192.png', 'icons/icon-512.png', 'icons/icon-maskable-512.png', 'icons/apple-touch-icon-180.png',
  'data/library.json', 'data/proposed.json', 'data/cards.json', 'data/standards.json'
];
var BASE = new URL('./', self.location).href;
var SHELL_URLS = new Set(SHELL_FILES.map(function (p) { return new URL(p, BASE).href; }));
var RUNTIME_PATHS = /^(data\/audits\/[^/]+\.json|templates\/[^/]+\.csv|figures\/.+)$/;

self.addEventListener('install', function (event) {
  event.waitUntil(
    caches.open(SHELL).then(function (cache) {
      return cache.addAll(SHELL_FILES.map(function (p) { return new Request(p, { cache: 'reload' }); }));
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
      var network = fetch(request).then(function (res) {
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
    event.respondWith(fetch(req, { cache: 'no-store' }));
    return;
  }

  // Opening the app (any hash or query) gets the cached app shell.
  if (req.mode === 'navigate' && (rel === '' || rel === 'index.html')) {
    event.respondWith(
      caches.match('index.html', { cacheName: SHELL }).then(function (cached) {
        return cached || fetch(req);
      }).catch(function () { return fetch(req); })
    );
    return;
  }

  var clean = url.origin + url.pathname;
  if (SHELL_URLS.has(clean)) {
    event.respondWith(
      caches.match(clean, { cacheName: SHELL }).then(function (cached) { return cached || fetch(req); })
    );
    return;
  }

  if (RUNTIME_PATHS.test(rel)) {
    event.respondWith(staleWhileRevalidate(req));
  }
});
