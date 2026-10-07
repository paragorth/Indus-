/* If the app has not started after 15 seconds (an old offline copy in this browser, a blocked
   script), offer a one-click repair: forget the offline copy and load the site afresh. */
(function () {
  'use strict';
  setTimeout(function () {
    var box = document.querySelector('main .loading');
    if (!box) return;
    var p = document.createElement('p');
    p.className = 'boot-fix';
    p.innerHTML = 'Taking too long? <button type="button">Reload Ai4Qi</button>';
    p.querySelector('button').addEventListener('click', function () {
      var jobs = [];
      if ('serviceWorker' in navigator) jobs.push(navigator.serviceWorker.getRegistrations().then(function (rs) { return Promise.all(rs.map(function (r) { return r.unregister(); })); }));
      if (window.caches) jobs.push(caches.keys().then(function (ks) { return Promise.all(ks.filter(function (k) { return k.indexOf('ai4qi-') === 0; }).map(function (k) { return caches.delete(k); })); }));
      Promise.all(jobs).catch(function () {}).then(function () { location.reload(); });
    });
    box.parentNode.appendChild(p);
  }, 15000);
})();
