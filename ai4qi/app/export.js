/*
 * Ai4Qi export helpers: data-collection workbook (.xlsx), results deck (.pptx),
 * proposal for a supervisor (.docx) and calendar (.ics). Plain ES5 in an IIFE, no build step.
 *
 * Libraries are loaded lazily from this site's vendor/ folder (copies of the cdnjs builds):
 *   ExcelJS 4.4.0  https://cdnjs.cloudflare.com/ajax/libs/exceljs/4.4.0/exceljs.min.js
 *   JSZip 3.10.1   https://cdnjs.cloudflare.com/ajax/libs/jszip/3.10.1/jszip.min.js
 * PptxGenJS is not hosted on cdnjs, so the deck is written directly as OOXML
 * (PresentationML, bars drawn as editable shapes, speaker notes) and zipped with JSZip.
 *
 * API: window.AI4QI_EXPORT = { ready, templateXlsx, deckPptx, proposalDocx, ics }
 */
(function () {
  'use strict';

  var LIBS = {
    ExcelJS: { url: 'vendor/exceljs.min.js', name: 'the spreadsheet library (ExcelJS)' },
    JSZip: { url: 'vendor/jszip.min.js', name: 'the zip library (JSZip)' }
  };
  var loading = {};

  var C = {
    blue: '217346', ink: '0E1626', muted: '586174', border: 'E0E3EB', tint: 'E7F2EA',   // 'blue' is the sheet accent: Excel green
    mint: '7CF2C0', passText: '0B6A4A', amber: 'E0A91A', amberTint: 'FDF5E1', grey: 'B9C0CE',
    green: '16A574', white: 'FFFFFF', zebra: 'F7F8FD'
  };
  var FONT = 'Calibri';
  // Results deck palette and fonts: change the deck's look here.
  var DECK = {
    navy: '0B1F3A',       // ink and dark backgrounds
    gold: 'B8924A',       // the single accent: key number, re-audit, thin rules
    grey: '9AA5B8',       // cycle 1 / secondary
    greyOnNavy: '9AA5B8', // secondary text on navy
    navySoft: '1E3556',   // tracks and faint marks on navy
    paper: 'FFFFFF',
    muted: '44506A',      // body captions
    hairline: 'E4E6EB',
    pass: '2E7D5B',       // reserved for essential traffic-light meaning
    goldDark: 'B8924A',   // accent on dark slides (lightened if needed for contrast)
    goldText: '87652A'    // accent for small text on light slides (darkened if needed)
  };
  var DECK_FONTS = { head: 'Georgia', body: 'Calibri' };
  var DECK_BASE = JSON.parse(JSON.stringify(DECK)), FONTS_BASE = JSON.parse(JSON.stringify(DECK_FONTS));

  /* 20 designs for the results deck, so decks from different audits do not all look alike.
     Each: [name, ink (dark slides and text), accent, paper, heading font, body font, title layout, slide motif].
     Title layouts: bleed, band, frame, split, stripe, topband, underline, centered. Motifs on light slides: none, rule, bar, corner.
     Fonts are ones Office ships on Windows and Mac. Keep DECK_THEME_NAMES in app.js in step. */
  var DECK_THEMES = [
    ['Classic navy', '0B1F3A', 'B8924A', 'FFFFFF', 'Georgia', 'Calibri', 'bleed', 'none'],
    ['NHS blue', '003087', '0072CE', 'FFFFFF', 'Arial', 'Arial', 'band', 'bar'],
    ['Forest', '1B4332', 'C9981A', 'FFFFFF', 'Cambria', 'Calibri', 'frame', 'rule'],
    ['Plum and coral', '3C1642', 'D9674E', 'FFFFFF', 'Georgia', 'Segoe UI', 'split', 'corner'],
    ['Slate and coral', '2F3E46', 'D8603F', 'FAFAF8', 'Trebuchet MS', 'Trebuchet MS', 'centered', 'bar'],
    ['Teal', '004E64', 'E09A2D', 'FFFFFF', 'Century Gothic', 'Calibri', 'stripe', 'rule'],
    ['Burgundy', '5C1A2B', 'B8924A', 'FCFAF7', 'Garamond', 'Calibri', 'centered', 'none'],
    ['Charcoal and lime', '222831', '6E9A1F', 'FFFFFF', 'Segoe UI', 'Segoe UI', 'band', 'corner'],
    ['Ocean', '0A3D62', '1F9A96', 'FFFFFF', 'Palatino Linotype', 'Calibri', 'split', 'rule'],
    ['Graphite and amber', '2B2D42', 'E0873A', 'FFFFFF', 'Franklin Gothic Medium', 'Calibri', 'topband', 'corner'],
    ['Emerald', '064E3B', '0E9F6E', 'FFFFFF', 'Constantia', 'Corbel', 'stripe', 'bar'],
    ['Indigo', '1E1B4B', 'D98E04', 'FFFFFF', 'Cambria', 'Candara', 'band', 'rule'],
    ['Terracotta', '3D2B1F', 'C0502F', 'FCF9F5', 'Rockwell', 'Calibri', 'topband', 'bar'],
    ['Midnight and rose', '14213D', 'C96F72', 'FFFFFF', 'Georgia', 'Verdana', 'topband', 'none'],
    ['Pine and sand', '2D3A3A', 'A8894F', 'FBFAF6', 'Book Antiqua', 'Calibri', 'underline', 'rule'],
    ['Royal', '26215C', 'C79214', 'FFFFFF', 'Palatino Linotype', 'Segoe UI', 'stripe', 'corner'],
    ['Steel and cyan', '263238', '0093A8', 'FFFFFF', 'Segoe UI', 'Segoe UI', 'frame', 'rule'],
    ['Aubergine and mint', '2E1A47', '1F9E6E', 'FFFFFF', 'Trebuchet MS', 'Trebuchet MS', 'band', 'bar'],
    ['Oxford', '002147', 'B8962E', 'FFFFFF', 'Garamond', 'Gill Sans MT', 'split', 'rule'],
    ['Sage', '283618', '7A8F4E', 'FAFAF5', 'Cambria', 'Calibri', 'underline', 'none']
  ];
  function hexMix(a, b, t) {                 // t = share of b
    var r = [0, 2, 4].map(function (i) {
      var x = parseInt(a.substr(i, 2), 16), y = parseInt(b.substr(i, 2), 16);
      return ('0' + Math.round(x + (y - x) * t).toString(16)).slice(-2);
    });
    return r.join('').toUpperCase();
  }
  function lum(h) {
    return [0, 2, 4].map(function (i) { var c = parseInt(h.substr(i, 2), 16) / 255; return c <= 0.03928 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4); })
      .reduce(function (a, c, i) { return a + c * [0.2126, 0.7152, 0.0722][i]; }, 0);
  }
  function contrast(a, b) { var x = lum(a), y = lum(b); return (Math.max(x, y) + 0.05) / (Math.min(x, y) + 0.05); }
  function towards(c, to, bg, min) { var t = 0, out = c; while (contrast(out, bg) < min && t < 1) { t += 0.08; out = hexMix(c, to, t); } return out; }
  function applyDeckTheme(i) {
    var t = DECK_THEMES[((i % DECK_THEMES.length) + DECK_THEMES.length) % DECK_THEMES.length];
    var ink = t[1], acc = t[2], paper = t[3];
    Object.assign(DECK, {
      navy: ink, gold: acc, paper: paper,
      grey: hexMix(ink, 'FFFFFF', 0.52), greyOnNavy: hexMix(ink, 'FFFFFF', 0.62), navySoft: hexMix(ink, 'FFFFFF', 0.16),
      muted: hexMix(ink, 'FFFFFF', 0.28), hairline: hexMix(ink, 'FFFFFF', 0.88),
      series: [acc, ink, hexMix(ink, 'FFFFFF', 0.5), hexMix(acc, 'FFFFFF', 0.45), hexMix(ink, acc, 0.5), hexMix(ink, 'FFFFFF', 0.78)],
      titleStyle: t[6], motif: t[7], themeName: t[0],
      goldDark: towards(acc, 'FFFFFF', ink, 3.4), goldText: towards(acc, '000000', paper, 4.5)
    });
    DECK_FONTS.head = t[4]; DECK_FONTS.body = t[5];
    return t[0];
  }
  function resetDeckTheme() { Object.keys(DECK).forEach(function (k) { delete DECK[k]; }); Object.assign(DECK, DECK_BASE); Object.assign(DECK_FONTS, FONTS_BASE); }
  function themeFor(run) {                   // the same audit keeps the same design unless one is chosen
    if (run && isFinite(run.deckTheme) && run.deckTheme !== null && run.deckTheme !== '') return +run.deckTheme;
    var k = String((run && (run.id || run.auditId)) || ''), h = 0;
    for (var i = 0; i < k.length; i++) h = (h * 31 + k.charCodeAt(i)) >>> 0;
    return h % DECK_THEMES.length;
  }
  var LOGO_SVG = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 48 48"><rect width="48" height="48" rx="11" fill="#0B1F3A"/><circle cx="20.5" cy="23" r="11.8" fill="none" stroke="#fff" stroke-width="3.2"/><path d="M26.6 30.2l5.8 6.6" stroke="#C9A45C" stroke-width="3.4" stroke-linecap="round"/><path d="M17.65 24.9V24.02H22.89V24.9ZM25.08 27.77 26.28 28.19V28.6H21.93V28.19L23.27 27.79L19.78 19.37H20.23L16.97 27.79L18.24 28.19V28.6H14.75V28.19L15.95 27.79L19.92 18H20.95Z" fill="#fff"/><rect x="37" y="26" width="3.2" height="10.6" rx="1.6" fill="#fff"/><circle cx="38.6" cy="21.4" r="2.2" fill="#C9A45C"/></svg>';

  /* ------------------------------------------------------------------ */
  /* Library loading                                                     */
  /* ------------------------------------------------------------------ */

  function loadLib(key) {
    if (window[key]) return Promise.resolve(window[key]);
    if (loading[key]) return loading[key];
    var lib = LIBS[key];
    loading[key] = new Promise(function (resolve, reject) {
      var s = document.createElement('script');
      s.src = lib.url;
      s.async = true;
      s.crossOrigin = 'anonymous';
      var timer = setTimeout(function () { fail('it took too long to download'); }, 30000);
      function fail(why) {
        clearTimeout(timer);
        delete loading[key];
        if (s.parentNode) s.parentNode.removeChild(s);
        reject(new Error('Could not load ' + lib.name + ' because ' + why +
          '. Check your internet connection, then try again.'));
      }
      s.onload = function () {
        clearTimeout(timer);
        if (window[key]) resolve(window[key]);
        else fail('the file downloaded but did not start correctly');
      };
      s.onerror = function () { fail('the download failed'); };
      (document.head || document.documentElement).appendChild(s);
    });
    return loading[key];
  }

  function ready() {
    return Promise.all([loadLib('ExcelJS'), loadLib('JSZip')]).then(function () { return true; });
  }

  /* ------------------------------------------------------------------ */
  /* Small helpers                                                       */
  /* ------------------------------------------------------------------ */

  function str(v) { return v === null || v === undefined ? '' : String(v); }
  // Standard sources in plain words (see humanSource in app.js); the raw reference when app.js is absent.
  function hsrc(v) { return clean(window.AI4QI_humanSource ? window.AI4QI_humanSource(v) : v); }
  function clean(v) { return str(v).replace(/[\u0000-\u0008\u000B\u000C\u000E-\u001F\uFFFE\uFFFF]/g, '').replace(/\s+$/, ''); }
  function trunc(s, n) {
    s = clean(s);
    if (s.length <= n) return s;
    var cut = s.slice(0, n - 1);
    var sp = cut.lastIndexOf(' ');
    if (sp > n * 0.7) cut = cut.slice(0, sp);
    return cut.replace(/[\s,;:.\u2013\u2014-]+$/, '') + '\u2026';
  }
  function esc(s) {
    return clean(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
  }
  function humanLabel(field) {
    var s = str(field).replace(/_/g, ' ').replace(/\s+/g, ' ').trim();
    return s ? s.charAt(0).toUpperCase() + s.slice(1) : '';
  }
  var MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
  function parseYmd(s) {
    var m = /^(\d{4})-(\d{2})-(\d{2})/.exec(str(s));
    if (!m) return null;
    var y = +m[1], mo = +m[2], d = +m[3];
    var dt = new Date(Date.UTC(y, mo - 1, d));
    if (dt.getUTCFullYear() !== y || dt.getUTCMonth() !== mo - 1 || dt.getUTCDate() !== d) return null;
    return dt;
  }
  function fmtDate(s) {
    var d = s instanceof Date ? s : parseYmd(s);
    if (!d) return '';
    return (s instanceof Date ? d.getDate() : d.getUTCDate()) + ' ' +
      MONTHS[s instanceof Date ? d.getMonth() : d.getUTCMonth()] + ' ' +
      (s instanceof Date ? d.getFullYear() : d.getUTCFullYear());
  }
  function isNum(v) { return typeof v === 'number' && isFinite(v); }
  function pctText(p) {
    if (!isNum(p)) return 'No data yet';
    return Math.round(p) + '%';
  }
  function pointsText(d) {
    var r = Math.round(Math.abs(d) * 10) / 10;
    var s = r % 1 === 0 ? String(r) : r.toFixed(1);
    return s + (r === 1 ? ' percentage point' : ' percentage points');
  }
  function colLetter(n) {
    var s = '';
    while (n > 0) { var m = (n - 1) % 26; s = String.fromCharCode(65 + m) + s; n = Math.floor((n - 1) / 26); }
    return s;
  }
  function withLibError(p) {
    return p.catch(function (e) {
      throw (e instanceof Error ? e : new Error('The export failed: ' + str(e)));
    });
  }

  /* ------------------------------------------------------------------ */
  /* 1. Data-collection workbook                                         */
  /* ------------------------------------------------------------------ */

  function estLines(text, widthChars, factor) {
    var lines = 0;
    str(text).split('\n').forEach(function (p) {
      lines += Math.max(1, Math.ceil((p.length * (factor || 1)) / Math.max(8, widthChars)));
    });
    return lines;
  }

  /* ---- Results sheet (live formulas) ---- */

  // Same rule as passField() in app.js.
  function pickPassField(fields) {
    var t = fields.filter(function (x) { return x.type !== 'cycle'; });
    return t.filter(function (x) { return /^(pass|met_standard|meets_standard|compliant|standard_met)$/i.test(x.field); })[0] ||
      t.filter(function (x) { return /pass|compliant|met/i.test(x.field) && /yes/i.test(x.type); })[0] ||
      t.filter(function (x) { return /yes/i.test(x.type); }).slice(-1)[0] || null;
  }
  function parseTargetFraction(t) {
    var m = /([\u2265\u2264<>]?)\s*(\d+(?:\.\d+)?)\s*%/.exec(str(t));
    if (!m) return null;
    var op = m[1] === '\u2264' ? '<=' : m[1] === '<' ? '<' : m[1] === '>' ? '>' : '>=';
    return { value: +m[2] / 100, op: op };
  }
  function codeSafe(s) { return clean(s).replace(/[|;=,]/g, ' ').replace(/\s+/g, ' ').trim(); }
  function xlStr(s) { return '"' + str(s).replace(/"/g, '""') + '"'; }

  function buildResultsSheet(rs, p, fields, FIRST, LAST) {
    function rng(i) { var L = colLetter(i + 1); return 'Data!$' + L + '$' + FIRST + ':$' + L + '$' + LAST; }
    var cyc = rng(0);
    var CYC = ['((' + cyc + '="Cycle 1")+(' + cyc + '=""))', '(' + cyc + '="Re-audit")'];
    var passF = pickPassField(fields);
    var passIdx = passF ? fields.indexOf(passF) : -1;
    var keyIdx = 1;
    var target = parseTargetFraction(p.target);
    var thin = { style: 'thin', color: { argb: 'FF' + C.border } };
    var box = { top: thin, left: thin, bottom: thin, right: thin };
    var fontInk = { name: FONT, size: 11, color: { argb: 'FF' + C.ink } };
    var fill = function (c) { return { type: 'pattern', pattern: 'solid', fgColor: { argb: 'FF' + c } }; };

    rs.getColumn(1).width = 34;
    rs.getColumn(2).width = 18;
    rs.getColumn(3).width = 18;
    rs.getColumn(4).width = 4;
    rs.getColumn(5).width = 40;

    rs.mergeCells('A1:E1');
    var t = rs.getCell('A1');
    t.value = 'Results' + (p.id ? ' \u00b7 ' + clean(p.id) : '');
    t.font = { name: FONT, size: 16, bold: true, color: { argb: 'FFFFFFFF' } };
    t.fill = fill(C.blue);
    t.alignment = { vertical: 'middle', indent: 1 };
    rs.getRow(1).height = 30;
    rs.mergeCells('A2:E2');
    var q = rs.getCell('A2');
    q.value = (clean(p.question) || '') + '  (Updates automatically from the Data sheet.)';
    q.font = { name: FONT, size: 10, italic: true, color: { argb: 'FF' + C.muted } };
    q.alignment = { wrapText: true, vertical: 'top', indent: 1 };
    rs.getRow(2).height = Math.max(18, estLines(q.value, 110, 1) * 13 + 6);

    function header(row, label) {
      var cells = [label, 'Cycle 1', 'Re-audit'];
      cells.forEach(function (v, k) {
        var c = rs.getCell(row, k + 1);
        c.value = v;
        c.font = { name: FONT, size: 11, bold: true, color: { argb: 'FFFFFFFF' } };
        c.fill = fill(C.blue);
        c.alignment = { vertical: 'middle', horizontal: k ? 'center' : 'left', indent: k ? 0 : 1 };
        c.border = box;
      });
      rs.getRow(row).height = 22;
    }
    function labelCell(row, text, bold) {
      var c = rs.getCell(row, 1);
      c.value = text;
      c.font = { name: FONT, size: 11, bold: !!bold, color: { argb: 'FF' + C.ink } };
      c.alignment = { vertical: 'middle', indent: 1, wrapText: true };
      c.border = box;
    }
    function valCell(row, col, formula, fmt, font) {
      var c = rs.getCell(row, col);
      c.value = { formula: formula };
      c.numFmt = fmt || '0';
      c.font = font || fontInk;
      c.alignment = { vertical: 'middle', horizontal: 'center' };
      c.border = box;
      return c;
    }

    var R = { rec: 5, met: 6, not: 7, na: 8, pct: 9, target: 10, tmet: 11, diff: 12 };
    header(4, 'Measure');
    var anyRow = passIdx >= 0 ? '(((' + rng(keyIdx) + '<>"")+(' + rng(passIdx) + '<>""))>0)' : '(' + rng(keyIdx) + '<>"")';
    labelCell(R.rec, 'Records entered');
    labelCell(R.met, 'Met the standard (Yes)' + (passF ? '' : ' \u2013 no pass field'));
    labelCell(R.not, 'Did not meet (No)');
    labelCell(R.na, 'Not applicable (N/A)');
    labelCell(R.pct, '% met  (Yes \u00f7 (Yes + No))', true);
    labelCell(R.target, 'Target');
    labelCell(R.tmet, 'Target met?');
    labelCell(R.diff, 'Change, Re-audit minus Cycle 1 (percentage points)');
    [0, 1].forEach(function (k) {
      var col = k + 2, L = colLetter(col);
      valCell(R.rec, col, 'SUMPRODUCT(' + anyRow + '*' + CYC[k] + ')');
      if (passIdx >= 0) {
        valCell(R.met, col, 'SUMPRODUCT((' + rng(passIdx) + '="Yes")*' + CYC[k] + ')');
        valCell(R.not, col, 'SUMPRODUCT((' + rng(passIdx) + '="No")*' + CYC[k] + ')');
        valCell(R.na, col, 'SUMPRODUCT((' + rng(passIdx) + '="N/A")*' + CYC[k] + ')');
      } else {
        [R.met, R.not, R.na].forEach(function (r) { valCell(r, col, '0'); });
      }
      valCell(R.pct, col, 'IF((' + L + R.met + '+' + L + R.not + ')=0,"",' + L + R.met + '/(' + L + R.met + '+' + L + R.not + '))', '0%',
        { name: FONT, size: 26, bold: true, color: { argb: 'FF' + C.blue } });
      var tc = rs.getCell(R.target, col);
      tc.value = target ? target.value : 'No target set';
      tc.numFmt = '0%';
      tc.font = fontInk; tc.alignment = { vertical: 'middle', horizontal: 'center' }; tc.border = box;
      if (target) {
        valCell(R.tmet, col, 'IF(' + L + R.pct + '="","",IF(' + L + R.pct + target.op + L + R.target + ',"Yes","No"))', '@',
          { name: FONT, size: 12, bold: true, color: { argb: 'FF' + C.ink } });
      } else {
        valCell(R.tmet, col, '""', '@');
      }
    });
    rs.getRow(R.pct).height = 40;
    rs.mergeCells(R.diff, 2, R.diff, 3);
    valCell(R.diff, 2, 'IF(OR(B' + R.pct + '="",C' + R.pct + '=""),"",ROUND((C' + R.pct + '-B' + R.pct + ')*100,1))', '+0.0;-0.0;0.0',
      { name: FONT, size: 16, bold: true, color: { argb: 'FF' + C.ink } });
    rs.getRow(R.diff).height = 30;

    rs.addConditionalFormatting({ ref: 'B' + R.pct + ':C' + R.pct, rules: [{ type: 'dataBar', priority: 1,
      cfvo: [{ type: 'num', value: 0 }, { type: 'num', value: 1 }], color: { argb: 'FF' + C.tint }, gradient: false, showValue: true }] });
    rs.addConditionalFormatting({ ref: 'B' + R.tmet + ':C' + R.tmet, rules: [
      { type: 'cellIs', operator: 'equal', formulae: ['"Yes"'], priority: 2, style: { fill: { type: 'pattern', pattern: 'solid', bgColor: { argb: 'FF' + C.mint } }, font: { color: { argb: 'FF' + C.passText }, bold: true } } },
      { type: 'cellIs', operator: 'equal', formulae: ['"No"'], priority: 3, style: { fill: { type: 'pattern', pattern: 'solid', bgColor: { argb: 'FF' + C.amberTint } }, font: { color: { argb: 'FF7A5A00' }, bold: true } } }] });
    rs.addConditionalFormatting({ ref: 'B' + R.diff, rules: [
      { type: 'cellIs', operator: 'greaterThan', formulae: ['0'], priority: 4, style: { fill: { type: 'pattern', pattern: 'solid', bgColor: { argb: 'FF' + C.mint } }, font: { color: { argb: 'FF' + C.passText }, bold: true } } }] });

    // Side notes
    rs.mergeCells('E4:E12');
    var side = rs.getCell('E4');
    side.value = 'Pass field: ' + (passF ? humanLabel(passF.field) + ' (Yes = met the standard)' : 'none found in this template') +
      '\n\nPass definition: ' + (clean(p.pass) || 'not defined') + '\n\nTarget: ' + (clean(p.target) || 'none set') +
      '\n\nN/A rows are left out of % met. Rows with a blank Cycle count as Cycle 1.';
    side.font = { name: FONT, size: 10, color: { argb: 'FF' + C.muted } };
    side.alignment = { wrapText: true, vertical: 'top' };
    side.fill = fill(C.tint);

    // Option tables
    var row = 21;
    var tables = [];
    fields.forEach(function (f, i) {
      var tt = str(f.type).toLowerCase();
      if (!((tt === 'choice' || tt === 'list' || tt === 'select') && f.options && f.options.length)) return;
      var opts = f.options.map(clean).filter(Boolean);
      header(row, humanLabel(f.field));
      var start = row + 1;
      opts.forEach(function (o, k) {
        var r = start + k;
        labelCell(r, o);
        [0, 1].forEach(function (c) {
          valCell(r, c + 2, 'SUMPRODUCT((' + rng(i) + '=' + xlStr(o) + ')*' + CYC[c] + ')');
        });
      });
      var end = start + opts.length - 1;
      rs.addConditionalFormatting({ ref: 'B' + start + ':C' + end, rules: [{ type: 'dataBar', priority: 10 + tables.length,
        cfvo: [{ type: 'num', value: 0 }, { type: 'max' }], color: { argb: 'FF' + C.grey }, gradient: false, showValue: true }] });
      tables.push({ field: f.field, opts: opts, start: start });
      row = end + 2;
    });

    rs.mergeCells('A18:E19');
    var cp = rs.getCell('A18');
    cp.value = 'Have Copilot in Excel? Try this prompt: \u201CSummarise this audit\u2019s results for a governance meeting in three sentences, using only the Results sheet.\u201D';
    cp.font = { name: FONT, size: 10, color: { argb: 'FF' + C.muted } };
    cp.alignment = { wrapText: true, vertical: 'top' };
  }

  function templateXlsx(protocol, opts) {
    opts = opts || {};
    return withLibError(loadLib('ExcelJS').then(function (ExcelJS) {
      var p = protocol || {};
      var std = p.standard || {};
      var fields = (p.template || []).filter(function (f) { return f && f.field; });
      if (!fields.length) fields = [{ field: 'audit_code', type: 'text' }];
      var tplFields = fields;
      fields = [{ field: 'cycle', type: 'cycle', note: 'Leave blank for Cycle 1. Choose Re-audit for cases in the second cycle.' }].concat(tplFields);
      var nRows = opts.rows || 100;
      var HEAD = 4, FIRST = HEAD + 1, LAST = HEAD + nRows;
      var n = fields.length;

      var wb = new ExcelJS.Workbook();
      wb.creator = '';
      wb.lastModifiedBy = '';
      wb.created = new Date();
      wb.modified = new Date();
      wb.title = trunc(p.question || 'Clinical audit data collection', 250);
      wb.subject = 'Clinical audit data collection template' + (p.id ? ' (' + p.id + ')' : '');
      wb.company = '';

      var ws = wb.addWorksheet('Data', {
        views: [{ state: 'frozen', xSplit: 0, ySplit: HEAD, topLeftCell: 'A' + FIRST, activeCell: 'A' + FIRST, showGridLines: false }],
        properties: { defaultRowHeight: 18 },
        pageSetup: { orientation: 'landscape', fitToPage: true, fitToWidth: 1, fitToHeight: 0, paperSize: 9 }
      });
      // Results goes straight after Data (filled in once Data is laid out).
      var rs = wb.addWorksheet('Results', {
        views: [{ showGridLines: false }],
        pageSetup: { orientation: 'portrait', fitToPage: true, fitToWidth: 1, fitToHeight: 0, paperSize: 9 }
      });
      var lists = null, listCol = 0;

      var thin = { style: 'thin', color: { argb: 'FF' + C.border } };
      var box = { top: thin, left: thin, bottom: thin, right: thin };

      // Column widths
      var widths = fields.map(function (f) {
        var label = humanLabel(f.field);
        var w = Math.max(12, Math.min(label.length + 4, 34));
        var t = str(f.type).toLowerCase();
        if (t === 'cycle') w = 13;
        if (t === 'date') w = Math.max(w, 14);
        if (t === 'datetime') w = Math.max(w, 18);
        if (t === 'text') w = Math.max(w, 22);
        if (t === 'choice' && f.options) {
          var ml = 0;
          f.options.forEach(function (o) { ml = Math.max(ml, str(o).length); });
          w = Math.max(w, Math.min(ml + 4, 40));
        }
        return w;
      });
      widths.forEach(function (w, i) { ws.getColumn(i + 1).width = w; });
      var totalW = widths.reduce(function (a, b) { return a + b; }, 0);

      function banner(row, text, font, fill) {
        var r = ws.getRow(row);
        if (n > 1) ws.mergeCells(row, 1, row, n);
        var c = r.getCell(1);
        c.value = text;
        c.font = font;
        c.alignment = { wrapText: true, vertical: 'middle', horizontal: 'left', indent: 1 };
        if (fill) c.fill = { type: 'pattern', pattern: 'solid', fgColor: { argb: 'FF' + fill } };
        var lines = estLines(text, totalW, (font.size || 11) / 11 * 1.1);
        r.height = Math.max(20, Math.min(160, lines * (font.size || 11) * 1.35 + 8));
      }
      // opts.code ties the sheet to one audit and one round ("NNA-074/r-ab12cd/RE"); the site reads it back on upload.
      var isRe = /\/RE$/.test(str(opts.code));
      banner(1, (isRe ? 'RE-AUDIT  \u00b7  ' : '') + (clean(p.question) || 'Clinical audit'), { name: FONT, size: 14, bold: true, color: { argb: 'FF' + C.ink } }, C.tint);
      var stdLine = 'Standard: ' + (clean(std.wording) || 'not recorded') + (std.source ? '  \u2014  ' + hsrc(std.source) : '');
      banner(2, stdLine, { name: FONT, size: 10, italic: true, color: { argb: 'FF' + C.muted } }, C.tint);
      banner(3, 'Pass: ' + (clean(p.pass) || 'not defined') + (p.target ? '   |   Target: ' + clean(p.target) : '') + (opts.code ? '   |   Sheet code: ' + opts.code : ''),
        { name: FONT, size: 10, bold: true, color: { argb: 'FF' + C.passText } }, C.tint);

      // Header row
      var hr = ws.getRow(HEAD);
      hr.height = 32;
      fields.forEach(function (f, i) {
        var c = hr.getCell(i + 1);
        c.value = humanLabel(f.field);
        c.font = { name: FONT, size: 11, bold: true, color: { argb: 'FFFFFFFF' } };
        c.fill = { type: 'pattern', pattern: 'solid', fgColor: { argb: 'FF' + C.blue } };
        c.alignment = { wrapText: true, vertical: 'middle', horizontal: 'left', indent: 1 };
        c.border = { top: thin, left: thin, right: thin, bottom: { style: 'medium', color: { argb: 'FF' + C.ink } } };
        var noteText = 'Field: ' + f.field + '\nType: ' + (f.type === 'cycle' ? 'Cycle 1 / Re-audit' : (f.type || 'text'));
        if (f.options && f.options.length) noteText += '\nOptions: ' + f.options.join(' | ');
        if (f.note) noteText += '\n' + clean(f.note);
        c.note = { texts: [{ text: noteText }], margins: { insetmode: 'auto' } };
      });
      ws.autoFilter = { from: { row: HEAD, column: 1 }, to: { row: HEAD, column: n } };

      // Validation per column
      function listFormula(f, colIdx) {
        var opts2 = (f.options || []).map(function (o) { return clean(o); }).filter(Boolean);
        var joined = opts2.join(',');
        var needsSheet = joined.length > 250 || opts2.some(function (o) { return /[,"]/.test(o); });
        if (!needsSheet) return '"' + joined + '"';
        if (!lists) {
          lists = wb.addWorksheet('Lists', { state: 'hidden' });
        }
        listCol += 1;
        var L = colLetter(listCol);
        lists.getCell(L + '1').value = f.field;
        opts2.forEach(function (o, k) { lists.getCell(L + (k + 2)).value = o; });
        return 'Lists!$' + L + '$2:$' + L + '$' + (opts2.length + 1);
      }

      var colSpec = fields.map(function (f, i) {
        var t = str(f.type).toLowerCase().replace(/\s/g, '');
        var dv = null, fmt = null, align = 'left';
        var prompt = f.note ? trunc(f.note, 250) : '';
        if (t === 'cycle') {
          dv = { type: 'list', allowBlank: true, formulae: ['"Cycle 1,Re-audit"'] };
          align = 'center';
        } else if (t === 'yes/no' || t === 'yesno' || t === 'boolean') {
          dv = { type: 'list', allowBlank: true, formulae: ['"Yes,No,N/A"'] };
          align = 'center';
        } else if ((t === 'choice' || t === 'list' || t === 'select') && f.options && f.options.length) {
          dv = { type: 'list', allowBlank: true, formulae: [listFormula(f, i + 1)] };
        } else if (t === 'date') {
          dv = { type: 'date', operator: 'greaterThan', allowBlank: true, formulae: [new Date(Date.UTC(1990, 0, 1))] };
          fmt = 'dd/mm/yyyy'; align = 'center';
        } else if (t === 'datetime' || t === 'date/time' || t === 'date-time') {
          dv = { type: 'date', operator: 'greaterThan', allowBlank: true, formulae: [new Date(Date.UTC(1990, 0, 1))] };
          fmt = 'dd/mm/yyyy hh:mm'; align = 'center';
        } else if (t === 'number' || t === 'numeric' || t === 'integer') {
          dv = { type: 'decimal', operator: 'greaterThanOrEqual', allowBlank: true, formulae: [0] };
          fmt = '0.##'; align = 'right';        } else if (/(^|_)(code|id|number|no|pseudonym\w*)(_|$)/i.test(f.field)) {
          // An audit code, never an NHS or hospital number: refuse a plain number of 7 digits or more.
          var ref = colLetter(i + 1) + FIRST;
          dv = { type: 'custom', allowBlank: true, formulae: ['NOT(AND(ISNUMBER(--SUBSTITUTE(' + ref + '," ","")),LEN(SUBSTITUTE(' + ref + '," ",""))>=7))'] };
          dv.idWarning = true;
        }
        if (dv) {
          dv.showErrorMessage = true;
          dv.errorStyle = 'stop';
          dv.errorTitle = humanLabel(f.field);
          if (dv.type === 'list') dv.error = 'Please choose a value from the list.';
          else if (dv.type === 'date') dv.error = 'Please enter a date as dd/mm/yyyy' + (fmt === 'dd/mm/yyyy hh:mm' ? ' hh:mm' : '') + '.';
          else if (dv.idWarning) { dv.error = 'This looks like an NHS or hospital number. Use a local audit code such as P001, and keep the key to the codes separately.'; delete dv.idWarning; }
          else dv.error = 'Please enter a number of 0 or more.';
          if (prompt) { dv.showInputMessage = true; dv.promptTitle = trunc(humanLabel(f.field), 32); dv.prompt = prompt; }
        }
        return { dv: dv, fmt: fmt, align: align };
      });

      // One validation per column range. (Setting it cell by cell makes ExcelJS 4.4
      // write overlapping ranges, which Excel reports as damaged content.)
      var rangeDv = !!(ws.dataValidations && typeof ws.dataValidations.add === 'function');
      if (rangeDv) {
        colSpec.forEach(function (spec, i) {
          if (spec.dv) ws.dataValidations.add(colLetter(i + 1) + FIRST + ':' + colLetter(i + 1) + LAST, spec.dv);
        });
      }
      for (var r = FIRST; r <= LAST; r++) {
        var row = ws.getRow(r);
        row.height = 18;
        var zebra = (r - FIRST) % 2 === 1;
        for (var ci = 0; ci < n; ci++) {
          var cell = row.getCell(ci + 1);
          var spec = colSpec[ci];
          cell.border = box;
          cell.font = { name: FONT, size: 11, color: { argb: 'FF' + C.ink } };
          cell.alignment = { vertical: 'middle', horizontal: spec.align };
          if (zebra) cell.fill = { type: 'pattern', pattern: 'solid', fgColor: { argb: 'FF' + C.zebra } };
          if (spec.fmt) cell.numFmt = spec.fmt;
          if (spec.dv && !rangeDv) cell.dataValidation = spec.dv;
          if (isRe && ci === 0 && str(fields[0].type) === 'cycle') cell.value = 'Re-audit';   // re-audit sheet: every row is the re-audit
        }
      }

      /* ---- How to use sheet ---- */
      var hw = wb.addWorksheet('How to use', {
        views: [{ showGridLines: false }],
        pageSetup: { orientation: 'portrait', fitToPage: true, fitToWidth: 1, fitToHeight: 0, paperSize: 9 }
      });
      hw.getColumn(1).width = 24;
      hw.getColumn(2).width = 96;
      var hr0 = hw.getRow(1);
      hw.mergeCells('A1:B1');
      hr0.getCell(1).value = 'How to use this audit template';
      hr0.getCell(1).font = { name: FONT, size: 16, bold: true, color: { argb: 'FFFFFFFF' } };
      hr0.getCell(1).fill = { type: 'pattern', pattern: 'solid', fgColor: { argb: 'FF' + C.blue } };
      hr0.getCell(1).alignment = { vertical: 'middle', indent: 1 };
      hr0.height = 30;

      var rowNo = 3;
      function info(label, value, link) {
        if (!clean(value)) return;
        var rr = hw.getRow(rowNo++);
        var a = rr.getCell(1), b = rr.getCell(2);
        a.value = label;
        a.font = { name: FONT, size: 11, bold: true, color: { argb: 'FF' + C.blue } };
        a.alignment = { vertical: 'top', wrapText: true };
        if (link) {
          b.value = { text: clean(value), hyperlink: clean(value) };
          b.font = { name: FONT, size: 11, underline: true, color: { argb: 'FF' + C.blue } };
        } else {
          b.value = clean(value);
          b.font = { name: FONT, size: 11, color: { argb: 'FF' + C.ink } };
        }
        b.alignment = { vertical: 'top', wrapText: true };
        a.border = { bottom: thin }; b.border = { bottom: thin };
        rr.height = Math.max(18, estLines(value, 92, 1.05) * 15 + 5);
      }
      info('Audit ID', p.id);
      info('Audit question', p.question);
      info('Clinical area', p.area);
      info('Standard', std.wording);
      info('Standard source', hsrc(std.source));
      info('Standard URL', std.url, /^https?:\/\//i.test(str(std.url)));
      info('Pass definition', p.pass);
      info('Population', p.population);
      info('Sample', p.sample);
      info('Data source', p.data_source);
      info('Timeline', p.timeline);
      info('Target', p.target);
      info('Re-audit', p.reaudit);

      rowNo++;
      var steps = [
        'Enter one row per case on the "Data" sheet, starting under the blue header row.',
        'Set the "Cycle" column to Re-audit for second-cycle cases; leave it blank (or Cycle 1) for the first audit.',
        'The "Results" sheet updates as you type.',
        'Use the drop-down lists where offered; dates as dd/mm/yyyy (and times as hh:mm).',
        'Hover over a header to see the field name and any collection notes.',
        'Record excluded cases separately; do not add them to the pass rate.',
        'When you are done, upload this sheet in My audits for your results, charts and slides. Only its own columns are read.'
      ];
      var sh = hw.getRow(rowNo++);
      sh.getCell(1).value = 'Steps';
      sh.getCell(1).font = { name: FONT, size: 11, bold: true, color: { argb: 'FF' + C.blue } };
      steps.forEach(function (s, k) {
        var rr = hw.getRow(rowNo++);
        rr.getCell(1).value = String(k + 1) + '.';
        rr.getCell(1).alignment = { horizontal: 'right', vertical: 'top' };
        rr.getCell(1).font = { name: FONT, size: 11, color: { argb: 'FF' + C.muted } };
        rr.getCell(2).value = s;
        rr.getCell(2).font = { name: FONT, size: 11, color: { argb: 'FF' + C.ink } };
        rr.getCell(2).alignment = { wrapText: true, vertical: 'top' };
      });

      rowNo++;
      var amberLine = { style: 'medium', color: { argb: 'FF' + C.amber } };
      var dpTitle = rowNo, dpBody = rowNo + 1;
      hw.mergeCells(dpTitle, 1, dpTitle, 2);
      hw.mergeCells(dpBody, 1, dpBody, 2);
      var t1 = hw.getRow(dpTitle).getCell(1);
      t1.value = 'Data protection';
      t1.font = { name: FONT, size: 13, bold: true, color: { argb: 'FF' + C.ink } };
      t1.fill = { type: 'pattern', pattern: 'solid', fgColor: { argb: 'FF' + C.amberTint } };
      t1.alignment = { vertical: 'middle', indent: 1 };
      t1.border = { top: amberLine, left: amberLine, right: amberLine };
      hw.getRow(dpTitle).getCell(2).border = { top: amberLine, right: amberLine };
      hw.getRow(dpTitle).height = 24;
      var b1 = hw.getRow(dpBody).getCell(1);
      b1.value = 'Do not enter names, NHS or hospital numbers, dates of birth, addresses or any free text that could identify a patient. ' +
        'Use a local audit code (e.g. P001) and keep the key on a secure Trust drive.';
      b1.font = { name: FONT, size: 12, color: { argb: 'FF' + C.ink } };
      b1.fill = { type: 'pattern', pattern: 'solid', fgColor: { argb: 'FF' + C.amberTint } };
      b1.alignment = { wrapText: true, vertical: 'top', indent: 1 };
      b1.border = { left: amberLine, bottom: amberLine, right: amberLine };
      hw.getRow(dpBody).getCell(2).border = { bottom: amberLine, right: amberLine };
      hw.getRow(dpBody).height = 52;

      var foot = hw.getRow(dpBody + 2).getCell(1);
      hw.mergeCells(dpBody + 2, 1, dpBody + 2, 2);
      foot.value = 'Template prepared ' + fmtDate(new Date()) + '.';
      foot.font = { name: FONT, size: 9, italic: true, color: { argb: 'FF' + C.muted } };

      wb.views = [{ activeTab: 0, firstSheet: 0, visibility: 'visible' }];

      buildResultsSheet(rs, p, fields, FIRST, LAST);

      return Promise.resolve(rs.protect('', {
        selectLockedCells: true, selectUnlockedCells: true, formatColumns: true, formatRows: true
      })).then(function () { return wb.xlsx.writeBuffer(); }).then(function (buf) {
        return new Blob([buf], { type: 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet' });
      });
    }));
  }

  /* ------------------------------------------------------------------ */
  /* 2. Results deck (OOXML writer)                                      */
  /* ------------------------------------------------------------------ */
  /*
   * A generated PresentationML package (no PptxGenJS on the CDN): one blank layout,
   * shapes drawn directly, speaker notes on every slide. Colours and fonts come
   * from DECK / DECK_FONTS at the top of this file; the logo from LOGO_SVG.
   */

  var EMU = 914400;
  var SW = 13.333, SH = 7.5;
  var X0 = 0.9, XW = SW - 2 * X0;
  var NS_A = 'http://schemas.openxmlformats.org/drawingml/2006/main';
  var NS_R = 'http://schemas.openxmlformats.org/officeDocument/2006/relationships';
  var NS_P = 'http://schemas.openxmlformats.org/presentationml/2006/main';
  var REL = 'http://schemas.openxmlformats.org/officeDocument/2006/relationships/';
  var XMLH = '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\r\n';
  function emu(inches) { return Math.round(inches * EMU); }

  // Rough text fitting: average glyph width as a fraction of the point size
  // (Calibri ~0.5, Georgia ~0.56; bold adds a little).
  function fitText(text, w, h, maxPt, minPt, opts) {
    opts = opts || {};
    var cw = (opts.serif ? 0.56 : 0.5) + (opts.bold ? 0.05 : 0);
    var ls = opts.lineSpacing || 1.2;
    function linesAt(pt, t) {
      var cpl = Math.max(4, Math.floor((w - 0.1) / (pt * cw / 72)));
      var total = 0;
      t.split('\n').forEach(function (p) {
        var words = p.split(/\s+/), line = 0, count = 1;
        words.forEach(function (wd) {
          var L = wd.length;
          if (line === 0) { line = L; while (line > cpl) { count++; line -= cpl; } }
          else if (line + 1 + L <= cpl) line += 1 + L;
          else { count++; line = L; while (line > cpl) { count++; line -= cpl; } }
        });
        total += count;
      });
      return total;
    }
    function fits(pt, t) {
      var extra = opts.paraSpace ? (t.split('\n').length - 1) * opts.paraSpace * pt / 72 : 0;
      return linesAt(pt, t) * pt * ls / 72 + extra + 0.1 <= h;
    }
    for (var pt = maxPt; pt >= minPt; pt -= (pt > 24 ? 2 : 1)) {
      if (fits(pt, text)) return { size: pt, text: text };
    }
    var t = str(text);
    var lo = 1, hi = t.length, best = trunc(t, 20);
    while (lo <= hi) {
      var mid = (lo + hi) >> 1;
      var cand = trunc(t, mid);
      if (fits(minPt, cand)) { best = cand; lo = mid + 1; } else hi = mid - 1;
    }
    return { size: minPt, text: best };
  }

  function Slide(deck, bg) {
    this.deck = deck;
    this.bg = bg || DECK.paper;
    this.dark = this.bg === DECK.navy;
    this.shapes = [];
    this.rels = [];
    this.nextId = 2;
    this.notes = [];
  }
  Slide.prototype.rel = function (type, target, external) {
    var id = 'rId' + (this.rels.length + 2); // rId1 = layout
    this.rels.push({ id: id, type: type, target: target, external: !!external });
    return id;
  };
  Slide.prototype.note = function () {
    for (var i = 0; i < arguments.length; i++) {
      var s = clean(arguments[i]);
      if (s) this.notes.push(s);
    }
  };
  function fillXml(color) { return color ? '<a:solidFill><a:srgbClr val="' + color + '"/></a:solidFill>' : '<a:noFill/>'; }
  function lnXml(color, wPt) {
    return color ? '<a:ln w="' + Math.round((wPt || 1) * 12700) + '"><a:solidFill><a:srgbClr val="' + color + '"/></a:solidFill></a:ln>' : '<a:ln><a:noFill/></a:ln>';
  }
  function xfrm(x, y, w, h) {
    return '<a:xfrm><a:off x="' + emu(x) + '" y="' + emu(y) + '"/><a:ext cx="' + emu(Math.max(w, 0.01)) + '" cy="' + emu(Math.max(h, 0.01)) + '"/></a:xfrm>';
  }
  function runXml(r, slide) {
    var face = r.font || DECK_FONTS.body;
    var pr = '<a:rPr lang="en-GB" sz="' + Math.round((r.size || 14) * 100) + '"' +
      (r.bold ? ' b="1"' : ' b="0"') + (r.italic ? ' i="1"' : '') + (r.link ? ' u="sng"' : '') +
      (r.spc ? ' spc="' + r.spc + '"' : '') + ' dirty="0">' +
      fillXml(r.color || DECK.navy) +
      '<a:latin typeface="' + face + '"/><a:cs typeface="' + face + '"/>';
    if (r.link && slide) pr += '<a:hlinkClick r:id="' + slide.rel(REL + 'hyperlink', r.link, true) + '"/>';
    pr += '</a:rPr>';
    return '<a:r>' + pr + '<a:t>' + esc(r.text) + '</a:t></a:r>';
  }
  function paraXml(p, slide) {
    var ppr = '<a:pPr algn="' + (p.align || 'l') + '" marL="0" indent="0">' +
      '<a:lnSpc><a:spcPct val="' + Math.round((p.lineSpacing || 1) * 100000) + '"/></a:lnSpc>' +
      '<a:spcBef><a:spcPts val="0"/></a:spcBef>' +
      '<a:spcAft><a:spcPts val="' + Math.round((p.spaceAfter || 0) * 100) + '"/></a:spcAft><a:buNone/></a:pPr>';
    var runs = p.runs.map(function (r) { return runXml(r, slide); }).join('');
    var r0 = p.runs[0] || {};
    return '<a:p>' + ppr + runs + '<a:endParaRPr lang="en-GB" sz="' + Math.round((r0.size || 14) * 100) + '" dirty="0"/></a:p>';
  }
  // o: {x,y,w,h, geom, fill, line, lineW, paras, anchor, inset}
  Slide.prototype.shape = function (o) {
    var id = this.nextId++;
    var ins = emu(o.inset === undefined ? 0 : o.inset);
    var self = this;
    var body = o.paras
      ? '<p:txBody><a:bodyPr wrap="square" lIns="' + ins + '" tIns="' + ins + '" rIns="' + ins + '" bIns="' + ins +
        '" anchor="' + (o.anchor || 't') + '" rtlCol="0"><a:noAutofit/></a:bodyPr><a:lstStyle/>' +
        o.paras.map(function (p) { return paraXml(p, self); }).join('') + '</p:txBody>'
      : '<p:txBody><a:bodyPr rtlCol="0" anchor="ctr"/><a:lstStyle/><a:p><a:endParaRPr lang="en-GB" dirty="0"/></a:p></p:txBody>';
    this.shapes.push('<p:sp><p:nvSpPr><p:cNvPr id="' + id + '" name="' + (o.name || ((o.paras ? 'Text ' : 'Shape ') + id)) + '"/><p:cNvSpPr' +
      (o.paras && !o.fill ? ' txBox="1"' : '') + '/><p:nvPr/></p:nvSpPr><p:spPr>' + xfrm(o.x, o.y, o.w, o.h) +
      '<a:prstGeom prst="' + (o.geom || 'rect') + '"><a:avLst>' + Object.keys(o.adj || {}).map(function (k) {
        return '<a:gd name="' + k + '" fmla="val ' + Math.round(o.adj[k]) + '"/>';
      }).join('') + '</a:avLst></a:prstGeom>' + fillXml(o.fill) + lnXml(o.line, o.lineW) +
      '</p:spPr>' + body + '</p:sp>');
  };
  Slide.prototype.rect = function (x, y, w, h, color, geom) { this.shape({ x: x, y: y, w: w, h: h, fill: color, geom: geom }); };
  // Pie or doughnut centred on (cx, cy), radius r. slices: [{v, color}], clockwise from 12 o'clock.
  // hole (0-1) cuts a doughnut hole in the slide's background colour.
  Slide.prototype.pie = function (cx, cy, r, slices, hole) {
    var tot = 0, self = this;
    slices.forEach(function (sl) { tot += Math.max(0, sl.v || 0); });
    if (!tot) { this.shape({ x: cx - r, y: cy - r, w: 2 * r, h: 2 * r, geom: 'ellipse', fill: DECK.hairline }); }
    else {
      var a = 0;
      slices.forEach(function (sl) {
        var v = Math.max(0, sl.v || 0); if (!v) return;
        var b = a + v / tot * 360;
        if (v >= tot) self.shape({ x: cx - r, y: cy - r, w: 2 * r, h: 2 * r, geom: 'ellipse', fill: sl.color });
        else self.shape({ x: cx - r, y: cy - r, w: 2 * r, h: 2 * r, geom: 'pie', fill: sl.color, line: self.bg, lineW: 1.5,
          adj: { adj1: ((a - 90 + 360) % 360) * 60000, adj2: ((b - 90 + 360) % 360) * 60000 } });
        a = b;
      });
    }
    if (hole) this.shape({ x: cx - r * hole, y: cy - r * hole, w: 2 * r * hole, h: 2 * r * hole, geom: 'ellipse', fill: this.bg });
  };
  // Single-style text box that shrinks to fit, then truncates with an ellipsis.
  Slide.prototype.text = function (text, x, y, w, h, o) {
    o = o || {};
    var serif = o.font === DECK_FONTS.head;
    var f = fitText(text, w, h, o.size || 14, o.min || Math.min(10, o.size || 14),
      { bold: o.bold, serif: serif, lineSpacing: (o.lineSpacing || 1) * 1.2, paraSpace: o.paraSpace });
    var paras = f.text.split('\n').map(function (t) {
      return { align: o.align, lineSpacing: o.lineSpacing, spaceAfter: o.paraSpace ? o.paraSpace * f.size : 0,
        runs: [{ text: t, size: f.size, bold: o.bold, italic: o.italic, color: o.color, link: o.link, spc: o.spc, font: o.font }] };
    });
    this.shape({ x: x, y: y, w: w, h: h, paras: paras, anchor: o.anchor, inset: o.inset });
    return f;
  };
  // Mixed runs in one paragraph (no fitting; for short labels only).
  Slide.prototype.runs = function (runs, x, y, w, h, o) {
    o = o || {};
    this.shape({ x: x, y: y, w: w, h: h, anchor: o.anchor, paras: [{ align: o.align, runs: runs }] });
  };
  Slide.prototype.line = function (x1, y1, x2, y2, color, wPt, dash) {
    var id = this.nextId++;
    var flipH = x2 < x1, flipV = y2 < y1;
    this.shapes.push('<p:cxnSp><p:nvCxnSpPr><p:cNvPr id="' + id + '" name="Line ' + id + '"/><p:cNvCxnSpPr/><p:nvPr/></p:nvCxnSpPr><p:spPr>' +
      '<a:xfrm' + (flipH ? ' flipH="1"' : '') + (flipV ? ' flipV="1"' : '') + '><a:off x="' + emu(Math.min(x1, x2)) + '" y="' + emu(Math.min(y1, y2)) +
      '"/><a:ext cx="' + emu(Math.abs(x2 - x1)) + '" cy="' + emu(Math.abs(y2 - y1)) + '"/></a:xfrm><a:prstGeom prst="line"><a:avLst/></a:prstGeom>' +
      '<a:ln w="' + Math.round((wPt || 1) * 12700) + '" cap="rnd"><a:solidFill><a:srgbClr val="' + color + '"/></a:solidFill>' +
      (dash ? '<a:prstDash val="' + (dash === true ? 'dash' : dash) + '"/>' : '') + '</a:ln></p:spPr></p:cxnSp>');
  };
  Slide.prototype.logo = function (x, y, size) {
    if (this.deck.logoPng) {
      var rid = this.rel(REL + 'image', '../media/logo.png');
      var id = this.nextId++;
      this.shapes.push('<p:pic><p:nvPicPr><p:cNvPr id="' + id + '" name="Logo" descr="Ai4Qi logo"/><p:cNvPicPr><a:picLocks noChangeAspect="1"/></p:cNvPicPr><p:nvPr/></p:nvPicPr>' +
        '<p:blipFill><a:blip r:embed="' + rid + '"/><a:stretch><a:fillRect/></a:stretch></p:blipFill><p:spPr>' + xfrm(x, y, size, size) +
        '<a:prstGeom prst="rect"><a:avLst/></a:prstGeom></p:spPr></p:pic>');
      return;
    }
    // Fallback when the canvas is unavailable: a simple gold monogram.
    this.text('Ai4Qi', x, y, 1.4, size, { size: 12, bold: true, color: DECK.gold, anchor: 'ctr' });
  };
  Slide.prototype.number = function (n) {
    this.text(String(n), SW - X0 - 0.8, SH - 0.62, 0.8, 0.3, { size: 10, color: this.dark ? DECK.greyOnNavy : DECK.grey, align: 'r', anchor: 'b' });
  };
  Slide.prototype.xml = function () {
    return XMLH + '<p:sld xmlns:a="' + NS_A + '" xmlns:r="' + NS_R + '" xmlns:p="' + NS_P + '"><p:cSld><p:bg><p:bgPr>' + fillXml(this.bg) +
      '<a:effectLst/></p:bgPr></p:bg><p:spTree><p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr>' +
      '<p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr>' +
      this.shapes.join('') + '</p:spTree></p:cSld><p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr></p:sld>';
  };
  Slide.prototype.relsXml = function (index) {
    var out = XMLH + '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">' +
      '<Relationship Id="rId1" Type="' + REL + 'slideLayout" Target="../slideLayouts/slideLayout1.xml"/>';
    this.rels.forEach(function (r) {
      out += '<Relationship Id="' + r.id + '" Type="' + r.type + '" Target="' + esc(r.target) + '"' + (r.external ? ' TargetMode="External"' : '') + '/>';
    });
    out += '<Relationship Id="rId' + (this.rels.length + 2) + '" Type="' + REL + 'notesSlide" Target="../notesSlides/notesSlide' + index + '.xml"/>';
    return out + '</Relationships>';
  };
  var GRP = '<p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr><p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr>';
  Slide.prototype.notesXml = function () {
    var paras = (this.notes.length ? this.notes : ['']).map(function (t) {
      return '<a:p><a:r><a:rPr lang="en-GB" dirty="0"/><a:t>' + esc(t) + '</a:t></a:r></a:p>';
    }).join('');
    return XMLH + '<p:notes xmlns:a="' + NS_A + '" xmlns:r="' + NS_R + '" xmlns:p="' + NS_P + '"><p:cSld><p:spTree>' + GRP +
      '<p:sp><p:nvSpPr><p:cNvPr id="2" name="Slide Image Placeholder 1"/><p:cNvSpPr><a:spLocks noGrp="1" noRot="1" noChangeAspect="1"/></p:cNvSpPr><p:nvPr><p:ph type="sldImg"/></p:nvPr></p:nvSpPr><p:spPr/></p:sp>' +
      '<p:sp><p:nvSpPr><p:cNvPr id="3" name="Notes Placeholder 2"/><p:cNvSpPr><a:spLocks noGrp="1"/></p:cNvSpPr><p:nvPr><p:ph type="body" idx="1"/></p:nvPr></p:nvSpPr><p:spPr/>' +
      '<p:txBody><a:bodyPr/><a:lstStyle/>' + paras + '</p:txBody></p:sp></p:spTree></p:cSld><p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr></p:notes>';
  };

  /* ---- Package parts ---- */
  function themeXml(name) {
    return XMLH + '<a:theme xmlns:a="' + NS_A + '" name="' + name + '"><a:themeElements><a:clrScheme name="Custom">' +
      '<a:dk1><a:srgbClr val="' + DECK.navy + '"/></a:dk1><a:lt1><a:srgbClr val="FFFFFF"/></a:lt1><a:dk2><a:srgbClr val="' + DECK.muted + '"/></a:dk2><a:lt2><a:srgbClr val="' + DECK.hairline + '"/></a:lt2>' +
      '<a:accent1><a:srgbClr val="' + DECK.gold + '"/></a:accent1><a:accent2><a:srgbClr val="' + DECK.grey + '"/></a:accent2><a:accent3><a:srgbClr val="' + DECK.navy + '"/></a:accent3>' +
      '<a:accent4><a:srgbClr val="' + DECK.muted + '"/></a:accent4><a:accent5><a:srgbClr val="' + DECK.navySoft + '"/></a:accent5><a:accent6><a:srgbClr val="' + DECK.pass + '"/></a:accent6>' +
      '<a:hlink><a:srgbClr val="' + DECK.navy + '"/></a:hlink><a:folHlink><a:srgbClr val="' + DECK.muted + '"/></a:folHlink></a:clrScheme>' +
      '<a:fontScheme name="Custom"><a:majorFont><a:latin typeface="' + DECK_FONTS.head + '"/><a:ea typeface=""/><a:cs typeface=""/></a:majorFont>' +
      '<a:minorFont><a:latin typeface="' + DECK_FONTS.body + '"/><a:ea typeface=""/><a:cs typeface=""/></a:minorFont></a:fontScheme>' +
      '<a:fmtScheme name="Office"><a:fillStyleLst><a:solidFill><a:schemeClr val="phClr"/></a:solidFill><a:solidFill><a:schemeClr val="phClr"/></a:solidFill><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:fillStyleLst>' +
      '<a:lnStyleLst><a:ln w="6350"><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:ln><a:ln w="12700"><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:ln><a:ln w="19050"><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:ln></a:lnStyleLst>' +
      '<a:effectStyleLst><a:effectStyle><a:effectLst/></a:effectStyle><a:effectStyle><a:effectLst/></a:effectStyle><a:effectStyle><a:effectLst/></a:effectStyle></a:effectStyleLst>' +
      '<a:bgFillStyleLst><a:solidFill><a:schemeClr val="phClr"/></a:solidFill><a:solidFill><a:schemeClr val="phClr"/></a:solidFill><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:bgFillStyleLst>' +
      '</a:fmtScheme></a:themeElements><a:objectDefaults/><a:extraClrSchemeLst/></a:theme>';
  }
  var EMPTY_TREE = '<p:spTree>' + GRP + '</p:spTree>';
  var CLRMAP = '<p:clrMap bg1="lt1" tx1="dk1" bg2="lt2" tx2="dk2" accent1="accent1" accent2="accent2" accent3="accent3" accent4="accent4" accent5="accent5" accent6="accent6" hlink="hlink" folHlink="folHlink"/>';
  function lvl(sz) { return '<a:lvl1pPr><a:defRPr sz="' + sz + '"><a:solidFill><a:schemeClr val="tx1"/></a:solidFill><a:latin typeface="+mn-lt"/></a:defRPr></a:lvl1pPr>'; }
  function masterXml() {
    return XMLH + '<p:sldMaster xmlns:a="' + NS_A + '" xmlns:r="' + NS_R + '" xmlns:p="' + NS_P + '"><p:cSld><p:bg><p:bgRef idx="1001"><a:schemeClr val="bg1"/></p:bgRef></p:bg>' + EMPTY_TREE + '</p:cSld>' +
      CLRMAP + '<p:sldLayoutIdLst><p:sldLayoutId id="2147483649" r:id="rId1"/></p:sldLayoutIdLst>' +
      '<p:txStyles><p:titleStyle>' + lvl(3200) + '</p:titleStyle><p:bodyStyle>' + lvl(1800) + '</p:bodyStyle><p:otherStyle>' + lvl(1800) + '</p:otherStyle></p:txStyles></p:sldMaster>';
  }
  var LAYOUT = XMLH + '<p:sldLayout xmlns:a="' + NS_A + '" xmlns:r="' + NS_R + '" xmlns:p="' + NS_P + '" type="blank" preserve="1"><p:cSld name="Blank">' + EMPTY_TREE +
    '</p:cSld><p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr></p:sldLayout>';
  var NOTES_MASTER = XMLH + '<p:notesMaster xmlns:a="' + NS_A + '" xmlns:r="' + NS_R + '" xmlns:p="' + NS_P + '"><p:cSld><p:bg><p:bgRef idx="1001"><a:schemeClr val="bg1"/></p:bgRef></p:bg><p:spTree>' + GRP +
    '<p:sp><p:nvSpPr><p:cNvPr id="2" name="Slide Image Placeholder 1"/><p:cNvSpPr><a:spLocks noGrp="1" noRot="1" noChangeAspect="1"/></p:cNvSpPr><p:nvPr><p:ph type="sldImg" idx="2"/></p:nvPr></p:nvSpPr>' +
    '<p:spPr><a:xfrm><a:off x="685800" y="1143000"/><a:ext cx="5486400" cy="3086100"/></a:xfrm><a:prstGeom prst="rect"><a:avLst/></a:prstGeom><a:noFill/><a:ln w="12700"><a:solidFill><a:srgbClr val="' + DECK.grey + '"/></a:solidFill></a:ln></p:spPr></p:sp>' +
    '<p:sp><p:nvSpPr><p:cNvPr id="3" name="Notes Placeholder 2"/><p:cNvSpPr><a:spLocks noGrp="1"/></p:cNvSpPr><p:nvPr><p:ph type="body" sz="quarter" idx="3"/></p:nvPr></p:nvSpPr>' +
    '<p:spPr><a:xfrm><a:off x="685800" y="4400550"/><a:ext cx="5486400" cy="3600450"/></a:xfrm><a:prstGeom prst="rect"><a:avLst/></a:prstGeom></p:spPr>' +
    '<p:txBody><a:bodyPr vert="horz" lIns="91440" tIns="45720" rIns="91440" bIns="45720" rtlCol="0"/><a:lstStyle/><a:p><a:pPr lvl="0"/><a:r><a:rPr lang="en-GB"/><a:t>Notes</a:t></a:r></a:p></p:txBody></p:sp>' +
    '</p:spTree></p:cSld>' + CLRMAP +
    '<p:notesStyle><a:lvl1pPr marL="0" algn="l" defTabSz="914400" rtl="0" eaLnBrk="1" latinLnBrk="0" hangingPunct="1"><a:defRPr sz="1200" kern="1200"><a:solidFill><a:schemeClr val="tx1"/></a:solidFill><a:latin typeface="+mn-lt"/><a:ea typeface="+mn-ea"/><a:cs typeface="+mn-cs"/></a:defRPr></a:lvl1pPr></p:notesStyle></p:notesMaster>';
  function relsDoc(list) {
    return XMLH + '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">' + list.map(function (r) {
      return '<Relationship Id="' + r[0] + '" Type="' + r[1] + '" Target="' + r[2] + '"/>';
    }).join('') + '</Relationships>';
  }

  // Logo: LOGO_SVG rendered to PNG on an offscreen canvas (swap LOGO_SVG to rebrand).
  function logoPng() {
    return new Promise(function (resolve) {
      try {
        var img = new Image();
        var done = false;
        var t = setTimeout(function () { if (!done) { done = true; resolve(null); } }, 3000);
        img.onload = function () {
          if (done) return;
          done = true; clearTimeout(t);
          try {
            var cv = document.createElement('canvas');
            cv.width = 192; cv.height = 192;
            cv.getContext('2d').drawImage(img, 0, 0, 192, 192);
            var url = cv.toDataURL('image/png');
            resolve(/^data:image\/png;base64,/.test(url) ? url.split(',')[1] : null);
          } catch (e) { resolve(null); }
        };
        img.onerror = function () { if (!done) { done = true; clearTimeout(t); resolve(null); } };
        img.src = 'data:image/svg+xml;base64,' + btoa(LOGO_SVG);
      } catch (e) { resolve(null); }
    });
  }

  /* ---- Wording helpers ---- */
  function cmp(op, v, t) {
    switch (op) {
      case '>': return v > t;
      case '<': return v < t;
      case '\u2264': case '<=': return v <= t;
      case '=': case '==': return v === t;
      default: return v >= t;
    }
  }
  function words(s, n) {
    var w = clean(s).split(/\s+/).filter(Boolean);
    if (w.length <= n) return w.join(' ');
    return w.slice(0, n).join(' ').replace(/[,;:\u2013\u2014-]+$/, '') + '\u2026';
  }
  function firstSentence(s) {
    s = clean(s);
    var m = /^[\s\S]*?[.!?](?=\s|$)/.exec(s);
    return (m ? m[0] : s).trim();
  }
  function stripParens(s) { return clean(s).replace(/\s*\([^)]*\)/g, '').replace(/\s+([,.;])/g, '$1'); }
  function capFirst(s) { s = str(s); return s.charAt(0).toUpperCase() + s.slice(1); }
  function noStop(s) { return clean(s).replace(/[.\s]+$/, ''); }
  var FRACTIONS = [[10, 'one in ten'], [20, 'one in five'], [25, 'one in four'], [100 / 3, 'one in three'], [40, 'two in five'],
    [50, 'half'], [60, 'three in five'], [200 / 3, 'two in three'], [70, 'seven in ten'], [75, 'three in four'], [80, 'four in five'], [90, 'nine in ten']];
  function nearPhrase(p) {
    if (p >= 99.5) return 'all';
    if (p <= 0.5) return 'none';
    var best = FRACTIONS[0];
    FRACTIONS.forEach(function (f) { if (Math.abs(f[0] - p) < Math.abs(best[0] - p)) best = f; });
    var d = p - best[0];
    if (Math.abs(d) < 1) return 'about ' + best[1];
    if (d > 0) return (d < 5 ? 'just over ' : 'over ') + best[1];
    return (d > -5 ? 'almost ' : 'under ') + best[1];
  }
  function fewerThan(p) {
    for (var i = 0; i < FRACTIONS.length; i++) if (FRACTIONS[i][0] > p + 0.5) return FRACTIONS[i][1];
    return null;
  }
  function den(c) { return c ? (c.passN || 0) + (c.failN || 0) : 0; }
  function ofText(c) { return (c.passN || 0) + ' of ' + den(c); }
  function dateRange(c) {
    if (!c || !c.from) return '';
    return fmtDate(c.from) + (c.to && c.to !== c.from ? ' \u2013 ' + fmtDate(c.to) : '');
  }

  /* ---- Slides ---- */
  function buildSlides(deck, run, stats) {
    var P = (run && typeof run.protocol === 'object' && run.protocol) || {};
    var D = (run && run.details) || {};
    var std = P.standard || {};
    stats = stats || {};
    var cycles = stats.cycles || [];
    var c1 = null, c2 = null;
    cycles.forEach(function (c) { if (c && c.key === 'c1') c1 = c; else if (c && c.key === 'c2') c2 = c; });
    if (!c1 && cycles[0] && cycles[0] !== c2) c1 = cycles[0];
    if (!c2 && cycles[1] && cycles[1] !== c1) c2 = cycles[1];
    var has1 = !!(c1 && isNum(c1.pct) && den(c1) > 0);
    var has2 = !!(c2 && c2.n > 0 && isNum(c2.pct) && den(c2) > 0);
    var target = stats.target && isNum(stats.target.value) ? stats.target : null;
    var tv = target ? target.value : null;
    var tOp = target ? (target.op || '\u2265') : '';
    var targetText = (stats.target && stats.target.text) || clean(P.target) || '';
    function met(p) { return target && isNum(p) ? cmp(tOp, p, tv) : null; }
    // Met / not met doughnut with a tick at the target, the count in the middle and a key below.
    function donut(sl, cx, cy, r, c, dark) {
      var metN = c.passN || 0, notN = c.failN || 0;
      var accD = dark ? DECK.goldDark : DECK.gold;
      sl.pie(cx, cy, r, [{ v: metN, color: accD }, { v: notN, color: dark ? DECK.navySoft : DECK.hairline }], 0.64);
      sl.text(String(metN), cx - 1.1, cy - 0.55, 2.2, 0.7, { size: 34, font: DECK_FONTS.head, color: dark ? DECK.paper : DECK.navy, align: 'ctr', anchor: 'b' });
      sl.text('of ' + den(c) + ' met', cx - 1.1, cy + 0.15, 2.2, 0.35, { size: 13, color: dark ? DECK.greyOnNavy : DECK.muted, align: 'ctr' });
      if (tv !== null) {
        var th = tv / 100 * 2 * Math.PI, r1 = r * 0.6, r2 = r * 1.1;
        sl.line(cx + r1 * Math.sin(th), cy - r1 * Math.cos(th), cx + r2 * Math.sin(th), cy - r2 * Math.cos(th), dark ? DECK.paper : DECK.navy, 2);
        var lx = cx + (r2 + 0.1) * Math.sin(th), ly = cy - (r2 + 0.1) * Math.cos(th);
        sl.text('Target', lx - 0.6, ly - 0.17, 1.2, 0.34, { size: 11, bold: true, color: dark ? DECK.paper : DECK.navy, align: 'ctr', anchor: 'ctr' });
      }
      sl.runs([{ text: '\u25cf', size: 13, color: accD }, { text: ' Met ' + metN + '     ', size: 12, color: dark ? DECK.greyOnNavy : DECK.muted },
        { text: '\u25cf', size: 13, color: dark ? DECK.navySoft : DECK.hairline }, { text: ' Not met ' + notN, size: 12, color: dark ? DECK.greyOnNavy : DECK.muted }],
        cx - r, cy + r + 0.35, 2 * r, 0.32, { align: 'ctr' });
    }
    var c1Label = (c1 && c1.label) || 'Cycle 1';
    var c2Label = (c2 && c2.label) || 'Re-audit';
    var title = clean(D.title) || clean(P.question) || 'Clinical audit results';
    var today = fmtDate(new Date());
    // For cause-type fields ("reason for delay"), the no-problem option is not a cause: leave it off the slides.
    var NO_CAUSE = /^(no delay|no delays|none|nil|no reason|no problem|no issue|not applicable|n\/?a|not delayed|on time)$/i;
    var breakdowns = (stats.breakdowns || []).map(function (b) {
      if (!b || !b.cycles || !/delay|reason|barrier|cause|why|fail/i.test(b.field + ' ' + (b.label || ''))) return b;
      var c = {}; Object.keys(b.cycles).forEach(function (k) { c[k] = (b.cycles[k] || []).filter(function (o) { return !NO_CAUSE.test(String(o.option).trim()); }); });
      return Object.assign({}, b, { cycles: c, causes: true });
    }).filter(function (b) {
      return b && b.cycles && ((b.cycles.c1 || []).some(function (o) { return o.n > 0; }) || (has2 && (b.cycles.c2 || []).some(function (o) { return o.n > 0; })));
    });
    var keyIsC2 = has2;

    function add(bg, plain) {
      var s = new Slide(deck, bg); deck.slides.push(s);
      if (!plain && !s.dark) {                    // the design's small recurring mark on light slides
        if (DECK.motif === 'rule') s.rect(X0, 0.42, 1.1, 0.07, DECK.gold);
        else if (DECK.motif === 'bar') s.rect(0, 0, 0.14, SH, DECK.gold);
        else if (DECK.motif === 'corner') s.rect(SW - 0.62, 0.5, 0.22, 0.22, DECK.gold);
      }
      if (deck.slides.length > 1) s.number(deck.slides.length);
      return s;
    }
    function eyebrow(s, t, color) {
      s.text(t.toUpperCase(), X0, 0.75, 8, 0.3, { size: 11, bold: true, spc: 200, color: color || (s.dark ? DECK.greyOnNavy : DECK.muted) });
    }
    function headline(s, t, y, o) {
      o = o || {};
      return s.text(t, X0, y || 0.7, o.w || XW, o.h || 1.0, { size: o.size || 32, min: 20, font: DECK_FONTS.head,
        color: s.dark ? DECK.paper : DECK.navy, anchor: o.anchor || 't', lineSpacing: 0.95 });
    }

    /* 1. Title, in the design's layout */
    var sub = [D.site, D.department, D.lead].map(function (v) { return trunc(v, 60); }).filter(Boolean).join('  \u00b7  ');
    var ts = DECK.titleStyle || 'bleed', s;
    if (ts === 'topband') {
      s = add(DECK.paper, true);
      s.rect(0, 0, SW, 4.3, DECK.navy);
      s.text('CLINICAL AUDIT', X0, 0.9, 6, 0.3, { size: 11, bold: true, spc: 300, color: DECK.goldDark });
      s.text(title, X0, 1.3, 11.2, 2.7, { size: 40, min: 24, font: DECK_FONTS.head, color: DECK.paper, lineSpacing: 0.95, anchor: 'b' });
      s.rect(X0, 4.3, 1.4, 0.1, DECK.gold);
      if (sub) s.text(sub, X0, 4.85, 11, 0.5, { size: 16, min: 11, color: DECK.navy });
      s.text(today, X0, 5.4, 6, 0.35, { size: 12, color: DECK.muted });
    } else if (ts === 'underline') {
      s = add(DECK.paper, true);
      s.text('CLINICAL AUDIT', X0, 1.2, 6, 0.3, { size: 11, bold: true, spc: 300, color: DECK.goldText });
      s.text(title, X0, 1.65, 11.2, 3.0, { size: 44, min: 26, font: DECK_FONTS.head, color: DECK.navy, lineSpacing: 0.95, anchor: 'b' });
      s.rect(X0, 4.9, 2.4, 0.14, DECK.gold);
      if (sub) s.text(sub, X0, 5.35, 11, 0.5, { size: 16, min: 11, color: DECK.muted });
      s.text(today, X0, 5.9, 6, 0.35, { size: 12, color: DECK.muted });
    } else if (ts === 'centered') {
      s = add(DECK.navy, true);
      s.rect(SW / 2 - 0.5, 1.5, 1.0, 0.06, DECK.goldDark);
      s.text('CLINICAL AUDIT', 1.5, 1.7, SW - 3, 0.3, { size: 11, bold: true, spc: 300, color: DECK.goldDark, align: 'ctr' });
      s.text(title, 1.5, 2.1, SW - 3, 2.6, { size: 42, min: 24, font: DECK_FONTS.head, color: DECK.paper, lineSpacing: 0.95, align: 'ctr', anchor: 'ctr' });
      if (sub) s.text(sub, 1.5, 4.95, SW - 3, 0.45, { size: 16, min: 11, color: DECK.greyOnNavy, align: 'ctr' });
      s.text(today, 1.5, 5.45, SW - 3, 0.35, { size: 12, color: DECK.greyOnNavy, align: 'ctr' });
    } else if (ts === 'band' || ts === 'split') {
      s = add(DECK.paper, true);
      var bandW = ts === 'band' ? 4.2 : 0, tx0 = ts === 'band' ? bandW + 0.7 : X0, tw0 = ts === 'band' ? SW - tx0 - 0.8 : 7.6;
      if (ts === 'band') {
        s.rect(0, 0, bandW, SH, DECK.navy);
        s.text('CLINICAL AUDIT', 0.7, 1.55, bandW - 1.0, 0.3, { size: 11, bold: true, spc: 300, color: DECK.goldDark });
        s.text(today, 0.7, SH - 1.9, bandW - 1.0, 0.35, { size: 12, color: DECK.greyOnNavy });
      } else {
        var bx = SW * 0.64;
        s.rect(bx, 0, SW - bx, SH, DECK.navy);
        s.pie(bx + (SW - bx) / 2, SH / 2, 1.55, [{ v: 1, color: DECK.goldDark }], 0);
        s.shape({ x: bx + (SW - bx) / 2 - 1.24, y: SH / 2 - 1.24, w: 2.48, h: 2.48, geom: 'ellipse', fill: DECK.navy });
        var ringP = has2 ? c2.pct : has1 ? c1.pct : null;          // the headline result inside the ring
        if (ringP !== null) {
          s.text(pctText(ringP), bx + 0.3, SH / 2 - 0.62, SW - bx - 0.6, 0.9, { size: 44, font: DECK_FONTS.head, color: DECK.paper, align: 'ctr', anchor: 'b' });
          s.text(has2 ? 'at re-audit' : 'met the standard', bx + 0.3, SH / 2 + 0.3, SW - bx - 0.6, 0.35, { size: 12, color: DECK.greyOnNavy, align: 'ctr' });
        }
        s.text('CLINICAL AUDIT', X0, 1.55, 6, 0.3, { size: 11, bold: true, spc: 300, color: DECK.goldText });
        s.text(today, X0, SH - 1.0, 6, 0.35, { size: 12, color: DECK.muted });
      }
      s.text(title, tx0, 2.0, tw0, 2.9, { size: 42, min: 24, font: DECK_FONTS.head, color: DECK.navy, lineSpacing: 0.95 });
      if (sub) s.text(sub, tx0, 5.0, tw0, 0.5, { size: 16, min: 11, color: DECK.muted });
    } else {
      s = add(DECK.navy, true);
      if (ts === 'frame') s.shape({ x: 0.35, y: 0.35, w: SW - 0.7, h: SH - 0.7, line: DECK.goldDark, lineW: 1.5 });
      if (ts === 'stripe') { s.rect(0, SH - 0.32, SW, 0.32, DECK.gold); s.rect(X0, 1.3, 0.9, 0.06, DECK.goldDark); }
      s.text('CLINICAL AUDIT', X0, 1.55, 6, 0.3, { size: 11, bold: true, spc: 300, color: DECK.goldDark });
      s.text(title, X0, 2.0, 10.4, 2.6, { size: 44, min: 26, font: DECK_FONTS.head, color: DECK.paper, lineSpacing: 0.95 });
      if (sub) s.text(sub, X0, 4.85, 10.4, 0.45, { size: 16, min: 11, color: DECK.greyOnNavy });
      s.text(today, X0, 5.3, 6, 0.35, { size: 12, color: DECK.greyOnNavy });
    }
    s.note(title, clean(P.question) && clean(P.question) !== title ? 'Audit question: ' + clean(P.question) : '',
      sub ? 'Presented by: ' + sub : '', clean(D.team) ? 'Team: ' + clean(D.team) : '', clean(D.supervisor) ? 'Supervisor: ' + clean(D.supervisor) : '',
      (run && run.auditId) || P.id ? 'Audit reference: ' + ((run && run.auditId) || P.id) : '');

    /* 2. Why it matters */
    if (clean(P.why)) {
      s = add();
      eyebrow(s, 'Why it matters');
      var why = firstSentence(P.why);
      if (why.split(/\s+/).length > 32) why = words(why, 30);
      s.text(why, X0, 1.5, 9.6, 3.6, { size: 34, min: 20, font: DECK_FONTS.head, color: DECK.navy, lineSpacing: 1.0, anchor: 'ctr' });
      if (clean(P.question)) {
        s.text('The question', X0, 5.55, 3, 0.3, { size: 11, bold: true, color: DECK.goldText });
        s.text(words(P.question, 35), X0, 5.85, 9.6, 0.75, { size: 14, min: 11, color: DECK.muted });
      }
      s.note(clean(P.why), clean(P.question) ? 'Audit question: ' + clean(P.question) : '');
    }

    /* 3. The standard */
    s = add();
    eyebrow(s, 'The standard');
    var qW = target ? 8.0 : XW - 0.4;
    s.line(X0, 1.6, X0, 4.9, DECK.gold, 1.25);
    var quote = noStop(std.wording) || 'Standard wording not recorded';
    if (quote.split(/\s+/).length > 45) quote = words(quote, 40);
    s.text('\u201c' + quote + '\u201d', X0 + 0.4, 1.5, qW, 3.5,
      { size: 30, min: 18, font: DECK_FONTS.head, italic: true, color: DECK.navy, lineSpacing: 1.02, anchor: 'ctr' });
    if (clean(std.source)) s.text(hsrc(std.source), X0 + 0.4, 5.3, qW, 0.7, { size: 12, min: 9, color: DECK.muted });
    if (/^https?:\/\//i.test(str(std.url))) s.text(trunc(std.url, 110), X0 + 0.4, 6.05, qW, 0.3, { size: 10, min: 8, color: DECK.grey, link: clean(std.url) });
    if (target) {
      var tx = X0 + 0.4 + qW + 0.5;
      s.text('Target', tx, 2.05, SW - X0 - tx, 0.3, { size: 11, bold: true, color: DECK.muted, spc: 200 });
      s.text(tv + '%', tx, 2.35, SW - X0 - tx, 1.3, { size: 80, min: 48, font: DECK_FONTS.head, color: DECK.gold });
      var rest = clean(targetText.replace(/^[^%]*%/, ''));
      if (rest) s.text(rest, tx, 3.7, SW - X0 - tx, 0.8, { size: 16, min: 11, color: DECK.navy });
    }
    s.note('Standard: ' + (clean(std.wording) || 'not recorded'), clean(std.source) ? 'Source: ' + clean(std.source) : '',
      clean(std.url) ? 'Link: ' + clean(std.url) : '', targetText ? 'Target: ' + targetText : '');

    /* 4. How we measured (2x2) */
    s = add();
    headline(s, 'How we measured', 0.7, { size: 32 });
    var howMany;
    if (c1 && c1.n > 0) {
      howMany = c1.n + ' cases in ' + c1Label.toLowerCase() + (c2 && c2.n > 0 ? ', ' + c2.n + ' at ' + c2Label.toLowerCase() : '');
      var dr = dateRange(c1);
      if (dr && !(c2 && c2.n > 0)) howMany += ', ' + dr;
    } else howMany = words(stripParens(firstSentence(P.sample)), 12) || 'Not yet collected';
    var dsrc = clean(P.data_source).split(/;\s*/).map(stripParens).filter(Boolean).slice(0, 3).join('; ');
    var facts = [
      ['Who', words(stripParens(firstSentence(P.population)), 12) || 'Not recorded'],
      ['How many', words(howMany, 12)],
      ['Where from', words(dsrc, 12) || 'Not recorded'],
      ['What counts as a pass', words(stripParens(firstSentence(P.pass)), 12) || 'Not defined']
    ];
    var gx = [X0, X0 + XW / 2 + 0.25], gy = [2.15, 4.35], cw2 = XW / 2 - 0.25;
    s.line(X0, gy[0] - 0.2, SW - X0, gy[0] - 0.2, DECK.hairline, 0.75);
    s.line(X0, gy[1] - 0.2, SW - X0, gy[1] - 0.2, DECK.hairline, 0.75);
    facts.forEach(function (f, k) {
      var x = gx[k % 2], y = gy[Math.floor(k / 2)];
      s.text(f[0].toUpperCase(), x, y, cw2, 0.3, { size: 11, bold: true, spc: 200, color: DECK.goldText });
      s.text(f[1], x, y + 0.4, cw2 - 0.3, 1.3, { size: 22, min: 14, color: DECK.navy, lineSpacing: 1.05 });
    });
    s.note('Population: ' + (clean(P.population) || 'not recorded'), 'Sample: ' + (clean(P.sample) || 'not recorded'),
      'Data source: ' + (clean(P.data_source) || 'not recorded'), 'Pass definition: ' + (clean(P.pass) || 'not defined'));
    cycles.forEach(function (c) {
      if (c && c.n > 0) s.note((c.label || c.key) + ': ' + c.n + ' records audited' + (dateRange(c) ? ', ' + dateRange(c) : '') + '.');
    });

    /* 5. What we found (cycle 1) */
    if (has1) {
      var dark1 = !keyIsC2;
      s = add(dark1 ? DECK.navy : DECK.paper);
      var p1 = c1.pct, m1 = met(p1), near1 = nearPhrase(p1), head1;
      if (m1 === null) head1 = capFirst(near1) + ' met the standard';
      else if (m1) head1 = capFirst(near1) + ' met the standard \u2014 target reached';
      else if (tv - p1 <= 10) head1 = 'Close, but short: ' + near1 + ' met the standard';
      else head1 = fewerThan(p1) ? 'Fewer than ' + fewerThan(p1) + ' met the standard' : capFirst(near1) + ' met the standard';
      eyebrow(s, 'What we found' + (has2 ? ' \u00b7 ' + c1Label : ''), dark1 ? DECK.greyOnNavy : DECK.muted);
      s.text(pctText(p1), X0 - 0.08, 1.35, 6.4, 2.4, { size: 140, min: 90, font: DECK_FONTS.head, color: dark1 ? DECK.goldDark : DECK.navy, anchor: 'ctr' });
      s.text(head1, X0, 3.85, 6.6, 1.6, { size: 30, min: 18, font: DECK_FONTS.head, color: dark1 ? DECK.paper : DECK.navy, lineSpacing: 1.0 });
      s.text(c1.n + ' audited' + (dateRange(c1) ? '  \u00b7  ' + dateRange(c1) : '') + (tv !== null ? '  \u00b7  target ' + tOp + ' ' + tv + '%' : ''),
        X0, 5.75, 6.6, 0.35, { size: 12, color: dark1 ? DECK.greyOnNavy : DECK.muted });
      donut(s, SW - X0 - 2.15, 3.35, 1.95, c1, dark1);
      s.note(head1 + '.', c1Label + ': ' + ofText(c1) + ' (' + (Math.round(p1 * 10) / 10) + '%) met the standard; records audited: ' + c1.n +
        (den(c1) < c1.n ? ' (' + (c1.n - den(c1)) + ' not applicable or blank, excluded from the percentage)' : '') + '.',
        targetText ? 'Target: ' + targetText + (m1 === null ? '' : m1 ? ' \u2014 met.' : ' \u2014 not met.') : '');
    }

    /* 6. Why it happened (small multiples) */
    if (breakdowns.length) {
      s = add();
      var causeLike = breakdowns.some(function (b) { return /delay|reason|barrier|cause|why|fail/i.test(b.field + ' ' + (b.label || '')); });
      headline(s, causeLike ? 'What got in the way' : 'How the cases break down', 0.7, { size: 32 });
      var shown = breakdowns.slice(0, 3);
      var anyC2 = has2 && shown.some(function (b) { return (b.cycles.c2 || []).some(function (o) { return o.n > 0; }); });
      if (anyC2) {
        s.runs([{ text: '\u25a0', size: 12, color: DECK.grey }, { text: ' ' + c1Label, size: 12, color: DECK.muted },
          { text: '      \u25a0', size: 12, color: DECK.gold }, { text: ' ' + c2Label, size: 12, color: DECK.muted }], X0, 1.55, 6, 0.3);
      } else {
        s.text('Top answers, ' + c1Label.toLowerCase() + (c1 && c1.n ? ' (' + c1.n + ' cases)' : ''), X0, 1.55, 8, 0.3, { size: 12, color: DECK.muted });
      }
      var k = shown.length, gap = 0.6, colW = (XW - gap * (k - 1)) / k;
      shown.forEach(function (b, bi) {
        var m1b = {}, m2b = {}, order = [];
        (b.cycles.c1 || []).forEach(function (o) { m1b[o.option] = o.n || 0; if (order.indexOf(o.option) < 0) order.push(o.option); });
        if (anyC2) (b.cycles.c2 || []).forEach(function (o) { m2b[o.option] = o.n || 0; if (order.indexOf(o.option) < 0) order.push(o.option); });
        order = order.filter(function (o) { return (m1b[o] || 0) + (m2b[o] || 0) > 0; });
        order.sort(function (a, c) { return ((m1b[c] || 0) - (m1b[a] || 0)) || ((m2b[c] || 0) - (m2b[a] || 0)); });
        var total = order.length;
        order = order.slice(0, 5);
        var mx = 1;
        order.forEach(function (o) { mx = Math.max(mx, m1b[o] || 0, m2b[o] || 0); });
        var x = X0 + bi * (colW + gap), y = 2.2;
        var wide = k === 1;
        s.text((clean(b.label) || humanLabel(b.field)).toUpperCase(), x, y, colW, 0.3, { size: 11, bold: true, spc: 200, color: DECK.goldText });
        y += 0.5;
        // A share of the whole with few answers reads best as a pie; causes and before/after pairs stay as bars.
        if (!b.causes && !anyC2 && total >= 2 && total <= 5) {
          var sum = 0; order.forEach(function (o) { sum += m1b[o] || 0; });
          var pr = wide ? 1.75 : Math.min(colW / 2 - 0.25, 1.25), pcx = wide ? x + pr + 0.2 : x + colW / 2, pcy = y + pr + 0.15;
          s.pie(pcx, pcy, pr, order.map(function (o, oi) { return { v: m1b[o] || 0, color: DECK.series[oi % DECK.series.length] }; }), 0);
          order.forEach(function (o, oi) {
            var lab = o === null || o === undefined || o === '' ? '(blank)' : trunc(capFirst(o), wide ? 40 : 24), v = m1b[o] || 0;
            var lx = wide ? x + 2 * pr + 0.9 : x, ly = wide ? pcy - order.length * 0.28 + oi * 0.56 : pcy + pr + 0.35 + oi * 0.36;
            s.runs([{ text: '\u25a0 ', size: wide ? 16 : 13, color: DECK.series[oi % DECK.series.length] },
              { text: lab + '  ', size: wide ? 16 : 12, color: DECK.navy },
              { text: v + ' (' + Math.round(v / sum * 100) + '%)', size: wide ? 16 : 12, bold: true, color: DECK.muted }], lx, ly, wide ? colW - 2 * pr - 0.9 : colW, wide ? 0.5 : 0.34);
          });
          return;
        }
        var rowH = anyC2 ? 0.78 : 0.66;
        var labW = wide ? 3.6 : colW;
        var barX = wide ? x + 3.8 : x, barW = (wide ? colW - 3.8 : colW) - 0.55;
        order.forEach(function (o, oi) {
          var ry = y + oi * rowH;
          var lab = o === null || o === undefined || o === '' ? '(blank)' : capFirst(o);
          if (wide) s.text(lab, x, ry, labW, 0.34, { size: 14, min: 10, color: DECK.navy, anchor: 'ctr' });
          else s.text(lab, x, ry, labW, 0.28, { size: 12, min: 9, color: DECK.navy });
          var by = wide ? ry + (anyC2 ? 0.02 : 0.1) : ry + 0.3;
          var bars = anyC2 ? [[m1b[o] || 0, DECK.grey], [m2b[o] || 0, DECK.gold]] : [[m1b[o] || 0, DECK.navy]];
          bars.forEach(function (bb, j) {
            var yy = by + j * 0.2, ww = barW * bb[0] / mx;
            if (bb[0] <= 0) return;
            s.rect(barX, yy, Math.max(0.03, ww), 0.14, bb[1]);
            s.text(String(bb[0]), barX + ww + 0.08, yy - 0.06, 0.5, 0.26, { size: 12, color: DECK.muted, anchor: 'ctr' });
          });
        });
        if (total > order.length) s.text('Top 5 of ' + total + ' answers', x, y + order.length * rowH + 0.05, colW, 0.3, { size: 10, color: DECK.grey });
      });
      breakdowns.forEach(function (b) {
        var fmt = function (list) { return (list || []).filter(function (o) { return o.n > 0; }).map(function (o) { return o.option + ' ' + o.n; }).join(', '); };
        s.note((clean(b.label) || humanLabel(b.field)) + ' \u2014 ' + c1Label + ': ' + (fmt(b.cycles.c1) || 'none') +
          (has2 ? '; ' + c2Label + ': ' + (fmt(b.cycles.c2) || 'none') : '') + '.');
      });
      if (breakdowns.length > 3) s.note('Only the first three breakdowns are shown on the slide.');
    }

    /* 7. What we changed */
    var cm = (run && run.changeMade) || {};
    var recorded = !!clean(cm.description);
    var changeText = recorded ? clean(cm.description) : clean(P.change);
    if (changeText) {
      s = add();
      var lbl = recorded ? (fmtDate(cm.date) ? 'Started ' + fmtDate(cm.date) : 'In place') : 'Planned';
      s.runs([{ text: 'THE CHANGE', size: 11, bold: true, spc: 200, color: DECK.muted },
        { text: '   \u00b7   ' + lbl.toUpperCase(), size: 11, bold: true, spc: 200, color: DECK.goldText }], X0, 0.75, 9, 0.3);
      var sents = changeText.match(/[^.!?]+[.!?]+(\s|$)/g) || [changeText];
      var short = clean(sents.slice(0, 2).join(' '));
      if (short.split(/\s+/).length > 40) short = words(short, 38);
      s.text(short, X0, 1.9, 10.2, 4.3, { size: 32, min: 18, font: DECK_FONTS.head, color: DECK.navy, lineSpacing: 1.02, anchor: 'ctr' });
      s.note((recorded ? 'Change made' : 'Planned change') + ': ' + changeText, recorded && fmtDate(cm.date) ? 'Introduced: ' + fmtDate(cm.date) + '.' : '');
    }

    /* 8. After the change (dark, key result) */
    var diff = has1 && has2 ? c2.pct - c1.pct : null;
    if (has2) {
      s = add(DECK.navy);
      var m2 = met(c2.pct), mBefore = has1 ? met(c1.pct) : null, head2;
      if (diff === null) head2 = capFirst(nearPhrase(c2.pct)) + ' met the standard at re-audit';
      else {
        var dp = Math.round(Math.abs(diff));
        var dir = dp === 0 ? 'No change after the intervention' : (diff > 0 ? 'Up ' : 'Down ') + dp + (dp === 1 ? ' point' : ' points');
        if (dp === 0) head2 = dir + (m2 === null ? '' : m2 ? ', target met' : ', still short of target');
        else if (m2 === null) head2 = dir + ' after the change';
        else if (m2) head2 = dir + (mBefore ? ', still above target' : ', now above target');
        else head2 = dir + ', still short of target';
      }
      eyebrow(s, 'After the change', DECK.greyOnNavy);
      s.text(head2, X0, 1.1, XW, 1.0, { size: 34, min: 22, font: DECK_FONTS.head, color: DECK.paper });
      var ny = 2.75, nh = 2.0;
      var lx = X0, rx = X0 + 7.3, nw = 3.6;
      if (has1) {
        s.text(pctText(c1.pct), lx - 0.06, ny, nw, nh, { size: 110, min: 72, font: DECK_FONTS.head, color: DECK.grey, anchor: 'ctr' });
        s.text(c1Label + '  \u00b7  ' + ofText(c1), lx, ny + nh + 0.05, nw, 0.35, { size: 13, color: DECK.greyOnNavy });
        // slope between the two figures
        var sx1 = lx + nw + 0.1, sx2 = rx - 0.35, mid = ny + nh / 2, k2 = 1.6;
        var y1 = mid + (c2.pct - c1.pct) / 100 * k2 / 2, y2 = mid - (c2.pct - c1.pct) / 100 * k2 / 2;
        s.line(sx1, y1, sx2, y2, DECK.goldDark, 1.5);
        s.shape({ x: sx1 - 0.07, y: y1 - 0.07, w: 0.14, h: 0.14, geom: 'ellipse', fill: DECK.grey });
        s.shape({ x: sx2 - 0.07, y: y2 - 0.07, w: 0.14, h: 0.14, geom: 'ellipse', fill: DECK.goldDark });
        var dtxt = (diff > 0 ? '+' : diff < 0 ? '\u2212' : '\u00b1') + (Math.round(Math.abs(diff) * 10) / 10) + ' pts';
        s.text(dtxt, (sx1 + sx2) / 2 - 1.0, Math.min(y1, y2) - 0.55, 2.0, 0.4, { size: 16, bold: true, color: DECK.goldDark, align: 'ctr' });
      }
      s.text(pctText(c2.pct), rx - 0.06, ny, nw, nh, { size: 110, min: 72, font: DECK_FONTS.head, color: DECK.goldDark, anchor: 'ctr' });
      s.text(c2Label + '  \u00b7  ' + ofText(c2), rx, ny + nh + 0.05, nw, 0.35, { size: 13, color: DECK.greyOnNavy });
      if (tv !== null) s.text('Target ' + tv + '%', X0, 6.45, 4, 0.3, { size: 12, color: DECK.greyOnNavy });
      s.note(head2 + '.', has1 ? c1Label + ': ' + ofText(c1) + ' (' + (Math.round(c1.pct * 10) / 10) + '%), ' + c1.n + ' records audited.' : '',
        c2Label + ': ' + ofText(c2) + ' (' + (Math.round(c2.pct * 10) / 10) + '%), ' + c2.n + ' records audited' + (dateRange(c2) ? ', ' + dateRange(c2) : '') + '.',
        diff !== null ? 'Change: ' + (diff >= 0 ? '+' : '') + (Math.round(diff * 10) / 10) + ' percentage points.' : '',
        targetText ? 'Target: ' + targetText + '.' : '');
    }

    // Shared axis for the charts below: 0-100% with light gridlines.
    function pctAxis(sl, x, y, w, h) {
      [0, 25, 50, 75, 100].forEach(function (g) {
        var gy = y + h - h * g / 100;
        sl.line(x, gy, x + w, gy, g === 0 ? DECK.grey : DECK.hairline, g === 0 ? 1 : 0.75);
        sl.text(g + '%', x - 0.75, gy - 0.15, 0.65, 0.3, { size: 10, color: DECK.muted, align: 'r', anchor: 'ctr' });
      });
      return function (p) { return y + h - h * Math.max(0, Math.min(100, p)) / 100; };
    }
    function targetLine(sl, x, w, yOf) {
      if (tv === null) return;
      var yT = yOf(tv);
      sl.line(x, yT, x + w, yT, DECK.navy, 1.5, 'dash');
      sl.text('Target ' + tv + '%', x + w - 1.6, yT - 0.36, 1.6, 0.3, { size: 11, bold: true, color: DECK.navy, align: 'r' });
    }

    /* 8b. Side by side: column chart against the target */
    if (has1 && has2) {
      s = add();
      headline(s, 'Before and after, side by side', 0.7, { size: 32 });
      var gx0 = X0 + 0.8, gy0 = 1.95, gw = 6.6, gh = 4.1, yOf = pctAxis(s, gx0, gy0, gw, gh);
      [[c1, DECK.grey, c1Label], [c2, DECK.gold, c2Label]].forEach(function (cc, i) {
        var bx = gx0 + 1.0 + i * 2.9, bw = 1.7, top = yOf(cc[0].pct);
        s.rect(bx, top, bw, gy0 + gh - top, cc[1]);
        s.text(pctText(cc[0].pct), bx - 0.4, top - 0.62, bw + 0.8, 0.55, { size: 26, font: DECK_FONTS.head, color: DECK.navy, align: 'ctr', anchor: 'b' });
        s.text(cc[2], bx - 0.4, gy0 + gh + 0.1, bw + 0.8, 0.3, { size: 13, bold: true, color: DECK.navy, align: 'ctr' });
        s.text(ofText(cc[0]) + ' met', bx - 0.4, gy0 + gh + 0.4, bw + 0.8, 0.3, { size: 11, color: DECK.muted, align: 'ctr' });
      });
      targetLine(s, gx0, gw, yOf);
      var px = gx0 + gw + 0.9, pw = SW - X0 - px;
      var dd = Math.round((c2.pct - c1.pct) * 10) / 10;
      [['Change', (dd > 0 ? '+' : dd < 0 ? '−' : '±') + Math.abs(dd) + ' pts'],
       ['Cases audited', c1.n + ' → ' + c2.n],
       ['Target', tv === null ? 'Not set' : (met(c2.pct) ? 'Met at re-audit' : 'Not yet met')]].forEach(function (f, i) {
        var yy = 2.0 + i * 1.35;
        s.text(f[0].toUpperCase(), px, yy, pw, 0.3, { size: 11, bold: true, spc: 200, color: DECK.goldText });
        s.text(f[1], px, yy + 0.32, pw, 0.7, { size: 26, min: 16, font: DECK_FONTS.head, color: DECK.navy });
      });
      s.note('Column chart: the share of cases meeting the standard in each cycle; the dashed line is the target.',
        c1Label + ': ' + ofText(c1) + ' (' + pctText(c1.pct) + '). ' + c2Label + ': ' + ofText(c2) + ' (' + pctText(c2.pct) + ').');
    }

    /* 8c. Month by month: a run chart, the usual QI view of change over time */
    var MON = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
    var pts = [];
    [[c1, 1], [c2, 2]].forEach(function (cc) {
      ((cc[0] && cc[0].months) || []).forEach(function (mm) {
        var d = (mm.pass || 0) + (mm.fail || 0);
        if (d > 0 && /^\d{4}-\d{2}$/.test(mm.m)) pts.push({ m: mm.m, pct: mm.pass / d * 100, n: d, cyc: cc[1] });
      });
    });
    pts.sort(function (a, b) { return a.m < b.m ? -1 : a.m > b.m ? 1 : a.cyc - b.cyc; });
    if (pts.length >= 4) {
      s = add();
      headline(s, 'Month by month', 0.7, { size: 32 });
      s.runs([{ text: '●', size: 13, color: DECK.grey }, { text: ' ' + c1Label + '     ', size: 12, color: DECK.muted },
        { text: '●', size: 13, color: DECK.gold }, { text: ' ' + c2Label + '     ', size: 12, color: DECK.muted },
        { text: '- - -', size: 12, color: DECK.muted }, { text: ' median', size: 12, color: DECK.muted }], X0, 1.5, 9, 0.3);
      var rx0 = X0 + 0.8, ry0 = 2.1, rw = SW - X0 - rx0 - 0.3, rh = 3.7, yOfR = pctAxis(s, rx0, ry0, rw, rh);
      var step = rw / pts.length, xs = pts.map(function (p, i) { return rx0 + step * (i + 0.5); });
      var sorted = pts.map(function (p) { return p.pct; }).sort(function (a, b) { return a - b; });
      var med = sorted.length % 2 ? sorted[(sorted.length - 1) / 2] : (sorted[sorted.length / 2 - 1] + sorted[sorted.length / 2]) / 2;
      s.line(rx0, yOfR(med), rx0 + rw, yOfR(med), DECK.muted, 1, 'sysDash');
      s.text('Median ' + Math.round(med) + '%', rx0, yOfR(med) - 0.34, 1.8, 0.3, { size: 10, color: DECK.muted });
      targetLine(s, rx0, rw, yOfR);
      var chg = clean(cm.date).slice(0, 7);
      if (/^\d{4}-\d{2}$/.test(chg)) {
        var ci = pts.findIndex(function (p) { return p.m >= chg; });
        if (ci > 0) {
          var cxm = (xs[ci - 1] + xs[ci]) / 2;
          s.line(cxm, ry0 - 0.1, cxm, ry0 + rh, DECK.gold, 1.25, 'dash');
          s.text('Change', cxm + 0.08, ry0 - 0.15, 1.2, 0.3, { size: 11, bold: true, color: DECK.goldText });
        }
      }
      pts.forEach(function (p, i) {
        if (i > 0) s.line(xs[i - 1], yOfR(pts[i - 1].pct), xs[i], yOfR(p.pct), DECK.navy, 1.5);
      });
      var every = pts.length > 12 ? 2 : 1;
      pts.forEach(function (p, i) {
        s.shape({ x: xs[i] - 0.09, y: yOfR(p.pct) - 0.09, w: 0.18, h: 0.18, geom: 'ellipse', fill: p.cyc === 2 ? DECK.gold : DECK.grey, line: DECK.paper, lineW: 1 });
        if (i % every === 0) s.text(MON[+p.m.slice(5) - 1] + ' ' + p.m.slice(2, 4), xs[i] - 0.5, ry0 + rh + 0.1, 1.0, 0.28, { size: 10, color: DECK.muted, align: 'ctr' });
      });
      s.note('Run chart: each point is one month’s share of cases meeting the standard; the dotted line is the median of all points.',
        'A run of six or more points on one side of the median, or five rising or falling in a row, suggests a real change rather than chance.',
        pts.map(function (p) { return MON[+p.m.slice(5) - 1] + ' ' + p.m.slice(0, 4) + ': ' + Math.round(p.pct) + '% of ' + p.n; }).join('; ') + '.');
    }

    /* 9. What next */
    var items = [];
    if (has2) {
      var mm2 = met(c2.pct);
      if (mm2 === true) items.push('Target reached \u2014 keep the change in place');
      else if (diff !== null && diff > 0) items.push('Better, but still short of the target');
      else if (diff !== null) items.push('No improvement yet \u2014 revisit how the change works');
      else items.push('Keep measuring against the standard');
    } else if (has1) {
      var mm1 = met(c1.pct);
      items.push(mm1 === null ? 'Agree a local target before re-auditing' : mm1 ? 'Standard met \u2014 keep monitoring' : 'Below target \u2014 a change is needed');
    } else items.push('Collect cycle 1 data with the data sheet');
    var top = null, topB = null;
    breakdowns.forEach(function (b) {
      if (top) return;
      (b.cycles.c1 || []).forEach(function (o) { if (o.n > 0 && (!top || o.n > top.n)) { top = o; topB = b; } });
    });
    if (top) items.push((topB.causes ? 'Commonest cause in cycle 1: ' : 'Biggest group in cycle 1: ') + words(top.option, 6) + ' (' + top.n + ')');
    if (has2 && clean(P.close_loop)) items.push(words(noStop(firstSentence(P.close_loop)), 10));
    else if (!has2) {
      var when = /^([^:;.]+)/.exec(clean(P.reaudit));
      items.push(when && /week|month|day/i.test(when[1]) ? 'Re-audit in ' + when[1].charAt(0).toLowerCase() + when[1].slice(1) + ', same template' : 'Re-audit with the same template');
    }
    items = items.slice(0, 3).map(function (t) { return words(t, 10); });
    s = add();
    headline(s, 'What next', 0.7, { size: 32 });
    var nsz = 28;
    items.forEach(function (t) { nsz = Math.min(nsz, fitText(t, 10.6, 0.8, 28, 16, {}).size); });
    items.forEach(function (t, i) {
      var y = 2.1 + i * 1.35;
      s.text(String(i + 1), X0, y - 0.12, 0.7, 0.8, { size: 44, font: DECK_FONTS.head, color: DECK.gold });
      s.text(t, X0 + 0.9, y, 10.6, 0.8, { size: nsz, min: nsz, color: DECK.navy });
    });
    s.note('Takeaways: ' + items.join('; ') + '.', clean(P.close_loop) ? 'Closing the loop: ' + clean(P.close_loop) : '',
      clean(P.reaudit) ? 'Re-audit: ' + clean(P.reaudit) : '', topB ? 'Largest single factor from "' + (clean(topB.label) || humanLabel(topB.field)) + '": ' + top.option + ' (' + top.n + ').' : '');

    /* 10. References */
    var refs = [];
    if (clean(std.source)) refs.push('Standard: ' + hsrc(std.source) + (clean(std.url) ? '. ' + clean(std.url) : ''));
    (P.evidence || []).map(clean).filter(Boolean).forEach(function (e) { refs.push(e); });
    if (refs.length) {
      s = add();
      s.text('References', X0, 0.7, 8, 0.7, { size: 24, font: DECK_FONTS.head, color: DECK.navy });
      refs = refs.slice(0, 12);
      var half = Math.ceil(refs.length / 2);
      var cols = refs.length > 3 ? [refs.slice(0, half), refs.slice(half)] : [refs];
      var rcw = cols.length > 1 ? (XW - 0.6) / 2 : XW * 0.7;
      cols.forEach(function (list, ci) {
        s.text(list.join('\n'), X0 + ci * (rcw + 0.6), 1.7, rcw, 4.5, { size: 12, min: 8, color: DECK.muted, paraSpace: 0.8, lineSpacing: 1.05 });
      });
      s.text(today + '  \u00b7  Figures are from local audit data.', X0, SH - 0.62, 9, 0.3, { size: 10, color: DECK.grey, anchor: 'b' });
      s.note('Full reference list:', refs.join('\n'));
    }
  }

  function deckPptx(run, stats) {
    var ttl = clean((run && run.details && run.details.title) || (run && run.protocol && run.protocol.question) || 'Clinical audit results');
    return packPptx(function (deck) {
      applyDeckTheme(themeFor(run));
      deck.author = clean(run && run.details && run.details.lead) || '';
      try { buildSlides(deck, run || {}, stats || {}); } finally { resetDeckTheme(); }
    }, ttl);
  }

  // Zips slides made by build(deck) into a .pptx package.
  function packPptx(build, ttl) {
    return withLibError(Promise.all([loadLib('JSZip'), null]).then(function (res) {   // no logo: the deck is the presenter's own
      var JSZip = res[0];
      var deck = { slides: [], logoPng: res[1] };
      build(deck);

      var zip = new JSZip();
      var nS = deck.slides.length;
      var ct = XMLH + '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">' +
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>' +
        '<Default Extension="xml" ContentType="application/xml"/><Default Extension="png" ContentType="image/png"/>' +
        '<Override PartName="/ppt/presentation.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.presentation.main+xml"/>' +
        '<Override PartName="/ppt/slideMasters/slideMaster1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slideMaster+xml"/>' +
        '<Override PartName="/ppt/slideLayouts/slideLayout1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slideLayout+xml"/>' +
        '<Override PartName="/ppt/notesMasters/notesMaster1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.notesMaster+xml"/>' +
        '<Override PartName="/ppt/theme/theme1.xml" ContentType="application/vnd.openxmlformats-officedocument.theme+xml"/>' +
        '<Override PartName="/ppt/theme/theme2.xml" ContentType="application/vnd.openxmlformats-officedocument.theme+xml"/>' +
        '<Override PartName="/ppt/presProps.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.presProps+xml"/>' +
        '<Override PartName="/ppt/viewProps.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.viewProps+xml"/>' +
        '<Override PartName="/ppt/tableStyles.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.tableStyles+xml"/>' +
        '<Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>' +
        '<Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>';
      for (var i = 1; i <= nS; i++) {
        ct += '<Override PartName="/ppt/slides/slide' + i + '.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slide+xml"/>';
        ct += '<Override PartName="/ppt/notesSlides/notesSlide' + i + '.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.notesSlide+xml"/>';
      }
      zip.file('[Content_Types].xml', ct + '</Types>');

      zip.file('_rels/.rels', relsDoc([
        ['rId1', REL + 'officeDocument', 'ppt/presentation.xml'],
        ['rId2', 'http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties', 'docProps/core.xml'],
        ['rId3', REL + 'extended-properties', 'docProps/app.xml']]));
      var now = new Date().toISOString().replace(/\.\d+Z$/, 'Z');
      zip.file('docProps/core.xml', XMLH + '<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:dcterms="http://purl.org/dc/terms/" xmlns:dcmitype="http://purl.org/dc/dcmitype/" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">' +
        '<dc:title>' + esc(trunc(ttl, 250)) + '</dc:title><dc:creator>' + esc(deck.author || '') + '</dc:creator><cp:lastModifiedBy>' + esc(deck.author || '') + '</cp:lastModifiedBy>' +
        '<dcterms:created xsi:type="dcterms:W3CDTF">' + now + '</dcterms:created><dcterms:modified xsi:type="dcterms:W3CDTF">' + now + '</dcterms:modified></cp:coreProperties>');
      zip.file('docProps/app.xml', XMLH + '<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties" xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">' +
        '<Slides>' + nS + '</Slides><Notes>' + nS + '</Notes><PresentationFormat>Widescreen</PresentationFormat></Properties>');

      var presRels = [['rId1', REL + 'slideMaster', 'slideMasters/slideMaster1.xml']];
      var sldIds = '';
      for (var k = 1; k <= nS; k++) {
        presRels.push(['rId' + (k + 1), REL + 'slide', 'slides/slide' + k + '.xml']);
        sldIds += '<p:sldId id="' + (255 + k) + '" r:id="rId' + (k + 1) + '"/>';
      }
      var nm = 'rId' + (nS + 2);
      presRels.push([nm, REL + 'notesMaster', 'notesMasters/notesMaster1.xml']);
      presRels.push(['rId' + (nS + 3), REL + 'theme', 'theme/theme1.xml']);
      presRels.push(['rId' + (nS + 4), REL + 'presProps', 'presProps.xml']);
      presRels.push(['rId' + (nS + 5), REL + 'viewProps', 'viewProps.xml']);
      presRels.push(['rId' + (nS + 6), REL + 'tableStyles', 'tableStyles.xml']);
      zip.file('ppt/_rels/presentation.xml.rels', relsDoc(presRels));
      zip.file('ppt/presentation.xml', XMLH + '<p:presentation xmlns:a="' + NS_A + '" xmlns:r="' + NS_R + '" xmlns:p="' + NS_P + '" saveSubsetFonts="1">' +
        '<p:sldMasterIdLst><p:sldMasterId id="2147483648" r:id="rId1"/></p:sldMasterIdLst>' +
        '<p:notesMasterIdLst><p:notesMasterId r:id="' + nm + '"/></p:notesMasterIdLst><p:sldIdLst>' + sldIds + '</p:sldIdLst>' +
        '<p:sldSz cx="12192000" cy="6858000"/><p:notesSz cx="6858000" cy="9144000"/></p:presentation>');
      zip.file('ppt/presProps.xml', XMLH + '<p:presentationPr xmlns:a="' + NS_A + '" xmlns:r="' + NS_R + '" xmlns:p="' + NS_P + '"/>');
      zip.file('ppt/viewProps.xml', XMLH + '<p:viewPr xmlns:a="' + NS_A + '" xmlns:r="' + NS_R + '" xmlns:p="' + NS_P + '"><p:normalViewPr><p:restoredLeft sz="15620"/><p:restoredTop sz="94660"/></p:normalViewPr><p:gridSpacing cx="76200" cy="76200"/></p:viewPr>');
      zip.file('ppt/tableStyles.xml', XMLH + '<a:tblStyleLst xmlns:a="' + NS_A + '" def="{5C22544A-7EE6-4342-B048-85BDC9FD1C3A}"/>');
      zip.file('ppt/theme/theme1.xml', themeXml('Office Theme'));
      zip.file('ppt/theme/theme2.xml', themeXml('Office Theme'));
      zip.file('ppt/slideMasters/slideMaster1.xml', masterXml());
      zip.file('ppt/slideMasters/_rels/slideMaster1.xml.rels', relsDoc([
        ['rId1', REL + 'slideLayout', '../slideLayouts/slideLayout1.xml'], ['rId2', REL + 'theme', '../theme/theme1.xml']]));
      zip.file('ppt/slideLayouts/slideLayout1.xml', LAYOUT);
      zip.file('ppt/slideLayouts/_rels/slideLayout1.xml.rels', relsDoc([['rId1', REL + 'slideMaster', '../slideMasters/slideMaster1.xml']]));
      zip.file('ppt/notesMasters/notesMaster1.xml', NOTES_MASTER);
      zip.file('ppt/notesMasters/_rels/notesMaster1.xml.rels', relsDoc([['rId1', REL + 'theme', '../theme/theme2.xml']]));
      if (deck.logoPng) zip.file('ppt/media/logo.png', deck.logoPng, { base64: true });
      deck.slides.forEach(function (sl, idx) {
        var n = idx + 1;
        zip.file('ppt/slides/slide' + n + '.xml', sl.xml());
        zip.file('ppt/slides/_rels/slide' + n + '.xml.rels', sl.relsXml(n));
        zip.file('ppt/notesSlides/notesSlide' + n + '.xml', sl.notesXml());
        zip.file('ppt/notesSlides/_rels/notesSlide' + n + '.xml.rels', relsDoc([
          ['rId1', REL + 'notesMaster', '../notesMasters/notesMaster1.xml'], ['rId2', REL + 'slide', '../slides/slide' + n + '.xml']]));
      });
      return zip.generateAsync({ type: 'blob', compression: 'DEFLATE',
        mimeType: 'application/vnd.openxmlformats-officedocument.presentationml.presentation' });
    }));
  }

  /* ------------------------------------------------------------------ */
  /* 2b. Proposal for a supervisor: Word (.docx)                        */
  /* ------------------------------------------------------------------ */

  // "Wk 1–2 collect; Wk 3 analyse" -> [{a:1,b:2,label:'collect'}, ...]; null if it does not parse.
  function timelineSteps(t) {
    var out = [];
    var ok = clean(t).split(/;\s*/).every(function (part) {
      var m = /^\s*(?:wk|weeks?)\s*(\d+)\s*(?:[–—-]\s*(\d+))?\s*[:,]?\s*(.+)$/i.exec(part);
      if (!m) return false;
      out.push({ a: +m[1], b: +(m[2] || m[1]), label: capFirst(clean(m[3])) });
      return true;
    });
    return ok && out.length ? out : null;
  }
  // Adds dates when a start date is known; "reminder" marks the steps Ai4Qi emails about when due
  // (the same steps as schedule() in app.js: collect, analyse/present, change, re-audit).
  function datedSteps(t, startIso) {
    var steps = timelineSteps(t), st = parseYmd(startIso);
    if (!steps) return null;
    return steps.map(function (x) {
      var o = { a: x.a, b: x.b, label: x.label, reminder: /collect|analy|present|change|re-?audit/i.test(x.label) && !/embed/i.test(x.label) };
      if (st) {
        o.from = new Date(st.getTime() + (x.a - 1) * 7 * 86400000);
        o.to = new Date(st.getTime() + (x.b * 7 - 1) * 86400000);
      }
      return o;
    });
  }
  function shortDate(d) { return d ? d.toLocaleDateString('en-GB', { day: 'numeric', month: 'short' }) : ''; }
  function weeksTotal(t) {
    var s = timelineSteps(t);
    return s ? Math.max.apply(null, s.map(function (x) { return x.b; })) : null;
  }
  function isNiceStd(st) { return /^\s*NICE\b/.test(str(st && st.source)) || /(^|\.)nice\.org\.uk\//i.test(str(st && st.url)); }
  function niceNotice(st) {
    var y = (str(st.source).match(/(19|20)\d\d(?!.*(19|20)\d\d)/) || [''])[0];
    return '© NICE ' + y + ' ' + clean(str(st.source).replace(/^NICE\s*/, '')) + '. Available from ' + (clean(st.url) || 'www.nice.org.uk') +
      '. All rights reserved. Subject to Notice of rights. NICE guidance is prepared for the National Health Service in England. ' +
      'All NICE guidance is subject to regular review and may be updated or withdrawn. NICE accepts no responsibility for the use of its content in this product.';
  }
  // Example cell for the spreadsheet preview of the template.
  function cellHint(f) {
    var t = str(f.type).toLowerCase();
    if (f.options && f.options.length) return '▾ ' + f.options.slice(0, 3).join(' / ') + (f.options.length > 3 ? ' …' : '');
    if (/yes/.test(t)) return '▾ yes / no';
    if (/datetime/.test(t)) return 'dd/mm/yyyy hh:mm';
    if (/date/.test(t)) return 'dd/mm/yyyy';
    if (/time/.test(t)) return 'hh:mm';
    if (/num|min|hour|score|count/.test(t)) return '123';
    if (/code|pseud|hospital_number|audit_code/i.test(f.field)) return 'P001';
    return 'text';
  }

  /* ---- Word ---- */
  var W_NS = 'xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" xmlns:r="' + NS_R + '"';
  function wRun(text, o) {
    o = o || {};
    var pr = (o.font ? '<w:rFonts w:ascii="' + o.font + '" w:hAnsi="' + o.font + '" w:cs="' + o.font + '"/>' : '') +
      (o.bold ? '<w:b/>' : '') + (o.italic ? '<w:i/>' : '') + (o.caps ? '<w:caps/>' : '') +
      (o.color ? '<w:color w:val="' + o.color + '"/>' : '') + (o.spacing ? '<w:spacing w:val="' + o.spacing + '"/>' : '') +
      (o.size ? '<w:sz w:val="' + Math.round(o.size * 2) + '"/><w:szCs w:val="' + Math.round(o.size * 2) + '"/>' : '') +
      (o.link ? '<w:u w:val="single"/>' : '');
    return str(text).split('\n').map(function (line, i) {
      return (i ? '<w:r><w:br/></w:r>' : '') + '<w:r>' + (pr ? '<w:rPr>' + pr + '</w:rPr>' : '') + '<w:t xml:space="preserve">' + esc(line) + '</w:t></w:r>';
    }).join('');
  }
  function wPara(runs, o) {
    o = o || {};
    var pr = (o.style ? '<w:pStyle w:val="' + o.style + '"/>' : '') + (o.keepNext ? '<w:keepNext/>' : '') +
      (o.border ? '<w:pBdr><w:left w:val="single" w:sz="18" w:space="12" w:color="' + o.border + '"/></w:pBdr>' : '') +
      (o.shade ? '<w:shd w:val="clear" w:color="auto" w:fill="' + o.shade + '"/>' : '') +
      '<w:spacing w:before="' + (o.before || 0) + '" w:after="' + (o.after === undefined ? 120 : o.after) + '"' + (o.line ? ' w:line="' + o.line + '" w:lineRule="auto"' : '') + '/>' +
      (o.indent ? '<w:ind w:left="' + o.indent + '"' + (o.hanging ? ' w:hanging="' + o.hanging + '"' : '') + '/>' : '') +
      (o.align ? '<w:jc w:val="' + o.align + '"/>' : '');
    return '<w:p><w:pPr>' + pr + '</w:pPr>' + (typeof runs === 'string' ? runs : runs.join('')) + '</w:p>';
  }
  function wCell(content, o) {
    o = o || {};
    return '<w:tc><w:tcPr><w:tcW w:w="' + o.w + '" w:type="dxa"/>' + (o.span ? '<w:gridSpan w:val="' + o.span + '"/>' : '') +
      (o.fill ? '<w:shd w:val="clear" w:color="auto" w:fill="' + o.fill + '"/>' : '') +
      '<w:tcMar><w:top w:w="' + (o.pad || 70) + '" w:type="dxa"/><w:left w:w="100" w:type="dxa"/><w:bottom w:w="' + (o.pad || 70) + '" w:type="dxa"/><w:right w:w="100" w:type="dxa"/></w:tcMar>' +
      '<w:vAlign w:val="' + (o.vAlign || 'top') + '"/></w:tcPr>' + (content || wPara('', { after: 0 })) + '</w:tc>';
  }
  function wTable(widths, rows, o) {
    o = o || {};
    var b = o.grid || DECK.hairline, bw = o.gridW || 4;
    var borders = ['top', 'left', 'bottom', 'right', 'insideH', 'insideV'].map(function (k) {
      var none = o.noVertical && (k === 'left' || k === 'right' || k === 'insideV');
      return '<w:' + k + ' w:val="' + (none ? 'nil' : 'single') + '" w:sz="' + bw + '" w:space="0" w:color="' + b + '"/>';
    }).join('');
    return '<w:tbl><w:tblPr><w:tblW w:w="' + widths.reduce(function (a, c) { return a + c; }, 0) + '" w:type="dxa"/>' +
      '<w:tblBorders>' + borders + '</w:tblBorders><w:tblLayout w:type="fixed"/><w:tblLook w:val="04A0" w:firstRow="1" w:lastRow="0" w:firstColumn="0" w:lastColumn="0" w:noHBand="1" w:noVBand="1"/></w:tblPr>' +
      '<w:tblGrid>' + widths.map(function (w) { return '<w:gridCol w:w="' + w + '"/>'; }).join('') + '</w:tblGrid>' +
      rows.map(function (r, i) {
        return '<w:tr>' + (i === 0 && o.header ? '<w:trPr><w:tblHeader/><w:cantSplit/></w:trPr>' : '<w:trPr><w:cantSplit/></w:trPr>') + r + '</w:tr>';
      }).join('') + '</w:tbl>' + wPara('', { after: 60 });
  }

  function proposalDocx(protocol, who) {
    var P = protocol || {}, D = who || {}, st = P.standard || {};
    var docLinks = [];
    return withLibError(loadLib('JSZip').then(function (JSZip) {
      var FULL = 9638;                      // A4 text width in twips (2 cm margins)
      var body = [], secN = 0, today = fmtDate(new Date());
      var navy = DECK.navy, gold = DECK.gold, muted = DECK.muted;
      function h1(t) { body.push(wPara(wRun(String(++secN), { color: gold, bold: true }) + '<w:r><w:tab/></w:r>' + wRun(t), { style: 'Heading1', keepNext: true })); }
      function label(t) { body.push(wPara(wRun(t, { bold: true, caps: true, color: gold, size: 8.5, spacing: 20 }), { keepNext: true, after: 40, before: 120 })); }
      function para(t, o) { if (clean(t)) body.push(wPara(wRun(clean(t), o), { line: 276 })); }
      function bullets(list) {
        list.forEach(function (t) {
          if (clean(t)) body.push(wPara(wRun('•', { color: gold, bold: true }) + wRun('\t' + clean(t)), { indent: 360, hanging: 360, after: 80, line: 264 }));
        });
      }
      var title = clean(D.title) || clean(P.question) || 'Clinical audit proposal';
      var place = [D.department, D.site].map(clean).filter(Boolean).join(', ');

      /* Masthead */
      body.push(wPara(wRun('CLINICAL AUDIT PROPOSAL', { bold: true, color: gold, size: 9, spacing: 60 }), { after: 120 }));
      body.push(wPara(wRun(title, { font: DECK_FONTS.head, size: 24, color: navy }), { after: 160, line: 252 }));
      if (clean(D.title) && clean(P.question) && clean(D.title) !== clean(P.question)) para('Audit question: ' + clean(P.question), { italic: true, color: muted });
      body.push(wPara(wRun([place, 'Prepared ' + today].filter(Boolean).join('   ·   '), { color: muted, size: 10 }),
        { after: 240, border: null }));

      /* At a glance */
      var weeks = weeksTotal(P.timeline);
      var glance = [
        ['Audit lead', clean([D.lead, D.role].filter(Boolean).join(', ')) || '—'],
        ['Supervisor', clean(D.supervisor) || '—'],
        ['Where', place || '—'],
        ['Proposed start', D.startDate ? fmtDate(D.startDate) : 'To agree'],
        ['Records per cycle', clean(D.sampleSize) || words(stripParens(firstSentence(P.sample)), 10) || '—'],
        ['Duration', weeks ? weeks + ' weeks to the first results, then re-audit' : '—'],
        ['Target', clean(P.target) ? words(P.target, 18) : 'To agree'],
        ['Reference', clean(P.id) || '—']
      ];
      var gw = [1900, FULL / 2 - 1900];
      var rowsG = [];
      for (var i = 0; i < glance.length; i += 2) {
        var cells = '';
        [glance[i], glance[i + 1]].forEach(function (g) {
          cells += wCell(wPara(wRun(g[0].toUpperCase(), { bold: true, color: muted, size: 8, spacing: 20 }), { after: 0 }), { w: gw[0], fill: 'F6F4EE', vAlign: 'center' }) +
            wCell(wPara(wRun(g[1], { color: navy, size: 10 }), { after: 0 }), { w: gw[1], vAlign: 'center' });
        });
        rowsG.push(cells);
      }
      body.push(wTable([gw[0], gw[1], gw[0], gw[1]], rowsG, { grid: 'E4E1D8' }));

      /* 1 Why */
      h1('Why this audit');
      para(P.why);

      /* 2 Standard */
      h1('The standard');
      if (clean(st.wording)) body.push(wPara(wRun('“' + clean(st.wording) + '”', { font: DECK_FONTS.head, italic: true, size: 12, color: navy }),
        { border: gold, indent: 240, after: 120, line: 288 }));
      if (clean(st.source)) body.push(wPara(wRun(hsrc(st.source), { color: muted, size: 9.5 }), { indent: 240, after: clean(st.url) ? 0 : 120 }));

      if (isNiceStd(st)) body.push(wPara(wRun(niceNotice(st), { color: muted, size: 7.5 }), { indent: 240, after: 160 }));

      /* 3 Method */
      h1('Method');
      [['Who is included', P.population], ['Sample', P.sample], ['Where the data come from', P.data_source], ['What counts as a pass', P.pass]].forEach(function (x) {
        if (!clean(x[1])) return;
        label(x[0]); para(x[1]);
      });

      /* 4 Data collection sheet (spreadsheet look) */
      var fields = (P.template || []).filter(function (f) { return f && f.field; });
      if (fields.length) {
        h1('Data collection sheet');
        para('One row per patient, one column per question. Audit codes only: no names, NHS numbers or dates of birth.', { color: muted, size: 9.5 });
        var shown = fields.slice(0, 8), rest = fields.slice(8);
        var nW = 380, colW = Math.floor((FULL - nW) / shown.length);
        var widths = [nW].concat(shown.map(function () { return colW; }));
        var head = wCell(wPara('', { after: 0 }), { w: nW, fill: 'EDEDED', pad: 30 }) + shown.map(function (f, k) {
          return wCell(wPara(wRun(String.fromCharCode(65 + k), { color: '6B6B6B', size: 8 }), { align: 'center', after: 0 }), { w: colW, fill: 'EDEDED', pad: 30 });
        }).join('');
        var hdr = wCell(wPara(wRun('1', { color: '6B6B6B', size: 8 }), { align: 'center', after: 0 }), { w: nW, fill: 'EDEDED' }) + shown.map(function (f) {
          return wCell(wPara(wRun(humanLabel(f.field), { bold: true, color: 'FFFFFF', size: 8.5 }), { after: 0 }), { w: colW, fill: C.blue });
        }).join('');
        var hint = wCell(wPara(wRun('2', { color: '6B6B6B', size: 8 }), { align: 'center', after: 0 }), { w: nW, fill: 'EDEDED' }) + shown.map(function (f) {
          return wCell(wPara(wRun(cellHint(f), { italic: true, color: '8A8A8A', size: 8 }), { after: 0 }), { w: colW });
        }).join('');
        var rowsX = [head, hdr, hint];
        for (var r = 3; r <= 5; r++) {
          rowsX.push(wCell(wPara(wRun(String(r), { color: '6B6B6B', size: 8 }), { align: 'center', after: 0 }), { w: nW, fill: 'EDEDED' }) +
            shown.map(function () { return wCell(wPara('', { after: 0 }), { w: colW }); }).join(''));
        }
        body.push(wTable(widths, rowsX, { grid: 'C8C8C8', gridW: 4, header: true }));
        if (rest.length) para('Further columns: ' + rest.map(function (f) { return humanLabel(f.field) + ' (' + cellHint(f).replace(/^▾ /, '') + ')'; }).join('; ') + '.', { size: 9.5, color: muted });
        var notes = fields.filter(function (f) { return clean(f.note); });
        if (notes.length) { label('Column notes'); bullets(notes.map(function (f) { return humanLabel(f.field) + ': ' + clean(f.note); })); }
        para('The Excel data sheet has these columns with drop-down lists and calculates the results automatically.', { size: 9.5, color: muted });
      }

      /* 5 Timeline */
      var steps = datedSteps(P.timeline, D.startDate);
      if (steps || clean(P.timeline)) {
        h1('Timeline');
        if (steps) {
          var dated = !!steps[0].from;
          var tw = dated ? [1400, 2300, FULL - 1400 - 2300 - 2400, 2400] : [1500, FULL - 1500 - 2400, 2400];
          var head2 = ['Weeks'].concat(dated ? ['Dates'] : []).concat(['Step', 'Reminder']);
          var trows = [head2.map(function (h, k) { return wCell(wPara(wRun(h.toUpperCase(), { bold: true, color: muted, size: 8, spacing: 20 }), { after: 0 }), { w: tw[k] }); }).join('')];
          steps.forEach(function (sx) {
            var cells = [wRun(sx.a === sx.b ? 'Week ' + sx.a : 'Weeks ' + sx.a + '\u2013' + sx.b, { bold: true, color: gold, size: 10 })];
            if (dated) cells.push(wRun(shortDate(sx.from) + ' \u2013 ' + shortDate(sx.to), { color: navy, size: 10 }));
            cells.push(wRun(sx.label, { color: navy, size: 10 }));
            cells.push(sx.reminder ? wRun('\u25cf ', { color: gold, size: 10 }) + wRun('Email ' + (dated ? 'on ' + shortDate(sx.to) : 'when due'), { color: muted, size: 9.5 }) : wRun('', {}));
            trows.push(cells.map(function (c, k) { return wCell(wPara(c, { after: 0 }), { w: tw[k] }); }).join(''));
          });
          body.push(wTable(tw, trows, { noVertical: true, grid: DECK.hairline, header: true }));
          para('Reminders: the audit lead gets an email when each marked step is due. The emails hold only the audit question, the step and its date.', { size: 9, color: muted });
        } else para(P.timeline);
      }

      /* 6-8 Change, re-audit, loop */
      if (clean(P.change)) { h1('The change we will make'); para(P.change); }
      if (clean(P.reaudit) || clean(P.target)) {
        h1('Re-audit and target');
        if (clean(P.target)) body.push(wPara(wRun('Target', { bold: true, caps: true, color: gold, size: 9, spacing: 20 }) + '<w:r><w:tab/></w:r>' + wRun(clean(P.target), { font: DECK_FONTS.head, size: 13, color: navy }), { after: 120 }));
        para(P.reaudit);
      }
      if (clean(P.close_loop)) { h1('Closing the loop'); para(P.close_loop); }
      if ((P.pitfalls || []).length) { h1('Risks and how we will handle them'); bullets(P.pitfalls.map(function (t) { return str(t).replace(/\s*→\s*/, ' → '); })); }
      if ((P.evidence || []).length) { h1('Previous audits on this topic'); bullets(P.evidence); }

      /* Links, all together at the end */
      var links = (D.links && D.links.length) ? D.links : (/^https?:\/\//i.test(str(st.url)) ? [[capFirst(hsrc(st.source)) || 'The standard', clean(st.url)]] : []);
      if (links.length) {
        h1('Links');
        links.forEach(function (l) {
          docLinks.push(clean(l[1]));
          body.push(wPara(wRun(clean(l[0]), { color: navy, size: 10 }) + '<w:r><w:br/></w:r><w:hyperlink r:id="rIdL' + docLinks.length + '">' +
            wRun(clean(l[1]), { color: '1F4E8C', size: 9, link: true }) + '</w:hyperlink>', { after: 80 }));
        });
      }

      /* Approval */
      body.push(wPara(wRun('Supervisor approval', { font: DECK_FONTS.head, size: 14, color: navy }), { before: 360, after: 120, keepNext: true }));
      var aw = [2600, FULL - 2600];
      body.push(wTable(aw, [
        ['Supervisor', clean(D.supervisor)], ['Decision', '☐ Approved as written     ☐ Approved with changes     ☐ Not approved'],
        ['Changes or comments', '\n\n'], ['Signature', ''], ['Date', '']
      ].map(function (x) {
        return wCell(wPara(wRun(x[0].toUpperCase(), { bold: true, color: muted, size: 8, spacing: 20 }), { after: 0 }), { w: aw[0], fill: 'F6F4EE', vAlign: 'center', pad: 140 }) +
          wCell(wPara(wRun(x[1], { color: navy, size: 10 }), { after: 0 }), { w: aw[1], vAlign: 'center', pad: 140 });
      }), { grid: 'E4E1D8' }));
      para('Register the audit with your clinical audit department before collecting data.', { size: 9, color: muted });

      var doc = XMLH + '<w:document ' + W_NS + '><w:body>' + body.join('') +
        '<w:sectPr><w:footerReference w:type="default" r:id="rIdF"/><w:pgSz w:w="11906" w:h="16838"/>' +
        '<w:pgMar w:top="1134" w:right="1134" w:bottom="1134" w:left="1134" w:header="567" w:footer="567" w:gutter="0"/></w:sectPr></w:body></w:document>';
      var footer = XMLH + '<w:ftr ' + W_NS + '>' + wPara(
        wRun('Audit proposal  ·  ' + trunc(title, 70) + '  ·  Page ', { color: DECK.grey, size: 8 }) +
        '<w:r><w:rPr><w:color w:val="' + DECK.grey + '"/><w:sz w:val="16"/></w:rPr><w:fldChar w:fldCharType="begin"/></w:r><w:r><w:rPr><w:color w:val="' + DECK.grey + '"/><w:sz w:val="16"/></w:rPr><w:instrText xml:space="preserve"> PAGE </w:instrText></w:r>' +
        '<w:r><w:rPr><w:color w:val="' + DECK.grey + '"/><w:sz w:val="16"/></w:rPr><w:fldChar w:fldCharType="separate"/></w:r><w:r><w:rPr><w:color w:val="' + DECK.grey + '"/><w:sz w:val="16"/></w:rPr><w:t>1</w:t></w:r>' +
        '<w:r><w:rPr><w:color w:val="' + DECK.grey + '"/><w:sz w:val="16"/></w:rPr><w:fldChar w:fldCharType="end"/></w:r>', { after: 0 }) + '</w:ftr>';
      var styles = XMLH + '<w:styles ' + W_NS + '><w:docDefaults><w:rPrDefault><w:rPr><w:rFonts w:ascii="' + DECK_FONTS.body + '" w:hAnsi="' + DECK_FONTS.body + '" w:cs="' + DECK_FONTS.body + '" w:eastAsia="' + DECK_FONTS.body + '"/>' +
        '<w:color w:val="' + DECK.navy + '"/><w:sz w:val="21"/><w:szCs w:val="21"/><w:lang w:val="en-GB"/></w:rPr></w:rPrDefault><w:pPrDefault><w:pPr><w:spacing w:after="120" w:line="264" w:lineRule="auto"/></w:pPr></w:pPrDefault></w:docDefaults>' +
        '<w:style w:type="paragraph" w:default="1" w:styleId="Normal"><w:name w:val="Normal"/><w:qFormat/></w:style>' +
        '<w:style w:type="paragraph" w:styleId="Heading1"><w:name w:val="heading 1"/><w:basedOn w:val="Normal"/><w:next w:val="Normal"/><w:qFormat/>' +
        '<w:pPr><w:keepNext/><w:pBdr><w:top w:val="single" w:sz="4" w:space="8" w:color="' + DECK.hairline + '"/></w:pBdr><w:spacing w:before="300" w:after="120"/><w:outlineLvl w:val="0"/></w:pPr>' +
        '<w:rPr><w:rFonts w:ascii="' + DECK_FONTS.head + '" w:hAnsi="' + DECK_FONTS.head + '" w:cs="' + DECK_FONTS.head + '"/><w:color w:val="' + DECK.navy + '"/><w:sz w:val="30"/><w:szCs w:val="30"/></w:rPr></w:style>' +
        '<w:style w:type="table" w:default="1" w:styleId="TableNormal"><w:name w:val="Normal Table"/><w:tblPr><w:tblInd w:w="0" w:type="dxa"/><w:tblCellMar><w:left w:w="100" w:type="dxa"/><w:right w:w="100" w:type="dxa"/></w:tblCellMar></w:tblPr></w:style></w:styles>';
      var zip = new JSZip();
      zip.file('[Content_Types].xml', XMLH + '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">' +
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/>' +
        '<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>' +
        '<Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>' +
        '<Override PartName="/word/footer1.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.footer+xml"/>' +
        '<Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>' +
        '<Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/></Types>');
      zip.file('_rels/.rels', relsDoc([['rId1', REL + 'officeDocument', 'word/document.xml'],
        ['rId2', 'http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties', 'docProps/core.xml'], ['rId3', REL + 'extended-properties', 'docProps/app.xml']]));
      var drels = XMLH + '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">' +
        '<Relationship Id="rIdS" Type="' + REL + 'styles" Target="styles.xml"/><Relationship Id="rIdF" Type="' + REL + 'footer" Target="footer1.xml"/>' +
        docLinks.map(function (u, k) { return '<Relationship Id="rIdL' + (k + 1) + '" Type="' + REL + 'hyperlink" Target="' + esc(u) + '" TargetMode="External"/>'; }).join('') + '</Relationships>';
      zip.file('word/_rels/document.xml.rels', drels);
      zip.file('word/document.xml', doc);
      zip.file('word/styles.xml', styles);
      zip.file('word/footer1.xml', footer);
      var now = new Date().toISOString().replace(/\.\d+Z$/, 'Z');
      zip.file('docProps/core.xml', XMLH + '<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:dcterms="http://purl.org/dc/terms/" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">' +
        '<dc:title>' + esc(trunc('Audit proposal: ' + title, 250)) + '</dc:title><dc:creator>' + esc(clean(D.lead) || '') + '</dc:creator>' +
        '<dcterms:created xsi:type="dcterms:W3CDTF">' + now + '</dcterms:created><dcterms:modified xsi:type="dcterms:W3CDTF">' + now + '</dcterms:modified></cp:coreProperties>');
      zip.file('docProps/app.xml', XMLH + '<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties"></Properties>');
      return zip.generateAsync({ type: 'blob', compression: 'DEFLATE', mimeType: 'application/vnd.openxmlformats-officedocument.wordprocessingml.document' });
    }));
  }

  /* ------------------------------------------------------------------ */
  /* 3. Calendar (.ics)                                                  */
  /* ------------------------------------------------------------------ */

  function icsEscape(s) {
    return clean(s).replace(/\\/g, '\\\\').replace(/;/g, '\\;').replace(/,/g, '\\,').replace(/\r\n|\r|\n/g, '\\n');
  }
  function utf8Len(ch) {
    var c = ch.codePointAt ? ch.codePointAt(0) : ch.charCodeAt(0);
    return c < 0x80 ? 1 : c < 0x800 ? 2 : c < 0x10000 ? 3 : 4;
  }
  function fold(line) {
    var out = [], cur = '', len = 0, limit = 75;
    var chars = Array.from ? Array.from(line) : line.split('');
    chars.forEach(function (ch) {
      var b = utf8Len(ch);
      if (len + b > limit) { out.push(cur); cur = ' '; len = 1; }
      cur += ch; len += b;
    });
    out.push(cur);
    return out.join('\r\n');
  }
  function ymd(d) {
    return d.getUTCFullYear() + ('0' + (d.getUTCMonth() + 1)).slice(-2) + ('0' + d.getUTCDate()).slice(-2);
  }
  function ics(run, events) {
    run = run || {};
    var now = new Date();
    var stamp = now.toISOString().replace(/[-:]/g, '').replace(/\.\d+Z$/, 'Z');
    var calName = clean((run.details && run.details.title) || (run.protocol && run.protocol.question) || 'Clinical audit');
    var lines = ['BEGIN:VCALENDAR', 'VERSION:2.0', 'PRODID:-//Ai4Qi//Clinical audit//EN', 'CALSCALE:GREGORIAN', 'METHOD:PUBLISH',
      'X-WR-CALNAME:' + icsEscape('Audit: ' + trunc(calName, 120))];
    (events || []).forEach(function (ev, i) {
      var d = ev && parseYmd(ev.date);
      if (!d) return;
      var end = new Date(d.getTime() + 86400000);
      lines.push('BEGIN:VEVENT');
      lines.push('UID:' + icsEscape(str(run.id || 'run') + '-' + i + '@ai4qi'));
      lines.push('DTSTAMP:' + stamp);
      lines.push('DTSTART;VALUE=DATE:' + ymd(d));
      lines.push('DTEND;VALUE=DATE:' + ymd(end));
      lines.push('SUMMARY:' + icsEscape(ev.title || 'Audit milestone'));
      if (clean(ev.description)) lines.push('DESCRIPTION:' + icsEscape(ev.description));
      lines.push('TRANSP:TRANSPARENT');
      lines.push('END:VEVENT');
    });
    lines.push('END:VCALENDAR');
    return lines.map(fold).join('\r\n') + '\r\n';
  }

  window.AI4QI_EXPORT = {
    ready: ready,
    templateXlsx: templateXlsx,
    deckPptx: deckPptx,
    proposalDocx: proposalDocx,
    ics: ics,
    libraries: { exceljs: LIBS.ExcelJS.url, jszip: LIBS.JSZip.url }
  };
})();
