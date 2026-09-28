/*
 * Ai4Qi export helpers: data-collection workbook (.xlsx), results deck (.pptx)
 * and calendar (.ics). Plain ES5 in an IIFE, no build step.
 *
 * Libraries are loaded lazily from this site's vendor/ folder (copies of the cdnjs builds):
 *   ExcelJS 4.4.0  https://cdnjs.cloudflare.com/ajax/libs/exceljs/4.4.0/exceljs.min.js
 *   JSZip 3.10.1   https://cdnjs.cloudflare.com/ajax/libs/jszip/3.10.1/jszip.min.js
 * PptxGenJS is not hosted on cdnjs, so the deck is written directly as OOXML
 * (PresentationML, bars drawn as editable shapes, speaker notes) and zipped with JSZip.
 *
 * API: window.AI4QI_EXPORT = { ready, templateXlsx, deckPptx, ics }
 */
(function () {
  'use strict';

  var LIBS = {
    ExcelJS: { url: 'vendor/exceljs.min.js', name: 'the spreadsheet library (ExcelJS)' },
    JSZip: { url: 'vendor/jszip.min.js', name: 'the zip library (JSZip)' }
  };
  var loading = {};

  var C = {
    blue: '3346D3', ink: '0E1626', muted: '586174', border: 'E0E3EB', tint: 'ECEEFC',
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
    pass: '2E7D5B'        // reserved for essential traffic-light meaning
  };
  var DECK_FONTS = { head: 'Georgia', body: 'Calibri' };
  var LOGO_SVG = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 48 48"><rect width="48" height="48" rx="11" fill="#0B1F3A"/><rect x="15" y="21" width="5" height="16" rx="2.5" fill="#fff"/><rect x="28" y="21" width="5" height="16" rx="2.5" fill="#fff"/><circle cx="17.5" cy="14.5" r="3.4" fill="#8C97AB"/><circle cx="30.5" cy="9.5" r="3.4" fill="#C9A45C"/></svg>';

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

    // Results code
    var parts = ['"AI4QI v1 | ' + codeSafe(p.id || 'audit').replace(/"/g, '') + ' | C1 "', 'B' + R.met, '"/"', '(B' + R.met + '+B' + R.not + ')', '" n="', 'B' + R.rec,
      '" | RE "', 'C' + R.met, '"/"', '(C' + R.met + '+C' + R.not + ')', '" n="', 'C' + R.rec];
    // Breakdown segments are assembled in hidden helper cells (columns G:H):
    // per table, G = Cycle 1 pairs and H = Re-audit pairs (count > 0 only),
    // and G17 = all segments joined with " ; ".
    rs.getColumn(7).hidden = true;
    rs.getColumn(8).hidden = true;
    var segRefs = [];
    tables.forEach(function (tb) {
      var labelTxt = codeSafe(str(tb.field).replace(/_/g, ' '));
      [['G', 'B', 'C1'], ['H', 'C', 'RE']].forEach(function (spec) {
        var pairs = tb.opts.map(function (o, k) {
          var ref = spec[1] + (tb.start + k);
          return 'IF(' + ref + '>0,' + xlStr(', ' + trunc(codeSafe(o), 120) + '=') + '&' + ref + ',"")';
        }).join('&');
        var cell = spec[0] + tb.start;
        rs.getCell(cell).value = { formula: 'MID(' + pairs + ',3,4000)' };
        var cond = spec[2] === 'RE' ? 'OR(C' + R.rec + '=0,' + cell + '="")' : cell + '=""';
        segRefs.push('IF(' + cond + ',"",' + xlStr(' ; ' + spec[2] + ' ' + labelTxt + ': ') + '&' + cell + ')');
      });
    });
    if (segRefs.length) {
      rs.getCell('G17').value = { formula: segRefs.join('&') };
      parts.push('IF(G17="",""," | "&MID(G17,4,8000))');
    }
    rs.mergeCells('A14:E14');
    var lab = rs.getCell('A14');
    lab.value = 'Results code \u2014 copy this one line into Ai4Qi (My audits \u203a Paste results) to make your dashboard and presentation. It contains totals only, no patient data.';
    lab.font = { name: FONT, size: 11, bold: true, color: { argb: 'FF' + C.ink } };
    lab.alignment = { wrapText: true, vertical: 'bottom' };
    rs.getRow(14).height = 32;
    rs.mergeCells('A15:E15');
    var code = rs.getCell('A15');
    code.value = { formula: parts.join('&') };
    code.font = { name: 'Consolas', size: 10, color: { argb: 'FF' + C.ink } };
    code.fill = fill(C.tint);
    code.alignment = { wrapText: true, vertical: 'middle', indent: 1 };
    var blue = { style: 'medium', color: { argb: 'FF' + C.blue } };
    ['A', 'B', 'C', 'D', 'E'].forEach(function (L) {
      rs.getCell(L + '15').border = { top: blue, bottom: blue, left: L === 'A' ? blue : undefined, right: L === 'E' ? blue : undefined };
    });
    rs.getRow(15).height = 48;

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
      wb.creator = 'Ai4Qi';
      wb.lastModifiedBy = 'Ai4Qi';
      wb.created = new Date();
      wb.modified = new Date();
      wb.title = trunc(p.question || 'Clinical audit data collection', 250);
      wb.subject = 'Clinical audit data collection template' + (p.id ? ' (' + p.id + ')' : '');
      wb.company = 'Ai4Qi';

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
      banner(1, clean(p.question) || 'Clinical audit', { name: FONT, size: 14, bold: true, color: { argb: 'FF' + C.ink } }, C.tint);
      var stdLine = 'Standard: ' + (clean(std.wording) || 'not recorded') + (std.source ? '  \u2014  ' + clean(std.source) : '');
      banner(2, stdLine, { name: FONT, size: 10, italic: true, color: { argb: 'FF' + C.muted } }, C.tint);
      banner(3, 'Pass: ' + (clean(p.pass) || 'not defined') + (p.target ? '   |   Target: ' + clean(p.target) : ''),
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
          fmt = '0.##'; align = 'right';
        }
        if (dv) {
          dv.showErrorMessage = true;
          dv.errorStyle = 'stop';
          dv.errorTitle = humanLabel(f.field);
          if (dv.type === 'list') dv.error = 'Please choose a value from the list.';
          else if (dv.type === 'date') dv.error = 'Please enter a date as dd/mm/yyyy' + (fmt === 'dd/mm/yyyy hh:mm' ? ' hh:mm' : '') + '.';
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
      info('Standard source', std.source);
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
        'The "Results" sheet updates as you type. Copy its one-line Results code into Ai4Qi (My audits \u203a Paste results).',
        'Use the drop-down lists where offered; dates as dd/mm/yyyy (and times as hh:mm).',
        'Hover over a header to see the field name and any collection notes.',
        'Record excluded cases separately; do not add them to the pass rate.',
        'Or upload the rows into Ai4Qi to calculate results.'
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
      foot.value = 'Template generated by Ai4Qi on ' + fmtDate(new Date()) + '.';
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
      '<a:prstGeom prst="' + (o.geom || 'rect') + '"><a:avLst/></a:prstGeom>' + fillXml(o.fill) + lnXml(o.line, o.lineW) +
      '</p:spPr>' + body + '</p:sp>');
  };
  Slide.prototype.rect = function (x, y, w, h, color, geom) { this.shape({ x: x, y: y, w: w, h: h, fill: color, geom: geom }); };
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
  Slide.prototype.line = function (x1, y1, x2, y2, color, wPt) {
    var id = this.nextId++;
    var flipH = x2 < x1, flipV = y2 < y1;
    this.shapes.push('<p:cxnSp><p:nvCxnSpPr><p:cNvPr id="' + id + '" name="Line ' + id + '"/><p:cNvCxnSpPr/><p:nvPr/></p:nvCxnSpPr><p:spPr>' +
      '<a:xfrm' + (flipH ? ' flipH="1"' : '') + (flipV ? ' flipV="1"' : '') + '><a:off x="' + emu(Math.min(x1, x2)) + '" y="' + emu(Math.min(y1, y2)) +
      '"/><a:ext cx="' + emu(Math.abs(x2 - x1)) + '" cy="' + emu(Math.abs(y2 - y1)) + '"/></a:xfrm><a:prstGeom prst="line"><a:avLst/></a:prstGeom>' +
      '<a:ln w="' + Math.round((wPt || 1) * 12700) + '" cap="rnd"><a:solidFill><a:srgbClr val="' + color + '"/></a:solidFill></a:ln></p:spPr></p:cxnSp>');
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
    return XMLH + '<a:theme xmlns:a="' + NS_A + '" name="' + name + '"><a:themeElements><a:clrScheme name="Ai4Qi">' +
      '<a:dk1><a:srgbClr val="' + DECK.navy + '"/></a:dk1><a:lt1><a:srgbClr val="FFFFFF"/></a:lt1><a:dk2><a:srgbClr val="' + DECK.muted + '"/></a:dk2><a:lt2><a:srgbClr val="' + DECK.hairline + '"/></a:lt2>' +
      '<a:accent1><a:srgbClr val="' + DECK.gold + '"/></a:accent1><a:accent2><a:srgbClr val="' + DECK.grey + '"/></a:accent2><a:accent3><a:srgbClr val="' + DECK.navy + '"/></a:accent3>' +
      '<a:accent4><a:srgbClr val="' + DECK.muted + '"/></a:accent4><a:accent5><a:srgbClr val="' + DECK.navySoft + '"/></a:accent5><a:accent6><a:srgbClr val="' + DECK.pass + '"/></a:accent6>' +
      '<a:hlink><a:srgbClr val="' + DECK.navy + '"/></a:hlink><a:folHlink><a:srgbClr val="' + DECK.muted + '"/></a:folHlink></a:clrScheme>' +
      '<a:fontScheme name="Ai4Qi"><a:majorFont><a:latin typeface="' + DECK_FONTS.head + '"/><a:ea typeface=""/><a:cs typeface=""/></a:majorFont>' +
      '<a:minorFont><a:latin typeface="' + DECK_FONTS.body + '"/><a:ea typeface=""/><a:cs typeface=""/></a:minorFont></a:fontScheme>' +
      '<a:fmtScheme name="Ai4Qi"><a:fillStyleLst><a:solidFill><a:schemeClr val="phClr"/></a:solidFill><a:solidFill><a:schemeClr val="phClr"/></a:solidFill><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:fillStyleLst>' +
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

    function add(bg) { var s = new Slide(deck, bg); deck.slides.push(s); if (deck.slides.length > 1) s.number(deck.slides.length); return s; }
    function eyebrow(s, t, color) {
      s.text(t.toUpperCase(), X0, 0.75, 8, 0.3, { size: 11, bold: true, spc: 200, color: color || (s.dark ? DECK.greyOnNavy : DECK.muted) });
    }
    function headline(s, t, y, o) {
      o = o || {};
      return s.text(t, X0, y || 0.7, o.w || XW, o.h || 1.0, { size: o.size || 32, min: 20, font: DECK_FONTS.head,
        color: s.dark ? DECK.paper : DECK.navy, anchor: o.anchor || 't', lineSpacing: 0.95 });
    }

    /* 1. Title (dark) */
    var s = add(DECK.navy);
    s.text('CLINICAL AUDIT', X0, 1.55, 6, 0.3, { size: 11, bold: true, spc: 300, color: DECK.gold });
    s.text(title, X0, 2.0, 10.4, 2.6, { size: 44, min: 26, font: DECK_FONTS.head, color: DECK.paper, lineSpacing: 0.95 });
    var sub = [D.site, D.department, D.lead].map(function (v) { return trunc(v, 60); }).filter(Boolean).join('  \u00b7  ');
    if (sub) s.text(sub, X0, 4.85, 10.4, 0.45, { size: 16, min: 11, color: DECK.greyOnNavy });
    s.text(today, X0, 5.3, 6, 0.35, { size: 12, color: DECK.greyOnNavy });
    s.logo(X0, SH - 1.15, 0.42);
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
        s.text('The question', X0, 5.55, 3, 0.3, { size: 11, bold: true, color: DECK.gold });
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
    if (clean(std.source)) s.text(clean(std.source), X0 + 0.4, 5.3, qW, 0.7, { size: 12, min: 9, color: DECK.muted });
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
      s.text(f[0].toUpperCase(), x, y, cw2, 0.3, { size: 11, bold: true, spc: 200, color: DECK.gold });
      s.text(f[1], x, y + 0.4, cw2 - 0.3, 1.3, { size: 22, min: 14, color: DECK.navy, lineSpacing: 1.05 });
    });
    s.note('Population: ' + (clean(P.population) || 'not recorded'), 'Sample: ' + (clean(P.sample) || 'not recorded'),
      'Data source: ' + (clean(P.data_source) || 'not recorded'), 'Pass definition: ' + (clean(P.pass) || 'not defined'));
    cycles.forEach(function (c) {
      if (c && c.n > 0) s.note((c.label || c.key) + ': ' + c.n + ' records audited' + (dateRange(c) ? ', ' + dateRange(c) : '') + '.');
    });

    /* 5. What we found (cycle 1) */
    function resultBar(sl, pct, x, y, w, fillCol, trackCol, tickCol) {
      sl.rect(x, y, w, 0.12, trackCol);
      if (pct > 0) sl.rect(x, y, Math.max(0.04, w * Math.min(pct, 100) / 100), 0.12, fillCol);
      if (tv !== null) {
        var tx2 = x + w * tv / 100;
        sl.line(tx2, y - 0.22, tx2, y + 0.34, tickCol, 1.5);
        sl.text('Target ' + tv + '%', Math.min(tx2 - 0.6, x + w - 1.2), y + 0.42, 1.2, 0.3, { size: 11, color: tickCol, align: 'ctr' });
      }
    }
    if (has1) {
      var dark1 = !keyIsC2;
      s = add(dark1 ? DECK.navy : DECK.paper);
      var p1 = c1.pct, m1 = met(p1), near1 = nearPhrase(p1), head1;
      if (m1 === null) head1 = capFirst(near1) + ' met the standard';
      else if (m1) head1 = capFirst(near1) + ' met the standard \u2014 target reached';
      else if (tv - p1 <= 10) head1 = 'Close, but short: ' + near1 + ' met the standard';
      else head1 = fewerThan(p1) ? 'Fewer than ' + fewerThan(p1) + ' met the standard' : capFirst(near1) + ' met the standard';
      eyebrow(s, 'What we found' + (has2 ? ' \u00b7 ' + c1Label : ''), dark1 ? DECK.greyOnNavy : DECK.muted);
      s.text(pctText(p1), X0 - 0.08, 1.55, 5.2, 2.6, { size: 150, min: 96, font: DECK_FONTS.head, color: dark1 ? DECK.gold : DECK.navy, anchor: 'ctr' });
      s.text(head1, X0 + 5.5, 1.75, SW - X0 - (X0 + 5.5), 2.2, { size: 32, min: 20, font: DECK_FONTS.head, color: dark1 ? DECK.paper : DECK.navy, anchor: 'ctr', lineSpacing: 1.0 });
      resultBar(s, p1, X0, 4.9, XW, dark1 ? DECK.gold : DECK.grey, dark1 ? DECK.navySoft : DECK.hairline, dark1 ? DECK.paper : DECK.navy);
      s.text(ofText(c1) + ' cases with a result  \u00b7  ' + c1.n + ' audited' + (dateRange(c1) ? '  \u00b7  ' + dateRange(c1) : ''),
        X0, 5.75, 9, 0.35, { size: 12, color: dark1 ? DECK.greyOnNavy : DECK.muted });
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
        s.text((clean(b.label) || humanLabel(b.field)).toUpperCase(), x, y, colW, 0.3, { size: 11, bold: true, spc: 200, color: DECK.gold });
        y += 0.5;
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
        { text: '   \u00b7   ' + lbl.toUpperCase(), size: 11, bold: true, spc: 200, color: DECK.gold }], X0, 0.75, 9, 0.3);
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
        s.line(sx1, y1, sx2, y2, DECK.gold, 1.5);
        s.shape({ x: sx1 - 0.07, y: y1 - 0.07, w: 0.14, h: 0.14, geom: 'ellipse', fill: DECK.grey });
        s.shape({ x: sx2 - 0.07, y: y2 - 0.07, w: 0.14, h: 0.14, geom: 'ellipse', fill: DECK.gold });
        var dtxt = (diff > 0 ? '+' : diff < 0 ? '\u2212' : '\u00b1') + (Math.round(Math.abs(diff) * 10) / 10) + ' pts';
        s.text(dtxt, (sx1 + sx2) / 2 - 1.0, Math.min(y1, y2) - 0.55, 2.0, 0.4, { size: 16, bold: true, color: DECK.gold, align: 'ctr' });
      }
      s.text(pctText(c2.pct), rx - 0.06, ny, nw, nh, { size: 110, min: 72, font: DECK_FONTS.head, color: DECK.gold, anchor: 'ctr' });
      s.text(c2Label + '  \u00b7  ' + ofText(c2), rx, ny + nh + 0.05, nw, 0.35, { size: 13, color: DECK.greyOnNavy });
      if (tv !== null) s.text('Target ' + tv + '%', X0, 6.45, 4, 0.3, { size: 12, color: DECK.greyOnNavy });
      s.note(head2 + '.', has1 ? c1Label + ': ' + ofText(c1) + ' (' + (Math.round(c1.pct * 10) / 10) + '%), ' + c1.n + ' records audited.' : '',
        c2Label + ': ' + ofText(c2) + ' (' + (Math.round(c2.pct * 10) / 10) + '%), ' + c2.n + ' records audited' + (dateRange(c2) ? ', ' + dateRange(c2) : '') + '.',
        diff !== null ? 'Change: ' + (diff >= 0 ? '+' : '') + (Math.round(diff * 10) / 10) + ' percentage points.' : '',
        targetText ? 'Target: ' + targetText + '.' : '');
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
    } else items.push('Collect cycle 1 data with the Ai4Qi template');
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
    if (clean(std.source)) refs.push('Standard: ' + clean(std.source) + (clean(std.url) ? '. ' + clean(std.url) : ''));
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
      s.text('Prepared with Ai4Qi  \u00b7  ' + today + '  \u00b7  Figures are from local audit data.', X0, SH - 0.62, 9, 0.3, { size: 10, color: DECK.grey, anchor: 'b' });
      s.note('Full reference list:', refs.join('\n'));
    }
  }

  function deckPptx(run, stats) {
    return withLibError(Promise.all([loadLib('JSZip'), logoPng()]).then(function (res) {
      var JSZip = res[0];
      var deck = { slides: [], logoPng: res[1] };
      buildSlides(deck, run || {}, stats || {});

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
      var ttl = clean((run && run.details && run.details.title) || (run && run.protocol && run.protocol.question) || 'Clinical audit results');
      zip.file('docProps/core.xml', XMLH + '<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:dcterms="http://purl.org/dc/terms/" xmlns:dcmitype="http://purl.org/dc/dcmitype/" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">' +
        '<dc:title>' + esc(trunc(ttl, 250)) + '</dc:title><dc:creator>Ai4Qi</dc:creator><cp:lastModifiedBy>Ai4Qi</cp:lastModifiedBy>' +
        '<dcterms:created xsi:type="dcterms:W3CDTF">' + now + '</dcterms:created><dcterms:modified xsi:type="dcterms:W3CDTF">' + now + '</dcterms:modified></cp:coreProperties>');
      zip.file('docProps/app.xml', XMLH + '<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties" xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">' +
        '<Application>Ai4Qi</Application><Slides>' + nS + '</Slides><Notes>' + nS + '</Notes><PresentationFormat>Widescreen</PresentationFormat><Company>Ai4Qi</Company></Properties>');

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
      zip.file('ppt/theme/theme1.xml', themeXml('Ai4Qi'));
      zip.file('ppt/theme/theme2.xml', themeXml('Ai4Qi Notes'));
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
      'X-WR-CALNAME:' + icsEscape('Ai4Qi: ' + trunc(calName, 120))];
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
    ics: ics,
    libraries: { exceljs: LIBS.ExcelJS.url, jszip: LIBS.JSZip.url }
  };
})();
