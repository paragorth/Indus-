/*
 * Ai4Qi export helpers: data-collection workbook (.xlsx), results deck (.pptx)
 * and calendar (.ics). Plain ES5 in an IIFE, no build step.
 *
 * Libraries are loaded lazily from this site's vendor/ folder (copies of the cdnjs builds):
 *   ExcelJS 4.4.0  https://cdnjs.cloudflare.com/ajax/libs/exceljs/4.4.0/exceljs.min.js
 *   JSZip 3.10.1   https://cdnjs.cloudflare.com/ajax/libs/jszip/3.10.1/jszip.min.js
 * PptxGenJS is not hosted on cdnjs, so the deck is written directly as OOXML
 * (PresentationML + native DrawingML charts) and zipped with JSZip. Chart data
 * workbooks are embedded (built with ExcelJS) so "Edit data" works in PowerPoint.
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
  var LOGO_SVG = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 48 48"><rect width="48" height="48" rx="12" fill="#3346D3"/><path d="M24 15.5A12 12 0 0 0 24 39.5" fill="none" stroke="#fff" stroke-width="5" stroke-linecap="round"/><path d="M24 8.5A12 12 0 0 1 24 32.5" fill="none" stroke="#7CF2C0" stroke-width="5" stroke-linecap="round"/></svg>';

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

  var EMU = 914400;
  var SW = 13.333, SH = 7.5;
  var NS_A = 'http://schemas.openxmlformats.org/drawingml/2006/main';
  var NS_R = 'http://schemas.openxmlformats.org/officeDocument/2006/relationships';
  var NS_P = 'http://schemas.openxmlformats.org/presentationml/2006/main';
  var NS_C = 'http://schemas.openxmlformats.org/drawingml/2006/chart';
  var REL = 'http://schemas.openxmlformats.org/officeDocument/2006/relationships/';
  var XMLH = '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\r\n';
  function emu(inches) { return Math.round(inches * EMU); }

  // Rough text-fitting for Calibri: average glyph width as a fraction of the font size.
  function fitText(text, w, h, maxPt, minPt, opts) {
    opts = opts || {};
    var cw = opts.bold ? 0.55 : 0.5;
    var ls = opts.lineSpacing || 1.2;
    function linesAt(pt, t) {
      var cpl = Math.max(4, Math.floor((w - 0.2) / (pt * cw / 72)));
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
      return linesAt(pt, t) * pt * ls / 72 + extra + 0.12 <= h;
    }
    for (var pt = maxPt; pt >= minPt; pt -= (pt > 20 ? 2 : 1)) {
      if (fits(pt, text)) return { size: pt, text: text };
    }
    // Still too long at the minimum size: truncate.
    var t = str(text);
    var lo = 1, hi = t.length, best = trunc(t, 20);
    while (lo <= hi) {
      var mid = (lo + hi) >> 1;
      var cand = trunc(t, mid);
      if (fits(minPt, cand)) { best = cand; lo = mid + 1; } else hi = mid - 1;
    }
    return { size: minPt, text: best };
  }

  function Slide(deck) {
    this.deck = deck;
    this.shapes = [];
    this.rels = [];
    this.nextId = 2;
    this.charts = [];
  }
  Slide.prototype.rel = function (type, target, external) {
    var id = 'rId' + (this.rels.length + 2); // rId1 = layout
    this.rels.push({ id: id, type: type, target: target, external: !!external });
    return id;
  };
  function fillXml(color) { return color ? '<a:solidFill><a:srgbClr val="' + color + '"/></a:solidFill>' : '<a:noFill/>'; }
  function lnXml(color, wPt) {
    return color ? '<a:ln w="' + Math.round((wPt || 1) * 12700) + '"><a:solidFill><a:srgbClr val="' + color + '"/></a:solidFill></a:ln>' : '<a:ln><a:noFill/></a:ln>';
  }
  function xfrm(x, y, w, h) {
    return '<a:xfrm><a:off x="' + emu(x) + '" y="' + emu(y) + '"/><a:ext cx="' + emu(Math.max(w, 0.01)) + '" cy="' + emu(Math.max(h, 0.01)) + '"/></a:xfrm>';
  }
  function runXml(r, slide) {
    var pr = '<a:rPr lang="en-GB" sz="' + Math.round((r.size || 14) * 100) + '"' +
      (r.bold ? ' b="1"' : ' b="0"') + (r.italic ? ' i="1"' : '') + (r.link ? ' u="sng"' : '') +
      (r.spc ? ' spc="' + r.spc + '"' : '') + ' dirty="0">' +
      fillXml(r.color || C.ink) +
      '<a:latin typeface="' + FONT + '"/><a:cs typeface="' + FONT + '"/>';
    if (r.link && slide) {
      var rid = slide.rel(REL + 'hyperlink', r.link, true);
      pr += '<a:hlinkClick r:id="' + rid + '"/>';
    }
    pr += '</a:rPr>';
    return '<a:r>' + pr + '<a:t>' + esc(r.text) + '</a:t></a:r>';
  }
  function paraXml(p, slide) {
    var ppr = '<a:pPr algn="' + (p.align || 'l') + '"' +
      (p.bullet ? ' marL="' + emu(0.32) + '" indent="-' + emu(0.32) + '"' : ' marL="0" indent="0"') + '>' +
      '<a:lnSpc><a:spcPct val="' + Math.round((p.lineSpacing || 1) * 100000) + '"/></a:lnSpc>' +
      '<a:spcBef><a:spcPts val="0"/></a:spcBef>' +
      '<a:spcAft><a:spcPts val="' + Math.round((p.spaceAfter || 0) * 100) + '"/></a:spcAft>';
    if (p.bullet) ppr += '<a:buClr><a:srgbClr val="' + C.blue + '"/></a:buClr><a:buSzPct val="100000"/><a:buFont typeface="Arial"/><a:buChar char="\u2022"/>';
    else ppr += '<a:buNone/>';
    ppr += '</a:pPr>';
    var runs = p.runs.map(function (r) { return runXml(r, slide); }).join('');
    var end = '<a:endParaRPr lang="en-GB" sz="' + Math.round(((p.runs[0] && p.runs[0].size) || 14) * 100) + '" dirty="0"/>';
    return '<a:p>' + ppr + runs + end + '</a:p>';
  }
  // Generic shape with optional text. o: {x,y,w,h, geom, fill, line, lineW, radius, paras, anchor, inset}
  Slide.prototype.shape = function (o) {
    var id = this.nextId++;
    var geom = o.geom || 'rect';
    var av = '<a:avLst/>';
    if (geom === 'roundRect') av = '<a:avLst><a:gd name="adj" fmla="val ' + Math.round((o.radius || 0.12) * 100000) + '"/></a:avLst>';
    var ins = emu(o.inset === undefined ? 0.05 : o.inset);
    var body = '';
    if (o.paras) {
      var self = this;
      body = '<p:txBody><a:bodyPr wrap="square" lIns="' + ins + '" tIns="' + ins + '" rIns="' + ins + '" bIns="' + ins +
        '" anchor="' + (o.anchor || 't') + '" rtlCol="0"><a:noAutofit/></a:bodyPr><a:lstStyle/>' +
        o.paras.map(function (p) { return paraXml(p, self); }).join('') + '</p:txBody>';
    }
    this.shapes.push('<p:sp><p:nvSpPr><p:cNvPr id="' + id + '" name="' + (o.paras ? 'TextBox ' : 'Shape ') + id + '"/><p:cNvSpPr' +
      (o.paras && !o.fill ? ' txBox="1"' : '') + '/><p:nvPr/></p:nvSpPr><p:spPr>' + xfrm(o.x, o.y, o.w, o.h) +
      '<a:prstGeom prst="' + geom + '">' + av + '</a:prstGeom>' + fillXml(o.fill) + lnXml(o.line, o.lineW) +
      '</p:spPr>' + (body || '<p:txBody><a:bodyPr rtlCol="0" anchor="ctr"/><a:lstStyle/><a:p><a:endParaRPr lang="en-GB" dirty="0"/></a:p></p:txBody>') + '</p:sp>');
  };
  // Single-style text box that shrinks to fit (then truncates).
  Slide.prototype.text = function (text, x, y, w, h, o) {
    o = o || {};
    var f = fitText(text, w, h, o.size || 14, o.min || Math.min(10, o.size || 14), { bold: o.bold, lineSpacing: (o.lineSpacing || 1) * 1.2, paraSpace: o.paraSpace });
    var paras = f.text.split('\n').map(function (t) {
      return { align: o.align, lineSpacing: o.lineSpacing, spaceAfter: o.paraSpace ? o.paraSpace * f.size : 0,
        runs: [{ text: t, size: f.size, bold: o.bold, italic: o.italic, color: o.color, link: o.link, spc: o.spc }] };
    });
    this.shape({ x: x, y: y, w: w, h: h, paras: paras, anchor: o.anchor, fill: o.fill, line: o.line, geom: o.geom, inset: o.inset });
    return f.size;
  };
  // Bulleted list that shrinks as a whole.
  Slide.prototype.bullets = function (items, x, y, w, h, o) {
    o = o || {};
    items = items.filter(function (s) { return clean(s); });
    if (!items.length) return;
    var joined = items.join('\n');
    var f = fitText(joined, w - 0.32, h, o.size || 18, o.min || 11, { paraSpace: 0.5 });
    var lines = f.text.split('\n');
    var paras = lines.map(function (t) {
      return { bullet: true, spaceAfter: f.size * 0.5, runs: [{ text: t, size: f.size, color: o.color || C.ink }] };
    });
    this.shape({ x: x, y: y, w: w, h: h, paras: paras });
  };
  Slide.prototype.line = function (x1, y1, x2, y2, color, wPt, dash) {
    var id = this.nextId++;
    var flipH = x2 < x1, flipV = y2 < y1;
    this.shapes.push('<p:cxnSp><p:nvCxnSpPr><p:cNvPr id="' + id + '" name="Line ' + id + '"/><p:cNvCxnSpPr/><p:nvPr/></p:nvCxnSpPr><p:spPr>' +
      '<a:xfrm' + (flipH ? ' flipH="1"' : '') + (flipV ? ' flipV="1"' : '') + '><a:off x="' + emu(Math.min(x1, x2)) + '" y="' + emu(Math.min(y1, y2)) +
      '"/><a:ext cx="' + emu(Math.abs(x2 - x1)) + '" cy="' + emu(Math.abs(y2 - y1)) + '"/></a:xfrm><a:prstGeom prst="line"><a:avLst/></a:prstGeom>' +
      '<a:ln w="' + Math.round((wPt || 1) * 12700) + '"><a:solidFill><a:srgbClr val="' + color + '"/></a:solidFill>' +
      (dash ? '<a:prstDash val="dash"/>' : '') + '</a:ln></p:spPr></p:cxnSp>');
  };
  Slide.prototype.logo = function (x, y, size) {
    if (this.deck.logoPng) {
      var rid = this.rel(REL + 'image', '../media/logo.png');
      var id = this.nextId++;
      this.shapes.push('<p:pic><p:nvPicPr><p:cNvPr id="' + id + '" name="Ai4Qi logo" descr="Ai4Qi logo"/><p:cNvPicPr><a:picLocks noChangeAspect="1"/></p:cNvPicPr><p:nvPr/></p:nvPicPr>' +
        '<p:blipFill><a:blip r:embed="' + rid + '"/><a:stretch><a:fillRect/></a:stretch></p:blipFill><p:spPr>' + xfrm(x, y, size, size) +
        '<a:prstGeom prst="rect"><a:avLst/></a:prstGeom></p:spPr></p:pic>');
      return;
    }
    // Fallback: draw the mark with shapes.
    this.shape({ x: x, y: y, w: size, h: size, geom: 'roundRect', radius: 0.25, fill: C.blue });
    var k = size / 48, lw = 5 * k * 72;
    this.arc(x + 12 * k, y + 15.5 * k, 24 * k, 5400000, 16200000, C.white, lw);
    this.arc(x + 12 * k, y + 8.5 * k, 24 * k, 16200000, 5400000, C.mint, lw);
  };
  Slide.prototype.arc = function (x, y, d, a1, a2, color, wPt) {
    var id = this.nextId++;
    this.shapes.push('<p:sp><p:nvSpPr><p:cNvPr id="' + id + '" name="Arc ' + id + '"/><p:cNvSpPr/><p:nvPr/></p:nvSpPr><p:spPr>' + xfrm(x, y, d, d) +
      '<a:prstGeom prst="arc"><a:avLst><a:gd name="adj1" fmla="val ' + a1 + '"/><a:gd name="adj2" fmla="val ' + a2 + '"/></a:avLst></a:prstGeom><a:noFill/>' +
      '<a:ln w="' + Math.round(wPt * 12700) + '" cap="rnd"><a:solidFill><a:srgbClr val="' + color + '"/></a:solidFill></a:ln></p:spPr></p:sp>');
  };
  Slide.prototype.chart = function (spec, x, y, w, h) {
    var n = this.deck.charts.length + 1;
    spec.index = n;
    this.deck.charts.push(spec);
    var rid = this.rel(REL + 'chart', '../charts/chart' + n + '.xml');
    var id = this.nextId++;
    this.shapes.push('<p:graphicFrame><p:nvGraphicFramePr><p:cNvPr id="' + id + '" name="Chart ' + id + '" descr="' + esc(spec.alt || 'Chart') + '"/><p:cNvGraphicFramePr/><p:nvPr/></p:nvGraphicFramePr>' +
      '<p:xfrm><a:off x="' + emu(x) + '" y="' + emu(y) + '"/><a:ext cx="' + emu(w) + '" cy="' + emu(h) + '"/></p:xfrm>' +
      '<a:graphic><a:graphicData uri="' + NS_C + '"><c:chart xmlns:c="' + NS_C + '" r:id="' + rid + '"/></a:graphicData></a:graphic></p:graphicFrame>');
  };
  Slide.prototype.xml = function () {
    return XMLH + '<p:sld xmlns:a="' + NS_A + '" xmlns:r="' + NS_R + '" xmlns:p="' + NS_P + '"><p:cSld><p:bg><p:bgPr>' + fillXml(C.white) +
      '<a:effectLst/></p:bgPr></p:bg><p:spTree><p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr>' +
      '<p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr>' +
      this.shapes.join('') + '</p:spTree></p:cSld><p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr></p:sld>';
  };
  Slide.prototype.relsXml = function () {
    var out = XMLH + '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">' +
      '<Relationship Id="rId1" Type="' + REL + 'slideLayout" Target="../slideLayouts/slideLayout1.xml"/>';
    this.rels.forEach(function (r) {
      out += '<Relationship Id="' + r.id + '" Type="' + r.type + '" Target="' + esc(r.target) + '"' + (r.external ? ' TargetMode="External"' : '') + '/>';
    });
    return out + '</Relationships>';
  };

  /* ---- Charts ---- */
  function txPr(size, color, bold) {
    return '<c:txPr><a:bodyPr/><a:lstStyle/><a:p><a:pPr><a:defRPr sz="' + Math.round(size * 100) + '" b="' + (bold ? 1 : 0) + '">' +
      fillXml(color || C.muted) + '<a:latin typeface="' + FONT + '"/><a:cs typeface="' + FONT + '"/></a:defRPr></a:pPr><a:endParaRPr lang="en-GB"/></a:p></c:txPr>';
  }
  function strCache(ref, vals) {
    return '<c:strRef><c:f>' + ref + '</c:f><c:strCache><c:ptCount val="' + vals.length + '"/>' +
      vals.map(function (v, i) { return '<c:pt idx="' + i + '"><c:v>' + esc(v) + '</c:v></c:pt>'; }).join('') + '</c:strCache></c:strRef>';
  }
  function numCache(ref, vals, fmt) {
    var pts = '';
    vals.forEach(function (v, i) { if (isNum(v)) pts += '<c:pt idx="' + i + '"><c:v>' + v + '</c:v></c:pt>'; });
    return '<c:numRef><c:f>' + ref + '</c:f><c:numCache><c:formatCode>' + esc(fmt || 'General') + '</c:formatCode><c:ptCount val="' + vals.length + '"/>' + pts + '</c:numCache></c:numRef>';
  }
  // spec: {dir:'col'|'bar', categories, series:[{name, values, color, pointColors?}], line?:{name, values, color}, max, pct, legend, labelSize}
  function chartXml(spec, hasEmbed) {
    var nCat = spec.categories.length;
    var catRef = 'Sheet1!$A$2:$A$' + (nCat + 1);
    var fmt = spec.pct ? '0"%"' : '0';
    var lblSize = spec.labelSize || 12;
    function ser(s, i, isLine) {
      var L = colLetter(i + 2);
      var out = '<c:ser><c:idx val="' + i + '"/><c:order val="' + i + '"/><c:tx>' + strCache('Sheet1!$' + L + '$1', [s.name]) + '</c:tx>';
      if (isLine) {
        out += '<c:spPr><a:ln w="28575" cap="rnd"><a:solidFill><a:srgbClr val="' + s.color + '"/></a:solidFill><a:prstDash val="dash"/><a:round/></a:ln></c:spPr>' +
          '<c:marker><c:symbol val="none"/></c:marker>';
      } else {
        out += '<c:spPr>' + fillXml(s.color) + '<a:ln><a:noFill/></a:ln></c:spPr><c:invertIfNegative val="0"/>';
        (s.pointColors || []).forEach(function (pc, k) {
          if (!pc) return;
          out += '<c:dPt><c:idx val="' + k + '"/><c:invertIfNegative val="0"/><c:bubble3D val="0"/><c:spPr>' + fillXml(pc) + '<a:ln><a:noFill/></a:ln></c:spPr></c:dPt>';
        });
        out += '<c:dLbls><c:numFmt formatCode="' + esc(fmt) + '" sourceLinked="0"/><c:spPr><a:noFill/><a:ln><a:noFill/></a:ln></c:spPr>' + txPr(lblSize, C.ink, true) +
          '<c:dLblPos val="outEnd"/><c:showLegendKey val="0"/><c:showVal val="1"/><c:showCatName val="0"/><c:showSerName val="0"/><c:showPercent val="0"/><c:showBubbleSize val="0"/></c:dLbls>';
      }
      out += '<c:cat>' + strCache(catRef, spec.categories) + '</c:cat>' +
        '<c:val>' + numCache('Sheet1!$' + L + '$2:$' + L + '$' + (nCat + 1), s.values, spec.pct ? '0.0' : 'General') + '</c:val>';
      if (isLine) out += '<c:smooth val="0"/>';
      return out + '</c:ser>';
    }
    var bar = '<c:barChart><c:barDir val="' + (spec.dir || 'col') + '"/><c:grouping val="clustered"/><c:varyColors val="0"/>' +
      spec.series.map(function (s, i) { return ser(s, i, false); }).join('') +
      '<c:gapWidth val="' + (spec.gap || 60) + '"/><c:overlap val="' + (spec.series.length > 1 ? -8 : 0) + '"/><c:axId val="50010"/><c:axId val="50020"/></c:barChart>';
    var line = '';
    if (spec.line) {
      line = '<c:lineChart><c:grouping val="standard"/><c:varyColors val="0"/>' + ser(spec.line, spec.series.length, true) +
        '<c:marker val="1"/><c:axId val="50010"/><c:axId val="50020"/></c:lineChart>';
    }
    var horiz = spec.dir === 'bar';
    var grid = '<c:majorGridlines><c:spPr><a:ln w="6350"><a:solidFill><a:srgbClr val="' + C.border + '"/></a:solidFill></a:ln></c:spPr></c:majorGridlines>';
    var scaling = '<c:scaling><c:orientation val="minMax"/>' + (isNum(spec.max) ? '<c:max val="' + spec.max + '"/>' : '') + '<c:min val="0"/></c:scaling>';
    var catAx = '<c:catAx><c:axId val="50010"/><c:scaling><c:orientation val="minMax"/></c:scaling><c:delete val="0"/><c:axPos val="' + (horiz ? 'l' : 'b') + '"/>' +
      '<c:numFmt formatCode="General" sourceLinked="0"/><c:majorTickMark val="none"/><c:minorTickMark val="none"/><c:tickLblPos val="nextTo"/>' +
      '<c:spPr><a:ln w="9525"><a:solidFill><a:srgbClr val="' + C.grey + '"/></a:solidFill></a:ln></c:spPr>' + txPr(spec.catSize || 12, C.ink) +
      '<c:crossAx val="50020"/><c:crosses val="autoZero"/><c:auto val="1"/><c:lblAlgn val="ctr"/><c:lblOffset val="100"/><c:noMultiLvlLbl val="0"/></c:catAx>';
    var valAx = '<c:valAx><c:axId val="50020"/>' + scaling + '<c:delete val="0"/><c:axPos val="' + (horiz ? 'b' : 'l') + '"/>' + grid +
      '<c:numFmt formatCode="' + esc(fmt) + '" sourceLinked="0"/><c:majorTickMark val="none"/><c:minorTickMark val="none"/><c:tickLblPos val="nextTo"/>' +
      '<c:spPr><a:ln><a:noFill/></a:ln></c:spPr>' + txPr(11, C.muted) + '<c:crossAx val="50010"/><c:crosses val="autoZero"/><c:crossBetween val="between"/>' +
      (spec.majorUnit ? '<c:majorUnit val="' + spec.majorUnit + '"/>' : '') + '</c:valAx>';
    var legend = spec.legend ? '<c:legend><c:legendPos val="' + (spec.legendPos || 'b') + '"/><c:overlay val="0"/>' + txPr(12, C.ink) + '</c:legend>' : '';
    return XMLH + '<c:chartSpace xmlns:c="' + NS_C + '" xmlns:a="' + NS_A + '" xmlns:r="' + NS_R + '"><c:date1904 val="0"/><c:lang val="en-GB"/><c:roundedCorners val="0"/>' +
      '<c:chart><c:autoTitleDeleted val="1"/><c:plotArea><c:layout/>' + bar + line + catAx + valAx +
      '<c:spPr><a:noFill/><a:ln><a:noFill/></a:ln></c:spPr></c:plotArea>' + legend + '<c:plotVisOnly val="1"/><c:dispBlanksAs val="gap"/></c:chart>' +
      '<c:spPr><a:noFill/><a:ln><a:noFill/></a:ln></c:spPr>' + txPr(12, C.ink) +
      (hasEmbed ? '<c:externalData r:id="rId1"><c:autoUpdate val="0"/></c:externalData>' : '') + '</c:chartSpace>';
  }
  function chartWorkbook(ExcelJS, spec) {
    var wb = new ExcelJS.Workbook();
    wb.creator = 'Ai4Qi';
    var ws = wb.addWorksheet('Sheet1');
    var all = spec.series.slice();
    if (spec.line) all.push(spec.line);
    all.forEach(function (s, i) { ws.getCell(1, i + 2).value = s.name; });
    spec.categories.forEach(function (c, r) {
      ws.getCell(r + 2, 1).value = c;
      all.forEach(function (s, i) { if (isNum(s.values[r])) ws.getCell(r + 2, i + 2).value = s.values[r]; });
    });
    ws.getColumn(1).width = 30;
    return wb.xlsx.writeBuffer();
  }

  /* ---- Package parts ---- */
  var THEME = XMLH + '<a:theme xmlns:a="' + NS_A + '" name="Ai4Qi"><a:themeElements><a:clrScheme name="Ai4Qi">' +
    '<a:dk1><a:srgbClr val="0E1626"/></a:dk1><a:lt1><a:srgbClr val="FFFFFF"/></a:lt1><a:dk2><a:srgbClr val="586174"/></a:dk2><a:lt2><a:srgbClr val="ECEEFC"/></a:lt2>' +
    '<a:accent1><a:srgbClr val="3346D3"/></a:accent1><a:accent2><a:srgbClr val="16A574"/></a:accent2><a:accent3><a:srgbClr val="B9C0CE"/></a:accent3>' +
    '<a:accent4><a:srgbClr val="E0A91A"/></a:accent4><a:accent5><a:srgbClr val="7CF2C0"/></a:accent5><a:accent6><a:srgbClr val="0B6A4A"/></a:accent6>' +
    '<a:hlink><a:srgbClr val="3346D3"/></a:hlink><a:folHlink><a:srgbClr val="586174"/></a:folHlink></a:clrScheme>' +
    '<a:fontScheme name="Ai4Qi"><a:majorFont><a:latin typeface="Calibri"/><a:ea typeface=""/><a:cs typeface=""/></a:majorFont>' +
    '<a:minorFont><a:latin typeface="Calibri"/><a:ea typeface=""/><a:cs typeface=""/></a:minorFont></a:fontScheme>' +
    '<a:fmtScheme name="Ai4Qi"><a:fillStyleLst><a:solidFill><a:schemeClr val="phClr"/></a:solidFill><a:solidFill><a:schemeClr val="phClr"/></a:solidFill><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:fillStyleLst>' +
    '<a:lnStyleLst><a:ln w="6350"><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:ln><a:ln w="12700"><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:ln><a:ln w="19050"><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:ln></a:lnStyleLst>' +
    '<a:effectStyleLst><a:effectStyle><a:effectLst/></a:effectStyle><a:effectStyle><a:effectLst/></a:effectStyle><a:effectStyle><a:effectLst/></a:effectStyle></a:effectStyleLst>' +
    '<a:bgFillStyleLst><a:solidFill><a:schemeClr val="phClr"/></a:solidFill><a:solidFill><a:schemeClr val="phClr"/></a:solidFill><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:bgFillStyleLst>' +
    '</a:fmtScheme></a:themeElements><a:objectDefaults/><a:extraClrSchemeLst/></a:theme>';
  var EMPTY_TREE = '<p:spTree><p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr><p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr></p:spTree>';
  var LVL = '<a:lvl1pPr><a:defRPr sz="1800"><a:solidFill><a:schemeClr val="tx1"/></a:solidFill><a:latin typeface="+mn-lt"/></a:defRPr></a:lvl1pPr>';
  var MASTER = XMLH + '<p:sldMaster xmlns:a="' + NS_A + '" xmlns:r="' + NS_R + '" xmlns:p="' + NS_P + '"><p:cSld><p:bg><p:bgRef idx="1001"><a:schemeClr val="bg1"/></p:bgRef></p:bg>' + EMPTY_TREE + '</p:cSld>' +
    '<p:clrMap bg1="lt1" tx1="dk1" bg2="lt2" tx2="dk2" accent1="accent1" accent2="accent2" accent3="accent3" accent4="accent4" accent5="accent5" accent6="accent6" hlink="hlink" folHlink="folHlink"/>' +
    '<p:sldLayoutIdLst><p:sldLayoutId id="2147483649" r:id="rId1"/></p:sldLayoutIdLst>' +
    '<p:txStyles><p:titleStyle>' + LVL + '</p:titleStyle><p:bodyStyle>' + LVL + '</p:bodyStyle><p:otherStyle>' + LVL + '</p:otherStyle></p:txStyles></p:sldMaster>';
  var LAYOUT = XMLH + '<p:sldLayout xmlns:a="' + NS_A + '" xmlns:r="' + NS_R + '" xmlns:p="' + NS_P + '" type="blank" preserve="1"><p:cSld name="Blank">' + EMPTY_TREE +
    '</p:cSld><p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr></p:sldLayout>';
  function relsDoc(list) {
    return XMLH + '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">' + list.map(function (r) {
      return '<Relationship Id="' + r[0] + '" Type="' + r[1] + '" Target="' + r[2] + '"/>';
    }).join('') + '</Relationships>';
  }

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
            var ctx = cv.getContext('2d');
            ctx.drawImage(img, 0, 0, 192, 192);
            var url = cv.toDataURL('image/png');
            resolve(/^data:image\/png;base64,/.test(url) ? url.split(',')[1] : null);
          } catch (e) { resolve(null); }
        };
        img.onerror = function () { if (!done) { done = true; clearTimeout(t); resolve(null); } };
        img.src = 'data:image/svg+xml;base64,' + btoa(LOGO_SVG);
      } catch (e) { resolve(null); }
    });
  }

  /* ---- Deck content ---- */
  function cmp(op, v, t) {
    switch (op) {
      case '>': return v > t;
      case '<': return v < t;
      case '\u2264': case '<=': return v <= t;
      case '=': case '==': return v === t;
      default: return v >= t;
    }
  }

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
    var hasC2 = !!(c2 && c2.n > 0);
    var target = stats.target || null;
    var targetText = (target && (target.text || ((target.op || '\u2265') + target.value + '%'))) || clean(P.target) || '';
    var targetShort = target && isNum(target.value) ? (target.op || '\u2265') + target.value + '%' : targetText;
    function metTarget(pct) { return target && isNum(target.value) && isNum(pct) ? cmp(target.op, pct, target.value) : null; }
    var c1Label = (c1 && c1.label) || 'Cycle 1';
    var c2Label = (c2 && c2.label) || 'Re-audit';

    var title = clean(D.title) || clean(P.question) || 'Clinical audit results';
    var today = fmtDate(new Date());
    var footer = [clean(D.site), clean(D.department)].filter(Boolean).join(' \u00b7 ');
    footer = trunc((footer ? footer + ' \u00b7 ' : '') + today, 90);

    function frame(kicker, heading) {
      var s = new Slide(deck);
      deck.slides.push(s);
      s.shape({ x: 0.6, y: 0.5, w: 0.45, h: 0.06, fill: C.blue });
      s.text(kicker.toUpperCase(), 1.15, 0.36, 8, 0.34, { size: 12, bold: true, color: C.blue, spc: 100, inset: 0 });
      if (heading) s.text(heading, 0.6, 0.72, 12.1, 0.95, { size: 30, min: 18, bold: true, color: C.ink, inset: 0, anchor: 't' });
      s.line(0.6, 6.88, SW - 0.6, 6.88, C.border, 0.75);
      s.logo(0.6, 6.98, 0.3);
      s.text(footer, 1.0, 6.98, 9.5, 0.3, { size: 10, color: C.muted, anchor: 'ctr', inset: 0 });
      s.text(String(deck.slides.length), SW - 1.6, 6.98, 1.0, 0.3, { size: 10, color: C.muted, align: 'r', anchor: 'ctr', inset: 0 });
      return s;
    }
    function label(s, t, x, y, w) { s.text(t.toUpperCase(), x, y, w, 0.3, { size: 11, bold: true, color: C.blue, spc: 80, inset: 0 }); }
    function chip(s, t, x, y, w, kind) {
      var fill = kind === 'good' ? C.mint : kind === 'warn' ? C.amberTint : C.tint;
      var col = kind === 'good' ? C.passText : kind === 'warn' ? '7A5A00' : C.blue;
      s.text(t, x, y, w, 0.44, { size: 14, bold: true, color: col, fill: fill, geom: 'roundRect', anchor: 'ctr', align: 'ctr', inset: 0.08,
        line: kind === 'warn' ? C.amber : null });
    }
    function cycleLine(c) {
      if (!c || !(c.n > 0) || !isNum(c.pct)) return 'No data yet';
      // Denominator is met + not met (N/A excluded), matching pct; n is records audited.
      return (c.passN || 0) + ' of ' + ((c.passN || 0) + (c.failN || 0)) + ' (' + pctText(c.pct) + ') met the standard';
    }

    /* 1. Title */
    var s = new Slide(deck);
    deck.slides.push(s);
    s.shape({ x: 9.9, y: 0, w: SW - 9.9, h: SH, fill: C.tint });
    s.logo(0.8, 0.75, 0.75);
    s.text('Ai4Qi', 1.7, 0.85, 3, 0.55, { size: 20, bold: true, color: C.ink, anchor: 'ctr', inset: 0 });
    s.text('CLINICAL AUDIT \u00b7 RESULTS', 0.8, 2.0, 8.6, 0.35, { size: 13, bold: true, color: C.blue, spc: 120, inset: 0 });
    s.text(title, 0.8, 2.45, 8.7, 2.2, { size: 38, min: 22, bold: true, color: C.ink, inset: 0 });
    var sub = [clean(D.site), clean(D.department)].filter(Boolean).join(' \u00b7 ');
    if (sub) s.text(sub, 0.8, 4.8, 8.7, 0.5, { size: 20, min: 14, color: C.muted, inset: 0 });
    var people = [];
    if (clean(D.lead)) people.push('Audit lead: ' + clean(D.lead));
    if (clean(D.team)) people.push('Team: ' + clean(D.team));
    if (clean(D.supervisor)) people.push('Supervisor: ' + clean(D.supervisor));
    if (people.length) s.text(people.join('\n'), 0.8, 5.4, 8.7, 1.1, { size: 14, min: 10, color: C.ink, inset: 0 });
    s.text(today, 0.8, 6.6, 6, 0.4, { size: 13, color: C.muted, inset: 0 });
    // Key number panel
    var keyC = hasC2 ? c2 : c1;
    label(s, hasC2 ? c2Label : c1Label, 10.3, 2.0, 2.7);
    if (keyC && isNum(keyC.pct)) {
      s.text(pctText(keyC.pct), 10.3, 2.35, 2.8, 1.3, { size: 66, min: 40, bold: true, color: C.blue, inset: 0 });
      s.text('met the standard\n' + (keyC.passN || 0) + ' of ' + ((keyC.passN || 0) + (keyC.failN || 0)) + ' \u00b7 ' + keyC.n + ' records audited',
        10.3, 3.65, 2.7, 0.8, { size: 14, min: 10, color: C.ink, inset: 0 });
    } else {
      s.text('No data yet', 10.3, 2.35, 2.8, 0.8, { size: 28, bold: true, color: C.muted, inset: 0 });
    }
    if (targetShort) {
      label(s, 'Target', 10.3, 4.6, 2.7);
      s.text(targetText, 10.3, 4.92, 2.7, 0.9, { size: 16, min: 11, bold: true, color: C.ink, inset: 0 });
    }
    if (clean(run && run.auditId) || clean(P.id)) s.text('Audit ' + (clean(run.auditId) || clean(P.id)), 10.3, 6.6, 2.7, 0.4, { size: 12, color: C.muted, inset: 0 });

    /* 2. Why */
    s = frame('Why this audit', 'Why this audit matters');
    s.text(clean(P.why) || 'Rationale not recorded.', 0.6, 1.9, 7.4, 4.7, { size: 22, min: 13, color: C.ink, lineSpacing: 1.1, inset: 0 });
    s.shape({ x: 8.5, y: 1.9, w: 4.23, h: 4.6, fill: C.tint, geom: 'roundRect', radius: 0.05 });
    label(s, 'Audit question', 8.8, 2.15, 3.7);
    s.text(clean(P.question) || title, 8.8, 2.55, 3.7, 3.75, { size: 18, min: 11, bold: true, color: C.ink, inset: 0 });

    /* 3. Standard */
    s = frame('The standard', null);
    s.text('\u201C', 0.45, 0.8, 1.2, 1.4, { size: 110, bold: true, color: C.blue, inset: 0 });
    s.text(clean(std.wording) || 'Standard wording not recorded.', 1.5, 1.2, 11.2, 2.9, { size: 32, min: 16, italic: true, color: C.ink, inset: 0, anchor: 'ctr', lineSpacing: 1.05 });
    s.text(clean(std.source) || '', 1.5, 4.25, 11.2, 0.8, { size: 15, min: 10, color: C.muted, inset: 0 });
    if (/^https?:\/\//i.test(str(std.url))) s.text(clean(std.url), 1.5, 5.05, 11.2, 0.4, { size: 12, min: 9, color: C.blue, link: clean(std.url), inset: 0 });
    if (targetText) {
      label(s, 'Target', 1.5, 5.75, 1.2);
      chip(s, trunc(targetText, 60), 2.6, 5.66, Math.min(8, 0.6 + targetText.length * 0.13), 'info');
    }

    /* 4. Method */
    s = frame('Method', 'How we audited');
    var items = [['Population', P.population], ['Sample', P.sample], ['Data source', P.data_source], ['Pass definition', P.pass]];
    items.forEach(function (it, k) {
      var col = k % 2, rowi = Math.floor(k / 2);
      var x = 0.6 + col * 4.25, y = 1.9 + rowi * 2.4;
      label(s, it[0], x, y, 3.95);
      s.text(clean(it[1]) || 'Not recorded', x, y + 0.34, 3.95, 1.9, { size: 14, min: 9, color: C.ink, inset: 0 });
    });
    s.shape({ x: 9.2, y: 1.9, w: 3.53, h: 4.7, fill: C.tint, geom: 'roundRect', radius: 0.05 });
    label(s, 'Data collection', 9.45, 2.1, 3.1);
    var cy = 2.5;
    cycles.slice(0, 4).forEach(function (c) {
      var dates = c.from ? fmtDate(c.from) + (c.to && c.to !== c.from ? ' \u2013 ' + fmtDate(c.to) : '') : 'Dates not recorded';
      s.text(clean(c.label) || c.key, 9.45, cy, 3.1, 0.35, { size: 15, bold: true, color: C.ink, inset: 0 });
      s.text(dates, 9.45, cy + 0.35, 3.1, 0.32, { size: 12, color: C.muted, inset: 0 });
      s.text((c.n || 0) + ' cases audited', 9.45, cy + 0.67, 3.1, 0.32, { size: 12, color: C.ink, inset: 0 });
      cy += 1.1;
    });
    if (!cycles.length) s.text('No data collected yet', 9.45, cy, 3.1, 0.4, { size: 14, color: C.muted, inset: 0 });

    /* 5. Cycle 1 results */
    s = frame(c1Label + ' results', null);
    var head = c1 && isNum(c1.pct) ? cycleLine(c1) + (targetShort ? '; target ' + targetShort : '') : 'No data yet for ' + c1Label.toLowerCase();
    s.text(head, 0.6, 0.72, 12.1, 0.95, { size: 28, min: 18, bold: true, color: C.ink, inset: 0 });
    if (c1 && isNum(c1.pct)) {
      s.text(pctText(c1.pct), 0.6, 2.0, 4.6, 1.8, { size: 96, min: 60, bold: true, color: C.blue, inset: 0, anchor: 'ctr' });
      s.text('met the standard', 0.6, 3.8, 4.6, 0.45, { size: 18, color: C.ink, inset: 0 });
      s.text('n = ' + c1.n + ' audited', 0.6, 4.25, 4.6, 0.4, { size: 14, color: C.muted, inset: 0 });
      var m1 = metTarget(c1.pct);
      if (m1 !== null) chip(s, m1 ? 'Target met (' + targetShort + ')' : 'Below target (' + targetShort + ')', 0.6, 4.95, 3.6, m1 ? 'good' : 'warn');
      s.chart({ dir: 'col', categories: ['Met standard', 'Did not meet'], alt: 'Cases meeting and not meeting the standard in ' + c1Label,
        series: [{ name: c1Label, values: [c1.passN || 0, c1.failN || 0], color: C.blue, pointColors: [C.blue, C.grey] }],
        labelSize: 16, catSize: 14, gap: 70 }, 5.6, 1.8, 7.1, 4.85);
    } else {
      s.text('No data yet', 0.6, 2.4, 8, 1.2, { size: 48, bold: true, color: C.muted, inset: 0 });
      s.text('Enter ' + c1Label.toLowerCase() + ' data in Ai4Qi to see results here.', 0.6, 3.7, 8, 0.5, { size: 18, color: C.muted, inset: 0 });
    }

    /* 6. Breakdowns */
    (stats.breakdowns || []).forEach(function (b) {
      if (!b || !b.cycles) return;
      var b1 = b.cycles.c1 || [], b2 = hasC2 ? (b.cycles.c2 || []) : [];
      var withC2 = b2.length > 0;
      if (!b1.length && !b2.length) return;
      var order = [], map1 = {}, map2 = {};
      b1.forEach(function (o) { if (order.indexOf(o.option) < 0) order.push(o.option); map1[o.option] = o.n; });
      b2.forEach(function (o) { if (order.indexOf(o.option) < 0) order.push(o.option); map2[o.option] = o.n; });
      order.sort(function (a, b) { return ((map1[b] || 0) - (map1[a] || 0)) || ((map2[b] || 0) - (map2[a] || 0)); });
      var totalOpts = order.length;
      order = order.slice(0, 14);
      var cats = order.map(function (o) { return trunc(o === null || o === undefined || o === '' ? '(blank)' : o, 42); }).reverse();
      var v1 = order.map(function (o) { return map1[o] || 0; }).reverse();
      var v2 = order.map(function (o) { return map2[o] || 0; }).reverse();
      var bl = clean(b.label) || humanLabel(b.field);
      var bs = frame('Breakdown', bl);
      var series = withC2
        ? [{ name: c2Label, values: v2, color: C.green }, { name: c1Label, values: v1, color: C.grey }]
        : [{ name: c1Label, values: v1, color: C.blue }];
      var total1 = b1.reduce(function (a, o) { return a + (o.n || 0); }, 0);
      var top = null;
      order.forEach(function (o) { if (!top || (map1[o] || 0) > (map1[top] || 0)) top = o; });
      var note = total1 && top !== null ? 'Most common in ' + c1Label.toLowerCase() + ': ' + trunc(top, 60) + ' (' + map1[top] + ' of ' + total1 + ')' : '';
      if (totalOpts > order.length) note += (note ? ' \u00b7 ' : '') + 'top ' + order.length + ' of ' + totalOpts + ' options shown';
      if (note) bs.text(note, 0.6, 1.55, 12.1, 0.4, { size: 15, color: C.muted, inset: 0 });
      bs.chart({ dir: 'bar', categories: cats, series: series, legend: withC2, legendPos: 't', alt: bl + ' by option',
        labelSize: order.length > 8 ? 10 : 12, catSize: order.length > 8 ? 10 : 12, gap: withC2 ? 50 : 40 }, 0.6, 2.0, 12.1, 4.7);
    });

    /* 7. Change */
    var cm = (run && run.changeMade) || {};
    var recorded = !!clean(cm.description);
    s = frame('The change', recorded ? 'What we changed' : 'Planned change');
    s.text(recorded ? clean(cm.description) : (clean(P.change) || 'No change recorded yet.'), 0.6, 1.9, 8.2, 4.6,
      { size: 22, min: 12, color: C.ink, lineSpacing: 1.1, inset: 0 });
    s.shape({ x: 9.3, y: 1.9, w: 3.43, h: 2.2, fill: C.tint, geom: 'roundRect', radius: 0.06 });
    label(s, recorded ? 'Introduced' : 'Status', 9.55, 2.1, 3);
    s.text(recorded ? (fmtDate(cm.date) || 'Date not recorded') : 'Planned \u2013 not yet recorded', 9.55, 2.45, 3, 1.4, { size: 20, min: 12, bold: true, color: C.ink, inset: 0 });

    /* 8. Re-audit */
    var diff = hasC2 && c1 && isNum(c1.pct) && isNum(c2.pct) ? c2.pct - c1.pct : null;
    if (hasC2) {
      s = frame(c2Label + ' results', null);
      var h2 = isNum(c2.pct) ? cycleLine(c2) + (diff !== null ? (diff > 0 ? ', up ' : diff < 0 ? ', down ' : ', unchanged') + (diff ? pointsText(diff) : '') : '')
        : 'No data yet for ' + c2Label.toLowerCase();
      s.text(h2, 0.6, 0.72, 12.1, 0.95, { size: 28, min: 18, bold: true, color: C.ink, inset: 0 });
      var tv = target && isNum(target.value) ? target.value : null;
      s.chart({ dir: 'col', pct: true, max: 100, majorUnit: 20, categories: [c1Label, c2Label], alt: 'Percentage meeting the standard, ' + c1Label + ' versus ' + c2Label,
        series: [{ name: '% meeting standard', values: [c1 && isNum(c1.pct) ? Math.round(c1.pct * 10) / 10 : null, isNum(c2.pct) ? Math.round(c2.pct * 10) / 10 : null], color: C.green, pointColors: [C.grey, C.green] }],
        line: tv !== null ? { name: 'Target ' + targetShort, values: [tv, tv], color: C.amber } : null,
        legend: tv !== null, labelSize: 16, catSize: 14, gap: 80 }, 0.6, 1.8, 7.4, 4.9);
      var px = 8.5;
      label(s, 'Change', px, 1.95, 4.2);
      if (diff !== null) {
        var better = diff > 0;
        s.text((diff > 0 ? '+' : diff < 0 ? '\u2212' : '\u00b1') + (Math.round(Math.abs(diff) * 10) / 10) , px, 2.3, 4.2, 1.3,
          { size: 72, min: 48, bold: true, color: better ? C.passText : diff < 0 ? C.ink : C.muted, inset: 0 });
        s.text('percentage points', px, 3.6, 4.2, 0.45, { size: 18, color: C.ink, inset: 0 });
        if (better) chip(s, 'Improved', px, 4.2, 1.6, 'good');
      } else {
        s.text('Not available', px, 2.3, 4.2, 0.8, { size: 28, bold: true, color: C.muted, inset: 0 });
      }
      var comp = c1Label + ': ' + cycleLine(c1) + '\n' + c2Label + ': ' + cycleLine(c2);
      s.text(comp, px, 4.85, 4.2, 1.2, { size: 13, min: 10, color: C.muted, inset: 0 });
      var m2 = metTarget(c2.pct);
      if (m2 !== null) chip(s, m2 ? 'Target met' : 'Below target', px, 6.1, 2.2, m2 ? 'good' : 'warn');
    }

    /* 9. Conclusions */
    s = frame('Conclusions', 'Conclusions and next steps');
    var bl2 = [];
    if (c1 && isNum(c1.pct)) {
      var m = metTarget(c1.pct);
      bl2.push(c1Label + ': ' + cycleLine(c1) + (m === null ? '.' : m ? ', meeting the target of ' + targetShort + '.' : ', below the target of ' + targetShort + '.'));
    } else bl2.push(c1Label + ': no data yet.');
    if (hasC2) {
      if (isNum(c2.pct)) {
        var mm = metTarget(c2.pct);
        bl2.push(c2Label + ': ' + cycleLine(c2) + (mm === null ? '.' : mm ? ', meeting the target.' : ', still below the target of ' + targetShort + '.'));
      }
      if (diff !== null) bl2.push(diff > 0 ? 'Compliance improved by ' + pointsText(diff) + ' after the change.' :
        diff < 0 ? 'Compliance fell by ' + pointsText(diff) + ' after the change; review how the change was implemented.' :
          'Compliance did not change after the change.');
    } else {
      var ra = trunc(P.reaudit, 120).replace(/[.\s]+$/, '');
      bl2.push((recorded ? 'Re-audit to measure the effect of the change' : 'Introduce the planned change and re-audit') + (ra ? ' (' + ra + ').' : '.'));
    }
    (stats.breakdowns || []).slice(0, 1).forEach(function (b) {
      var list = (b.cycles && b.cycles.c1) || [];
      var tot = 0, best = null;
      list.forEach(function (o) { tot += o.n || 0; if (!best || o.n > best.n) best = o; });
      if (best && best.n) bl2.push((clean(b.label) || humanLabel(b.field)) + ': most often \u201C' + trunc(best.option, 60) + '\u201D (' + best.n + ' of ' + tot + ') in ' + c1Label.toLowerCase() + '.');
    });
    if (clean(P.close_loop)) bl2.push(clean(P.close_loop));
    s.bullets(bl2, 0.6, 1.9, 12.1, 4.75, { size: 20, min: 12 });

    /* 10. References */
    s = frame('References', 'References and evidence');
    var refs = (P.evidence || []).map(clean).filter(Boolean).slice(0, 10);
    if (clean(std.source)) refs.unshift('Standard: ' + clean(std.source) + (clean(std.url) ? ' \u2014 ' + clean(std.url) : ''));
    if (!refs.length) refs.push('No references recorded.');
    s.bullets(refs, 0.6, 1.9, 12.1, 4.2, { size: 16, min: 10, color: C.ink });
    s.text('Prepared with Ai4Qi. Figures are from local audit data; check before sharing outside the department.', 0.6, 6.25, 12.1, 0.4,
      { size: 11, italic: true, color: C.muted, inset: 0 });
  }

  function deckPptx(run, stats) {
    return withLibError(Promise.all([loadLib('JSZip'), loadLib('ExcelJS').catch(function () { return null; }), logoPng()]).then(function (res) {
      var JSZip = res[0], ExcelJS = res[1];
      var deck = { slides: [], charts: [], logoPng: res[2] };
      buildSlides(deck, run || {}, stats || {});

      var zip = new JSZip();
      var nS = deck.slides.length, nC = deck.charts.length;
      var ct = XMLH + '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">' +
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>' +
        '<Default Extension="xml" ContentType="application/xml"/><Default Extension="png" ContentType="image/png"/>' +
        '<Default Extension="xlsx" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"/>' +
        '<Override PartName="/ppt/presentation.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.presentation.main+xml"/>' +
        '<Override PartName="/ppt/slideMasters/slideMaster1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slideMaster+xml"/>' +
        '<Override PartName="/ppt/slideLayouts/slideLayout1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slideLayout+xml"/>' +
        '<Override PartName="/ppt/theme/theme1.xml" ContentType="application/vnd.openxmlformats-officedocument.theme+xml"/>' +
        '<Override PartName="/ppt/presProps.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.presProps+xml"/>' +
        '<Override PartName="/ppt/viewProps.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.viewProps+xml"/>' +
        '<Override PartName="/ppt/tableStyles.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.tableStyles+xml"/>' +
        '<Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>' +
        '<Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>';
      for (var i = 1; i <= nS; i++) ct += '<Override PartName="/ppt/slides/slide' + i + '.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slide+xml"/>';
      for (var j = 1; j <= nC; j++) ct += '<Override PartName="/ppt/charts/chart' + j + '.xml" ContentType="application/vnd.openxmlformats-officedocument.drawingml.chart+xml"/>';
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
        '<Application>Ai4Qi</Application><Slides>' + nS + '</Slides><PresentationFormat>Widescreen</PresentationFormat><Company>Ai4Qi</Company></Properties>');

      var presRels = [['rId1', REL + 'slideMaster', 'slideMasters/slideMaster1.xml']];
      var sldIds = '';
      for (var k = 1; k <= nS; k++) {
        presRels.push(['rId' + (k + 1), REL + 'slide', 'slides/slide' + k + '.xml']);
        sldIds += '<p:sldId id="' + (255 + k) + '" r:id="rId' + (k + 1) + '"/>';
      }
      presRels.push(['rId' + (nS + 2), REL + 'theme', 'theme/theme1.xml']);
      presRels.push(['rId' + (nS + 3), REL + 'presProps', 'presProps.xml']);
      presRels.push(['rId' + (nS + 4), REL + 'viewProps', 'viewProps.xml']);
      presRels.push(['rId' + (nS + 5), REL + 'tableStyles', 'tableStyles.xml']);
      zip.file('ppt/_rels/presentation.xml.rels', relsDoc(presRels));
      zip.file('ppt/presentation.xml', XMLH + '<p:presentation xmlns:a="' + NS_A + '" xmlns:r="' + NS_R + '" xmlns:p="' + NS_P + '" saveSubsetFonts="1">' +
        '<p:sldMasterIdLst><p:sldMasterId id="2147483648" r:id="rId1"/></p:sldMasterIdLst><p:sldIdLst>' + sldIds + '</p:sldIdLst>' +
        '<p:sldSz cx="12192000" cy="6858000"/><p:notesSz cx="6858000" cy="9144000"/></p:presentation>');
      zip.file('ppt/presProps.xml', XMLH + '<p:presentationPr xmlns:a="' + NS_A + '" xmlns:r="' + NS_R + '" xmlns:p="' + NS_P + '"/>');
      zip.file('ppt/viewProps.xml', XMLH + '<p:viewPr xmlns:a="' + NS_A + '" xmlns:r="' + NS_R + '" xmlns:p="' + NS_P + '"><p:normalViewPr><p:restoredLeft sz="15620"/><p:restoredTop sz="94660"/></p:normalViewPr><p:gridSpacing cx="76200" cy="76200"/></p:viewPr>');
      zip.file('ppt/tableStyles.xml', XMLH + '<a:tblStyleLst xmlns:a="' + NS_A + '" def="{5C22544A-7EE6-4342-B048-85BDC9FD1C3A}"/>');
      zip.file('ppt/theme/theme1.xml', THEME);
      zip.file('ppt/slideMasters/slideMaster1.xml', MASTER);
      zip.file('ppt/slideMasters/_rels/slideMaster1.xml.rels', relsDoc([
        ['rId1', REL + 'slideLayout', '../slideLayouts/slideLayout1.xml'], ['rId2', REL + 'theme', '../theme/theme1.xml']]));
      zip.file('ppt/slideLayouts/slideLayout1.xml', LAYOUT);
      zip.file('ppt/slideLayouts/_rels/slideLayout1.xml.rels', relsDoc([['rId1', REL + 'slideMaster', '../slideMasters/slideMaster1.xml']]));
      if (deck.logoPng) zip.file('ppt/media/logo.png', deck.logoPng, { base64: true });
      deck.slides.forEach(function (sl, idx) {
        zip.file('ppt/slides/slide' + (idx + 1) + '.xml', sl.xml());
        zip.file('ppt/slides/_rels/slide' + (idx + 1) + '.xml.rels', sl.relsXml());
      });

      var embeds = deck.charts.map(function (spec) {
        if (!ExcelJS) return Promise.resolve(null);
        return chartWorkbook(ExcelJS, spec).catch(function () { return null; });
      });
      return Promise.all(embeds).then(function (bufs) {
        deck.charts.forEach(function (spec, idx) {
          var n = idx + 1, buf = bufs[idx];
          zip.file('ppt/charts/chart' + n + '.xml', chartXml(spec, !!buf));
          if (buf) {
            zip.file('ppt/embeddings/Microsoft_Excel_Worksheet' + n + '.xlsx', buf);
            zip.file('ppt/charts/_rels/chart' + n + '.xml.rels', relsDoc([['rId1', REL + 'package', '../embeddings/Microsoft_Excel_Worksheet' + n + '.xlsx']]));
          }
        });
        return zip.generateAsync({ type: 'blob', compression: 'DEFLATE',
          mimeType: 'application/vnd.openxmlformats-officedocument.presentationml.presentation' });
      });
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
