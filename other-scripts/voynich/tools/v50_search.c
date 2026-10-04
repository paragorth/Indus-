/* v50 path search: read a page-grid set (tools/v50_build.py) and score millions of reading paths.
 *
 * usage: v50_search SET.txt OUT.bin MODE NRAND SEED [PAGEMASK]
 *   MODE det  = deterministic family (enumerated; see build_det), rand = NRAND random multi-move paths (seeded)
 *   PAGEMASK  = "all" | "even" | "odd" (pages by index parity, for held-out tests)
 * Output: binary float32 records per path: n_pairs, MI1 (bits, plug-in), MI1 of the stream shuffled within pages,
 *         n_units, R3 (glyph/word trigrams recurring on >= 3 pages, minus the shuffled stream's count)
 * Path spec: unit mode W (word grid) or X (char-offset grid, glyph at offset, spaces skipped); selector
 * (W: 0 first glyph, 1 last glyph, 2 second, 3 penultimate, 4 top-63 word id, 5 word hash mod 64); start rule
 * S0 single path from (0,j0), S1 restart in each line (within-line steps only), S2 restart at each paragraph start;
 * fromright flag; edge mode M0 wrap, M1 row-major overflow, M2 skip out-of-line cells, M3 boustrophedon overflow;
 * move cycle (a_k,b_k), k<m: a lines down, b columns right. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

#define MAXW 64
typedef struct { int wid, hid, ng; int g[24]; } Word;
typedef struct { int ps, nw, nc; Word *w; int *ch; } Line;   /* ch: glyph codes with -1 for spaces */
typedef struct { int nl; Line *l; } Page;
static Page *P; static int NP;

typedef struct { uint8_t unit, sel, start, fr, mode, m, j0; int8_t a[6], b[6]; } Spec;

static double *NLOGN; static int NLN;
static double nlogn(int c) { return c < NLN ? NLOGN[c] : c * log2((double)c); }

static void readset(const char *fn) {
    FILE *f = fopen(fn, "r"); if (!f) { perror(fn); exit(1); }
    if (fscanf(f, "%d", &NP) != 1) exit(2);
    P = calloc(NP, sizeof(Page));
    for (int p = 0; p < NP; p++) {
        if (fscanf(f, "%d", &P[p].nl) != 1) exit(2);
        P[p].l = calloc(P[p].nl, sizeof(Line));
        for (int i = 0; i < P[p].nl; i++) {
            Line *L = &P[p].l[i];
            if (fscanf(f, "%d %d", &L->ps, &L->nw) != 2) exit(2);
            L->w = calloc(L->nw, sizeof(Word)); int nc = 0;
            for (int j = 0; j < L->nw; j++) {
                Word *w = &L->w[j]; int ng;
                if (fscanf(f, "%d %d %d", &w->wid, &w->hid, &ng) != 3) exit(2);
                for (int k = 0; k < ng; k++) { int g; if (fscanf(f, "%d", &g) != 1) exit(2); if (k < 24) w->g[k] = g; }
                w->ng = ng < 24 ? ng : 24; nc += w->ng + 1;
            }
            L->ch = malloc(sizeof(int) * (nc + 1)); int c = 0;
            for (int j = 0; j < L->nw; j++) { for (int k = 0; k < L->w[j].ng; k++) L->ch[c++] = L->w[j].g[k]; if (j < L->nw - 1) L->ch[c++] = -1; }
            L->nc = c;
        }
    }
    fclose(f);
}

/* ---------- stream generation ---------- */
static int *BUF; static int BN, BCAP;   /* units; -1 = segment break; -2 = page break */
static void emit(int u) { if (BN >= BCAP) { BCAP *= 2; BUF = realloc(BUF, sizeof(int) * BCAP); } BUF[BN++] = u; }

static inline int rowlen(const Spec *s, Line *L) { return s->unit == 0 ? L->nw : L->nc; }
static int cell(const Spec *s, Line *L, int col) {
    if (s->unit == 1) return L->ch[col];
    Word *w = &L->w[col];
    switch (s->sel) {
        case 0: return w->g[0];
        case 1: return w->g[w->ng - 1];
        case 2: return w->ng >= 2 ? w->g[1] : -1;
        case 3: return w->ng >= 2 ? w->g[w->ng - 2] : -1;
        case 4: return w->wid;
        default: return w->hid;
    }
}
static int physcol(const Spec *s, int i, int j, int len) {
    int fr = s->fr; if (s->mode == 3 && (i & 1)) fr = !fr;
    return fr ? len - 1 - j : j;
}

/* walk one sub-path on page p from line i0 (stop at line iend), start column j0 */
static void walk(const Spec *s, Page *pg, int i0, int iend, int within_line) {
    int i = i0, j = s->j0, steps = 0, k = 0;
    if (i >= iend) return;
    int len = rowlen(s, &pg->l[i]);
    if (s->mode == 1 || s->mode == 3) { while (j >= len) { j -= len; i++; if (i >= iend) return; len = rowlen(s, &pg->l[i]); } }
    while (steps++ < 4000) {
        Line *L = &pg->l[i]; len = rowlen(s, L);
        int col = -1;
        if (s->mode == 0) { col = ((j % len) + len) % len; }
        else if (j >= 0 && j < len) col = j;
        if (col >= 0) { int u = cell(s, L, physcol(s, i, col, len)); if (u >= 0) emit(u); }
        int a = s->a[k], b = s->b[k]; k = (k + 1) % s->m;
        if (within_line) { j += b; if (j < 0 || j >= len) return; continue; }
        i += a; if (i >= iend) return;
        if (s->mode == 1 || s->mode == 3) {
            int nl = rowlen(s, &pg->l[i]); if (j >= nl) j = nl - 1;
            j += b;
            while (j >= rowlen(s, &pg->l[i])) { j -= rowlen(s, &pg->l[i]); i++; if (i >= iend) return; }
            while (j < 0) { i--; if (i < i0) return; j += rowlen(s, &pg->l[i]); }
        } else j += b;
    }
}

static int MASK = 0; /* 0 all, 1 even, 2 odd */
static void make_stream(const Spec *s) {
    BN = 0;
    for (int p = 0; p < NP; p++) {
        if (MASK == 1 && (p & 1)) continue;
        if (MASK == 2 && !(p & 1)) continue;
        Page *pg = &P[p];
        if (s->start == 0) walk(s, pg, 0, pg->nl, 0);
        else if (s->start == 1) { for (int i = 0; i < pg->nl; i++) { walk(s, pg, i, i + 1, 1); emit(-1); } }
        else {
            int i = 0;
            while (i < pg->nl) { int e = i + 1; while (e < pg->nl && !pg->l[e].ps) e++; walk(s, pg, i, e, 0); emit(-1); i = e; }
        }
        emit(-2);
    }
}

/* ---------- scoring ---------- */
static int CNT[MAXW * MAXW], RC[MAXW], CC[MAXW];
static double mi1(const int *u, int n, int *npairs) {
    memset(CNT, 0, sizeof CNT); memset(RC, 0, sizeof RC); memset(CC, 0, sizeof CC);
    int N = 0;
    for (int t = 0; t + 1 < n; t++) { int x = u[t], y = u[t + 1]; if (x < 0 || y < 0) continue; CNT[x * MAXW + y]++; RC[x]++; CC[y]++; N++; }
    *npairs = N; if (N < 2) return 0;
    double s = 0; for (int c = 0; c < MAXW * MAXW; c++) if (CNT[c]) s += nlogn(CNT[c]);
    for (int c = 0; c < MAXW; c++) { if (RC[c]) s -= nlogn(RC[c]); if (CC[c]) s -= nlogn(CC[c]); }
    return (s + nlogn(N)) / N;
}
static uint64_t RS;
static inline uint64_t xr(void) { RS ^= RS << 13; RS ^= RS >> 7; RS ^= RS << 17; return RS; }
static int *SH; static int SHCAP;
static void shuffle_pages(void) {
    if (SHCAP < BN) { SHCAP = BN * 2; SH = realloc(SH, sizeof(int) * SHCAP); }
    memcpy(SH, BUF, sizeof(int) * BN);
    int st = 0;
    static int *idx = NULL; static int icap = 0;
    for (int t = 0; t <= BN; t++) {
        if (t == BN || SH[t] == -2) {
            int k = 0; if (icap < t - st + 1) { icap = (t - st + 1) * 2; idx = realloc(idx, sizeof(int) * icap); }
            for (int q = st; q < t; q++) if (SH[q] >= 0) idx[k++] = q;
            for (int q = k - 1; q > 0; q--) { int r = xr() % (q + 1); int tmp = SH[idx[q]]; SH[idx[q]] = SH[idx[r]]; SH[idx[r]] = tmp; }
            st = t + 1;
        }
    }
}
/* trigram recurrence across pages: count distinct trigrams seen on >= 3 pages (hash table) */
#define HT (1 << 18)
static uint32_t HK[HT]; static int HP[HT], HN[HT]; static int HSTAMP[HT]; static int STAMP = 0;
static int recur3(const int *u, int n) {
    STAMP++; int page = 0, out = 0;
    for (int t = 0; t + 2 < n; t++) {
        if (u[t] == -2) { page++; continue; }
        if (u[t] < 0 || u[t + 1] < 0 || u[t + 2] < 0) continue;
        uint32_t key = (u[t] << 12) | (u[t + 1] << 6) | u[t + 2];
        uint32_t h = (key * 2654435761u) >> 14;
        while (HSTAMP[h] == STAMP && HK[h] != key) h = (h + 1) & (HT - 1);
        if (HSTAMP[h] != STAMP) { HSTAMP[h] = STAMP; HK[h] = key; HP[h] = page; HN[h] = 1; continue; }
        if (HP[h] != page) { HP[h] = page; HN[h]++; if (HN[h] == 3) out++; }
    }
    return out;
}

/* ---------- path families ---------- */
static Spec *SP; static int NS, SCAP;
static void add(Spec s) { if (NS >= SCAP) { SCAP = SCAP ? SCAP * 2 : 1 << 16; SP = realloc(SP, sizeof(Spec) * SCAP); } SP[NS++] = s; }

static void build_det(void) {
    int mv[68][2], nm = 0;
    for (int a = 0; a <= 3; a++) for (int b = -4; b <= 12; b++) { mv[nm][0] = a; mv[nm][1] = b; nm++; }
    int cyc[3000][2], nc = 0;
    for (int x = 0; x < nm; x++) { cyc[nc][0] = x; cyc[nc][1] = -1; nc++; }
    for (int x = 0; x < nm; x++) for (int y = x + 1; y < nm; y++) { cyc[nc][0] = x; cyc[nc][1] = y; nc++; }
    static const int XJ0[10] = {0, 1, 2, 3, 4, 6, 8, 12, 16, 24};
    for (int unit = 0; unit <= 1; unit++)
    for (int sel = 0; sel < (unit ? 1 : 6); sel++)
    for (int c = 0; c < nc; c++) {
        Spec s; memset(&s, 0, sizeof s); s.unit = unit; s.sel = sel;
        s.m = cyc[c][1] < 0 ? 1 : 2;
        s.a[0] = mv[cyc[c][0]][0]; s.b[0] = mv[cyc[c][0]][1];
        if (s.m == 2) { s.a[1] = mv[cyc[c][1]][0]; s.b[1] = mv[cyc[c][1]][1]; }
        int A = 0, B = 0, allpos = 1; for (int k = 0; k < s.m; k++) { A += s.a[k]; B += s.b[k]; if (s.a[k] != 0 || s.b[k] <= 0) allpos = 0; }
        for (int fr = 0; fr <= 1; fr++) {
            s.fr = fr;
            for (int jj = 0; jj < (unit ? 10 : 6); jj++) {
                s.j0 = unit ? XJ0[jj] : jj;
                if (allpos) { s.start = 1; s.mode = 2; add(s); }
                for (int st = 0; st <= 2; st += 2) for (int mode = 0; mode <= 3; mode++) {
                    if (A == 0 && (mode == 0 || mode == 2)) continue;
                    if (A == 0 && B <= 0) continue;
                    if (A > 0 && B < 0 && (mode == 1 || mode == 3) ) { /* allowed */ }
                    s.start = st; s.mode = mode; add(s);
                }
            }
        }
    }
}
static void build_rand(int n, uint64_t seed) {
    RS = seed * 0x9E3779B97F4A7C15ull + 1;
    while (NS < n) {
        Spec s; memset(&s, 0, sizeof s);
        s.unit = (xr() % 4) == 0; s.sel = s.unit ? 0 : xr() % 6;
        s.m = 3 + xr() % 4; int A = 0, B = 0;
        for (int k = 0; k < s.m; k++) { s.a[k] = xr() % 5 == 0 ? 1 + xr() % 4 : 0; s.b[k] = (int)(xr() % 29) - 8; A += s.a[k]; B += s.b[k]; }
        s.fr = xr() & 1; s.j0 = s.unit ? xr() % 30 : xr() % 8; s.start = (xr() % 2) * 2; s.mode = xr() % 4;
        if (A == 0 && (s.mode == 0 || s.mode == 2)) continue;
        if (A == 0 && B <= 0) continue;
        add(s);
    }
}

int main(int argc, char **argv) {
    if (argc < 6) { fprintf(stderr, "usage\n"); return 1; }
    readset(argv[1]);
    if (argc > 6) MASK = !strcmp(argv[6], "even") ? 1 : !strcmp(argv[6], "odd") ? 2 : 0;
    NLN = 1 << 20; NLOGN = malloc(sizeof(double) * NLN); NLOGN[0] = 0; for (int c = 1; c < NLN; c++) NLOGN[c] = c * log2((double)c);
    BCAP = 1 << 16; BUF = malloc(sizeof(int) * BCAP);
    /* MODE: det | count | rand | subdet IDXFILE | subrand:N:SEED IDXFILE (argv[4] = idx file of int32 path indices) */
    int *sub = NULL, nsub = -1;
    if (!strncmp(argv[3], "sub", 3)) {
        FILE *fi = fopen(argv[4], "rb"); fseek(fi, 0, SEEK_END); nsub = ftell(fi) / 4; fseek(fi, 0, SEEK_SET);
        sub = malloc(4 * nsub); if (fread(sub, 4, nsub, fi) != (size_t)nsub) return 3; fclose(fi);
        if (!strcmp(argv[3], "subdet")) build_det();
        else { int nr; unsigned long long sd; sscanf(argv[3], "subrand:%d:%llu", &nr, &sd); build_rand(nr, sd); }
    } else if (!strcmp(argv[3], "det") || !strcmp(argv[3], "count")) build_det(); else build_rand(atoi(argv[4]), strtoull(argv[5], 0, 10));
    if (!strcmp(argv[3], "count")) { printf("%d\n", NS); return 0; }
    if (!strcmp(argv[2], "SPEC")) { /* print specs: v50_search SET SPEC det|subdet IDX... */
        for (int q = 0; q < NS; q++) { Spec *s = &SP[q];
            printf("%d unit=%d sel=%d start=%d fr=%d mode=%d j0=%d m=%d moves=", sub ? sub[q] : q, s->unit, s->sel, s->start, s->fr, s->mode, s->j0, s->m);
            for (int k = 0; k < s->m; k++) printf("(%d,%d)", s->a[k], s->b[k]); printf("\n"); }
        return 0; }
    if (sub) { Spec *T = malloc(sizeof(Spec) * nsub); for (int q = 0; q < nsub; q++) T[q] = SP[sub[q]]; SP = T; NS = nsub; }
    if (!strcmp(argv[2], "STREAM")) { /* print each listed path's unit stream (-1 segment, -2 page break) */
        for (int q = 0; q < NS; q++) { make_stream(&SP[q]); printf("%d", sub ? sub[q] : q);
            for (int t = 0; t < BN; t++) printf(" %d", BUF[t]); printf("\n"); }
        return 0; }
    FILE *o = fopen(argv[2], "wb");
    float *rec = malloc(sizeof(float) * 5 * NS);
    for (int q = 0; q < NS; q++) {
        make_stream(&SP[q]);
        int np, np2; double m = mi1(BUF, BN, &np);
        RS = 0x1234567ull + q * 7919ull; shuffle_pages();
        double ms = mi1(SH, BN, &np2);
        int r3 = recur3(BUF, BN), r3s = recur3(SH, BN);
        int nu = 0; for (int t = 0; t < BN; t++) if (BUF[t] >= 0) nu++;
        rec[5 * q] = np; rec[5 * q + 1] = m; rec[5 * q + 2] = ms; rec[5 * q + 3] = nu; rec[5 * q + 4] = r3 - r3s;
    }
    fwrite(rec, sizeof(float), 5 * NS, o); fclose(o);
    fprintf(stderr, "%s: %d paths\n", argv[1], NS);
    return 0;
}
