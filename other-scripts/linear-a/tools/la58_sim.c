/* LA-58: agent-based simulator of Bronze Age administrations + paperwork statistic panel.
 * Build: gcc -O2 -shared -fPIC -o la58_sim.so la58_sim.c -lm   (done by la58_common.build)
 *
 * World types: 0 SINGLE palace + dependents, 1 CAPSAT capital + satellite offices, 2 PEERS independent centres,
 *              3 TEMPLE centre receiving offerings, 4 MERCH merchant houses (lateral exchange), 5 NULL no institutions.
 * Roles: 0 independent, 1 centre, 2 dependent (parent = centre index).
 * Word classes (for read-off only): 0 common, 1 place word, 2 title (office), 3 local name, 4 institution personnel.
 * Document types: 0 TAB, 1 ROU, 2 NOD, 3 OTH.
 */
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define MAXK 16
#define NSITE 13
#define NNUIS 30
#define VMAX 6000

/* ---------------------------------------------------------------- rng */
typedef struct { uint64_t s[4]; } rng_t;
static uint64_t rotl(uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }
static uint64_t nxt(rng_t *r) {
    uint64_t *s = r->s, res = rotl(s[1] * 5, 7) * 9, t = s[1] << 17;
    s[2] ^= s[0]; s[3] ^= s[1]; s[1] ^= s[2]; s[0] ^= s[3]; s[2] ^= t; s[3] = rotl(s[3], 45);
    return res;
}
static void rseed(rng_t *r, uint64_t sd) {
    for (int i = 0; i < 4; i++) { sd += 0x9e3779b97f4a7c15ULL; uint64_t z = sd;
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL; z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL; r->s[i] = z ^ (z >> 31); }
}
static double U(rng_t *r) { return (nxt(r) >> 11) * (1.0 / 9007199254740992.0); }
static double Uab(rng_t *r, double a, double b) { return a + (b - a) * U(r); }
static double LU(rng_t *r, double a, double b) { return exp(Uab(r, log(a), log(b))); }
static int RI(rng_t *r, int n) { return (int)(U(r) * n) % (n > 0 ? n : 1); }
static double NRM(rng_t *r) { double u = U(r) + 1e-300, v = U(r); return sqrt(-2 * log(u)) * cos(6.283185307179586 * v); }

/* ---------------------------------------------------------------- growable doc store */
typedef struct {
    int n, cap; int *site, *dtype, *wst, *nst;
    int nw, wcap; int *W, *WC;
    int nx, xcap; double *X;
} store_t;
static void st_init(store_t *s) { memset(s, 0, sizeof(*s));
    s->cap = 1024; s->site = malloc(4 * s->cap); s->dtype = malloc(4 * s->cap); s->wst = malloc(4 * (s->cap + 1)); s->nst = malloc(4 * (s->cap + 1));
    s->wcap = 8192; s->W = malloc(4 * s->wcap); s->WC = malloc(4 * s->wcap); s->xcap = 8192; s->X = malloc(8 * s->xcap);
    s->wst[0] = 0; s->nst[0] = 0; }
static void st_free(store_t *s) { free(s->site); free(s->dtype); free(s->wst); free(s->nst); free(s->W); free(s->WC); free(s->X); }
static void st_word(store_t *s, int w, int c) { if (s->nw >= s->wcap) { s->wcap *= 2; s->W = realloc(s->W, 4 * s->wcap); s->WC = realloc(s->WC, 4 * s->wcap); }
    s->W[s->nw] = w; s->WC[s->nw] = c; s->nw++; }
static void st_num(store_t *s, double x) { if (s->nx >= s->xcap) { s->xcap *= 2; s->X = realloc(s->X, 8 * s->xcap); } s->X[s->nx++] = x; }
static void st_close(store_t *s, int site, int dt) {
    if (s->n + 1 >= s->cap) { s->cap *= 2; s->site = realloc(s->site, 4 * s->cap); s->dtype = realloc(s->dtype, 4 * s->cap);
        s->wst = realloc(s->wst, 4 * (s->cap + 1)); s->nst = realloc(s->nst, 4 * (s->cap + 1)); }
    s->site[s->n] = site; s->dtype[s->n] = dt; s->n++; s->wst[s->n] = s->nw; s->nst[s->n] = s->nx; }

/* ---------------------------------------------------------------- world */
typedef struct {
    int K, type, role[MAXK], parent[MAXK], lat[MAXK][MAXK], ndep[MAXK];
    double a, Vc, Vl, ntit, mtab, pwe, pne, pcom, phead, ptotc, ptotl, muc, mul, sig, pseal, ptrav, prou, pdw,
           share, pm, ifrac, rr, plat, svt, svr, svn, poth, prod, latshare, trav_local;
    double cdfc[VMAX], cdfl[VMAX]; int nc, nl;
    int PL0, TIT0, LOC0, INS0;
} world_t;

static int TABONLY = 0;
void la58_set_tabonly(int x) { TABONLY = x; }
static int ntheta_(int K) { return 2 + 2 * K + NNUIS; }
int la58_ntheta(int K) { return ntheta_(K); }

static void mkcdf(double *c, int n, double a) { double s = 0; for (int i = 0; i < n; i++) { s += pow(i + 1, -a); c[i] = s; } for (int i = 0; i < n; i++) c[i] /= s; }
static int zipf(rng_t *r, const double *c, int n) { double u = U(r); int lo = 0, hi = n - 1; while (lo < hi) { int m = (lo + hi) / 2; if (c[m] < u) lo = m + 1; else hi = m; } return lo; }

static void draw_world(world_t *w, rng_t *r, int K, int ftype) {
    memset(w, 0, sizeof(*w)); w->K = K;
    w->type = ftype >= 0 ? ftype : RI(r, 6);
    w->a = Uab(r, 0.7, 1.4); w->Vc = LU(r, 30, 3000); w->Vl = LU(r, 100, VMAX); w->ntit = 1 + RI(r, 8);
    w->mtab = Uab(r, 1, 10); w->pwe = Uab(r, 0.4, 1); w->pne = Uab(r, 0.4, 1); w->pcom = Uab(r, 0, 0.6);
    w->phead = U(r); w->ptotc = Uab(r, 0, 0.6); w->ptotl = Uab(r, 0, 0.4); w->muc = Uab(r, 0, 4.5); w->mul = Uab(r, 0, 3);
    w->sig = Uab(r, 0.4, 1.6); w->pseal = U(r); w->ptrav = U(r); w->prou = U(r); w->pdw = U(r);
    w->share = U(r); w->pm = Uab(r, 0, 0.5); w->ifrac = Uab(r, 0.1, 0.9); w->rr = U(r); w->plat = Uab(r, 0.1, 1);
    w->svt = LU(r, 0.2, 5); w->svr = LU(r, 0.2, 5); w->svn = LU(r, 0.2, 5); w->poth = Uab(r, 0, 0.25);
    w->prod = Uab(r, 1.2, 3); w->latshare = Uab(r, 0.1, 0.6); w->trav_local = U(r) * 0.3;
    if (TABONLY) { w->pseal = 0; w->poth = 0; w->trav_local = 0; }
    for (int k = 0; k < K; k++) { w->role[k] = 0; w->parent[k] = -1; }
    int t = w->type;
    if (t == 0 || t == 1 || t == 3) {
        int c = RI(r, K); w->role[c] = 1;
        double patt = Uab(r, 0.5, 1);
        for (int k = 0; k < K; k++) if (k != c && U(r) < patt) { w->role[k] = 2; w->parent[k] = c; }
        if (t == 1) { w->share = Uab(r, 0.6, 1); w->pm = Uab(r, 0.3, 0.8); }
        if (t == 0) { w->share = Uab(r, 0, 0.6); w->pm = Uab(r, 0, 0.3); }
        if (t == 3) { w->rr = 0; w->muc = w->mul * Uab(r, 0.3, 1); }
    } else if (t == 2) {
        int m = 2 + RI(r, K - 1); if (m > K) m = K;
        int idx[MAXK]; for (int k = 0; k < K; k++) idx[k] = k;
        for (int k = K - 1; k > 0; k--) { int j = RI(r, k + 1); int tmp = idx[k]; idx[k] = idx[j]; idx[j] = tmp; }
        for (int i = 0; i < m; i++) w->role[idx[i]] = 1;
        for (int i = m; i < K; i++) { int k = idx[i]; w->role[k] = 2; w->parent[k] = idx[RI(r, m)]; }
        for (int k = 0; k < K; k++) for (int l = k + 1; l < K; l++) if (w->role[k] == 1 && w->role[l] == 1 && U(r) < w->plat * 0.3) w->lat[k][l] = w->lat[l][k] = 1;
    } else if (t == 4) {
        for (int k = 0; k < K; k++) for (int l = k + 1; l < K; l++) if (U(r) < w->plat) w->lat[k][l] = w->lat[l][k] = 1;
    }
    for (int k = 0; k < K; k++) if (w->role[k] == 2) w->ndep[w->parent[k]]++;
    w->nc = (int)w->Vc; w->nl = (int)w->Vl; if (w->nl > VMAX) w->nl = VMAX; if (w->nc > VMAX) w->nc = VMAX;
    mkcdf(w->cdfc, w->nc, w->a); mkcdf(w->cdfl, w->nl, w->a);
    w->PL0 = w->nc; w->TIT0 = w->PL0 + K; w->LOC0 = w->TIT0 + K * 8; w->INS0 = w->LOC0 + K * w->nl;
}

static void put_theta(world_t *w, double *th) {
    int K = w->K, nc = 0; for (int k = 0; k < K; k++) nc += w->role[k] == 1;
    th[0] = w->type; th[1] = nc;
    for (int k = 0; k < K; k++) { th[2 + k] = w->role[k]; th[2 + K + k] = w->parent[k]; }
    double v[NNUIS] = {w->a, log(w->Vc), log(w->Vl), w->ntit, w->mtab, w->pwe, w->pne, w->pcom, w->phead, w->ptotc,
                       w->ptotl, w->muc, w->mul, w->sig, w->pseal, w->ptrav, w->prou, w->pdw, w->share, w->pm, w->ifrac,
                       w->rr, w->plat, log(w->svt), log(w->svr), log(w->svn), w->poth, w->prod, w->latshare, w->trav_local};
    memcpy(th + 2 + 2 * K, v, sizeof(v));
}
static void get_theta(world_t *w, const double *th, int K) {
    /* planted worlds: take type/roles/parents/nuisance from th */
    memset(w, 0, sizeof(*w)); w->K = K; w->type = (int)th[0];
    for (int k = 0; k < K; k++) { w->role[k] = (int)th[2 + k]; w->parent[k] = (int)th[2 + K + k]; }
    const double *v = th + 2 + 2 * K;
    w->a = v[0]; w->Vc = exp(v[1]); w->Vl = exp(v[2]); w->ntit = v[3]; w->mtab = v[4]; w->pwe = v[5]; w->pne = v[6]; w->pcom = v[7];
    w->phead = v[8]; w->ptotc = v[9]; w->ptotl = v[10]; w->muc = v[11]; w->mul = v[12]; w->sig = v[13]; w->pseal = v[14];
    w->ptrav = v[15]; w->prou = v[16]; w->pdw = v[17]; w->share = v[18]; w->pm = v[19]; w->ifrac = v[20]; w->rr = v[21];
    w->plat = v[22]; w->svt = exp(v[23]); w->svr = exp(v[24]); w->svn = exp(v[25]); w->poth = v[26]; w->prod = v[27];
    w->latshare = v[28]; w->trav_local = v[29];
    for (int k = 0; k < K; k++) if (w->role[k] == 2) w->ndep[w->parent[k]]++;
    if (w->type == 4) for (int k = 0; k < K; k++) for (int l = 0; l < K; l++) w->lat[k][l] = (k != l);
    w->nc = (int)w->Vc; w->nl = (int)w->Vl; if (w->nl > VMAX) w->nl = VMAX; if (w->nc > VMAX) w->nc = VMAX;
    mkcdf(w->cdfc, w->nc, w->a); mkcdf(w->cdfl, w->nl, w->a);
    w->PL0 = w->nc; w->TIT0 = w->PL0 + K; w->LOC0 = w->TIT0 + K * 8; w->INS0 = w->LOC0 + K * w->nl;
}

/* word draws */
static void wcommon(world_t *w, rng_t *r, store_t *s) { st_word(s, zipf(r, w->cdfc, w->nc), 0); }
static void wplace(world_t *w, store_t *s, int k) { st_word(s, w->PL0 + k, 1); }
static void wtitle(world_t *w, rng_t *r, store_t *s, int k) { st_word(s, w->TIT0 + k * 8 + RI(r, (int)w->ntit), 2); }
static void wlocal(world_t *w, rng_t *r, store_t *s, int k) { st_word(s, w->LOC0 + k * w->nl + zipf(r, w->cdfl, w->nl), 3); }
static void wpers(world_t *w, rng_t *r, store_t *s, int i) { st_word(s, w->INS0 + i * w->nl + zipf(r, w->cdfl, w->nl), 4); }
static int inst_of(world_t *w, int k) { return w->role[k] == 2 ? w->parent[k] : k; }
/* the office whose title a scribe at k writes */
static int office(world_t *w, rng_t *r, int k) {
    if (w->type == 5) return -1;
    if (w->role[k] == 2 && U(r) < w->share) return w->parent[k];
    return k;
}
static double amount(world_t *w, rng_t *r, double mu) { double x = floor(exp(mu + w->sig * NRM(r)) + 0.5); return x < 1 ? 1 : (x > 9999 ? 9999 : x); }
static int nentries(world_t *w, rng_t *r, double mean) {
    int g = 1; double p = 1.0 / mean; while (U(r) > p && g < 40) g++; return g; }

static void header(world_t *w, rng_t *r, store_t *s, int k) {
    if (U(r) >= w->phead) return;
    int o = office(w, r, k);
    if (o < 0) wcommon(w, r, s); else wtitle(w, r, s, o);
}
static void finish_tab(world_t *w, rng_t *r, store_t *s, int at, int nstart, double ptot) {
    int nn = s->nx - nstart;
    if (nn >= 2 && U(r) < ptot) { double t = 0; for (int i = nstart; i < s->nx; i++) t += s->X[i]; if (U(r) < 0.5) wcommon(w, r, s); st_num(s, t); }
    st_close(s, at, 0);
}
/* entry-word sources */
enum { E_LOCAL, E_RECEIPT, E_RATION, E_DISPATCH, E_LATERAL };
static void entry_word(world_t *w, rng_t *r, store_t *s, int k, int kind, int other) {
    if (U(r) >= w->pwe) return;
    if (U(r) < w->pcom) { wcommon(w, r, s); return; }
    switch (kind) {
    case E_RECEIPT: { /* centre lists who delivered: a dependent's place, or one of its people */
        double u = U(r); if (u < 0.45) wplace(w, s, other); else if (u < 0.75) wlocal(w, r, s, other); else wpers(w, r, s, k); return; }
    case E_RATION: wpers(w, r, s, inst_of(w, k)); return;
    case E_LATERAL: if (U(r) < 0.4) wplace(w, s, other); else wlocal(w, r, s, U(r) < 0.5 ? other : k); return;
    default:
        if (w->type != 5 && w->role[k] != 0 && U(r) < w->pm) wpers(w, r, s, inst_of(w, k)); else wlocal(w, r, s, k);
    }
}
static int pick_dep(world_t *w, rng_t *r, int c) {
    if (!w->ndep[c]) return -1; int j = RI(r, w->ndep[c]);
    for (int k = 0; k < w->K; k++) if (w->role[k] == 2 && w->parent[k] == c) { if (!j) return k; j--; }
    return -1;
}
static void tablet(world_t *w, rng_t *r, store_t *s, int k, int kind, int other, double mean, double mu, double ptot) {
    int n0 = s->nx; header(w, r, s, k);
    if (kind == E_DISPATCH && U(r) < 0.5) wplace(w, s, other);
    if (kind == E_LATERAL && U(r) < 0.3) wplace(w, s, other);
    int g = nentries(w, r, mean);
    for (int e = 0; e < g; e++) { entry_word(w, r, s, k, kind, other); if (U(r) < w->pne) st_num(s, amount(w, r, mu)); }
    finish_tab(w, r, s, k, n0, ptot);
}
static void device(world_t *w, rng_t *r, store_t *s, int k, int dest) {
    int dt = U(r) < w->prou ? 1 : 2;
    if (U(r) < w->pdw) {
        double u = U(r); int o = office(w, r, k);
        if (u < 0.35 && w->type != 5) wplace(w, s, k);
        else if (u < 0.65 && o >= 0) wtitle(w, r, s, o);
        else if (u < 0.85) wlocal(w, r, s, k); else wcommon(w, r, s);
    }
    if (dt == 1 && U(r) < 0.5) st_num(s, amount(w, r, w->mul));
    st_close(s, dest, dt);
}
static void object(world_t *w, rng_t *r, store_t *s, int k, int at) {
    int n = 1 + RI(r, 3); for (int i = 0; i < n; i++) { if (U(r) < 0.5) wcommon(w, r, s); else wlocal(w, r, s, k); }
    st_close(s, at, 3);
}
static void local_event(world_t *w, rng_t *r, store_t *s, int k) {
    double u = U(r);
    if (u < w->poth) object(w, r, s, k, k);
    else if (u < w->poth + 0.5 * w->pseal * (1 - w->poth)) device(w, r, s, k, k);
    else tablet(w, r, s, k, E_LOCAL, -1, w->mtab, w->mul, w->ptotl);
}
static void event(world_t *w, rng_t *r, store_t *s, int k) {
    int t = w->type, ro = w->role[k];
    if (t == 5) { local_event(w, r, s, k); return; }
    if (U(r) < w->ifrac) {
        if (ro == 1) {
            int d = pick_dep(w, r, k);
            if (t == 3) { /* temple: offerings come in; lists of contributors, dedications */
                if (d >= 0 && U(r) < 0.5) { object(w, r, s, d, k); return; }
                if (d >= 0) { tablet(w, r, s, k, E_RECEIPT, d, w->mtab * 1.5, w->muc, w->ptotc); return; }
            } else {
                if (U(r) < w->rr) { tablet(w, r, s, k, E_RATION, -1, w->mtab * 1.5, w->mul, w->ptotc); return; }
                if (d >= 0) { tablet(w, r, s, k, E_RECEIPT, d, w->mtab * 1.5, w->muc, w->ptotc); return; }
            }
        } else if (ro == 2) {
            int c = w->parent[k];
            if (t == 1 && U(r) < 0.5) { tablet(w, r, s, k, E_RATION, -1, w->mtab * 1.5, w->mul, w->ptotc); return; }
            if (U(r) < w->pseal) { device(w, r, s, k, U(r) < w->ptrav ? c : k); return; }
            tablet(w, r, s, k, E_DISPATCH, c, w->mtab, w->muc, w->ptotl); return;
        }
    }
    /* lateral trade (merchant houses, peer centres) */
    int np = 0, ps[MAXK]; for (int l = 0; l < w->K; l++) if (w->lat[k][l]) ps[np++] = l;
    if (np && U(r) < w->latshare) {
        int l = ps[RI(r, np)];
        if (U(r) < w->pseal) device(w, r, s, k, U(r) < w->ptrav ? l : k);
        else tablet(w, r, s, k, E_LATERAL, l, w->mtab, w->muc, w->ptotl);
        return;
    }
    if (U(r) < w->trav_local && w->type != 5) { /* local sealed goods sent anywhere nearby */
        device(w, r, s, k, RI(r, w->K)); return; }
    local_event(w, r, s, k);
}

/* generate + thin; returns surviving store in out (indices of kept docs) */
static int gen_world(world_t *w, rng_t *r, const int *nk, store_t *s, int *keep) {
    int K = w->K; int cnt[MAXK] = {0};
    for (int k = 0; k < K; k++) { int P = (int)ceil(w->prod * nk[k]) + 2; for (int i = 0; i < P; i++) event(w, r, s, k); }
    for (int i = 0; i < s->n; i++) cnt[s->site[i]]++;
    for (int k = 0; k < K; k++) while (cnt[k] < nk[k] + 1) { int n0 = s->n; local_event(w, r, s, k); if (s->n > n0) cnt[s->site[s->n - 1]]++; }
    /* weighted thinning per site (Efraimidis-Spirakis) */
    double *key = malloc(sizeof(double) * s->n); int nkeep = 0;
    for (int i = 0; i < s->n; i++) { double wt = s->dtype[i] == 0 ? w->svt : s->dtype[i] == 1 ? w->svr : s->dtype[i] == 2 ? w->svn : 1.0;
        key[i] = log(U(r) + 1e-300) / wt; }
    for (int k = 0; k < K; k++) {
        /* select nk[k] largest keys among site k */
        int m = 0; int *idx = malloc(sizeof(int) * cnt[k]);
        for (int i = 0; i < s->n; i++) if (s->site[i] == k) idx[m++] = i;
        /* partial selection sort is fine at these sizes? use qsort-like via simple sort */
        for (int a = 0; a < nk[k] && a < m; a++) { int b = a; for (int c = a + 1; c < m; c++) if (key[idx[c]] > key[idx[b]]) b = c;
            int t = idx[a]; idx[a] = idx[b]; idx[b] = t; keep[nkeep++] = idx[a]; }
        free(idx);
    }
    free(key);
    return nkeep;
}

/* ---------------------------------------------------------------- statistic panel */
#define PS 13
int la58_nstat(int K) { return K * PS + K * (K - 1) / 2 + 2; }

static int cmpd(const void *a, const void *b) { double x = *(double *)a, y = *(double *)b; return (x > y) - (x < y); }

/* docs given by index list (keep) into store-like arrays */
static void stats_core(int K, int n, const int *idx, const int *site, const int *dtype, const int *wst, const int *W,
                       const int *nst, const double *X, int nvocab, double *out) {
    int ns = la58_nstat(K); for (int i = 0; i < ns; i++) out[i] = 0;
    int *freq = calloc(nvocab, sizeof(int)), *smask = calloc(nvocab, sizeof(int)), *dcnt = calloc((size_t)nvocab * K, sizeof(int));
    int *last = malloc(sizeof(int) * nvocab); for (int v = 0; v < nvocab; v++) last[v] = -1;
    for (int ii = 0; ii < n; ii++) { int i = idx ? idx[ii] : ii; for (int j = wst[i]; j < wst[i + 1]; j++) freq[W[j]]++; }
    /* top-30 types excluded from sharing stats */
    int top[30], ntop = 0; char *istop = calloc(nvocab, 1);
    for (int t = 0; t < 30; t++) { int b = -1; for (int v = 0; v < nvocab; v++) if (!istop[v] && freq[v] > 0 && (b < 0 || freq[v] > freq[b])) b = v; if (b < 0) break; istop[b] = 1; top[ntop++] = b; }
    long tottok = 0, toptok = 0;
    for (int ii = 0; ii < n; ii++) { int i = idx ? idx[ii] : ii; int k = site[i];
        for (int j = wst[i]; j < wst[i + 1]; j++) { int v = W[j]; tottok++; if (istop[v]) { toptok++; continue; }
            smask[v] |= 1 << k; if (last[v] != ii) { dcnt[(size_t)v * K + k]++; last[v] = ii; } } }
    double ndoc[MAXK] = {0}, ntab[MAXK] = {0}, nrou[MAXK] = {0}, nnod[MAXK] = {0}, lnum[MAXK] = {0}, tab3[MAXK] = {0}, tot3[MAXK] = {0};
    double tok[MAXK] = {0}, tokx[MAXK] = {0}, dtok[MAXK] = {0}, dtokx[MAXK] = {0}, ftok[MAXK] = {0}, ftokx[MAXK] = {0}, nw[MAXK] = {0}, nwd[MAXK] = {0};
    double *lx[MAXK]; int nlx[MAXK] = {0}, clx[MAXK];
    for (int k = 0; k < K; k++) { clx[k] = 256; lx[k] = malloc(sizeof(double) * clx[k]); }
    for (int ii = 0; ii < n; ii++) { int i = idx ? idx[ii] : ii; int k = site[i], dt = dtype[i];
        ndoc[k]++; nw[k] += wst[i + 1] - wst[i]; if (wst[i + 1] > wst[i]) nwd[k]++;
        if (dt == 1) nrou[k]++; else if (dt == 2) nnod[k]++;
        if (dt == 0) { ntab[k]++; int nn = nst[i + 1] - nst[i]; lnum[k] += log(1.0 + nn);
            for (int j = nst[i]; j < nst[i + 1]; j++) if (X[j] > 0) { if (nlx[k] >= clx[k]) { clx[k] *= 2; lx[k] = realloc(lx[k], sizeof(double) * clx[k]); } lx[k][nlx[k]++] = log10(X[j]); }
            if (nn >= 3) { tab3[k]++; double run = 0; int terms = 0, hit = 0;
                for (int j = nst[i]; j < nst[i + 1]; j++) { if (terms >= 2 && fabs(X[j] - run) < 1e-6 && X[j] > 0) { hit = 1; run = 0; terms = 0; } else { run += X[j]; terms++; } }
                tot3[k] += hit; }
            if (wst[i + 1] > wst[i]) { int v = W[wst[i]]; if (!istop[v]) { ftok[k]++; if (smask[v] & ~(1 << k)) ftokx[k]++; } } }
        for (int j = wst[i]; j < wst[i + 1]; j++) { int v = W[j]; if (istop[v]) continue; int other = (smask[v] & ~(1 << k)) != 0;
            tok[k]++; tokx[k] += other; if (dt == 1 || dt == 2) { dtok[k]++; dtokx[k] += other; } } }
    double types[MAXK] = {0}, rec[MAXK] = {0}; int shared[MAXK][MAXK]; memset(shared, 0, sizeof(shared));
    double multi = 0, alltypes = 0;
    for (int v = 0; v < nvocab; v++) { int m = smask[v]; if (!m) continue; alltypes++; if (m & (m - 1)) multi++;
        for (int k = 0; k < K; k++) if (m >> k & 1) { types[k]++; if (dcnt[(size_t)v * K + k] >= 2) rec[k]++;
            for (int l = k + 1; l < K; l++) if (m >> l & 1) shared[k][l]++; } }
    for (int k = 0; k < K; k++) {
        double *o = out + k * PS, nd = ndoc[k] > 0 ? ndoc[k] : 1;
        o[0] = ntab[k] / nd; o[1] = nrou[k] / nd; o[2] = nnod[k] / nd;
        o[3] = ntab[k] > 0 ? lnum[k] / ntab[k] : -1;
        o[4] = tab3[k] > 0 ? tot3[k] / tab3[k] : -1;
        if (nlx[k]) { qsort(lx[k], nlx[k], sizeof(double), cmpd); o[5] = lx[k][nlx[k] / 2]; } else o[5] = -1;
        o[6] = tok[k] > 0 ? tokx[k] / tok[k] : -1;
        o[7] = types[k] > 0 ? rec[k] / types[k] : -1;
        o[8] = dtok[k] > 0 ? dtokx[k] / dtok[k] : -1;
        o[9] = ftok[k] > 0 ? ftokx[k] / ftok[k] : -1;
        o[10] = nw[k] / nd; o[11] = nwd[k] / nd; o[12] = log(1 + types[k]) / log(2 + ndoc[k]);
        free(lx[k]);
    }
    int p = K * PS;
    for (int k = 0; k < K; k++) for (int l = k + 1; l < K; l++) out[p++] = (types[k] > 0 && types[l] > 0) ? shared[k][l] / sqrt(types[k] * types[l]) : 0;
    out[p++] = alltypes > 0 ? multi / alltypes : 0; out[p++] = tottok > 0 ? (double)toptok / tottok : 0;
    free(freq); free(smask); free(dcnt); free(last); free(istop);
}

void la58_stats(int K, int n, const int *site, const int *dtype, const int *wst, const int *W, const int *nst, const double *X, int nvocab, double *out) {
    stats_core(K, n, NULL, site, dtype, wst, W, nst, X, nvocab > 0 ? nvocab : 1, out);
}

static int vocab_size(world_t *w) { return w->INS0 + w->K * w->nl; }

void la58_batch(int K, const int *nk, int nworld, unsigned long long sd, int ftype, int use_theta, const double *fth, double *S, double *TH) {
    rng_t r; rseed(&r, sd); int tot = 0; for (int k = 0; k < K; k++) tot += nk[k];
    int *keep = malloc(sizeof(int) * (tot + 16)); world_t *w = malloc(sizeof(world_t));
    for (int it = 0; it < nworld; it++) {
        if (use_theta) get_theta(w, fth, K); else draw_world(w, &r, K, ftype);
        store_t s; st_init(&s); int nkp = gen_world(w, &r, nk, &s, keep);
        stats_core(K, nkp, keep, s.site, s.dtype, s.wst, s.W, s.nst, s.X, vocab_size(w), S + (size_t)it * la58_nstat(K));
        put_theta(w, TH + (size_t)it * ntheta_(K)); st_free(&s);
    }
    free(keep); free(w);
}

int la58_world_docs(int K, const int *nk, unsigned long long sd, int ftype, int use_theta, const double *fth, int *site, int *dt, int *wst, int *W, int *WC,
                    int *nst, double *X, double *th) {
    rng_t r; rseed(&r, sd); int tot = 0; for (int k = 0; k < K; k++) tot += nk[k];
    int *keep = malloc(sizeof(int) * (tot + 16)); world_t *w = malloc(sizeof(world_t));
    if (use_theta) get_theta(w, fth, K); else draw_world(w, &r, K, ftype);
    store_t s; st_init(&s); int nkp = gen_world(w, &r, nk, &s, keep);
    int nwp = 0, nxp = 0; wst[0] = 0; nst[0] = 0;
    for (int ii = 0; ii < nkp; ii++) { int i = keep[ii]; site[ii] = s.site[i]; dt[ii] = s.dtype[i];
        for (int j = s.wst[i]; j < s.wst[i + 1] && nwp < tot * 64 - 1; j++) { W[nwp] = s.W[j]; WC[nwp] = s.WC[j]; nwp++; }
        for (int j = s.nst[i]; j < s.nst[i + 1] && nxp < tot * 64 - 1; j++) X[nxp++] = s.X[j];
        wst[ii + 1] = nwp; nst[ii + 1] = nxp; }
    put_theta(w, th); st_free(&s); free(keep); free(w);
    return nkp;
}
