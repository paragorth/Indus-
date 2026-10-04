/* pe38 TYPE INFERENCE core: tempered Gibbs sampler over sign-type roles with a line type-checker.
 * Energy (minimised) E = -log DM(occurrence features | roles)          [off if alpha <= 0]
 *                        - sum_r log(Mt[r] + beta)                      (role-size prior)
 *                        + sum_t cost[t][role_t]                        (type-level rules / learned emission)
 *                        + lam_g * sum_lines viol(line)                 (accounting type-check of every line)
 * Roles: 0 NAM 1 PRO 2 COM 3 ANI 4 MEA 5 HDR 6 VRB 7 TOT 8 QUA.  Line tokens with type -1 are wildcards.
 * Line kinds: 0 header, 1 unnumbered other, 2 entry, 3 total (closure / reserved reverse), 4 other numbered.
 * Build: gcc -O2 -shared -fPIC -o pe38_core.so pe38_core.c -lm
 */
#include <math.h>
#include <stdlib.h>
#include <string.h>

#define MAXR 10
#define MAXV 16
enum { NAM, PRO, COM, ANI, MEA, HDR, VRB, TOT, QUA };

static unsigned long long rs;
static double urand(void) { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return (rs >> 11) * (1.0 / 9007199254740992.0); }

static const int *g_ls, *g_lt, *g_lk, *g_lp;
static int *g_as;
static int *g_lc;      /* per line role counts, R slots */
static int *g_wild;    /* per line wildcard count */
static int g_R;

#define LC(l, r) g_lc[(size_t)(l) * g_R + (r)]
static int heads_mask(int l) {
    int m = 0;
    for (int r = PRO; r <= MEA; r++) if (LC(l, r)) m |= 1 << r;
    return m;
}
/* number of broken accounting rules on line l under the current assignment */
static int line_viol(int l) {
    int k = g_lk[l];
    if (g_ls[l] == g_ls[l + 1]) return 0;
    int v = 0, w = g_wild[l] > 0;
    if (k == 2 || k == 4) {            /* entry: role + number; needs a countable head, no header/total sign */
        int ok = w || LC(l, NAM) || heads_mask(l);
        v += !ok;
        v += (LC(l, HDR) > 0) + (LC(l, TOT) > 0);
    } else if (k == 3) {               /* total: a total marker, or the same head role as the entry it closes */
        int ok = w || LC(l, TOT);
        if (!ok && g_lp[l] >= 0) ok = (heads_mask(l) & heads_mask(g_lp[l])) != 0 || g_wild[g_lp[l]] > 0;
        v += !ok;
        v += LC(l, HDR) > 0;
    } else if (k == 0) {               /* header: names an office, a person or a transaction; never a total */
        int ok = w || LC(l, HDR) || LC(l, VRB) || LC(l, NAM);
        v += !ok;
        v += LC(l, TOT) > 0;
    } else {                            /* unnumbered other line: no total marker */
        v += LC(l, TOT) > 0;
    }
    return v;
}
static void init_lines(int nL, int R) {
    g_R = R;
    g_lc = calloc((size_t)nL * R, sizeof(int)); g_wild = calloc(nL, sizeof(int));
    for (int l = 0; l < nL; l++) for (int j = g_ls[l]; j < g_ls[l + 1]; j++) {
        int t = g_lt[j]; if (t < 0) g_wild[l]++; else LC(l, g_as[t])++;
    }
}
static void free_lines(void) { free(g_lc); free(g_wild); g_lc = NULL; g_wild = NULL; }

int run_chain(int nT, int R, int nF, const int *card, int nO, const int *otype, const int *ofeat,
              const double *cost, int nL, const int *ls, const int *lt, const int *lk, const int *lp,
              const int *dstart, const int *dlist, const int *fixed,
              double alpha, double beta, double lam_g, long steps, long burn, long thin,
              double T0, unsigned long long seed, int *assign, int *samples, int maxs, double *trace)
{
    rs = seed * 2654435761ULL + 88172645463325252ULL; if (!rs) rs = 1;
    for (int i = 0; i < 20; i++) urand();
    g_ls = ls; g_lt = lt; g_lk = lk; g_lp = lp; g_as = assign;
    init_lines(nL, R);
    /* per type: lines it sits on with multiplicity */
    int *tlstart = calloc((size_t)nT + 1, sizeof(int)), *tlcnt = calloc(nT, sizeof(int));
    for (int l = 0; l < nL; l++) for (int j = ls[l]; j < ls[l + 1]; j++) if (lt[j] >= 0) tlcnt[lt[j]]++;
    for (int t = 0; t < nT; t++) tlstart[t + 1] = tlstart[t] + tlcnt[t];
    int *tlline = malloc(sizeof(int) * (tlstart[nT] + 1)); memset(tlcnt, 0, sizeof(int) * nT);
    for (int l = 0; l < nL; l++) for (int j = ls[l]; j < ls[l + 1]; j++) if (lt[j] >= 0) { int t = lt[j]; tlline[tlstart[t] + tlcnt[t]++] = l; }
    int use_dm = alpha > 0;
    /* per-type sparse feature counts */
    int *no = calloc(nT, sizeof(int));
    int *fc = calloc((size_t)nT * nF * MAXV, sizeof(int));
    for (int o = 0; o < nO; o++) { int t = otype[o]; no[t]++; for (int k = 0; k < nF; k++) fc[((size_t)t * nF + k) * MAXV + ofeat[o * nF + k]]++; }
    double *n = calloc((size_t)R * nF * MAXV, sizeof(double));
    double *No = calloc(R, sizeof(double)), *Mt = calloc(R, sizeof(double));
    for (int t = 0; t < nT; t++) {
        int r = assign[t];
        for (int k = 0; k < nF; k++) for (int v = 0; v < card[k]; v++) n[(r * nF + k) * MAXV + v] += fc[((size_t)t * nF + k) * MAXV + v];
        No[r] += no[t]; Mt[r] += 1;
    }
    int *movable = malloc(sizeof(int) * nT), nm = 0;
    for (int t = 0; t < nT; t++) if (!fixed || fixed[t] < 0) movable[nm++] = t;
    double lp_[MAXR];
    long ns = 0;
    for (long s = 0; s < steps && nm > 0; s++) {
        double T = 1.0;
        long anneal = burn / 2;
        if (s < anneal) T = T0 * pow(1.0 / T0, (double)s / anneal);
        int t = movable[(int)(urand() * nm)];
        int r0 = assign[t];
        for (int k = 0; k < nF; k++) for (int v = 0; v < card[k]; v++) n[(r0 * nF + k) * MAXV + v] -= fc[((size_t)t * nF + k) * MAXV + v];
        No[r0] -= no[t]; Mt[r0] -= 1;
        double mx = -1e300;
        for (int r = 0; r < R; r++) {
            double l = 0;
            if (use_dm) {
                for (int k = 0; k < nF; k++) {
                    for (int v = 0; v < card[k]; v++) {
                        int c = fc[((size_t)t * nF + k) * MAXV + v];
                        if (c) { double a = n[(r * nF + k) * MAXV + v]; l += lgamma(a + c + alpha) - lgamma(a + alpha); }
                    }
                    l -= lgamma(No[r] + no[t] + card[k] * alpha) - lgamma(No[r] + card[k] * alpha);
                }
            }
            l += log(Mt[r] + beta);
            l -= cost[(size_t)t * R + r];
            if (lam_g > 0) {
                int prev = assign[t];
                if (prev != r) { for (int j = tlstart[t]; j < tlstart[t + 1]; j++) { LC(tlline[j], prev)--; LC(tlline[j], r)++; } assign[t] = r; }
                int vv = 0;
                for (int j = dstart[t]; j < dstart[t + 1]; j++) vv += line_viol(dlist[j]);
                l -= lam_g * vv;
            }
            lp_[r] = l / T; if (lp_[r] > mx) mx = lp_[r];
        }
        double z = 0; for (int r = 0; r < R; r++) { lp_[r] = exp(lp_[r] - mx); z += lp_[r]; }
        double u = urand() * z; int r1 = R - 1;
        for (int r = 0; r < R; r++) { u -= lp_[r]; if (u <= 0) { r1 = r; break; } }
        if (assign[t] != r1) { int prev = assign[t]; for (int j = tlstart[t]; j < tlstart[t + 1]; j++) { LC(tlline[j], prev)--; LC(tlline[j], r1)++; } }
        assign[t] = r1;
        for (int k = 0; k < nF; k++) for (int v = 0; v < card[k]; v++) n[(r1 * nF + k) * MAXV + v] += fc[((size_t)t * nF + k) * MAXV + v];
        No[r1] += no[t]; Mt[r1] += 1;
        if (s >= burn && ((s - burn) % thin) == 0 && ns < maxs) { memcpy(samples + (size_t)ns * nT, assign, sizeof(int) * nT); ns++; }
    }
    /* final total line violations */
    if (trace) { int vv = 0; for (int l = 0; l < nL; l++) vv += line_viol(l); trace[0] = vv; }
    free(no); free(fc); free(n); free(No); free(Mt); free(movable); free(tlstart); free(tlcnt); free(tlline); free_lines();
    return (int)ns;
}

/* total violations for an assignment (for scoring) */
int count_viol(int nL, const int *ls, const int *lt, const int *lk, const int *lp, int *assign) {
    g_ls = ls; g_lt = lt; g_lk = lk; g_lp = lp; g_as = assign;
    init_lines(nL, MAXR);
    int vv = 0; for (int l = 0; l < nL; l++) vv += line_viol(l);
    free_lines();
    return vv;
}
