/* LA-40 type inference core: tempered Gibbs sampler over word-type roles.
 * Energy E = -log DM(occurrence features | roles) - log DM(type-level bin | roles)
 *            - log DM(role sizes) + sum_t viol[t][role_t] + lam_h * sum_docs heterogeneity
 * Heterogeneity of a doc = (#entry heads typed P/L/C) - (max count among P, L, C).
 * Build: gcc -O2 -shared -fPIC -o la40_core.so la40_core.c -lm
 */
#include <math.h>
#include <stdlib.h>
#include <string.h>

#define MAXR 8
#define MAXF 8
#define MAXV 16

static unsigned long long rs;
static double urand(void) { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return (rs >> 11) * (1.0 / 9007199254740992.0); }

/* sparse per-type data */
typedef struct { int nc; int *k, *v, *c; int no; int nh; int *hd, *hc; } TT;

int run_chain(int nT, int R, int nF, const int *card, int nO, const int *otype, const int *ofeat,
              const int *ohead_doc, int nD, const double *viol, const int *tbin, int tcard,
              const int *hetero_mask /* R: 1 if role counts for homogeneity */,
              double alpha, double beta, double lam_h, long steps, long burn, long thin,
              double T0, unsigned long long seed, int *assign, int *samples, int maxs)
{
    rs = seed * 2654435761ULL + 88172645463325252ULL; if (!rs) rs = 1;
    for (int i = 0; i < 20; i++) urand();
    TT *tt = calloc(nT, sizeof(TT));
    /* per type feature counts */
    int *tmp = calloc((size_t)nF * MAXV, sizeof(int));
    int *hcnt = calloc(nD, sizeof(int));
    for (int t = 0; t < nT; t++) { tt[t].no = 0; }
    for (int o = 0; o < nO; o++) tt[otype[o]].no++;
    /* build per-type: gather occurrences */
    int *start = calloc((size_t)nT + 1, sizeof(int)), *ord = calloc((size_t)(nO > 0 ? nO : 1), sizeof(int)), *fill = calloc(nT, sizeof(int));
    for (int t = 0; t < nT; t++) start[t + 1] = start[t] + tt[t].no;
    for (int o = 0; o < nO; o++) { int t = otype[o]; ord[start[t] + fill[t]++] = o; }
    for (int t = 0; t < nT; t++) {
        memset(tmp, 0, sizeof(int) * nF * MAXV);
        int nh = 0;
        for (int j = start[t]; j < start[t + 1]; j++) {
            int o = ord[j];
            for (int k = 0; k < nF; k++) tmp[k * MAXV + ofeat[o * nF + k]]++;
            if (ohead_doc[o] >= 0) { if (hcnt[ohead_doc[o]] == 0) nh++; hcnt[ohead_doc[o]]++; }
        }
        int nc = 0;
        for (int k = 0; k < nF; k++) for (int v = 0; v < card[k]; v++) if (tmp[k * MAXV + v]) nc++;
        tt[t].nc = nc; tt[t].k = malloc(sizeof(int) * (nc + 1)); tt[t].v = malloc(sizeof(int) * (nc + 1)); tt[t].c = malloc(sizeof(int) * (nc + 1));
        nc = 0;
        for (int k = 0; k < nF; k++) for (int v = 0; v < card[k]; v++) if (tmp[k * MAXV + v]) { tt[t].k[nc] = k; tt[t].v[nc] = v; tt[t].c[nc] = tmp[k * MAXV + v]; nc++; }
        tt[t].nh = nh; tt[t].hd = malloc(sizeof(int) * (nh + 1)); tt[t].hc = malloc(sizeof(int) * (nh + 1));
        nh = 0;
        for (int j = start[t]; j < start[t + 1]; j++) {
            int o = ord[j], d = ohead_doc[o];
            if (d >= 0 && hcnt[d] > 0) { tt[t].hd[nh] = d; tt[t].hc[nh] = hcnt[d]; hcnt[d] = 0; nh++; }
        }
    }
    /* state counts */
    double *n = calloc((size_t)R * nF * MAXV, sizeof(double));
    double *No = calloc(R, sizeof(double));
    double *m = calloc((size_t)R * MAXV, sizeof(double));
    double *Mt = calloc(R, sizeof(double));
    int *dc = calloc((size_t)nD * R, sizeof(int));
    for (int t = 0; t < nT; t++) {
        int r = assign[t];
        for (int j = 0; j < tt[t].nc; j++) n[(r * nF + tt[t].k[j]) * MAXV + tt[t].v[j]] += tt[t].c[j];
        No[r] += tt[t].no; m[r * MAXV + tbin[t]] += 1; Mt[r] += 1;
        for (int j = 0; j < tt[t].nh; j++) dc[tt[t].hd[j] * R + r] += tt[t].hc[j];
    }
    double lp[MAXR];
    long ns = 0;
    for (long s = 0; s < steps; s++) {
        double T = 1.0;
        long anneal = burn / 2;
        if (s < anneal) T = T0 * pow(1.0 / T0, (double)s / anneal);
        int t = (int)(urand() * nT);
        int r0 = assign[t];
        /* remove */
        for (int j = 0; j < tt[t].nc; j++) n[(r0 * nF + tt[t].k[j]) * MAXV + tt[t].v[j]] -= tt[t].c[j];
        No[r0] -= tt[t].no; m[r0 * MAXV + tbin[t]] -= 1; Mt[r0] -= 1;
        for (int j = 0; j < tt[t].nh; j++) dc[tt[t].hd[j] * R + r0] -= tt[t].hc[j];
        double mx = -1e300;
        for (int r = 0; r < R; r++) {
            double l = 0;
            for (int j = 0; j < tt[t].nc; j++) {
                double a = n[(r * nF + tt[t].k[j]) * MAXV + tt[t].v[j]];
                l += lgamma(a + tt[t].c[j] + alpha) - lgamma(a + alpha);
            }
            for (int k = 0; k < nF; k++) l -= lgamma(No[r] + tt[t].no + card[k] * alpha) - lgamma(No[r] + card[k] * alpha);
            l += log(m[r * MAXV + tbin[t]] + alpha) - log(Mt[r] + tcard * alpha);
            l += log(Mt[r] + beta);
            l -= viol[t * R + r];
            if (hetero_mask[r] && lam_h > 0) {
                double dh = 0;
                for (int j = 0; j < tt[t].nh; j++) {
                    int d = tt[t].hd[j], c = tt[t].hc[j];
                    int tot = 0, mxb = 0, tot2 = 0, mxa = 0;
                    for (int q = 0; q < R; q++) if (hetero_mask[q]) {
                        int x = dc[d * R + q]; tot += x; if (x > mxb) mxb = x;
                        int y = x + (q == r ? c : 0); tot2 += y; if (y > mxa) mxa = y;
                    }
                    dh += (tot2 - mxa) - (tot - mxb);
                }
                l -= lam_h * dh;
            }
            lp[r] = l / T; if (lp[r] > mx) mx = lp[r];
        }
        double z = 0; for (int r = 0; r < R; r++) { lp[r] = exp(lp[r] - mx); z += lp[r]; }
        double u = urand() * z; int r1 = R - 1;
        for (int r = 0; r < R; r++) { u -= lp[r]; if (u <= 0) { r1 = r; break; } }
        assign[t] = r1;
        for (int j = 0; j < tt[t].nc; j++) n[(r1 * nF + tt[t].k[j]) * MAXV + tt[t].v[j]] += tt[t].c[j];
        No[r1] += tt[t].no; m[r1 * MAXV + tbin[t]] += 1; Mt[r1] += 1;
        for (int j = 0; j < tt[t].nh; j++) dc[tt[t].hd[j] * R + r1] += tt[t].hc[j];
        if (s >= burn && ((s - burn) % thin) == 0 && ns < maxs) { memcpy(samples + (size_t)ns * nT, assign, sizeof(int) * nT); ns++; }
    }
    for (int t = 0; t < nT; t++) { free(tt[t].k); free(tt[t].v); free(tt[t].c); free(tt[t].hd); free(tt[t].hc); }
    free(tt); free(tmp); free(hcnt); free(start); free(ord); free(fill); free(n); free(No); free(m); free(Mt); free(dc);
    return (int)ns;
}
