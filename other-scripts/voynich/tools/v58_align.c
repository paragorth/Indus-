/* v58: local length alignment (Smith-Waterman on log lengths with merges and skips).
   x[0..n) : Voynich unit lengths (words), y[0..m) : source entry lengths.
   A move takes k Voynich units (1..KX) against l source entries (1..LY):
       d = log(sum x) - log(sum y) - c
       score = lam - d*d/(2 s^2) - pm*(k+l-2)
   Skips: one Voynich unit or one source entry at cost -g.
   Local alignment (floor 0). Scale offset c scanned over a grid; best over grid returned.
   Build: gcc -O3 -shared -fPIC -o v58_align.so v58_align.c -lm
*/
#include <math.h>
#include <stdlib.h>
#include <string.h>

#define KX 2
#define LY 3

static double sw_one(const double *cx, const double *cy, int n, int m, double c,
                     double lam, double s, double pm, double g,
                     int *bi, int *bj, int *trace)
{
    /* cx, cy: cumulative sums (length n+1, m+1) */
    double *H = (double *)malloc(sizeof(double) * (size_t)(n + 1) * (m + 1));
    double best = 0; int b_i = 0, b_j = 0;
    double inv = 1.0 / (2 * s * s);
    for (int i = 0; i <= n; i++) H[(size_t)i * (m + 1)] = 0;
    for (int j = 0; j <= m; j++) H[j] = 0;
    for (int i = 1; i <= n; i++) {
        for (int j = 1; j <= m; j++) {
            double h = 0; int tb = 0;
            for (int k = 1; k <= KX && k <= i; k++) {
                double lx = log(cx[i] - cx[i - k]);
                for (int l = 1; l <= LY && l <= j; l++) {
                    double d = lx - log(cy[j] - cy[j - l]) - c;
                    double v = H[(size_t)(i - k) * (m + 1) + (j - l)] + lam - d * d * inv - pm * (k + l - 2);
                    if (v > h) { h = v; tb = k * 10 + l; }
                }
            }
            double v1 = H[(size_t)(i - 1) * (m + 1) + j] - g;
            if (v1 > h) { h = v1; tb = 100; }
            double v2 = H[(size_t)i * (m + 1) + j - 1] - g;
            if (v2 > h) { h = v2; tb = 200; }
            H[(size_t)i * (m + 1) + j] = h;
            if (trace) trace[(size_t)i * (m + 1) + j] = tb;
            if (h > best) { best = h; b_i = i; b_j = j; }
        }
    }
    free(H);
    if (bi) *bi = b_i;
    if (bj) *bj = b_j;
    return best;
}

/* best score over a grid of c; returns best, writes best c index */
double align_best(const double *x, int n, const double *y, int m,
                  const double *cgrid, int nc, double lam, double s, double pm, double g,
                  int *best_c)
{
    double *cx = (double *)malloc(sizeof(double) * (n + 1));
    double *cy = (double *)malloc(sizeof(double) * (m + 1));
    cx[0] = 0; for (int i = 0; i < n; i++) cx[i + 1] = cx[i] + x[i];
    cy[0] = 0; for (int j = 0; j < m; j++) cy[j + 1] = cy[j] + y[j];
    double best = -1; int bc = 0;
    for (int t = 0; t < nc; t++) {
        double v = sw_one(cx, cy, n, m, cgrid[t], lam, s, pm, g, 0, 0, 0);
        if (v > best) { best = v; bc = t; }
    }
    free(cx); free(cy);
    if (best_c) *best_c = bc;
    return best;
}

/* full traceback at a given c: path written as pairs (i_end, j_end, k, l) into out (max 4*(n+m)); returns number of steps */
int align_path(const double *x, int n, const double *y, int m, double c,
               double lam, double s, double pm, double g, int *out, double *score)
{
    double *cx = (double *)malloc(sizeof(double) * (n + 1));
    double *cy = (double *)malloc(sizeof(double) * (m + 1));
    int *tr = (int *)calloc((size_t)(n + 1) * (m + 1), sizeof(int));
    cx[0] = 0; for (int i = 0; i < n; i++) cx[i + 1] = cx[i] + x[i];
    cy[0] = 0; for (int j = 0; j < m; j++) cy[j + 1] = cy[j] + y[j];
    int bi, bj;
    double best = sw_one(cx, cy, n, m, c, lam, s, pm, g, &bi, &bj, tr);
    /* recompute H to know when to stop: stop when tb==0 */
    int i = bi, j = bj, steps = 0;
    while (i > 0 && j > 0) {
        int tb = tr[(size_t)i * (m + 1) + j];
        if (tb == 0) break;
        if (tb == 100) { out[4 * steps] = i; out[4 * steps + 1] = j; out[4 * steps + 2] = 1; out[4 * steps + 3] = 0; i -= 1; }
        else if (tb == 200) { out[4 * steps] = i; out[4 * steps + 1] = j; out[4 * steps + 2] = 0; out[4 * steps + 3] = 1; j -= 1; }
        else { int k = tb / 10, l = tb % 10; out[4 * steps] = i; out[4 * steps + 1] = j; out[4 * steps + 2] = k; out[4 * steps + 3] = l; i -= k; j -= l; }
        steps++;
    }
    free(cx); free(cy); free(tr);
    *score = best;
    return steps;
}
