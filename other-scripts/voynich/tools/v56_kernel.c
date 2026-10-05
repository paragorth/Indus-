/* v56 pointer-rule scoring kernel: for each candidate value of one glyph, the pairing-corrected score
   z = (sum_i R[row_i, t_i] - sum_t rbar[t] * hist[t] / n) / sqrt(n)
   kind 0 abs_mod, 1 abs_clip, 2 rel_fwd, 3 rel_back. Build: cc -O3 -shared -fPIC -o v56_kernel.so v56_kernel.c -lm */
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
void cand_scores(int n, const int64_t *b0, const int64_t *cg, int V, const int64_t *vals, int kind, int off, int Nsp,
                 const int32_t *own, const float *R, int ncols, const int64_t *rowidx, const double *rbar, double *out) {
    int32_t *hist = (int32_t *)malloc(sizeof(int32_t) * Nsp);
    double sq = sqrt(n > 0 ? (double)n : 1.0);
    for (int k = 0; k < V; k++) {
        memset(hist, 0, sizeof(int32_t) * Nsp);
        double obs = 0.0;
        int64_t v = vals[k];
        for (int i = 0; i < n; i++) {
            int64_t a = b0[i] + cg[i] * v + off;
            int64_t t;
            if (kind == 0) { t = a % Nsp; if (t < 0) t += Nsp; }
            else if (kind == 1) { t = a; if (t < 0 || t >= Nsp) continue; }
            else if (kind == 2) { if (a == 0) continue; t = own[i] + a; if (t < 0 || t >= Nsp) continue; }
            else { if (a == 0) continue; t = own[i] - a; if (t < 0 || t >= Nsp) continue; }
            obs += R[rowidx[i] * (int64_t)ncols + t];
            hist[t]++;
        }
        double pe = 0.0;
        for (int t = 0; t < Nsp; t++) pe += rbar[t] * hist[t];
        out[k] = (obs - pe / (n > 0 ? n : 1)) / sq;
    }
    free(hist);
}
