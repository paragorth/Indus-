/* LA-19 core: collapsed Gibbs sampler for a mixture of K sign-level bigram "languages".
   Each word = BOS s1 s2 ... sm EOS. Component k has a transition table prev -> next
   (prev in signs + BOS, next in signs + EOS), Dirichlet(beta * g[next]) prior, with g the corpus-wide
   next-symbol distribution. Initial-sign preferences, final-sign preferences and length (via EOS) are
   all carried by the BOS row and the EOS column. Mixture weights ~ Dirichlet(alpha).
   Build: gcc -O2 -shared -fPIC -o la19_core.so la19_core.c -lm */
#include <math.h>
#include <stdlib.h>
#include <string.h>

static unsigned long long rs;
static double urand(void){ rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return (rs >> 11) * (1.0/9007199254740992.0); }

/* word log predictive under component k given counts (sequential, handles repeats inside the word) */
static double wlp(int k, const int *w, int m, int V, const double *g, double beta,
                  int *n, int *r, int add)
{
    int W = V + 1; double lp = 0; int j, p = V, x;
    int *nk = n + (long)k * W * W, *rk = r + (long)k * W;
    for (j = 0; j <= m; j++) {
        x = (j < m) ? w[j] : V;
        lp += log((nk[p * W + x] + beta * g[x]) / (rk[p] + beta));
        if (add) { nk[p * W + x]++; rk[p]++; }
        p = (j < m) ? w[j] : V;
    }
    if (add) { /* undo temporary adds */
        p = V;
        for (j = 0; j <= m; j++) { x = (j < m) ? w[j] : V; nk[p * W + x]--; rk[p]--; p = (j < m) ? w[j] : V; }
    }
    return lp;
}

static void upd(int k, const int *w, int m, int V, int *n, int *r, int d)
{
    int W = V + 1, j, p = V, x;
    int *nk = n + (long)k * W * W, *rk = r + (long)k * W;
    for (j = 0; j <= m; j++) { x = (j < m) ? w[j] : V; nk[p * W + x] += d; rk[p] += d; p = (j < m) ? w[j] : V; }
}

/* N train words (off[N+1], sym), H held-out words (hoff[H+1], hsym).
   Returns final training joint log-lik; z (len N) gets final assignment;
   held_lp (len H) gets log of posterior-averaged predictive prob (avg over samples after burn, every thin);
   zprob (N x K) gets assignment frequencies over the same samples. */
double gibbs(int N, const int *off, const int *sym, int H, const int *hoff, const int *hsym,
             int V, int K, double alpha, double beta, const double *g, int sweeps, int burn, int thin,
             unsigned long long seed, int *z, double *held_lp, double *zprob, const int *zinit)
{
    int W = V + 1, i, k, s, h;
    int *n = calloc((long)K * W * W, sizeof(int)), *r = calloc((long)K * W, sizeof(int)), *mk = calloc(K, sizeof(int));
    double *lp = malloc(sizeof(double) * K), *acc = calloc(H > 0 ? H : 1, sizeof(double)), *ref = malloc(sizeof(double) * (H > 0 ? H : 1));
    int nsamp = 0;
    rs = seed * 2654435761ULL + 88172645463325252ULL; for (i = 0; i < 10; i++) urand();
    for (i = 0; i < N; i++) { z[i] = zinit ? zinit[i] : (int)(urand() * K); if (z[i] >= K) z[i] = K - 1; upd(z[i], sym + off[i], off[i+1]-off[i], V, n, r, 1); mk[z[i]]++; }
    if (zprob) memset(zprob, 0, sizeof(double) * N * K);
    for (s = 0; s < sweeps; s++) {
        for (i = 0; i < N; i++) {
            const int *w = sym + off[i]; int m = off[i+1] - off[i];
            upd(z[i], w, m, V, n, r, -1); mk[z[i]]--;
            double mx = -1e300, tot = 0;
            for (k = 0; k < K; k++) { lp[k] = log(mk[k] + alpha) + wlp(k, w, m, V, g, beta, n, r, 1); if (lp[k] > mx) mx = lp[k]; }
            for (k = 0; k < K; k++) { lp[k] = exp(lp[k] - mx); tot += lp[k]; }
            double u = urand() * tot; for (k = 0; k < K - 1; k++) { u -= lp[k]; if (u <= 0) break; }
            z[i] = k; upd(k, w, m, V, n, r, 1); mk[k]++;
        }
        if (s >= burn && ((s - burn) % thin == 0)) {
            for (h = 0; h < H; h++) {
                const int *w = hsym + hoff[h]; int m = hoff[h+1] - hoff[h];
                double mx = -1e300, tot = 0;
                for (k = 0; k < K; k++) { lp[k] = log((mk[k] + alpha) / (N + K * alpha)) + wlp(k, w, m, V, g, beta, n, r, 1); if (lp[k] > mx) mx = lp[k]; }
                for (k = 0; k < K; k++) tot += exp(lp[k] - mx);
                double v = mx + log(tot);
                if (nsamp == 0) { ref[h] = v; acc[h] = 1.0; } else acc[h] += exp(v - ref[h]);
            }
            if (zprob) for (i = 0; i < N; i++) zprob[(long)i * K + z[i]] += 1.0;
            nsamp++;
        }
    }
    for (h = 0; h < H; h++) held_lp[h] = ref[h] + log(acc[h] / nsamp);
    if (zprob) for (i = 0; i < (long)N * K; i++) zprob[i] /= nsamp;
    /* final joint log-lik (collapsed, sequential) */
    double ll = 0;
    memset(n, 0, sizeof(int) * (long)K * W * W); memset(r, 0, sizeof(int) * (long)K * W); memset(mk, 0, sizeof(int) * K);
    for (i = 0; i < N; i++) { const int *w = sym + off[i]; int m = off[i+1]-off[i];
        ll += log((mk[z[i]] + alpha) / (i + K * alpha)) + wlp(z[i], w, m, V, g, beta, n, r, 1);
        upd(z[i], w, m, V, n, r, 1); mk[z[i]]++; }
    free(n); free(r); free(mk); free(lp); free(acc); free(ref);
    return ll;
}
