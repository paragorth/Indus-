/* LA-11 core: simulated annealing for the best one-to-one map (quadratic assignment).
   Source symbols 0..K-1 are mapped injectively into target symbols 0..M-1 (M >= K);
   fixed: K -> M (OTHER), K+1 -> M+1 (BOS), K+2 -> M+2 (EOS).
   Score(m) = sum_{i,j} C[i][j] * L[m(i)][m(j)]   (C: (K+3)^2 bigram counts, L: (M+3)^2 log2 probabilities)
   Build: gcc -O3 -shared -fPIC -o la11_core.so la11_core.c -lm */
#include <math.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>

static uint64_t rs;
static inline uint64_t xr(void) { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return rs; }
static inline double ur(void) { return (xr() >> 11) * (1.0 / 9007199254740992.0); }

static double score(const double *C, const double *L, const int *m, int K, int M) {
    int n = K + 3, mm = M + 3; double s = 0;
    for (int i = 0; i < n; i++) for (int j = 0; j < n; j++) { double c = C[i * n + j]; if (c != 0) s += c * L[m[i] * mm + m[j]]; }
    return s;
}

/* delta of moving source i from image a to image b (b currently unused or = image of k, handled by caller) */
static double delta_move(const double *C, const double *L, const int *m, int K, int M, int i, int b) {
    int n = K + 3, mm = M + 3, a = m[i]; double d = 0;
    for (int j = 0; j < n; j++) {
        if (j == i) continue;
        double c1 = C[i * n + j], c2 = C[j * n + i];
        if (c1 != 0) d += c1 * (L[b * mm + m[j]] - L[a * mm + m[j]]);
        if (c2 != 0) d += c2 * (L[m[j] * mm + b] - L[m[j] * mm + a]);
    }
    double cii = C[i * n + i];
    if (cii != 0) d += cii * (L[b * mm + b] - L[a * mm + a]);
    return d;
}

/* returns best score; best map written to out_map (length K+3). restarts R, steps S per restart, T0 -> T1 geometric. */
double anneal(const double *C, const double *L, int K, int M, int R, int S, double T0, double T1, uint64_t seed,
              int *out_map, double *rest_scores) {
    int n = K + 3;
    int *m = malloc(sizeof(int) * n), *inv = malloc(sizeof(int) * (M + 3)), *perm = malloc(sizeof(int) * M);
    double best = -1e300;
    rs = seed * 2654435761ULL + 88172645463325252ULL; if (!rs) rs = 1;
    for (int r = 0; r < R; r++) {
        for (int t = 0; t < M; t++) perm[t] = t;
        for (int t = M - 1; t > 0; t--) { int u = xr() % (t + 1); int tmp = perm[t]; perm[t] = perm[u]; perm[u] = tmp; }
        for (int t = 0; t < M + 3; t++) inv[t] = -1;
        for (int i = 0; i < K; i++) { m[i] = perm[i]; inv[perm[i]] = i; }
        m[K] = M; m[K + 1] = M + 1; m[K + 2] = M + 2;
        double cur = score(C, L, m, K, M), rbest = cur;
        double lg = log(T1 / T0);
        for (int s = 0; s < S; s++) {
            double T = T0 * exp(lg * s / (double)S);
            int i = xr() % K, b = xr() % M;
            if (b == m[i]) continue;
            int k = inv[b];
            double d;
            if (k < 0) {
                d = delta_move(C, L, m, K, M, i, b);
                if (d >= 0 || ur() < exp(d / T)) { inv[m[i]] = -1; m[i] = b; inv[b] = i; cur += d; }
            } else {
                int a = m[i];
                /* swap images of i and k: do it, rescore the affected rows exactly */
                double before = 0, after = 0; int nn = K + 3, mm = M + 3;
                for (int j = 0; j < nn; j++) {
                    before += C[i * nn + j] * L[m[i] * mm + m[j]] + C[k * nn + j] * L[m[k] * mm + m[j]];
                    if (j != i && j != k) before += C[j * nn + i] * L[m[j] * mm + m[i]] + C[j * nn + k] * L[m[j] * mm + m[k]];
                }
                m[i] = b; m[k] = a;
                for (int j = 0; j < nn; j++) {
                    after += C[i * nn + j] * L[m[i] * mm + m[j]] + C[k * nn + j] * L[m[k] * mm + m[j]];
                    if (j != i && j != k) after += C[j * nn + i] * L[m[j] * mm + m[i]] + C[j * nn + k] * L[m[j] * mm + m[k]];
                }
                d = after - before;
                if (d >= 0 || ur() < exp(d / T)) { inv[b] = i; inv[a] = k; cur += d; }
                else { m[i] = a; m[k] = b; }
            }
            if (cur > rbest) { rbest = cur; }
            if (cur > best) { best = cur; memcpy(out_map, m, sizeof(int) * n); }
        }
        /* greedy polish */
        int improved = 1;
        while (improved) {
            improved = 0;
            for (int i = 0; i < K; i++) for (int b = 0; b < M; b++) {
                if (b == m[i]) continue;
                int k = inv[b];
                if (k < 0) {
                    double d = delta_move(C, L, m, K, M, i, b);
                    if (d > 1e-9) { inv[m[i]] = -1; m[i] = b; inv[b] = i; cur += d; improved = 1; }
                } else {
                    int a = m[i]; double s0 = score(C, L, m, K, M);
                    m[i] = b; m[k] = a; double s1 = score(C, L, m, K, M);
                    if (s1 > s0 + 1e-9) { inv[b] = i; inv[a] = k; cur = s1; improved = 1; } else { m[i] = a; m[k] = b; }
                }
            }
        }
        cur = score(C, L, m, K, M);
        if (cur > rbest) rbest = cur;
        if (cur > best) { best = cur; memcpy(out_map, m, sizeof(int) * n); }
        if (rest_scores) rest_scores[r] = rbest;
    }
    free(m); free(inv); free(perm);
    return best;
}

double score_map(const double *C, const double *L, const int *m, int K, int M) { return score(C, L, m, K, M); }
