/* v27 card sequencing: best Hamiltonian path with node 0 fixed as start (free end).
   Exact Held-Karp DP for n-1 <= 14 body nodes, else simulated annealing with restarts.
   Also a large-n global open-path SA (free start, free end) with O(1) move deltas.   */
#include <stdlib.h>
#include <string.h>
#include <math.h>

static unsigned long long rs;
static inline unsigned long long xr(void){ rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return rs; }
static inline double ur(void){ return (xr() >> 11) * (1.0/9007199254740992.0); }
static inline int ri(int n){ return (int)(xr() % (unsigned long long)n); }

static double pcost(int n, const double *M, const int *p){
    double s = 0; for (int i = 0; i + 1 < n; i++) s += M[p[i]*n + p[i+1]]; return s;
}

static double exact(int n, const double *M, int *out){
    int m = n - 1;                    /* body nodes 1..m, bit b <-> node b+1 */
    int S = 1 << m;
    double *D = malloc(sizeof(double) * S * m); int *B = malloc(sizeof(int) * S * m);
    for (int i = 0; i < S*m; i++) D[i] = -1e300;
    for (int b = 0; b < m; b++){ D[(1<<b)*m + b] = M[0*n + b+1]; B[(1<<b)*m + b] = -1; }
    for (int s = 1; s < S; s++) for (int b = 0; b < m; b++){
        double v = D[s*m + b]; if (v < -1e299) continue;
        for (int c = 0; c < m; c++){ if (s & (1<<c)) continue;
            int t = s | (1<<c); double w = v + M[(b+1)*n + c+1];
            if (w > D[t*m + c]){ D[t*m + c] = w; B[t*m + c] = b; } }
    }
    int full = S - 1, bb = 0; double best = -1e300;
    for (int b = 0; b < m; b++) if (D[full*m + b] > best){ best = D[full*m + b]; bb = b; }
    int s = full, k = m; out[0] = 0;
    while (bb >= 0){ out[k--] = bb + 1; int pb = B[s*m + bb]; s &= ~(1<<bb); bb = pb; }
    free(D); free(B); return best;
}

static double anneal(int n, const double *M, int restarts, int iters, int *out){
    int *p = malloc(sizeof(int)*n), *q = malloc(sizeof(int)*n);
    double best = -1e300;
    double mu = 0, sd = 0; int cnt = 0;
    for (int i = 0; i < n*n; i++){ if (i % (n+1)) { mu += M[i]; cnt++; } }
    mu /= cnt; for (int i = 0; i < n*n; i++) if (i % (n+1)) sd += (M[i]-mu)*(M[i]-mu);
    sd = sqrt(sd / cnt) + 1e-9;
    for (int r = 0; r < restarts; r++){
        for (int i = 0; i < n; i++) p[i] = i;
        for (int i = n-1; i > 1; i--){ int j = 1 + ri(i); int t = p[i]; p[i] = p[j]; p[j] = t; }
        double cur = pcost(n, M, p);
        double T0 = sd, T1 = sd * 1e-3;
        for (int it = 0; it < iters; it++){
            double T = T0 * pow(T1/T0, (double)it/iters);
            memcpy(q, p, sizeof(int)*n);
            int mv = ri(3), i = 1 + ri(n-1), j = 1 + ri(n-1);
            if (i == j) continue;
            if (i > j){ int t = i; i = j; j = t; }
            if (mv == 0){ for (int a = i, b = j; a < b; a++, b--){ int t = q[a]; q[a] = q[b]; q[b] = t; } }
            else if (mv == 1){ int t = q[i]; q[i] = q[j]; q[j] = t; }
            else { /* move segment [i..i+L-1] to position j (or-opt) */
                int L = 1 + ri(3); if (i + L - 1 >= n) L = n - i;
                int seg[3]; for (int a = 0; a < L; a++) seg[a] = p[i+a];
                int rest[4096], m = 0; for (int a = 1; a < n; a++) if (a < i || a >= i+L) rest[m++] = p[a];
                int pos = ri(m+1); int k = 1;
                for (int a = 0; a < pos; a++) q[k++] = rest[a];
                for (int a = 0; a < L; a++) q[k++] = seg[a];
                for (int a = pos; a < m; a++) q[k++] = rest[a];
            }
            double nw = pcost(n, M, q);
            if (nw >= cur || ur() < exp((nw-cur)/T)){ memcpy(p, q, sizeof(int)*n); cur = nw; }
        }
        if (cur > best){ best = cur; memcpy(out, p, sizeof(int)*n); }
    }
    free(p); free(q); return best;
}

double solve_path(int n, const double *M, int restarts, int iters, unsigned long long seed, int *out){
    rs = seed * 2654435761ULL + 88172645463325252ULL;
    if (n <= 1){ out[0] = 0; return 0; }
    if (n - 1 <= 14) return exact(n, M, out);
    return anneal(n, M, restarts, iters, out);
}

/* ---- global open path over N cards, float matrix (row i -> col j), free start and end.
   Moves: or-opt segment move (orientation kept) and node swap; O(1) deltas, O(N) applying.   */
static const float *GM; static int GN;
#define E(a,b) ((a) < 0 || (b) < 0 ? 0.0 : (double)GM[(long)(a)*GN + (b)])

double global_path(int N, const float *M, long iters, double T0, double T1, unsigned long long seed, int *p){
    rs = seed * 2654435761ULL + 88172645463325252ULL; GM = M; GN = N;
    /* p holds an initial permutation on entry */
    double cur = 0; for (int i = 0; i + 1 < N; i++) cur += E(p[i], p[i+1]);
    int *tmp = malloc(sizeof(int) * N);
    for (long it = 0; it < iters; it++){
        double T = T0 * pow(T1/T0, (double)it/iters);
        int mv = ri(2);
        if (mv == 0){ /* swap positions i<j */
            int i = ri(N), j = ri(N); if (i == j) continue; if (i > j){ int t=i;i=j;j=t; }
            int a = p[i], b = p[j];
            int ip = i>0?p[i-1]:-1, in = p[i+1], jp = p[j-1], jn = j+1<N?p[j+1]:-1;
            double d;
            if (j == i+1) d = E(ip,b)+E(b,a)+E(a,jn) - E(ip,a)-E(a,b)-E(b,jn);
            else d = E(ip,b)+E(b,in)+E(jp,a)+E(a,jn) - E(ip,a)-E(a,in)-E(jp,b)-E(b,jn);
            if (d >= 0 || ur() < exp(d/T)){ p[i] = b; p[j] = a; cur += d; }
        } else { /* or-opt: segment [i..i+L-1] moved between positions k-1 and k (k outside segment) */
            int L = 1 + ri(8); int i = ri(N - L + 1); int k = ri(N + 1);
            if (k >= i && k <= i + L) continue;
            int s0 = p[i], s1 = p[i+L-1];
            int pv = i>0?p[i-1]:-1, nx = i+L<N?p[i+L]:-1;
            int kp = k>0?p[k-1]:-1, kn = k<N?p[k]:-1;
            double d = E(pv,nx) - E(pv,s0) - E(s1,nx) + E(kp,s0) + E(s1,kn) - E(kp,kn);
            if (d >= 0 || ur() < exp(d/T)){
                memcpy(tmp, p+i, sizeof(int)*L);
                if (k < i){ memmove(p+k+L, p+k, sizeof(int)*(i-k)); memcpy(p+k, tmp, sizeof(int)*L); }
                else { memmove(p+i, p+i+L, sizeof(int)*(k-i-L)); memcpy(p+k-L, tmp, sizeof(int)*L); }
                cur += d;
            }
        }
    }
    free(tmp);
    cur = 0; for (int i = 0; i + 1 < N; i++) cur += E(p[i], p[i+1]);
    return cur;
}
