/* LA-24 core: apportionment fits.
 * For one list x[0..n-1] (values in main units, doubles), for every share alphabet S
 * (Sflat with offsets/lengths) and one rule, return the largest number of entries that
 * the rule reproduces with a common unit share u, and the cheapest code length (bits)
 * of a u that achieves it.
 * rules: 0 EXACT  x = u*w exactly (fractions absorb the remainder)
 *        1 FLOOR  x = g*floor(u*w/g)       (divisor method, D'Hondt-like)
 *        2 ROUND  x = g*round(u*w/g)       (divisor method, Sainte-Lague-like)
 *        3 CEIL   x = g*ceil(u*w/g)        (divisor method, Adams-like)
 *        4 HAMIL  largest remainder: T = sum x, u = T/W, floors plus +g to the largest remainders
 * build: gcc -O2 -shared -fPIC -o la24_core.so la24_core.c -lm
 */
#include <math.h>
#include <stdlib.h>
#include <string.h>

#define EPS 1e-7
static const int QS[] = {1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 30, 32, 48, 60};
#define NQ 16

static double elias(long p) { /* Elias gamma length for p >= 1 */
    if (p < 1) p = 1;
    return 2.0 * floor(log2((double)p)) + 1.0;
}

/* cheapest rational p/q (q in QS) inside [lo, hi]; returns bits, or 60 if none */
static double UBEST;
static double ucost(double lo, double hi) {
    double best = 60.0;
    for (int k = 0; k < NQ; k++) {
        int q = QS[k];
        double p = ceil(lo * q - EPS);
        if (p < 1) p = 1;
        if (p / q <= hi + EPS) {
            double c = 1.0 + 2.0 * log2(k + 1.0) + elias((long)p);
            if (c < best) { best = c; UBEST = p / q; }
        }
    }
    return best;
}

typedef struct { double v; int type; int ent; } Ev;  /* type +1 open, -1 close */
static int evcmp(const void* a, const void* b) {
    const Ev* x = a; const Ev* y = b;
    if (x->v < y->v - 1e-12) return -1;
    if (x->v > y->v + 1e-12) return 1;
    return y->type - x->type;  /* opens before closes at the same point */
}

static int on_grid(double x, double g) { double r = x / g; return fabs(r - floor(r + 0.5)) < 1e-6; }

/* one alphabet, per-entry rules 0-3 */
static void fit_one(const double* x, const int* cn, int n, const int* S, int k, int rule, double g,
                    int* cover, double* cost, double* ub) {
    *cover = 0; *cost = 60.0; *ub = 0;
    if (rule == 0) {
        for (int i = 0; i < n; i++) for (int a = 0; a < k; a++) {
            double u = x[i] / S[a];
            if (u <= 0) continue;
            int c = 0;
            for (int j = 0; j < n; j++) {
                int hit = 0;
                for (int b = 0; b < k && !hit; b++) if (fabs(u * S[b] - x[j]) < EPS * (1 + x[j])) hit = 1;
                c += hit * cn[j];
            }
            double cc = ucost(u, u);
            if (c > *cover || (c == *cover && cc < *cost)) { *cover = c; *cost = cc; *ub = u; }
        }
        return;
    }
    Ev* ev = malloc(sizeof(Ev) * 2 * n * k);
    int m = 0;
    for (int i = 0; i < n; i++) {
        if (!on_grid(x[i], g)) continue;
        for (int a = 0; a < k; a++) {
            double w = S[a], lo, hi;
            if (rule == 1) { lo = x[i] / w; hi = (x[i] + g) / w - 1e-9; }
            else if (rule == 2) { lo = (x[i] - g / 2) / w; hi = (x[i] + g / 2) / w - 1e-9; }
            else { lo = (x[i] - g) / w + 1e-9; hi = x[i] / w; }
            if (hi <= 0) continue;
            if (lo < 1e-9) lo = 1e-9;
            ev[m].v = lo; ev[m].type = 1; ev[m].ent = i; m++;
            ev[m].v = hi; ev[m].type = -1; ev[m].ent = i; m++;
        }
    }
    qsort(ev, m, sizeof(Ev), evcmp);
    int* cnt = calloc(n, sizeof(int));
    int covered = 0;
    for (int e = 0; e < m; e++) {
        if (ev[e].type == 1) {
            if (cnt[ev[e].ent]++ == 0) covered += cn[ev[e].ent];
            if (covered >= *cover) {
                /* region from ev[e].v to next event */
                double lo = ev[e].v, hi = (e + 1 < m) ? ev[e + 1].v : lo;
                double cc = ucost(lo, hi);
                if (covered > *cover || cc < *cost) { *cover = covered; *cost = cc; *ub = UBEST; }
            }
        } else {
            if (--cnt[ev[e].ent] == 0) covered -= cn[ev[e].ent];
        }
    }
    free(cnt); free(ev);
}

/* largest remainder */
static void fit_hamil(const double* x, int n, const int* S, int k, double g, int* cover, double* cost, double* ub) {
    *cover = 0; *cost = 60.0; *ub = 0;
    double T = 0; for (int i = 0; i < n; i++) { if (!on_grid(x[i], g)) return; T += x[i]; }
    int smin = S[0], smax = S[0];
    for (int a = 1; a < k; a++) { if (S[a] < smin) smin = S[a]; if (S[a] > smax) smax = S[a]; }
    int* w = malloc(sizeof(int) * n);
    double* rem = malloc(sizeof(double) * n);
    int* up = malloc(sizeof(int) * n);
    long Tg = lround(T / g);
    for (int W = n * smin; W <= n * smax; W++) {
        double u = T / W;  int ok = 1, sw = 0;
        for (int i = 0; i < n && ok; i++) {
            int found = 0;
            for (int a = 0; a < k && !found; a++) {
                double q = u * S[a] / g, f = floor(q + 1e-9);
                double xi = x[i] / g;
                if (fabs(xi - f) < 1e-6) { w[i] = S[a]; rem[i] = q - f; up[i] = 0; found = 1; }
                else if (fabs(xi - f - 1) < 1e-6) { w[i] = S[a]; rem[i] = q - f; up[i] = 1; found = 1; }
            }
            if (!found) ok = 0; else sw += w[i];
        }
        if (!ok || sw != W) continue;
        /* every 'up' entry must have remainder >= every non-up entry */
        double minup = 2, maxdn = -1; int nup = 0;
        for (int i = 0; i < n; i++) {
            if (up[i]) { nup++; if (rem[i] < minup) minup = rem[i]; }
            else if (rem[i] > maxdn) maxdn = rem[i];
        }
        if (nup > 0 && minup < maxdn - 1e-9) continue;
        if (nup > 0 && minup < 1e-9) continue;  /* +1 on an exact quota is not largest remainder */
        *cover = n; *cost = 1.0 + elias(Tg); *ub = u;
        break;
    }
    free(w); free(rem); free(up);
}

/* fit all alphabets; out arrays of length nS */
void fit_all(const double* x, int n, const int* Sflat, const int* Soff, const int* Slen, int nS,
             int rule, double g, int* cover, double* cost, double* ub) {
    double* xd = malloc(sizeof(double) * n); int* cn = malloc(sizeof(int) * n); int nd = 0;
    for (int i = 0; i < n; i++) {
        int f = -1;
        for (int j = 0; j < nd; j++) if (fabs(xd[j] - x[i]) < 1e-9) { f = j; break; }
        if (f < 0) { xd[nd] = x[i]; cn[nd] = 1; nd++; } else cn[f]++;
    }
    for (int s = 0; s < nS; s++) {
        if (rule == 4) fit_hamil(x, n, Sflat + Soff[s], Slen[s], g, cover + s, cost + s, ub + s);
        else fit_one(xd, cn, nd, Sflat + Soff[s], Slen[s], rule, g, cover + s, cost + s, ub + s);
    }
    free(xd); free(cn);
}

/* the share vector chosen for one alphabet and rule at the best u (for reporting) */
void assign_one(const double* x, int n, const int* S, int k, int rule, double g, double u, int* w) {
    for (int i = 0; i < n; i++) {
        w[i] = 0;
        for (int a = 0; a < k; a++) {
            double v = u * S[a], r;
            if (rule == 0) r = v;
            else if (rule == 1) r = g * floor(v / g + 1e-9);
            else if (rule == 2) r = g * floor(v / g + 0.5);
            else r = g * ceil(v / g - 1e-9);
            if (fabs(r - x[i]) < 1e-6 * (1 + x[i])) { w[i] = S[a]; break; }
        }
    }
}
