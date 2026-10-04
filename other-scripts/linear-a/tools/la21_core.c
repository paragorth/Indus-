/* LA-21 core: let universals build the grid.
 * Signs are assigned blindly to a grid of (R1 = 1 pure-vowel row + R consonant rows) x K vowel columns.
 * Score (nats) of an assignment, all from in-word sign bigram counts B, word-initial counts I, word-final counts F:
 *   LL  = tier-factorised log-likelihood: P(next) = P(row'|row) P(col'|col) P(sign|cell), with word start/end as an
 *         extra boundary state on both tiers (consonant and vowel tiers independent: autosegmental universal).
 *   OCP = w_ocp * sum over consonant rows of (expected - observed) same-row adjacency (distinct signs).
 *   VIN = w_vin * sum over pure-vowel-row signs of (initial count - p0 * token count): pure vowels mainly word-initial.
 *   cell capacity <= cap signs (equal-sized classes; variants allowed up to cap).
 * Simulated annealing, many restarts; returns every restart's assignment and score.
 */
#include <math.h>
#include <stdlib.h>
#include <string.h>

typedef unsigned long long u64;
static u64 rs;
static inline u64 rnd(void) { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return rs; }
static inline double urand(void) { return (rnd() >> 11) * (1.0 / 9007199254740992.0); }

#define MAXS 160
#define MAXR 40
#define MAXK 12

static int S, R1, K, cap;
static const int *B; /* S*S */
static const int *I, *F;
static int e[MAXS], outd[MAXS], ind[MAXS];
static int row[MAXS], col[MAXS];
static int RR[MAXR + 1][MAXR + 1], VV[MAXK + 1][MAXK + 1];
static int cellc[MAXR][MAXK], celln[MAXR][MAXK];
static double OUTr[MAXR], INr[MAXR], OIr[MAXR], SSr[MAXR];
static double wocp, wvin, alpha, p0, Nb;
static double *tfR, *tnR, *tfV, *tnV, *te;
static int NT;

static void place(int s, int r, int v, int sg) {
    /* sg=+1 add sign s at (r,v); sg=-1 remove it */
    int b;
    for (b = 0; b < S; b++) {
        if (b == s) continue;
        int rb = row[b], vb = col[b];
        if (rb < 0) continue;
        int x = B[s * S + b], y = B[b * S + s];
        if (x) { RR[r][rb] += sg * x; VV[v][vb] += sg * x; }
        if (y) { RR[rb][r] += sg * y; VV[vb][v] += sg * y; }
    }
    int self = B[s * S + s];
    RR[r][r] += sg * self; VV[v][v] += sg * self;
    RR[R1][r] += sg * I[s]; VV[K][v] += sg * I[s];
    RR[r][R1] += sg * F[s]; VV[v][K] += sg * F[s];
    cellc[r][v] += sg * e[s]; celln[r][v] += sg;
    OUTr[r] += sg * outd[s]; INr[r] += sg * ind[s]; OIr[r] += sg * (double)outd[s] * ind[s]; SSr[r] += sg * self;
    if (sg > 0) { row[s] = r; col[s] = v; } else { row[s] = -1; col[s] = -1; }
}

static double comp[4];
static double score(void) {
    int x, y;
    double llr = 0, llv = 0, em = 0, ocp = 0, vin = 0;
    for (x = 0; x <= R1; x++) {
        int n = 0;
        for (y = 0; y <= R1; y++) { n += RR[x][y]; llr += tfR[RR[x][y]]; }
        llr -= tnR[n];
    }
    for (x = 0; x <= K; x++) {
        int n = 0;
        for (y = 0; y <= K; y++) { n += VV[x][y]; llv += tfV[VV[x][y]]; }
        llv -= tnV[n];
    }
    for (x = 0; x < R1; x++) for (y = 0; y < K; y++) em -= te[cellc[x][y]];
    for (x = 1; x < R1; x++) {
        double E = (OUTr[x] * INr[x] - OIr[x]) / Nb;
        double O = RR[x][x] - SSr[x];
        ocp += E - O;
    }
    int s;
    for (s = 0; s < S; s++) if (row[s] == 0) vin += I[s] - p0 * e[s];
    comp[0] = llr + llv; comp[1] = em; comp[2] = ocp; comp[3] = vin;
    return llr + llv + em + wocp * ocp + wvin * vin;
}

static void reset(void) {
    memset(RR, 0, sizeof RR); memset(VV, 0, sizeof VV); memset(cellc, 0, sizeof cellc); memset(celln, 0, sizeof celln);
    memset(OUTr, 0, sizeof OUTr); memset(INr, 0, sizeof INr); memset(OIr, 0, sizeof OIr); memset(SSr, 0, sizeof SSr);
    for (int s = 0; s < S; s++) { row[s] = -1; col[s] = -1; }
}

/* run nrest annealing restarts; out_row/out_col: nrest*S, out_score: nrest*5 (total, LL, em, ocp, vin) */
int anneal_many(int S_, int R1_, int K_, int cap_, const int *B_, const int *I_, const int *F_,
                double wocp_, double wvin_, double alpha_, int nrest, long moves, double T0, double T1,
                u64 seed, int *out_row, int *out_col, double *out_score) {
    S = S_; R1 = R1_; K = K_; cap = cap_; B = B_; I = I_; F = F_; wocp = wocp_; wvin = wvin_; alpha = alpha_;
    if (S > MAXS || R1 > MAXR - 1 || K > MAXK - 1) return -1;
    long tot = 0, nb = 0, totI = 0;
    for (int s = 0; s < S; s++) {
        outd[s] = 0; ind[s] = 0;
        for (int b = 0; b < S; b++) { outd[s] += B[s * S + b]; ind[s] += B[b * S + s]; }
        e[s] = I[s] + ind[s];
        tot += e[s] + F[s] + outd[s]; nb += outd[s]; totI += I[s];
    }
    long esum = 0; for (int s = 0; s < S; s++) esum += e[s];
    p0 = (double)totI / esum; Nb = nb > 0 ? nb : 1;
    NT = (int)(tot + 10);
    tfR = malloc(sizeof(double) * NT); tnR = malloc(sizeof(double) * NT); tfV = malloc(sizeof(double) * NT);
    tnV = malloc(sizeof(double) * NT); te = malloc(sizeof(double) * NT);
    for (int n = 0; n < NT; n++) {
        tfR[n] = n * log(n + alpha); tfV[n] = tfR[n];
        tnR[n] = n * log(n + alpha * (R1 + 1)); tnV[n] = n * log(n + alpha * (K + 1));
        te[n] = n > 0 ? n * log((double)n) : 0;
    }
    rs = seed * 0x9E3779B97F4A7C15ULL + 12345; if (!rs) rs = 1;
    for (int it = 0; it < nrest; it++) {
        reset();
        /* random feasible start */
        for (int s = 0; s < S; s++) {
            int r, v, tries = 0;
            do { r = rnd() % R1; v = rnd() % K; tries++; } while (celln[r][v] >= cap && tries < 100000);
            place(s, r, v, +1);
        }
        double cur = score(), best = cur;
        int brow[MAXS], bcol[MAXS];
        memcpy(brow, row, sizeof(int) * S); memcpy(bcol, col, sizeof(int) * S);
        for (long m = 0; m < moves; m++) {
            double T = T0 * pow(T1 / T0, (double)m / moves);
            int typ = rnd() % 4;
            int s = rnd() % S, t = rnd() % S;
            int rs0 = row[s], vs0 = col[s], rt0 = row[t], vt0 = col[t];
            int nrs, nvs, nrt, nvt, two = 0;
            if (typ == 0) { nrs = rnd() % R1; nvs = rnd() % K; if (nrs == rs0 && nvs == vs0) continue;
                if (celln[nrs][nvs] >= cap) continue; }
            else {
                if (s == t) continue; two = 1;
                if (typ == 1) { nrs = rt0; nvs = vt0; nrt = rs0; nvt = vs0; }
                else if (typ == 2) { nrs = rt0; nvs = vs0; nrt = rs0; nvt = vt0; }
                else { nrs = rs0; nvs = vt0; nrt = rt0; nvt = vs0; }
                if (nrs == rs0 && nvs == vs0) continue;
            }
            place(s, rs0, vs0, -1);
            if (two) place(t, rt0, vt0, -1);
            if (two && (celln[nrs][nvs] >= cap)) { place(s, rs0, vs0, +1); place(t, rt0, vt0, +1); continue; }
            place(s, nrs, nvs, +1);
            if (two && celln[nrt][nvt] >= cap) { place(s, nrs, nvs, -1); place(s, rs0, vs0, +1); place(t, rt0, vt0, +1); continue; }
            if (two) place(t, nrt, nvt, +1);
            double nw = score();
            double d = nw - cur;
            if (d >= 0 || urand() < exp(d / T)) {
                cur = nw;
                if (cur > best) { best = cur; memcpy(brow, row, sizeof(int) * S); memcpy(bcol, col, sizeof(int) * S); }
            } else {
                place(s, nrs, nvs, -1); if (two) place(t, nrt, nvt, -1);
                place(s, rs0, vs0, +1); if (two) place(t, rt0, vt0, +1);
            }
        }
        reset();
        for (int s = 0; s < S; s++) place(s, brow[s], bcol[s], +1);
        double sc = score();
        for (int s = 0; s < S; s++) { out_row[it * S + s] = brow[s]; out_col[it * S + s] = bcol[s]; }
        out_score[it * 5 + 0] = sc; out_score[it * 5 + 1] = comp[0]; out_score[it * 5 + 2] = comp[1];
        out_score[it * 5 + 3] = comp[2]; out_score[it * 5 + 4] = comp[3];
    }
    free(tfR); free(tnR); free(tfV); free(tnV); free(te);
    return 0;
}
