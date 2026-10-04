/* LA-8 core: score a typed-slot probabilistic automaton on token streams.
   Tokens carry a lexicon id (or -1 = novel) and a novel-key id. Genome: class of each lexicon type,
   class of each novel key, number of hidden states per class. Emission: class-conditional
   (closed class = listed members only; open class = also emits novel words through its keys).
   Transitions between states: Baum-Welch on training folds. Score: held-out bits (2-fold CV).
   Build: gcc -O2 -shared -fPIC -o la8_core.so la8_core.c -lm */
#include <math.h>
#include <stdlib.h>
#include <string.h>

#define MAXS 128
static unsigned int rs;
static double urand(void){ rs = rs*1103515245u + 12345u; return ((rs>>8)&0xffffff)/16777216.0; }

/* returns heldout bits; out[0]=emission bits, out[1]=transition bits, out[2]=edges (avg over folds),
   if mode==1: fit on all docs, trans_out gets (S+1)x(S+1) probs (rows: states..,START; cols: states..,END),
   out[0], out[1] = training bits. */
double score(int T, int ndoc, const int *doc_off, const int *fold, const int *ttype, const int *tkey,
             const double *logp0, int V, int NK, int K, const int *type_class, const int *key_class,
             const int *nst, double beta, double alpha0, double gamma, double delta, int iters,
             int mode, double *out, double *trans_out, double *emit_out)
{
    int S = 0, first[64], i, t, d, f, s, s2;
    int cls_of_state[MAXS];
    for (i = 0; i < K; i++) { first[i] = S; for (s = 0; s < nst[i]; s++) cls_of_state[S++] = i; }
    if (S > MAXS - 2) return 1e18;
    int R = S + 1, C = S + 1; /* rows: S states + START(S); cols: S states + END(S) */
    int *tc = malloc(sizeof(int) * T);
    for (t = 0; t < T; t++) tc[t] = ttype[t] >= 0 ? type_class[ttype[t]] : key_class[tkey[t]];
    int *M = calloc(K, sizeof(int)), *isopen = calloc(K, sizeof(int)), *nkc = calloc(K, sizeof(int));
    for (i = 0; i < V; i++) M[type_class[i]]++;
    for (i = 0; i < NK; i++) { isopen[key_class[i]] = 1; nkc[key_class[i]]++; }
    double *ncw = malloc(sizeof(double) * V), *Nc = malloc(sizeof(double) * K), *Ac = malloc(sizeof(double) * K);
    double *nck = malloc(sizeof(double) * NK);
    double *A = malloc(sizeof(double) * R * C), *E = malloc(sizeof(double) * R * C);
    int maxn = 0; for (d = 0; d < ndoc; d++) if (doc_off[d+1]-doc_off[d] > maxn) maxn = doc_off[d+1]-doc_off[d];
    double *al = malloc(sizeof(double) * (maxn + 1) * 4), *be = malloc(sizeof(double) * (maxn + 1) * 4), *sc = malloc(sizeof(double) * (maxn + 2));
    double emit_bits = 0, trans_bits = 0, edges = 0;
    int nf = mode == 1 ? 1 : 2;
    for (f = 0; f < nf; f++) {
        /* ---- emission counts on training docs ---- */
        memset(ncw, 0, sizeof(double) * V); memset(Nc, 0, sizeof(double) * K); memset(Ac, 0, sizeof(double) * K); memset(nck, 0, sizeof(double) * NK);
        double ntrain = 0;
        for (d = 0; d < ndoc; d++) { if (mode == 0 && fold[d] == f) continue;
            for (t = doc_off[d]; t < doc_off[d+1]; t++) { ntrain++;
                if (ttype[t] >= 0) { ncw[ttype[t]]++; Nc[tc[t]]++; } else { Ac[tc[t]]++; nck[tkey[t]]++; } } }
        /* ---- Baum-Welch for transitions ---- */
        rs = 12345u;
        for (i = 0; i < R * C; i++) A[i] = 1.0 + 0.5 * urand();
        for (int it = 0; it <= iters; it++) {
            /* normalise A (from E after first pass) */
            for (i = 0; i < R; i++) { double z = 0; for (s = 0; s < C; s++) z += A[i*C+s]; for (s = 0; s < C; s++) A[i*C+s] /= z; }
            if (it == iters) break;
            memset(E, 0, sizeof(double) * R * C);
            for (d = 0; d < ndoc; d++) { if (mode == 0 && fold[d] == f) continue;
                int b = doc_off[d], n = doc_off[d+1] - b; if (n == 0) continue;
                /* forward with scaling; al[t*4 + j] for j-th state of class tc[b+t] */
                for (t = 0; t < n; t++) {
                    int c = tc[b+t], m = nst[c]; double z = 0;
                    for (int j = 0; j < m; j++) { s = first[c] + j; double v = 0;
                        if (t == 0) v = A[S*C + s];
                        else { int cp = tc[b+t-1]; for (int jp = 0; jp < nst[cp]; jp++) v += al[(t-1)*4+jp] * A[(first[cp]+jp)*C + s]; }
                        al[t*4+j] = v; z += v; }
                    if (z <= 0) z = 1e-300;
                    for (int j = 0; j < m; j++) al[t*4+j] /= z; sc[t] = z; }
                { int c = tc[b+n-1]; double z = 0; for (int j = 0; j < nst[c]; j++) z += al[(n-1)*4+j] * A[(first[c]+j)*C + S]; sc[n] = z > 0 ? z : 1e-300; }
                /* backward (scaled with same factors) */
                { int c = tc[b+n-1]; for (int j = 0; j < nst[c]; j++) be[(n-1)*4+j] = A[(first[c]+j)*C + S] / sc[n]; }
                for (t = n - 2; t >= 0; t--) { int c = tc[b+t], cn = tc[b+t+1];
                    for (int j = 0; j < nst[c]; j++) { double v = 0; s = first[c]+j;
                        for (int jn = 0; jn < nst[cn]; jn++) v += A[s*C + first[cn]+jn] * be[(t+1)*4+jn];
                        be[t*4+j] = v / sc[t+1]; } }
                /* expected counts */
                { int c = tc[b]; for (int j = 0; j < nst[c]; j++) E[S*C + first[c]+j] += al[j] * be[j]; }
                for (t = 0; t < n - 1; t++) { int c = tc[b+t], cn = tc[b+t+1];
                    for (int j = 0; j < nst[c]; j++) for (int jn = 0; jn < nst[cn]; jn++) {
                        s = first[c]+j; s2 = first[cn]+jn;
                        E[s*C + s2] += al[t*4+j] * A[s*C+s2] * be[(t+1)*4+jn] / sc[t+1]; } }
                { int c = tc[b+n-1]; for (int j = 0; j < nst[c]; j++) { s = first[c]+j; E[s*C + S] += al[(n-1)*4+j] * A[s*C+S] / sc[n]; } }
            }
            for (i = 0; i < R * C; i++) A[i] = E[i] + delta;
        }
        for (i = 0; i < R * C; i++) if (E[i] >= 0.5) edges += 1.0 / nf;
        /* ---- score held-out (or training if mode 1) ---- */
        for (d = 0; d < ndoc; d++) { if (mode == 0 && fold[d] != f) continue;
            int b = doc_off[d], n = doc_off[d+1] - b; if (n == 0) continue;
            for (t = b; t < b + n; t++) { int c = tc[t]; double den = Nc[c] + beta * M[c] + (isopen[c] ? Ac[c] + alpha0 : 0), p;
                if (ttype[t] >= 0) p = (ncw[ttype[t]] + beta) / den;
                else p = (Ac[c] + alpha0) / den * (nck[tkey[t]] + gamma) / (Ac[c] + gamma * nkc[c]) * exp(logp0[t]);
                emit_bits -= log2(p); }
            for (t = 0; t < n; t++) { int c = tc[b+t], m = nst[c]; double z = 0;
                for (int j = 0; j < m; j++) { s = first[c]+j; double v = 0;
                    if (t == 0) v = A[S*C+s]; else { int cp = tc[b+t-1]; for (int jp = 0; jp < nst[cp]; jp++) v += al[(t-1)*4+jp] * A[(first[cp]+jp)*C+s]; }
                    al[t*4+j] = v; z += v; }
                for (int j = 0; j < m; j++) al[t*4+j] /= z; trans_bits -= log2(z); }
            { int c = tc[b+n-1]; double z = 0; for (int j = 0; j < nst[c]; j++) z += al[(n-1)*4+j] * A[(first[c]+j)*C+S]; trans_bits -= log2(z); }
        }
        if (mode == 1) { memcpy(trans_out, A, sizeof(double) * R * C); if (emit_out) for (i = 0; i < R * C; i++) emit_out[i] = E[i]; }
        (void)ntrain;
    }
    out[0] = emit_bits; out[1] = trans_bits; out[2] = edges; out[3] = S;
    free(tc); free(M); free(isopen); free(nkc); free(ncw); free(Nc); free(Ac); free(nck); free(A); free(E); free(al); free(be); free(sc);
    return emit_bits + trans_bits;
}
