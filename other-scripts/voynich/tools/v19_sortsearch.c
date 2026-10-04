/* v19 'SOMEWHERE THE TEXT IS SORTED' - search engine.
 *
 * For each stretch (an ordered list of entries, each a glyph-id string), find the
 * glyph order (permutation of the alphabet) under which the entries are most
 * nearly sorted, measured by Kendall-type tau between position and sort key:
 *   tau = (concordant - discordant) / decided pairs   (ties ignored)
 * The key is lexicographic on the entry (optionally reversed, optionally truncated
 * to `depth` glyphs; depth 1 = first glyph only). The end of word is a symbol of
 * its own (id 0) and takes part in the order (so prefix-before-word is learned).
 * Every decided pair is decided by one glyph pair (a,b) at the first difference,
 * so tau is linear in the pairwise precedence matrix W and the search is a
 * linear-ordering problem: random orders (phase 1) then iterated local search with
 * insertion moves (phase 2).
 *
 * Null: entries shuffled among entries of the same class (class 0 = all) inside
 * the stretch (per-stretch mode) or across the whole file (pooled mode, classes
 * global, nullmode 2), or blocks (lines) permuted inside the stretch (nullmode 1); then the
 * SAME optimisation is re-run on each null replicate.
 *
 * Input (stdin):
 *   NSYM A
 *   STRETCH id n
 *   class block len g1 .. glen       (n lines; glyph ids 1..A-1)
 * Args: depth(0=full) reverse R nullmode pooled restarts ils nrand seed
 *   fixed-order mode: if env V19_FIXED is set to a comma list of ids (an order),
 *   no optimisation is done: tau is evaluated under that order (held-out test).
 * Output: one line per stretch (or one for the pooled file):
 *   id n decided tau_real nullmean nullsd nullmax p randmax order(comma ids)
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define MAXA 64
typedef struct { int cls, blk, len; int *g; } Entry;
typedef struct { char id[128]; int n; Entry *e; } Stretch;

static int A, depth, rev, R, nullmode, pooled, restarts, ils, nrand;
static unsigned long long rs = 88172645463325252ULL;
static inline unsigned long long xr(void){ rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return rs; }
static inline int rint_(int n){ return (int)(xr() % (unsigned long long)n); }

static Stretch *S; static int NS;
static int *fixed_order = NULL; static int nfixed = 0;

static inline int keyglyph(const Entry *e, int t){
  int L = e->len; if (depth > 0 && L > depth) L = depth;
  if (t >= L) return 0;
  return rev ? e->g[e->len - 1 - t] : e->g[t];
}

/* add pair (x before y) decision into W */
static void addpairs(Entry **seq, int n, double *W){
  for (int i = 0; i < n; i++){
    const Entry *x = seq[i];
    int Lx = x->len; if (depth > 0 && Lx > depth) Lx = depth;
    for (int j = i + 1; j < n; j++){
      const Entry *y = seq[j];
      int Ly = y->len; if (depth > 0 && Ly > depth) Ly = depth;
      int m = Lx > Ly ? Lx : Ly, t;
      for (t = 0; t < m; t++){
        int a = keyglyph(x, t), b = keyglyph(y, t);
        if (a != b){ W[a * A + b] += 1.0; break; }
      }
    }
  }
}

static double score(const int *ord, int k, const double *W){
  double s = 0;
  for (int i = 0; i < k; i++) for (int j = i + 1; j < k; j++) s += W[ord[i]*A + ord[j]];
  return s;
}

/* insertion local search to a local optimum; returns score */
static double localsearch(int *ord, int k, const double *W){
  int improved = 1;
  while (improved){
    improved = 0;
    for (int i = 0; i < k; i++){
      int x = ord[i]; double best = 1e-9; int bj = i; double g = 0;
      for (int j = i + 1; j < k; j++){ g += W[x*A + ord[j]] * -1 + W[ord[j]*A + x]; if (g > best){ best = g; bj = j; } }
      g = 0;
      for (int j = i - 1; j >= 0; j--){ g += W[x*A + ord[j]] - W[ord[j]*A + x]; if (g > best){ best = g; bj = j; } }
      if (bj != i){
        if (bj > i){ memmove(ord + i, ord + i + 1, (bj - i) * sizeof(int)); ord[bj] = x; }
        else { memmove(ord + bj + 1, ord + bj, (i - bj) * sizeof(int)); ord[bj] = x; }
        improved = 1;
      }
    }
  }
  return score(ord, k, W);
}

/* optimise order over the symbols present; returns best score, fills best order */
static double optimise(const double *W, int *present, int k, int *bestord, double *randmax){
  int ord[MAXA], cur[MAXA]; double best = -1;
  /* phase 1: random orders */
  double rm = -1;
  for (int r = 0; r < (randmax ? nrand : 0); r++){
    for (int i = 0; i < k; i++) ord[i] = present[i];
    for (int i = k - 1; i > 0; i--){ int j = rint_(i + 1); int t = ord[i]; ord[i] = ord[j]; ord[j] = t; }
    double s = score(ord, k, W); if (s > rm) rm = s;
  }
  if (randmax) *randmax = rm;
  /* phase 2: Borda start + random restarts, each with iterated local search */
  for (int r = 0; r < restarts; r++){
    for (int i = 0; i < k; i++) ord[i] = present[i];
    if (r == 0){ /* Borda: sort by net outflow */
      double net[MAXA];
      for (int i = 0; i < k; i++){ net[i] = 0; for (int j = 0; j < k; j++) net[i] += W[ord[i]*A+ord[j]] - W[ord[j]*A+ord[i]]; }
      for (int i = 0; i < k; i++) for (int j = i + 1; j < k; j++) if (net[j] > net[i]){ double t=net[i];net[i]=net[j];net[j]=t; int u=ord[i];ord[i]=ord[j];ord[j]=u; }
    } else for (int i = k - 1; i > 0; i--){ int j = rint_(i + 1); int t = ord[i]; ord[i] = ord[j]; ord[j] = t; }
    double s = localsearch(ord, k, W);
    for (int it = 0; it < ils && k > 2; it++){
      memcpy(cur, ord, k * sizeof(int));
      int nk = 2 + rint_(3);
      for (int q = 0; q < nk; q++){ int a = rint_(k), b = rint_(k); int t = cur[a]; cur[a] = cur[b]; cur[b] = t; }
      double s2 = localsearch(cur, k, W);
      if (s2 >= s){ s = s2; memcpy(ord, cur, k * sizeof(int)); }
    }
    if (s > best){ best = s; memcpy(bestord, ord, k * sizeof(int)); }
  }
  return best;
}

static double evaluate(Entry **seqs[], int *ns, int nst, int *ordout, int *kout, double *Tout, double *randmax){
  static double W[MAXA*MAXA];
  memset(W, 0, sizeof(W));
  for (int s = 0; s < nst; s++) addpairs(seqs[s], ns[s], W);
  double T = 0; int present[MAXA], k = 0, pres[MAXA] = {0};
  for (int a = 0; a < A; a++) for (int b = 0; b < A; b++) if (W[a*A+b] > 0){ T += W[a*A+b]; pres[a] = pres[b] = 1; }
  for (int a = 0; a < A; a++) if (pres[a]) present[k++] = a;
  *Tout = T; *kout = k;
  if (T == 0){ if (randmax) *randmax = 0; return 0; }
  double best;
  if (fixed_order){
    int ord[MAXA], m = 0;
    for (int i = 0; i < nfixed; i++) if (fixed_order[i] < A && pres[fixed_order[i]]) ord[m++] = fixed_order[i];
    /* symbols not in the fixed order: appended in Borda order (rare) */
    for (int a = 0; a < A; a++) if (pres[a]){ int f = 0; for (int i = 0; i < m; i++) if (ord[i]==a) f=1; if(!f) ord[m++]=a; }
    best = score(ord, m, W); memcpy(ordout, ord, m*sizeof(int)); *kout = m;
    if (randmax) *randmax = 0;
  } else best = optimise(W, present, k, ordout, randmax);
  if (randmax && *randmax >= 0) *randmax = (2 * *randmax - T) / T;
  return (2 * best - T) / T;
}

static void shuffle_seq(Entry **seq, int n){
  if (nullmode == 0){
    /* Fisher-Yates within each class (classes are small ints) */
    int *pos = malloc(n * sizeof(int)), *used = calloc(n, sizeof(int));
    for (int i = 0; i < n; i++){
      if (used[i]) continue;
      int c = seq[i]->cls, m = 0;
      for (int j = i; j < n; j++) if (!used[j] && seq[j]->cls == c){ pos[m++] = j; used[j] = 1; }
      for (int q = m - 1; q > 0; q--){ int r = rint_(q + 1); Entry *t = seq[pos[q]]; seq[pos[q]] = seq[pos[r]]; seq[pos[r]] = t; }
    }
    free(pos); free(used);
  } else {
    /* permute blocks keeping internal order */
    int nb = 0, starts[4096], lens[4096];
    for (int i = 0; i < n; i++){ if (i == 0 || seq[i]->blk != seq[i-1]->blk){ starts[nb] = i; lens[nb] = 0; nb++; } lens[nb-1]++; }
    int perm[4096]; for (int b = 0; b < nb; b++) perm[b] = b;
    for (int b = nb - 1; b > 0; b--){ int j = rint_(b + 1); int t = perm[b]; perm[b] = perm[j]; perm[j] = t; }
    Entry **tmp = malloc(n * sizeof(Entry*)); int p = 0;
    for (int b = 0; b < nb; b++) for (int q = 0; q < lens[perm[b]]; q++) tmp[p++] = seq[starts[perm[b]] + q];
    memcpy(seq, tmp, n * sizeof(Entry*)); free(tmp);
  }
}

static void print_order(FILE *f, int *ord, int k){ for (int i = 0; i < k; i++) fprintf(f, i ? ",%d" : "%d", ord[i]); }

int main(int argc, char **argv){
  if (argc < 10){ fprintf(stderr, "usage: depth rev R nullmode pooled restarts ils nrand seed\n"); return 1; }
  depth = atoi(argv[1]); rev = atoi(argv[2]); R = atoi(argv[3]); nullmode = atoi(argv[4]);
  pooled = atoi(argv[5]); restarts = atoi(argv[6]); ils = atoi(argv[7]); nrand = atoi(argv[8]);
  rs ^= (unsigned long long)atoll(argv[9]) * 0x9E3779B97F4A7C15ULL; for (int i=0;i<10;i++) xr();
  char *fx = getenv("V19_FIXED");
  if (fx && *fx){ fixed_order = malloc(MAXA*sizeof(int)); char *tok = strtok(fx, ","); while (tok && nfixed < MAXA){ fixed_order[nfixed++] = atoi(tok); tok = strtok(NULL, ","); } }
  char buf[256];
  if (scanf("%255s %d", buf, &A) != 2 || strcmp(buf, "NSYM")){ fprintf(stderr, "bad header\n"); return 1; }
  if (A > MAXA){ fprintf(stderr, "alphabet too large\n"); return 1; }
  int cap = 1024; S = malloc(cap * sizeof(Stretch)); NS = 0;
  while (scanf("%255s", buf) == 1){
    if (strcmp(buf, "STRETCH")) { fprintf(stderr, "parse error %s\n", buf); return 1; }
    if (NS == cap){ cap *= 2; S = realloc(S, cap * sizeof(Stretch)); }
    Stretch *st = &S[NS++];
    if (scanf("%127s %d", st->id, &st->n) != 2) return 1;
    st->e = malloc(st->n * sizeof(Entry));
    for (int i = 0; i < st->n; i++){
      Entry *e = &st->e[i];
      if (scanf("%d %d %d", &e->cls, &e->blk, &e->len) != 3) return 1;
      e->g = malloc((e->len > 0 ? e->len : 1) * sizeof(int));
      for (int t = 0; t < e->len; t++) if (scanf("%d", &e->g[t]) != 1) return 1;
    }
  }
  int ord[MAXA], k; double T, rmax;
  if (!pooled){
    for (int s = 0; s < NS; s++){
      Stretch *st = &S[s];
      Entry **seq = malloc(st->n * sizeof(Entry*));
      for (int i = 0; i < st->n; i++) seq[i] = &st->e[i];
      Entry **seqs[1] = { seq }; int ns[1] = { st->n };
      double tr = evaluate(seqs, ns, 1, ord, &k, &T, &rmax);
      int ordr[MAXA], kr = k; memcpy(ordr, ord, k * sizeof(int));
      double sum = 0, sum2 = 0, mx = -2; int ge = 0; int nok = 0;
      for (int r = 0; r < R; r++){
        shuffle_seq(seq, st->n);
        int o2[MAXA], k2; double T2, rm2;
        double tn = evaluate(seqs, ns, 1, o2, &k2, &T2, NULL);
        (void)rm2; sum += tn; sum2 += tn * tn; if (tn > mx) mx = tn; if (tn >= tr - 1e-12) ge++; nok++;
      }
      double m = nok ? sum / nok : 0, sd = nok ? sqrt(fmax(0, sum2 / nok - m * m)) : 0;
      printf("%s %d %.0f %.5f %.5f %.5f %.5f %.5f %.5f ", st->id, st->n, T, tr, m, sd, mx, (ge + 1.0) / (R + 1.0), rmax);
      print_order(stdout, ordr, kr); printf("\n"); fflush(stdout);
      free(seq);
    }
  } else {
    /* pooled: one order for all stretches. class-shuffle nulls are global across the file */
    Entry ***seqs = malloc(NS * sizeof(Entry**)); int *ns = malloc(NS * sizeof(int)); int N = 0;
    for (int s = 0; s < NS; s++){ seqs[s] = malloc(S[s].n * sizeof(Entry*)); ns[s] = S[s].n; for (int i = 0; i < S[s].n; i++) seqs[s][i] = &S[s].e[i]; N += S[s].n; }
    double tr = evaluate(seqs, ns, NS, ord, &k, &T, &rmax);
    int ordr[MAXA], kr = k; memcpy(ordr, ord, k * sizeof(int));
    double sum = 0, sum2 = 0, mx = -2; int ge = 0;
    Entry **all = malloc(N * sizeof(Entry*)); int *own = malloc(N * sizeof(int));
    for (int r = 0; r < R; r++){
      if (nullmode == 2){
        /* global shuffle within class across all stretches: bucket by class */
        int p = 0; for (int s = 0; s < NS; s++) for (int i = 0; i < ns[s]; i++){ all[p] = seqs[s][i]; own[p] = p; p++; }
        /* sort indices by class, shuffle entries within class buckets */
        /* simple: for each position, swap with random same-class position (Fisher-Yates per class) */
        int maxc = 0; for (int i = 0; i < N; i++) if (all[i]->cls > maxc) maxc = all[i]->cls;
        int *cnt = calloc(maxc + 2, sizeof(int)); for (int i = 0; i < N; i++) cnt[all[i]->cls + 1]++;
        for (int c = 1; c <= maxc + 1; c++) cnt[c] += cnt[c-1];
        int *idx = malloc(N * sizeof(int)); int *fill = calloc(maxc + 1, sizeof(int));
        for (int i = 0; i < N; i++){ int c = all[i]->cls; idx[cnt[c] + fill[c]++] = i; }
        Entry **perm = malloc(N * sizeof(Entry*)); for (int i = 0; i < N; i++) perm[i] = all[i];
        for (int c = 0; c <= maxc; c++){
          int lo = cnt[c], hi = cnt[c+1];
          for (int i = hi - 1; i > lo; i--){ int j = lo + rint_(i - lo + 1); int a = idx[i], b = idx[j]; Entry *t = perm[a]; perm[a] = perm[b]; perm[b] = t; }
        }
        p = 0; for (int s = 0; s < NS; s++) for (int i = 0; i < ns[s]; i++) seqs[s][i] = perm[p++];
        free(cnt); free(idx); free(fill); free(perm);
      } else for (int s = 0; s < NS; s++) shuffle_seq(seqs[s], ns[s]);
      int o2[MAXA], k2; double T2;
      double tn = evaluate(seqs, ns, NS, o2, &k2, &T2, NULL);
      sum += tn; sum2 += tn * tn; if (tn > mx) mx = tn; if (tn >= tr - 1e-12) ge++;
    }
    double m = R ? sum / R : 0, sd = R ? sqrt(fmax(0, sum2 / R - m * m)) : 0;
    printf("POOLED %d %.0f %.5f %.5f %.5f %.5f %.5f %.5f ", N, T, tr, m, sd, mx, (ge + 1.0) / (R + 1.0), rmax);
    print_order(stdout, ordr, kr); printf("\n");
  }
  return 0;
}
