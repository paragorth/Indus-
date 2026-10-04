/* R2 random-PROGRAM search engine.
 *
 * A program is a small postfix expression (stack machine) over per-position terminals
 * (previous symbols, symbol above in the previous line, same slot in previous words,
 * positions and counters), integer ops (add, sub, mul, mod, min, max, eq, lt, if),
 * learned lookup tables (TAB1, TAB2: a table-and-copy machine, entries = majority next
 * symbol on training documents A) and an L-system template (LSYS: a 4-rule L-system
 * string indexed by its argument).  Its output at each position is a predicted symbol
 * (or abstain).  Description length of held-out text = program bits + table bits +
 * residual bits, where the residual codes each symbol with
 *     p = lam * [sym == prediction] + (1 - lam) * p_baseline      (lam fitted on B)
 * i.e. the program is an expert added to a Kneser-Ney baseline; GAIN = baseline bits -
 * (program bits + residual bits).  Fitness = gain on split B; final test on split C.
 *
 * Phases: E_rand random programs, then evolution (mu+lambda, tournament, subtree
 * mutation/crossover, point mutation, restarts) for E_evo evaluations.  Checkpointed.
 *
 * usage: r2_search corpus.bin out_prefix seed E_rand E_evo [mode]
 *   mode 0 = full language, 1 = no tables (pure arithmetic/copy programs)
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

#define MAXI 15
#define HOF 30
enum { TERM, CONST, ADD, SUB, MUL, MOD, MIN, MAX, EQ, LT, IF, TAB1, TAB2, LSYS, NOPS };
static const char *OPN[] = {"T", "K", "add", "sub", "mul", "mod", "min", "max", "eq", "lt", "if", "tab1", "tab2", "lsys"};
static const int AR[] = {0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 3, 1, 2, 1};
static const char *TN[] = {"P1", "P2", "P3", "P4", "P6", "P8", "AB", "AW", "AW2", "FW", "FL",
                           "PW", "PL", "WI", "LI", "CW", "LPW", "LLW"};

typedef struct { uint8_t op, arg; } Ins;
typedef struct {
    int n; Ins c[MAXI];
    uint8_t rule[4][3]; uint8_t rlen[4];
    double fit, gainB, lam[4], bits, tabbits, gainC; int hitsB, predB;
} Prog;

static int N, NT, V, ND, MODE;
static int32_t *X, *SP, *DOC; static double *PB; static uint8_t **F;
static double *LGPB;      /* -log2 pb */
static int nA, nB, nC; static int *idxA;
static uint8_t *stk[MAXI + 2];
static int32_t *cnt; static int32_t *best, *tot; static uint8_t *bsym;
static uint64_t rs;
static double rnd(void) { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return (rs >> 11) * (1.0 / 9007199254740992.0); }
static int ri(int n) { return (int)(rnd() * n); }

/* ---------------------------------------------------------- program generation */
static int gen_tree(Ins *out, int maxlen, int depth, int root) {
    /* writes postfix into out, returns length or -1 */
    int leafp = depth <= 0 ? 1 : (root ? 0 : (rnd() < 0.35));
    if (leafp) {
        if (maxlen < 1) return -1;
        if (rnd() < 0.75) { out[0].op = TERM; out[0].arg = ri(NT); }
        else { out[0].op = CONST; out[0].arg = ri(32); }
        return 1;
    }
    int op;
    if (root && MODE == 0 && rnd() < 0.5) op = rnd() < 0.5 ? TAB1 : TAB2;
    else {
        do { op = ADD + ri(NOPS - ADD); } while (MODE == 1 && (op == TAB1 || op == TAB2));
    }
    int len = 0;
    for (int a = 0; a < AR[op]; a++) {
        int l = gen_tree(out + len, maxlen - len - 1, depth - 1, 0);
        if (l < 0) return -1;
        len += l;
    }
    if (len + 1 > maxlen) return -1;
    out[len].op = op; out[len].arg = 0;
    return len + 1;
}
static void rand_lsys(Prog *p) {
    for (int r = 0; r < 4; r++) { p->rlen[r] = 1 + ri(3); for (int j = 0; j < 3; j++) p->rule[r][j] = ri(4); }
}
static void rand_prog(Prog *p) {
    int l;
    do { l = gen_tree(p->c, MAXI, 1 + ri(4), 1); } while (l < 0);
    p->n = l; rand_lsys(p);
}
static int subtree_start(const Ins *c, int i) {
    int need = 1, j = i;
    while (j >= 0) { need += AR[c[j].op] - 1; if (need == 0) return j; j--; }
    return 0;
}
static int uses(const Prog *p, int op) { for (int i = 0; i < p->n; i++) if (p->c[i].op == op) return 1; return 0; }

static void mutate(Prog *p) {
    double u = rnd();
    if (u < 0.35) {           /* point mutation */
        int i = ri(p->n); Ins *x = &p->c[i];
        if (x->op == TERM) x->arg = ri(NT);
        else if (x->op == CONST) x->arg = rnd() < 0.5 ? ri(32) : (x->arg + (rnd() < 0.5 ? 1 : 31)) % 32;
        else {
            int a = AR[x->op], op;
            for (int t = 0; t < 20; t++) { op = ADD + ri(NOPS - ADD); if (AR[op] == a && !(MODE == 1 && (op == TAB1 || op == TAB2))) { x->op = op; break; } }
        }
    } else if (u < 0.45 && uses(p, LSYS)) {
        int r = ri(4); p->rlen[r] = 1 + ri(3); p->rule[r][ri(3)] = ri(4);
    } else {                  /* subtree replacement */
        int i = ri(p->n), s = subtree_start(p->c, i);
        Ins sub[MAXI]; int room = MAXI - (p->n - (i - s + 1)), l;
        for (int t = 0; t < 30; t++) { l = gen_tree(sub, room, ri(3), 0); if (l > 0) break; }
        if (l <= 0) return;
        Ins nc[MAXI]; int k = 0;
        for (int j = 0; j < s; j++) nc[k++] = p->c[j];
        for (int j = 0; j < l; j++) nc[k++] = sub[j];
        for (int j = i + 1; j < p->n; j++) nc[k++] = p->c[j];
        memcpy(p->c, nc, k * sizeof(Ins)); p->n = k;
    }
}
static void crossover(Prog *p, const Prog *q) {
    int i = ri(p->n), s = subtree_start(p->c, i);
    int j = ri(q->n), t = subtree_start(q->c, j);
    int l = j - t + 1, nn = p->n - (i - s + 1) + l;
    if (nn > MAXI) return;
    Ins nc[MAXI]; int k = 0;
    for (int a = 0; a < s; a++) nc[k++] = p->c[a];
    for (int a = t; a <= j; a++) nc[k++] = q->c[a];
    for (int a = i + 1; a < p->n; a++) nc[k++] = p->c[a];
    memcpy(p->c, nc, k * sizeof(Ins)); p->n = k;
    if (rnd() < 0.5) { memcpy(p->rule, q->rule, sizeof p->rule); memcpy(p->rlen, q->rlen, sizeof p->rlen); }
}

/* ---------------------------------------------------------- evaluation */
static uint8_t lstr[256];
static void build_lsys(const Prog *p) {
    uint8_t a[600], b[600]; int na = 1, nb; a[0] = 0;
    for (int it = 0; it < 12 && na < 256; it++) {
        nb = 0;
        for (int i = 0; i < na && nb < 590; i++) for (int j = 0; j < p->rlen[a[i]]; j++) b[nb++] = p->rule[a[i]][j];
        memcpy(a, b, nb); na = nb;
    }
    for (int i = 0; i < 256; i++) lstr[i] = a[i % na];
}

static uint8_t *TBIN, *BINS;
static double tab_train(uint8_t *ka, uint8_t *kb, uint8_t *out, int two) {
    /* train on A positions, write predictions for all positions; return table bits */
    int K = two ? 65536 : 256;
    for (int t = 0; t < nA; t++) {
        int i = idxA[t]; int k = two ? (ka[i] << 8 | kb[i]) : ka[i]; int y = X[i];
        int c = ++cnt[(size_t)k * V + y]; tot[k]++;
        if (c > best[k]) { best[k] = c; bsym[k] = y; }
    }
    int stored = 0, keys = 0;
    for (int i = 0; i < N; i++) {
        int k = two ? (ka[i] << 8 | kb[i]) : ka[i];
        out[i] = best[k] >= 2 ? bsym[k] : 255;
        if (best[k] >= 2) { double pu = (double)best[k] / tot[k]; TBIN[i] = pu < 0.25 ? 0 : pu < 0.5 ? 1 : pu < 0.75 ? 2 : 3; }
        else TBIN[i] = 0;
    }
    for (int t = 0; t < nA; t++) {
        int i = idxA[t]; int k = two ? (ka[i] << 8 | kb[i]) : ka[i];
        if (tot[k] > 0) { keys++; if (best[k] >= 2) stored++; tot[k] = -1; }
    }
    for (int t = 0; t < nA; t++) {
        int i = idxA[t]; int k = two ? (ka[i] << 8 | kb[i]) : ka[i];
        cnt[(size_t)k * V + X[i]] = 0; best[k] = 0; tot[k] = 0;
    }
    (void)K;
    double bits = stored * log2((double)V);
    if (keys > 0 && stored > 0 && stored < keys) {
        double f = (double)stored / keys;
        bits += keys * (-f * log2(f) - (1 - f) * log2(1 - f));
    }
    return bits + 2 * log2(keys + 2.0);
}

static uint8_t *PRED;
static double run_prog(Prog *p) {   /* fills PRED, returns table bits */
    int sp = 0; double tb = 0;
    if (uses(p, LSYS)) build_lsys(p);
    for (int k = 0; k < p->n; k++) {
        Ins x = p->c[k];
        int ar = AR[x.op];
        uint8_t *o = stk[sp - ar];   /* write result into lowest operand slot */
        uint8_t *a = ar >= 1 ? stk[sp - ar] : NULL, *b = ar >= 2 ? stk[sp - ar + 1] : NULL, *c = ar >= 3 ? stk[sp - ar + 2] : NULL;
        if (ar == 0) o = stk[sp];
        switch (x.op) {
        case TERM: memcpy(o, F[x.arg], N); break;
        case CONST: memset(o, x.arg, N); break;
        case ADD: for (int i = 0; i < N; i++) o[i] = a[i] + b[i]; break;
        case SUB: for (int i = 0; i < N; i++) o[i] = a[i] - b[i]; break;
        case MUL: for (int i = 0; i < N; i++) o[i] = a[i] * b[i]; break;
        case MOD: for (int i = 0; i < N; i++) o[i] = b[i] ? a[i] % b[i] : a[i]; break;
        case MIN: for (int i = 0; i < N; i++) o[i] = a[i] < b[i] ? a[i] : b[i]; break;
        case MAX: for (int i = 0; i < N; i++) o[i] = a[i] > b[i] ? a[i] : b[i]; break;
        case EQ: for (int i = 0; i < N; i++) o[i] = a[i] == b[i]; break;
        case LT: for (int i = 0; i < N; i++) o[i] = a[i] < b[i]; break;
        case IF: for (int i = 0; i < N; i++) o[i] = a[i] ? b[i] : c[i]; break;
        case TAB1: { tb += tab_train(a, NULL, PRED, 0); memcpy(o, PRED, N); break; }
        case TAB2: { tb += tab_train(a, b, PRED, 1); memcpy(o, PRED, N); break; }
        case LSYS: for (int i = 0; i < N; i++) o[i] = lstr[a[i]]; break;
        }
        sp = sp - ar + 1;
    }
    memcpy(PRED, stk[0], N);
    int rop = p->c[p->n - 1].op;
    if (rop == TAB1 || rop == TAB2) memcpy(BINS, TBIN, N); else memset(BINS, 0, N);
    return tb;
}

static double prog_bits(const Prog *p) {
    double b = log2((double)MAXI);
    for (int i = 0; i < p->n; i++) {
        b += log2((double)NOPS);
        if (p->c[i].op == TERM) b += log2((double)NT);
        if (p->c[i].op == CONST) b += 5;
    }
    if (uses(p, LSYS)) b += 4 * (log2(3.0) + 3 * 2);
    return b;
}

/* fit one lam per purity bin on split s (or use given), return gain bits on split s */
#define NBIN 4
static double gain_on(int s, double *lam, int fit, int *hits, int *npred, double *perdoc) {
    static double *hp[NBIN]; static int init = 0;
    if (!init) { for (int b = 0; b < NBIN; b++) hp[b] = malloc(sizeof(double) * N); init = 1; }
    int h[NBIN] = {0}, m[NBIN] = {0};
    for (int i = 0; i < N; i++) {
        if (SP[i] != s || PRED[i] >= V) continue;
        int b = BINS[i];
        if (PRED[i] == X[i]) hp[b][h[b]++] = PB[i]; else m[b]++;
    }
    *hits = 0; *npred = 0;
    double g = 0;
    for (int b = 0; b < NBIN; b++) {
        *hits += h[b]; *npred += h[b] + m[b];
        if (fit) {
            lam[b] = 0;
            if (h[b] > 0) {
                /* histogram of log2 pb over hits (256 bins) for a fast concave 1-D fit */
                double hc[256] = {0}, hv[256] = {0};
                for (int k = 0; k < h[b]; k++) { double lp = -log2(hp[b][k]); int j = (int)(lp * 8); if (j > 255) j = 255; hc[j] += 1; hv[j] += hp[b][k]; }
                double lo = 0, hi = 1 - 1e-9;
                for (int it = 0; it < 40; it++) {
                    double l = 0.5 * (lo + hi), d = -m[b] / (1 - l);
                    for (int j = 0; j < 256; j++) if (hc[j] > 0) { double q = hv[j] / hc[j]; d += hc[j] * (1 - q) / (l + (1 - l) * q); }
                    if (d > 0) lo = l; else hi = l;
                }
                lam[b] = 0.5 * (lo + hi);
            }
        }
        double l = lam[b];
        if (l <= 0 || perdoc) continue;
        double gb = m[b] * log2(1 - l);
        for (int k = 0; k < h[b]; k++) gb += log2((l + (1 - l) * hp[b][k]) / hp[b][k]);
        g += gb;
    }
    if (perdoc) {
        for (int i = 0; i < N; i++) {
            if (SP[i] != s || PRED[i] >= V) continue;
            double l = lam[BINS[i]]; if (l <= 0) continue;
            double gi = PRED[i] == X[i] ? log2((l + (1 - l) * PB[i]) / PB[i]) : log2(1 - l);
            perdoc[DOC[i]] += gi; g += gi;
        }
    }
    return g;
}

static long nevals = 0;
static void evaluate(Prog *p) {
    p->tabbits = run_prog(p);
    p->gainB = gain_on(1, p->lam, 1, &p->hitsB, &p->predB, NULL);
    int nl = 0; for (int b = 0; b < NBIN; b++) nl += p->lam[b] > 0;
    p->bits = prog_bits(p) + p->tabbits + 0.5 * log2((double)nB + 1) * (nl ? nl : 1);   /* + lams */
    p->fit = p->gainB - p->bits;
    nevals++;
}

static void show(FILE *f, const Prog *p) {
    for (int i = 0; i < p->n; i++) {
        if (p->c[i].op == TERM) fprintf(f, "%s ", TN[p->c[i].arg]);
        else if (p->c[i].op == CONST) fprintf(f, "%d ", p->c[i].arg);
        else fprintf(f, "%s ", OPN[p->c[i].op]);
    }
    if (uses(p, LSYS)) {
        fprintf(f, "[L:");
        for (int r = 0; r < 4; r++) { fprintf(f, " %d>", r); for (int j = 0; j < p->rlen[r]; j++) fprintf(f, "%d", p->rule[r][j]); }
        fprintf(f, "]");
    }
}
static int same_code(const Prog *a, const Prog *b) {
    if (a->n != b->n) return 0;
    for (int i = 0; i < a->n; i++) if (a->c[i].op != b->c[i].op || a->c[i].arg != b->c[i].arg) return 0;
    if (uses(a, LSYS) && (memcmp(a->rule, b->rule, sizeof a->rule) || memcmp(a->rlen, b->rlen, sizeof a->rlen))) return 0;
    return 1;
}

static Prog hof[HOF]; static int nhof = 0;
static void hof_add(const Prog *p) {
    for (int i = 0; i < nhof; i++) {
        if (same_code(&hof[i], p)) return;
        /* behavioural duplicate: same fitness to 1e-6 */
        if (fabs(hof[i].fit - p->fit) < 1e-6 && hof[i].hitsB == p->hitsB) return;
    }
    if (nhof < HOF) { hof[nhof++] = *p; }
    else { int w = 0; for (int i = 1; i < HOF; i++) if (hof[i].fit < hof[w].fit) w = i; if (p->fit > hof[w].fit) hof[w] = *p; else return; }
}

static int cmpfit(const void *a, const void *b) { double x = ((Prog *)a)->fit, y = ((Prog *)b)->fit; return (x < y) - (x > y); }

int main(int argc, char **argv) {
    if (argc < 6) { fprintf(stderr, "usage\n"); return 1; }
    const char *pre = argv[2]; unsigned seed = atoi(argv[3]); long Er = atol(argv[4]), Ee = atol(argv[5]);
    MODE = argc > 6 ? atoi(argv[6]) : 0;
    FILE *f = fopen(argv[1], "rb"); int hd[4];
    if (!f || fread(hd, 4, 4, f) != 4) { fprintf(stderr, "read\n"); return 1; }
    N = hd[0]; NT = hd[1]; V = hd[2]; ND = hd[3];
    X = malloc(4 * N); SP = malloc(4 * N); DOC = malloc(4 * N); PB = malloc(8 * N);
    fread(X, 4, N, f); fread(SP, 4, N, f); fread(DOC, 4, N, f); fread(PB, 8, N, f);
    F = malloc(sizeof(uint8_t *) * NT); int32_t *tmp = malloc(4 * N);
    for (int t = 0; t < NT; t++) { F[t] = malloc(N); fread(tmp, 4, N, f); for (int i = 0; i < N; i++) F[t][i] = (uint8_t)tmp[i]; }
    fclose(f);
    nA = nB = nC = 0; idxA = malloc(4 * N);
    for (int i = 0; i < N; i++) { if (SP[i] == 0) idxA[nA++] = i; else if (SP[i] == 1) nB++; else nC++; }
    for (int k = 0; k < MAXI + 2; k++) stk[k] = malloc(N);
    PRED = malloc(N); TBIN = calloc(N, 1); BINS = calloc(N, 1);
    cnt = calloc((size_t)65536 * V, 4); best = calloc(65536, 4); tot = calloc(65536, 4); bsym = calloc(65536, 1);
    rs = 0x9E3779B97F4A7C15ULL ^ (seed * 0x2545F4914F6CDD1DULL); for (int i = 0; i < 10; i++) rnd();

    if (MODE == 9) {   /* evaluate one program given as text (argv[7]) */
        Prog p; memset(&p, 0, sizeof p); rand_lsys(&p);
        char buf[4096]; strncpy(buf, argv[7], 4095); char *tok = strtok(buf, " ");
        while (tok) {
            int done = 0;
            for (int t = 0; t < NT && !done; t++) if (!strcmp(tok, TN[t])) { p.c[p.n].op = TERM; p.c[p.n++].arg = t; done = 1; }
            for (int o = ADD; o < NOPS && !done; o++) if (!strcmp(tok, OPN[o])) { p.c[p.n].op = o; p.c[p.n++].arg = 0; done = 1; }
            if (!done) { p.c[p.n].op = CONST; p.c[p.n++].arg = atoi(tok); }
            tok = strtok(NULL, " ");
        }
        MODE = 0; evaluate(&p);
        if (argc > 8) { FILE *pf = fopen(argv[8], "wb"); fwrite(PRED, 1, N, pf); fwrite(BINS, 1, N, pf); fclose(pf); }
        int hc, pc; double *pd = calloc(ND, sizeof(double)); double gC = gain_on(2, p.lam, 0, &hc, &pc, pd);
        printf("fitB %.2f gainB %.2f bits %.2f hitsB %d predB %d gainC %.2f hitsC %d lam %.4f %.4f %.4f %.4f\n", p.fit, p.gainB, p.bits, p.hitsB, p.predB, gC, hc, p.lam[0], p.lam[1], p.lam[2], p.lam[3]);
        return 0;
    }
    char path[1024]; sprintf(path, "%s.ckpt", pre);
    enum { P = 300 };
    static Prog pop[2 * P];
    long done_r = 0, done_e = 0; int gen = 0;
    FILE *ck = fopen(path, "rb");
    if (ck) {
        fread(&done_r, 8, 1, ck); fread(&done_e, 8, 1, ck); fread(&gen, 4, 1, ck); fread(&rs, 8, 1, ck);
        fread(&nhof, 4, 1, ck); fread(hof, sizeof(Prog), HOF, ck); fread(pop, sizeof(Prog), P, ck); fclose(ck);
        fprintf(stderr, "resume r=%ld e=%ld\n", done_r, done_e);
    }
    /* baseline sanity */
    double bB = 0; for (int i = 0; i < N; i++) if (SP[i] == 1) bB -= log2(PB[i]);
    fprintf(stderr, "N=%d V=%d A=%d B=%d C=%d baseline B bits %.0f (%.4f/tok)\n", N, V, nA, nB, nC, bB, bB / nB);

    /* random phase: keep best P in pop */
    int npop = done_r ? P : 0;
    if (!done_r) for (int i = 0; i < P; i++) pop[i].fit = -1e18;
    while (done_r < Er) {
        Prog p; rand_prog(&p); evaluate(&p); hof_add(&p); done_r++;
        int w = 0; for (int i = 1; i < P; i++) if (pop[i].fit < pop[w].fit) w = i;
        if (p.fit > pop[w].fit) pop[w] = p;
        if (done_r % 20000 == 0 || done_r == Er) {
            qsort(hof, nhof, sizeof(Prog), cmpfit);
            fprintf(stderr, "rand %ld best fit %.1f gainB %.1f bits %.1f : ", done_r, hof[0].fit, hof[0].gainB, hof[0].bits); show(stderr, &hof[0]); fprintf(stderr, "\n");
            ck = fopen(path, "wb");
            fwrite(&done_r, 8, 1, ck); fwrite(&done_e, 8, 1, ck); fwrite(&gen, 4, 1, ck); fwrite(&rs, 8, 1, ck);
            fwrite(&nhof, 4, 1, ck); fwrite(hof, sizeof(Prog), HOF, ck); fwrite(pop, sizeof(Prog), P, ck); fclose(ck);
        }
    }
    npop = P;
    /* evolution */
    double lastbest = -1e18; int stale = 0;
    while (done_e < Ee) {
        for (int c = 0; c < P; c++) {
            Prog *ch = &pop[P + c];
            int a = ri(npop); for (int t = 0; t < 3; t++) { int b = ri(npop); if (pop[b].fit > pop[a].fit) a = b; }
            *ch = pop[a];
            double u = rnd();
            if (u < 0.3) { int b = ri(npop); for (int t = 0; t < 3; t++) { int d = ri(npop); if (pop[d].fit > pop[b].fit) b = d; } crossover(ch, &pop[b]); }
            int nm = 1 + (rnd() < 0.3); for (int t = 0; t < nm; t++) mutate(ch);
            if (rnd() < 0.03) rand_prog(ch);
            evaluate(ch); hof_add(ch); done_e++;
        }
        qsort(pop, 2 * P, sizeof(Prog), cmpfit);
        /* dedupe by code: push duplicates to the end */
        for (int i = 1; i < P; i++) for (int j = 0; j < i; j++) if (same_code(&pop[i], &pop[j])) { rand_prog(&pop[i]); evaluate(&pop[i]); break; }
        gen++;
        if (pop[0].fit > lastbest + 0.5) { lastbest = pop[0].fit; stale = 0; } else stale++;
        if (stale >= 40) {   /* restart: keep top 10, reseed the rest */
            for (int i = 10; i < P; i++) { rand_prog(&pop[i]); evaluate(&pop[i]); hof_add(&pop[i]); }
            stale = 0; lastbest = -1e18;
            fprintf(stderr, "restart at gen %d\n", gen);
        }
        if (gen % 20 == 0 || done_e >= Ee) {
            qsort(hof, nhof, sizeof(Prog), cmpfit);
            fprintf(stderr, "gen %d evo %ld best fit %.1f gainB %.1f bits %.1f : ", gen, done_e, hof[0].fit, hof[0].gainB, hof[0].bits); show(stderr, &hof[0]); fprintf(stderr, "\n");
            ck = fopen(path, "wb");
            fwrite(&done_r, 8, 1, ck); fwrite(&done_e, 8, 1, ck); fwrite(&gen, 4, 1, ck); fwrite(&rs, 8, 1, ck);
            fwrite(&nhof, 4, 1, ck); fwrite(hof, sizeof(Prog), HOF, ck); fwrite(pop, sizeof(Prog), P, ck); fclose(ck);
        }
    }
    /* final: hall of fame tested on held-out C */
    qsort(hof, nhof, sizeof(Prog), cmpfit);
    sprintf(path, "%s.out", pre); FILE *o = fopen(path, "w");
    fprintf(o, "{\"evals\": %ld, \"N\": %d, \"V\": %d, \"nB\": %d, \"nC\": %d, \"hof\": [\n", done_r + done_e, N, V, nB, nC);
    double *pd = calloc(ND, sizeof(double));
    for (int h = 0; h < nhof; h++) {
        Prog *p = &hof[h];
        run_prog(p);
        memset(pd, 0, ND * sizeof(double));
        int hc, pc;
        double gC = gain_on(2, p->lam, 0, &hc, &pc, pd);
        fprintf(o, "{\"prog\": \"");
        show(o, p);
        fprintf(o, "\", \"fitB\": %.3f, \"gainB\": %.3f, \"bits\": %.3f, \"tabbits\": %.3f, \"lam\": [%.4f, %.4f, %.4f, %.4f], \"hitsB\": %d, \"predB\": %d, \"gainC\": %.3f, \"hitsC\": %d, \"predC\": %d, \"perdoc\": [",
                p->fit, p->gainB, p->bits, p->tabbits, p->lam[0], p->lam[1], p->lam[2], p->lam[3], p->hitsB, p->predB, gC, hc, pc);
        int first = 1;
        for (int d = 0; d < ND; d++) if (pd[d] != 0) { fprintf(o, "%s[%d, %.4f]", first ? "" : ", ", d, pd[d]); first = 0; }
        fprintf(o, "]}%s\n", h + 1 < nhof ? "," : "");
    }
    fprintf(o, "]}\n"); fclose(o);
    fprintf(stderr, "done\n");
    return 0;
}
