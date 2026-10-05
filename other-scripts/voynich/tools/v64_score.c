/* v64 concept-alphabet scorer (held-out word prediction).
 * Corpus file (argv[1]): lines "ctrain ctest word" (one byte per glyph unit).
 * Alphabets on stdin, one per line, concepts separated by spaces.
 * Output per alphabet:  cov canon bfree bm1 bmk2 Gfree Gm1 meanm
 * (bm1: same gaps, concept sequence by concept Markov-1 instead of the free-combination term)
 *
 * Parse: greedy longest-match left to right; unmatched glyphs are padding.
 * CONCEPT model (a Lullian wheel/table model):
 *   P(word) = P(m) * multinomial(m; w) * order term * prod gap-string probs
 *   order term: pi if concepts appear in the canonical order (Copeland order of
 *   train precedence counts), else (1-pi)/(Nperm-1).
 *   gaps: initial / medial / final padding strings, each a smoothed categorical
 *   over train gap strings with an escape that spells the string glyph by glyph.
 * BASELINE: glyph Markov-2 with end symbol (alphabet-free).
 * Both mixed 50:50 with the train word-unigram (so fixed free words cost the same).
 * G = bits_mk2 - bits_concept per test token (positive = words are better told as
 * concept combinations than as glyph strings).
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define MAXT 60000
#define MAXK 40
#define MAXM 24
#define HS 65536
typedef struct { double tr, te; char w[64]; int len; } Type;
static Type T[MAXT]; static int NT = 0;
static int seqs[MAXT][MAXM]; static int ms[MAXT];
static int gst[MAXT][MAXM+1], gen_[MAXT][MAXM+1]; /* gap start,end offsets */
static double TRN=0, TEN=0;
static double pmk[MAXT]; /* markov-2 prob per type */

/* gap string hash tables: 3 kinds */
typedef struct { char s[64]; double c; int used; } HE;
static HE H[3][HS]; static double HN[3]; static double HTYPES[3];
static unsigned hsh(const char *s, int n){ unsigned h=2166136261u; for(int i=0;i<n;i++){h^=(unsigned char)s[i]; h*=16777619u;} return h; }
static HE* hget(int k, const char *s, int n, int create){
  unsigned h = hsh(s,n) & (HS-1);
  while(H[k][h].used){ if((int)strlen(H[k][h].s)==n && memcmp(H[k][h].s,s,n)==0) return &H[k][h]; h=(h+1)&(HS-1); }
  if(!create) return NULL;
  H[k][h].used=1; memcpy(H[k][h].s,s,n); H[k][h].s[n]=0; H[k][h].c=0; HTYPES[k]+=1; return &H[k][h];
}
static double gunif[256]; static double NG=0;

static double gapprob(int k, const char *s, int n){
  HE *e = hget(k,s,n,0);
  double alpha = HTYPES[k] + 1.0;
  double denom = HN[k] + alpha;
  if(e) return e->c/denom;
  /* escape: spell */
  double p = alpha/denom * 0.5;
  for(int i=0;i<n;i++) p *= 0.5*gunif[(unsigned char)s[i]];
  return p;
}

int main(int argc, char **argv){
  FILE *f = fopen(argv[1], "r"); if(!f){perror("corpus"); return 1;}
  char buf[512];
  while(fgets(buf, sizeof buf, f)){
    double a,b; char w[256];
    if(sscanf(buf, "%lf %lf %255s", &a, &b, w) != 3) continue;
    if(strlen(w) > 60) continue;
    T[NT].tr = a; T[NT].te = b; strcpy(T[NT].w, w); T[NT].len = strlen(w); TRN+=a; TEN+=b; NT++;
    if(NT >= MAXT) break;
  }
  fclose(f);
  /* glyph unigram + markov-2 */
  static double tri[257][257][257]; /* index 256 = boundary */
  static double bic[257][257];
  for(int i=0;i<256;i++) gunif[i]=0;
  for(int t=0;t<NT;t++){ double c=T[t].tr; if(c<=0) continue; int a=256,b=256; const unsigned char *w=(const unsigned char*)T[t].w;
    for(int i=0;i<=T[t].len;i++){ int x = i<T[t].len? w[i]:256; tri[a][b][x]+=c; bic[a][b]+=c; if(x<256){gunif[x]+=c; NG+=c;} a=b; b=x; } }
  int nsym=0; for(int i=0;i<256;i++) if(gunif[i]>0) nsym++;
  for(int i=0;i<256;i++) gunif[i] = (gunif[i]+0.5)/(NG+0.5*256);
  for(int t=0;t<NT;t++){ int a=256,b=256; const unsigned char *w=(const unsigned char*)T[t].w; double lp=0;
    for(int i=0;i<=T[t].len;i++){ int x = i<T[t].len? w[i]:256; lp += log2((tri[a][b][x]+0.1)/(bic[a][b]+0.1*(nsym+1))); a=b; b=x; }
    pmk[t]=pow(2.0, lp); }
  static char line[8192];
  while(fgets(line, sizeof line, stdin)){
    char *con[MAXK]; int clen[MAXK]; int K = 0;
    char *tok = strtok(line, " \t\n");
    while(tok && K < MAXK){ con[K] = tok; clen[K] = strlen(tok); K++; tok = strtok(NULL, " \t\n"); }
    if(K == 0){ printf("NA\n"); fflush(stdout); continue; }
    int ord[MAXK]; for(int i=0;i<K;i++) ord[i]=i;
    for(int i=0;i<K;i++) for(int j=i+1;j<K;j++) if(clen[ord[j]] > clen[ord[i]]){int t=ord[i];ord[i]=ord[j];ord[j]=t;}
    for(int k=0;k<3;k++){ memset(H[k],0,sizeof(H[k])); HN[k]=0; HTYPES[k]=0; }
    double covg=0, totg=0, cnt[MAXK]; memset(cnt,0,sizeof cnt);
    static double prec[MAXK][MAXK]; memset(prec,0,sizeof prec);
    double mcount[MAXM+1]; memset(mcount,0,sizeof mcount);
    static double trn[MAXK+1][MAXK+1]; memset(trn,0,sizeof trn);
    for(int t=0;t<NT;t++){
      int p=0, m=0, pad=0; const char *w=T[t].w; int L=T[t].len; int gs=0;
      gst[t][0]=0;
      while(p < L){
        int hit=-1;
        for(int ii=0;ii<K;ii++){ int c=ord[ii]; if(clen[c] <= L-p && memcmp(w+p, con[c], clen[c])==0){hit=c;break;} }
        if(hit>=0 && m<MAXM){ gen_[t][m]=p; seqs[t][m++]=hit; p+=clen[hit]; gst[t][m]=p; }
        else { pad++; p++; }
      }
      gen_[t][m]=L; ms[t]=m; (void)gs;
      double c = T[t].tr;
      if(c>0){
        covg += c*(L-pad); totg += c*L; mcount[m] += c;
        { int pv=K; for(int i=0;i<m;i++){ trn[pv][seqs[t][i]]+=c; pv=seqs[t][i]; } trn[pv][K]+=c; }
        for(int i=0;i<m;i++){ cnt[seqs[t][i]] += c;
          for(int j=i+1;j<m;j++) if(seqs[t][i]!=seqs[t][j]) prec[seqs[t][i]][seqs[t][j]] += c; }
        for(int g=0; g<=m; g++){ int kind = (g==0)?0:((g==m)?2:1); if(m==0) kind=0;
          HE *e = hget(kind, w+gst[t][g], gen_[t][g]-gst[t][g], 1); e->c += c; HN[kind]+=c; }
      }
    }
    double score[MAXK]; int rank[MAXK];
    for(int a=0;a<K;a++){ score[a]=0; for(int b=0;b<K;b++) if(a!=b){ double s=prec[a][b]+prec[b][a]; score[a]+= s>0? prec[a][b]/s : 0.5; } }
    int co[MAXK]; for(int i=0;i<K;i++) co[i]=i;
    for(int i=0;i<K;i++) for(int j=i+1;j<K;j++) if(score[co[j]]>score[co[i]]){int x=co[i];co[i]=co[j];co[j]=x;}
    for(int i=0;i<K;i++) rank[co[i]]=i;
    double W=0, wv[MAXK]; for(int a=0;a<K;a++){ wv[a]=cnt[a]+0.5; W+=wv[a]; } for(int a=0;a<K;a++) wv[a]/=W;
    double M=0; for(int m=0;m<=MAXM;m++){ mcount[m]+=0.1; M+=mcount[m]; }
    double canon_tr=0, tot2_tr=0;
    for(int t=0;t<NT;t++){ if(T[t].tr<=0 || ms[t]<2) continue; int ok=1; for(int i=1;i<ms[t];i++) if(rank[seqs[t][i]]<rank[seqs[t][i-1]]) {ok=0;break;}
      tot2_tr+=T[t].tr; if(ok) canon_tr+=T[t].tr; }
    double pi = (canon_tr+0.5)/(tot2_tr+1.0);
    double rs[MAXK+1]; for(int a=0;a<=K;a++){ rs[a]=0; for(int b=0;b<=K;b++) rs[a]+=trn[a][b]+0.5; }
    double bc=0, bm=0, b1=0, pc=0, pt=0, msum=0;
    for(int t=0;t<NT;t++){
      double c=T[t].te; if(c<=0) continue; int m=ms[t]; msum+=c*m;
      const char *w=T[t].w;
      double ll = log(mcount[m>MAXM?MAXM:m]/M);
      int nc[MAXK]; memset(nc,0,sizeof nc);
      for(int i=0;i<m;i++){ nc[seqs[t][i]]++; ll += log(wv[seqs[t][i]]); }
      double lperm = lgamma(m+1.0); for(int a=0;a<K;a++) lperm -= lgamma(nc[a]+1.0);
      ll += lperm;
      double nperm = exp(lperm);
      int ok=1; for(int i=1;i<m;i++) if(rank[seqs[t][i]]<rank[seqs[t][i-1]]) {ok=0;break;}
      if(nperm > 1.5){ if(ok) ll += log(pi); else ll += log((1-pi)/(nperm-1)); }
      double l1=0; { int pv=K; for(int i=0;i<m;i++){ l1+=log((trn[pv][seqs[t][i]]+0.5)/rs[pv]); pv=seqs[t][i]; } l1+=log((trn[pv][K]+0.5)/rs[pv]); }
      double lg=0;
      for(int g=0; g<=m; g++){ int kind = (g==0)?0:((g==m)?2:1); if(m==0) kind=0;
        lg += log(gapprob(kind, w+gst[t][g], gen_[t][g]-gst[t][g])); }
      ll += lg; l1 += lg;
      double puni = T[t].tr / (TRN + 1.0);
      double pcon = exp(ll);
      b1 += -c*log2(0.5*puni + 0.5*exp(l1));
      bc += -c*log2(0.5*puni + 0.5*pcon);
      bm += -c*log2(0.5*puni + 0.5*pmk[t]);
      for(int i=0;i<m;i++) for(int j=i+1;j<m;j++){ int a=seqs[t][i], b=seqs[t][j]; if(a==b) continue; pt+=c; if(rank[a]<rank[b]) pc+=c; }
    }
    double cov = totg>0? covg/totg : 0;
    double canon = pt>0? pc/pt : 0;
    printf("%.4f %.4f %.4f %.4f %.4f %.4f %.4f %.3f\n", cov, canon, bc/TEN, b1/TEN, bm/TEN, (bm-bc)/TEN, (bm-b1)/TEN, msum/TEN);
    fflush(stdout);
  }
  return 0;
}
