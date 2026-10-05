/* v55 cycle 3: HOROSCOPE search. For every day 1290-1611, every alignment (rotation x direction)
   and every binary word class, count how many of the 7 bodies (Sun, Moon, Mercury..Saturn) stand
   on a degree whose nymph label is in the class. Score = -log10 hypergeometric tail.
   Input (binary, little-endian): header int32 ndates, nbody, nal, ncls, L;
     int16 deg[ndates*nbody]; int16 yearidx[ndates]; int16 map[nal*360] (label or -1);
     uint8 cls[ncls*L].
   Output: float32 best[ncls*nyears], int32 bestdate[ncls*nyears], int16 bestal[ncls*nyears]. */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
static double lchoose(int n, int k){ if(k<0||k>n) return -INFINITY; return lgamma(n+1.0)-lgamma(k+1.0)-lgamma(n-k+1.0); }
int main(int argc, char **argv){
  FILE *f = fopen(argv[1], "rb"); int nyears = atoi(argv[3]);
  int hd[5]; fread(hd, 4, 5, f); int nd=hd[0], nb=hd[1], nal=hd[2], nc=hd[3], L=hd[4];
  short *deg = malloc(2L*nd*nb); fread(deg, 2, (long)nd*nb, f);
  short *yi = malloc(2L*nd); fread(yi, 2, nd, f);
  short *map = malloc(2L*nal*360); fread(map, 2, nal*360, f);
  unsigned char *cls = malloc((long)nc*L); fread(cls, 1, (long)nc*L, f); fclose(f);
  /* tail table T[m][B][h] = -log10 P(H>=h), population L, m successes, B draws (B<=nb) */
  double *T = malloc(sizeof(double)*(L+1)*(nb+1)*(nb+1));
  for(int m=0;m<=L;m++) for(int B=0;B<=nb;B++){
    double p[16]; for(int h=0;h<=B;h++) p[h]=exp(lchoose(m,h)+lchoose(L-m,B-h)-lchoose(L,B));
    for(int h=0;h<=nb;h++){ double s=0; for(int k=h;k<=B;k++) s+= (k<=B? p[k]:0); T[((long)m*(nb+1)+B)*(nb+1)+h] = (h<=B && s>0)? -log10(s):0; }
  }
  int *msum = calloc(nc, sizeof(int));
  for(int c=0;c<nc;c++) for(int l=0;l<L;l++) msum[c]+=cls[(long)c*L+l];
  float *best = calloc((long)nc*nyears, 4); int *bd = calloc((long)nc*nyears, 4); short *ba = calloc((long)nc*nyears, 2);
  int lab[16];
  for(int d=0; d<nd; d++){
    int y = yi[d];
    for(int a=0;a<nal;a++){
      int B=0;
      for(int b=0;b<nb;b++){ int l = map[a*360+deg[(long)d*nb+b]]; if(l<0) continue;
        int dup=0; for(int q=0;q<B;q++) if(lab[q]==l) dup=1; if(!dup) lab[B++]=l; }
      if(B==0) continue;
      for(int c=0;c<nc;c++){
        const unsigned char *cc = cls+(long)c*L; int h=0;
        for(int q=0;q<B;q++) h+=cc[lab[q]];
        float s = (float)T[((long)msum[c]*(nb+1)+B)*(nb+1)+h];
        long o=(long)c*nyears+y; if(s>best[o]){best[o]=s; bd[o]=d; ba[o]=a;}
      }
    }
  }
  FILE *g = fopen(argv[2], "wb"); fwrite(best,4,(long)nc*nyears,g); fwrite(bd,4,(long)nc*nyears,g); fwrite(ba,2,(long)nc*nyears,g); fclose(g);
  return 0;
}
