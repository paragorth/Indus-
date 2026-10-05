// v66: injective graph matching (concepts -> keywords) by simulated annealing.
// score(m) = sum_{i<j} C[i][j] * B[m_i][m_j] / npairs ; C is n x n, B is K x K (K >= n), both standardised.
// Many random restarts; returns best map and every restart's final score.
// build: gcc -O3 -march=native -shared -fPIC -o v66_qap.so v66_qap.c -lm
#include <stdlib.h>
#include <math.h>
#include <string.h>

static unsigned long long rs;
static inline unsigned long long xr(void){ rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return rs; }
static inline double ur(void){ return (xr() >> 11) * (1.0/9007199254740992.0); }

static double full(int n, int K, const double*C, const double*B, const int*m){
  double s=0; for(int i=0;i<n;i++) for(int j=i+1;j<n;j++) s += C[i*n+j]*B[m[i]*K+m[j]];
  return s;
}

// init: if init_map != NULL and restart 0 uses it as start.
double qap_sa(int n, int K, const double*C, const double*B, long iters, int restarts,
              unsigned long long seed, double T0, double T1, int*best_map, double*rest_scores,
              const int*init_map, int*rest_maps){
  rs = seed*2654435761ULL + 88172645463325252ULL; if(!rs) rs=1;
  int *m = malloc(sizeof(int)*n), *used = malloc(sizeof(int)*K), *pos = malloc(sizeof(int)*K);
  double best=-1e300; double np = n*(n-1)/2.0;
  for(int r=0;r<restarts;r++){
    // random injective start
    for(int k=0;k<K;k++) pos[k]=k;
    for(int k=K-1;k>0;k--){ int j=xr()%(k+1); int t=pos[k]; pos[k]=pos[j]; pos[j]=t; }
    if(r==0 && init_map){ memset(used,0,sizeof(int)*K); for(int i=0;i<n;i++){ m[i]=init_map[i]; used[m[i]]=1; } }
    else { memset(used,0,sizeof(int)*K); for(int i=0;i<n;i++){ m[i]=pos[i]; used[m[i]]=1; } }
    double s = full(n,K,C,B,m), bs=s; int *bm = malloc(sizeof(int)*n); memcpy(bm,m,sizeof(int)*n);
    for(long it=0; it<iters; it++){
      double T = T0*pow(T1/T0, (double)it/iters);
      int i = xr()%n; double d=0;
      if(K>n && (xr()&1)){ // reassign i to unused keyword
        int k = xr()%K; if(used[k]) continue;
        int a=m[i];
        for(int j=0;j<n;j++) if(j!=i) d += C[i*n+j]*(B[k*K+m[j]]-B[a*K+m[j]]);
        if(d>=0 || ur()<exp(d/T)){ used[a]=0; used[k]=1; m[i]=k; s+=d; }
      } else { // swap i,j
        int j = xr()%n; if(j==i) continue;
        int a=m[i], b=m[j];
        for(int l=0;l<n;l++){ if(l==i||l==j) continue;
          d += (C[i*n+l]-C[j*n+l])*(B[b*K+m[l]]-B[a*K+m[l]]); }
        if(d>=0 || ur()<exp(d/T)){ m[i]=b; m[j]=a; s+=d; }
      }
      if(s>bs){ bs=s; memcpy(bm,m,sizeof(int)*n); }
    }
    bs = full(n,K,C,B,bm);
    rest_scores[r]=bs/np;
    if(rest_maps) memcpy(rest_maps + (long)r*n, bm, sizeof(int)*n);
    if(bs>best){ best=bs; memcpy(best_map,bm,sizeof(int)*n); }
    free(bm);
  }
  free(m); free(used); free(pos);
  return best/np;
}
