/* v54: near-repeated word trigrams at scale.
   stdin:  V, then V type strings (one per line, printable bytes), then N, then N lines "line page type".
   stdout: one line per pair of trigrams "i j d0 d1 d2" (token start indices), where trigram i and j lie
           inside one line each, do not overlap, and each aligned word pair is within the allowed edit distance
           (<=1, or <=2 when both words have >=6 units); total distance <= 3.  Exact repeats (all 0) included.
   Build: gcc -O2 -o v54_pairs v54_pairs.c */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static int lev(const char*a,int la,const char*b,int lb,int cap){
  int d[64][64]; if(la>62||lb>62) return cap+1;
  if(abs(la-lb)>cap) return cap+1;
  for(int i=0;i<=la;i++) d[i][0]=i; for(int j=0;j<=lb;j++) d[0][j]=j;
  for(int i=1;i<=la;i++){int mn=1000; for(int j=1;j<=lb;j++){int c=d[i-1][j-1]+(a[i-1]!=b[j-1]);
    if(d[i-1][j]+1<c)c=d[i-1][j]+1; if(d[i][j-1]+1<c)c=d[i][j-1]+1; d[i][j]=c; if(c<mn)mn=c;} if(mn>cap) return cap+1;}
  return d[la][lb];
}
int main(){
  int V; if(scanf("%d",&V)!=1) return 1;
  char **T=malloc(V*sizeof(char*)); int *L=malloc(V*sizeof(int)); char buf[256];
  for(int v=0;v<V;v++){ if(scanf("%255s",buf)!=1) return 1; T[v]=strdup(buf); L[v]=strlen(buf); }
  int N; if(scanf("%d",&N)!=1) return 1;
  int *ln=malloc(N*sizeof(int)),*pg=malloc(N*sizeof(int)),*ty=malloc(N*sizeof(int));
  for(int i=0;i<N;i++) if(scanf("%d %d %d",&ln[i],&pg[i],&ty[i])!=3) return 1;
  /* neighbour lists: types within allowed distance (incl. self) */
  int **nb=malloc(V*sizeof(int*)); unsigned char **nd=malloc(V*sizeof(unsigned char*)); int *nn=calloc(V,sizeof(int)); int *cap=calloc(V,sizeof(int));
  for(int v=0;v<V;v++){cap[v]=8;nb[v]=malloc(8*sizeof(int));nd[v]=malloc(8);}
  for(int a=0;a<V;a++) for(int b=a;b<V;b++){
    int allow = (L[a]>=6 && L[b]>=6)?2:1; int d = (a==b)?0:lev(T[a],L[a],T[b],L[b],allow);
    if(d<=allow){ int pr[2][2]={{a,b},{b,a}}; for(int k=0;k<(a==b?1:2);k++){int x=pr[k][0],y=pr[k][1];
        if(nn[x]==cap[x]){cap[x]*=2;nb[x]=realloc(nb[x],cap[x]*sizeof(int));nd[x]=realloc(nd[x],cap[x]);}
        nb[x][nn[x]]=y; nd[x][nn[x]]=d; nn[x]++;}}
  }
  /* trigram starts by first type */
  int *tri=malloc(N*sizeof(int)); int nt=0;
  for(int i=0;i+2<N;i++) if(ln[i]==ln[i+1]&&ln[i]==ln[i+2]) tri[nt++]=i;
  int *cnt=calloc(V+1,sizeof(int)); for(int k=0;k<nt;k++) cnt[ty[tri[k]]+1]++;
  for(int v=0;v<V;v++) cnt[v+1]+=cnt[v];
  int *idx=malloc(nt*sizeof(int)); int *fill=calloc(V,sizeof(int));
  for(int k=0;k<nt;k++){int t=ty[tri[k]]; idx[cnt[t]+fill[t]++]=tri[k];}
  /* lookup d between two types via neighbour list scan (lists are short) */
  long np=0;
  for(int k=0;k<nt;k++){ int i=tri[k]; int t0=ty[i];
    for(int a=0;a<nn[t0];a++){ int u=nb[t0][a]; int d0=nd[t0][a];
      for(int m=cnt[u];m<cnt[u+1];m++){ int j=idx[m]; if(j<=i) continue; if(ln[j]==ln[i] && j<i+3) continue;
        int d1=-1,d2=-1;
        for(int b=0;b<nn[ty[i+1]];b++) if(nb[ty[i+1]][b]==ty[j+1]){d1=nd[ty[i+1]][b];break;}
        if(d1<0) continue;
        for(int b=0;b<nn[ty[i+2]];b++) if(nb[ty[i+2]][b]==ty[j+2]){d2=nd[ty[i+2]][b];break;}
        if(d2<0) continue;
        if(d0+d1+d2>3) continue;
        printf("%d %d %d %d %d\n",i,j,d0,d1,d2); np++; }}}
  fprintf(stderr,"types %d tokens %d trigrams %d pairs %ld\n",V,N,nt,np);
  return 0;
}
