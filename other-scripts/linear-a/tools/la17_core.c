/* LA-17 per-word SIR outbreak on a site network. Words are independent lineages. */
#include <stdint.h>
#include <math.h>
static uint64_t s0, s1;
static inline uint64_t nx(void){uint64_t a=s0,b=s1;s0=b;a^=a<<23;s1=a^b^(a>>17)^(b>>26);return s1+b;}
static inline double ur(void){return (nx()>>11)*(1.0/9007199254740992.0);}
void seed(uint64_t a){s0=a*0x9E3779B97F4A7C15ULL+1;s1=(a+7)*0xBF58476D1CE4E5B9ULL+3;for(int i=0;i<20;i++)nx();}
/* lit[j*ns+k]; start[j] first lit step (-1 none); parent[j] (-1 none); last[j] last lit step;
   W[i*K+j] contact i->j; bsite,bstep per word; out snap[w*K+j] */
void sim(int M,int K,int ns,const uint8_t*lit,const int*start,const int*parent,const int*last,
         const double*W,double beta,double dt,double phi,double gamma,
         const int*bsite,const int*bstep,uint8_t*snap){
  uint8_t H[64],R[64],Hn[64];
  for(int w=0;w<M;w++){
    for(int j=0;j<K;j++){H[j]=0;R[j]=0;}
    uint8_t*o=snap+(long)w*K; for(int j=0;j<K;j++)o[j]=0;
    int any=0;
    for(int k=0;k<ns;k++){
      if(k<bstep[w] ){ /* still founding possible but word does not exist yet */ continue; }
      for(int j=0;j<K;j++) if(start[j]==k && parent[j]>=0 && H[parent[j]] && !R[j] && ur()<phi){H[j]=1;}
      if(k==bstep[w]){H[bsite[w]]=1;any=1;}
      if(any){
        for(int j=0;j<K;j++){Hn[j]=H[j];
          if(!H[j]&&!R[j]&&lit[j*ns+k]){double p=0;for(int i=0;i<K;i++)if(H[i])p+=W[i*K+j];
            if(p>0 && ur()< 1-exp(-beta*p*dt)) Hn[j]=1;}}
        for(int j=0;j<K;j++){H[j]=Hn[j]; if(H[j]&&gamma>0&&ur()<gamma*dt){H[j]=0;R[j]=1;}}
      }
      for(int j=0;j<K;j++) if(last[j]==k) o[j]=H[j];
    }
  }
}
