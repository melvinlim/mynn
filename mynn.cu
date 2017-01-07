#include <cuda.h>
#include <stdio.h>
#include <string.h>

#define LAYERS 3
#define L1N 40
#define L1M 40
#define L2N 40
#define L2M 40
#define L3N 40
#define L3M 40

const int nDim[LAYERS]={L1N,L2N,L3N};
const int mDim[LAYERS]={L1M,L2M,L3M};

struct Layer{
	float **L;
	int *N;
	int *M;
};
struct Net{
	Layer **L;
};
int main(){
	int i,j;
	Net *net;
	net=(Net *)malloc(sizeof(Net));
	net->L=(Layer **)malloc(LAYERS*sizeof(Layer *));
	for(i=0;i<LAYERS;i++){
		net->L[i]=(Layer *)malloc(sizeof(Layer));
		net->L[i]->N=(int *)malloc(sizeof(int));
		net->L[i]->M=(int *)malloc(sizeof(int));
		memcpy(net->L[i]->N,&nDim[i],sizeof(int));
		memcpy(net->L[i]->M,&mDim[i],sizeof(int));
		net->L[i]->L=(float **)malloc(nDim[i]*sizeof(float *));
		for(j=0;j<nDim[i];j++){
			net->L[i]->L[j]=(float *)malloc(mDim[i]*sizeof(float));
		}
	}
}
