#include <cuda.h>
#include <stdio.h>
#include <string.h>

#define LAYERS 3
#define L1N 5
#define L1M 7
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
void PRINTMATRIX(float **x,int n,int m){
	int i,j;
	for(i=0;i<n;i++){
		for(j=0;j<m;j++){
			printf("[%i,%i]%.02f\t",i,j,x[i][j]);
		}
	}
	printf("\n");
}
void PRINTARRAY(float *x,int sz){
	int i;
	for(i=0;i<sz;i++){
		printf("[%i]%.02f\t",i,*x++);
	}
	printf("\n");
}
int main(){
	int i,j,k;
	float test[L1N*L1M];
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
	for(i=0;i<L1N*L1M;i++){
		test[i]=i;
	}
	PRINTARRAY(test,L1N*L1M);
	PRINTARRAY(test,L1N*L1M);
	PRINTMATRIX(net->L[0]->L,*net->L[0]->N,*net->L[0]->M);
	//memcpy(net->L[0]->L,test,L1N*L1M*sizeof(float));
	k=0;
	for(i=0;i<*net->L[0]->N;i++){
		for(j=0;j<*net->L[0]->M;j++){
			net->L[0]->L[i][j]=k++;
		}
	}
	PRINTMATRIX(net->L[0]->L,*net->L[0]->N,*net->L[0]->M);
}
