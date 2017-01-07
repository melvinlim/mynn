#include <cuda.h>
#include <stdio.h>
#include <string.h>

#include "matrixmul.cu"

#define LAYERS 3
#define L1N 6
#define L1M 6
#define L2N 40
#define L2M 40
#define L3N 40
#define L3M 40

const int nDim[LAYERS]={L1N,L2N,L3N};
const int mDim[LAYERS]={L1M,L2M,L3M};

struct Layer{
	Matrix *M;
};
struct Net{
	Layer **L;
};
void PRINTMATRIX(Matrix *M){
	int i,j;
	for(i=0;i<M->height;i++){
		for(j=0;j<M->width;j++){
			printf("[%i,%i]%.02f\t",i,j,M->elements[i*M->stride+j]);
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
		net->L[i]->M=(Matrix *)malloc(sizeof(Matrix));
		memcpy(&net->L[i]->M->height,&nDim[i],sizeof(int));
		memcpy(&net->L[i]->M->width,&mDim[i],sizeof(int));
		memcpy(&net->L[i]->M->stride,&mDim[i],sizeof(int));
		net->L[i]->M->elements=(float *)malloc(nDim[i]*mDim[i]*sizeof(float));
	}
	for(i=0;i<L1N*L1M;i++){
		test[i]=i;
	}
	PRINTARRAY(test,L1N*L1M);
	PRINTMATRIX(net->L[0]->M);
	//memcpy(net->L[0]->L,test,L1N*L1M*sizeof(float));
	k=0;
	for(i=0;i<net->L[0]->M->height;i++){
		for(j=0;j<net->L[0]->M->width;j++){
			net->L[0]->M->elements[i*net->L[0]->M->stride+j]=k++;
		}
	}
	PRINTMATRIX(net->L[0]->M);
	Matrix *Mptr=net->L[0]->M;
	//MatMul requires matrices to be multiples of BLOCK_SIZE (declared in matmul.cu) and possibly to be square.
	MatMul(*Mptr,*Mptr,*Mptr);
	PRINTMATRIX(net->L[0]->M);
}
