#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "matrixmul.cu"

#define LAYERS 3
#define L1N 6
#define L1M 6
#define L2N 6
#define L2M 40
#define L3N 40
#define L3M 40

#define RANDSCALING 10	//scale random weights to be from -0.1 to +0.1

const int nDim[LAYERS]={L1N,L2N,L3N};
const int mDim[LAYERS]={L1M,L2M,L3M};

struct Layer{
	Array *in;
	Matrix *M;
	Array *out;
};
struct Net{
	Layer **L;
	int size;
};
void PRINTMATRIX(Matrix *M){
	int i,j;
	for(i=0;i<M->height;i++){
		for(j=0;j<M->width;j++){
			printf("[%i,%i]%.04f\t",i,j,M->elements[i*M->stride+j]);
		}
	}
	printf("\n");
}
void PRINTARRAY(Array *A){
	int i;
	float *x;
	int sz=A->len;
	x=A->el;
	for(i=0;i<sz;i++){
		printf("[%i]%.02f\t",i,*x++);
	}
	printf("\n");
}
void randMatrix(Matrix *M){
	int i,j;
	for(i=0;i<M->height;i++){
		for(j=0;j<M->width;j++){
			M->elements[i*M->stride+j]=
			(random()-(RAND_MAX/2))*2.0/((float)RAND_MAX)/((float)RANDSCALING);
		}
	}
}
void randArray(Array *A){
	int i;
	for(i=0;i<A->len;i++){
		A->el[i]=
		(random()-(RAND_MAX/2))*2.0/((float)RAND_MAX)/((float)RANDSCALING);
	}
}
void nnRand(Net *N){
	Matrix *pM;
	int i;
	int n=N->size;
	for(i=0;i<n;i++){
		pM=N->L[i]->M;
		randMatrix(pM);
	}
}
void nnInsert(Array *A){
}
const float ex1[L1N]={-1,-1};
const float ex2[L1N]={-1,+1};
const float ex3[L1N]={+1,-1};
const float ex4[L1N]={+1,+1};
int main(){
	int i,j,k;
	Net *net;
	net=(Net *)malloc(sizeof(Net));
	net->L=(Layer **)malloc(LAYERS*sizeof(Layer *));
	net->size=LAYERS;
	net->L[0]=(Layer *)malloc(sizeof(Layer));
	net->L[0]->in=(Array *)malloc(sizeof(Array));
	net->L[0]->out=(Array *)malloc(sizeof(Array));
	net->L[0]->in->len=L1N;
	net->L[0]->in->el=(float *)malloc(L1N*sizeof(float));
	net->L[0]->out->len=L1M;
	net->L[0]->out->el=(float *)malloc(L1M*sizeof(float));
	for(i=0;i<LAYERS;i++){
		if(i>0){
			net->L[i]=(Layer *)malloc(sizeof(Layer));
			net->L[i]->in=net->L[i-1]->out;
			net->L[i]->out=(Array *)malloc(sizeof(Array));
			net->L[i]->in->len=nDim[i];
			net->L[i]->in->el=(float *)malloc(nDim[i]*sizeof(float));
			net->L[i]->out->len=mDim[i];
			net->L[i]->out->el=(float *)malloc(mDim[i]*sizeof(float));
		}
//		net->L[i]=(Layer *)malloc(sizeof(Layer));
		net->L[i]->M=(Matrix *)malloc(sizeof(Matrix));
		memcpy(&net->L[i]->M->height,&nDim[i],sizeof(int));
		memcpy(&net->L[i]->M->width,&mDim[i],sizeof(int));
		memcpy(&net->L[i]->M->stride,&mDim[i],sizeof(int));
		net->L[i]->M->elements=(float *)malloc(nDim[i]*mDim[i]*sizeof(float));
	}
	Matrix *pM=net->L[0]->M;
	PRINTMATRIX(net->L[0]->M);
	nnRand(net);
	PRINTMATRIX(net->L[0]->M);
	Array *pA=net->L[0]->in;
	PRINTARRAY(pA);
	randArray(pA);
	PRINTARRAY(pA);

	//pA=&ex1;
	//nnInsert(pA);

	Array *py=net->L[0]->out;
	PRINTARRAY(py);
	MatMul(*pM,*pA,*py);
	PRINTARRAY(py);
}
