#include <cuda.h>
#include <stdio.h>
#include <string.h>

#include "matrixmul.cu"

#define LAYERS 3
#define L1N 6
#define L1M 6
#define L2N 6
#define L2M 40
#define L3N 40
#define L3M 40

const int nDim[LAYERS]={L1N,L2N,L3N};
const int mDim[LAYERS]={L1M,L2M,L3M};

struct Layer{
	Array *in;
	Matrix *M;
	Array *out;
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
int main(){
	int i,j,k;
	Net *net;
	net=(Net *)malloc(sizeof(Net));
	net->L=(Layer **)malloc(LAYERS*sizeof(Layer *));
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
	PRINTMATRIX(net->L[0]->M);
	k=0;
	for(i=0;i<net->L[0]->M->height;i++){
		for(j=0;j<net->L[0]->M->width;j++){
			net->L[0]->M->elements[i*net->L[0]->M->stride+j]=k++;
		}
	}
	PRINTMATRIX(net->L[0]->M);
	Array *pA=net->L[0]->in;
	PRINTARRAY(pA);
	for(i=0;i<pA->len;i++){
		pA->el[i]=i;
	}
	PRINTARRAY(pA);
	Matrix *Mptr=net->L[0]->M;
	//MatMul requires matrices to be multiples of BLOCK_SIZE (declared in matmul.cu) and possibly to be square.
	MatMul(*Mptr,*Mptr,*Mptr);
	PRINTMATRIX(net->L[0]->M);
	Array *py=net->L[0]->out;
	PRINTARRAY(py);
	MatMul(*Mptr,*pA,*py);
	PRINTARRAY(py);
}
