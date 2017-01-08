#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "manip.cu"
#include "matrixmul.cu"

#define L1M 2
#define L1N 40
#define L2M 40
#define L2N 2

const int nDim[LAYERS]={L1N,L2N};//,L3N};
const int mDim[LAYERS]={L1M,L2M};//,L3M};

void nnInsert(Net *N,Array *x){
	memcpy(N->L[0]->in->el,x->el,x->len*sizeof(float));
}
/*
Array *nnForward(Net *N){
	int i;
	for(i=0;i<LAYERS;i++){
//printf("***********%d\n",i);
//		PRINTARRAY(N->L[i]->in);
//		PRINTARRAY(N->L[i]->out);
		MatMul(*N->L[i]->M,*N->L[i]->in,*N->L[i]->out,*N->L[i]->deriv);
//		PRINTARRAY(N->L[i]->in);
//		PRINTARRAY(N->L[i]->out);
	}
	return N->L[LAYERS-1]->out;
}
*/
void nnError(Array *err,const Array *y0,const Array *yTarget){
	int i;
	int n=y0->len;
	for(i=0;i<n;i++){
		err->el[i]=(y0->el[i]-yTarget->el[i]);
	}
}
float nnTotalError(const Array *y0,const Array *y){
	int i;
	int n=y0->len;
	float ret=0;
	for(i=0;i<n;i++){
		ret+=fabs(y0->el[i]-y->el[i]);
		ret*=ret;
	}
	return(ret/2.0);
}
const float ex1[L1M]={-1,-1};
const float ex2[L1M]={-1,+1};
const float ex3[L1M]={+1,-1};
const float ex4[L1M]={+1,+1};
//const float ans1[L2N]={-1,+1};
//const float ans2[L2N]={+1,-1};
//const float ans3[L2N]={+1,-1};
//const float ans4[L2N]={-1,+1};
const float ans1[L2N]={-1,+1};
const float ans2[L2N]={+1,-1};
const float ans3[L2N]={+1,-1};
const float ans4[L2N]={+1,-1};
int main(){
	int i;
	Net *net;
	net=(Net *)malloc(sizeof(Net));
	net->L=(Layer **)malloc(LAYERS*sizeof(Layer *));
	net->size=LAYERS;
	for(i=0;i<LAYERS;i++){
		net->L[i]=(Layer *)malloc(sizeof(Layer));
		if(i>0){
			net->L[i]->in=net->L[i-1]->out;
		}else{
			net->L[0]->in=(Array *)malloc(sizeof(Array));
		}
		net->L[i]->out=(Array *)malloc(sizeof(Array));
		net->L[i]->deriv=(Array *)malloc(sizeof(Array));
		net->L[i]->delta=(Array *)malloc(sizeof(Array));
		net->L[i]->in->len=mDim[i];
		net->L[i]->in->el=(float *)malloc(mDim[i]*sizeof(float));
		net->L[i]->out->len=nDim[i];
		net->L[i]->out->el=(float *)malloc(nDim[i]*sizeof(float));
		net->L[i]->deriv->len=nDim[i];
		net->L[i]->deriv->el=(float *)malloc(nDim[i]*sizeof(float));
		net->L[i]->delta->len=nDim[i];
		net->L[i]->delta->el=(float *)malloc(nDim[i]*sizeof(float));

		net->L[i]->M=(Matrix *)malloc(sizeof(Matrix));
		net->L[i]->M->height=nDim[i];
		net->L[i]->M->width=mDim[i];
		net->L[i]->M->stride=mDim[i];
		net->L[i]->M->elements=(float *)malloc(nDim[i]*mDim[i]*sizeof(float));
		net->L[i]->dW=(Matrix *)malloc(sizeof(Matrix));
		net->L[i]->dW->height=nDim[i];
		net->L[i]->dW->width=mDim[i];
		net->L[i]->dW->stride=mDim[i];
		net->L[i]->dW->elements=(float *)malloc(nDim[i]*mDim[i]*sizeof(float));
	}

	nnRand(net);
	for(i=0;i<LAYERS;i++){
		PRINTMATRIX(net->L[i]->M);
	}

	Array *p1,*p2,*p3,*p4,*ret;
	p1=CREATEARRAY(ex1,L1M);
	p2=CREATEARRAY(ex2,L1M);
	p3=CREATEARRAY(ex3,L1M);
	p4=CREATEARRAY(ex4,L1M);
	Array *pAns1,*pAns2,*pAns3,*pAns4;
	pAns1=CREATEARRAY(ans1,L2N);
	pAns2=CREATEARRAY(ans2,L2N);
	pAns3=CREATEARRAY(ans3,L2N);
	pAns4=CREATEARRAY(ans4,L2N);

	Array *pError;
	pError=CREATEARRAY(ans4,L2N);

	ret=CREATEARRAY(0,L2N);

	nnInsert(net,p1);
	ret=nnForward(net);
	PRINTARRAY(ret);
	nnError(pError,ret,pAns1);
	float err=nnTotalError(ret,pAns1);
	printf("err:%f\n",err);
	nnBackProp(net,pError);


	Array **pInputs=(Array **)malloc(4*sizeof(Array *));
	pInputs[0]=p1;
	pInputs[1]=p2;
	pInputs[2]=p3;
	pInputs[3]=p4;
	Array **pOutputs=(Array **)malloc(4*sizeof(Array *));
	pOutputs[0]=pAns1;
	pOutputs[1]=pAns2;
	pOutputs[2]=pAns3;
	pOutputs[3]=pAns4;
	Array *pIn,*pOut;
	int tmpvar;
	for(i=0;i<1000;i++){
		tmpvar=rand()%4;
		pIn=pInputs[tmpvar];
		pOut=pOutputs[tmpvar];
		nnInsert(net,pIn);
		ret=nnForward(net);
		nnError(pError,ret,pOut);
		err=nnTotalError(ret,pOut);
		printf("out:[%f,%f] targ:[%f,%f] err:%f\n",net->L[LAYERS-1]->out->el[0],net->L[LAYERS-1]->out->el[1],pOut->el[0],pOut->el[1],err);
		nnBackProp(net,pError);
	}

	for(i=0;i<LAYERS;i++){
		PRINTARRAY(net->L[i]->out);
	}
}
