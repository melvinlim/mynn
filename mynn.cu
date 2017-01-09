#ifndef _MYNN
#define _MYNN

#include <cuda.h>
#include <math.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <vector>

using namespace std;

#include "types.h"

#define NINPUTS (2)
#define NOUTPUTS (2)

#define L1M (NINPUTS)
#define L1N (7)
#define L2M (7)
#define L2N (NOUTPUTS)

const int nDim[LAYERS]={L1N,L2N};//,L3N};
const int mDim[LAYERS]={L1M,L2M};//,L3M};
//const int nDim[LAYERS]={L1N,L2N,L3N};
//const int mDim[LAYERS]={L1M,L2M,L3M};

void PRINTINFO(Array<float> *pIn,Array<float> *answer,Array<float> *pOut,Array<float> *pErr){
	printf("in:[%.0f,%.0f] out:[%f,%f] targ:[%.0f,%.0f] err:[%f,%f]\n",
	pIn->el[0],pIn->el[1],
	answer->el[0],answer->el[1],
	pOut->el[0],pOut->el[1],
	pErr->el[0],pErr->el[1]
	);
}
float ex1[NINPUTS]={-1,-1};
float ex2[NINPUTS]={-1,+1};
float ex3[NINPUTS]={+1,-1};
float ex4[NINPUTS]={+1,+1};
float ans1[NOUTPUTS]={-1,+1};
float ans2[NOUTPUTS]={+1,-1};
float ans3[NOUTPUTS]={+1,-1};
float ans4[NOUTPUTS]={-1,+1};
//const float ans1[NOUTPUTS]={-1,+1};
//const float ans2[NOUTPUTS]={+1,-1};
//const float ans3[NOUTPUTS]={+1,-1};
//const float ans4[NOUTPUTS]={+1,-1};
int main(){
	int i;
	srand(time(0));
	Net *net=new Net(LAYERS);
	for(i=0;i<LAYERS;i++){
		net->insertLayer(i,nDim[i],mDim[i]);
	}

	net->print();
	net->rand();
	net->print();
/*
	Array<float> *p1,*p2,*p3,*p4;
	Array<float> *pAns1,*pAns2,*pAns3,*pAns4;

	p1=new Array<float>(ex1,NINPUTS);
	p2=new Array<float>(ex2,NINPUTS);
	p3=new Array<float>(ex3,NINPUTS);
	p4=new Array<float>(ex4,NINPUTS);

	pAns1=new Array<float>(ans1,NOUTPUTS);
	pAns2=new Array<float>(ans2,NOUTPUTS);
	pAns3=new Array<float>(ans3,NOUTPUTS);
	pAns4=new Array<float>(ans4,NOUTPUTS);
*/
	float **pInputs=(float **)malloc(4*sizeof(float *));
	pInputs[0]=ex1;
	pInputs[1]=ex2;
	pInputs[2]=ex3;
	pInputs[3]=ex4;
	float **pOutputs=(float **)malloc(4*sizeof(float *));
	pOutputs[0]=ans1;
	pOutputs[1]=ans2;
	pOutputs[2]=ans3;
	pOutputs[3]=ans4;
	Array<float> *pIn,*pOut;
	int tmpvar;
	for(i=0;i<EPOCHS;i++){
		tmpvar=rand()%4;
		pIn=new Array<float>(pInputs[tmpvar],NINPUTS);
		pOut=new Array<float>(pOutputs[tmpvar],NOUTPUTS);
		net->train(pIn,pOut);
		PRINTINFO(pIn,net->answer,pOut,net->error);
		delete pIn;
		delete pOut;
	}
}

#endif
