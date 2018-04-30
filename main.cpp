#include"net.h"
#include"defs.h"

#define NINPUTS (2)
#define NOUTPUTS (2)

#define L1M (NINPUTS)
#define L1N (10)
#define L2M (10)
#define L2N (NOUTPUTS)

const int mDim[LAYERS]={L1M,L2M};//,L3M};
const int nDim[LAYERS]={L1N,L2N};//,L3N};
//const int mDim[LAYERS]={L1M,L2M,L3M};
//const int nDim[LAYERS]={L1N,L2N,L3N};

const double ex1[NINPUTS]={-1,-1};
const double ex2[NINPUTS]={-1,+1};
const double ex3[NINPUTS]={+1,-1};
const double ex4[NINPUTS]={+1,+1};
const double ans1[NOUTPUTS]={-1,+1};
const double ans2[NOUTPUTS]={+1,-1};
const double ans3[NOUTPUTS]={+1,-1};
const double ans4[NOUTPUTS]={-1,+1};
//const double ans1[NOUTPUTS]={-1,+1};
//const double ans2[NOUTPUTS]={+1,-1};
//const double ans3[NOUTPUTS]={+1,-1};
//const double ans4[NOUTPUTS]={+1,-1};
int main(){
	int i;
	Net *net=new Net(LAYERS);
	for(i=0;i<LAYERS;i++){
		net->insertLayer(i,mDim[i],nDim[i]);
	}

	net->print();
	net->rand();
	net->print();

	Array *p1,*p2,*p3,*p4;
	Array *pAns1,*pAns2,*pAns3,*pAns4;

	p1=new Array(ex1,NINPUTS);
	p2=new Array(ex2,NINPUTS);
	p3=new Array(ex3,NINPUTS);
	p4=new Array(ex4,NINPUTS);

	pAns1=new Array(ans1,NOUTPUTS);
	pAns2=new Array(ans2,NOUTPUTS);
	pAns3=new Array(ans3,NOUTPUTS);
	pAns4=new Array(ans4,NOUTPUTS);

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
	for(i=0;i<EPOCHS;i++){
//		tmpvar=(tmpvar+1)%4;
		tmpvar=rand()%3;
		pIn=pInputs[tmpvar];
		pOut=pOutputs[tmpvar];
		net->train(pIn,pOut);
		if(i%4){
			net->updateWeights();
		}
	}
}
