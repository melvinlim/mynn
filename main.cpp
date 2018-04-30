#include"net.h"
#include"data.h"
#include"defs.h"

#define NINPUTS (2)
#define NOUTPUTS (2)

#define L1M (NINPUTS+1)
#define L1N (10)
#define L2M (10+1)
#define L2N (NOUTPUTS)

const int mDim[LAYERS]={L1M,L2M};//,L3M};
const int nDim[LAYERS]={L1N,L2N};//,L3N};

int main(){
	int i;
	Net *net=new Net(LAYERS);
	for(i=0;i<LAYERS;i++){
		net->insertLayer(i,mDim[i],nDim[i]);
	}

	net->print();
	net->rand();
	net->print();

	Array **arrays;
	Array *pIn,*pOut;

	XorData data;
	for(int i=0;i<8;i++){
		printf("%d:\n",i);
		arrays=data.fillIOArrays();
		arrays[0]->print();
		arrays[1]->print();
		printf("\n");
	}

	for(i=0;i<EPOCHS;i++){
/*
Net *asdf=new Net(10);
for(int j=0;j<10;j++){
	asdf->insertLayer(j,10,10);
}
delete asdf;
*/
		arrays=data.fillIOArrays();
		pIn=arrays[0];
		pOut=arrays[1];
		net->train(pIn,pOut);
		if(i%4){
			net->updateWeights();
		}
	}
}
