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

	Data data=Data();
	Array pIn=Array(2);
	Array pOut=Array(2);
	data.fillIOArrays(pIn,pOut);
	pIn.print();

	return 0;
	for(i=0;i<EPOCHS;i++){
/*
		pIn=pInputs[tmpvar];
		pOut=pOutputs[tmpvar];
		net->train(pIn,pOut);
		if(i%4){
			net->updateWeights();
		}
*/
	}
}
