#include"net.h"
#include"data.h"
#include"defs.h"
#include"idx.h"

int main(){
	int i;
	Net *net;
	Array **arrays;
	Array *pIn,*pOut;
	net=new SingleHidden(NINPUTS,HIDDEN,NOUTPUTS);
#ifdef SOLVEXOR
	XorData data;
#else
	MNISTData data;
#endif

	for(int i=0;i<8;i++){
		printf("%d:\n",i);
		arrays=data.fillIOArrays(true);
		arrays[1]->print();
		arrays[0]->print();
	}
	int hidden=HIDDEN;
	for(int network=0;network<4;network++){
		delete net;
		net=new SingleHidden(NINPUTS,hidden++,NOUTPUTS);
		for(i=0;i<EPOCHS;i++){
			arrays=data.fillIOArrays();
			pIn=arrays[0];
			pOut=arrays[1];
#ifdef BATCH
			net->trainBatch(pIn,pOut);
			if(i%4){
				net->updateWeights();
			}
#else
			net->trainOnce(pIn,pOut);
			printf("epoch: %i\n",i);
			data.status(arrays,net->response,net->error);
#endif
		}
		printf("net: %d\n",network);
		for(int i=0;i<data.nOutputs;i++){
			arrays=data.fillIOArrays();
			pIn=arrays[0];
			net->forward(pIn);
			data.status(arrays,net->response,net->error);
		}
	}
}
